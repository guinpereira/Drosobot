"""
O que esta maquina de fato oferece de compute. Sonda, nao adivinha.

    .venv\\Scripts\\python tools\\gpu_probe\\probe.py
    .venv\\Scripts\\python tools\\gpu_probe\\probe.py --json

Escreve benchmarks/hardware/<host>_<data>.json e imprime um resumo.

## Por que existe

A decisao de backend do Drosobot depende de capacidade real do dispositivo, nao
de o que a documentacao de um projeto assume. Duas coisas ja mordemos por
confiar em suposicao:

  - "Warp acelera em GPU" -- acelera em CUDA. Em AMD ele cai pro device CPU.
  - "HIP SDK suporta Windows" -- suporta, mas nao esta GPU (gfx1031 nao consta
    na matriz de suporte).

Entao a regra passou a ser: antes de escolher backend, rodar isto e guardar a
saida junto do benchmark. Um numero de desempenho sem o hardware que o produziu
nao vale muito.

O probe nao instala nada e nao falha se algo estiver ausente -- ausencia e
resultado, e fica registrada.
"""
from __future__ import annotations

import argparse
import json
import platform
import re
import shutil
import socket
import subprocess
from datetime import datetime
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[2]
SAIDA = RAIZ / "benchmarks" / "hardware"


def _roda(cmd: list[str], timeout: int = 60) -> str | None:
    """Roda e devolve stdout, ou None se o programa nao existe/falhou."""
    if shutil.which(cmd[0]) is None and not Path(cmd[0]).exists():
        return None
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout,
                           errors="replace")
        return r.stdout
    except (OSError, subprocess.SubprocessError):
        return None


# ------------------------------------------------------------------ sistema

def sonda_sistema() -> dict:
    d = {
        "hostname": socket.gethostname(),
        "os": platform.system(),
        "os_release": platform.release(),
        "os_version": platform.version(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "cpu": platform.processor(),
    }
    if platform.system() == "Windows":
        out = _roda(["powershell", "-NoProfile", "-Command",
                     "(Get-CimInstance Win32_Processor | Select-Object -First 1 "
                     "Name,NumberOfCores,NumberOfLogicalProcessors | ConvertTo-Json)"])
        if out:
            try:
                info = json.loads(out)
                d["cpu"] = (info.get("Name") or d["cpu"]).strip()
                d["cpu_cores"] = info.get("NumberOfCores")
                d["cpu_threads"] = info.get("NumberOfLogicalProcessors")
            except ValueError:
                pass
    return d


# ---------------------------------------------------------------------- GPU

def sonda_gpu_windows() -> list[dict]:
    out = _roda(["powershell", "-NoProfile", "-Command",
                 "Get-CimInstance Win32_VideoController | Select-Object "
                 "Name,DriverVersion,DriverDate,VideoProcessor | ConvertTo-Json"])
    if not out:
        return []
    try:
        dados = json.loads(out)
    except ValueError:
        return []
    if isinstance(dados, dict):
        dados = [dados]
    return [{"name": g.get("Name"), "driver": g.get("DriverVersion"),
             "processor": g.get("VideoProcessor")} for g in dados]


# ------------------------------------------------------------------- Vulkan

def sonda_vulkan() -> dict:
    """Capacidades de compute via Vulkan. E o caminho mais portavel que temos."""
    d: dict = {"available": False}
    out = _roda(["vulkaninfo"], timeout=120)
    if not out:
        d["note"] = "vulkaninfo ausente ou falhou"
        return d
    d["available"] = True

    def num(chave: str):
        m = re.search(rf"{re.escape(chave)}\s*=\s*(\d+)", out)
        return int(m.group(1)) if m else None

    def txt(chave: str):
        m = re.search(rf"{re.escape(chave)}\s*=\s*(.+)", out)
        return m.group(1).strip() if m else None

    d["device_name"] = txt("deviceName")
    d["driver_name"] = txt("driverName")
    d["api_version"] = txt("apiVersion")
    d["subgroup_size"] = num("subgroupSize")
    d["max_shared_memory_bytes"] = num("maxComputeSharedMemorySize")
    d["max_workgroup_invocations"] = num("maxComputeWorkGroupInvocations")
    d["shader_float64"] = "shaderFloat64                           = true" in out

    # Extensoes que decidem se da pra portar as tecnicas do MJWarp.
    # atomic_add em float NAO e core do Vulkan -- por isso esta na lista.
    interesse = [
        "VK_EXT_shader_atomic_float", "VK_EXT_shader_atomic_float2",
        "VK_KHR_shader_atomic_int64", "VK_KHR_cooperative_matrix",
        "VK_KHR_buffer_device_address", "VK_KHR_16bit_storage",
        "VK_KHR_shader_float16_int8", "VK_EXT_subgroup_size_control",
        "VK_KHR_timeline_semaphore", "VK_KHR_shader_subgroup_extended_types",
    ]
    d["extensions"] = {e: (e in out) for e in interesse}
    return d


# ---------------------------------------------------------------------- HIP

def sonda_hip() -> dict:
    """
    HIP/ROCm. Ausente nao e erro: e o estado desta maquina, e importa registrar.
    """
    d: dict = {"available": False, "gfx_target": None}
    out = _roda(["hipInfo"]) or _roda(["rocminfo"])
    if out:
        d["available"] = True
        d["raw_head"] = "\n".join(out.splitlines()[:40])
        m = re.search(r"gfx\d+", out)
        if m:
            d["gfx_target"] = m.group(0)
    ver = _roda(["hipcc", "--version"])
    if ver:
        d["hipcc_version"] = ver.splitlines()[0] if ver.splitlines() else None
        d["available"] = True
    if not d["available"]:
        d["note"] = ("HIP SDK nao instalado. Para esta GPU isso nao e acidente: "
                     "ver docs/research/HIP_WINDOWS_API_AUDIT.md -- gfx1031 "
                     "(RX 6700 XT) nao consta na matriz de suporte do HIP SDK "
                     "para Windows.")
    return d


# -------------------------------------------------------------------- OpenCL

def sonda_opencl() -> dict:
    d: dict = {"available": False}
    if platform.system() == "Windows":
        icd = Path("C:/Windows/System32/OpenCL.dll")
        d["icd_present"] = icd.exists()
        d["amd_runtime_present"] = Path("C:/Windows/System32/amdocl64.dll").exists()
        d["available"] = d["icd_present"]
    out = _roda(["clinfo"])
    if out:
        m = re.search(r"Device Name\s+(.+)", out)
        if m:
            d["device_name"] = m.group(1).strip()
    return d


# --------------------------------------------------------------------- CUDA

def sonda_cuda() -> dict:
    d: dict = {"available": False}
    out = _roda(["nvidia-smi", "--query-gpu=name,driver_version,memory.total",
                 "--format=csv,noheader"])
    if out and out.strip():
        d["available"] = True
        d["raw"] = out.strip()
    return d


# -------------------------------------------------------------------- Python

def sonda_python() -> dict:
    import importlib
    pacotes = ["numpy", "numba", "torch", "jax", "cupy", "pyopencl", "warp",
               "mujoco", "mujoco_warp", "flygym", "brian2"]
    d = {}
    for p in pacotes:
        try:
            mod = importlib.import_module(p)
            d[p] = getattr(mod, "__version__", "instalado")
        except Exception:      # noqa: BLE001  -- import de terceiro pode explodir feio
            d[p] = None
    return d


def main():
    ap = argparse.ArgumentParser(description="Sonda de compute do Drosobot")
    ap.add_argument("--json", action="store_true", help="so o JSON, sem resumo")
    ap.add_argument("--out", default=None, help="caminho do JSON")
    args = ap.parse_args()

    relatorio = {
        "probed_at": datetime.now().isoformat(timespec="seconds"),
        "system": sonda_sistema(),
        "gpus": sonda_gpu_windows() if platform.system() == "Windows" else [],
        "vulkan": sonda_vulkan(),
        "hip": sonda_hip(),
        "opencl": sonda_opencl(),
        "cuda": sonda_cuda(),
        "python_packages": sonda_python(),
    }

    SAIDA.mkdir(parents=True, exist_ok=True)
    carimbo = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    destino = Path(args.out) if args.out else (
        SAIDA / f"{relatorio['system']['hostname']}_{carimbo}.json")
    destino.write_text(json.dumps(relatorio, indent=2), encoding="utf-8")

    if args.json:
        print(json.dumps(relatorio, indent=2))
        return

    s, v, h = relatorio["system"], relatorio["vulkan"], relatorio["hip"]
    print(f"SISTEMA   {s['os']} {s['os_release']}  |  {s.get('cpu')}"
          f"  ({s.get('cpu_cores')}C/{s.get('cpu_threads')}T)")
    for g in relatorio["gpus"]:
        print(f"GPU       {g['name']}  driver {g['driver']}")
    print()
    if v.get("available"):
        print(f"VULKAN    {v.get('device_name')}  api {v.get('api_version')}")
        print(f"          subgroup {v.get('subgroup_size')}   "
              f"LDS {v.get('max_shared_memory_bytes')} B   "
              f"maxWG {v.get('max_workgroup_invocations')}   "
              f"fp64 {v.get('shader_float64')}")
        faltando = [e for e, ok in v.get("extensions", {}).items() if not ok]
        print(f"          extensoes de interesse: "
              f"{len(v.get('extensions', {})) - len(faltando)}"
              f"/{len(v.get('extensions', {}))} presentes"
              + (f"  faltando: {', '.join(faltando)}" if faltando else ""))
    else:
        print("VULKAN    indisponivel")
    print(f"HIP       {'disponivel ' + str(h.get('gfx_target')) if h['available'] else 'ausente'}")
    if not h["available"]:
        print(f"          {h.get('note','')}")
    print(f"OPENCL    {'ICD presente' if relatorio['opencl'].get('available') else 'ausente'}")
    print(f"CUDA      {'presente' if relatorio['cuda']['available'] else 'ausente'}")
    print()
    print(f"salvo em {destino.relative_to(RAIZ)}")


if __name__ == "__main__":
    main()
