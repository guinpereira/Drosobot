r"""
Quantos estagios sequenciais cabem em 117 us? O teto de QUALQUER implementacao.

    .venv-flygym2\Scripts\python benchmarks\physics\gpu\orcamento_de_estagios.py

A cinematica portada e um ponto so da curva: 12 estagios sequenciais, 12,3 us,
~1,0 us por estagio. Um passo de fisica inteiro tem muito mais estagios, e vale
saber quantos **antes** de escrever solver, porque o resultado e independente da
qualidade da implementacao.

Um estagio sequencial e um trecho que **nao pode comecar antes de o anterior
terminar**, e que por isso termina numa `barrier()`. Num passo do MuJoCo sobre
este modelo:

    cinematica direta            10   niveis da arvore
    quadros inerciais + geoms     2
    comPos                        2
    inercia composta (CRB)       10   niveis, para tras
    montagem de qM               17   profundidade da cadeia de dof
    factorM (Cholesky esparsa)   17   profundidade da cadeia
    RNE (bias)                   20   10 para frente, 10 para tras
    colisao                       1   os 55 pares em paralelo
    makeConstraint                3
    projectConstraint            36   dois solves triangulares + produto
    atuacao                       1
    solver Newton                      1 fatoracao 24x24 + 8 iteracoes de
                                 ~300  (linesearch + solve + update de posto 1)
    Euler (M + h*D)              51   fatorar + dois solves + integrar
                                ----
                                ~470  estagios sequenciais por passo

A contagem e aproximada de proposito -- a ordem de grandeza e o que decide, e
ela nao muda se for 400 ou 600.

Este script mede a outra metade: **quanto custa um estagio sequencial**, em
funcao de quanto trabalho ele carrega. O modelo que sai e

    t_passo  ~  N_estagios * (piso_por_estagio + trabalho / vazao)

e com ele da para dizer, sem escrever o solver, se ha ou nao orcamento.

O kernel e sintetico por necessidade: se usasse a conta real, mediria a
implementacao. Ele faz trabalho DEPENDENTE sobre um vetor de `nv` elementos em
`__local`, que e a forma do problema, e separa os estagios com `barrier()`.
"""
from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pyopencl as cl

RAIZ = Path(__file__).resolve().parents[3]

NV = 72
# Medido em fk_gpu_vs_cpu.json: 12 estagios, 12,3 us (fp32).
ESTAGIOS_FK = 12
US_FK = 12.3
# O passo do MuJoCo sobre este modelo. Ver o cabecalho.
ESTAGIOS_PASSO = 470
US_MJ_STEP = 117.0
US_ORCAMENTO_TEMPO_REAL = 100.0

FONTE = r"""
// K estagios sequenciais sobre um vetor de NV elementos em __local.
// `flops` controla quanto trabalho DEPENDENTE cada estagio carrega: a cadeia
// e serial dentro do estagio, que e a forma de uma eliminacao ou de um passo
// de arvore. Sem a dependencia o compilador vetorizaria e mediriamos vazao,
// que nao e o que limita este problema.
__kernel void estagios(__global float* saida, const int K, const int nv,
                       const int flops) {
    __local float v[256];
    int t = get_local_id(0), W = get_local_size(0);
    for (int i = t; i < nv; i += W) v[i] = 1.0f + 0.001f*i;
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int k = 0; k < K; ++k) {
        for (int i = t; i < nv; i += W) {
            float x = v[i];
            float y = v[(i + 1) % 256];
            for (int f = 0; f < flops; ++f) x = fma(x, 0.999f, 0.001f*y);
            v[i] = x;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (t == 0) saida[0] = v[0];
}
"""


def _dev():
    for p in cl.get_platforms():
        for d in p.get_devices(cl.device_type.GPU):
            return d
    raise SystemExit("sem GPU OpenCL")


def main() -> int:
    dev = _dev()
    ctx = cl.Context([dev])
    fila = cl.CommandQueue(ctx)
    k = cl.Kernel(cl.Program(ctx, FONTE).build(options="-cl-std=CL1.2"),
                  "estagios")
    buf = cl.Buffer(ctx, cl.mem_flags.WRITE_ONLY, size=4)

    def med(K: int, flops: int, W: int, n: int = 100) -> float:
        k.set_args(buf, np.int32(K), np.int32(NV), np.int32(flops))
        for _ in range(10):
            cl.enqueue_nd_range_kernel(fila, k, (W,), (W,))
        fila.finish()
        t0 = time.perf_counter_ns()
        for _ in range(n):
            cl.enqueue_nd_range_kernel(fila, k, (W,), (W,))
        fila.finish()
        return (time.perf_counter_ns() - t0) / n / 1000.0

    r = {
        "gerado_em": datetime.now().isoformat(timespec="seconds"),
        "device": dev.name.strip(),
        "nv": NV,
        "referencias": {
            "cinematica_portada": {
                "estagios": ESTAGIOS_FK, "us": US_FK,
                "us_por_estagio": round(US_FK / ESTAGIOS_FK, 3),
            },
            "mj_step_cpu_us": US_MJ_STEP,
            "orcamento_tempo_real_us": US_ORCAMENTO_TEMPO_REAL,
            "estagios_estimados_por_passo": ESTAGIOS_PASSO,
        },
        "custo_por_estagio": {},
    }

    # O custo marginal de um estagio, por carga de trabalho. Duas medidas (K
    # pequeno e K grande) tiram o custo do dispatch da conta.
    for W in (64, 256):
        for flops in (0, 4, 16, 64):
            a = med(20, flops, W)
            b = med(220, flops, W)
            us = (b - a) / 200.0
            r["custo_por_estagio"][f"W{W}_flops{flops}"] = round(us, 4)

    piso = min(r["custo_por_estagio"].values())
    realista = r["custo_por_estagio"]["W256_flops16"]
    medido_fk = US_FK / ESTAGIOS_FK

    r["projecao"] = {
        "piso_por_estagio_us": piso,
        "estagio_com_trabalho_moderado_us": realista,
        "estagio_medido_na_cinematica_us": round(medido_fk, 3),
        "passo_no_piso_us": round(piso * ESTAGIOS_PASSO, 1),
        "passo_com_trabalho_moderado_us": round(realista * ESTAGIOS_PASSO, 1),
        "passo_na_taxa_da_cinematica_us": round(medido_fk * ESTAGIOS_PASSO, 1),
        "estagios_que_cabem_no_mj_step": int(US_MJ_STEP / max(piso, 1e-9)),
        "estagios_que_cabem_no_tempo_real": int(
            US_ORCAMENTO_TEMPO_REAL / max(piso, 1e-9)),
    }
    p = r["projecao"]
    r["leitura"] = (
        f"No piso absoluto ({piso:.3f} us/estagio, trabalho zero) cabem "
        f"{p['estagios_que_cabem_no_mj_step']} estagios dentro dos "
        f"{US_MJ_STEP:.0f} us do mj_step, contra ~{ESTAGIOS_PASSO} necessarios. "
        f"Na taxa REAL medida na cinematica portada "
        f"({medido_fk:.2f} us/estagio) o passo projetado sai a "
        f"{p['passo_na_taxa_da_cinematica_us']:.0f} us. O piso nao proibe; a "
        f"taxa real, sim -- e a diferenca entre os dois e exatamente o que uma "
        f"implementacao melhor teria que recuperar.")

    dest = RAIZ / "benchmarks" / "physics" / "gpu" / "orcamento_de_estagios.json"
    dest.write_text(json.dumps(r, indent=1), encoding="utf-8")
    print(f"device {r['device']}, nv={NV}\n")
    print("  custo marginal de UM estagio sequencial:")
    for nome, us in r["custo_por_estagio"].items():
        print(f"    {nome:<18} {us:>8.4f} us")
    print()
    print(f"  na cinematica portada (real)   {medido_fk:>8.2f} us/estagio")
    print(f"  piso sintetico                 {piso:>8.4f} us/estagio")
    print()
    print(f"  passo de ~{ESTAGIOS_PASSO} estagios:")
    print(f"    no piso                      {p['passo_no_piso_us']:>8.1f} us")
    print(f"    com trabalho moderado        {p['passo_com_trabalho_moderado_us']:>8.1f} us")
    print(f"    na taxa real da cinematica   {p['passo_na_taxa_da_cinematica_us']:>8.1f} us")
    print(f"    MuJoCo CPU hoje              {US_MJ_STEP:>8.1f} us")
    print(f"\n  {r['leitura']}")
    print(f"\n-> {dest.relative_to(RAIZ)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
