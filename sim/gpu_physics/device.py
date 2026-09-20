r"""
Device, buffers e dispatch. Nada de fisica.

O mesmo criterio de `sim/neural/compute/opencl.py`, aplicado do lado do corpo:
este arquivo nao sabe o que e uma junta, uma inercia nem um contato. Ele sabe
abrir um device, subir array, despachar kernel e esperar.

## Por que OpenCL, e nao D3D12 ou Vulkan

A escolha nao e de gosto, e tem uma medida por tras de cada metade dela.

**Contra o despacho por passo, qualquer que seja a API.** Medido nesta placa
(`benchmarks/physics/gpu/piso_de_latencia.json`): um despacho sincrono custa
84 us; o orcamento de tempo real com dt=1e-4 e 100 us/passo. Nenhuma API salva
um desenho com a CPU esperando a GPU a cada passo de fisica. O que salva e
manter o estado no device e rodar o laco por dentro do kernel -- onde uma
iteracao com barreira custa 0,076 us, mil vezes menos.

**A favor do OpenCL, dado isso.** O motor neural deste projeto ja e OpenCL e ja
e residente na GPU. Fisica no MESMO contexto significa que o handoff
`motor -> corpo -> sensor -> motor` e uma troca de ponteiro entre buffers do
mesmo device, sem interop, sem copia e sem passar pela CPU. Fisica em D3D12 com
cerebro em OpenCL exigiria uma camada de interop que nao existe, para ganhar
uma latencia de despacho que este desenho ja nao paga.

Os outros criterios ficam empatados ou a favor: atomics e barreiras de
work-group existem nos tres; AMD e NVIDIA rodam OpenCL em Windows e Linux sem
CUDA; e o host em Python ja esta validado aqui.

O que se perde e honesto declarar: OpenCL 2.0 e o teto da AMD no Windows, nao
ha depuracao decente, e portar para Vulkan depois significa traduzir kernels
(via SPIR-V), nao so trocar este arquivo. A fronteira deste modulo esconde o
host, nao a linguagem dos kernels.

## Precisao

O programa e compilado duas vezes, com `-D USA_FP64` e sem. Nao ha conversao
silenciosa: `Device.programa(fp64=True|False)` diz qual esta em uso, e o
metadata registra. A RX 6700 XT faz FP64 a 1/16 da taxa de FP32, entao a
diferenca de custo e real e tem que ser medida, nao suposta.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

KERNELS = Path(__file__).resolve().parent / "kernels"


class DeviceIndisponivel(RuntimeError):
    """Sem GPU OpenCL utilizavel. Mensagem com o que fazer, nunca silencio."""


class Device:
    """Um contexto, uma fila, um cache de programas compilados."""

    def __init__(self, preferir_gpu: bool = True):
        try:
            import pyopencl as cl
        except ImportError as exc:                              # noqa: BLE001
            raise DeviceIndisponivel(
                "pyopencl ausente. `pip install pyopencl` no venv que roda a "
                "fisica.") from exc
        self.cl = cl
        tipo = cl.device_type.GPU if preferir_gpu else cl.device_type.ALL
        dispositivos = [d for p in cl.get_platforms() for d in p.get_devices(tipo)]
        if not dispositivos:
            raise DeviceIndisponivel(
                "nenhum device OpenCL encontrado. Em AMD/Windows isso costuma "
                "ser driver sem o runtime OpenCL instalado.")
        self.dev = dispositivos[0]
        self.ctx = cl.Context([self.dev])
        self.fila = cl.CommandQueue(self.ctx)
        self._programas: dict[tuple[str, bool], object] = {}
        self._kernels: dict[tuple[str, bool, str], object] = {}

    # ----------------------------------------------------------- capacidades

    @property
    def tem_fp64(self) -> bool:
        return "cl_khr_fp64" in self.dev.extensions

    def capacidades(self) -> dict:
        """Vai para o metadata da corrida. Quem le precisa saber onde rodou."""
        d = self.dev
        return {
            "nome": d.name.strip(),
            "vendor": d.vendor.strip(),
            "versao_opencl": d.version.strip(),
            "driver": d.driver_version.strip(),
            "compute_units": int(d.max_compute_units),
            "clock_mhz": int(d.max_clock_frequency),
            "max_work_group_size": int(d.max_work_group_size),
            "local_mem_bytes": int(d.local_mem_size),
            "global_mem_bytes": int(d.global_mem_size),
            "fp64": self.tem_fp64,
        }

    # -------------------------------------------------------------- programas

    def programa(self, arquivo, fp64: bool, defines: tuple = ()):
        """
        Compila (uma vez) `kernels/<arquivo>` no modo de precisao pedido.

        `arquivo` pode ser uma tupla de arquivos, concatenados na ordem. O
        OpenCL nao tem `#include` entre unidades de traducao, e duplicar as
        primitivas de quaternio num segundo arquivo criaria duas copias da
        mesma conta, livres para divergir.

        `defines` sao `-D NOME=valor`. Existem porque `__local` de tamanho fixo
        precisa das dimensoes do modelo em tempo de compilacao, e um `__local`
        dimensionado por argumento perde o endereco estatico -- que e o motivo
        de usar `__local` em primeiro lugar.
        """
        arquivos = (arquivo,) if isinstance(arquivo, str) else tuple(arquivo)
        chave = (arquivos, fp64, defines)
        if chave in self._programas:
            return self._programas[chave]
        if fp64 and not self.tem_fp64:
            raise DeviceIndisponivel(
                f"{self.dev.name.strip()} nao expoe cl_khr_fp64; use fp64=False "
                "e leia a divergencia medida em docs/GPU_PHYSICS.md.")
        fonte = chr(10).join((KERNELS / a).read_text(encoding="utf-8")
                             for a in arquivos)
        opcoes = "-cl-std=CL1.2"
        if fp64:
            opcoes += " -D USA_FP64"
        for nome, valor in defines:
            opcoes += f" -D {nome}={valor}"
        prog = self.cl.Program(self.ctx, fonte).build(options=opcoes)
        self._programas[chave] = prog
        return prog

    def kernel_proprio(self, arquivo, nome: str, fp64: bool,
                       defines: tuple = ()):
        """
        Um `cl.Kernel` NOVO, do programa ja compilado.

        O programa e caro e fica no cache; o objeto de kernel e barato e NAO
        pode ser compartilhado por dois motores. Os argumentos ligados vivem
        dentro dele: quem liga por ultimo ganha. Com cache de argumentos por
        motor, o outro acha que ja ligou e despacha com os buffers alheios --
        o sintoma foi a cinematica de dois motores no mesmo device divergindo
        no segundo passo com o mesmo `qpos`.
        """
        arquivos = (arquivo,) if isinstance(arquivo, str) else tuple(arquivo)
        return self.cl.Kernel(self.programa(arquivos, fp64, defines), nome)

    def kernel(self, arquivo, nome: str, fp64: bool, defines: tuple = ()):
        """
        Um objeto `cl.Kernel` por nome, reaproveitado.

        Recuperar pelo atributo (`prog.nome(...)`) cria um kernel novo por
        chamada. Custa centenas de microssegundos, e foi assim que a primeira
        versao do benchmark de latencia mediu 250 us por despacho nulo.
        """
        arquivos = (arquivo,) if isinstance(arquivo, str) else tuple(arquivo)
        chave = (arquivos, fp64, nome, defines)
        if chave not in self._kernels:
            self._kernels[chave] = self.cl.Kernel(
                self.programa(arquivos, fp64, defines), nome)
        return self._kernels[chave]

    # ---------------------------------------------------------------- memoria

    def sobe(self, a: np.ndarray, somente_leitura: bool = True):
        """Array do host -> buffer do device. Chamado no setup, nunca no laco."""
        mf = self.cl.mem_flags
        flags = (mf.READ_ONLY if somente_leitura else mf.READ_WRITE) | mf.COPY_HOST_PTR
        return self.cl.Buffer(self.ctx, flags, hostbuf=np.ascontiguousarray(a))

    def vazio(self, n: int, dtype):
        buf = self.cl.Buffer(self.ctx, self.cl.mem_flags.READ_WRITE,
                             size=max(1, int(n)) * np.dtype(dtype).itemsize)
        self.zera(buf, max(1, int(n)), dtype)
        return buf

    def zera(self, buf, n: int, dtype) -> None:
        self.cl.enqueue_fill_buffer(self.fila, buf, np.dtype(dtype).type(0), 0,
                                    int(n) * np.dtype(dtype).itemsize)

    def escreve(self, buf, a: np.ndarray) -> None:
        self.cl.enqueue_copy(self.fila, buf, np.ascontiguousarray(a))

    def le(self, buf, n: int, dtype) -> np.ndarray:
        saida = np.empty(int(n), dtype=dtype)
        self.cl.enqueue_copy(self.fila, saida, buf)
        return saida

    # --------------------------------------------------------------- dispatch

    def roda(self, kernel, global_size: int, local_size: int | None = None,
             args: tuple = ()) -> None:
        if args:
            kernel.set_args(*args)
        gs = (int(global_size),)
        ls = (int(local_size),) if local_size else None
        self.cl.enqueue_nd_range_kernel(self.fila, kernel, gs, ls)

    def espera(self) -> None:
        self.fila.finish()
