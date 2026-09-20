"""
Backend OpenCL: device, memoria, dispatch e sincronizacao. Nada de ciencia.

PyOpenCL entra aqui como BINDING de host -- o equivalente a usar C++ com a API
de OpenCL. Os kernels sao nossos, em `sim/neural/kernels/*.cl`, e nenhuma regra
cientifica vive dentro de chamada de biblioteca de terceiro. Este arquivo nao
sabe o que e um neurotransmissor, por que o atraso e de 4 passos nem qual
populacao e LC4.

Se um dia trocarmos por Vulkan ou D3D12, o que muda e este arquivo. O
`engine.py` e o `model.py` ficam como estao -- e e exatamente esse o criterio
declarado em `compute/base.py`.

Nesta maquina o device se identifica como `gfx1031` (Radeon RX 6700 XT), 20 CUs,
12 GiB, com atomics em int32 e int64.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from ..model import ESCALA, V_REST, V_RESET, V_TH, Coeficientes, Conectoma, Estado
from .base import BackendIndisponivel

KERNELS = Path(__file__).resolve().parents[1] / "kernels"
ARQUIVOS = ("lif.cl", "synaptic_scatter.cl", "delay.cl", "reductions.cl")
GRUPO = 64


def dispositivos() -> list[str]:
    """O que ha nesta maquina. Nunca levanta -- ausencia e informacao."""
    try:
        import pyopencl as cl
    except ImportError:
        return []
    try:
        return [f"{d.name.strip()} ({p.name.strip()})"
                for p in cl.get_platforms() for d in p.get_devices()]
    except Exception:      # noqa: BLE001 -- driver quebrado nao derruba o import
        return []


class OpenCLBackend:
    nome = "drosobot-opencl"

    def __init__(self, preferir_gpu: bool = True):
        try:
            import pyopencl as cl
        except ImportError as exc:
            raise BackendIndisponivel("pyopencl ausente") from exc
        self.cl = cl
        self.ctx = self._contexto(preferir_gpu)
        self.fila = cl.CommandQueue(self.ctx)
        d = self.ctx.devices[0]
        self._dev = d
        self.device = (f"{d.name.strip()} ({d.max_compute_units} CUs, "
                       f"{d.global_mem_size / 2**30:.1f} GiB)")

        fonte = "\n".join((KERNELS / a).read_text(encoding="utf-8") for a in ARQUIVOS)
        self.prog = cl.Program(self.ctx, fonte).build()
        # Kernel retido uma vez. Buscar pelo atributo a cada chamada cria um
        # objeto novo por despacho, e no caminho quente isso aparece.
        self.k_lif = cl.Kernel(self.prog, "lif_step")
        self.k_scatter = cl.Kernel(self.prog, "scatter")
        self.k_conta = cl.Kernel(self.prog, "acumula_contagem")
        self.k_grupo = cl.Kernel(self.prog, "soma_por_grupo")
        self.k_forcado = cl.Kernel(self.prog, "escreve_forcados_esparso")
        self.k_compacta = cl.Kernel(self.prog, "compacta_spikes")
        self.k_frontier = cl.Kernel(self.prog, "scatter_frontier")
        self.k_coleta = cl.Kernel(self.prog, "coleta_indices")
        self.k_limpa = cl.Kernel(self.prog, "limpa_anel")

        self.c = None
        self.coef = None
        self.grupos = None
        self.n_grupos = 1

    def _contexto(self, preferir_gpu: bool):
        cl = self.cl
        plats = cl.get_platforms()
        if not plats:
            raise BackendIndisponivel("nenhuma plataforma OpenCL")
        ordem = ([cl.device_type.GPU, cl.device_type.CPU] if preferir_gpu
                 else [cl.device_type.CPU, cl.device_type.GPU])
        for tipo in ordem:
            for p in plats:
                try:
                    devs = p.get_devices(device_type=tipo)
                except cl.LogicError:
                    continue
                if devs:
                    return cl.Context(devices=[devs[0]])
        raise BackendIndisponivel("nenhum device OpenCL utilizavel")

    # ------------------------------------------------------------- ciclo

    def prepara(self, conectoma: Conectoma, coef: Coeficientes,
                grupos: np.ndarray | None) -> None:
        cl = self.cl
        mf = cl.mem_flags
        self.c, self.coef, self.grupos = conectoma, coef, grupos
        self.n_grupos = (int(grupos.max()) + 1
                         if grupos is not None and len(grupos) else 1)

        ro = np.ascontiguousarray(conectoma.row_offsets, dtype=np.int32)
        tg = np.ascontiguousarray(conectoma.targets, dtype=np.int32)
        wt = np.ascontiguousarray(conectoma.weights, dtype=np.float32)
        # Um grafo sem aresta e valido (rede isolada, subgrafo degenerado), mas
        # OpenCL recusa buffer de tamanho zero. Um elemento morto custa 4 bytes
        # e evita um caso especial em todo lugar que usa o buffer.
        if tg.size == 0:
            tg = np.zeros(1, dtype=np.int32)
            wt = np.zeros(1, dtype=np.float32)

        self.b_row = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=ro)
        self.b_tgt = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=tg)
        self.b_w = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=wt)
        self.bytes_estaticos = ro.nbytes + tg.nbytes + wt.nbytes

        gr = (np.ascontiguousarray(grupos, dtype=np.int32) if grupos is not None
              else np.full(conectoma.n, -1, dtype=np.int32))
        self.b_grupo = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=gr)
        self.reset()

    def reset(self) -> None:
        cl = self.cl
        mf = cl.mem_flags
        n, D = self.c.n, self.coef.atraso_passos

        def buf(arr, flags=mf.READ_WRITE):
            return cl.Buffer(self.ctx, flags | mf.COPY_HOST_PTR, hostbuf=arr)

        v = np.full(n, V_REST, dtype=np.float32)
        g = np.zeros(n, dtype=np.float32)
        ref = np.full(n, -1, dtype=np.int32)
        spk = np.zeros(n, dtype=np.uint8)
        anel = np.zeros(D * n, dtype=np.int32)
        cont = np.zeros(n, dtype=np.int32)
        soma = np.zeros(self.n_grupos, dtype=np.int32)
        zero = np.zeros(n, dtype=np.float32)
        zero_u8 = np.zeros(n, dtype=np.uint8)

        self.b_v, self.b_g, self.b_ref = buf(v), buf(g), buf(ref)
        self.b_forcado = buf(zero_u8, mf.READ_ONLY)
        self._forcado_zerado = True
        self.b_spk, self.b_anel = buf(spk), buf(anel)
        self.b_cont, self.b_soma = buf(cont), buf(soma)

        # Frontier: lista dos que dispararam no passo. No pior caso todos
        # disparam, entao ela tem n posicoes -- 658 KB no Male CNS inteiro,
        # contra os 195 MB do CSR. O contador e um int so.
        self.b_frontier = buf(np.zeros(n, dtype=np.int32))
        self.b_nfront = buf(np.zeros(1, dtype=np.int32))
        self.b_ext = buf(zero, mf.READ_ONLY)
        self._ext_zerado = True
        self.bytes_dinamicos = (v.nbytes + g.nbytes + ref.nbytes + spk.nbytes
                                + anel.nbytes + cont.nbytes + zero.nbytes
                                + zero_u8.nbytes)

        # os kernels guardam ponteiro de buffer: trocar buffer sem refixar
        # deixaria o kernel escrevendo no buffer velho, em silencio
        self._fixa_args()

    def _fixa_args(self) -> None:
        """
        Prende nos kernels todos os argumentos que NAO mudam entre passos.

        pyopencl remarshala a lista inteira a cada chamada: o LIF tem 18
        argumentos, e sete deles sao buffers e oito sao constantes de modelo
        que nunca mudam durante a corrida. Medido: 247 us por chamada com
        marshalling completo contra 39 us so trocando os dois escalares que
        mudam -- 6,4x, e isso a 2.000 chamadas por segundo simulado.

        Tem que ser chamado depois de TODA troca de buffer (`reset`, e o
        caminho esparso), senao o kernel continua apontando pro buffer velho.
        """
        k = self.coef
        self.k_lif.set_args(
            self.b_v, self.b_g, self.b_ref, self.b_spk, self.b_anel,
            self.b_ext, self.b_forcado,
            np.int32(self.c.n), np.int32(0), np.int32(0),
            np.int32(k.ref_passos),
            np.float32(V_REST), np.float32(V_RESET), np.float32(V_TH),
            np.float32(k.dec_v), np.float32(k.dec_g), np.float32(k.acopla),
            np.float32(1.0 / ESCALA), self.b_nfront)
        self.k_scatter.set_args(
            self.b_spk, self.b_row, self.b_tgt, self.b_w, self.b_anel,
            np.int32(self.c.n), np.int32(0), np.float32(ESCALA))
        self.k_compacta.set_args(self.b_spk, self.b_frontier, self.b_nfront,
                                 np.int32(self.c.n))
        self.k_frontier.set_args(
            self.b_frontier, self.b_nfront, self.b_row, self.b_tgt, self.b_w,
            self.b_anel, np.int32(self.c.n), np.int32(0), np.float32(ESCALA))
        # Tamanho de lancamento FIXO pro scatter balanceado: o kernel percorre
        # a frontier com passo largo, entao nao precisa saber quantos
        # dispararam -- e descobrir isso exigiria trazer o contador pra CPU, ou
        # seja uma sincronizacao por passo.
        self._gl_frontier = (65536,)
        self.k_conta.set_args(self.b_spk, self.b_cont, np.int32(self.c.n))
        if self.grupos is not None and self.n_grupos > 1:
            self.k_grupo.set_args(self.b_spk, self.b_grupo, self.b_soma,
                                  np.int32(self.c.n))
        self._gl_n = self._gl(self.c.n)

    @staticmethod
    def _gl(n: int) -> tuple[int]:
        return (((n + GRUPO - 1) // GRUPO) * GRUPO,)

    # --------------------------------------------------------- primitivas

    def escreve_externo(self, externo_mV: np.ndarray | None) -> None:
        cl = self.cl
        if externo_mV is None:
            if not self._ext_zerado:
                cl.enqueue_copy(self.fila, self.b_ext,
                                np.zeros(self.c.n, dtype=np.float32))
                self._ext_zerado = True
            return
        cl.enqueue_copy(self.fila, self.b_ext,
                        np.ascontiguousarray(externo_mV, dtype=np.float32))
        self._ext_zerado = False

    def prepara_forcados_esparsos(self, indices) -> bool:
        """
        Declara QUAIS neuronios podem receber spike forcado.

        Os indices sao fixos durante a corrida (sao os sensores), entao sobem
        uma vez. Depois disso cada passo manda so os valores -- 311 bytes no
        lugar de 164.451.

        Devolve False se nao houver indices: ai o caminho denso continua.
        """
        cl = self.cl
        mf = cl.mem_flags
        idx = np.ascontiguousarray(indices, dtype=np.int32)
        if idx.size == 0:
            self._idx_forcado = None
            return False
        self._idx_forcado = idx
        self.b_idx_forcado = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR,
                                       hostbuf=idx)
        # buffer de valores reaproveitado: alocar por passo devolveria o
        # overhead que este caminho veio remover
        self.b_val_forcado = cl.Buffer(self.ctx, mf.READ_ONLY, size=int(idx.size))
        self._val_host = np.zeros(idx.size, dtype=np.uint8)
        self.k_forcado.set_args(self.b_forcado, self.b_idx_forcado,
                                self.b_val_forcado, np.int32(idx.size))
        self._gl_forcado = self._gl(int(idx.size))
        # a mascara comeca zerada e SO estes indices serao escritos daqui pra
        # frente, entao o resto permanece zero pra sempre
        cl.enqueue_copy(self.fila, self.b_forcado,
                        np.zeros(self.c.n, dtype=np.uint8))
        self._forcado_zerado = True
        return True

    def escreve_forcados_esparso(self, valores) -> None:
        """Valores nos indices declarados. Nao bloqueia o host."""
        cl = self.cl
        np.copyto(self._val_host, np.asarray(valores, dtype=np.uint8))
        cl.enqueue_copy(self.fila, self.b_val_forcado, self._val_host,
                        is_blocking=False)
        cl.enqueue_nd_range_kernel(self.fila, self.k_forcado, self._gl_forcado,
                                   None)
        self._forcado_zerado = False

    def escreve_forcados(self, forcados) -> None:
        """Mascara de spike forcado deste passo (a fonte Poisson)."""
        cl = self.cl
        if forcados is None:
            if not self._forcado_zerado:
                cl.enqueue_copy(self.fila, self.b_forcado,
                                np.zeros(self.c.n, dtype=np.uint8))
                self._forcado_zerado = True
            return
        cl.enqueue_copy(self.fila, self.b_forcado,
                        np.ascontiguousarray(forcados, dtype=np.uint8))
        self._forcado_zerado = False

    def lif(self, passo: int, cursor: int) -> None:
        # so `passo` e `cursor` mudam; o resto esta preso desde `_fixa_args`
        self.k_lif.set_arg(8, np.int32(passo))
        self.k_lif.set_arg(9, np.int32(cursor))
        self.cl.enqueue_nd_range_kernel(self.fila, self.k_lif, self._gl_n, None)

    def scatter(self, cursor: int) -> None:
        """
        Propaga os spikes. Duas etapas, uma thread por ARESTA.

        O caminho de uma thread por NEURONIO (`k_scatter`, ainda no .cl) fica
        como referencia: com 1.500 disparos em 164.451 neuronios ele deixa
        99,1% das threads saindo na primeira linha e concentra o trabalho nos
        hubs -- um neuronio de 11.203 arestas percorrido por uma thread so.
        Medido: 859 us de atomics contra 19 us de piso.

        O resultado e o mesmo bit a bit: a soma e `atomic_add` em INT, e adicao
        inteira nao depende da ordem das parcelas.
        """
        cl = self.cl
        cl.enqueue_nd_range_kernel(self.fila, self.k_compacta, self._gl_n, None)
        self.k_frontier.set_arg(7, np.int32(cursor))
        cl.enqueue_nd_range_kernel(self.fila, self.k_frontier,
                                   self._gl_frontier, None)

    def acumula(self) -> None:
        cl = self.cl
        cl.enqueue_nd_range_kernel(self.fila, self.k_conta, self._gl_n, None)
        if self.grupos is not None and self.n_grupos > 1:
            cl.enqueue_nd_range_kernel(self.fila, self.k_grupo, self._gl_n, None)

    def sincroniza(self) -> None:
        self.fila.finish()

    # ------------------------------------------------------------ leitura

    def le_indices(self, indices: np.ndarray) -> Estado:
        cl = self.cl
        mf = cl.mem_flags
        idx = np.ascontiguousarray(indices, dtype=np.int32)
        k = len(idx)
        if k == 0:
            z = np.zeros(0, dtype=np.float32)
            return Estado(idx, z, z, np.zeros(0, dtype=np.uint8))

        b_idx = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=idx)
        v_o = np.empty(k, dtype=np.float32)
        g_o = np.empty(k, dtype=np.float32)
        s_o = np.empty(k, dtype=np.uint8)
        b_v = cl.Buffer(self.ctx, mf.WRITE_ONLY, v_o.nbytes)
        b_g = cl.Buffer(self.ctx, mf.WRITE_ONLY, g_o.nbytes)
        b_s = cl.Buffer(self.ctx, mf.WRITE_ONLY, s_o.nbytes)
        self.k_coleta(self.fila, self._gl(k), None,
                      self.b_v, self.b_g, self.b_spk, b_idx, b_v, b_g, b_s,
                      np.int32(k))
        cl.enqueue_copy(self.fila, v_o, b_v)
        cl.enqueue_copy(self.fila, g_o, b_g)
        cl.enqueue_copy(self.fila, s_o, b_s)
        self.fila.finish()
        return Estado(idx, v_o, g_o, s_o)

    def le_contagem_grupos(self) -> np.ndarray:
        saida = np.empty(self.n_grupos, dtype=np.int32)
        self.cl.enqueue_copy(self.fila, saida, self.b_soma)
        self.fila.finish()
        return saida.astype(np.int64)

    def zera_contagem_grupos(self) -> None:
        self.cl.enqueue_copy(self.fila, self.b_soma,
                             np.zeros(self.n_grupos, dtype=np.int32))

    def le_estado_completo(self):
        n = self.c.n
        v = np.empty(n, dtype=np.float32)
        g = np.empty(n, dtype=np.float32)
        s = np.empty(n, dtype=np.uint8)
        cl = self.cl
        cl.enqueue_copy(self.fila, v, self.b_v)
        cl.enqueue_copy(self.fila, g, self.b_g)
        cl.enqueue_copy(self.fila, s, self.b_spk)
        self.fila.finish()
        return v, g, s

    def le_contagem_total(self) -> np.ndarray:
        saida = np.empty(self.c.n, dtype=np.int32)
        self.cl.enqueue_copy(self.fila, saida, self.b_cont)
        self.fila.finish()
        return saida.astype(np.int64)

    def resumo(self) -> dict:
        d = self._dev
        return {
            "backend": self.nome,
            "device": self.device,
            "api": "OpenCL",
            "precision": "fp32 (acumulacao em ponto fixo)",
            "compute_units": int(d.max_compute_units),
            "vram_mib": round((self.bytes_estaticos + self.bytes_dinamicos)
                              / 1048576, 1),
            "static_mib": round(self.bytes_estaticos / 1048576, 1),
        }
