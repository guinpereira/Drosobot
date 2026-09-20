"""
Drosobot Lab: um comando, o laboratorio inteiro.

    .venv-flygym2\\Scripts\\python sim\\drosobot_lab.py ^
        --physics flygym2 --neural opencl --cns whole ^
        --experiment looming --telemetry --duracao 600

    python sim/drosobot_lab.py --info          # so imprime o que seria montado
    python sim/drosobot_lab.py --lista         # catalogo de experimentos

    PhysicsAdapter          <- autoridade da fisica (FlyGym 1 ou 2 / MuJoCo)
        SensorFrame
            v
    Drosobot Neural Engine  <- autoridade do comportamento (CPU, OpenCL, D3D12)
        Male CNS ou circuito
            v
        MotorFrame
            v
    PhysicsAdapter
        (repete)

        e em paralelo: telemetria -> Unity Drosobot Lab
                       controle   <- Unity Drosobot Lab

Este arquivo NAO conhece a API do FlyGym. Toda a fisica passa por
`sim/physics/`, e todo o cerebro por `sim/neural/`. Trocar de simulador ou de
backend de GPU nao mexe aqui.

## Quem manda em que

    MuJoCo / FlyGym    fisica. Posicao, contato, retina saem do SensorFrame.
    conectoma          comportamento. O drive motor sai de spike, sempre.
    controle (Unity)   qual experimento, com que semente, e quando comeca.

A terceira linha nao encosta nas outras duas: nao existe comando que altere
peso, limiar, entrada ou drive. Ver `sim/telemetry/control.py`.

## ESCOPO -- a distincao que nunca pode sumir

    WHOLE CONNECTOME SIMULATED     164.451 neuronios, 25,5M arestas participam
    FULL FUNCTIONAL BRAIN          NAO

So a via de looming (LC4/LPLC2) e a motora (DNp01, TTMn) tem semantica
sensorial/motora modelada. O resto participa pelo que chega pela conectividade.
As duas coisas sao diferentes, e o runtime reporta as duas separadas -- em
`--info`, na telemetria e na interface.

## Sem entrada artificial

Os 311 sensores sao a unica origem. Nao ha entrada tonica global; o
`TONIC_INHIB_HZ` dos experimentos de circuito NAO e usado aqui e nao deve
voltar -- no conectoma inteiro os inibitorios tem fonte de verdade.

## Sem readback do cerebro

v, g, refratario e o anel ficam na GPU. A CPU le, por janela: a saida motora, os
papeis declarados, o balanco no Giant Fiber e a atividade agregada por
populacao. O conectoma inteiro nunca volta pra CPU.
"""
from __future__ import annotations

import argparse
import csv
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np

AQUI = Path(__file__).resolve().parent
RAIZ = AQUI.parent
sys.path.insert(0, str(AQUI))

from profiler import Profiler  # noqa: E402
from transducao import TransducaoLooming, TransducaoOptomotor  # noqa: E402

PAPEIS_JSON = RAIZ / "connectome" / "gf_roles.json"
PAPEIS_OPTO_JSON = RAIZ / "connectome" / "opto_roles.json"
PROPRIEDADES_CSV = RAIZ / "connectome" / "neuron_properties.csv"

# transducao retina -> taxa: escolha NOSSA (ASSUMPTION), nao esta no conectoma
DARK_THRESHOLD = 0.4
LOOM_GAIN = 240.0
LOOM_MAX_HZ = 20.0
VISION_HZ = 100
JANELA_MS = 10.0
DT = 1e-4
BASE_DRIVE, ESCAPE_DRIVE, ESCAPE_MS = 1.0, -0.5, 120.0
# Giro por desequilibrio entre os DNa02. ASSUMPTION: o conectoma nao diz quanto
# um spike descendente vale em velocidade de perna. O sinal segue a convencao
# de `TransducaoOptomotor.hz_por_lado`, que tambem e declarada.
GANHO_GIRO = 0.25
GIRO_MAX = 0.8
# Ganho e teto da transducao optomotora, em Hz por unidade de fluxo. ASSUMPTION.
OPTO_GANHO_HZ = 4.0
OPTO_MAX_HZ = 20.0
# pose vai na cadencia da VISUALIZACAO, nao na da fisica
POSE_HZ = 30
# a retina sao 1442 floats: a mensagem mais cara, entao vai a cada N janelas
RETINA_A_CADA = 5
# Aviso de limitacao do modelo. O LIF de Shiu et al. nao tem potencial de
# reversao inibitorio, entao inibicao convergente forte leva v a valores nao
# fisiologicos. NAO ha clamp: o valor vai como veio, e o aviso e so registro.
V_AVISO_MV = -150.0

# Catalogo do que ESTE runtime sabe rodar. E o que vai pro seletor da Unity.
# Nao ha entrada aqui sem implementacao atras -- prometer experimento que nao
# roda e pior que nao oferecer.
CATALOGO = [
    {
        "id": "looming_whole",
        "name": "Looming -- Male CNS inteiro",
        "description": ("Esfera se aproximando. LC4/LPLC2 recebem Poisson pela "
                        "retina, o conectoma inteiro propaga, o TTMn decide a "
                        "fuga. 164.451 neuronios na GPU."),
        "arena": "looming", "cns": "whole",
    },
    {
        "id": "looming_circuit",
        "name": "Looming -- circuito do Giant Fiber",
        "description": ("Mesmo estimulo, so o circuito do GF: 1.261 neuronios. "
                        "E a comparacao que mostra o gate inibitorio."),
        "arena": "looming", "cns": "circuit",
    },
    {
        "id": "optomotor_whole",
        "name": "Optomotor -- Male CNS inteiro",
        "description": ("Tambor de postes girando em torno da mosca. O fluxo "
                        "optico horizontal excita os T4/T5 de um lado, HS "
                        "propaga, e o desequilibrio entre os DNa02 vira giro. "
                        "Via e transducao diferentes das de looming."),
        "arena": "optomotor", "cns": "whole",
    },
    {
        "id": "optomotor_circuit",
        "name": "Optomotor -- circuito T4/T5 -> HS -> DNa02",
        "description": ("Mesmo estimulo, so a via optomotora isolada. E a "
                        "comparacao que diz o que o resto do conectoma faz "
                        "com o sinal de giro."),
        "arena": "optomotor", "cns": "circuit",
    },
    {
        "id": "obstaculos_whole",
        "name": "Campo de obstaculos -- Male CNS inteiro",
        "description": ("Pilares fixos no caminho. Usa a MESMA via de looming: "
                        "um obstaculo que se aproxima e, para a retina, um "
                        "estimulo em expansao. A pergunta e se o reflexo de "
                        "fuga guia desvio, nao so reacao."),
        "arena": "obstaculos", "cns": "whole",
    },
    {
        "id": "obstaculos_circuit",
        "name": "Campo de obstaculos -- circuito do Giant Fiber",
        "description": ("Mesmos pilares, so o circuito do GF. E a comparacao "
                        "que diz se o gate inibitorio muda o desvio."),
        "arena": "obstaculos", "cns": "circuit",
    },
    {
        "id": "flat_whole",
        "name": "Marcha livre -- Male CNS inteiro",
        "description": ("Chao plano, sem estimulo. Linha de base: o que a rede "
                        "faz sem nada chegando pela retina."),
        "arena": "flat", "cns": "whole",
    },
]


# Que populacoes cada via declara. O resto do conectoma participa pela
# conectividade, sem semantica sensorial ou motora modelada.
PAPEIS_POR_VIA = {
    "looming": ("LC4/LPLC2", "DNp01", "TTMn"),
    "optomotor": ("T4T5_L", "T4T5_R", "HS_L", "HS_R", "DNa02_L", "DNa02_R",
                  "opto_motor"),
}


def _ids_da_via(via: str) -> dict:
    caminho = PAPEIS_OPTO_JSON if via == "optomotor" else PAPEIS_JSON
    return json.loads(caminho.read_text(encoding="utf-8"))


def monta_cerebro(escopo: str, backend: str, via: str = "looming"):
    """
    Devolve (engine, papeis, nomes_grupos). `papeis`: nome -> indices.

    `via` escolhe QUAIS populacoes tem semantica declarada -- a de looming
    (LC4/LPLC2 -> DNp01 -> TTMn) ou a optomotora (T4/T5 -> HS -> DNa02). O
    conectoma carregado e o mesmo; o que muda e quem recebe entrada sensorial e
    de quem se le saida motora.
    """
    from neural import NeuralEngine, carrega_male_cns, subgrafo

    ids = _ids_da_via(via)
    nomes_papeis = PAPEIS_POR_VIA[via]
    c = carrega_male_cns()

    if escopo == "circuit":
        if via == "optomotor":
            # o subgrafo da via optomotora e a uniao das suas populacoes: nao
            # ha lista de upstream publicada como a do GF
            alvo = np.unique(np.concatenate(
                [np.asarray(ids[n], dtype=np.int64) for n in nomes_papeis]))
        else:
            alvo = np.unique(np.concatenate([
                np.asarray(ids["upstream_gf_total"], dtype=np.int64),
                np.asarray(ids["DNp01"], dtype=np.int64),
                np.asarray(ids["TTMn"], dtype=np.int64)]))
        idx = c.indice_de(alvo)
        c = subgrafo(c, idx[idx >= 0])

    grupos = np.full(c.n, -1, dtype=np.int32)
    nomes, papeis = [], {}
    for nome in nomes_papeis:
        idx = c.indice_de(ids[nome])
        idx = idx[idx >= 0]
        papeis[nome] = idx
        if len(idx):
            grupos[idx] = len(nomes)
            nomes.append(nome)

    eng = NeuralEngine(c, backend=backend, grupos=grupos, nomes_grupos=nomes)
    return eng, papeis, nomes


def nomes_por_body_id() -> dict[int, str]:
    """
    bodyId -> tipo (`DNp01`, `GNG300`, `SAD073`...).

    So pra ROTULAR o que ja foi medido. O gate e calculado dos pesos e dos
    spikes; este mapa nunca entra na conta -- se o CSV faltar, o numero
    continua o mesmo e o rotulo vira o proprio bodyId.
    """
    if not PROPRIEDADES_CSV.exists():
        return {}
    mapa = {}
    with PROPRIEDADES_CSV.open(encoding="utf-8", newline="") as f:
        for linha in csv.DictReader(f):
            tipo = (linha.get("type") or linha.get("instance") or "").strip()
            if not tipo:
                continue
            try:
                mapa[int(linha["bodyId"])] = tipo
            except (KeyError, ValueError):
                continue
    return mapa


def entradas_do_gf(c, idx_gf):
    """
    Quem entra no Giant Fiber, com peso e sinal.

    Sem saber QUEM entra nao da pra mostrar de onde vem a inibicao. Calculado
    uma vez: e uma varredura do CSR, e o grafo nao muda durante a corrida.
    """
    alvo = set(int(i) for i in idx_gf)
    pre, peso = [], []
    ro, tg, w = c.row_offsets, c.targets, c.weights
    for i in range(c.n):
        a, b = ro[i], ro[i + 1]
        if b <= a:
            continue
        m = np.isin(tg[a:b], list(alvo))
        if m.any():
            pre.append(np.full(int(m.sum()), i, dtype=np.int64))
            peso.append(w[a:b][m])
    if not pre:
        return np.zeros(0, np.int32), np.zeros(0, np.float32)
    return (np.concatenate(pre).astype(np.int32),
            np.concatenate(peso).astype(np.float32))


def indices_por_tipo(eng, gf_pre, nomes):
    """
    Para cada aresta que entra no GF, a que TIPO o pre-sinaptico pertence.

    Calculado uma vez. Permite somar a contribuicao por populacao com um
    `bincount` por janela em vez de um dicionario montado na mao -- e e o que
    responde "de onde vem a inibicao" com numero, nao com os cinco maiores do
    instante.
    """
    if not len(gf_pre):
        return np.zeros(0, np.int32), []
    rotulos, idx = [], np.zeros(len(gf_pre), dtype=np.int32)
    pos = {}
    for k, i in enumerate(gf_pre):
        bid = int(eng.c.body_ids[i])
        nome = (nomes or {}).get(bid, str(bid))
        if nome not in pos:
            pos[nome] = len(rotulos)
            rotulos.append(nome)
        idx[k] = pos[nome]
    return idx, rotulos


def mede_gate(eng, idx_gf, gf_pre, gf_peso, nomes=None, tipo_idx=None,
              n_tipos=0):
    """
    Balanco de entrada no Giant Fiber neste passo.

    Dado real, medido do estado da GPU -- nao animacao decorativa. E o que
    permite VER o gate acontecer: no circuito o liquido fica proximo de zero e
    o GF dispara; no conectoma inteiro a inibicao multiplica e o liquido troca
    de sinal.
    """
    if not len(idx_gf):
        return {"exc_mV": 0.0, "inib_mV": 0.0, "liquido_mV": 0.0,
                "v_min_mV": 0.0, "spikes_gf": 0, "top_inib": [],
                "por_tipo": None}
    est_gf = eng.le(idx_gf)
    exc = inib = 0.0
    top = []
    por_tipo = None
    if len(gf_pre):
        est_pre = eng.le(gf_pre)
        disp = est_pre.spike.astype(bool)
        if disp.any():
            contrib = gf_peso[disp]
            exc = float(contrib[contrib > 0].sum())
            inib = float(contrib[contrib < 0].sum())
            if tipo_idx is not None and n_tipos:
                # contribuicao por populacao, com sinal: uma passada sobre as
                # 1.455 arestas que entram no GF
                por_tipo = np.bincount(tipo_idx[disp], weights=contrib,
                                       minlength=n_tipos)
            neg = np.flatnonzero(disp & (gf_peso < 0))
            if len(neg):
                for k in neg[np.argsort(gf_peso[neg])][:5]:
                    bid = int(eng.c.body_ids[gf_pre[k]])
                    top.append({"body_id": bid,
                                "type": (nomes or {}).get(bid, str(bid)),
                                "mV": round(float(gf_peso[k]), 2)})
    return {"exc_mV": round(exc, 2), "inib_mV": round(inib, 2),
            "liquido_mV": round(exc + inib, 2),
            "v_min_mV": round(float(est_gf.v_mV.min()), 2),
            "spikes_gf": int(est_gf.spike.sum()), "top_inib": top,
            "por_tipo": por_tipo}


def dispositivo() -> dict:
    """
    Quem esta rodando isto. Vai pra telemetria e pro painel.

    A GPU nao e perguntada aqui: quem sabe o dispositivo de compute e o
    backend, e ele ja reporta em `engine.resumo()`. Duplicar a deteccao daria
    duas respostas possiveis pra mesma pergunta.
    """
    return {
        "os": f"{platform.system()} {platform.release()}",
        "cpu": platform.processor() or platform.machine(),
        "python": platform.python_version(),
    }


class Laboratorio:
    """
    Estado do laboratorio: idle -> loading -> running <-> paused -> finished.

    O cerebro e montado uma vez POR ESCOPO: ler os CSR do Male CNS e subir
    25,5M arestas pra GPU leva segundos, e refazer isso a cada troca de
    experimento tornaria o seletor inutil. Trocar de experimento dentro do
    mesmo escopo so reseta o estado dinamico. O CORPO e remontado sempre que a
    arena muda, porque arena diferente e modelo diferente.
    """

    def __init__(self, args, tel, ctl, protocol, registro=None,
                 estimulo=None):
        self.args = args
        # Gravacao estruturada da corrida. Independente da telemetria: uma
        # bateria de experimentos roda sem interface nenhuma aberta, e e
        # justamente ela que precisa deixar dado em disco.
        self.registro = registro
        self.params_estimulo = dict(estimulo or {})
        self.tel = tel
        self.ctl = ctl
        self.protocol = protocol
        self.estado = "idle"
        self.sair = False
        self.seed = args.seed
        self.exp_id = None
        self.corpo = None
        self.eng = None
        self.escopo_montado = None
        self.via_montada = None
        self.arena_montada = None
        self.nomes_tipo = nomes_por_body_id()
        self.passo_atual = 0
        self.t_s = 0.0
        self.escapes = 0
        self.hz = 0.0
        self.spikes_sensoriais = 0
        self.fluxo_optico = 0.0
        self.desequilibrio = 0
        self._looming_ativo = False
        self.gf_acumulado = self._zera_acumulado()
        self.prof = Profiler(["physics", "vision", "neural", "leitura",
                              "telemetry"])

    # ------------------------------------------------------------- montagem

    @staticmethod
    def item(exp_id):
        for e in CATALOGO:
            if e["id"] == exp_id:
                return e
        return None

    def _cerebro_para(self, escopo: str, via: str = "looming"):
        """Monta o cerebro do escopo e da via pedidos, reaproveitando se der."""
        if (self.eng is not None and self.escopo_montado == escopo
                and self.via_montada == via):
            self.eng.reset()
            return
        t0 = time.perf_counter()
        self.eng, self.papeis, self.nomes_grupos = monta_cerebro(
            escopo, self.args.neural, via=via)
        self.escopo_montado = escopo
        self.via_montada = via
        self.idx_gf = self.papeis.get("DNp01", np.zeros(0, np.int32))
        self.gf_pre, self.gf_peso = entradas_do_gf(self.eng.c, self.idx_gf)
        self.gf_tipo_idx, self.gf_tipos = indices_por_tipo(
            self.eng, self.gf_pre, self.nomes_tipo)
        r = self.eng.resumo()
        print(f"  cerebro   {r['neurons_simulated']:,} neuronios, "
              f"{r['edges_simulated']:,} arestas, {r['backend']} em "
              f"{r['device']}   ({time.perf_counter() - t0:.1f} s)")

    def _corpo_para(self, arena: str, seed: int):
        """Remonta o corpo so quando a arena muda; senao, reset e barato."""
        from physics import cria

        t0 = time.perf_counter()
        if self.corpo is not None and self.arena_montada == arena:
            self.frame = self.corpo.reset(seed=seed)
            return
        if self.corpo is not None:
            self.corpo.fecha()
        self.corpo = cria(self.args.physics, arena=arena,
                          self_collisions=self.args.colisao,
                          timestep=DT, com_visao=True,
                          estimulo=self.params_estimulo)
        self.arena_montada = arena
        self.frame = self.corpo.reset(seed=seed)
        rc = self.corpo.resumo()
        print(f"  corpo     {rc['physics_backend']}, arena={arena}, "
              f"{rc['collision_pairs']} pares, nv={rc['nv']}   "
              f"({time.perf_counter() - t0:.1f} s)")

    def monta(self, exp_id: str, seed: int) -> bool:
        item = self.item(exp_id)
        if item is None:
            self._falha(f"experimento desconhecido: {exp_id!r}")
            return False
        self.estado = "loading"
        self.exp_id, self.seed = exp_id, seed
        self._publica_estado(message="montando cerebro, arena e mosca")
        via = "optomotor" if item["arena"] == "optomotor" else "looming"
        try:
            self._cerebro_para(item["cns"], via)
            self._corpo_para(item["arena"], seed)
        except Exception as e:                                # noqa: BLE001
            self._falha(f"{type(e).__name__}: {e}")
            return False

        # estado do laco, zerado junto com o experimento
        self.via = via
        if via == "optomotor":
            self.sens_l = self.papeis.get("T4T5_L", np.zeros(0, np.int32))
            self.sens_r = self.papeis.get("T4T5_R", np.zeros(0, np.int32))
            self.sens = np.concatenate([self.sens_l, self.sens_r]).astype(np.int64)
            self.motor_l = self.papeis.get("DNa02_L", np.zeros(0, np.int32))
            self.motor_r = self.papeis.get("DNa02_R", np.zeros(0, np.int32))
            self.motor = np.concatenate([self.motor_l, self.motor_r]).astype(np.int64)
            self.transducao = TransducaoOptomotor(ganho_hz_por_unidade=OPTO_GANHO_HZ,
                                                  teto_hz=OPTO_MAX_HZ)
            self.transducao.prepara()
        else:
            self.sens = self.papeis.get("LC4/LPLC2", np.zeros(0, np.int32))
            self.motor = self.papeis.get("TTMn", np.zeros(0, np.int32))
            self.transducao = TransducaoLooming(DARK_THRESHOLD, LOOM_GAIN,
                                                LOOM_MAX_HZ, VISION_HZ)
        self.taxas = np.zeros(self.eng.c.n, dtype=np.float64)
        # Semente do EXPERIMENTO, nao semente global: a realizacao de Poisson
        # tem que ser reproduzivel por corrida pra que duas corridas com a
        # mesma semente sejam comparaveis.
        self.rng = np.random.default_rng(seed)
        self.escuro_lento = None
        self.drive = np.array([BASE_DRIVE, BASE_DRIVE])
        self.escape_ate, self.escapes = -1.0, 0
        self.passo_atual, self.t_s = 0, 0.0
        self.hz, self.janelas = 0.0, 0
        self.spikes_sensoriais = 0
        self.fluxo_optico = 0.0
        self.desequilibrio = 0
        self._looming_ativo = False
        self.gf_acumulado = self._zera_acumulado()
        self.proxima_pose = 0.0
        self.avisou_v = False
        self.prof = Profiler(["physics", "vision", "neural", "leitura",
                              "telemetry"])
        self._abertura()
        self.estado = "running"
        self._publica_estado()
        return True

    def _zera_acumulado(self) -> dict:
        """
        Agregados da corrida inteira.

        Acumular aqui, e nao reprocessar a serie temporal depois, e o que
        permite comparar dez corridas sem reabrir dez CSV -- e o que garante
        que o resumo e do mesmo laco que produziu os eventos.
        """
        n = len(getattr(self, "gf_tipos", []) or [])
        return {"spikes": 0, "exc": 0.0, "inib": 0.0, "v_min": 0.0,
                "sensoriais": 0, "ttmn": 0, "hz_max": 0.0,
                "soma_tipo": np.zeros(n, dtype=np.float64), "por_tipo": {}}

    def _acumula(self, gate, ttmn) -> None:
        a = self.gf_acumulado
        a["spikes"] += gate["spikes_gf"]
        a["exc"] += gate["exc_mV"]
        a["inib"] += gate["inib_mV"]
        a["v_min"] = min(a["v_min"], gate["v_min_mV"])
        a["sensoriais"] += self.spikes_sensoriais
        a["ttmn"] += ttmn
        a["hz_max"] = max(a["hz_max"], self.hz)
        if gate.get("por_tipo") is not None and len(a["soma_tipo"]):
            a["soma_tipo"] += gate["por_tipo"]

    def _fecha_acumulado(self) -> None:
        """Converte a soma por tipo num dicionario ordenado por magnitude."""
        a = self.gf_acumulado
        if not len(a["soma_tipo"]):
            return
        ordem = np.argsort(a["soma_tipo"])
        a["por_tipo"] = {self.gf_tipos[i]: round(float(a["soma_tipo"][i]), 2)
                         for i in ordem if abs(a["soma_tipo"][i]) > 0.005}

    # -------------------------------------------------------------- comandos

    def trata(self, msg):
        cmd = msg.get("command")
        sock = msg.get("_sock")
        resp = self.ctl.responder

        if cmd == "list":
            self._publica_catalogo()
            resp(sock, True, experiments=CATALOGO, current=self.exp_id,
                 state=self.estado)

        elif cmd == "select":
            alvo = msg.get("experiment_id")
            if self.item(alvo) is None:
                resp(sock, False, error=f"id desconhecido: {alvo!r}")
                return
            self.exp_id = alvo
            self.seed = int(msg.get("seed", self.seed))
            self.estado = "idle"
            self._publica_estado()
            self._publica_catalogo()
            resp(sock, True, experiment_id=self.exp_id, seed=self.seed)

        elif cmd == "start":
            alvo = msg.get("experiment_id") or self.exp_id
            if self.item(alvo) is None:
                resp(sock, False, error="nenhum experimento selecionado")
                return
            seed = int(msg.get("seed", self.seed))
            # O ack vai ANTES de montar: subir a arena (e, na primeira vez, o
            # conectoma) leva segundos, e segurar a conexao de controle nesse
            # tempo faz o cliente estourar o timeout achando que morreu.
            self.estado = "loading"
            self._publica_estado(message="montando")
            resp(sock, True, experiment_id=alvo, seed=seed, state="loading")
            self.monta(alvo, seed)

        elif cmd == "pause":
            if self.estado == "running":
                self.estado = "paused"
                self.prof.pausa()
                self._publica_estado()
            resp(sock, self.estado == "paused", state=self.estado)

        elif cmd == "resume":
            if self.estado == "paused":
                self.estado = "running"
                self.prof.retoma()
                self._publica_estado()
            resp(sock, self.estado == "running", state=self.estado)

        elif cmd == "reset":
            seed = int(msg.get("seed", self.seed))
            alvo = self.exp_id or CATALOGO[0]["id"]
            resp(sock, True, seed=seed, state="loading")
            if self.monta(alvo, seed):
                self.tel.enviar(self.protocol.event(0.0, "experiment_reset",
                                                    {"seed": seed}))

        elif cmd == "stop":
            self._encerra("parado pela interface")
            resp(sock, True, state=self.estado)

        elif cmd == "quit":
            resp(sock, True)
            self._encerra("processo encerrado")
            self.sair = True

    # ----------------------------------------------------------------- laco

    def passo(self) -> bool:
        """Um passo de fisica. Devolve True se uma janela neural rodou."""
        from physics import MotorFrame

        self.t_s = self.passo_atual * DT
        self.corpo.antes_do_passo(self.t_s)

        with self.prof("physics", "passo de fisica"):
            self.frame = self.corpo.passo(MotorFrame(drive=self.drive))
        self.prof.avanca_sim(DT * 1000)
        self.passo_atual += 1

        frame = self.frame
        if not frame.retina_atualizou or frame.retina is None:
            return False

        with self.prof("vision", "quadro de retina"):
            self.taxas[:] = 0.0
            if self.via == "optomotor":
                fluxo, escuro = self.transducao.taxa(frame.retina)
                hz_l, hz_r = self.transducao.hz_por_lado(fluxo)
                if len(self.sens_l):
                    self.taxas[self.sens_l] = hz_l
                if len(self.sens_r):
                    self.taxas[self.sens_r] = hz_r
                self.hz = max(hz_l, hz_r)
                self.fluxo_optico = float(np.mean(fluxo))
            else:
                self.hz, escuro = self.transducao.taxa(frame.retina)
                if len(self.sens):
                    self.taxas[self.sens] = self.hz
                self.fluxo_optico = 0.0

        with self.prof("neural", "janela de 10 ms"):
            self.eng.roda_poisson(JANELA_MS, self.taxas, self.rng,
                                  indices=self.sens)

        with self.prof("leitura", "janela"):
            gate = mede_gate(self.eng, self.idx_gf, self.gf_pre, self.gf_peso,
                             self.nomes_tipo, self.gf_tipo_idx,
                             len(self.gf_tipos))
            est_m = self.eng.le(self.motor) if len(self.motor) else None
            ttmn = int(est_m.spike.sum()) if est_m is not None else 0
            # Quantos sensores estao disparando AGORA. E a entrada do circuito,
            # e sem ela nao da pra dizer se uma diferenca entre escopos veio da
            # rede ou de terem recebido estimulos diferentes.
            est_s = self.eng.le(self.sens) if len(self.sens) else None
            self.spikes_sensoriais = (int(est_s.spike.sum())
                                      if est_s is not None else 0)

        eventos = self._eventos(gate, ttmn)
        if self.via == "optomotor":
            # Giro pelo desequilibrio entre os DNa02. ASSUMPTION: o conectoma
            # nao diz quanto um spike descendente vale em velocidade de perna.
            est_l = self.eng.le(self.motor_l) if len(self.motor_l) else None
            est_r = self.eng.le(self.motor_r) if len(self.motor_r) else None
            nl = int(est_l.spike.sum()) if est_l is not None else 0
            nr = int(est_r.spike.sum()) if est_r is not None else 0
            self.desequilibrio = nr - nl
            giro = float(np.clip(self.desequilibrio * GANHO_GIRO,
                                 -GIRO_MAX, GIRO_MAX))
            self.drive = np.array([BASE_DRIVE - giro, BASE_DRIVE + giro])
        else:
            self.drive = (np.array([ESCAPE_DRIVE, ESCAPE_DRIVE])
                          if self.t_s < self.escape_ate
                          else np.array([BASE_DRIVE, BASE_DRIVE]))

        self.janelas += 1
        self._acumula(gate, ttmn)
        if self.registro is not None:
            self._registra(gate, ttmn, eventos)
        with self.prof("telemetry", "quadro"):
            if self.tel.ativo:
                self._publica_quadro(gate, ttmn, eventos, escuro)
        return True

    def _registra(self, gate, ttmn, eventos) -> None:
        """Uma linha por janela neural, mais os eventos raros."""
        pos = np.asarray(self.frame.posicao, dtype=float)
        self.registro.linha(
            t_s=round(self.t_s, 6), passo=self.passo_atual,
            entrada_hz=round(self.hz, 4),
            spikes_sensoriais=self.spikes_sensoriais,
            gf_exc_mV=gate["exc_mV"], gf_inib_mV=gate["inib_mV"],
            gf_liquido_mV=gate["liquido_mV"], gf_v_min_mV=gate["v_min_mV"],
            gf_spikes=gate["spikes_gf"], ttmn_spikes=ttmn,
            fugas_ate_agora=self.escapes,
            drive_esq=round(float(self.drive[0]), 4),
            drive_dir=round(float(self.drive[1]), 4),
            pos_x_mm=round(float(pos[0]), 4),
            pos_y_mm=round(float(pos[1]), 4),
            pos_z_mm=round(float(pos[2]), 4))
        for tipo, detalhe in eventos:
            self.registro.evento(self.t_s, tipo, detalhe)

    def _eventos(self, gate, ttmn):
        """Eventos CIENTIFICOS. Nao ha evento por passo de fisica aqui."""
        eventos = []
        # Borda de SUBIDA, nao estado. Emitir a cada janela em que ha looming
        # enche a lista com 100 linhas por segundo e empurra pra fora da tela
        # justamente os eventos raros -- spike do GF, fuga, limitacao do modelo
        # -- que sao o motivo de a lista existir.
        forte = self.hz > 1.0
        if forte and not self._looming_ativo:
            eventos.append(("looming_start", {"hz": round(self.hz, 2)}))
        elif not forte and self._looming_ativo:
            eventos.append(("looming_end", {"hz": round(self.hz, 2)}))
        self._looming_ativo = forte
        if gate["spikes_gf"]:
            eventos.append(("gf_spike", {"net_mV": gate["liquido_mV"]}))
        if ttmn:
            eventos.append(("ttmn_spike", {"n": ttmn}))
        if ttmn and self.t_s > self.escape_ate:
            self.escape_ate = self.t_s + ESCAPE_MS / 1000.0
            self.escapes += 1
            eventos.append(("escape", {"t_s": round(self.t_s, 3)}))
        if not self.avisou_v and gate["v_min_mV"] < V_AVISO_MV:
            self.avisou_v = True
            eventos.append(("model_limitation", {
                "v_mV": gate["v_min_mV"], "limiar_aviso": V_AVISO_MV,
                "id": "no_inhibitory_reversal"}))
        return eventos

    def roda(self):
        """Laco principal. Controle e simulacao no mesmo fio, de proposito."""
        n_passos = int(self.args.duracao / DT) if self.args.duracao > 0 else -1
        ultimo_log = time.time()
        while not self.sair:
            # o controle e drenado SEMPRE, inclusive pausado -- senao nao ha
            # como despausar
            while True:
                msg = self.ctl.proximo()
                if msg is None:
                    break
                self.trata(msg)
                if self.sair:
                    return

            if self.estado != "running":
                # sem interface nao ha quem mande continuar: acabou, acabou
                if not self.ctl.ativo and self.estado in ("finished", "error"):
                    return
                time.sleep(0.02)
                continue

            self.passo()

            if n_passos > 0 and self.passo_atual >= n_passos:
                self._encerra("duracao alcancada")
                continue

            agora = time.time()
            if agora - ultimo_log > 2.0:
                ultimo_log = agora
                self._log()

    def _log(self):
        v = self.prof.valores()
        print(f"  t={self.t_s:5.2f}s  RTF {v['_total']['rtf']:.4f}  "
              f"looming {self.hz:5.1f}Hz  fugas {self.escapes}")

    def _encerra(self, motivo):
        if self.estado in ("finished", "error"):
            return
        self.estado = "finished"
        self._publica_estado(message=motivo)
        print(f"\n== profiler ==\n{self.prof.relatorio()}")
        print(f"\n  fugas: {self.escapes}")

    def _falha(self, texto):
        self.estado = "error"
        print(f"[erro] {texto}", file=sys.stderr)
        self._publica_estado(message=texto)

    # ----------------------------------------------------------- publicacao

    def _publica_estado(self, message=None):
        self.tel.enviar(self.protocol.run_state(
            self.estado, experiment_id=self.exp_id, seed=self.seed,
            sim_time=self.t_s, step=self.passo_atual,
            detail={"message": message} if message else {}))

    def _publica_catalogo(self):
        self.tel.enviar(self.protocol.experiment_list(CATALOGO,
                                                      current=self.exp_id))

    def _abertura(self):
        protocol, eng, args = self.protocol, self.eng, self.args
        r, rc = eng.resumo(), self.corpo.resumo()
        item = self.item(self.exp_id)
        circuitos = [{"name": n, "role": n,
                      "body_ids": [int(b) for b in eng.c.body_ids[idx]],
                      "sides": [None] * len(idx), "types": [n] * len(idx),
                      "neurotransmitters": [None] * len(idx)}
                     for n, idx in self.papeis.items() if len(idx)]
        dev = dispositivo()
        self.tel.enviar(protocol.experiment_info(
            experiment_id=self.exp_id,
            name=item["name"],
            description=item["description"],
            parameters={"dt_physics_s": DT, "dt_neural_ms": eng.dt,
                        "vision_hz": VISION_HZ, "loom_gain": LOOM_GAIN,
                        "collision_set": args.colisao,
                        "collision_pairs": rc["collision_pairs"],
                        "seed": self.seed, "entrada": "poisson_por_taxa",
                        "v_aviso_mV": V_AVISO_MV},
            provenance={
                "body_ids": protocol.DATA, "synapse_weight": protocol.DATA,
                "neurotransmitter": protocol.DATA,
                "membrane_potential": protocol.MODEL, "spikes": protocol.MODEL,
                "synapse_sign": protocol.MODEL,
                "gf_excitation": protocol.MODEL, "gf_inhibition": protocol.MODEL,
                "loom_gain": protocol.ASSUMPTION,
                "dark_fraction_detector": protocol.ASSUMPTION,
                "motor_to_gait_mapping": protocol.ASSUMPTION,
                "collision_set": protocol.ASSUMPTION,
            },
            circuits=circuitos,
            runtime={
                "os": dev["os"], "cpu": dev["cpu"], "python": dev["python"],
                "physics_backend": rc["physics_backend"],
                "physics_adapter": rc["adapter"],
                "neural_backend": r["backend"], "neural_device": r["device"],
                "neurons_simulated": r["neurons_simulated"],
                "edges_simulated": r["edges_simulated"],
                "vram_mib": r.get("vram_mib", 0),
                "collision_set": args.colisao,
                "collision_pairs": rc["collision_pairs"],
                "population_names": self.nomes_grupos,
            },
            scope={
                "simulated": ("whole_connectome" if item["cns"] == "whole"
                              else "circuit"),
                "functional_brain": "no",
                # Simulado e visualizado sao numeros DIFERENTES e a interface
                # nao pode somar os dois. Quantas morfologias existem e
                # propriedade do asset da Unity, nao do runtime -- por isso o
                # campo vai nulo aqui em vez de chutado.
                "neurons_simulated": r["neurons_simulated"],
                "morphologies_visualized": None,
                "note": ("conectoma inteiro simulado; NAO e um cerebro funcional "
                         "completo -- so a via de looming e a motora tem semantica "
                         "sensorial/motora modelada"),
            },
            model_limitations=[{
                "id": "no_inhibitory_reversal",
                "text": ("The Shiu et al. LIF formulation used here has no "
                         "inhibitory reversal potential. In whole-CNS simulations, "
                         "strong convergent inhibition can therefore drive membrane "
                         "potential to non-physiological negative values."),
                "warning_threshold_mV": V_AVISO_MV,
                "action": "reported, never clamped",
            }],
            gf_gate={
                "edges": int(len(self.gf_peso)),
                "excitatory": int((self.gf_peso > 0).sum()),
                "inhibitory": int((self.gf_peso < 0).sum()),
                "peso_exc_mV": round(float(self.gf_peso[self.gf_peso > 0].sum()), 2),
                "peso_inib_mV": round(float(self.gf_peso[self.gf_peso < 0].sum()), 2),
            },
        ))
        self.tel.enviar(protocol.scene_info(
            arena={"kind": item["arena"]},
            stimulus=({"kind": "approaching_sphere", "radius": 3.0,
                       "start_distance": 30.0, "end_distance": 4.0,
                       "cycle_s": 0.8}
                      if item["arena"] == "looming" else {"kind": "none"})))
        self._publica_catalogo()

    def _publica_quadro(self, gate, ttmn, eventos, escuro):
        protocol, eng = self.protocol, self.eng
        v = self.prof.valores()
        extra = {}
        if self.t_s >= self.proxima_pose:
            nomes, pos, quat = self.corpo.pose_corpo()
            # Pose a 30 Hz, nao a 10.000. A Unity pode interpolar entre
            # quadros -- nada do que ela fizer volta pra fisica.
            extra["body_pose"] = {
                "segments": nomes,
                "pos": np.asarray(pos, dtype=np.float32).round(4).tolist(),
                "quat": np.asarray(quat, dtype=np.float32).round(5).tolist(),
            }
            self.proxima_pose = self.t_s + 1.0 / POSE_HZ

        # onde o estimulo esta AGORA, medido de quem o move. A Unity nao
        # recalcula a trajetoria: se recalculasse, estaria simulando.
        est = self.corpo.estado_estimulo()
        if est is not None:
            extra["stimulus"] = {"pos": [round(v, 4) for v in est[:3]],
                                 "radius": round(est[3], 3)}

        self.tel.enviar(protocol.frame(
            step=self.passo_atual, sim_time=self.t_s, wall_time=time.time(),
            real_time_factor=v["_total"]["rtf"], position=self.frame.posicao,
            drive=self.drive,
            profile={k: v[k]["ms_por_seg_simulado"]
                     for k in ("physics", "vision", "neural", "leitura",
                               "telemetry")},
            **extra))

        camadas = []
        for nome, idx in self.papeis.items():
            if not len(idx):
                continue
            est = eng.le(idx)
            camadas.append({"name": nome, "spikes": est.spike.tolist(),
                            "v_mV": est.v_mV.tolist(), "g_mV": est.g_mV.tolist()})
        self.tel.enviar(protocol.neural_activity(self.t_s, camadas))

        if self.janelas % RETINA_A_CADA == 0 and self.frame.retina is not None:
            ret = self.frame.retina
            self.tel.enviar(protocol.retina(
                self.t_s, left=ret[0], right=ret[1],
                derived={"dark_fraction": {"L": round(float(escuro[0]), 4),
                                           "R": round(float(escuro[1]), 4)},
                         "input_hz": {"L": round(self.hz, 2),
                                      "R": round(self.hz, 2)}}))

        soma = eng.atividade_por_grupo(zerar=True)
        self.tel.enviar(protocol.statistics(self.t_s, {
            "population_activity": {n: int(s)
                                    for n, s in zip(self.nomes_grupos, soma)},
            "looming_hz": self.hz, "ttmn_spikes": ttmn, "gf_gate": gate,
            "escapes": self.escapes,
        }))
        for tipo, detalhe in eventos:
            self.tel.enviar(protocol.event(self.t_s, tipo, detalhe))


def _resolve_experimento(args) -> str:
    """
    `--experiment looming` e atalho pro item do catalogo que casa com a arena
    e o escopo de `--cns`. E o que faz a linha de comando documentada funcionar
    sem obrigar ninguem a decorar id.
    """
    if args.experiment and Laboratorio.item(args.experiment):
        return args.experiment
    arena = args.experiment or "looming"
    casa = [e for e in CATALOGO if e["arena"] == arena and e["cns"] == args.cns]
    if casa:
        return casa[0]["id"]
    casa = [e for e in CATALOGO if e["arena"] == arena]
    return casa[0]["id"] if casa else CATALOGO[0]["id"]


def main():
    ap = argparse.ArgumentParser(description="Drosobot Lab")
    ap.add_argument("--physics", choices=["flygym1", "flygym2"], default="flygym2",
                    help="flygym2 e o caminho principal; flygym1 e a referencia")
    ap.add_argument("--neural", default="auto", help="auto | opencl | cpu | d3d12")
    ap.add_argument("--cns", choices=["whole", "circuit"], default="whole")
    ap.add_argument("--experiment", default=None,
                    help="id do catalogo, ou looming/flat (atalho)")
    ap.add_argument("--colisao", choices=["legs", "tarsi", "none"], default="legs",
                    help="legs = padrao cientifico validado; tarsi = otimizacao "
                         "por cenario (REPROVOU em curva, ver COLLISION_PAIR_AUDIT)")
    ap.add_argument("--duracao", type=float, default=2.0,
                    help="segundos de mosca; 0 = ate a interface mandar parar")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--telemetry", action="store_true")
    ap.add_argument("--porta", type=int, default=8765)
    ap.add_argument("--porta-controle", type=int, default=8766)
    ap.add_argument("--sem-controle", action="store_true",
                    help="nao abre o canal de controle da Unity")
    ap.add_argument("--espera", action="store_true",
                    help="nao comeca sozinho; espera a interface mandar")
    ap.add_argument("--info", action="store_true")
    ap.add_argument("--lista", action="store_true", help="imprime o catalogo e sai")
    args = ap.parse_args()

    if args.lista:
        for e in CATALOGO:
            print(f"  {e['id']:18s} {e['name']}")
        return

    print("== Drosobot Lab ==")
    dev = dispositivo()
    print(f"  maquina   {dev['os']}, {dev['cpu']}, Python {dev['python']}")

    if args.info:
        eng, papeis, _ = monta_cerebro(args.cns, args.neural)
        r = eng.resumo()
        print(f"  cerebro   {r['neurons_simulated']:,} neuronios, "
              f"{r['edges_simulated']:,} arestas")
        print(f"            {r['backend']} em {r['device']}   "
              f"VRAM {r.get('vram_mib', 0):.1f} MiB")
        for nome, idx in papeis.items():
            print(f"            {nome:10s} {len(idx):5d}")
        print(f"  escopo    {'WHOLE CONNECTOME SIMULATED' if args.cns == 'whole' else 'circuito do Giant Fiber'}")
        print("            NAO e um cerebro funcional completo: so a via de")
        print("            looming e a motora tem semantica modelada.")
        return

    from telemetry import protocol
    from telemetry.control import abrir as abrir_controle
    from telemetry.server import abrir as abrir_telemetria

    exp_id = _resolve_experimento(args)
    tel = abrir_telemetria(porta=args.porta, ativo=args.telemetry)
    ctl = abrir_controle(porta=args.porta_controle, ativo=not args.sem_controle)
    lab = Laboratorio(args, tel, ctl, protocol)
    lab._publica_catalogo()

    try:
        if args.espera:
            lab.exp_id = exp_id
            lab._publica_estado(message="esperando a interface")
            print(f"  esperando a interface escolher "
                  f"({len(CATALOGO)} experimentos no catalogo)")
        elif not lab.monta(exp_id, args.seed):
            return
        lab.roda()
    except KeyboardInterrupt:
        print("\ninterrompido")
        lab._encerra("interrompido pelo teclado")
    finally:
        if lab.corpo is not None:
            lab.corpo.fecha()
        if tel.ativo:
            tel.enviar(protocol.bye("corrida terminada"))
            tel.fechar()
        ctl.fechar()


if __name__ == "__main__":
    main()
