"""
RESULTADO NEGATIVO: o circuito nao consegue guiar desvio de obstaculo.

A pergunta era se o reflexo serve pra alguma coisa: a mosca anda livre entre
postes, o fluxo optico vem do movimento dela, e o circuito deveria desviar.

    poste passando ao lado (porque a mosca anda)
      -> retina -> fluxo optico por olho
      -> T4/T5 -> HS -> DNa02 -> motoneuronio de perna   [conectoma]
      -> vira

Testamos as duas convencoes possiveis (virar PRA o obstaculo e PRA LONGE dele)
contra um controle cego, porque o conectoma da o lado mas nao diz para que lado
virar. Nenhuma funcionou, e a causa nao e ajuste de parametro.

## Por que nao funciona

Medindo o circuito isolado, com as taxas que um poste lateral realmente produz
(medidas na propria simulacao: ~205 Hz no olho de perto, ~122 no outro):

    entrada          motor L      motor R      L - R
    122/122 (sem)      0.0         16.0        -16.0
    205/122 (ESQ)      5.8         16.0        -10.2
    122/205 (DIR)      0.0         38.6        -38.6

O sinal de (L - R) e SEMPRE negativo, esteja o obstaculo de que lado for. O canal
esquerdo fica mudo com entrada simetrica e so acorda perto de 205 Hz. Um
controlador que leia "de que lado" pela comparacao bilateral nao tem como
funcionar -- nessa faixa o circuito reporta QUANTO, nao DE QUE LADO.

A causa e a assimetria de reconstrucao do Male CNS, ja documentada no README: o
hemisferio direito e mais forte em toda etapa (T4/T5 721 contra 637, DNa02->motor
442 contra 334). Com limiar nao-linear, o lado esquerdo fica abaixo do limiar e
nao contribui. Com contraste extremo entre os olhos (200 contra 15 Hz) a
lateralizacao e limpa -- ver sim/optomotor_network.py. Um poste passando nao gera
contraste nem perto disso.

## O que foi tentado antes de chegar aqui

1. Circuito do Giant Fiber, postes NA FRENTE. Ameaca frontal da n_L ~ n_R, o
   sinal de lado zera e a mosca anda reto no obstaculo. O GF informa QUE LADO, e
   lado so existe quando a ameaca esta de lado.
2. Giant Fiber, postes AO LADO. A expansao e bem lateralizada (pico L=0.106
   contra R=0.044), mas dura poucos quadros e o GF solta 1 ou 2 spikes por poste.
   Detecta, nao guia.
3. Optomotor com freio de emergencia herdado do GF. O optomotor e continuo,
   cruzava o limiar do freio o tempo todo, e a mosca passava a corrida inteira
   andando de re.
4. Optomotor com ganho alto. Os dois olhos saturavam em 250 Hz e a saturacao
   matava a assimetria.
5. Optomotor com o desequilibrio bruto no comando. O vies permanente do dataset
   fazia a mosca girar sem parar; a folga maior era artefato de nao ter avancado.
6. Optomotor com o vies descontado (media lenta). A mosca anda normal, mas a
   folga fica identica ao controle: 6.91 contra 6.88 e 6.88 mm.

O passo 6 e o que este script roda. O resultado e o item acima: nao ha janela de
ajuste entre "gira sem parar" e "nao gira", porque a informacao de lado nao esta
na saida do circuito nessa faixa de entrada.

Roda:  .venv\Scripts\python sim\flygym_avoidance.py
Sai:   docs/images/flygym_avoidance.png
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from flygym import Fly
from flygym.arena import FlatTerrain
from flygym.examples.locomotion import HybridTurningController

sys.path.insert(0, str(Path(__file__).parent))
from connectome_model import (  # noqa: E402
    CONNECTOME, load_properties, signed_weights, side_of,
)
import fast_lif  # noqa: E402

HERE = Path(__file__).parent
OUT = HERE.parent / "docs" / "images"
props = load_properties()

# ---------------- circuito optomotor, bilateral ----------------
sensor_hs = pd.read_csv(CONNECTOME / "opto_sensor_hs.csv")
hs_dna02 = pd.read_csv(CONNECTOME / "opto_hs_dna02.csv")
dna02_motor = pd.read_csv(CONNECTOME / "opto_dna02_motor.csv")

sensor_ids = sorted(sensor_hs["bodyId_pre"].unique().tolist())
hs_ids = sorted(set(sensor_hs["bodyId_post"]) | set(hs_dna02["bodyId_pre"]))
dna_ids = sorted(set(hs_dna02["bodyId_post"]) | set(dna02_motor["bodyId_pre"]))
motor_ids = sorted(dna02_motor["bodyId_post"].unique().tolist())

lado_sensor = np.array([side_of(props, b) for b in sensor_ids])
lado_motor = np.array([side_of(props, b) for b in motor_ids])
MOTOR_L = np.where(lado_motor == "L")[0]
MOTOR_R = np.where(lado_motor == "R")[0]

ix = lambda ids: {b: i for i, b in enumerate(ids)}
CONEXOES = []
for conn, si, ti in [(sensor_hs, ix(sensor_ids), ix(hs_ids)),
                     (hs_dna02, ix(hs_ids), ix(dna_ids)),
                     (dna02_motor, ix(dna_ids), ix(motor_ids))]:
    agg = signed_weights(conn, props)
    agg = agg[agg["bodyId_pre"].isin(si) & agg["bodyId_post"].isin(ti)]
    CONEXOES.append(([si[b] for b in agg["bodyId_pre"]],
                     [ti[b] for b in agg["bodyId_post"]],
                     agg["w_mV"].values))
TAMANHOS = [len(hs_ids), len(dna_ids), len(motor_ids)]
CAMADA_MOTOR = 2

print(f"circuito: T4/T5 {len(sensor_ids)} (L={np.sum(lado_sensor=='L')} "
      f"R={np.sum(lado_sensor=='R')}) -> HS {len(hs_ids)} -> DNa02 {len(dna_ids)} "
      f"-> motor {len(motor_ids)} (L={len(MOTOR_L)} R={len(MOTOR_R)})")

# ---------------- campo de obstaculos ----------------
# Postes ALTERNANDO dos lados da linha de caminhada, nao em cima dela.
#
# A primeira versao punha os postes na frente da mosca, e nenhum dos modos passava
# do primeiro: ameaca frontal produz n_L ~ n_R, o sinal de lado da zero e ela anda
# reto no obstaculo. Nao e falha de ajuste, e o que o circuito e -- o Giant Fiber
# informa QUE LADO, e lado so existe quando a ameaca esta de lado. (Na biologia
# vale o mesmo: a fuga do GF e um pulo pra tras, nao uma curva.)
#
# A +/-3.5 mm ainda era perto demais: o poste tem raio 1.5, a mosca deriva pra
# y ~ 1 andando reto, e ela raspava e TRAVAVA no primeiro (avanco parava em 11 mm
# dos 52 esperados). A +/-5 mm a cega passa limpa, e sobra espaco pra medir se
# virar aumenta ou diminui a folga.
POSTES = np.array([(12.0 + 10.0 * k, 7.0 if k % 2 == 0 else -7.0) for k in range(6)])
RAIO_POSTE = 3.0
# "tocou o poste": centro da mosca dentro do raio dele mais um corpo de mosca.
# Contar TRAVESSIAS desse limiar nao funciona -- o corpo balanca ao andar, a
# distancia oscila em volta do valor e uma unica batida virava dezenas de
# "encostoes". Contamos postes DISTINTOS tocados na corrida inteira.
RAIO_COLISAO = RAIO_POSTE + 1.0

DT_MS = 0.5
VISION_HZ = 100
# fluxo optico por olho -> taxa de T4/T5.
# Ganho calibrado NESTA geometria, nao copiado de flygym_optomotor.py: medimos o
# fluxo com a mosca passando por um poste lateral e deu ~0.012 de base nos dois
# olhos e ~0.0186 contra 0.0111 na passagem. Com o ganho de 30000 do outro script
# os dois olhos saturavam em 250 Hz, e saturacao MATA a assimetria, que e
# exatamente o sinal. 11000 poe o olho estimulado perto de 200 Hz e o outro em
# ~120, dentro da faixa em que o circuito responde.
FLOW_GAIN = 11000.0
FLOW_MAX_HZ = 250.0
BASE_DRIVE = 1.0
# Sem freio de emergencia. A versao anterior recuava quando a saida motora passava
# de um limiar por janela -- regra pensada pro Giant Fiber, que dispara esparso.
# O optomotor e continuo: soltava 1225 spikes, cruzava o limiar o tempo todo e a
# mosca passava a corrida inteira andando de re. Os dois modos davam identico
# porque nenhum estava virando.
GIRO = 0.6          # quanto o lado escolhido encurta o passo
GIRO_TAU = 4.0      # quanto o comando de giro demora a acumular
# O desequilibrio entre os lados tem um VIES PERMANENTE, nao so a resposta ao
# obstaculo: medimos fluxo de base L=0.0133 contra R=0.0112 longe de qualquer
# poste, e o hemisferio direito do dataset e mais forte em toda etapa (o mesmo
# vies ja documentado no circuito optomotor). Com o viés entrando direto no
# comando, a mosca girava sem parar e nem chegava nos postes -- a folga maior era
# artefato de nao ter avancado. Descontamos a media lenta do proprio sinal, entao
# so DESVIO em relacao ao normal dela vira curva. Mesmo principio do detector de
# looming, e igualmente suposicao nossa.
VIES_TAU = 120.0    # atualizacoes de retina (~1.2 s a 100 Hz)
JANELA_MS = 10.0
DURACAO_S = 6.0   # a ~13 mm/s cobre os 6 postes

CONTATOS = [f"{perna}{seg}"
            for perna in ["LF", "LM", "LH", "RF", "RM", "RH"]
            for seg in ["Tibia", "Tarsus1", "Tarsus2", "Tarsus3", "Tarsus4", "Tarsus5"]]


def _monta_arena():
    import importlib.util
    import os
    import flygym.examples as fex
    spec = importlib.util.spec_from_file_location(
        "flygym_vision_arena", os.path.join(fex.__path__[0], "vision", "arena.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.ObstacleOdorArena(
        terrain=FlatTerrain(),
        obstacle_positions=POSTES,
        obstacle_colors=(0, 0, 0, 1),
        obstacle_radius=RAIO_POSTE,
        obstacle_height=4.0,
        odor_source=np.array([[1000.0, 0.0, 2.0]]),   # longe: nao usamos odor
        marker_colors=[],
    )


def roda(modo, semente=0):
    np.random.seed(semente)
    arena = _monta_arena()
    fly = Fly(enable_vision=True, vision_refresh_rate=VISION_HZ, spawn_pos=(0, 0, 0.3),
              contact_sensor_placements=CONTATOS)
    sim = HybridTurningController(fly=fly, arena=arena, timestep=1e-4)
    obs, info = sim.reset(seed=semente)

    rede = fast_lif.Rede(len(sensor_ids), TAMANHOS, CONEXOES, DT_MS)
    rng = np.random.default_rng(semente)

    retina_ant = np.asarray(obs["vision"]).mean(axis=2)
    drive = np.array([BASE_DRIVE, BASE_DRIVE])
    bal = 0.0
    vies = 0.0
    traj, ttmn_total = [], 0
    tocados = np.zeros(len(POSTES), dtype=bool)
    folga_min = np.full(len(POSTES), np.inf)   # menor distancia a cada poste

    n_passos = int(DURACAO_S / 1e-4)
    for passo in range(n_passos):
        t_s = passo * 1e-4
        obs, _, _, _, info = sim.step(drive)

        pos = np.asarray(obs["fly"][0])[:2]
        d = np.linalg.norm(POSTES - pos, axis=1)
        tocados |= d < RAIO_COLISAO
        folga_min = np.minimum(folga_min, d)

        if passo % 50 == 0:
            traj.append((pos[0], pos[1]))

        if not info.get("vision_updated", False):
            continue

        retina = np.asarray(obs["vision"]).mean(axis=2)
        flow = np.abs(retina - retina_ant).mean(axis=1)     # um valor por olho
        retina_ant = retina
        hz = np.clip(flow * FLOW_GAIN, 0, FLOW_MAX_HZ)

        if modo == "cego":
            drive = np.array([BASE_DRIVE, BASE_DRIVE])
            continue

        taxa = np.where(lado_sensor == "L", hz[0], hz[1])
        saida = rede.roda(JANELA_MS, taxa, rng=rng)[CAMADA_MOTOR]
        n_L = int(saida[MOTOR_L].sum())
        n_R = int(saida[MOTOR_R].sum())
        ttmn_total += n_L + n_R

        lado = 0.0
        if n_L + n_R > 0:
            lado = (n_R - n_L) / (n_L + n_R)     # +1 obstaculo visto a direita
        vies += (lado - vies) / VIES_TAU          # o "normal" dela, sem obstaculo
        desvio = lado - vies
        if modo == "para_longe":
            desvio = -desvio
        bal += (desvio - bal) / GIRO_TAU

        drive = np.array([BASE_DRIVE, BASE_DRIVE])
        if bal > 0:
            drive[1] -= bal * GIRO
        else:
            drive[0] -= -bal * GIRO
        drive = np.clip(drive, -0.5, 1.5)

    # so conta a folga dos postes que a mosca chegou a passar
    passou = folga_min < 20.0
    return dict(traj=np.array(traj), tocados=int(tocados.sum()), ttmn=ttmn_total,
                avanco=float(np.asarray(obs["fly"][0])[0]),
                folga=float(np.mean(folga_min[passou])) if passou.any() else float("nan"),
                n_passou=int(passou.sum()))


MODOS = [("cego", "sem circuito (controle)", "#888888"),
         ("para_ameaca", "vira PRA o obstaculo", "#d62728"),
         ("para_longe", "vira PRA LONGE do obstaculo", "#1f77b4")]
SEMENTES = [0, 1, 2]


def _tarefa(args):
    modo, semente = args
    return modo, semente, roda(modo, semente)


def main():
    import multiprocessing as mp
    import time

    tarefas = [(m, s) for m, _, _ in MODOS for s in SEMENTES]
    # As 9 corridas sao independentes, entao vao em paralelo. Uma simulacao
    # sozinha nao usa mais de um nucleo -- o passo do MuJoCo e serial pra um
    # modelo so, e o laco passa por Python a cada passo. O paralelismo que existe
    # aqui e entre CORRIDAS, nao dentro de uma.
    n_proc = min(len(tarefas), mp.cpu_count())
    print()
    print(f"{len(POSTES)} postes, {DURACAO_S:.0f} s por corrida, "
          f"{len(tarefas)} corridas em {n_proc} processos")
    print()

    t0 = time.time()
    with mp.Pool(n_proc) as pool:
        brutos = pool.map(_tarefa, tarefas)
    print(f"(tempo de relogio: {time.time() - t0:.0f} s)")
    print()
    resultados = {m: [] for m, _, _ in MODOS}
    for modo, _semente, r in brutos:
        resultados[modo].append(r)

    for modo, rotulo, _ in MODOS:
        corridas = resultados[modo]
        av = [c["avanco"] for c in corridas]
        toc = [c["tocados"] for c in corridas]
        fol = [c["folga"] for c in corridas]
        print(f"{rotulo:<30} avanco {np.mean(av):5.1f} mm   "
              f"folga media {np.nanmean(fol):5.2f} +/- {np.nanstd(fol):4.2f} mm   "
              f"tocados {np.mean(toc):4.1f}   motor {np.mean([c['ttmn'] for c in corridas]):6.0f}")

    fig, axes = plt.subplots(1, 3, figsize=(15, 5.5), sharex=True, sharey=True)
    for ax, (modo, rotulo, cor) in zip(axes, MODOS):
        for p in POSTES:
            ax.add_patch(plt.Circle(p, RAIO_POSTE, color="black", alpha=0.75))
        for k, c in enumerate(resultados[modo]):
            t = c["traj"]
            ax.plot(t[:, 0], t[:, 1], color=cor, lw=1.4, alpha=0.85)
            ax.plot(t[0, 0], t[0, 1], "o", color="k", ms=5)
        av = [c["avanco"] for c in resultados[modo]]
        toc = [c["tocados"] for c in resultados[modo]]
        fol = [c["folga"] for c in resultados[modo]]
        titulo = (f"{rotulo}" + chr(10) +
                  f"folga {np.nanmean(fol):.2f} mm | tocados {np.mean(toc):.1f} | "
                  f"avanco {np.mean(av):.0f} mm")
        ax.set_title(titulo, fontsize=10)
        ax.set_xlabel("x (mm)")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("y (mm)")
    axes[0].set_ylim(-14, 14)
    fig.suptitle(
        "A mosca anda livre entre postes; o fluxo optico vem do movimento dela."
        + chr(10) +
        "O conectoma da o lado do obstaculo -- para que lado virar e o que esta sendo testado.",
        fontsize=11)
    plt.tight_layout()
    png = OUT / "flygym_avoidance.png"
    plt.savefig(png, dpi=120)
    print()
    print(f"{len(POSTES)} postes, {DURACAO_S:.0f} s por corrida, "
          f"{len(tarefas)} corridas em {n_proc} processos")
    print()
if __name__ == "__main__":
    main()
