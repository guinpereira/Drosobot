"""
Circuito 2, optomotor, agora com os dois hemisferios separados:
T4/T5 -> HS -> DNa02 -> Sternal anterior rotator MN, em cada lado.

O que mudou em relacao a v1:

- a rede nao e mais um pool unico. Os dados mostram que as tres etapas sao
  ESTRITAMENTE ipsilaterais (HS->DNa02: L->L 66, R->R 72, zero cruzando;
  DNa02->motor: L->L 334, R->R 442). Nada aqui forca essa separacao no codigo --
  a rede e montada par a par a partir do conectoma e os dois canais aparecem
  sozinhos. A separacao e um achado do dado, nao uma suposicao nossa.
- por isso a DIRECAO do giro agora sai do circuito: basta olhar de que lado o
  motoneuronio de perna disparou. Antes o desenho usava a tecla que o usuario
  estava segurando, porque a amostra de T4/T5 era 17 do lado R contra 3 do L e
  nao dava pra comparar os lados.
- sinal da sinapse pelo neurotransmissor e biofisica de Shiu et al. 2024,
  igual ao circuito do Giant Fiber (ver sim/connectome_model.py).

O estimulo modela guinada (yaw): girar para um lado acelera o fluxo optico num
olho e desacelera no outro. Qual olho recebe mais e SUPOSICAO nossa sobre o
estimulo; o que vem do dado e o que a rede faz com essa diferenca.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from brian2 import (
    Network, NeuronGroup, Synapses, SpikeMonitor, ms, mV, Hz, defaultclock,
)

sys.path.insert(0, str(Path(__file__).parent))
from connectome_model import (  # noqa: E402
    CONNECTOME, LIF_EQS, LIF_KWARGS, T_DELAY, V_REST,
    load_properties, signed_weights, side_of, describe_drive,
)

HERE = Path(__file__).parent
props = load_properties()

sensor_hs = pd.read_csv(CONNECTOME / "opto_sensor_hs.csv")
hs_dna02 = pd.read_csv(CONNECTOME / "opto_hs_dna02.csv")
dna02_motor = pd.read_csv(CONNECTOME / "opto_dna02_motor.csv")

sensor_ids = sorted(sensor_hs["bodyId_pre"].unique().tolist())
hs_ids = sorted(set(sensor_hs["bodyId_post"]) | set(hs_dna02["bodyId_pre"]))
dna02_ids = sorted(set(hs_dna02["bodyId_post"]) | set(dna02_motor["bodyId_pre"]))
motor_ids = sorted(dna02_motor["bodyId_post"].unique().tolist())

sides = {
    "sensor": [side_of(props, b) for b in sensor_ids],
    "hs": [side_of(props, b) for b in hs_ids],
    "dna02": [side_of(props, b) for b in dna02_ids],
    "motor": [side_of(props, b) for b in motor_ids],
}
for nome, ids in [("T4/T5", sensor_ids), ("HS", hs_ids), ("DNa02", dna02_ids), ("Motor perna", motor_ids)]:
    s = [side_of(props, b) for b in ids]
    print(f"{nome:<12} {len(ids):3d} neuronios   L={s.count('L')}  R={s.count('R')}")

# taxa do olho que recebe o fluxo optico no sentido preferido, e do outro.
# 200 Hz porque abaixo disso o canal ESQUERDO nao chega no motoneuronio: o lado
# direito e mais forte no dado (T4/T5 721 contra 637, DNa02->motor 442 contra
# 334), provavelmente por reconstrucao mais completa desse hemisferio, e o
# limiar transforma ~10% de diferenca de peso em varias vezes de resposta.
YAW_FORTE_HZ = 200
YAW_FRACO_HZ = 15
DURATION = 300 * ms
N_TRIALS = 8


def build():
    """Monta a rede par a par a partir do conectoma. Nada de hemisferio hardcoded."""
    six = {b: i for i, b in enumerate(sensor_ids)}
    hix = {b: i for i, b in enumerate(hs_ids)}
    dix = {b: i for i, b in enumerate(dna02_ids)}
    mix = {b: i for i, b in enumerate(motor_ids)}

    Sensor = NeuronGroup(len(sensor_ids), "rate : Hz", threshold="rand()<rate*dt", method="euler")
    HS = NeuronGroup(len(hs_ids), LIF_EQS, **LIF_KWARGS); HS.v = V_REST
    DNa02 = NeuronGroup(len(dna02_ids), LIF_EQS, **LIF_KWARGS); DNa02.v = V_REST
    Motor = NeuronGroup(len(motor_ids), LIF_EQS, **LIF_KWARGS); Motor.v = V_REST

    syn = []
    for src, tgt, conn, si, ti in [
        (Sensor, HS, sensor_hs, six, hix),
        (HS, DNa02, hs_dna02, hix, dix),
        (DNa02, Motor, dna02_motor, dix, mix),
    ]:
        agg = signed_weights(conn, props)
        agg = agg[agg["bodyId_pre"].isin(si) & agg["bodyId_post"].isin(ti)]
        S = Synapses(src, tgt, "w : volt", on_pre="g_post += w", delay=T_DELAY)
        S.connect(i=[si[b] for b in agg["bodyId_pre"]], j=[ti[b] for b in agg["bodyId_post"]])
        S.w = agg["w_mV"].values * mV
        syn.append(S)
    return Sensor, HS, DNa02, Motor, syn


def trial(olho_forte, seed):
    """olho_forte: 'L' ou 'R'. Devolve spikes do motor de cada lado."""
    np.random.seed(seed)
    defaultclock.dt = 0.1 * ms
    Sensor, HS, DNa02, Motor, syn = build()
    Sensor.rate = [YAW_FORTE_HZ if s == olho_forte else YAW_FRACO_HZ
                   for s in sides["sensor"]] * Hz
    mon_m = SpikeMonitor(Motor)
    mon_d = SpikeMonitor(DNa02)
    # Network explicito em vez de run() magico: o magico varre as variaveis
    # locais de quem chama e NAO olha dentro de listas, entao as Synapses
    # guardadas em `syn` ficavam de fora e a rede rodava muda.
    net = Network(Sensor, HS, DNa02, Motor, *syn, mon_m, mon_d)
    net.run(DURATION)
    out = {}
    for lado in ("L", "R"):
        idx = [i for i, s in enumerate(sides["motor"]) if s == lado]
        out["motor_" + lado] = int(np.isin(mon_m.i, idx).sum())
        idx_d = [i for i, s in enumerate(sides["dna02"]) if s == lado]
        out["dna02_" + lado] = int(np.isin(mon_d.i, idx_d).sum())
    return out


print("\nBalanco do drive:")
for nome, conn in [("T4/T5 -> HS   ", sensor_hs), ("HS    -> DNa02 ", hs_dna02),
                   ("DNa02 -> Motor ", dna02_motor)]:
    describe_drive(signed_weights(conn, props), props, "  " + nome)

print(f"\nEstimulo de guinada: olho estimulado a {YAW_FORTE_HZ} Hz, o outro a {YAW_FRACO_HZ} Hz")
print(f"media de {N_TRIALS} corridas de {int(DURATION/ms)} ms\n")

res = {}
for olho in ("L", "R"):
    runs = [trial(olho, seed=s) for s in range(N_TRIALS)]
    res[olho] = {k: np.mean([r[k] for r in runs]) for k in runs[0]}
    r = res[olho]
    vencedor = "L" if r["motor_L"] > r["motor_R"] else ("R" if r["motor_R"] > r["motor_L"] else "empate")
    print(f"olho {olho} estimulado -> DNa02 L={r['dna02_L']:5.1f} R={r['dna02_R']:5.1f} | "
          f"motor perna L={r['motor_L']:5.1f} R={r['motor_R']:5.1f}   => giro pelo lado {vencedor}")

# ---------- figura ----------
fig, ax = plt.subplots(figsize=(8, 5))
x = np.arange(2)
largura = 0.35
esq = [res["L"]["motor_L"], res["R"]["motor_L"]]
dir_ = [res["L"]["motor_R"], res["R"]["motor_R"]]
ax.bar(x - largura / 2, esq, largura, label="motoneuronio de perna ESQUERDO", color="#1f77b4")
ax.bar(x + largura / 2, dir_, largura, label="motoneuronio de perna DIREITO", color="#ff7f0e")
ax.set_xticks(x)
ax.set_xticklabels([f"fluxo optico no olho\nESQUERDO ({YAW_FORTE_HZ} Hz)",
                    f"fluxo optico no olho\nDIREITO ({YAW_FORTE_HZ} Hz)"])
ax.set_ylabel(f"spikes do motoneuronio em {int(DURATION/ms)} ms")
ax.set_title("Optomotor bilateral: a direcao do giro sai do circuito\n"
             "as tres etapas sao ipsilaterais no conectoma, nada disso e hardcoded", fontsize=11)
ax.legend()
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
out = HERE.parent / "docs" / "images" / "optomotor_raster.png"
plt.savefig(out, dpi=120)
print(f"\nSalvo: {out}")
