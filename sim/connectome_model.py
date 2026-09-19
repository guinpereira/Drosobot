"""
Base compartilhada das redes: biofisica publicada + sinal sinaptico vindo do
neurotransmissor real. Antes cada script repetia um LIF unitless (v de 0 a 1,
padrao tutorial do Brian2) com ganho ajustado na mao por camada, e tratava toda
sinapse como excitatoria.

Os parametros aqui sao os de Shiu et al. 2024 (Nature 634:210), "A Drosophila
computational brain model reveals sensorimotor processing" -- um LIF do cerebro
inteiro da mosca, tambem em Brian2, a partir da conectividade do FlyWire mais
predicao de neurotransmissor. Usar os numeros deles troca varios ganhos
arbitrarios por um unico parametro livre (W_SYN), que e exatamente como o paper
trata o problema.

Modelo (alpha-synapse, tres equacoes):

    dv/dt = (-(v - V_rest) + g) / TAU_MBR
    dg/dt = -g / TAU_SYN
    ao spike de j:  g_i += w_ji

    w_ji = (numero de sinapses EM de j para i) * sinal(j) * W_SYN

O sinal vem do neurotransmissor do neuronio PRE-sinaptico: um neuronio e
inteiramente excitatorio ou inteiramente inibitorio (lei de Dale), nao a
sinapse individual.
"""
from pathlib import Path

import pandas as pd
from brian2 import mV, ms, uF, kohm, cmeter, prefs

# gera codigo em numpy em vez de C++. Sem isso o Brian2 tenta compilar e morre
# com 'Unable to find a compatible Visual Studio installation' nesta maquina.
# As redes aqui tem dezenas de neuronios, a diferenca de velocidade nao importa.
prefs.codegen.target = "numpy"

CONNECTOME = Path(__file__).parent.parent / "connectome"

# ---------- parametros biofisicos (Shiu et al. 2024, Methods) ----------
V_REST = -52 * mV          # potencial de repouso
V_RESET = -52 * mV         # volta pro repouso depois do spike
V_THRESHOLD = -45 * mV     # limiar de disparo -- 7 mV acima do repouso
R_MBR = 10 * kohm * cmeter ** 2
C_MBR = 2 * uF / cmeter ** 2
TAU_MBR = C_MBR * R_MBR    # 20 ms -- constante de tempo da membrana (circuito RC)
T_REFRACTORY = 2.2 * ms
TAU_SYN = 5 * ms           # decaimento da conductancia sinaptica
T_DELAY = 1.8 * ms         # atraso entre o spike e o efeito no alvo
W_SYN = 0.275 * mV         # UNICO parametro livre: efeito de UMA sinapse no alvo

# equacoes para NeuronGroup(...) -- mesmas dos tres scripts de rede
LIF_EQS = """
dv/dt = (-(v - V_REST) + g) / TAU_MBR : volt (unless refractory)
dg/dt = -g / TAU_SYN : volt
"""
# as constantes vao no namespace explicito: sem isso o Brian2 tenta resolver
# V_REST/TAU_MBR/etc no escopo de QUEM chama, e um NeuronGroup criado dentro de
# uma funcao quebra com 'The identifier "TAU_MBR" could not be resolved'.
LIF_NAMESPACE = {
    "V_REST": V_REST,
    "V_RESET": V_RESET,
    "V_THRESHOLD": V_THRESHOLD,
    "TAU_MBR": TAU_MBR,
    "TAU_SYN": TAU_SYN,
}
LIF_KWARGS = dict(
    threshold="v > V_THRESHOLD",
    reset="v = V_RESET",
    refractory=T_REFRACTORY,
    method="exact",
    namespace=LIF_NAMESPACE,
)

# ---------- neurotransmissor -> sinal ----------
# Aqui os dois papers de referencia DISCORDAM, e a escolha muda o resultado:
#
#   Shiu et al. 2024 (Nature)  -- GABA e glutamato inibitorios; dopamina,
#                                 octopamina e serotonina no balde excitatorio.
#   Jin et al. (FlyGM, arXiv)  -- so GABA e glicina inibitorios; glutamato,
#                                 aspartato e histamina excitatorios.
#
# Seguimos Shiu: na mosca o glutamato costuma agir em GluCl, canal de cloreto
# ativado por glutamato, que hiperpolariza. Nao e detalhe academico -- 6 dos 8
# neuronios pre-sinapticos do Giant Fiber sao GABA ou glutamato e carregam 59%
# do peso total, entao a escolha inverte o sinal da maior parte do drive.
NT_SIGN = {
    "acetylcholine": +1,
    "dopamine": +1,
    "octopamine": +1,
    "serotonin": +1,
    "gaba": -1,
    "glutamate": -1,
    "histamine": -1,   # HisCl1, canal de cloreto -- inibitorio na mosca
}
# "unclear" cai aqui. Nos nossos circuitos so os motoneuronios terminais
# (TTMn, Sternal anterior rotator MN) ficam sem predicao, e eles nao tem alvo
# dentro do modelo, entao o sinal deles nunca e usado.
NT_SIGN_DEFAULT = +1


def load_properties():
    """neuron_properties.csv indexado por bodyId (gera com fetch_neuron_properties.py)."""
    path = CONNECTOME / "neuron_properties.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} nao existe. Rode: .venv\Scripts\python connectome\fetch_neuron_properties.py"
        )
    return pd.read_csv(path).set_index("bodyId")


def nt_sign(props, body_id):
    """+1 se o neuronio excita o alvo, -1 se inibe."""
    if body_id not in props.index:
        return NT_SIGN_DEFAULT
    nt = props.at[body_id, "consensusNt"]
    if not isinstance(nt, str):
        return NT_SIGN_DEFAULT
    return NT_SIGN.get(nt.strip().lower(), NT_SIGN_DEFAULT)


def side_of(props, body_id):
    """'L', 'R' ou 'M' (linha media). None se o dataset nao diz."""
    if body_id not in props.index:
        return None
    side = props.at[body_id, "somaSide"]
    return side if isinstance(side, str) else None


def signed_weights(conn, props):
    """
    Agrega uma tabela de conexoes em pares (pre, post) com peso ja com sinal.

    conn: DataFrame com bodyId_pre, bodyId_post, weight (uma linha por ROI).
    Retorna DataFrame com bodyId_pre, bodyId_post, weight, sign, w_mV.
    """
    agg = conn.groupby(["bodyId_pre", "bodyId_post"], as_index=False)["weight"].sum()
    agg["sign"] = [nt_sign(props, b) for b in agg["bodyId_pre"]]
    agg["w_mV"] = agg["weight"] * agg["sign"] * float(W_SYN / mV)
    return agg


def describe_drive(agg, props, label):
    """Imprime quanto do drive e excitatorio vs inibitorio -- some silenciosamente e mentira."""
    exc = agg[agg["sign"] > 0]["weight"].sum()
    inh = agg[agg["sign"] < 0]["weight"].sum()
    total = exc + inh
    if total == 0:
        print(f"{label}: sem conexoes")
        return
    print(f"{label}: {int(exc)} sinapses excitatorias ({exc/total:.0%}), "
          f"{int(inh)} inibitorias ({inh/total:.0%})")


# ---------- populacao de entrada do Giant Fiber ----------
# Os detectores de looming que entram no GF sao LC4 e LPLC2 -- e o dataset que
# diz: sao os unicos tipos com peso relevante entre os upstream de superclass
# "visual_projection" (6362 e 4862 sinapses; o terceiro colocado tem 20). Bate
# com a literatura, que descreve os dois como a via de aproximacao que dispara
# a fuga.
#
# Isso corrigiu um erro: antes o "sensor visual" era os 8 upstream de maior peso,
# que pegava DNp70 e neuronios SAD e NAO incluia nenhum LC4 ou LPLC2. Cada celula
# LC4 tem peso pequeno (6362 espalhados em centenas de celulas) enquanto DNp70
# sao 2 celulas de 799 e 617 -- ordenar por peso por celula escondia a via certa.
LOOMING_TYPES = {"LC4", "LPLC2"}

# Taxa de base dos inibitorios. SUPOSICAO nossa: no cerebro inteiro eles sao
# disparados pelo resto da rede, que nao simulamos. Todo o resto fica em 0 Hz,
# seguindo o protocolo de Shiu et al. (baseline de 0 Hz, so o sensorio e
# estimulado) -- por Poisson de fundo em tudo, o GF dispara ate em repouso.
TONIC_INHIB_HZ = 5.0


def gf_input_population(props, upstream):
    """
    Separa os upstream do GF em looming / inibitorio / silencioso.

    Devolve (ids, mascara_looming, mascara_inibitoria), na mesma ordem.
    """
    ids = sorted(upstream["bodyId_pre"].unique())
    tipos = [props.at[b, "type"] if b in props.index else None for b in ids]
    is_looming = [t in LOOMING_TYPES for t in tipos]
    is_inhib = [nt_sign(props, b) < 0 for b in ids]
    return ids, is_looming, is_inhib
