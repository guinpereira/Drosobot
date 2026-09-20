"""
O pedaço quente do controlador, compilado — com a MESMA aritmética.

Só uma coisa mora aqui: a avaliação do polinômio por partes que o passo
pré-programado usa. É o maior item que sobrou no laço de física depois do
`fastpath.py` — ~158 µs por passo, 6 chamadas ao scipy a ~26 µs de overhead de
Python cada, para ~1 µs de conta.

## Por que dá para substituir o scipy aqui

Porque a substituição é **exata**, e isso foi testado antes de ser escrita.

`PPoly` expõe publicamente `x` (os nós) e `c` (os coeficientes). O que não é
público é a ordem em que o scipy soma os termos — e a ordem decide o último
bit. Foram testadas três formas contra 2.000 sorteios de fase:

    Horner              0/2000 idênticos, pior erro 4,4e-16
    forma potência      0/2000 idênticos, pior erro 4,4e-16
    potência corrente   2000/2000 idênticos, pior erro 0,0

A que bate é a terceira: somar do termo de MENOR grau para o maior, com a
potência acumulada por multiplicação sucessiva (`z *= s`), que é o que o
`evaluate_poly1` do scipy faz. Horner é matematicamente igual e numericamente
diferente — e essa diferença, realimentada 10.000 vezes por segundo simulado
num sistema com contato, diverge.

Não há API privada do scipy envolvida: só `x` e `c`, e aritmética nossa.

## Por que Numba

Medido neste ambiente (Python 3.14.4, Windows): Numba 0.67 compila e o
overhead de chamada é 1,28 µs. Cython e uma extensão C exigiriam toolchain de
build no repositório; Numba compila em processo, guarda em cache no disco, e
some sozinho se não estiver instalado — o `fastpath.py` volta ao scipy.

A degradação é silenciosa **de propósito**: sem Numba o laboratório roda mais
devagar, não quebra.
"""
from __future__ import annotations

import numpy as np

try:
    from numba import njit

    NUMBA = True
except ImportError:                                           # pragma: no cover
    NUMBA = False

    def njit(*a, **k):                                        # type: ignore
        def deco(f):
            return f
        return deco


@njit(cache=True, fastmath=False, nogil=True)
def avalia_ppoly_periodico(x, c, fases, saida):
    """
    `PPoly(fases)` com `extrapolate='periodic'`, para várias fases de uma vez.

    x      (nx,)            nós, crescentes
    c      (k+1, nx-1, m)   coeficientes, grau decrescente no eixo 0
    fases  (p,)             onde avaliar
    saida  (p, m)           escrito no lugar

    `fastmath=False` é obrigatório: ele autorizaria reassociação e é exatamente
    o que quebraria a igualdade bit a bit com o scipy.
    """
    nx = x.shape[0]
    ordem = c.shape[0] - 1
    m = c.shape[2]
    x0 = x[0]
    periodo = x[nx - 1] - x0

    for pi in range(fases.shape[0]):
        # mesmo wrap do scipy: resto em relação ao início do intervalo base
        xw = (fases[pi] - x0) % periodo + x0

        # busca binária equivalente a `searchsorted(x, xw, side='right') - 1`
        lo = 0
        hi = nx
        while lo < hi:
            meio = (lo + hi) // 2
            if x[meio] <= xw:
                lo = meio + 1
            else:
                hi = meio
        k = lo - 1
        if k < 0:
            k = 0
        elif k > nx - 2:
            k = nx - 2

        s = xw - x[k]
        for j in range(m):
            # do MENOR grau para o maior, com potência acumulada: e a ordem do
            # `evaluate_poly1` do scipy. Trocar por Horner muda o ultimo bit.
            res = 0.0
            z = 1.0
            for kp in range(ordem + 1):
                res += c[ordem - kp, k, j] * z
                z *= s
            saida[pi, j] = res
    return saida


def disponivel() -> bool:
    """Numba compilou? Quem chama decide o que fazer sem ele."""
    return NUMBA


def aquece(x, c, p: int) -> None:
    """
    Compila antes da corrida começar.

    Sem isto a primeira janela neural pagaria o compilador, o que apareceria
    como um pico de ~0,4 s no profiler e seria lido como problema de física.
    """
    if not NUMBA:
        return
    fases = np.zeros(p, dtype=np.float64)
    saida = np.zeros((p, c.shape[2]), dtype=np.float64)
    avalia_ppoly_periodico(x, c, fases, saida)


@njit(cache=True, fastmath=False, nogil=True)
def passo_controlador(
    # estado, escrito no lugar
    fases, magnitudes, retracao, tropeco, contador,
    # entrada do passo
    torax_z, tarsus5_z, forcas, rumo, sinal,
    # constantes do CPG
    acoplamento, vies_fase, freqs_base, convergencia, dt,
    # constantes do controlador
    limiar_retracao, limiar_persistencia, passos_persistencia,
    taxas_retracao, taxas_tropeco, limiar_forca, max_correcao,
    # tabelas do passo pre-programado
    x_pp, c_pp, neutro, pontos_fase, incrementos, vetores_corr, indices,
    swing_ini, swing_fim,
    # saida
    angulos, adesao, psi_buf, correcoes,
):
    """
    Um passo do HybridTurningController, inteiro, em codigo compilado.

    Mesma aritmetica e mesma ORDEM de operacoes do caminho em NumPy -- a ordem
    decide o ultimo bit, e o ultimo bit realimentado 10.000 vezes por segundo
    simulado diverge. A prova esta em tests/test_fastpath_equivalencia.py, que
    exige igualdade exata contra o controlador do upstream.

    O ganho nao vem de calcular menos: vem de nao pagar despacho do NumPy em
    arrays de SEIS elementos. Sao ~30 operacoes desse tamanho por passo, cada
    uma com overhead de chamada maior que a propria conta.
    """
    n_pernas = fases.shape[0]
    dois_pi = 2.0 * np.pi

    # --- amplitude e frequencia intrinsecas, do sinal descendente ---
    amps = np.empty(n_pernas)
    freqs = np.empty(n_pernas)
    for i in range(n_pernas):
        lado = 0 if i < 3 else 1
        amps[i] = abs(sinal[lado])
        freqs[i] = freqs_base[i] * (1.0 if sinal[lado] >= 0.0 else -1.0)

    # --- perna a retrair: a mais alta, se destacada das outras ---
    altura = np.empty(n_pernas)
    for i in range(n_pernas):
        altura[i] = torax_z - tarsus5_z[i]
    ordem = np.argsort(altura)
    perna_retracao = -1
    if altura[ordem[n_pernas - 1]] > altura[ordem[n_pernas - 3]] + limiar_retracao:
        perna_retracao = ordem[n_pernas - 1]

    if perna_retracao >= 0 and retracao[perna_retracao] > limiar_persistencia:
        contador[perna_retracao] = 1

    for i in range(n_pernas):
        if contador[i] > 0:
            contador[i] += 1
        if contador[i] > passos_persistencia:
            contador[i] = 0

    # --- tropeco: forca de contato projetada no rumo da mosca ---
    mascara = np.zeros(n_pernas, dtype=np.bool_)
    n_elos = forcas.shape[1]
    for i in range(n_pernas):
        for j in range(n_elos):
            proj = (forcas[i, j, 0] * rumo[0] + forcas[i, j, 1] * rumo[1]
                    + forcas[i, j, 2] * rumo[2])
            if proj < limiar_forca:
                mascara[i] = True

    # --- CPG: uma passada de Euler nos osciladores acoplados ---
    dtheta = np.empty(n_pernas)
    dr = np.empty(n_pernas)
    for i in range(n_pernas):
        acopla = 0.0
        for j in range(n_pernas):
            acopla += (magnitudes[j] * acoplamento[i, j]
                       * np.sin(fases[j] - fases[i] - vies_fase[i, j]))
        dtheta[i] = dois_pi * freqs[i] + acopla
        dr[i] = convergencia[i] * (amps[i] - magnitudes[i])
    for i in range(n_pernas):
        fases[i] += dtheta[i] * dt
        magnitudes[i] += dr[i] * dt

    # --- angulos das juntas ---
    avalia_ppoly_periodico(x_pp, c_pp, fases, psi_buf)
    n_dof = neutro.shape[1]
    for i in range(n_pernas):
        if contador[i] > 0 or i == perna_retracao:
            retracao[i] += taxas_retracao[0] * dt
        else:
            v = retracao[i] - taxas_retracao[1] * dt
            retracao[i] = v if v > 0.0 else 0.0
        if mascara[i]:
            tropeco[i] += taxas_tropeco[0] * dt
        else:
            v = tropeco[i] - taxas_tropeco[1] * dt
            tropeco[i] = v if v > 0.0 else 0.0

        if retracao[i] > 0.0:
            correcao = retracao[i]
            tropeco[i] = 0.0
        else:
            correcao = tropeco[i]
        if correcao < 0.0:
            correcao = 0.0
        elif correcao > max_correcao:
            correcao = max_correcao

        fase_mod = fases[i] % dois_pi
        ganho = np.interp(fase_mod, pontos_fase[i], incrementos)
        correcoes[i] = correcao * ganho

        base = i * n_dof
        for j in range(n_dof):
            ang = neutro[i, j] + magnitudes[i] * (psi_buf[i, base + j] - neutro[i, j])
            angulos[indices[i, j]] = ang + correcao * ganho * vetores_corr[i, j]

        adesao[i] = not (swing_ini[i] < fase_mod < swing_fim[i])
    return perna_retracao
