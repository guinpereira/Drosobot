"""
Retina -> taxa de disparo. É aqui que a ASSUMPTION mora.

O conectoma diz quem liga em quem e com que peso. Ele **não** diz como a
intensidade de um omatídeo vira probabilidade de disparo — isso é escolha
nossa, e este arquivo é o único lugar onde ela é feita.

Duas transduções, porque são dois experimentos diferentes:

    Looming      detecta EXPANSÃO: a fração escura crescendo
    Optomotor    detecta MOVIMENTO HORIZONTAL, com sinal, por olho

A distinção não é cosmética. Rodar o optomotor com a transdução de looming
mediria expansão num estímulo que não expande — o resultado seria zero, ou
ruído, e pareceria um resultado científico.
"""
from __future__ import annotations

import numpy as np


class TransducaoLooming:
    """
    Fração escura crescendo -> taxa, igual nos dois olhos.

    É a transdução que produziu todos os resultados de Giant Fiber deste
    projeto. Mantida sem uma vírgula de diferença: a média móvel da fração
    escura, a parte positiva da diferença, o ganho e o teto.

    ASSUMPTION: `DARK_THRESHOLD`, `LOOM_GAIN`, `LOOM_MAX_HZ` e a constante de
    tempo da média móvel. Nenhum vem do conectoma.
    """

    def __init__(self, limiar_escuro: float, ganho: float, teto_hz: float,
                 vision_hz: float):
        self.limiar = limiar_escuro
        self.ganho = ganho
        self.teto = teto_hz
        self.tau = 0.3 * vision_hz
        self.lento = None

    def taxa(self, retina) -> tuple[float, np.ndarray]:
        """Devolve (hz único, fração escura por olho)."""
        escuro = (retina < self.limiar).mean(axis=1)
        if self.lento is None:
            self.lento = escuro.copy()
        expansao = np.clip(escuro - self.lento, 0, None)
        self.lento += (escuro - self.lento) / self.tau
        hz = float(np.clip(expansao * self.ganho, 0, self.teto).max())
        return hz, escuro

    def reset(self) -> None:
        self.lento = None


class TransducaoOptomotor:
    """
    Fluxo óptico horizontal, com sinal, por olho.

    ## O estimador

    Restrição do fluxo óptico em uma dimensão: para um padrão que se desloca
    horizontalmente com velocidade `u`, vale `∂I/∂t + u·∂I/∂x = 0`. Com muitos
    omatídeos, o `u` que melhor explica o quadro sai por mínimos quadrados:

        u = -Σ(Iₜ·Iₓ) / Σ(Iₓ² + ε)

    É Lucas–Kanade de uma dimensão, aplicado ao olho inteiro. `Iₓ` é a
    derivada espacial horizontal, estimada entre omatídeos vizinhos em coluna;
    `Iₜ` é a diferença entre quadros consecutivos.

    Dá um número **com sinal** por olho, que é o que separa "da direita para a
    esquerda" de "da esquerda para a direita". Sem sinal, as duas condições do
    experimento seriam indistinguíveis — e reportar diferença entre elas seria
    reportar ruído.

    ## Onde estão os omatídeos

    O `ommatidia_id_map` do FlyGym diz qual omatídeo cobre qual pixel. O
    centroide desses pixels dá a posição de cada omatídeo no campo visual, e é
    daí que sai a vizinhança horizontal. Geometria do modelo, não suposição
    nossa.

    ## O que é ASSUMPTION

    O ganho `u -> Hz`, o teto, e a decisão de que cada olho excita a população
    T4/T5 do **seu** lado quando o fluxo é de frente para trás. Nenhum desses
    números vem do conectoma. A direção preferida dos T4/T5 reais é uma
    propriedade medida da célula; aqui ela é uma convenção declarada.
    """

    def __init__(self, ganho_hz_por_unidade: float = 40.0,
                 teto_hz: float = 20.0, n_vizinhos: int = 6):
        self.ganho = float(ganho_hz_por_unidade)
        self.teto = float(teto_hz)
        self.n_vizinhos = int(n_vizinhos)
        self.anterior = None
        self._vizinhos = None
        self._dx = None

    # ------------------------------------------------------------ geometria

    def prepara(self) -> None:
        """
        Resolve a vizinhança horizontal dos omatídeos. Uma vez.

        Para cada omatídeo, os `n_vizinhos` mais próximos em coluna dentro de
        uma faixa de linhas parecida — é entre eles que a derivada espacial faz
        sentido. Fora dessa faixa, dois omatídeos próximos em coluna podem
        estar em alturas muito diferentes do campo visual.
        """
        from flygym.vision.retina import Retina

        mapa = np.asarray(Retina().ommatidia_id_map)
        n = int(mapa.max())
        linhas, colunas = np.zeros(n), np.zeros(n)
        for i in range(1, n + 1):
            ys, xs = np.nonzero(mapa == i)
            if len(xs):
                linhas[i - 1] = ys.mean()
                colunas[i - 1] = xs.mean()

        # vizinho horizontal: mesma faixa de linha, coluna mais próxima à
        # direita. É a diferença que aproxima ∂I/∂x.
        tol = (linhas.max() - linhas.min()) / 40.0
        viz = np.full(n, -1, dtype=np.int64)
        dx = np.zeros(n)
        for i in range(n):
            perto = np.nonzero(np.abs(linhas - linhas[i]) <= tol)[0]
            direita = perto[colunas[perto] > colunas[i]]
            if len(direita):
                j = direita[np.argmin(colunas[direita])]
                viz[i] = j
                dx[i] = colunas[j] - colunas[i]
        self._vizinhos = viz
        self._dx = dx

    # -------------------------------------------------------------- estimador

    def taxa(self, retina) -> tuple[np.ndarray, np.ndarray]:
        """
        Devolve (fluxo por olho, fração escura por olho).

        Fluxo positivo = padrão indo para colunas maiores naquele olho.
        """
        if self._vizinhos is None:
            self.prepara()
        r = np.asarray(retina, dtype=np.float64)
        escuro = (r < 0.4).mean(axis=1)
        if self.anterior is None or self.anterior.shape != r.shape:
            self.anterior = r.copy()
            return np.zeros(r.shape[0]), escuro

        it = r - self.anterior
        self.anterior = r.copy()

        fluxo = np.zeros(r.shape[0])
        viz, dx = self._vizinhos, self._dx
        val = viz >= 0
        for olho in range(r.shape[0]):
            ix = np.zeros(r.shape[1])
            ix[val] = (r[olho, viz[val]] - r[olho, val]) / dx[val]
            num = float(np.dot(it[olho], ix))
            den = float(np.dot(ix, ix)) + 1e-9
            fluxo[olho] = -num / den
        return fluxo, escuro

    def hz_por_lado(self, fluxo) -> tuple[float, float]:
        """
        Fluxo -> taxa dos T4/T5 esquerdos e direitos. Devolve (hz_L, hz_R).

        ## Por que o MODO COMUM, e não a diferença entre olhos

        Medido, com a mosca praticamente parada (drive 0) e o tambor a 180°/s:

            tambor +180°/s    fluxo  L -1,48   R -1,48
            tambor -180°/s    fluxo  L +4,35   R +4,48
            tambor parado     fluxo  L +0,05   R +0,09

        Os dois olhos dão o MESMO sinal, e ele inverte com a direção. Isso é o
        esperado: um tambor girando em torno do animal é rotação do campo
        visual (yaw), não translação. Cada câmera tem a sua orientação, então
        "coluna crescente" aponta para lados opostos do corpo nos dois olhos —
        e o sinal comum é justamente a medida do giro.

        Usar a diferença L-R aqui mediria translação, que este estímulo não
        produz, e daria ruído em volta de zero.

        ## O que é convenção, e não resultado

        Qual sinal excita qual lado é **convenção declarada**. T4/T5 reais são
        seletivos a direção, e as populações que alimentam HS de cada lado
        respondem a direções opostas — mas qual delas corresponde ao nosso
        `+` depende da orientação das câmeras do modelo, que não foi validada
        contra medida biológica.

        Consequência: este experimento mede **se as duas condições diferem**, e
        não se a mosca vira para o lado biologicamente correto. Afirmar a
        segunda coisa exigiria calibrar contra dado de optomotor real.

        ## Assimetria medida, não explicada

        |−1,48| contra |+4,35|: o estimador não é simétrico entre as duas
        direções. Fica registrado. Não foi ajustado nenhum ganho para
        compensar — compensar sem entender seria fabricar simetria.
        """
        u = float(np.mean(np.asarray(fluxo)))
        taxa = min(abs(u) * self.ganho, self.teto)
        return (taxa, 0.0) if u > 0 else (0.0, taxa)

    def reset(self) -> None:
        self.anterior = None
