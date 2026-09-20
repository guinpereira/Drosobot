"""
Carrega o Male CNS real como CSR, pronto pra subir na GPU.

    from sim.neural import carrega_male_cns
    c = carrega_male_cns()              # 164.451 neuronios, 25,5M arestas
    c = carrega_male_cns(subconjunto=50000)

Le os binarios que `benchmarks/neural/build_csr.py` gera a partir dos downloads
oficiais do dataset. Eles nao sao versionados (195 MiB); o script os reconstroi.

## Sinapse nao e aresta

O modelo do Drosobot agrega por par de neuronios: `signed_weights()` soma as
sinapses entre dois neuronios e produz UM peso. Entao a unidade computacional e a
aresta.

    sinapses biologicas   123.967.037
    arestas computacionais 25.550.583      (4,85 sinapses por aresta)

O peso continua representando a quantidade de sinapses, como sempre:
`w_mV = n_sinapses * sinal * W_SYN`.

## Estatico e dinamico

O que este modulo devolve e SO o estatico: conectividade, peso, sinal, bodyId.
Nada de v, g ou spike -- isso nasce no backend e mora na GPU.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .model import Conectoma

RAIZ = Path(__file__).resolve().parents[2]
CSR = RAIZ / "data" / "male-cns" / "csr"


class ConectomaAusente(FileNotFoundError):
    pass


def _exige(caminho: Path) -> Path:
    if not caminho.exists():
        raise ConectomaAusente(
            f"falta {caminho}.\n"
            "Gere com:\n"
            "  .venv\\Scripts\\python benchmarks\\neural\\build_csr.py\n"
            "(precisa dos feather de https://male-cns.janelia.org/download/ "
            "em data/male-cns/)")
    return caminho


def carrega_male_cns(subconjunto: int | None = None) -> Conectoma:
    """
    CSR do Male CNS. `subconjunto` usa o arquivo reduzido, se existir.

    O subconjunto e escolhido por GRAU DE SAIDA no build_csr.py, nao por sorteio:
    cortar aleatoriamente daria um grafo mais facil que o real e o benchmark
    ficaria otimista.
    """
    sufixo = f"_{subconjunto}" if subconjunto else "_full"
    ro = np.fromfile(_exige(CSR / f"row_offsets{sufixo}.bin"), dtype=np.int32)
    tg = np.fromfile(_exige(CSR / f"targets{sufixo}.bin"), dtype=np.int32)
    wt = np.fromfile(_exige(CSR / f"weights{sufixo}.bin"), dtype=np.float32)
    ids = np.fromfile(_exige(CSR / f"body_ids{sufixo}.bin"), dtype=np.int64)
    return Conectoma(row_offsets=ro, targets=tg, weights=wt, body_ids=ids,
                     nome=f"male-cns-v1.0{sufixo}")


def metadados(subconjunto: int | None = None) -> dict:
    sufixo = f"_{subconjunto}" if subconjunto else "_full"
    arq = CSR / f"meta{sufixo}.json"
    return json.loads(arq.read_text(encoding="utf-8")) if arq.exists() else {}


def existe(subconjunto: int | None = None) -> bool:
    sufixo = f"_{subconjunto}" if subconjunto else "_full"
    return (CSR / f"row_offsets{sufixo}.bin").exists()


def subgrafo(c: Conectoma, indices: np.ndarray) -> Conectoma:
    """
    Recorta um subgrafo induzido, renumerado de 0 a k-1.

    Serve pra validacao: comparar GPU e referencia sobre um pedaco REAL do
    conectoma, com a distribuicao de grau e os pesos que o dataset tem, em vez
    de uma rede sintetica que nao se parece com nada.
    """
    sel = np.sort(np.asarray(indices, dtype=np.int64))
    mapa = np.full(c.n, -1, dtype=np.int32)
    mapa[sel] = np.arange(len(sel), dtype=np.int32)

    linhas_ro = [0]
    alvos, pesos = [], []
    for i in sel:
        a, b = c.row_offsets[i], c.row_offsets[i + 1]
        t, w = c.targets[a:b], c.weights[a:b]
        novo = mapa[t]
        manter = novo >= 0
        alvos.append(novo[manter])
        pesos.append(w[manter])
        linhas_ro.append(linhas_ro[-1] + int(manter.sum()))

    return Conectoma(
        row_offsets=np.asarray(linhas_ro, dtype=np.int32),
        targets=(np.concatenate(alvos) if alvos else np.zeros(0, np.int32)
                 ).astype(np.int32),
        weights=(np.concatenate(pesos) if pesos else np.zeros(0, np.float32)
                 ).astype(np.float32),
        body_ids=c.body_ids[sel],
        nome=f"{c.nome}-sub{len(sel)}",
    )


def rede_sintetica(n: int, grau: int, semente: int = 0,
                   frac_inibitoria: float = 0.4) -> Conectoma:
    """
    Rede pequena e controlada, pra testes de corretude que precisam de um caso
    conhecido (convergencia no mesmo alvo, peso negativo, grau zero).

    Nao serve pra medir desempenho: a distribuicao de grau e uniforme, e a real
    nao e (media 155,6, mediana 114, max 11.203).
    """
    rng = np.random.default_rng(semente)
    graus = np.full(n, grau, dtype=np.int32)
    if n > 2:
        graus[0] = 0                     # um neuronio sem saida nenhuma
        graus[1] = min(n, grau * 3)      # e um com grau alto
    ro = np.zeros(n + 1, dtype=np.int32)
    np.cumsum(graus, out=ro[1:])
    e = int(ro[-1])
    tg = rng.integers(0, n, size=e, dtype=np.int64).astype(np.int32)
    w = rng.uniform(0.2, 2.0, size=e).astype(np.float32)
    neg = rng.random(e) < frac_inibitoria
    w[neg] *= -1
    return Conectoma(row_offsets=ro, targets=tg, weights=w,
                     body_ids=np.arange(n, dtype=np.int64),
                     nome=f"sintetica-{n}x{grau}")
