"""
Converte os videos das simulacoes em GIF pro README.

O GitHub nao embute .mp4 local em markdown -- vira so um link. GIF anima inline.
Os .mp4 continuam no repo em resolucao cheia; o GIF e a versao que aparece na
pagina.

Roda:  .venv\Scripts\python sim\make_gifs.py
"""
from pathlib import Path

import imageio.v3 as iio
import numpy as np

OUT = Path(__file__).parent.parent / "docs" / "images"

# (video, gif, de quantos em quantos quadros pegar, largura alvo)
TRABALHOS = [
    ("flygym_optomotor.mp4", "flygym_optomotor.gif", 2, 400),
    ("flygym_escape.mp4", "flygym_escape.gif", 3, 400),
]


def reduzir(frame, largura):
    """Subamostra por passo inteiro -- sem dependencia de PIL/cv2 so pra isso."""
    h, w = frame.shape[:2]
    passo = max(1, round(w / largura))
    return frame[::passo, ::passo]


for nome_mp4, nome_gif, salto, largura in TRABALHOS:
    origem = OUT / nome_mp4
    if not origem.exists():
        print(f"pulando {nome_mp4}: nao existe (rode o script da simulacao antes)")
        continue
    quadros = iio.imread(origem, plugin="pyav")
    selecao = [reduzir(f, largura) for f in quadros[::salto]]
    destino = OUT / nome_gif
    iio.imwrite(destino, np.asarray(selecao), extension=".gif", fps=12, loop=0)
    mb = destino.stat().st_size / 1e6
    print(f"{nome_gif}: {len(selecao)} quadros, {selecao[0].shape[1]}x{selecao[0].shape[0]}, {mb:.2f} MB")
