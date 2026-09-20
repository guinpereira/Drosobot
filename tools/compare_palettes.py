"""Monta a tira comparativa Clay | Flybody | Drosophila.

As tres vem de `CaptureBodyShots`, na MESMA pose e na MESMA camera (bind pose,
perspectiva, yaw -40 pitch 22). So o esquema de cor muda -- e esse o ponto: com
pose ou enquadramento diferente a comparacao nao diria nada sobre material.

    python tools/compare_palettes.py
"""

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

RAIZ = Path(__file__).resolve().parents[1]
IMAGENS = RAIZ / "docs" / "images"

PAINEIS = [
    ("fly_clay_perspectiva.png", "CLAY", "DATA - cinza 0,5 do modelo"),
    ("fly_flybody_perspectiva.png", "FLYBODY", "referencia visual upstream"),
    ("fly_bindpose_perspectiva.png", "DROSOPHILA", "ASSUMPTION - apresentacao"),
]

LADO = 800          # cada painel, depois de reduzir
FAIXA = 64          # altura da tarja de rotulo


def fonte(tamanho: int):
    # Arial existe em qualquer Windows; em outro sistema cai no bitmap padrao,
    # que fica pequeno mas nao quebra a geracao
    for caminho in (r"C:\Windows\Fonts\arialbd.ttf", r"C:\Windows\Fonts\arial.ttf"):
        try:
            return ImageFont.truetype(caminho, tamanho)
        except OSError:
            continue
    return ImageFont.load_default()


def main() -> None:
    faltando = [n for n, _, _ in PAINEIS if not (IMAGENS / n).exists()]
    if faltando:
        raise SystemExit(
            "faltam capturas: " + ", ".join(faltando) + "\n"
            "gere com: Unity.exe -batchmode -quit -projectPath unity\\DrosobotLab "
            "-executeMethod Drosobot.EditorTools.CaptureBodyShots.Capturar")

    tira = Image.new("RGB", (LADO * len(PAINEIS), LADO + FAIXA), (11, 12, 16))
    desenho = ImageDraw.Draw(tira)
    f_titulo, f_nota = fonte(30), fonte(20)

    for i, (arquivo, titulo, nota) in enumerate(PAINEIS):
        img = Image.open(IMAGENS / arquivo).convert("RGB")
        img = img.resize((LADO, LADO), Image.LANCZOS)
        x = i * LADO
        tira.paste(img, (x, 0))
        desenho.text((x + 18, LADO + 10), titulo, font=f_titulo, fill=(235, 238, 245))
        desenho.text((x + 18, LADO + 40), nota, font=f_nota, fill=(150, 156, 168))
        if i:
            desenho.line([(x, 0), (x, LADO + FAIXA)], fill=(40, 43, 50), width=2)

    saida = IMAGENS / "fly_paletas_comparacao.png"
    tira.save(saida)
    print(f"salvo em {saida}  ({tira.width}x{tira.height})")


if __name__ == "__main__":
    main()
