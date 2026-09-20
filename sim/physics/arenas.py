"""
As arenas dos experimentos, além do looming.

    optomotor     um anel de postes que gira em torno da mosca
    obstaculos    pilares fixos no chão, à esquerda / no centro / à direita

Ambas são construídas com `MjSpec` sobre o `FlatGroundWorld` do FlyGym 2.x, a
mesma via da arena de looming (`looming_world.py`). O chão é o mesmo nas três,
então mudar de experimento não muda a física do contato com o solo.

## Optomotor: por que postes que giram, e não uma textura

O reflexo optomotor responde a **movimento de padrão no campo visual**. A forma
canônica é um tambor listrado girando em volta do animal.

Aqui o tambor é um anel de cilindros verticais, todos filhos de um corpo
**mocap** que gira. Mocap pela mesma razão do looming: são estímulo, não objetos
com que a mosca interage. Se tivessem massa e colisão, um encontro mudaria a
física e o experimento deixaria de ser visual.

Uma textura listrada no horizonte seria mais fiel ao tambor de laboratório, mas
exigiria material e mapeamento de textura na câmera dos omatídeos — e a retina
do FlyGym lê **intensidade por omatídeo**, que é exatamente o que postes escuros
contra fundo claro produzem. Postes dão o mesmo sinal com geometria que já
sabemos que a retina enxerga.

## Obstáculos: estes SIM colidem

Ao contrário dos dois estímulos acima, os pilares são obstáculos físicos. É o
ponto do experimento — medir se o circuito guia desvio, e não apenas se reage.

**Mas não por `contype`/`conaffinity`.** Este modelo não usa esse mecanismo:
todos os 69 geoms da mosca têm `(0, 0)`, e o contato com o chão vem de 55
`<pair>` declarados um a um. Um pilar com `contype=1` atravessa a mosca sem um
único contato — `0 & 1 = 0`. Ver `declara_pares_obstaculo`.

Consequência honesta: cada pilar soma 55 pares de colisão (55 → 110 com um
pilar), e o custo da física sobe com eles. Isso aparece em `resumo()` do
adaptador e vai para o `metadata.json` de cada corrida.
"""
from __future__ import annotations

import numpy as np

import mujoco as mj

# Postes do anel optomotor: altura, raio e quantos. Doze postes num anel de
# 20 mm dão ~30° de período angular, dentro da faixa em que o optomotor de
# Drosophila responde bem. Não é um ajuste fino — é uma escolha declarada.
OPTO_N_POSTES = 12
OPTO_RAIO_ANEL_MM = 20.0
OPTO_ALTURA_MM = 8.0
OPTO_RAIO_POSTE_MM = 1.2
OPTO_VEL_GRAUS_S = 60.0

# Pilares do campo de obstáculos
OBST_RAIO_MM = 1.5
OBST_ALTURA_MM = 4.0
OBST_DISTANCIA_MM = 12.0


def constroi_mundo_optomotor(n_postes: int = OPTO_N_POSTES,
                             raio_anel_mm: float = OPTO_RAIO_ANEL_MM,
                             altura_mm: float = OPTO_ALTURA_MM,
                             raio_poste_mm: float = OPTO_RAIO_POSTE_MM):
    """
    `FlatGroundWorld` mais um anel de postes num corpo mocap.

    Devolve (world, nome_do_corpo). O corpo é único e os postes são geoms dele:
    girar o corpo gira o anel inteiro, com uma escrita em `mocap_quat` por
    passo em vez de N.
    """
    from flygym.compose import FlatGroundWorld

    world = FlatGroundWorld(name="optomotor_world")
    spec = world.mjcf_root

    corpo = spec.worldbody.add_body(name="optomotor_tambor", pos=[0.0, 0.0, 0.0])
    corpo.mocap = True
    for i in range(n_postes):
        a = 2.0 * np.pi * i / n_postes
        corpo.add_geom(
            name=f"opto_poste_{i}",
            type=mj.mjtGeom.mjGEOM_CYLINDER,
            size=[raio_poste_mm, altura_mm * 0.5, 0.0],
            pos=[raio_anel_mm * np.cos(a), raio_anel_mm * np.sin(a),
                 altura_mm * 0.5],
            rgba=[0.05, 0.05, 0.05, 1.0],
            # estímulo visual, não obstáculo: não colide com nada
            contype=0,
            conaffinity=0,
            group=0,       # grupo 0 é o que a câmera dos omatídeos enxerga
        )
    return world, "optomotor_tambor"


def constroi_mundo_obstaculos(posicoes,
                              raio_mm: float = OBST_RAIO_MM,
                              altura_mm: float = OBST_ALTURA_MM):
    """
    `FlatGroundWorld` mais pilares FIXOS que colidem.

    `posicoes` é uma lista de (x, y) em mm, no referencial do mundo. Sem corpo
    mocap: eles não se movem, e por isso não há nada a atualizar por passo.
    """
    from flygym.compose import FlatGroundWorld

    world = FlatGroundWorld(name="obstacle_world")
    spec = world.mjcf_root

    nomes = []
    for i, (x, y) in enumerate(posicoes):
        corpo = spec.worldbody.add_body(name=f"obstaculo_{i}",
                                        pos=[float(x), float(y),
                                             altura_mm * 0.5])
        corpo.add_geom(
            name=f"obstaculo_geom_{i}",
            type=mj.mjtGeom.mjGEOM_CYLINDER,
            size=[raio_mm, altura_mm * 0.5, 0.0],
            rgba=[0.10, 0.10, 0.12, 1.0],
            group=0,
        )
        nomes.append(f"obstaculo_geom_{i}")
    return world, nomes


def declara_pares_obstaculo(spec, nomes_obstaculo, margem: float = 0.001) -> int:
    """
    Faz os pilares colidirem com a mosca. Devolve quantos pares foram criados.

    O FlyGym 2.x **não** usa `contype`/`conaffinity`: todos os 69 geoms da mosca
    têm `(0, 0)`, e o contato com o chão vem de 55 `<pair>` declarados um a um
    (`ground_plane × fly/<geom>`). Um geom novo com `contype=1` nunca encosta em
    nada — `0 & 1 = 0`.

    Isso custou uma corrida de 3 s em que a mosca atravessou o pilar sem um
    único contato, com a retina vendo o obstáculo o tempo todo. O sintoma
    ("passou por dentro") não aponta para a causa ("o modelo usa pares
    explícitos"), e por isso está escrito aqui.

    Os pares novos espelham os que já existem: os mesmos geoms da mosca, a mesma
    margem. Cada pilar soma 55 pares, e isso aparece em `resumo()`.
    """
    fly_geoms = []
    for par in spec.pairs:
        for nome in (par.geomname1, par.geomname2):
            if nome.startswith("fly/") and nome not in fly_geoms:
                fly_geoms.append(nome)
    n = 0
    for alvo in nomes_obstaculo:
        for g in fly_geoms:
            par = spec.add_pair()
            par.geomname1 = alvo
            par.geomname2 = g
            par.margin = margem
            n += 1
    return n


class EstimuloOptomotor:
    """
    O tambor girando. Só cinemática do experimento.

    `vel_graus_s` positivo gira no sentido anti-horário visto de cima, que a
    mosca (olhando para +x) vê como padrão indo da **direita para a esquerda**.
    Negativo inverte. É o parâmetro que separa as duas condições.
    """

    def __init__(self, vel_graus_s: float = OPTO_VEL_GRAUS_S,
                 atraso_s: float = 0.2):
        self.vel = float(vel_graus_s)
        # O tambor fica parado no começo: sem isso a mosca recebe movimento
        # antes de o controlador de marcha assentar, e a resposta mede as duas
        # coisas misturadas.
        self.atraso = float(atraso_s)

    def angulo(self, t_s: float) -> float:
        """Ângulo do tambor, em radianos."""
        if t_s < self.atraso:
            return 0.0
        return np.radians(self.vel) * (t_s - self.atraso)

    def quaternion(self, t_s: float) -> np.ndarray:
        """Rotação em torno de z, no formato do MuJoCo (w, x, y, z)."""
        meio = self.angulo(t_s) * 0.5
        return np.array([np.cos(meio), 0.0, 0.0, np.sin(meio)], dtype=float)


class CampoObstaculos:
    """
    Onde ficam os pilares. Fixo: não há o que atualizar por passo.

    `lado` decide o arranjo, e é a condição experimental:

        centro     um pilar à frente, na linha de marcha
        esquerda   um pilar deslocado para +y
        direita    um pilar deslocado para -y
        campo      três pilares, um de cada
    """

    ARRANJOS = {
        "centro": [(OBST_DISTANCIA_MM, 0.0)],
        "esquerda": [(OBST_DISTANCIA_MM, 4.0)],
        "direita": [(OBST_DISTANCIA_MM, -4.0)],
        "campo": [(OBST_DISTANCIA_MM, 4.0), (OBST_DISTANCIA_MM + 6.0, 0.0),
                  (OBST_DISTANCIA_MM + 12.0, -4.0)],
    }

    def __init__(self, lado: str = "centro", distancia_mm: float | None = None):
        if lado not in self.ARRANJOS:
            raise ValueError(
                f"arranjo desconhecido: {lado!r} (de {sorted(self.ARRANJOS)})")
        self.lado = lado
        base = self.ARRANJOS[lado]
        if distancia_mm is None:
            self.posicoes = list(base)
        else:
            # desloca o arranjo inteiro, mantendo a geometria relativa
            d = float(distancia_mm) - OBST_DISTANCIA_MM
            self.posicoes = [(x + d, y) for x, y in base]
