"""
Arena de looming pro FlyGym 2.x.

Reproduz o EXPERIMENTO que o caminho FlyGym 1.x roda, nao a implementacao linha
por linha. O que tem que ser igual e o que a mosca ve:

    esfera de raio 3 mm
    aproxima de 30 mm ate 4 mm
    ciclo de 0,8 s, reinicia longe
    fica na altura do torax, a frente da mosca

No 1.x isso vem de `MovingObjArena` do `flygym.examples.vision`, com a posicao
imposta pelo laco (a arena tem `step()` vazio). Aqui a esfera e um corpo
**mocap**, que e o equivalente em MjSpec: posicao imposta por fora, sem dinamica
propria e sem entrar no solver.

Mocap importa por uma razao cientifica, nao de conveniencia: a esfera e
ESTIMULO, nao um objeto fisico com que a mosca interage. Se ela tivesse massa e
colisao, um encontro mudaria a fisica -- e ai os dois caminhos deixariam de
rodar o mesmo experimento.

## O que ainda difere do 1.x

O 1.x usa `MovingObjArena`, que tem chao proprio e parametros de contato
proprios. Aqui herdamos `FlatGroundWorld`. Chao plano nos dois casos, mas os
parametros de contato do 2.x sao os dele. Isso esta declarado em `resumo()` do
adaptador -- a comparacao de CUSTO vale, a de comportamento carrega essa
ressalva.
"""
from __future__ import annotations

import mujoco as mj
import numpy as np


# Raio da esfera de looming, em mm. Um so lugar: o mundo do MuJoCo e a esfera
# que a Unity desenha tem que ter o mesmo tamanho, senao a tela mente sobre o
# estimulo que a retina de fato viu.
RAIO_ESTIMULO = 3.0


def constroi_mundo_looming(raio_mm: float = RAIO_ESTIMULO,
                           pos_inicial=(30.0, 0.0, 2.5)):
    """
    `FlatGroundWorld` mais uma esfera mocap. Devolve (world, nome_do_corpo).

    O import e tardio porque este modulo so existe no `.venv-flygym2`.
    """
    from flygym.compose import FlatGroundWorld

    world = FlatGroundWorld(name="looming_world")
    spec = world.mjcf_root

    corpo = spec.worldbody.add_body(name="looming_obj", pos=list(pos_inicial))
    corpo.mocap = True
    corpo.add_geom(
        name="looming_geom",
        type=mj.mjtGeom.mjGEOM_SPHERE,
        size=[raio_mm, 0.0, 0.0],
        rgba=[0.0, 0.0, 0.0, 1.0],
        # contype/conaffinity 0: a esfera nao colide com nada. Ela e estimulo
        # visual, e um contato mudaria a fisica que estamos comparando.
        contype=0,
        conaffinity=0,
        # grupo 0 e o que a camera dos olhos enxerga (ela ignora 1 e 2)
        group=0,
    )
    return world, "looming_obj"


class Estimulo:
    """
    Onde a esfera esta em cada instante. E so geometria do experimento.

    Fica separada do adaptador de proposito: os dois caminhos (FlyGym 1 e 2)
    usam a MESMA regra de distancia, entao o estimulo nao pode estar escrito
    duas vezes.
    """

    def __init__(self, ciclo_s: float = 0.8, dist_longe: float = 30.0,
                 dist_perto: float = 4.0, altura_mm: float = 2.5,
                 azimute_graus: float = 0.0):
        self.ciclo_s = ciclo_s
        self.dist_longe = dist_longe
        self.dist_perto = dist_perto
        self.altura = altura_mm
        # De que lado a esfera se aproxima. 0 = de frente; positivo = pela
        # ESQUERDA da mosca (+y do MuJoCo). E o parametro que separa
        # "looming lateral" de "looming central" -- o mesmo estimulo, chegando
        # por outro ponto do campo visual, que e o que decide qual olho o ve.
        self.azimute = float(azimute_graus)

    def distancia(self, t_s: float) -> float:
        fase = (t_s % self.ciclo_s) / self.ciclo_s
        return self.dist_longe + (self.dist_perto - self.dist_longe) * fase

    def posicao(self, t_s: float, pos_mosca) -> np.ndarray:
        """
        Onde a esfera esta, em MUNDO.

        O azimute gira a direcao de aproximacao em torno da mosca, mantendo a
        distancia: a esfera anda no arco, nao se afasta. Com azimute 0 isto
        reduz exatamente ao caso antigo (`p[0] + d`, `p[1]`), entao as corridas
        centrais continuam comparaveis com o que ja foi medido.
        """
        d = self.distancia(t_s)
        p = np.asarray(pos_mosca, dtype=float)
        if self.azimute == 0.0:
            return np.array([p[0] + d, p[1], self.altura], dtype=np.float32)
        a = np.radians(self.azimute)
        return np.array([p[0] + d * np.cos(a), p[1] + d * np.sin(a),
                         self.altura], dtype=np.float32)
