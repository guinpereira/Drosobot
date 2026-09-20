# Que subset do MuJoCo o NeuroMechFly realmente usa

Inventario medido no modelo que o Drosobot roda hoje (`flygym 1.2.1`,
`mujoco 3.2.7`), montado como em `sim/experiments/corpo.py`: `Fly` +
`HybridTurningController`, `timestep=1e-4`, 36 sensores de contato nas pernas.

Serve pra responder: **o que um backend nosso precisaria suportar**, e
**onde o tempo realmente vai**.

---

## 1. Inventario do modelo

```
nq        94      nv       93      nu       48
nbody     72      njnt     88      ngeom    70
nmesh     69      nsite    37      nsensor 175
ntendon    0      neq       0      nM     1089
```

| feature | usa? | detalhe |
|---|---|---|
| free joint | sim | 1 (o torax) |
| hinge | sim | **87** |
| ball / slide | **nao** | 0 |
| tendoes | **nao** | `ntendon = 0` |
| restricoes de igualdade | **nao** | `neq = 0` |
| atuadores | sim | 48: **42 de junta** + **6 de adesao** (`trntype` 5) |
| atuadores musculares | **nao** | nenhum neste caminho |
| geoms primitivos | quase nao | 1 plano (o chao) |
| geoms de malha | **sim, 69** | e onde esta o problema |
| sensores | sim | 175 |
| camera / render | sim | retina dos dois olhos a 100 Hz |
| integrador | **Euler** | `opt.integrator = 0` |
| solver | **Newton** | `opt.solver = 2`, `iterations = 1000` |
| cone de atrito | piramidal | `opt.cone = 0`, `impratio = 1.0` |
| noslip | **sim** | `noslip_iterations = 100` |

Em regime de caminhada: `ncon ~ 2`, `nefc ~ 8`, `solver_niter = 3`.

O `iterations = 1000` e teto, nao trabalho: o Newton converge em 3. O
`noslip_iterations = 100` e trabalho de verdade e **o MJWarp nao suporta noslip**
(ele zera a opcao e avisa) -- ou seja, o caminho GPU do FlyGym ja muda esta
fisica.

---

## 2. Onde o tempo vai -- medido

Timers internos do MuJoCo (`mjcb_time`), 800 passos:

| etapa | ms/passo | % |
|---|---|---|
| STEP | 1,526 | 100,0 |
| FORWARD | 1,515 | 99,3 |
| POSITION | 1,406 | 92,1 |
| POS_COLLISION | 1,361 | 89,2 |
| **COL_NARROW** | **1,330** | **87,2** |
| CONSTRAINT | 0,086 | 5,7 |
| POS_PROJECT | 0,015 | 1,0 |
| POS_KINEMATICS | 0,011 | 0,7 |
| POS_INERTIA | 0,010 | 0,7 |
| ADVANCE | 0,010 | 0,6 |
| COL_BROAD | 0,002 | 0,1 |
| ACTUATION | 0,001 | 0,1 |

E o passo completo, do ponto de vista do Drosobot (3000 passos):

| caminho | us/passo | s de relogio / s simulado |
|---|---|---|
| `mj_step` puro | 1583 | 15,8 |
| `Simulation.step()` 1.x sem visao | 2610 | 26,1 |
| `Simulation.step()` 1.x com visao 100 Hz | 3002 | 30,0 |

---

## 3. A causa do narrowphase

```
npair (pares de colisao explicitos)  : 2220
   perna x perna                     : 2172   (98%)
   corpo x perna (inclui o chao)     :   48
geoms de malha                       : 69 de 70
vertices totais                      : 247.724  (mediana 1283, max 41.147)
faces totais                         : 502.781
cascos convexos pre-computados       : 0 de 69
```

2220 pares por passo x 10.000 passos por segundo simulado =
**22,2 milhoes de testes narrowphase por segundo simulado**, ~0,6 us cada.

**98% disso e auto-colisao entre pernas.** A mosca passa 87% do orcamento de
fisica verificando se as proprias pernas se tocam.

Das maiores malhas, varias nem participam de colisao util:

```
mesh_Head      41.147 vertices
mesh_RArista   26.595
mesh_LArista   26.595
mesh_REye      26.066
mesh_LEye      26.066
```

---

## 4. Matriz de features por backend

| feature | MuJoCo CPU | MJX-JAX | MJWarp | backend nosso (futuro) |
|---|---|---|---|---|
| hinge + free joint | sim | sim | sim | necessario |
| atuador de junta | sim | sim | sim | necessario |
| **atuador de adesao** | sim | sim | sim | **necessario** (6 pernas) |
| colisao de malha | sim | limitado | sim | **evitar** (ver secao 5) |
| colisao de primitiva | sim | sim | sim | necessario |
| solver Newton | sim | sim | sim | necessario |
| cone piramidal | sim | sim | sim | necessario |
| **noslip** | sim | nao sei | **nao** | decidir e documentar |
| integrador Euler | sim | sim | sim | necessario |
| tendoes | sim | sim | sim | **nao usamos** |
| restricoes de igualdade | sim | sim | sim | **nao usamos** |
| atuador muscular | sim | - | - | nao usamos (hoje) |
| render de camera | sim | limitado | sim (batch) | necessario pra retina |

O subset e pequeno: hinge, free joint, atuador de posicao, adesao, contato com
cone piramidal, Newton, Euler. Sem tendao, sem equality, sem musculo.

Isso e uma boa noticia pra ambicao de longo prazo e uma ma noticia pra pressa: o
subset e viavel, mas **o custo nao esta no subset, esta na geometria**.

---

## 5. O caminho obvio, e por que ele nao e automatico

O FlyGym 2.x expoe `GeomFittingOption.ALL_TO_CAPSULES`, e o benchmark oficial
tem um parametro `simplify_geom` que a liga. Trocar 69 malhas por capsulas
transforma 2172 testes malha-malha em testes analiticos segmento-segmento.

Pela estrutura do problema, e o maior ganho isolado disponivel -- e e **CPU**,
sem GPU, sem Vulkan, sem NVIDIA.

**Mas isso muda a fisica de contato.** Uma capsula nao tem a forma de um tarso.
Muda onde o pe toca, muda o atrito efetivo, pode mudar a marcha.

Regra do projeto: otimizacao que altera a ciencia e rejeitada; otimizacao que
*pode* alterar a ciencia e medida antes. Entao a ordem e:

1. gravar trajetoria de referencia com o modelo atual (posicao, orientacao,
   angulos de junta, eventos de contato, trajetoria de pe, frequencia de marcha,
   velocidade)
2. rodar com capsulas, mesma semente, mesmo drive
3. comparar com tolerancia declarada
4. **so entao** decidir -- e, decidindo a favor, deixar a opcao visivel na
   interface, porque passa a ser outro modelo de contato

Alternativa menos invasiva, que vale medir junto: **podar a lista de pares**.
Muitos dos 2172 sao geometricamente impossiveis numa caminhada (tarso dianteiro
esquerdo contra tarso dianteiro direito, lados opostos do corpo). Podar pares
impossiveis nao muda a fisica dos contatos que de fato ocorrem -- mas "de fato
ocorrem" precisa ser medido numa corrida longa, nao presumido.

---

## 6. Consequencia pro plano

Antes desta medicao, o plano implicito era "fisica e lenta -> precisamos de GPU".

A medicao diz outra coisa:

- 87% do passo e colisao de malha, e o remedio e geometria, na CPU
- 5,7% e o solver de restricoes, que e onde o MJWarp poe a engenharia
- 39% de overhead do wrapper 1.x sai com a migracao pro 2.x, tambem na CPU

**Ha um caminho de CPU plausivel de ~30 s/s para a casa de poucos s/s, sem GPU
nenhuma.** Fisica em GPU continua interessante pra multi-mundo, mas deixa de ser
o proximo passo.

O que continua justificando GPU de verdade e o **neural**: 166.691 neuronios e
125M sinapses nao cabem no orcamento de CPU do laco em tempo interativo. E por
isso que a Fase 9 (neural GPU) vem antes da Fase 14 (fisica GPU) -- e agora tem
medicao por tras, nao so intuicao.
