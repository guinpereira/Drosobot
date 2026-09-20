# O gargalo da física não era a física

Medição, diagnóstico e correção do custo do laço de física do Drosobot.
Resultado: **RTF 0,055× → 0,113×**, sem tocar em física, modelo, controlador ou
timestep, com igualdade bit a bit verificada.

Ferramenta: `benchmarks/physics/profile_flygym2_step.py`.
Código: `sim/physics/fastpath.py`. Prova: `tests/test_fastpath_equivalencia.py`.

---

## A medição que mudou o plano

A suposição em vigor era: *a física é 79% do relógio, logo o alvo é o modelo de
colisão*. A decomposição do passo disse outra coisa.

Looming, FlyGym 2.1.0 / MuJoCo 3.9, 3.000 passos, RX 6700 XT:

| etapa | ms/s simulado | % | µs/passo |
|---|---|---|---|
| controlador | 9.187 | 63,5% | 918 |
| observação | 2.435 | 16,8% | 243 |
| **`sim.step()`** | **1.588** | **11,0%** | **159** |
| retina | 879 | 6,1% | — |
| aplica ação | 163 | 1,1% | 16 |
| estímulo | 133 | 0,9% | 13 |
| **parede** | **14.474** | **100%** | |

**Só 11% do tempo estava dentro do `sim.step()`.** Os outros 89% eram Python em
volta da física.

Isso reorientou tudo: mexer em pares de colisão atacaria o menor pedaço, com o
maior risco científico. O trabalho estava em outro lugar — e era trabalho
*repetido*.

## O que estava sendo refeito 10.000 vezes por segundo

| o que | quanto |
|---|---|
| `dof_spec_to_jointdof` | 4 dataclasses congeladas por DOF, **42×/passo**, só para virar chave de dicionário |
| `_step_phase_gain` | dois arrays de 5 elementos remontados, 6×/passo |
| `get_joint_angles` | revalidação do nome da perna + `np.asarray`, 6×/passo |
| `_get_adhesion_onoff` | lookup por nome de perna, 6×/passo |
| `get_bodysegment_contact_forces` | lista de `BodySegment`, dict geom→saída e array de geoms reconstruídos por chamada, para 30 segmentos |
| `np.isin` | 4 varreduras/passo sobre um conjunto fixo de 18 geoms |
| `np.clip` escalar | array de zero dimensões alocado, 6×/passo |
| `interp1d` | 6 chamadas ao scipy, ~26 µs de overhead de Python cada, para ~1 µs de conta |

Nada disso é conta. É bookkeeping — e todo ele sobre dados que **não mudam
durante a corrida**.

## O que foi feito

`sim/physics/fastpath.py` pré-resolve o que é constante e escreve em buffers já
alocados:

- o mapa `(perna, dof) → posição na saída` vira uma tabela de inteiros;
- `step_points`, vetores de correção e `swing_period` estendido viram constantes
  por perna;
- a tabela geom→saída e as máscaras de contato viram arrays booleanos indexados
  pelo próprio id do geom, no lugar de `np.isin`;
- as seis `CubicSpline` do passo pré-programado **compartilham os mesmos 45
  nós**, então viram um `PPoly` só e **uma** chamada ao scipy no lugar de seis.

A última avalia mais do que precisa — as seis pernas em cada uma das seis fases,
usando só a diagonal — e ainda sai na frente, porque o custo é overhead **por
chamada**, não por ponto.

## Por que isto não é uma mudança de física

Mesmo modelo, mesmo `timestep`, mesmo controlador, mesmo `float64`, mesma ordem
de operações. Não há aproximação, não há atalho numérico, não há tolerância.

O critério é **igualdade exata**. `tests/test_fastpath_equivalencia.py` roda os
dois caminhos lado a lado por 2.000 passos, a partir do mesmo estado, e reprova
na primeira diferença de bit:

```
controlador: 2000 passos, diferenca maxima 0.0e+00 rad, adesao divergente 0x
contato:     2000 passos, 2000 com forca nao nula, diferenca maxima 0.0e+00
```

Não há limiar para afrouxar: **1 ULP por passo, realimentado 10.000 vezes por
segundo simulado num sistema com contato, diverge.**

O teste já pagou por si. A primeira versão usava o `get_adhesion_onoff` do passo
pré-programado, mas o controlador **estende** o fim do swing por
`swing_extension` antes de decidir a adesão. A divergência apareceu no passo 0.

### E o upstream ficou intacto

`research/upstream/flygym` é clone de leitura. Editar lá faria o projeto depender
de um fork silencioso: quem clonasse o repositório teria comportamento diferente
do nosso sem nenhum sinal. A composição é nossa, aparece no diff, e o teste é a
prova.

## O resultado

Laço de física, mesma medição:

| etapa | antes | depois |
|---|---|---|
| controlador | 918 µs | **228 µs** |
| observação | 243 µs | **72 µs** |
| `sim.step()` | 159 µs | 153 µs |
| **parede** | **14.474 ms/s** | **5.655 ms/s** |

Ponta a ponta, whole CNS, 1 s de mosca:

| | antes | depois |
|---|---|---|
| **RTF** | 0,055× | **0,113×** |
| física | 14.405 ms/s (78,9%) | **5.167 ms/s (58,3%)** |
| neural | 3.363 ms/s (18,4%) | 3.068 ms/s (34,6%) |
| visão | 14 ms/s | 14 ms/s |

O neural não foi tocado; ele só passou a pesar mais porque a física encolheu.

**Comportamento idêntico**, verificado na comparação 4-vias com a mesma semente:
circuito 2 fugas / 8 spikes do GF, whole 0 fugas / 1 spike, e excitação,
inibição, líquido, `v` mínimo e posição final todos iguais ao que estava gravado
antes da mudança.

## Onde o custo está agora

Dentro do laço de física: controlador 40%, `sim.step()` 27%, retina 13%,
observação 13%.

O `mj_step` passou de 11% para 27% do laço — não porque ficou mais lento, mas
porque o que estava em volta encolheu 2,5×. O que sobra é cada vez mais física
de verdade.

## O que sobrou, e o que não vale a pena

**O que ainda dá para atacar sem tocar em física:** cerca de 158 µs/passo são a
chamada do scipy para a interpolação do passo pré-programado, já em lote. Reduzir
isso exigiria evaluar `PPoly` por dentro, acoplando o projeto a API privada do
scipy — frágil demais para o ganho.

**O que NÃO foi tocado, de propósito:**

- **pares de colisão.** `legs` continua o padrão. `tarsi` reprovou em curva (ver
  [`COLLISION_PAIR_AUDIT.md`](COLLISION_PAIR_AUDIT.md)) e não voltou a ser
  considerado. Com `mj_step` em 27% do laço, o teto de qualquer poda de colisão
  é menor que o ganho que já veio de graça.
- **geometria de colisão simplificada.** A hipótese de trocar malhas por
  cápsulas/elipsoides não foi implementada: o NeuroMechFly já usa primitivas
  para contato, e a narrowphase é 5% do `mj_step`. Não há gordura ali.
- **timestep.** `1e-4` s, intocado.

## Achado colateral

Os elos de tropeço (`tibia`, `tarsus1`, `tarsus2`) **não encostam no chão** neste
experimento: 0 passos com força não nula em 6.000 medidos. Só `tarsus4`/`tarsus5`
tocam. A detecção de tropeço do controlador recebe zero o tempo todo aqui.

Fica registrado. Não foi alterado — o controlador é do upstream e o experimento
é o que é.
