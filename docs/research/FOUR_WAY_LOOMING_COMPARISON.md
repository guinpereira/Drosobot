# Comparação 4-vias: FlyGym 1.x × 2.x, circuito × whole CNS

Refeita em 20/09/2026, depois que a arena de looming passou a ser **a mesma nos
dois simuladores**. A versão anterior desta comparação rodava o adaptador 2.x em
chão plano, então a coluna de comportamento não valia nada — a mosca não via o
mesmo estímulo.

Corridas: `sim/compare_runtimes.py`, 1,0 s de mosca, `dt = 1e-4`,
`self_collisions="legs"`, backend `drosobot-opencl` (gfx1031).
Dados em `benchmarks/runtime/run_*.json`.

---

## Custo — ms de relógio por segundo simulado

Esta é a única base em que as etapas se somam (ver `sim/profiler.py`).

| configuração | RTF | physics | neural | vision | leitura | outro | **total** |
|---|---|---|---|---|---|---|---|
| flygym1+circuito | 0,0308 | 29.506 | 2.350 | 9 | 363 | 260 | **32.487** |
| flygym1+whole | 0,0301 | 28.937 | 3.793 | 16 | 211 | 239 | **33.196** |
| flygym2+circuito | 0,0534 | 15.937 | 2.204 | 11 | 342 | 238 | **18.733** |
| flygym2+whole | 0,0492 | 16.006 | 3.694 | 19 | 375 | 246 | **20.340** |

O que se lê daqui:

- **A física domina em todos os quatro**, entre 79% e 91% do relógio. Continua
  valendo depois de o conectoma inteiro ter ido pra GPU.
- **Trocar 1.261 neurônios por 164.451 custa ~1,5 s de relógio por segundo
  simulado** (2.350 → 3.793 ms no 1.x; 2.204 → 3.694 no 2.x). São 243× mais
  arestas por 1,6× mais tempo — é o que a GPU comprou.
- O custo neural é praticamente o mesmo nos dois simuladores, como tem que ser:
  é o mesmo backend rodando o mesmo conectoma. Serve de controle — se diferisse,
  alguma coisa estaria errada na medição.

## Comportamento e Giant Fiber

| configuração | fugas | GF spk | exc mV | inib mV | líquido mV | v mín mV | arestas |
|---|---|---|---|---|---|---|---|
| flygym1+circuito | 1 | 9 | 1.812,6 | −1.898,9 | −86,3 | −175,6 | 1.455 |
| flygym1+whole | 1 | 1 | 2.003,3 | −7.723,2 | −5.719,8 | −651,8 | 1.455 |
| flygym2+circuito | 2 | 8 | 1.981,2 | −2.056,5 | −75,3 | −178,6 | 1.455 |
| flygym2+whole | 0 | 1 | 1.849,8 | −12.372,6 | −10.522,8 | −576,2 | 1.455 |

As **1.455 arestas entrando no GF são as mesmas nas quatro corridas** — é o
mesmo conectoma. O que muda é quanta corrente chega por elas:

- no **circuito**, excitação e inibição quase se cancelam (líquido −86 e −75 mV)
  e o GF dispara 8–9 vezes em 1 s;
- no **whole CNS**, a inibição multiplica por 4–6 (−7.723 e −12.373 mV), o
  líquido despenca pra −5.720 / −10.523 mV e o GF dispara **uma vez só** —
  no transiente de partida, antes de a rede inibitória carregar. A análise
  causal desse único disparo está em
  [WHOLE_CNS_GIANT_FIBER.md](WHOLE_CNS_GIANT_FIBER.md#o-gate-nao-esta-ligado-em-t0-a-janela-de-partida).

**Nenhum parâmetro foi ajustado** pra produzir esses números: nem entrada, nem
limiar, nem peso, nem inibição, nem a lista de neurônios.

## Ganho observado × causa atribuída

**Observado:** ponta a ponta com whole CNS, o 2.x é **1,63× mais rápido**
(33.196 → 20.340 ms por segundo simulado). Na física isolada, **1,81×**
(28.937 → 16.006).

**Atribuído: não sabemos.** Pelo menos três causas mudaram juntas e a corrida
não as separa:

| o que mudou | 1.x | 2.x |
|---|---|---|
| pares de colisão | 2.268 | 55 |
| `nv` do modelo | 93 | 72 |
| chão | `MovingObjArena` | `FlatGroundWorld` |
| controlador | `flygym.examples.locomotion` | `HybridTurningController` do demo |
| MuJoCo | 3.2.7 | 3.9 |

Dizer "o wrapper do FlyGym 2.x é 1,6× mais rápido" seria errado: o **modelo
padrão** dele também é mais leve. Com 41× menos pares de colisão, a narrowphase
tem 41× menos trabalho antes de qualquer diferença de wrapper.

Separar as causas exigiria rodar o 2.x com o modelo do 1.x (mesmos pares, mesmo
`nv`, mesmo chão). Isso não foi feito, e até ser feito o número **1,63×** é uma
medição de ponta a ponta de duas configurações diferentes, não um speedup
atribuível ao FlyGym 2.

## O que foi igualado e o que não dá pra igualar

**Igual nas quatro:** semente (0), a classe `Estimulo` (literalmente a mesma —
`sim/physics/looming_world.py`, esfera de 3 mm, 30 → 4 mm, ciclo de 0,8 s),
duração, timestep, a realização de Poisson (`default_rng(0)`) e a semântica do
experimento.

**Não dá pra igualar**, e vai gravado em `diferencas_declaradas` de cada JSON:
os pares de colisão, o `nv`, o chão e a implementação do controlador. Estão na
tabela acima.

## O que continua em aberto

- O 2.x com o modelo do 1.x, que é o que separaria wrapper de modelo.
- 1 s por configuração é pouco pra contagem de fugas: a diferença entre 0, 1 e 2
  fugas está dentro do que um evento isolado explica. Para tratar fuga como
  variável de saída seria preciso repetir com várias sementes.
- A física continua sendo 80–90% do custo. Qualquer ganho grande daqui pra
  frente vem dali, não do neural.
