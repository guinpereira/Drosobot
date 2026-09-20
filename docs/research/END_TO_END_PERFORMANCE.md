# Desempenho ponta a ponta: 0,055× → 0,204×

Três rodadas de otimização do laço completo (FlyGym 2.x + Male CNS inteiro na
GPU + telemetria), **sem tocar em ciência**: mesmo timestep, mesmo modelo,
mesmo controlador, mesmo conectoma, mesmos parâmetros de Shiu et al.

RX 6700 XT (gfx1031, 20 CUs), Windows 11, looming, `legs`, whole CNS.

| | baseline | fastpath | esta rodada |
|---|---|---|---|
| **RTF** | 0,055× | 0,113× | **0,204×** |
| física | 14.405 (78,9%) | 5.167 (58,3%) | **3.328 (67,8%)** |
| neural | 3.363 (18,4%) | 3.068 (34,6%) | **916 (18,7%)** |
| visão | 14 | 14 | 15 |
| leitura | 220 | 174 | 179 |
| telemetria | 191 | 191 | 208 |
| **parede** | 18.266 | 8.860 | **4.907** |

Unidade: ms de relógio por segundo simulado. **3,7× no total.**

O comportamento é **idêntico** nas três: mesma semente, mesmas fugas, mesmos
spikes do Giant Fiber, mesma excitação, inibição, líquido, `v` mínimo e posição
final. Verificado contra os JSON gravados antes de cada rodada.

---

## O método, nas três rodadas

Sempre o mesmo: **medir antes de mexer**, e em cada caso a medição contrariou a
suposição em vigor.

| suposição | medição | o que era de fato |
|---|---|---|
| "a física é 79%, logo o alvo é colisão" | só 11% estava dentro do `sim.step()` | bookkeeping Python refeito 10.000×/s |
| "o neural é compute na GPU" | scatter tinha piso de 19 µs e 859 µs de atomics | desbalanceamento: 1 thread por neurônio, hub de 11.203 arestas |
| "o custo do controlador é o scipy" | trocar o scipy economizou 24 µs de 228 | despacho do NumPy em arrays de 6 elementos |

---

## Rodada 3 — o que entrou

### 3.1 Scatter sináptico por frontier — 859 → 171 µs/passo

O kernel dava **uma thread por neurônio**: com 1.500 disparos em 164.451
neurônios, 99,1% das threads saíam na primeira linha e o trabalho ficava
concentrado nos hubs — um neurônio de 11.203 arestas percorrido por **uma**
thread enquanto 20 CUs esperavam.

Agora são dois kernels: `compacta_spikes` junta quem disparou numa lista curta,
e `scatter_frontier` distribui as arestas dessa lista entre as threads, 32 por
neurônio.

**Por que é exato:** a soma é `atomic_add` em **int**, e adição inteira é
associativa e comutativa — a ordem das parcelas não muda o total. (Seria
diferente com float; é mais uma razão para o acumulador ser ponto fixo.)

O contador da frontier é zerado dentro do kernel do LIF, pelo item 0 — um
`enqueue_fill_buffer` só para isso custaria mais um lançamento por passo
(~19 µs medidos).

### 3.2 Argumentos de kernel fixados — LIF 247 → 39 µs por chamada

`pyopencl` remarshala a lista inteira a cada chamada. O LIF tem 18 argumentos,
e 15 deles nunca mudam durante a corrida. Fixando-os uma vez e trocando só os
dois escalares por passo: **6,4×** na chamada.

Precisa ser refeito a cada troca de buffer (`reset`, caminho esparso), senão o
kernel continua apontando para o buffer velho — em silêncio.

### 3.3 Máscara de spike forçado esparsa — 164 KB → 311 bytes

A máscara tem um byte por neurônio, mas só os 311 sensores mudam de valor. Pior:
`enqueue_copy` do pyopencl é **bloqueante por padrão** — o host parava esperando
uma transferência que a GPU fazia em 75 µs.

Agora sobem 311 bytes, sem bloquear, e um kernel de 311 threads escreve nos
lugares certos. O conteúdo final da máscara é idêntico: os outros 164.140 bytes
já eram zero e continuam zero.

Ganho isolado modesto (1,14×) porque a fila estava saturada por outra coisa —
mas necessário para que os ganhos seguintes aparecessem.

### 3.4 Passo do controlador compilado — 204 → 32 µs/passo

Numba (`sim/physics/native_fastpath.py`). O ganho **não vem de calcular
menos**: vem de não pagar despacho do NumPy em arrays de seis elementos, ~30
vezes por passo.

O ponto delicado era a avaliação do polinômio por partes. `PPoly` expõe `x` e
`c` publicamente, mas **não** a ordem em que soma os termos — e a ordem decide o
último bit. Três formas foram testadas contra 2.000 sorteios de fase:

```
Horner              0/2000 idênticos, pior erro 4,4e-16
forma potência      0/2000 idênticos, pior erro 4,4e-16
potência corrente   2000/2000 idênticos, pior erro 0,0
```

A que bate é somar do termo de **menor** grau para o maior, com a potência
acumulada por multiplicação sucessiva. Se nenhuma tivesse batido, a
substituição seria rejeitada — não há ganho que justifique perder determinismo.

`fastmath=False` é obrigatório: ele autorizaria reassociação, que é exatamente o
que quebraria a igualdade.

Numba é **opcional**. Sem ele, `fastpath.py` volta ao scipy e o laboratório roda
mais devagar, não quebra.

---

## O que foi medido e rejeitado

**Substituir a avaliação do PPoly por Horner.** Matematicamente igual,
numericamente diferente (4,4e-16). Rejeitado antes de ser escrito.

**Poda de pares de colisão.** Não reconsiderada. `legs` continua o padrão;
`tarsi` reprovou em curva (ver [`COLLISION_PAIR_AUDIT.md`](COLLISION_PAIR_AUDIT.md)).

**Geometria de colisão simplificada.** O NeuroMechFly já usa primitivas para
contato e a narrowphase é 5% do `mj_step`. Não há gordura.

**Vulkan.** Fora de escopo desta rodada por decisão registrada em
[`VULKAN_STRATEGY.md`](VULKAN_STRATEGY.md). E a evidência desta rodada reforça:
o gargalo do neural era nosso padrão de dispatch, não a API.

**Batching de vários passos neurais num lançamento.** Não implementado: a
janela de 10 ms já enfileira 20 passos sem sincronizar entre eles, e a
sincronização só acontece no fim. Não havia round-trip para remover.

---

## Onde o custo está agora

Laço de física (4.052 ms/s):

| etapa | µs/passo | % |
|---|---|---|
| `mj_step` | 160 | 39,5% |
| retina | 106 | 26,0% |
| observação | 69 | 17,1% |
| **controlador** | **32** | 7,9% |
| estímulo | 16 | 3,9% |
| aplica ação | 14 | 3,4% |

Passo neural (407 µs):

| etapa | µs/passo |
|---|---|
| scatter (frontier) | 171 |
| forçados (cópia + kernel) | 106 |
| LIF | 102 |
| reduções | ~0 |

**O `mj_step` é agora o maior item isolado do laço de física** — 39,5%, contra
11% no começo. Não porque ficou mais lento (160 µs contra 159 no baseline), mas
porque tudo em volta encolheu 3,6×. O que sobra é física de verdade.

## O teto, e por que parar aqui

Para dobrar de novo seria preciso atacar `mj_step` (MuJoCo, 160 µs) e o render
da retina (106 µs). Os dois são trabalho real do modelo:

- **`mj_step`** só encolhe mudando o modelo físico — pares de colisão, `nv`,
  geometria. Fora de escopo, e com risco científico que a medição não justifica.
- **a retina** renderiza dois olhos a 100 Hz. A resolução é do modelo.

O que ainda é overhead removível: ~106 µs/passo de forçados no neural (duas
enfileiradas para 311 bytes) e ~69 µs de observação. Juntos valem talvez mais
15% — abaixo do que já foi colhido, e com complexidade crescente.

**RTF 0,204× é o ponto onde o custo passa a ser dominado por coisas que só
encolhem mudando a ciência.**

---

## Como reproduzir

```bat
.venv-flygym2\Scripts\python benchmarks\physics\profile_flygym2_step.py
.venv-flygym2\Scripts\python benchmarks\neural\profile_neural_step.py
.venv-flygym2\Scripts\python sim\drosobot_lab.py --physics flygym2 ^
    --neural opencl --cns whole --experiment looming --telemetry ^
    --sem-controle --duracao 1.5
```

Equivalência e regressão:

```bat
.venv-flygym2\Scripts\python tests\test_fastpath_equivalencia.py
.venv-flygym2\Scripts\python tests\test_neural_backend.py
.venv-flygym2\Scripts\python tests\test_drosobot_lab.py
.venv-flygym2\Scripts\python sim\compare_runtimes.py --physics flygym2 --duracao 1.0
```
