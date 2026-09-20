# Drosobot GPU Physics

Um backend de física acelerado por GPU para o NeuroMechFly, sem CUDA, medido
contra o MuJoCo CPU como referência.

Este documento é o estado real do trabalho: o que foi construído, o que foi
medido, e — a parte mais importante — **o que a medição refuta e o que ela
ainda permite**.

Estado em uma linha: **o passo de física completo — `state(t) → state(t+dt)` —
roda na GPU para o subconjunto plano × malha, sem o MuJoCo executar nenhuma
etapa dinâmica. Falta colisão cilindro × malha, e a latência ainda é 50× a da
CPU, dominada por enfileiramento no host.**

---

## A pergunta

Não é "a GPU consegue rodar MuJoCo?". É:

> Conseguimos executar a física necessária ao NeuroMechFly na GPU com fidelidade
> suficiente e **latência de um mundo** menor que a do MuJoCo CPU atual?

Um mundo, uma mosca, malha fechada, baixa latência. Não vazão com mil mundos.

Hardware do alvo primário: **AMD Radeon RX 6700 XT / Windows 11**, `gfx1031`,
20 CUs, OpenCL 2.0.

---

## A resposta curta

**Ainda não — mas a medição não diz que é impossível, e essa distinção é o
resultado mais importante desta rodada.**

O modelo tem `nv = 72`. Isso é minúsculo para uma GPU, e três medidas
independentes fecham o espaço de desenhos possíveis:

| medida | valor | o que refuta |
|---|---|---|
| despacho OpenCL síncrono | **84 us** | CPU no laço a cada passo, em qualquer API |
| Cholesky densa 72×72, um work-group | **40 us** | porte direto do solver Newton no primal |
| cinemática direta portada, fp32 | **9,3 us** contra **5,0–7,3 us** da CPU | que bastasse portar bem |

Mas o teto **não** proíbe. Um passo de física sobre este modelo tem ~470
estágios sequenciais — trechos que não podem começar antes de o anterior
terminar. O custo de um estágio nesta placa, medido em isolamento:

```
piso absoluto, trabalho zero          0,060 us   ->  470 estagios =  28 us
com trabalho moderado                 0,248 us   ->  470 estagios = 116 us
na taxa real da cinematica portada    0,77  us   ->  470 estagios = 362 us
MuJoCo CPU hoje                                                     117 us
```

O hardware comporta um passo inteiro em 28 us. Uma implementação com trabalho
moderado por estágio empata com o MuJoCo. **O que falta é implementação, não
placa** — e a distância exata é 3×, do que temos hoje até a paridade.

Uma parte desses 3× já foi recuperada nesta sessão, por diagnóstico e não por
tentativa: o kernel lia as constantes do modelo da memória global a cada nível,
com ~7 corpos ativos e nada para esconder a latência. Passando-as para
registrador, a cinemática em fp64 caiu de 26,4 para 20,6 us. Em fp32 a diferença
ficou dentro do ruído de corrida a corrida (~10 us nas duas variantes).

A cinemática está **correta** — erro máximo 6,7e-16 contra o `mjData` em fp64 —
e ainda assim é 1,3–2× mais lenta que a CPU. E o tempo dela é **plano de 16 a
256 threads**: não falta paralelismo. O que limita é a cadeia serial de 10
níveis da árvore de corpos, onde cada elo é uma operação escalar dependente da
anterior.

Ver [O que mudaria a resposta](#o-que-mudaria-a-resposta).

---

## O que foi medido primeiro: onde vai o tempo

### O `mj_step` não custava o que se pensava

O perfil anterior atribuía ~160 us ao `mj_step`. Eram 160 us do **wrapper**. Os
timers internos do MuJoCo saíam todos zerados porque dependem de um callback de
relógio (`mjcb_time`) que, por padrão, não existe — sem ele todo `TM_START` vira
no-op. Instalado o callback (`benchmarks/physics/profile_mj_step.py`):

```
looming, dt=1e-4, nv=72, ncon~8, nefc~32     116,9 us por mj_step
                                             205,5 us por ciclo do experimento

  solver                  39,1%     45,7 us
  project_constraint      12,4%     14,5 us
  colisao                  9,1%     10,6 us
  cinematica               8,4%      9,8 us
  make_constraint          7,3%      8,5 us
  resto do passo           6,6%      7,8 us
  inercia                  5,6%      6,5 us
  integracao               4,8%      5,6 us
  velocidade               3,9%      4,5 us
  posicao (outros)         1,6%      1,9 us
  atuacao                  1,3%      1,5 us
```

**O gargalo é o solver de restrição, não a colisão.** Isso inverte a intuição —
e inverte também a prioridade de qualquer backend novo.

### Um detalhe de método que mudou o número em 40%

Medir `mj_step` num laço apertado depois de aquecer mede **outra física**. Sem o
controlador reescrevendo `ctrl` a cada passo, os atuadores congelam, a mosca
desaba no chão, `ncon` sobe e o solver trabalha mais. Isso inflou a medida de
~95 us para 149 us sem nada ter mudado no modelo. Os números acima vêm de
`_CronometraStep`, que cronometra `sim.step()` **por dentro do laço real**, com
o controlador de marcha acionando.

### O custo que ninguém estava olhando

```
ciclo do experimento    190 us/passo
  dentro do mj_step     108 us
  Python em volta        82 us
```

**82 us por passo são Python em volta da física** — observação, controlador,
escrita nos atuadores. É quase tanto quanto a física inteira, e é o item mais
barato de atacar no orçamento de 100 us/passo que o tempo real exige com
dt = 1e-4. Nenhum backend de física, por melhor que seja, melhora esse número.

---

## Por que o upstream anuncia um número muito maior

Lido no clone local
(`research/upstream/flygym/scripts/dev/run_gpu_benchmark.py`):

```
n_worlds     16 -> 16384, dobrando
backend      flygym.warp.GPUSimulation  (NVIDIA Warp -> CUDA)
rendering    desligado
controle     angulos gravados, reproduzidos -- sem malha fechada
geometria    tambem com ALL_TO_CAPSULES: malhas viram capsulas

realtime_factor = sim_steps * n_worlds / walltime * sim_timestep
```

O `n_worlds` está **dentro** da métrica. Um fator de 100× com 16.384 mundos é
0,006× por mundo. Não há nada errado nisso — para treinar política o que importa
é vazão agregada. Mas não é a mesma grandeza que a nossa, e pôr os dois números
lado a lado compara vazão com latência.

Consequência prática: o benchmark oficial **não roda nesta máquina**. Ele chama
`check_gpu()` e `nvidia-smi` diretamente, e o `flygym.warp` gera CUDA em tempo
de execução. Não existe caminho HIP, Vulkan ou OpenCL no Warp upstream — isso
está verificado, não suposto, em `benchmarks/physics/upstream_vs_drosobot.py`,
que tenta o import e registra o erro.

O que dá para comparar honestamente no mesmo PC é um mundo só:

```
                 fisica pura   ciclo do experimento   vertical completo
looming            108 us            190 us            rtf 0,19-0,23
obstaculos         104 us            180 us            (runs/*/summary.json)
```

---

## O piso de latência da GPU

Medido antes de escrever uma linha de solver
(`benchmarks/physics/gpu/piso_de_latencia.py`):

```
despacho sincrono (1 kernel nulo, enfileira e espera)      84,3 us
despacho em lote, 10 na fila                                8,9 us por kernel
despacho em lote, 100 na fila                               2,8 us por kernel
despacho em lote, 1000 na fila                              1,6 us por kernel
iteracao dentro de kernel persistente (2 barreiras)         0,076 us
varredura de suporte, 1024 vertices, 1 work-group           0,66 us
banda de um work-group                                     24,9 GB/s
```

Três leituras, e cada uma é uma decisão de arquitetura:

1. **84 us por despacho síncrono.** O orçamento inteiro de tempo real é 100 us.
   Qualquer desenho com a CPU esperando a GPU a cada passo está morto — em
   OpenCL, em D3D12 e em Vulkan. Isso não é escolha de API.
2. **1,6 a 8,9 us por kernel enfileirado.** Um passo com 15 kernels custaria
   24 a 130 us só em despacho, antes de calcular qualquer coisa.
3. **0,076 us por iteração dentro do kernel.** Mil vezes mais barato. É a única
   forma viável: **o laço de física mora dentro do kernel**, não em volta dele.

Daí a arquitetura: estado residente no device, laço por dentro do kernel, um
work-group. Usar 1 CU de 20 não é desperdício por descuido — o que decide este
problema é latência, e mais work-groups exigiriam sincronização global, que
custa um despacho.

### O solver, medido antes de ser escrito

O Newton do MuJoCo fatora o Hessiano **uma vez por passo** e depois usa
atualizações Cholesky de posto 1 (`HessianIncremental`, lido em
`src/engine/engine_solver.c`). Então o custo da GPU depende de uma pergunta só:
quanto custa uma fatoração num work-group?

```
Cholesky densa, um work-group, melhor configuracao medida
  N = 24      7,7 us
  N = 72     40,3 us
```

Uma fatoração de 72×72 são 62 mil flops. Um CU da RX 6700 XT faz ~150 GFLOP/s
em FP32 — ou seja, a conta deveria levar 0,4 us. Leva 40. O fator 100 é latência
pura: 72 etapas de eliminação estritamente sequenciais, cada uma com barreiras e
cadeias de dependência em LDS. O solver inteiro do MuJoCo, com as 8 iterações de
Newton que este modelo usa, custa 45,7 us na CPU. Uma única fatoração custa
40 us na GPU.

---

## O que foi construído

```
sim/gpu_physics/
    inventario.py     o que o modelo do NeuroMechFly usa do MuJoCo
    compilador.py     mjModel -> modelo plano + physics_model_hash + validacao
    estrutura.py      a arvore de corpos reorganizada em niveis paralelos
    device.py         device, buffers, dispatch. Nada de fisica
    cinematica.py     host da cinematica residente
    kernels/
        cinematica.cl porte de mj_kinematics1/2, um nivel por vez
```

### O subconjunto MJCF suportado

Descoberto inspecionando o modelo real, não escolhido no papel
(`benchmarks/physics/mjmodel_inventario.json`):

| | |
|---|---|
| dimensões | `nq` 73, `nv` 72, `nu` 48, `nbody` 70, `njnt` 67, `ngeom` 73 |
| juntas | `free` × 1, `hinge` × 66 — mais nada |
| geoms | `plane`, `sphere`, `cylinder`, `mesh` |
| colisão | `plane`×`mesh` (55 pares), `cylinder`×`mesh` (55 no campo de obstáculos) |
| `condim` | 3 — atrito deslizante, sem torcional nem rolante |
| atuadores | 42 `joint` com bias afim (servo de posição), 6 `body` (adesão) |
| integrador | Euler |
| solver | Newton, cone piramidal, tol 1e-8 |
| **ausentes** | tendões, restrições de igualdade, limites de junta, atuadores com estado, atrito seco, forças de fluido, flex, hfield, SDF |

`compilador.valida()` **recusa** qualquer modelo fora disso, nomeando o recurso.
Um solver que ignora em silêncio um tendão que existe no modelo produz número
errado com cara de número certo.

### O achado dos pilares, virado em contrato

O FlyGym 2.x / NeuroMechFly **não depende de `contype`/`conaffinity`** para os
contatos que este modelo usa. Todos os 69 geoms da mosca são `(0, 0)`; a
topologia de contato vem de `<pair>` explícitos — 55 com o chão, mais 55 com o
pilar quando a arena de obstáculos os declara.

Um backend GPU que inferisse a topologia de colisão pelas máscaras teria **zero
contatos** e a mosca atravessaria o pilar. Foi exatamente o que aconteceu antes
da correção: 3 s de corrida, retina vendo o obstáculo o tempo todo, nenhum
contato.

Isso está em dois lugares, de propósito:

* `compilador.valida()` registra `colisao_por_pares_explicitos: True` e a nota
  no metadata de toda corrida;
* `tests/test_gpu_physics.py::test_pilar_barra_a_mosca` mede **contato de
  verdade** — 3 s de mosca andando contra o pilar central, e reprova se der
  zero. Contar `npair` não serviria: antes da correção os pares existiam no XML
  e os contatos não aconteciam.

Medido hoje: 55 pares com o pilar, **7.179 contatos em 3 s**.

### `physics_model_hash`

O par de `hash_ciencia()`, para a outra metade do problema.

* `hash_ciencia` cobre o **cérebro**: constantes de Shiu et al., transdução
  retina → taxa, timestep neural, escala de ponto fixo.
* `physics_model_hash` cobre o **corpo**: massas, inércias, juntas, damping,
  atrito, pares de contato, parâmetros de geom, solver, integrador, timestep.

É derivado do `mjModel` **compilado**, não de uma lista escrita à mão. Trocar
uma malha, uma massa ou um `<pair>` muda o hash sem ninguém lembrar de atualizar
nada. Trocar a cor de um geom não muda — e isso é testado nas duas direções.

É o que permite dizer, comparando MuJoCo CPU com Drosobot GPU, que os dois
rodaram **o mesmo corpo**.

### O backend entrou na receita

```
flygym1           FlyGym 1.2.1 / mujoco 3.2.7    referencia historica
flygym2-mujoco    FlyGym 2.1.0 / mujoco 3.9      referencia atual, baseline
drosobot-gpu      Drosobot GPU Physics           o motor proprio
```

O campo `physics` da `Receita` agora nomeia o **backend físico**, não o
adaptador. `flygym2` continua aceito e vira `flygym2-mujoco`, porque corridas
gravadas antes desta mudança não podem deixar de ser legíveis.

Cada `metadata.json` ganhou um bloco `backend_fisico` com backend, família,
adaptador, versão, timestep, pares de colisão, solver, integrador,
`physics_model_hash` e o subconjunto validado — e, quando houver, device e
precisão.

**`drosobot-gpu` recusa montar.** `physics.cria("drosobot-gpu")` levanta
`BackendIncompleto` nomeando o que roda na GPU (cinemática) e o que ainda é
MuJoCo (inércia, colisão, restrições, solver, atuação, integração). Gravar
`physics: drosobot-gpu` no metadata com o MuJoCo integrando por trás seria uma
procedência falsa — exatamente o tipo de dependência mascarada que este projeto
não pode ter.

### Comparação entre backends

`lab.compara_backends(pasta_a, pasta_b)` põe duas corridas da mesma receita, com
motores diferentes, lado a lado — e reporta **a primeira janela em que cada
camada se separa**, na ordem causal:

```
FISICA      posicao do torax
SENSORIAL   taxa de entrada, spikes sensoriais
NEURAL      excitacao, inibicao e spikes do Giant Fiber
MOTOR       drive descendente
DESFECHO    fugas
```

Comparar só a posição final não distingue "divergiu no primeiro contato e se
reencontrou por acaso" de "andou junto 590 ms e se separou no fim". A primeira
camada a divergir é a que aponta a causa; as seguintes herdam.

Exige `hash_ciencia` igual **e** `physics_model_hash` igual. O backend é a única
variável — é o que está sendo medido.

**O que ela ainda não vê:** `qpos`, `qvel`, forças de restrição e contatos não
estão no `timeseries.csv`, cujas colunas são fixas de propósito. A camada FÍSICA
é a trajetória do tórax, que é integral do resto: detecta divergência, mas não a
localiza dentro do passo. Um arquivo próprio para o estado físico por passo é o
que falta.

---

## A cinemática direta, em detalhe

Porte de `mj_kinematics1`/`mj_kinematics2` (MuJoCo, Apache-2.0). A ordem das
operações de ponto flutuante segue o original termo a termo — `mju_mulQuat`,
`mju_rotVecQuat`, `mju_quat2Mat`, com os mesmos atalhos para quaternião
identidade e vetor nulo. Mudar a ordem daria o mesmo resultado matemático e um
resultado numérico diferente, e aí a comparação mediria o porte em vez da
física.

O laço do MuJoCo é serial sobre os corpos (`parentid < id` garante a ordem).
Aqui é serial sobre **níveis** e paralelo dentro de cada nível:

```
nbody = 70      profundidade = 10      corpos por nivel: 1 2 16 10 9 7 7 6 6 6
```

### Corretude

| | erro máximo contra `mjData` |
|---|---|
| fp64 | **6,7e-16** (`ximat`) |
| fp32 | **3,6e-07** (`xipos`) |

Nove campos conferidos: `xpos`, `xquat`, `xmat`, `xipos`, `ximat`, `xanchor`,
`xaxis`, `geom_xpos`, `geom_xmat`. O fp64 fica na ordem do épsilon da máquina —
o que sobra é arredondamento de ULP, com `sin`/`cos` da GPU respondendo pela
maior parte.

### Latência, e o que ela diz

Arena de looming, melhor work-group de cada variante:

```
                              fp64      fp32
modelo em memoria global      26,4       9,8   us
constantes em registrador     20,6      10,2   us

MuJoCo CPU (mj_kinematics)               5,0   us
```

**O tempo é plano de 16 a 256 threads.** Dezesseis vezes mais threads, o mesmo
tempo. O kernel não está limitado por paralelismo, nem por barreiras (12
barreiras × 0,035 us = 0,4 us do total), nem por banda. Está limitado pela
cadeia serial de 10 níveis de operações escalares dependentes.

Três otimizações foram tentadas e medidas, não supostas:

* **estado dos corpos em `__local`** em vez de memória global entre níveis:
  32 → 27 us em fp64, sem efeito em fp32. Quase nada — o estado não era o
  problema.
* **constantes do modelo em registrador**, uma thread adotando um corpo:
  26,4 → 20,6 us em fp64 (−24%); em fp32 as duas variantes ficam dentro do
  ruído de corrida a corrida. A leitura: fp64 é mais sensível porque cada
  leitura é duas vezes mais larga, e com ~7 corpos ativos por nível não há
  trabalho para escondê-la.
* **varredura de tamanho de work-group** de 16 a 256: acima.

O que sobra, em número: **0,77–0,85 us por estágio sequencial**, contra 0,248 us
de um estágio sintético com trabalho comparável. Esse fator ~3 é a distância que
separa o que está escrito do que a placa comporta — e é exatamente o fator que
decide o projeto inteiro, porque ele multiplica os ~470 estágios do passo.

### Precisão

Não houve conversão silenciosa para fp32. O programa é compilado duas vezes,
`-D USA_FP64` e sem, e o metadata registra qual está em uso. O custo da diferença
é real e está medido: fp64 custa ~2,2× o fp32 nesta placa (não 16×, porque o
kernel não é limitado por ALU).

Para a cinemática isolada, fp32 a 3,6e-07 seria suficiente — o erro é menor que
a espessura de um tarso. **Mas isso não se estende ao passo inteiro sem
medição**: um erro de 1e-7 por passo realimentado 10.000 vezes por segundo
simulado, num sistema com contato, pode divergir. A decisão de precisão por
estágio só pode ser tomada com a trajetória inteira comparada, e isso exige o
solver, que não existe.

---

## O que roda onde, hoje

Sem eufemismo.

### Roda no Drosobot GPU Physics, validado contra o `mjData`

| etapa | campos conferidos | pior erro relativo |
|---|---|---|
| cinemática direta | `xpos`, `xquat`, `xmat`, `xipos`, `ximat`, `xanchor`, `xaxis`, `geom_xpos`, `geom_xmat` | 6,7e-16 |
| centro de massa e inércias | `subtree_com`, `cinert`, `cdof` | ~1e-16 |
| matriz de massa e fatoração | `crb`, `M`, `qLD`, `qLDiagInv` | 2,7e-15 |
| velocidades de corpo | `cvel`, `cdof_dot` | 3,3e-16 |
| forças passivas | `qfrc_passive` | 1,5e-16 |
| bias (RNE) | `qfrc_bias` | 5,9e-18 |
| atuação de junta | `actuator_force`, `qfrc_actuator` | exato |
| **adesão** (`mjTRN_BODY`) | `actuator_force`, `qfrc_actuator`, com `ctrl = 1` | 1,2e-16 |
| aceleração sem restrição | `qacc_smooth` | 3,2e-15 |
| colisão plano × malha | `ncon`, ordem, `dist`, `pos` | 4,5e-16 |
| **construção das restrições** | `efc_J`, `efc_pos`, `efc_margin`, `efc_id`, ordem | 4,6e-16 |
| | `efc_diagApprox`, `efc_R`, `efc_D` | **exatos** |
| | `efc_vel`, `efc_aref` | 1,0e-13 |
| **solver de restrição** | `efc_force`, `qfrc_constraint`, `qacc` | 2,0e-14 |
| integração Euler | `qvel`, `qpos` | — |
| **passo completo** | dois passos consecutivos contra `mj_step` | dqpos 2,2e-16, dqvel 1,9e-13 |

### Continua no MuJoCo CPU

* colisão **cilindro × malha** (GJK/EPA), usada só na arena de obstáculos — e o
  compilador **recusa** o modelo em vez de devolver zero contatos com o pilar
* parsing, compilação do `mjModel`, pré-processamento estático, e a referência
  de comparação
* **todos os experimentos gravados**: `looming`, `obstáculos` e `optomotor`
  rodaram e continuam rodando 100% em `flygym2-mujoco`

`physics.cria('drosobot-gpu')` continua recusando montar: falta o adaptador que
liga o motor ao laço do laboratório (retina, arena, pose dos segmentos), e sem
cilindro × malha a arena de obstáculos não roda.

### O solver, e por que ele não "bate com o MuJoCo"

O objetivo é o do 3.9.0, termo a termo:

```
custo(a) = 1/2 (a - a_s)' M (a - a_s) + sum_i 1/2 D_i min(jar_i, 0)^2
jar      = J a - aref
```

estritamente convexo e C¹, com mínimo único. O caminho até ele difere de
propósito: o MuJoCo fatora o Hessiano uma vez e faz atualizações de posto 1 com
busca de linha exata; aqui é refatoração densa com recuo de Armijo, num
work-group, com `H` (72×72) em `__local`.

Medido, no estado inicial, com as entradas dos dois batendo a 1e-13:

```
            custo           |grad|
GPU         1,394463e+09    1,3e-13
MuJoCo      1,394733e+09    5,0e+01
```

**O ponto da GPU é estacionário e tem custo menor.** O MuJoCo pára antes do
mínimo nesse passo — com `solref[0] = 2e-4` e `dt = 1e-4` a restrição é muito
rígida, e a busca de linha dele estagna.

Ao longo de **1000 passos**, o ponto da GPU permanece estacionário em todos os
marcos conferidos — `|grad|` entre 7e-14 e 1,3e-13, sem degradação, inclusive
com warm start. A divergência de trajetória cresce devagar e `ncon` acompanha:

```
passo    dqpos      dqvel     ncon g/mj   |grad| GPU
    2   5,4e-05   3,5e-01    12/12       9,4e-14
   10   6,8e-04   1,0e+00    12/12       1,1e-13
  100   1,1e-02   1,1e+00    12/12       9,1e-14
 1000   5,6e-02   3,9e-01    14/12       7,6e-14
```

Consequência prática: as trajetórias coincidem a 1e-13 enquanto os dois solvers
concordam e separam quando não concordam mais. Com adesão desligada isso são
dois passos a 1e-16/1e-13; com adesão ligada, a separação começa no primeiro
passo. **Isso não é erro do porte**, e o teste reflete isso: ele exige
estacionariedade (`|grad| ≈ 0`), não igualdade com o MuJoCo — exigir igualdade
faria da parada antecipada dele a especificação.

### Latência do passo completo

Agora faz sentido medir, porque o passo existe:

```
Drosobot GPU passo completo     6345 us
  host para enfileirar          4390 us      80 despachos, ~53 us cada
  alem do enfileiramento        1956 us
MuJoCo CPU mj_step               104 us
```

**70% do custo é Python enfileirando kernels**, não a GPU calculando. Os 1956 us
de GPU ainda são ~19× a CPU, e a maior parte está nos estágios com laço por
nível — `com_pos`, `massa`, `com_vel`, `bias` somam ~56 dos 80 despachos, cada um
com poucas dezenas de threads ativas.

Isso é exatamente o que a fase de fusão/kernel persistente ataca, e foi deixado
para depois de propósito: primeiro uma física completa e correta, depois uma
implementação de baixa latência.

### Trace físico

`sim/lab/trace_fisico.py` grava `physics_trace.jsonl` — `qpos`, `qvel`, `qacc`,
contatos e forças de restrição por passo, com passo configurável e **desligado
por padrão**. Arquivo próprio, não colunas novas no `timeseries.csv`: são
centenas de números por passo de física contra poucos por janela neural, e as
colunas do timeseries são fixas de propósito.

`primeira_divergencia()` compara dois traces e diz o primeiro passo e o primeiro
campo que se separam, na ordem causal.

### Dois bugs que só a trajetória pegou

Nenhum dos dois aparece num teste de campo isolado, porque os dois dependem de
estado residente entre passos:

* **`crb_monta_M` acumulava em `M` sem zerar.** O passo 1 batia porque o buffer
  nasce zerado; o passo 2 somava sobre o passo 1 e a trajetória ia a 1e71.
* **`factor_M` recebia o mesmo buffer como entrada `const` e como saída.**
  Aliasing é comportamento indefinido, e o compilador tem licença para supor que
  não há.

É por isso que o teste de passo roda **dois** passos, não um.

---

## O que mudaria a resposta

A conclusão é sobre **este modelo, nesta placa, com um mundo**. Três coisas a
mudariam, e vale dizer quais para que a medição não seja lida como mais geral do
que é:

0. **Fechar o fator 3 até o piso.** É a que está mais perto e a única sobre a
   qual há número: 0,77 us por estágio hoje, 0,248 us no sintético com trabalho
   comparável. Recuperar isso põe o passo projetado em ~116 us, empatando com o
   MuJoCo CPU — antes de qualquer outra mudança. O caminho que a medição aponta
   é reduzir o que cada estágio espera: menos leituras dependentes, mais
   trabalho por thread ativa, e menos estágios (fundir níveis onde a árvore
   permite).
1. **Um modelo maior.** `nv` na casa dos milhares muda tudo: a cadeia serial da
   árvore cresce como o logaritmo da profundidade, o trabalho paralelo cresce
   linearmente. É o regime em que o MJX e o MuJoCo Warp ganham.
2. **Muitos mundos.** É o regime do benchmark upstream, e é vazão. Explicitamente
   não é o objetivo deste projeto.
3. **Fusão com o motor neural.** Esta é a única que continua viável aqui, e é a
   maior: o vertical completo gasta **82 us/passo em Python em volta da física**,
   quase tanto quanto a física inteira. Um laço que não volte para o Python a
   cada passo vale mais que qualquer ganho disponível dentro do `mj_step`.

Sobre o solver, se alguém retomar: o caminho que a medição **não** refuta é
resolver no **dual** (`nefc ≈ 24`) em vez de no primal (`nv = 72`) — uma
fatoração de 24×24 custa 7,7 us contra 40 us. A física preservada não é o
algoritmo, é a **solução**: o problema é um QP convexo com certificado de
convergência a 1e-8, e qualquer método que alcance a mesma tolerância dá a mesma
resposta. Isso não é mudança científica. Mas é uma aposta que precisa de medição
própria, e não foi feita.

---

## Licença e procedência

MuJoCo e MuJoCo Warp são Apache-2.0 — confirmado nos clones locais
(`research/upstream/mujoco/LICENSE`, `research/upstream/mujoco_warp/`).

`sim/gpu_physics/kernels/cinematica.cl` é porte direto de
`src/engine/engine_core_smooth.c` e `src/engine/engine_util_spatial.c`. A origem
está no cabeçalho do arquivo e em `THIRD_PARTY_NOTICES.md`. A ordem das
operações foi preservada deliberadamente, o que torna o parentesco com o
original explícito e não acidental.

Nada do runtime do NVIDIA Warp foi portado. O MuJoCo Warp foi lido como
referência de engenharia — decomposição dos algoritmos, layout orientado a
dados, estrutura do solver — não copiado: os kernels dele são escritos no DSL do
Warp e o CUDA é gerado em tempo de execução, então não existem `.cu` para
traduzir.

---

## Como executar

```bat
REM trajetoria GPU x MuJoCo, trace fisico e latencia do passo completo
.venv-flygym2\Scripts\python benchmarks\physics\gpu\trajetoria_gpu_vs_mujoco.py

REM quantos estagios sequenciais cabem no orcamento
.venv-flygym2\Scripts\python benchmarks\physics\gpu\orcamento_de_estagios.py

REM o que o modelo usa do MuJoCo
.venv-flygym2\Scripts\python -m sim.gpu_physics.inventario

REM onde vai o tempo do passo de fisica
.venv-flygym2\Scripts\python benchmarks\physics\profile_mj_step.py

REM o piso de latencia desta placa
.venv-flygym2\Scripts\python benchmarks\physics\gpu\piso_de_latencia.py

REM cinematica: GPU contra CPU, corretude e latencia
.venv-flygym2\Scripts\python benchmarks\physics\gpu\fk_gpu_vs_cpu.py

REM por que o numero do upstream e outro
.venv-flygym2\Scripts\python benchmarks\physics\upstream_vs_drosobot.py

REM as cinco propriedades que o backend nao pode quebrar
.venv-flygym2\Scripts\python tests\test_gpu_physics.py
```

Requisitos: `.venv-flygym2` (Python 3.14, mujoco 3.9, flygym 2.1.0) e `pyopencl`
com um device OpenCL. Sem GPU, o teste de cinemática se declara pulado em vez de
falhar; os demais não precisam de GPU.

---

## Portabilidade

Os kernels são OpenCL C99 e não usam nada específico de `gfx1031`. `Device`
descobre o dispositivo, reporta capacidades (`cl_khr_fp64`, `max_work_group_size`,
`local_mem_size`) e recusa fp64 com mensagem quando o device não o expõe. Não há
`#ifdef` de vendor nem tamanho de wavefront assumido.

O que **não** é portátil de graça: se um dia o alvo for Vulkan ou D3D12, os
kernels precisam ser traduzidos (via SPIR-V, por clspv ou reescrita em HLSL).
A fronteira de `device.py` esconde o host, não a linguagem dos kernels — e é
honesto dizer isso em vez de prometer uma troca de arquivo.

A escolha do OpenCL não foi por conveniência de host. A medida de 84 us por
despacho síncrono descarta o desenho CPU-no-laço em qualquer API, e o desenho
que sobra — estado residente, laço dentro do kernel — não paga a latência de
despacho em que D3D12 seria melhor. Dado isso, o critério decisivo passou a ser
outro: **o motor neural deste projeto já é OpenCL e já é residente na GPU.**
Física no mesmo contexto torna o handoff motor → corpo → sensor → motor uma
troca de ponteiro entre buffers do mesmo device, sem interop e sem passar pela
CPU. Física em D3D12 com cérebro em OpenCL exigiria uma camada de interop que
não existe.
