# Drosobot GPU Physics

Um backend de física acelerado por GPU para o NeuroMechFly, sem CUDA, medido
contra o MuJoCo CPU como referência.

Este documento é o estado real do trabalho: o que foi construído, o que foi
medido, e — a parte mais importante — **o que a medição refuta**.

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

**Para este modelo, nesta placa, não.** E a razão é estrutural, não de
implementação.

O modelo tem `nv = 72`. Isso é minúsculo para uma GPU. Três medidas
independentes, cada uma refutando uma esperança diferente:

| medida | valor | o que refuta |
|---|---|---|
| despacho OpenCL síncrono | **84 us** | CPU no laço a cada passo, em qualquer API |
| Cholesky densa 72×72, um work-group | **40 us** | porte direto do solver Newton |
| cinemática direta portada, fp32 | **12,3 us** contra **6,3 us** da CPU | que o problema fosse só o solver |

A cinemática está **correta** — erro máximo 6,7e-16 contra o `mjData` em fp64 —
e ainda assim é 2× mais lenta que a CPU em fp32 e 4× em fp64. E o tempo dela é
**plano de 16 a 256 threads**: não falta paralelismo. O que limita é a cadeia
serial de 10 níveis da árvore de corpos, onde cada elo é uma operação escalar
dependente da anterior. Um núcleo de CPU com execução fora de ordem e cache L1
resolve essa cadeia melhor que um CU de GPU, e nenhuma quantidade de threads a
encurta.

O que isso **não** significa: que o trabalho foi perdido, ou que a resposta seja
a mesma para outro modelo. Ver [O que mudaria a resposta](#o-que-mudaria-a-resposta).

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

```
                       fp64      fp32
work-group 16         32,9      15,3   us
work-group 32         29,9      14,0
work-group 64         27,2      13,2
work-group 128        27,0      12,3
work-group 256        27,3      12,8

MuJoCo CPU (mj_kinematics)        6,3   us
```

**Plano.** Dezesseis vezes mais threads, o mesmo tempo. Esse é o diagnóstico
inteiro: o kernel não está limitado por paralelismo, nem por barreiras (12
barreiras × 0,035 us = 0,4 us dos 12,3), nem por banda. Está limitado pela
cadeia serial de 10 níveis de operações escalares dependentes.

Duas otimizações foram tentadas e medidas, não supostas:

* **estado dos corpos em `__local`** em vez de memória global entre níveis:
  32 → 27 us em fp64, 12,9 → 13,0 em fp32. Praticamente nada. Não era memória.
* **varredura de tamanho de work-group** de 16 a 256: acima.

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

Sem eufemismo:

### Roda no Drosobot GPU Physics

* cinemática direta completa: quadros dos corpos, quadros inerciais, quadros dos
  geoms, âncoras e eixos das juntas — validada contra o `mjData`

### Continua no MuJoCo CPU

* inércia composta e matriz de massa
* forças de Coriolis/gravidade (RNE)
* colisão (broadphase e narrowphase)
* construção e projeção das restrições
* **o solver de restrição** — 39% do passo
* atuação
* integração
* **todos os experimentos gravados**: `looming`, `obstáculos` e `optomotor`
  rodaram e continuam rodando 100% em `flygym2-mujoco`

Nenhuma corrida da plataforma experimental usou GPU Physics. Não há como usar:
o backend recusa montar enquanto for parcial.

---

## O que mudaria a resposta

A conclusão é sobre **este modelo, nesta placa, com um mundo**. Três coisas a
mudariam, e vale dizer quais para que a medição não seja lida como mais geral do
que é:

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
