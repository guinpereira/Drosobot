# MuJoCo Warp: como acelera, e o que disso sobrevive sem CUDA

Lido em `mujoco_warp@87e742d` (v3.13.0-9) e `warp@015e5a17`. Ver
`research/UPSTREAM_LOCK.md`.

A pergunta que este documento tem que responder e a que foi feita:

> "Se nao pudessemos usar CUDA/Warp, quais ideias algoritmicas ainda seriam
> portaveis para Vulkan, DirectX ou outro backend?"

---

## 0. O fato que decide tudo: Warp so tem CPU e CUDA

Antes de estudar kernel, precisa saber em que ele roda. Lido em
`warp/_src/context.py`, a classe `Device` tem exatamente dois casos:

```python
ordinal (int): A Warp-specific label for the device. ``-1`` for CPU devices.
...
elif ordinal >= 0 and ordinal < runtime.core.wp_cuda_device_get_count():
    self.name = runtime.core.wp_cuda_device_get_name(ordinal).decode()
    self.arch = runtime.core.wp_cuda_device_get_arch(ordinal)
```

E o README do Warp:

> "The Windows x86-64 and Linux wheels support CPU execution and CUDA
> acceleration. CUDA acceleration requires a supported NVIDIA GPU and driver."

Procurei HIP, ROCm, Vulkan, Metal e OpenCL em `warp/native/`. As unicas
ocorrencias sao a palavra "graph" dentro de comentarios sobre CUDA graphs e
"ownership". **Nao existe backend AMD, nem experimental.**

E o README do MJWarp:

> "MJWarp is a GPU-accelerated version of MuJoCo, designed for NVIDIA hardware...
> MuJoCo Warp requires an NVIDIA GPU for fast simulation but supports CPU for
> development and debugging."

Conclusao operacional pra RX 6700 XT: `mujoco_warp` **roda** (no device CPU do
Warp, via LLVM/Clang JIT) e **nao acelera**. Nao e questao de configurar.

Isso nao torna o codigo inutil pra nos -- torna-o uma fonte de ALGORITMO, nao de
implementacao. Que e como este documento o trata.

---

## 1. A estrategia de paralelizacao, em uma linha

231 kernels em `mujoco_warp/_src/`. O padrao de indexacao de thread:

| padrao | ocorrencias |
|---|---|
| `worldid = wp.tid()` | 49 |
| `worldid, dofid = wp.tid()` | 26 |
| `worldid, bodyid = wp.tid()` | 15 |
| `worldid, actid = wp.tid()` | 12 |
| `worldid, elemid = wp.tid()` | 11 |
| `worldid, efcid = wp.tid()` | 10 |
| `worldid, tenid / nodeid / treeid` | 23 |
| sem `worldid` (conid, tid) | ~14 |

**`worldid` e a dimensao externa de quase tudo.** O paralelismo total e
`n_worlds x entidades_por_mundo`.

### Por que isso e decisivo pro Drosobot

Medido nesta maquina, o NeuroMechFly tem `nv = 93` graus de liberdade,
`nbody = 72`, `nu = 48`, `nefc ~ 8` em regime de caminhada.

Com `n_worlds = 1`, um kernel `(worldid, dofid)` lanca **93 threads**. A RX 6700
XT tem 2304 stream processors. O ocupante util seria ~4% da GPU, e o passo teria
que atravessar dezenas de kernels em sequencia -- cada um com latencia de
lancamento na casa de microssegundos.

Com `timestep = 1e-4`, sao 10.000 passos por segundo simulado. Mesmo
optimisticamente, so a cadencia de lancamento de kernel domina.

**Fisica GPU para UMA mosca nao e um problema de portar codigo. E um problema de
paralelismo insuficiente.** O MJWarp nao resolve isso porque nao se propoe a:
ele resolve vazao entre muitos mundos.

---

## 2. As tecnicas, e quais sao portaveis

O cabecalho de `solver.py:780` lista as otimizacoes do linesearch de forma
incomumente explicita. Vale traduzir cada uma para "isto depende de CUDA?".

| tecnica (MJWarp) | depende de CUDA? | equivalente portavel |
|---|---|---|
| **1. Fusao de kernels** -- junta `jv`, `quad`, `gauss`, `qacc/Ma`, `Jaref` num kernel so pra cortar lancamento | **nao** | vale igual em Vulkan/D3D12; lancamento tambem custa la |
| **2. Reducoes paralelas** -- `wp.tile_reduce` sobre linhas EFC dentro de cada mundo; empacota 3 reducoes `vec3` numa `mat33` | **nao** | reducao em workgroup: shared memory + `subgroupAdd` (Vulkan) / wave intrinsics (HLSL) |
| **3. Especializacao em tempo de compilacao** -- gera kernel por `(block_dim, ls_iterations, cone_type, fuse_jv)`; elimina ramos de cone eliptico quando o modelo e so piramidal | **nao** | SPIR-V **specialization constants**, ou gerar variantes de shader. E o mesmo padrao |
| **4. Avaliacao direta** -- calcula custo/gradiente/hessiana sem coeficientes intermediarios | **nao** | algebra, nao API |
| **5. Avaliacao de 3 alphas em lote** -- ramifica por tipo de restricao uma vez, carrega dados uma vez, avalia 3 pontos | **nao** | reduz divergencia de branch: vale mais ainda em AMD (wave64) |
| **6. Carregamento adiado** -- so le `efc_D`/`frictionloss` dentro do ramo que usa, pra baixar pressao de registrador | **nao** | idem; AMD tem orcamento de VGPR analogo |
| `wp.ScopedCapture()` -- CUDA graph capture | **sim** | nao ha equivalente exato; Vulkan chega perto com command buffers pre-gravados e reusados |
| `wp.tile_*` (239 usos) -- modelo de execucao em tile | parcialmente | e abstracao do Warp sobre shared memory + sync de bloco; reimplementavel, nao importavel |

**Leitura importante:** cinco das seis otimizacoes que o proprio upstream
destaca sao **algoritmicas**, nao de API. Elas sobrevivem a troca de backend. O
que nao sobrevive e o graph capture e a linguagem de kernel.

### Atomics usados

Contagem em `_src/*.py`:

```
147  wp.atomic_add
 18  wp.atomic_or
  9  wp.atomic_sub
  2  wp.atomic_min
  1  wp.atomic_max
  1  wp.atomic_cas
```

Tudo isso existe em Vulkan compute. **Com uma ressalva que importa**:
`atomic_add` em **float** nao e core no Vulkan -- precisa de
`VK_EXT_shader_atomic_float`. Em inteiro e core. Se formos por Vulkan, essa
extensao entra na lista de capacidades a verificar no dispositivo (e a RX 6700
XT precisa ser testada, nao presumida).

---

## 3. Broadphase e narrowphase

`collision_driver.py` implementa duas estrategias de broadphase:

- **`nxn`** -- todos os pares candidatos de uma lista pre-computada
  (`nxn_pairid`), um thread por par.
- **`SAP` (sweep and prune)**, incluindo `SAP_SEGMENTED`, com
  `sap_project` / `sap_binary_search` / `sap_range` em `collision_core.py`.

Narrowphase esta espalhado por `collision_primitive.py` (1611),
`collision_convex.py` (1491), `collision_gjk.py` (2729), `collision_sdf.py`
(1092), `collision_flex.py` (3747).

A separacao primitivo / convexo / GJK / SDF e ela mesma uma ideia portavel:
**pares que dao pra resolver analiticamente nunca chegam ao GJK**. Isso e
relevante direto pro nosso gargalo medido -- ver secao 5.

---

## 4. Pipeline de forward dynamics

`forward.py` mantem a estrutura do MuJoCo CPU:

```
fwd_position   -> kinematics, com/inertia, projecao, make constraint
fwd_velocity   -> velocidades de atuador e tendao
fwd_actuation  -> forca de atuador
solver         -> resolve restricoes
_advance       -> integra (euler / rungekutta4 / implicit)
```

Os integradores estao todos la (`euler`, `rungekutta4`, `implicit`), o que
significa que a correspondencia numerica com o MuJoCo CPU foi levada a serio pelo
upstream, e nao e um "physics engine parecido".

`block_cholesky.py` tem fatoracao de Cholesky em blocos para o passo de Newton,
criada por fabrica (`create_blocked_cholesky_factorize_solve_func`) -- de novo
especializacao em tempo de compilacao.

---

## 5. O que isso diz sobre o NOSSO gargalo

Medido nesta maquina (timers do proprio MuJoCo, callback `mjcb_time`, 800 passos,
modelo NeuroMechFly de `flygym 1.2.1`):

| etapa | ms/passo | % do passo |
|---|---|---|
| STEP (total) | 1,526 | 100% |
| FORWARD | 1,515 | 99,3% |
| POSITION | 1,406 | 92,1% |
| POS_COLLISION | 1,361 | 89,2% |
| **COL_NARROW** | **1,330** | **87,2%** |
| CONSTRAINT | 0,086 | 5,7% |
| POS_PROJECT | 0,015 | 1,0% |
| ADVANCE | 0,010 | 0,6% |
| COL_BROAD | 0,002 | 0,1% |

Com `ncon = 2`, `nefc = 8`, `solver_niter = 3`.

**O solver de restricoes -- que e onde o MJWarp concentra a engenharia mais
sofisticada -- e 5,7% do nosso passo.** Portar aquilo perfeitamente pra GPU nos
daria, no limite teorico, ~6%.

O que consome o passo e narrowphase. A causa, medida:

```
npair (pares de colisao explicitos) : 2220
  perna x perna                      : 2172   (98%)
  corpo x perna (inclui chao)        :   48
geoms de malha                       : 69 de 70
vertices totais nas malhas           : 247.724   (mediana 1283, max 41.147)
cascos convexos pre-computados       : 0 de 69
```

2220 pares de malha testados por passo, 10.000 passos por segundo simulado =
**22,2 milhoes de testes narrowphase por segundo simulado**, a ~0,6 us cada.

98% desse trabalho e a mosca checando se as pernas dela batem umas nas outras.

### A licao do MJWarp aplicada aqui

A ideia portavel mais valiosa do MJWarp pro Drosobot **nao e um kernel**. E a
disciplina de separar primitivo de convexo de GJK, e nunca mandar ao GJK um par
que uma primitiva resolve.

O FlyGym 2.x ja expoe exatamente isso:
`GeomFittingOption.ALL_TO_CAPSULES`. Trocar malha por capsula transforma
colisao capsula-capsula em teste analitico de segmento-segmento.

**Isso e CPU. Nao precisa de GPU, de Vulkan nem de NVIDIA.** E muda a fisica de
contato, entao exige validacao contra trajetoria de referencia antes de virar
padrao (ver `NEUROMECHFLY_PHYSICS_FEATURES.md`).

---

## 6. Se um dia formos fazer fisica em GPU

Registrando o que a leitura sugere, pra nao termos que reler:

1. **Nao comecar pelo solver.** E 5,7% do nosso passo. A engenharia bonita do
   MJWarp esta ali, e nao e onde esta o nosso problema.
2. **So faz sentido com muitos mundos.** Varredura de parametros, replicas com
   sementes diferentes, busca de ganho. Nao pro laco interativo de uma mosca.
   Se um dia quisermos "rodar 256 moscas com 256 sementes", MJWarp e a resposta
   pronta -- em NVIDIA.
3. **Colisao paralela precisa de pares, nao de mundos.** 2220 pares e
   paralelismo de verdade mesmo com um mundo so: um thread por par satura melhor
   que 93 threads de DOF. Se houver um caminho de fisica GPU util pra nos com um
   unico mundo, e esse -- e nao e o caminho que o MJWarp otimiza.
4. **Especializacao em tempo de compilacao e o padrao a copiar.** SPIR-V
   specialization constants dao o mesmo efeito do `create_*_func` do MJWarp.

---

## 7. O que eu nao verifiquei

- Nao rodei `mujoco_warp` nesta maquina, nem no device CPU. As afirmacoes sobre
  ausencia de backend AMD vem de leitura de codigo e do README, nao de execucao.
- Nao medi latencia de lancamento de kernel na RX 6700 XT. O argumento de
  "paralelismo insuficiente" e estrutural (93 threads contra 2304 SPs) e apoiado
  em contagem de kernels, nao em medicao de lancamento. Isso sera medido no
  microbenchmark.
- Nao conferi se `VK_EXT_shader_atomic_float` esta presente nesta GPU.
