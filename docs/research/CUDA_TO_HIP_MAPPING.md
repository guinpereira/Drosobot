# CUDA -> HIP: o que existe pra portar, e o que nao existe

Fontes primarias:
[HIP porting guide](https://rocm.docs.amd.com/projects/HIP/en/latest/how-to/hip_porting_guide.html),
[HIPIFY](https://rocm.docs.amd.com/projects/HIPIFY/en/latest/),
[HIP performance guidelines](https://rocm.docs.amd.com/projects/HIP/en/latest/how-to/performance_guidelines.html).
Upstreams em `research/UPSTREAM_LOCK.md`.

---

## 0. Antes do mapeamento: onde esta o CUDA

A tarefa pedia mapear "qualquer codigo CUDA/NVIDIA encontrado em MuJoCo Warp,
Warp, MJX". Contei os arquivos antes de comecar:

| repositorio | `.cu` | `.cuh` | `.py` | chamadas a runtime CUDA |
|---|---|---|---|---|
| **mujoco_warp** | **0** | **0** | 93 | **nenhuma** |
| **mujoco** (inclui MJX) | **0** | 0 | 265 | nenhuma |
| **warp** (NVIDIA) | **18** | 0 | 677 | sim (15.038 linhas de `.cu`) |

Busquei `cudaMalloc`, `cudaMemcpy`, `cudaStream`, `__global__`, `__shared__`,
`threadIdx`, `blockIdx` em `mujoco_warp/`: **zero ocorrencias**.

Isso muda a natureza da tarefa e vale explicar por que.

**MJWarp nao e codigo CUDA.** Sao 231 kernels escritos em Python com o DSL do
Warp:

```python
@wp.kernel
def _some_kernel(m: Model, d: Data):
    worldid, dofid = wp.tid()
    ...
```

Quem transforma isso em CUDA e o **Warp**, em tempo de execucao: ele gera C++/CUDA
a partir da AST do Python e compila com NVRTC. O CUDA nunca existe como arquivo
no repositorio do MJWarp.

Consequencia direta: **HIPIFY nao tem o que converter no MJWarp.** Rodar
`hipify-clang` ou `hipify-perl` ali produziria um relatorio vazio -- nao por
limitacao da ferramenta, mas porque nao ha entrada. Nao rodei, e o motivo e esse;
rodar pra gerar um relatorio vazio seria teatro.

Os 18 `.cu` do Warp tambem nao sao os kernels de fisica. Sao a **infraestrutura
de runtime**: `bvh.cu`, `hashgrid.cu`, `mesh.cu`, `reduce.cu`, `scan.cu`,
`sort.cu`, `sparse.cu`, `deterministic.cu`, `runlength_encode.cu`. Util, mas nao
e o MuJoCo.

### O que "portar MJWarp para AMD" significaria de verdade

Nao e traduzir kernels. Seria **escrever um backend HIP para o NVIDIA Warp**:
fazer o codegen dele emitir HIP em vez de CUDA, trocar NVRTC por hiprtc, e portar
os 15k linhas de runtime. Isso e construir um compilador alternativo dentro de um
projeto da NVIDIA que tem exatamente dois devices (CPU e CUDA -- verificado em
`warp/_src/context.py`).

Nao e "nao vale a pena". E que nao e uma tarefa de porte, e um projeto proprio, e
entregaria aceleracao de **fisica multi-mundo**, que (medido, ver
`NEUROMECHFLY_PHYSICS_FEATURES.md`) nao e o nosso gargalo.

**Onde HIP continua interessante pra nos: nos NOSSOS kernels**, os de LIF esparso
da Fase 9. O mapeamento abaixo serve pra isso.

---

## 1. Mapeamento de primitivas

Para o que o nosso backend neural precisaria. "Windows?" refere-se ao HIP SDK
para Windows *em GPU suportada* -- que nao e a nossa (ver
`HIP_WINDOWS_API_AUDIT.md`).

| CUDA | HIP | Windows? | diferenca semantica | implicacao |
|---|---|---|---|---|
| `cudaMalloc` | `hipMalloc` | sim | -- | nenhuma |
| `cudaFree` | `hipFree` | sim | -- | nenhuma |
| `cudaMemcpy` | `hipMemcpy` | sim | -- | nenhuma |
| `cudaMemcpyHostToDevice` | `hipMemcpyHostToDevice` | sim | -- | nenhuma |
| `cudaMemcpyToSymbol` | `hipMemcpyToSymbol` | sim | -- | nenhuma |
| `cudaDeviceSynchronize` | `hipDeviceSynchronize` | sim | -- | custa igual; evitar no laco |
| `cudaMallocHost` (pinned) | `hipHostMalloc` | sim | -- | AMD recomenda pinned pra transferencia |
| streams | `hipStream*`, `hipStreamPerThread` | sim | -- | mesma ideia de sobreposicao |
| events | `hipEvent*` (`cuEventCreate` -> `hipEventCreate`) | sim | -- | usar pra medir kernel |
| `cuLaunchKernel` (driver API) | `hipModuleLaunchKernel` | sim | HIP oferece **as duas** formas de memcpy (direcao no nome e por parametro) | -- |
| atomics inteiros | iguais | sim | -- | -- |
| `atomicAdd` float | igual, **mas condicional** | sim | guardado por `__HIP_ARCH_HAS_FLOAT_ATOMIC_ADD__`; consultavel em runtime | **checar em vez de assumir** |
| `__shared__` | `__shared__` (LDS) | sim | LDS da AMD e 32 KB aqui (medido via Vulkan) | orcamento menor que os 48-164 KB comuns em NVIDIA |
| `__syncthreads()` | igual | sim | -- | "cada barreira trava todos ate o mais lento" |
| **warp size 32** | **`warpSize`** | sim | **muda de verdade** | ver secao 2 |
| `__shfl_*`, lane masks | equivalentes | sim | "shift >31 limparia o registrador de 32 bits" em wave64 | **fonte classica de bug silencioso** |
| `__launch_bounds__(T, B)` | `__launch_bounds__` | sim | **segundo parametro tem outro significado**: warps/EU, nao blocos/SM | traduzir, nao copiar |
| `--maxregcount` | **nao existe** | -- | `amdclang++` nao suporta | controlar registrador por outro caminho |
| CUDA graphs | HIP graphs | **verificar** | o porting guide **nao menciona** graphs | ver `HIP_WINDOWS_API_AUDIT.md` |
| contexto por device | **espaco de enderecos unico** | sim | "HIP-Clang define um espaco de enderecos de processo inteiro" | simplifica multi-device |

---

## 2. Porte nao e trocar nome -- as tres armadilhas reais

**2.1 Tamanho de warp.** O guia e explicito:

> "Code should not assume a warp size of 32 or 64, as AMD GPU architectures have
> different warp sizes. The `warpSize` built-in should be used in device code."

E o guia de desempenho:

> "AMD Instinct GPUs execute threads in warps of 64, while AMD Radeon GPUs
> execute threads in warps of 32."

Mas o Vulkan **nesta placa** reporta `subgroupSize = 64`. RDNA2 executa wave32 ou
wave64 conforme o shader; o driver reportou 64 como tamanho de subgrupo. **Nao
resolvi essa divergencia** e nao vou escolher um numero pra escrever num
documento: qualquer kernel nosso usa `warpSize` / `gl_SubgroupSize` e nunca uma
constante.

Isso pega qualquer reducao em nivel de warp -- e o MJWarp usa `wp.tile_reduce`
239 vezes.

**2.2 `__launch_bounds__`.** O segundo parametro do CUDA e
`MIN_BLOCKS_PER_MULTIPROCESSOR`; no HIP e em termos de warps e unidades de
execucao. Copiar o numero da fonte NVIDIA da ocupancia errada sem erro de
compilacao.

**2.3 Lane masks.** Em wave64, deslocar mais de 31 bits num registrador de 32
zera tudo. Bug silencioso, so aparece como resultado errado.

---

## 3. Bibliotecas: quais o upstream usa

A tarefa pedia cruzar cuBLAS / cuSPARSE / cuSOLVER / cuRAND / cuFFT / CUB.

Levantei o que o upstream de fato usa:

| biblioteca | usada? | onde |
|---|---|---|
| cuBLAS | **nao** | MJWarp faz Cholesky em blocos com kernel proprio (`block_cholesky.py`) |
| cuSPARSE | **nao** | esparsidade tratada em kernel proprio |
| cuSOLVER | **nao** | idem |
| cuRAND | **nao** | -- |
| cuFFT | **nao** | -- |
| CUB | indiretamente | o Warp tem `scan.cu`, `sort.cu`, `reduce.cu` proprios |

**Nenhuma biblioteca CUDA de alto nivel entra na conta.** Isso e boa noticia: nao
ha `hipBLAS` / `rocBLAS` / `hipSPARSE` a arrastar, e a orientacao da AMD de
preferir `rocXXX` a `hipXXX` em hardware AMD **nao se aplica ao nosso caso**,
porque nao usariamos nenhuma das duas.

Nosso kernel de LIF esparso tambem nao precisa: CSR + scatter atomico e codigo
nosso, nao chamada de biblioteca.

---

## 4. Guia de desempenho da AMD aplicado aos nossos kernels

Do [performance guidelines](https://rocm.docs.amd.com/projects/HIP/en/latest/how-to/performance_guidelines.html),
o que incide em cada carga que vamos escrever. Vale **independente do backend**
-- as mesmas regras valem em Vulkan e D3D12.

**A) Atualizacao LIF (densa, N neuronios)**
- coalescencia: `v`, `g`, `ref_until` em arrays separados (SoA), thread `i` toca
  elemento `i`. Sai de graca se nao usarmos struct-of-neuron.
- divergencia: o ramo `if refratario` divide o warp. Melhor calcular os dois
  lados e selecionar sem ramo.
- ocupancia: poucos registradores por thread; o kernel e aritmetica simples.
- **nao** precisa de LDS.

**B) Propagacao sinaptica esparsa (CSR, 125M arestas)**
- coalescencia: a leitura de `indices[]`/`weight[]` e sequencial por neuronio
  pre-sinaptico -- bom. A **escrita** e dispersa -- ruim, e inevitavel.
- atomics: `atomicAdd` em float sobre `g[pos]`. Na AMD isso e condicional
  (`__HIP_ARCH_HAS_FLOAT_ATOMIC_ADD__`); em Vulkan exige
  `VK_EXT_shader_atomic_float`, que **medi presente** nesta GPU.
- contencao: neuronio muito convergente vira ponto quente. Reducao em LDS antes
  do atomico global e a mitigacao padrao -- e e onde os 32 KB de LDS pesam.
- transferencia: os ~1,1 GB de CSR sobem **uma vez**. "Minimizar transferencia
  host/device" aqui e trivial de obedecer, e e o principal argumento a favor de
  GPU pro neural.

**C) Solver de contato experimental**
- adiado. E 5,7% do passo (medido). O guia nao muda isso.

**D) Buffers de telemetria / reducao**
- "consolidar transferencias pequenas numa grande": a telemetria deve ler **um**
  buffer por quadro, nao um por camada.
- assincronia: `AsyncGPUReadback` (Unity) ou copia em stream separada, pra nao
  serializar o laco.
- **e onde mais arriscamos perder o ganho**: ler estado por passo desmonta a
  sobreposicao. Medir, nao supor.

---

## 5. Conclusao

1. **Nao ha CUDA pra converter no MJWarp.** HIPIFY nao se aplica; o alvo real
   seria escrever um backend HIP pro Warp, que e outro projeto.
2. **O mapeamento CUDA->HIP e quase 1:1** nas primitivas que usariamos. As
   armadilhas sao tres: `warpSize`, `__launch_bounds__`, lane masks.
3. **Nenhuma biblioteca CUDA de alto nivel esta em jogo**, entao nao ha
   dependencia ROCm a arrastar.
4. **O guia de desempenho da AMD vale mesmo se nao usarmos HIP.** Coalescencia,
   LDS, divergencia, ocupancia e minimizar transferencia sao propriedades da GPU,
   nao da API.
5. **Mas nada disso roda nesta placa via HIP** -- ver `HIP_WINDOWS_API_AUDIT.md`.
