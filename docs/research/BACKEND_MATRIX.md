# Matriz de backends

Estado em 2026-09-20, commits em `research/UPSTREAM_LOCK.md`.

**Como ler a coluna "fonte".** Nada aqui e opiniao. Cada celula e uma de tres
coisas:

- **medido** -- rodei nesta maquina e tenho o numero
- **codigo** -- li no fonte upstream e cito onde
- **docs** -- esta na documentacao oficial do projeto e nao verifiquei rodando

Onde nao tenho nenhuma das tres, escrevi "nao sei". Isso e informacao, nao
lacuna.

---

## 1. Disponibilidade por plataforma

| backend | Win AMD | Win NVIDIA | Linux AMD | Linux NVIDIA | fonte |
|---|---|---|---|---|---|
| CPU (NumPy/Numba) | sim | sim | sim | sim | medido |
| MuJoCo CPU | sim | sim | sim | sim | medido |
| CUDA | **nao** | sim | **nao** | sim | - |
| NVIDIA Warp | **so CPU** | sim | **so CPU** | sim | codigo: `warp/_src/context.py`, so ordinal -1 (CPU) e CUDA |
| MJWarp | **so CPU** | sim | **so CPU** | sim | codigo: herda do Warp; README "requires an NVIDIA GPU" |
| MJX-JAX | **nao (CPU)** | nao (CPU) | sim (ROCm) | sim | docs: JAX nao publica wheel GPU pra Windows |
| HIP / ROCm | nao sei | n/a | sim | n/a | nao instalado aqui; nao testei |
| Vulkan compute | **sim** | sim | sim | sim | medido: Vulkan 1.4.315 no dispositivo |
| DirectX 12 compute | **sim** | sim | nao | nao | medido: `d3d12.dll`, Unity em D3D12 |
| Unity ComputeShader | **sim** | sim | sim (Vulkan) | sim | medido: `supportsComputeShaders=True` |
| OpenCL | **sim** | sim | sim | sim | medido: `OpenCL.dll` + `amdocl64.dll` |
| DirectML | sim | sim | nao | nao | medido: DLL presente (mas ver secao 3) |

---

## 2. Adequacao ao nosso workload

Duas cargas diferentes, e elas nao tem a mesma resposta.

### 2.1 Neural: LIF esparso sobre conectoma

Alvo: 166.691 neuronios, ~125M sinapses, CSR, propagacao dirigida por evento.

| backend | esparso | atomics | maturidade | complexidade | veredito |
|---|---|---|---|---|---|
| CPU NumPy | ok | n/a | e a nossa referencia validada | baixa | **referencia de correcao** |
| Unity ComputeShader | ok | sim (`InterlockedAdd`) | alta | **baixa** | **primeiro alvo** |
| Vulkan compute | ok | sim, incl. float (medido) | alta | alta | **alvo definitivo candidato** |
| D3D12 direto | ok | sim | alta | media-alta | so Windows; Vulkan domina |
| OpenCL | ok | sim | media | media | plano B |
| CUDA | ok | sim | alta | media | so NVIDIA |
| MJWarp | n/a | n/a | n/a | n/a | e fisica, nao neural |
| DirectML | **nao** | n/a | alta | media | catalogo de operador denso; ver secao 3 |
| JAX | ok em teoria | - | alta | media | sem GPU em Windows |

### 2.2 Fisica: NeuroMechFly, uma mosca

Alvo medido: `nv=93`, `nbody=72`, `nu=48`, `ncon~2`, `nefc~8`, `timestep=1e-4`.

| backend | 1 mundo | N mundos | veredito |
|---|---|---|---|
| MuJoCo CPU | **15,8 s/s** (medido) | escala por processo | **e o que temos, e e a referencia** |
| MJWarp (NVIDIA) | ruim por construcao | otimo | paralelismo e `worldid`; ver abaixo |
| MJX-JAX | ruim | bom | mesma logica, e sem GPU no Windows |
| fisica GPU propria | nao sei | nao sei | nao ha evidencia de que valha |

**Por que "ruim por construcao" com um mundo:** em `mujoco_warp/_src/`, 231
kernels, e o padrao de thread e `worldid` como dimensao externa (49 kernels
`worldid = wp.tid()`, 26 `worldid, dofid`, 15 `worldid, bodyid`...). Com
`n_worlds=1` um kernel de DOF lanca 93 threads numa GPU de 2304 SPs, e o passo
atravessa dezenas de kernels em sequencia, 10.000 vezes por segundo simulado.

E o benchmark oficial do FlyGym confirma que o numero publicado e agregado:

```python
df["steps_per_second"] = sim_steps * df["n_worlds"] / df["walltime_s"]
df["realtime_factor"]  = df["steps_per_second"] * sim_timestep
```

---

## 3. Por que DirectML esta descartado

Nao e por indisponibilidade -- a DLL esta no sistema. E por incompatibilidade de
forma.

DirectML expoe operadores de rede neural densa: convolucao, gemm, normalizacao,
ativacao, reducao. Nosso nucleo e:

1. travessia esparsa de grafo em CSR com 125M arestas
2. estado LIF por neuronio (decaimento, limiar, refratario, atraso)
3. scatter atomico so a partir de quem disparou

O item 3 e o que torna o problema tratavel -- em atividade fisiologica, a fracao
de neuronios que dispara por passo e pequena. Um operador denso obriga a pagar
por todos. Usar DirectML aqui trocaria o algoritmo certo por um operador
conveniente.

**Registrado como pedido: DirectML nao serve, e a razao e o event-driven.**

---

## 4. Onde cada plataforma fica

| plataforma | neural preferido | fisica preferida |
|---|---|---|
| Windows AMD (nosso alvo) | Unity ComputeShader -> Vulkan | MuJoCo CPU |
| Windows NVIDIA | mesmo caminho portavel; CUDA opcional | MuJoCo CPU; MJWarp se multi-mundo |
| Linux AMD | Vulkan | MuJoCo CPU |
| Linux NVIDIA | Vulkan; CUDA opcional | MJWarp se multi-mundo |
| sem GPU | CPU | MuJoCo CPU |

**Nenhuma dessas escolhas esta gravada.** Elas saem de capacidade sondada e de
leitura de codigo. A decisao de backend so vale depois do microbenchmark, que e
o proximo passo.

---

## 5. O que falta medir

- Latencia de lancamento de kernel na RX 6700 XT (D3D12 e Vulkan).
- Throughput de LIF por backend, em 1k / 10k / 50k / 166.691 neuronios.
- Custo de `AsyncGPUReadback` por quadro de telemetria.
- Se HIP no Windows funciona nesta GPU.
- Se MuJoCo 3.9 (o que o FlyGym 2.x pede) muda o perfil de narrowphase.
