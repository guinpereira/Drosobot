# Drosobot Compute -- proposta de arquitetura

Proposta, nao decisao. Sai da pesquisa em `FLYGYM2_ARCHITECTURE.md`,
`MJWARP_IMPLEMENTATION.md`, `NEUROMECHFLY_PHYSICS_FEATURES.md`,
`WINDOWS_AMD_RESEARCH.md` e `BACKEND_MATRIX.md`.

---

## 1. O que a pesquisa mudou no plano

O plano entrou assumindo "fisica e lenta, precisamos de GPU". As medicoes dizem
outra coisa, e vale registrar antes de desenhar qualquer camada:

| achado | consequencia |
|---|---|
| 87% do passo e narrowphase de malha (2172 pares perna-perna) | o maior ganho de fisica e **geometria, na CPU** |
| o solver de restricoes e 5,7% do passo | portar o solver pra GPU renderia ~6% |
| MJWarp paraleliza sobre `worldid` | fisica GPU **nao ajuda uma mosca so** |
| Warp so tem CPU e CUDA | MJWarp **nao roda acelerado em AMD**, nem experimentalmente |
| JAX nao tem GPU em Windows | MJX-JAX tambem nao e caminho aqui |
| overhead do wrapper FlyGym 1.x = 39% do passo | migrar pro 2.x vale ~1,7x, **na CPU** |
| Vulkan na RX 6700 XT tem atomic float, int64, subgroup 64, 32 KB LDS | o caminho GPU portavel **existe e e capaz** |

**Inversao de prioridade:** GPU e pro **neural** (166.691 neuronios, 125M
sinapses -- nao cabe em CPU). Fisica melhora primeiro na CPU, e so vai pra GPU se
um dia quisermos muitos mundos.

Isso reordena as fases do pedido: Fase 9 (neural GPU) continua antes da Fase 14
(fisica GPU), agora com medicao sustentando, e a Fase 3/8 rende um ganho de CPU
que nao dependia de nada disso.

---

## 2. Principios

1. **A referencia manda.** Todo backend e verificado contra o caminho de
   referencia (`fast_lif` NumPy, validado contra Brian2; MuJoCo CPU pra fisica).
   Backend que diverge fora da tolerancia declarada esta errado, mesmo que rapido.
2. **Backend e detalhe, nao arquitetura.** Nenhum codigo de experimento importa
   Vulkan, CUDA ou `ComputeBuffer`.
3. **Fallback sempre existe.** Sem GPU, o Drosobot roda. Mais devagar, igual.
4. **O que produziu o dado fica visivel.** A interface diz sempre qual backend
   rodou. Ver Fase 19 do pedido.
5. **Nada de escolha gravada antes de benchmark.** A deteccao de dispositivo
   propoe; a medicao decide.

---

## 3. Estrutura proposta

```
sim/                          # o que existe hoje, intocado
    connectome_model.py
    fast_lif.py               # REFERENCIA neural; nao sai do lugar
    experiments/
    telemetry/
    lab_runner.py

drosobot/
    compute/
        device.py             # deteccao e enumeracao
        capabilities.py       # o que o dispositivo sabe fazer
        registry.py           # backend disponivel -> escolha

    neural/
        backend.py            # interface NeuralBackend
        reference.py          # adaptador sobre sim/fast_lif.py
        connectivity.py       # CSR, sinal, atraso: DADO ESTATICO
        state.py              # v, g, spike, refratario: ESTADO DINAMICO
        kernels/
            lif.hlsl          # Unity ComputeShader
            lif.comp          # Vulkan/SPIR-V
        unity_compute.py
        vulkan.py
        cuda.py               # opcional, se houver NVIDIA

    physics/
        backend.py            # interface PhysicsBackend
        mujoco_cpu.py         # REFERENCIA; envolve o que ja usamos
        flygym2_cpu.py        # depois da migracao
        # nada de GPU aqui ate haver evidencia

    validation/
        neural_equivalence.py # backend x referencia, mesma semente
        physics_trajectory.py # trajetoria de referencia e tolerancias

benchmarks/
    neural/                   # JSON/CSV reproduziveis
    physics/
```

`drosobot/` nasce ao lado de `sim/`, nao por cima. Nada em `sim/` muda enquanto
a validacao nao passar. `fast_lif.py` continua sendo a referencia -- ele e
pequeno, compreensivel e ja validado contra o Brian2, e e exatamente por isso
que serve de padrao-ouro.

---

## 4. Interfaces

### Dispositivo

```python
@dataclass(frozen=True)
class Capabilities:
    vendor: str              # "AMD" | "NVIDIA" | "Intel" | "CPU"
    name: str
    platform: str            # "Windows" | "Linux"
    vram_mb: int | None
    backends: tuple[str, ...]        # ("unity_compute", "vulkan", "cpu")
    fp32: bool
    fp64: bool
    atomic_float_add: bool           # medido; nao e garantido em Vulkan core
    subgroup_size: int | None
    shared_memory_bytes: int | None

def detect_devices() -> list[Capabilities]: ...
def detect_best_device(workload: str) -> Capabilities: ...
```

`workload` existe porque a resposta certa difere entre neural e fisica -- hoje a
melhor fisica e CPU mesmo com GPU boa disponivel.

### Neural

```python
class NeuralBackend(Protocol):
    def build(self, conn: Connectivity, dt_ms: float) -> None: ...
    def run(self, duracao_ms: float, taxas_hz, rng=None) -> Counts: ...
    def snapshot(self, nomes=None) -> list[dict]: ...
    @property
    def name(self) -> str: ...
    @property
    def device(self) -> Capabilities: ...
```

Assinatura escolhida pra casar com `fast_lif.Rede.roda()` e `.snapshot()`, que e
o que `sim/experiments/corpo.py` ja chama. O adaptador de referencia deve ser
quase vazio -- se nao for, a interface esta errada.

### Fisica

```python
class PhysicsBackend(Protocol):
    def reset(self, seed: int) -> None: ...
    def step(self) -> None: ...
    def joint_angles(self): ...
    def body_positions(self): ...
    def contacts(self): ...
    def retina(self): ...
```

Deliberadamente **pull-based**, que e a licao de CPU do FlyGym 2.x: o 1.x monta
uma observacao completa 10.000 vezes por segundo simulado pra gente usar uma
fracao 100 vezes.

---

## 5. Dados do conectoma: estatico e dinamico separados

Pro alvo de 166.691 neuronios e ~125M sinapses. Nunca matriz densa: 166k x 166k
em fp32 seriam ~111 TB.

**Estatico** (sobe uma vez, nunca muda durante a corrida):

```
indptr      int32[N+1]        CSR, offset de cada neuronio
indices     int32[E]          alvo de cada sinapse
weight      float32[E]        peso ja com sinal (regra de Dale)
delay_slot  uint8[E]          balde de atraso
```

**Dinamico** (por passo):

```
v           float32[N]
g           float32[N]
ref_until   int32[N]
spike       uint32[N/32]      bitmask
fila        ring buffer de atraso
```

Com E = 125M: `indices` 500 MB + `weight` 500 MB + `delay_slot` 125 MB = **~1,1
GB de dado estatico**. Cabe nos 12 GB da RX 6700 XT com folga. O estado dinamico
e ~2,7 MB, desprezivel.

A conta importa porque decide a estrategia: **o estatico domina, e ele nao se
move.** Uma vez na VRAM, o trafego por passo e so o estado -- que e o argumento
central a favor de GPU aqui, e contra GPU na fisica de um mundo so.

**Denso por passo x dirigido por evento**: a Fase 12 pede benchmark dos dois. A
estrutura acima serve aos dois; o hibrido provavel e decaimento barato em todos
os N mais scatter atomico so a partir de quem disparou. Nao assumir -- medir.

---

## 6. Escolha de backend (proposta, nao gravada)

| plataforma | neural | fisica |
|---|---|---|
| Windows AMD | Unity ComputeShader -> Vulkan | MuJoCo CPU |
| Windows NVIDIA | mesmo portavel; CUDA opcional | MuJoCo CPU |
| Linux AMD | Vulkan | MuJoCo CPU |
| Linux NVIDIA | Vulkan; CUDA opcional | MuJoCo CPU (MJWarp se multi-mundo) |
| sem GPU | CPU (`fast_lif`) | MuJoCo CPU |

Por que Unity ComputeShader primeiro: funciona hoje nesta maquina (medido,
D3D12, `supportsComputeShaders=True`), custa zero dependencia nova, e os
`ComputeBuffer` sao os mesmos que o renderer le -- o que resolve de graca a Fase
16 (estado neural na GPU alimentando a visualizacao do cerebro sem ida e volta
pela CPU).

Por que Vulkan como alvo: e o unico que cobre as quatro plataformas com o mesmo
codigo, e a GPU expoe tudo que precisamos.

---

## 7. Onde isto encosta na fronteira atual

Hoje o Drosobot tem uma fronteira dura: simulacao em Python, Unity so visualiza,
telemetria de mao unica, controle num socket curto e declarado.

Se o backend neural for Unity ComputeShader, **o circuito passa a rodar dentro do
processo da Unity**, e essa fronteira muda de forma. Precisa de desenho
explicito, nao de acidente.

Proposta: manter a autoridade onde esta, movendo apenas execucao.

- o conectoma, os parametros de Shiu et al. e a decisao de quem dispara
  continuam definidos em Python e conferidos contra a referencia
- a Unity executa um kernel cujo resultado e **verificavel bit a bit** contra
  `fast_lif` com a mesma semente
- se a verificacao nao rodar, o Lab marca o backend como nao validado na interface

O que nao pode acontecer: a Unity virar a autoridade do comportamento porque foi
conveniente. Um kernel que a gente nao consegue comparar com a referencia nao
entra.

---

## 8. Ordem proposta

**P0 -- pesquisa (feito)**
Clones, lock, auditorias, matriz de backends, esta proposta.

**P1 -- prova de compute**
Microbenchmark na RX 6700 XT. Kernel LIF em ComputeShader contra `fast_lif` em
CPU, em 1k / 10k / 50k / 166.691 neuronios. Medir updates/s, latencia, VRAM,
custo de readback. Salvar JSON em `benchmarks/neural/`.

**P2 -- equivalencia**
Portar o circuito optomotor. Comparar com Brian2 e `fast_lif` na mesma semente:
contagem de spike, tempo de spike, v, g, refratario, atraso, saida motora.
Nenhuma otimizacao passa se mudar o comportamento.

**P3 -- ganho de CPU na fisica (independente, pode ir em paralelo)**
Trajetoria de referencia -> podar pares de colisao -> medir -> avaliar capsulas.
E aqui que estao os 87%.

**P4 -- rede grande**
CSR do CNS inteiro, denso x event-driven, benchmark de memoria.

**P5 -- FlyGym 2.x**
Migrar um experimento, medir contra o atual.

**P6 -- Unity Lab**
Backend e device na interface; buffers compartilhados se a medicao justificar.

---

## 9. O que esta proposta pode ter de errado

- **Unity ComputeShader pode nao valer.** Se o custo de readback por passo comer
  o ganho, o caminho e processo proprio em Vulkan. So o benchmark diz.
- **O event-driven pode perder pro denso** nesta escala. Scatter atomico tem
  contencao; 166k neuronios com decaimento denso e so ~2,7 MB de estado. Medir.
- **O ganho de CPU na fisica pode ser menor do que a conta sugere.** Podar pares
  reduz narrowphase, mas broadphase e construcao de restricao nao somem.
- **Capsulas podem quebrar a marcha.** Ai o ganho nao existe, e isso e resultado,
  nao fracasso.
