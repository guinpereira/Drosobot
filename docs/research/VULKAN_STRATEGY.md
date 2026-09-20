# Vulkan compute: a estratégia, e por que não foi implementado agora

**Decisão: não implementar nesta rodada.** A interface está pronta, o caminho
está mapeado, e o laboratório funciona sem ele. Este documento existe para que
a próxima pessoa não precise refazer a análise.

---

## O que já está pronto

`IComputeBackend` (`sim/neural/compute/base.py`) é um `Protocol` de 13 métodos e
nenhum deles menciona API gráfica:

```
prepara            reset                escreve_externo     escreve_forcados
lif                scatter              acumula             sincroniza
le_indices         le_contagem_grupos   zera_contagem_grupos
le_estado_completo le_contagem_total    resumo
```

A ciência não passa por aqui. Ela está em:

| onde | o que |
|---|---|
| `model.py` | parâmetros de Shiu et al., coeficientes do passo exato, escala de ponto fixo |
| `engine.py` | **ordem** do passo e semântica do atraso — `lif → scatter → acumula → avança cursor` |
| `connectome_loader.py` | CSR, bodyIds, subgrafo |

Um backend novo implementa os 13 métodos e não escolhe nada científico. Isso já
foi exercido duas vezes: `CPUBackend` (fp64, referência) e `OpenCLBackend`
(gfx1031) dão **0 divergência** em 512 e em 1.500 neurônios reais.
`D3D12Backend` existe como esqueleto e levanta `BackendIndisponivel` com a
instrução do que falta — ele não finge estar pronto.

## O trabalho que Vulkan exige

O que **não** é problema:

- disponibilidade — medido: Vulkan 1.4.315 na RX 6700 XT, com
  `VK_KHR_shader_atomic_float` e `shaderBufferInt64Atomics`;
- portabilidade — é o único caminho que cobre Windows, Linux, AMD, NVIDIA e
  Intel com uma implementação só;
- o formato dos dados — CSR, ring buffer de atraso e acumulador em ponto fixo
  são buffers simples, sem nada específico de OpenCL.

O que **é** trabalho, e é onde a estimativa mora:

1. **Os kernels precisam ser reescritos.** Hoje são OpenCL C
   (`sim/neural/kernels/*.cl`). Vulkan consome SPIR-V, então seriam GLSL
   compute + `glslangValidator`, ou os mesmos `.cl` via `clspv`. Os quatro
   kernels são pequenos, mas a tradução tem que ser **verificada**, não
   presumida — a equivalência com a CPU em fp64 é o critério, e já existe teste
   para ela.

2. **Não existe binding Python decente.** `pyopencl` faz o host inteiro em
   OpenCL; para Vulkan, as opções são `vulkan` (ctypes, pouco mantido),
   `kompute` (C++ com binding), ou escrever o host em C++ e chamar por
   `ctypes`/`pybind11`. Isso é a maior parte do custo: descritores, pipeline
   layouts, memória, barreiras e sincronização são explícitos em Vulkan de um
   jeito que OpenCL esconde.

3. **Gerência de memória manual.** `VkDeviceMemory`, `vkMapMemory`, staging
   buffers e escolha de heap passam a ser nossos. Com 25,5M arestas isso importa
   — hoje são ~400 MiB de CSR na GPU.

## Por que não bloqueia

O objetivo desta rodada era o laboratório funcionando em Windows + AMD com o
Male CNS inteiro. Isso já acontece, medido: OpenCL na RX 6700 XT, 164.451
neurônios, 25.550.583 arestas, RTF 0,055×.

E o número que decide: **a física é 79–86% do relógio.** O neural é 18% com o
conectoma inteiro. Reescrever o backend neural numa API nova, no melhor caso,
mexe em menos de um quinto do custo — e não há evidência de que Vulkan seja mais
rápido que OpenCL para *este* kernel nesta GPU. Seria trocar a API pela API.

Trocar por pureza arquitetural não vale o vertical funcionando. A regra que o
projeto segue é a mesma de sempre: medir antes.

## Quando fazer

Vulkan passa a valer a pena se qualquer uma destas acontecer:

- **Linux virar alvo de execução, não só de compatibilidade.** OpenCL em AMD no
  Linux depende de ROCm/Mesa Rusticl e é mais frágil que no Windows.
- **A física deixar de ser o gargalo.** Se a física cair para menos da metade do
  relógio, o neural passa a valer otimização de verdade.
- **pyopencl virar um problema.** Hoje ele é só o *host binding* — os kernels
  são nossos. Se ele quebrar numa versão de Python ou de driver, a saída já está
  desenhada.
- **Precisarmos de atômicos de 64 bits.** O acumulador é int32 com folga de
  3,98× (medida) contra o teto. Um conectoma maior ou pesos maiores derrubam
  essa folga, e aí `shaderBufferInt64Atomics` deixa de ser curiosidade.

## O que não fazer

Não implementar Vulkan "pela metade" e deixar `VulkanBackend` levantando
exceção junto com o D3D12. Um esqueleto já basta como marcador de intenção; dois
viram ruído. A interface é o contrato — ela já diz o que um backend precisa
fazer, e isso é o que precisava estar pronto.

---

## Estado dos backends

| backend | estado | verificação |
|---|---|---|
| `CPUBackend` | operacional | referência em fp64 |
| `OpenCLBackend` | **operacional, é o que roda** | 0 divergência vs CPU em 512 e 1.500 neurônios |
| `D3D12Backend` | esqueleto declarado | LIF já executou correto em HLSL; falta host nativo |
| `VulkanBackend` | não existe | este documento |

Ver também [`BACKEND_MATRIX.md`](BACKEND_MATRIX.md) para a medição de
disponibilidade e adequação por plataforma, e
[`DROSOBOT_COMPUTE_ARCHITECTURE.md`](DROSOBOT_COMPUTE_ARCHITECTURE.md) para a
fronteira entre ciência e execução.
