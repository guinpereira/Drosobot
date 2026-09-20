# Toolchain do HIP SDK no Windows -- e o toolchain que vamos precisar de verdade

Consultado em 2026-09-20.

Contexto: este documento foi pedido pra garantir que um backend nosso consiga
compilar de forma reproduzivel. A pesquisa mudou qual backend e esse (ver
`HIP_WINDOWS_API_AUDIT.md`), entao ele cobre os dois -- o que a AMD exige, e o
que passa a valer pra nos.

---

## 1. HIP SDK no Windows: o que consegui confirmar

| item | estado |
|---|---|
| plataforma | Windows 11 22H2 x86-64 |
| plugin de IDE | instalador oferece "Visual Studio 2017, 2019, 2022 Plugin" como componente opcional |
| compilador | `hipcc` / `amdclang++` (o porting guide cita `amdclang++` e nota que ele **nao** suporta `--maxregcount`) |
| versao de VS exigida, workloads, Windows SDK, gerador CMake, linker, redistribuiveis | **nao documentado na pagina de instalacao** |

A pagina de instalacao lista componentes e passos, mas nao especifica versao de
Visual Studio, workloads, Windows SDK, gerador do CMake nem o que precisa ser
redistribuido pra enviar um aplicativo a terceiros.

**Nao vou preencher essa tabela por inferencia.** Sao exatamente os detalhes que,
errados, custam um dia de build quebrado. Se HIP voltar a ser candidato, isso se
resolve instalando o SDK e medindo, nao lendo.

E ha uma razao pratica pra nao gastar mais nisso agora: **o SDK nao suporta a
nossa GPU** (gfx1031 marcada como nao suportada em Runtime, HIP SDK e Debugger).
Levantar requisito de compilacao pra um toolchain que nao roda aqui seria
trabalho sem consumidor.

---

## 2. O toolchain que passa a valer

Com Vulkan e Unity ComputeShader como caminhos recomendados, os requisitos de
build mudam -- e ficam bem mais leves.

### 2.1 Unity ComputeShader (primeiro alvo)

| item | requisito | estado |
|---|---|---|
| compilador de shader | embutido na Unity (HLSL -> DXIL/DXBC) | ja temos |
| runtime | Unity 6000.3.2f1, Built-in RP, D3D12 | ja temos |
| Visual Studio | **nao necessario** pro shader | -- |
| reproducibilidade | `.compute` versionado no repo; a Unity compila no import | boa |
| distribuicao | sai no build da Unity, sem redistribuivel extra | boa |

Este e o motivo de ele ser o primeiro alvo: **custo de toolchain zero**. O
projeto ja compila shader hoje.

### 2.2 Vulkan compute (alvo definitivo candidato)

| item | requisito | estado |
|---|---|---|
| runtime | `vulkan-1.dll` do driver | **presente** (Vulkan 1.4.309 / device 1.4.315) |
| compilador de shader | `glslc` ou `dxc` (Vulkan SDK) | **Vulkan SDK nao instalado** |
| compilar GLSL/HLSL -> SPIR-V | offline, no build | SPIR-V versionado = build reproduzivel |
| host | C++ (MSVC/clang) **ou** binding Python | a decidir |
| Visual Studio | so se o host for C++ nativo | -- |
| distribuicao | SPIR-V pre-compilado + runtime do driver | nada a redistribuir |

Ponto a favor da reproducibilidade: **SPIR-V compilado offline e versionado no
repositorio**. O artefato que roda e o que esta no git, nao o que a maquina de
alguem compilou. E o mesmo principio de `research/UPSTREAM_LOCK.md`.

Decisao em aberto: host em C++ nativo (mais trabalho, controle total, vira plugin
nativo da Unity se precisar) ou binding Python (mais rapido de prototipar, uma
dependencia a mais). **Nao decidir antes do benchmark.**

---

## 3. O que instalar quando chegar a hora

Nao instalar agora -- listado pra nao ter que pesquisar de novo:

- **Vulkan SDK (LunarG)** -- traz `glslc`, `dxc`, camadas de validacao e
  `vulkaninfo`. As camadas de validacao importam: compute shader errado em GPU
  falha em silencio, com resultado errado em vez de excecao.
- **RenderDoc** -- captura e inspeciona dispatch de compute, em D3D12 e Vulkan.
  Funciona em AMD no Windows, que e mais do que da pra dizer do `rocprofv3`.

Sobre perfilamento: a tarefa pedia `rocprofv3` ou equivalente documentado.
`rocprofv3` faz parte do ROCm e **nao se aplica** aqui pelo mesmo motivo do
resto. Os equivalentes que funcionam nesta maquina:

| ferramenta | serve pra | disponivel |
|---|---|---|
| Unity Profiler + Recorder de GPU | tempo de dispatch dentro da Unity | sim, ja |
| `AsyncGPUReadback` com timestamp | latencia de readback | sim, ja |
| RenderDoc | inspecao de dispatch, buffers, correcao | instalar |
| AMD Radeon GPU Profiler (RGP) | ocupancia, banda, wavefront, em D3D12/Vulkan | instalar |
| `hipEvent*` / `rocprofv3` | -- | **nao se aplica** |

O RGP e o mais proximo do que a AMD recomenda e roda em Windows com Radeon, que e
exatamente a nossa configuracao.

---

## 4. Resumo

1. O toolchain do HIP no Windows nao esta totalmente documentado na pagina de
   instalacao, e **nao investiguei alem disso** porque o SDK nao suporta esta GPU.
2. O caminho recomendado tem toolchain mais simples: Unity hoje sem instalar
   nada; Vulkan depois, com o SDK da LunarG.
3. Reproducibilidade vem de versionar o artefato compilado (SPIR-V) e o
   `.compute`, nao de fixar versao de compilador na maquina de cada um.
4. Perfilamento existe e e bom nesta plataforma -- RGP e RenderDoc -- so nao se
   chama `rocprofv3`.
