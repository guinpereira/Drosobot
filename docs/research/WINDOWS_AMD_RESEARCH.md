# Compute pesado em Windows + AMD, sem Linux e sem WSL

Pergunta: **como rodar compute pesado numa RX 6700 XT no Windows sem depender de
Linux nem de WSL?**

Tudo abaixo foi sondado nesta maquina em 2026-09-20. Onde eu nao rodei, esta
marcado.

Hardware: Ryzen 7 5700X, Radeon RX 6700 XT 12 GB, Windows 11 Pro 10.0.26200,
driver AMD 32.0.21045.5002.

---

## 1. O que existe nesta maquina, agora

| componente | estado |
|---|---|
| `vulkan-1.dll`, `vulkaninfo.exe` | presentes, instance **Vulkan 1.4.309** |
| driver Vulkan do dispositivo | apiVersion **1.4.315**, "AMD proprietary driver" |
| `d3d12.dll` | presente |
| `DirectML.dll` | presente no sistema |
| `OpenCL.dll` + `amdocl64.dll` | presentes (ICD do driver AMD) |
| ROCm / HIP SDK | **nao instalado** |
| Unity | rodando em **Direct3D12** na RX 6700 XT |

### Capacidades Vulkan medidas (`vulkaninfo`)

```
VK_EXT_shader_atomic_float          OK
VK_EXT_shader_atomic_float2         OK
VK_KHR_shader_atomic_int64          OK
VK_KHR_cooperative_matrix           OK
VK_KHR_buffer_device_address        OK
VK_KHR_16bit_storage                OK
VK_KHR_shader_float16_int8          OK
VK_EXT_subgroup_size_control        OK
VK_KHR_timeline_semaphore           OK

maxComputeSharedMemorySize      = 32768   (32 KB de LDS)
maxComputeWorkGroupInvocations  = 1024
subgroupSize                    = 64      (wave64)
shaderFloat64                   = true
```

Isto fecha a pergunta em aberto do `MJWARP_IMPLEMENTATION.md`: **todos os atomics
que o MJWarp usa existem nesta GPU**, incluindo `atomic_add` em float, que nao e
core do Vulkan e precisava de confirmacao.

### Capacidades via Unity (medido no Editor em Play)

```
graphicsDeviceType          Direct3D12
graphicsDeviceName          AMD Radeon RX 6700 XT
supportsComputeShaders      True
supportsAsyncCompute        True
supportsAsyncGPUReadback    True
maxComputeWorkGroupSize     1024
graphicsMemorySize          12243 MB
graphicsShaderLevel         50
```

---

## 2. Avaliacao das opcoes

### 2.1 Unity ComputeShader (HLSL -> D3D12)

**Funciona hoje, zero dependencia nova.** Unity ja esta no projeto, ja usa D3D12
nesta GPU, ja reporta compute + async + readback assincrono.

A favor:
- nada pra instalar; o pipeline de build/distribuicao ja existe
- `ComputeBuffer`/`GraphicsBuffer` sao os MESMOS buffers que o renderer le. Isso
  e o ponto da Fase 16: estado neural calculado na GPU pode ir direto pra
  visualizacao do cerebro sem GPU->CPU->GPU
- `AsyncGPUReadback` pra trazer so o que a telemetria precisa, sem travar
- funciona igual em NVIDIA; em Linux a Unity usa Vulkan com o mesmo HLSL

Contra:
- **amarra o compute ao processo da Unity.** Hoje o Drosobot roda simulacao em
  Python e visualizacao na Unity, como processos separados, e essa separacao e o
  que garante que a Unity nao decide nada. Mover o circuito pra dentro da Unity
  mexe nessa fronteira e precisa de desenho cuidadoso
- HLSL em SM 5.0 e mais pobre que SPIR-V moderno (sem buffer device address, sem
  subgroup explicito do jeito do Vulkan)
- depurar compute shader na Unity e pior que em Vulkan com ferramentas dedicadas

### 2.2 Vulkan compute (SPIR-V), processo proprio

**Tecnicamente o mais capaz.** A GPU expoe tudo que precisamos, incluindo as
specialization constants que substituem a especializacao em tempo de compilacao
do MJWarp (secao 2 do `MJWARP_IMPLEMENTATION.md`).

A favor:
- portavel de verdade: Windows AMD, Windows NVIDIA, Linux AMD, Linux NVIDIA, com
  o mesmo SPIR-V
- controle total de memoria, filas, sincronizacao
- command buffers pre-gravados e reusados sao o analogo mais proximo do CUDA
  graph capture
- interop com D3D12/Vulkan da Unity e possivel via memoria externa

Contra:
- **muito mais codigo pra escrever.** Vulkan nao tem "ola mundo" curto
- precisa de um caminho de build C++ ou de um binding Python maduro
- interop de buffer com a Unity e trabalho real, nao configuracao

### 2.3 DirectX 12 Compute direto

Mesmas capacidades do caminho Unity, sem a Unity. Faz sentido se quisermos um
processo de compute proprio e so no Windows. Perde a portabilidade Linux que o
Vulkan da de graca. **Nao vejo vantagem sobre Vulkan** pro nosso caso, exceto
familiaridade com HLSL.

### 2.4 HIP / ROCm no Windows

Nao instalado aqui. A AMD publica HIP SDK para Windows, mas o suporte a GPUs de
consumo RDNA2 historicamente fica atras do Linux e varia por release. **Nao
testei.** Nao vou afirmar que funciona nem que nao funciona.

O que da pra dizer com seguranca: mesmo que HIP funcione, ele **nao destrava
`mujoco_warp`**, porque o Warp nao tem backend HIP (ver
`MJWARP_IMPLEMENTATION.md`, secao 0). HIP seria pra codigo NOSSO, e ai Vulkan
cobre o mesmo terreno com portabilidade maior.

### 2.5 JAX em Windows AMD

JAX nao publica wheel de GPU para Windows. O caminho de GPU AMD para JAX e ROCm
em **Linux**. Em Windows, JAX roda em CPU.

Consequencia: **MJX-JAX nao e caminho de GPU nesta maquina.** Ele era a aposta
mais plausivel de "XLA alcanca AMD e NVIDIA" e nao alcanca, no Windows.

Nao testei; isto vem da matriz de distribuicao do projeto JAX. Se algum dia
mudar, vale remedir -- MJX-JAX continua sendo a alternativa conceitualmente mais
limpa para vazao multi-mundo fora do ecossistema NVIDIA.

### 2.6 DirectML

Presente no sistema. **Provavelmente inadequado pro nosso workload, e vale dizer
por que** em vez de so descartar.

DirectML e uma biblioteca de operadores de rede neural: convolucao, gemm,
normalizacao, ativacao. O catalogo e denso e orientado a tensor.

Nosso workload e:
- propagacao esparsa em grafo (CSR, ~125M sinapses)
- dinamica LIF (decaimento exponencial, limiar, refratario) -- elemento a
  elemento, com estado
- scatter atomico dirigido por evento (so quem disparou propaga)

Nada disso e um operador DirectML. Daria pra torturar um `gemm` esparso pra
fazer parte, mas perderiamos o ganho do event-driven e ganhariamos uma
dependencia. **Descartado, com motivo registrado.**

Mesmo raciocinio vale para ONNX Runtime DirectML: e um executor de grafo de
inferencia, nao um lancador de kernel arbitrario.

### 2.7 OpenCL

O ICD da AMD esta instalado. OpenCL roda em AMD e NVIDIA, Windows e Linux, e
`pyopencl` e um binding Python maduro -- seria o caminho de menor esforco pra um
primeiro kernel fora da Unity.

Contra: a NVIDIA mantem OpenCL em nivel minimo, as ferramentas sao piores que as
de Vulkan, e o futuro e menos claro. Serve como **plano B de prototipagem**, nao
como aposta.

### 2.8 Numba

Instalado. `numba.cuda` e NVIDIA. `numba` em CPU (njit + prange) **e relevante
mesmo assim**: e o que o FlyGym usa na retina, e e um ganho de CPU real sem GPU
nenhuma. Entra como parte do caminho CPU, nao como backend GPU.

---

## 3. O que isto recomenda

Em ordem, pro que temos:

1. **Unity ComputeShader (D3D12)** para o primeiro marco de "compute rodando de
   verdade na RX 6700 XT". Custo de entrada quase zero, e alinhado com a Fase 16
   (buffers compartilhados com a visualizacao).
2. **Vulkan compute** como alvo do backend definitivo, se o microbenchmark
   mostrar que vale. E o unico caminho que cobre Windows AMD, Windows NVIDIA,
   Linux AMD e Linux NVIDIA com o mesmo codigo.
3. **CUDA** so como caminho opcional acelerado quando houver NVIDIA -- nunca como
   requisito.
4. **CPU (NumPy + Numba)** como fallback universal e como referencia de
   correcao. Continua sendo o `fast_lif` validado contra o Brian2.

Nada disso deve ser gravado antes do benchmark. Esta secao e recomendacao a
partir de capacidade sondada, nao de desempenho medido.

---

## 4. Requisitos que isto nao viola

- Nao exige WSL.
- Nao exige Docker.
- Nao exige dual boot.
- Nao exige NVIDIA.
- Nao exige ROCm.

A configuracao principal testada e Windows + AMD, que era o pedido.
