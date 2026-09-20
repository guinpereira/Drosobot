# Pesquisa: Drosobot Compute

Branch `research/gpu-core`. Nada em `sim/` foi alterado -- os experimentos, os
parametros cientificos e o `timestep` estao como estavam.

Commits upstream: `research/UPSTREAM_LOCK.md`. Reproduzir:
`bash research/clone_upstream.sh`.

---

## Os cinco achados que mudam o plano

**1. O ganho de GPU do FlyGym 2.x e vazao, nao latencia.**
O benchmark oficial calcula `realtime_factor = steps_per_second * timestep` com
`steps_per_second` **somado sobre `n_worlds`**. Um "300x" com 1024 mundos e ~0,3x
por mosca. O Drosobot roda uma mosca num laco fechado. Nao se aplica.

**2. O Warp so tem CPU e CUDA.** Verificado em `warp/_src/context.py`: device e
ordinal −1 (CPU, via LLVM) ou CUDA. Sem HIP, sem Vulkan, nem experimental. Na
RX 6700 XT, `mujoco_warp` roda no device CPU e nao acelera.

**3. 87% do nosso passo de fisica e colisao de malha.** Medido com os timers do
MuJoCo: `COL_NARROW` 1,330 ms de 1,526 ms. O solver de restricoes -- onde o
MJWarp concentra a engenharia -- e 5,7%. A causa: **2220 pares de colisao
explicitos, 2172 deles perna-contra-perna**, entre malhas sem casco convexo
pre-computado. O remedio e geometria, na CPU.

**4. A RX 6700 XT nao e suportada pelo HIP SDK no Windows.** gfx1031 aparece com
X em Runtime, HIP SDK e Debugger; toda a RDNA2 de consumo esta fora. E
`HSA_OVERRIDE_GFX_VERSION`, que contorna isso no Linux, nao funciona no Windows.
HIP sai do caminho principal.

**5. Vulkan nesta placa tem tudo que precisamos.** 10/10 das extensoes de
interesse, incluindo `VK_EXT_shader_atomic_float` (que nao e core e era a
duvida). subgroup 64, 32 KB de LDS, fp64.

---

## Prova de compute: rodando na RX 6700 XT

Kernel LIF em HLSL/D3D12 pela Unity, 2000 passos, contra NumPy no Ryzen 7 5700X.

| neuronios | CPU fp64 | CPU fp32 | **GPU** | ganho vs fp32 | ganho vs fp64 |
|---|---|---|---|---|---|
| 1.000 | 19,1 us | 29,5 us | **8,1 us** | 3,6x | 2,4x |
| 10.000 | 70,6 us | 57,5 us | **8,5 us** | 6,8x | 8,3x |
| 50.000 | 425,7 us | 220,8 us | **11,3 us** | 19,5x | 37,7x |
| **166.691** | 4076,5 us | 1030,4 us | **17,9 us** | **57,6x** | **228x** |

166.691 e o numero de neuronios do Male CNS. A 17,9 us por passo, uma janela de
rede de 10 ms (20 passos) custa **358 us** -- cabe num laco interativo.

Note o formato da curva: em n=1000 a GPU leva 8,1 us, praticamente o mesmo que em
n=10000. Abaixo de ~10k neuronios estamos medindo **custo de lancamento**, nao
trabalho. E exatamente o argumento da secao 1 do `MJWARP_IMPLEMENTATION.md` sobre
por que fisica GPU de um mundo so nao funciona -- so que agora medido.

### Validado, nao so medido

`tests/test_lif_gpu_equivalencia.py` compara o estado final da GPU contra o
integrador de referencia (o mesmo que e validado contra o Brian2):

```
n=1000 passos=400
spikes  GPU   5461   referencia   5460
contagem difere em 1/1000 neuronios (0.10%)
  neuronio 460: GPU 6 x ref 5   margem ao limiar 1.873e-07 mV
                                (dentro do eps32 3.815e-06)
|dv| max 6.295e-05 mV (nos 999 que concordam)   |dg| max 4.898e-06 mV
```

O unico neuronio que divergiu passou a **1,873e-07 mV do limiar**, e o epsilon do
fp32 naquele ponto e **3,815e-06 mV** -- vinte vezes maior que a margem. Ele esta
dentro do vao representavel: o fp32 nao consegue dizer de que lado do limiar ele
esta.

**Isso e propriedade do fp32, nao erro de formula**, e o teste distingue os dois
casos: divergencia perto do limiar passa (com teto de 0,5%), divergencia longe do
limiar falha, porque ai seria conta errada.

Vale registrar como limitacao cientifica de qualquer backend neural em fp32:
spike e evento discreto, e eventos amplificam diferenca numerica minuscula.

---

## O que isto NAO prova

- Nao prova que Unity ComputeShader e o backend certo. Falta o custo do laco
  fechado com readback por passo, e falta a **propagacao sinaptica esparsa** --
  125M arestas com scatter atomico, que e a parte dificil e nao foi medida.
- Nao prova nada sobre fisica em GPU.
- O benchmark usa entrada deterministica, nao Poisson, e sem sinapse nenhuma.
  E o passo de estado denso, isolado de proposito.

---

## Documentos

| arquivo | assunto |
|---|---|
| [FLYGYM2_ARCHITECTURE.md](FLYGYM2_ARCHITECTURE.md) | o que mudou do 1.x pro 2.x, e de onde vem cada ganho |
| [MJWARP_IMPLEMENTATION.md](MJWARP_IMPLEMENTATION.md) | como o MJWarp acelera, e o que disso sobrevive sem CUDA |
| [NEUROMECHFLY_PHYSICS_FEATURES.md](NEUROMECHFLY_PHYSICS_FEATURES.md) | subset do MuJoCo que usamos; onde o tempo vai |
| [WINDOWS_AMD_RESEARCH.md](WINDOWS_AMD_RESEARCH.md) | compute pesado em Windows+AMD, sem WSL |
| [BACKEND_MATRIX.md](BACKEND_MATRIX.md) | matriz por plataforma, com fonte por celula |
| [CUDA_TO_HIP_MAPPING.md](CUDA_TO_HIP_MAPPING.md) | mapeamento de primitivas, e por que HIPIFY nao se aplica |
| [HIP_WINDOWS_API_AUDIT.md](HIP_WINDOWS_API_AUDIT.md) | suporte do HIP SDK a esta GPU |
| [HIP_WINDOWS_TOOLCHAIN.md](HIP_WINDOWS_TOOLCHAIN.md) | toolchain do HIP e o que passa a valer |
| [DROSOBOT_COMPUTE_ARCHITECTURE.md](DROSOBOT_COMPUTE_ARCHITECTURE.md) | proposta de arquitetura |

## Ferramentas

```
tools/gpu_probe/probe.py           sonda o hardware -> benchmarks/hardware/
benchmarks/neural/bench_cpu.py     LIF na CPU
unity ... LifBenchmark.RodarTudo() LIF na GPU
unity ... LifBenchmark.Validar()   estado da GPU pro teste de equivalencia
tests/test_lif_gpu_equivalencia.py GPU x referencia
```

## Proximo passo sugerido

**Propagacao sinaptica esparsa.** E a parte nao medida e a que decide se GPU
serve pro conectoma inteiro. O passo de estado denso ja se mostrou barato; o
scatter atomico sobre 125M arestas e outra historia.

Em paralelo, e independente: **podar os 2172 pares de colisao perna-perna**, com
trajetoria de referencia antes e depois. E o ganho de fisica mais barato que
existe, e nao depende de nenhuma decisao de backend.
