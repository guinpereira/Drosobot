# HIP no Windows: auditoria de suporte para a RX 6700 XT

Consultado em **2026-09-20**, na matriz publicada atual, como pedido ("nao
confie em documentacao antiga").

---

## 1. O resultado, primeiro

**A Radeon RX 6700 XT (gfx1031) nao consta como suportada pelo HIP SDK para
Windows.**

Da [matriz de requisitos de sistema do HIP SDK para Windows](https://rocm.docs.amd.com/projects/install-on-windows/en/latest/reference/system-requirements.html)
(HIP SDK for Windows 7.2.0), a RX 6700 XT aparece com **X nas tres colunas** --
Runtime, HIP SDK e Debugger. E nao e so ela: **toda a linha RDNA2 de consumo, da
RX 6950 XT ate a RX 6600, esta marcada como nao suportada.**

A propria tabela define o que "X" significa:

> unsupported (no official support; prebuilt libraries may cause errors)

Confirmei com segunda fonte, porque a afirmacao e forte e contraria a premissa
com que a tarefa chegou. As discussoes no rastreador da AMD
([ROCm#2770](https://github.com/ROCm/ROCm/issues/2770),
[ROCm#2353](https://github.com/ROCm/ROCm/issues/2353)) tratam exatamente da
ausencia de gfx1031, e sao consistentes com a matriz.

### O contorno do Linux nao existe no Windows

No Linux, `HSA_OVERRIDE_GFX_VERSION=10.3.0` faz o ROCm tratar gfx1031 como
gfx1030 (que **e** suportada), e na pratica funciona porque as duas sao RDNA2
proximas.

**Essa variavel nao funciona no Windows.** Ha pedido aberto pra isso
([TheRock#1719](https://github.com/ROCm/TheRock/issues/1719)) e relatos de que
nao surte efeito ([ollama#3107](https://github.com/ollama/ollama/issues/3107)).

Ou seja, o unico contorno conhecido exige justamente o Linux que o projeto disse
pra nao exigir.

---

## 2. Auditoria de API: por que ela nao foi feita como planejada

A tarefa pedia, pra cada API HIP que nosso backend precisaria, verificar
"supported on Windows? / restricted? / missing? / version introduced?".

Essa auditoria pressupoe que o runtime **carregue** nesta GPU. Ele nao carrega:
sem dispositivo suportado, a distincao entre "API existe no Windows" e "API
existe aqui" e irrelevante -- nenhuma API HIP roda nesta maquina.

Fazer a tabela assim mesmo produziria um documento que parece conhecimento e nao
e. O que da pra registrar com honestidade e o nivel acima:

| camada | estado nesta maquina | fonte |
|---|---|---|
| HIP SDK instalado | **nao** | probe (`tools/gpu_probe/probe.py`) |
| GPU na matriz de suporte Windows | **nao** (gfx1031, X em Runtime/SDK/Debugger) | matriz oficial |
| contorno via `HSA_OVERRIDE_GFX_VERSION` | **nao funciona no Windows** | issues AMD/ollama |
| Windows exigido pela matriz | Windows 11 22H2 x86-64 | matriz oficial |
| Windows desta maquina | Windows 11 Pro 10.0.26200 | probe |
| HIP graphs documentados no porting guide | **nao mencionados** | porting guide |

A lacuna do HIP graphs vale nota: o porting guide nao fala deles, e no CUDA o
graph capture e o que amortiza o custo de lancar kernel em passos curtos (o
`wp.ScopedCapture()` do benchmark do FlyGym). Se um dia HIP entrar na conta, essa
e a primeira coisa a verificar -- nao assumir paridade.

---

## 3. O que isto decide

**HIP sai do caminho principal.** Nao por preferencia de API -- o mapeamento
CUDA->HIP e limpo (ver `CUDA_TO_HIP_MAPPING.md`) e a ideia de um unico conjunto
de kernels cross-vendor era boa. Sai porque **o requisito "AMD-first validado no
hardware que temos" e incompativel com uma GPU fora da matriz de suporte**.

Adotar HIP significaria uma de tres coisas, e nenhuma serve:

1. exigir Linux -- vetado pelo projeto
2. exigir outra GPU -- contraria "AMD-first no hardware que temos"
3. rodar fora de suporte, com bibliotecas pre-compiladas que a propria AMD avisa
   que "may cause errors" -- inaceitavel num projeto que valida ciencia

### O que fica de pe

**Vulkan compute.** Medido nesta placa, com todas as extensoes de interesse
presentes (10/10), incluindo `VK_EXT_shader_atomic_float`, que era o requisito
duvidoso:

```
device      AMD Radeon RX 6700 XT
api         1.4.315   driver: AMD proprietary driver
subgroup    64
LDS         32768 B
maxWG       1024
fp64        true
```

Vulkan nao tem matriz de suporte por modelo: se o driver expoe a extensao, ela
funciona. E cobre as quatro plataformas que o projeto exige com o mesmo SPIR-V.

**Unity ComputeShader (D3D12)**, que ja roda nesta GPU hoje, como caminho de
prova rapida e de integracao com a visualizacao.

---

## 4. Reavaliacao das opcoes da tarefa

A tarefa pedia decisao fundamentada entre cinco arquiteturas:

| opcao | veredito | motivo |
|---|---|---|
| **A) HIP como backend principal cross-vendor** | **descartada** | gfx1031 fora da matriz Windows; sem contorno no Windows |
| **B) HIP AMD + CUDA NVIDIA** | **descartada** | mesma razao do lado AMD, e duplica manutencao |
| **C) Vulkan cross-platform** | **recomendada** | unico que cobre Win/Linux x AMD/NVIDIA com um codigo; capacidade medida nesta placa |
| **D) D3D12 Windows + outra no Linux** | **nao** | duas implementacoes pra ganhar nada sobre C |
| **E) hibrido** | **parcial** | Unity ComputeShader pra prova e visualizacao; Vulkan como alvo; CUDA opcional se houver NVIDIA; CPU sempre |

Recomendacao: **C, chegando por E.** Comecar pelo Unity ComputeShader (custo de
entrada quase zero, ja roda, e resolve a Fase 16 de buffers compartilhados),
medir, e so entao investir em Vulkan como backend definitivo.

**Isto continua sendo recomendacao, nao decisao.** O benchmark decide.

---

## 5. Se o cenario mudar

Coisas que reabririam o HIP, e que vale reconferir de tempos em tempos:

- AMD adicionar gfx1031 a matriz Windows
- `HSA_OVERRIDE_GFX_VERSION` (ou equivalente) passar a funcionar no Windows
- o projeto ganhar uma GPU AMD que esteja na matriz
- o requisito de Windows-first ser relaxado para alguma carga especifica

O probe (`tools/gpu_probe/probe.py`) ja detecta HIP e registra `gfx_target`, entao
no dia em que isso mudar a mudanca aparece no JSON em vez de passar batido.
