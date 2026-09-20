# Whole-CNS na GPU: cabe em tempo real?

Medido na RX 6700 XT / Windows / D3D12, sobre o conectoma **real** do Male CNS
v1.0. Dados e scripts em `benchmarks/neural/`.

---

## 1. O grafo real -- e a correcao que ele obrigou

A pergunta "125M sinapses ou 125M arestas?" estava certa em ser feita primeiro,
porque as duas respostas dimensionam kernels diferentes.

O modelo do Drosobot ja agrega por par: `signed_weights()` soma as sinapses entre
dois neuronios e produz um peso. A unidade computacional e a **aresta**.

Baixamos `connectome-weights-male-cns-v1.0-minconf-0.5.feather` (1,0 GB) de
[male-cns.janelia.org](https://male-cns.janelia.org/download/) e medimos.

### Primeira medicao: errada, e vale registrar por que

| | |
|---|---|
| corpos | **88.384.522** |
| arestas | 151.856.684 |
| sinapses | 311.833.243 |

88 milhoes de "neuronios" nao existem. O arquivo traz **todos os segmentos**,
inclusive fragmentos minusculos nao revisados -- a propria pagina diz
"segment-to-segment... for all segments". Segmento nao e neuronio.

### Segunda medicao: filtrada pelas anotacoes

Mantendo corpos com tracado util (`Roughly traced`, `Reviewed`,
`Prelim Roughly traced`, `RT Hard to trace`) e excluindo glia:

| | |
|---|---|
| neuronios anotados | 164.451  (Janelia publica ~166.700) |
| neuronios com aresta | **164.249** |
| **arestas dirigidas** | **25.550.583** |
| sinapses | **123.967.037** |
| sinapses por aresta | **4,85** |
| arestas sobreviventes do grafo de segmentos | 16,83% |

**Os "125M" sao sinapses. O grafo computacional tem 25,5M arestas** -- 4,85x
menor. Dimensionar por 125M teria inflado memoria e banda por um fator 5.

### Grau

| | media | mediana | p95 | p99 | max |
|---|---|---|---|---|---|
| out-degree | 155,6 | 114 | 408 | 798 | 11.203 |
| in-degree | 155,6 | 100 | 456 | 862 | 11.526 |

Os 1000 neuronios de maior out-degree concentram **7,41%** das arestas. A cauda
existe mas nao domina -- nao precisa de estrategia especial pra hub, ao menos
nao antes de medir com ela.

### Memoria

```
row_offsets  (int32, N+1)      0,6 MiB
targets      (int32, E)       97,5 MiB
weights      (fp32, E)        97,5 MiB
-------------------------- estatico  195,6 MiB
estado dinamico (v, g, ref)    1,9 MiB
spike bitmask                  0,02 MiB
-------------------------- TOTAL     197,5 MiB   de 12.288 MiB
```

**1,6% da VRAM.** Cabe com folga absurda -- e isso muda o que da pra planejar.

O `delay` nao ocupa nada: o modelo usa **atraso fixo** (`T_DELAY = 1,8 ms`), entao
nao existe campo por aresta. Um delay arbitrario custaria +24,4 MiB e uma
estrutura de baldes. A simplificacao vale a pena e esta registrada.

### Sinal

Peso ja sai com sinal, pela regra de Dale sobre o neurotransmissor do
pre-sinaptico (a mesma de `sim/connectome_model.py`):

```
excitatorias   15.164.662
inibitorias     9.799.540
mudas             586.381   (neurotransmissor desconhecido)
```

3.125 neuronios (1,9%) nao tem NT conhecido. Eles ficam com **sinal 0**, nao +1:
assumir excitatorio por omissao inventaria conectoma. As arestas deles somem do
efeito e o numero fica na tela.

---

## 2. Propagacao esparsa: as duas estrategias

`Assets/Compute/SynapticScatter.compute`, D3D12, grupo de 64.

**DENSE_SCAN** -- um thread por neuronio; quem disparou percorre suas arestas.
**FRONTIER** -- compacta os que dispararam num vetor, depois so eles percorrem.
Lancado por `DispatchIndirect`, entao **o tamanho da frontier nunca passa pela
CPU**.

Conductancia acumulada em ponto fixo inteiro: HLSL SM 5.0 nao tem
`InterlockedAdd` em float. Escala 1024/mV deixa o degrau em ~0,001 mV, uma ordem
abaixo do erro de fp32 que ja medimos (6e-05 mV).

### Resultado (melhor de 3, 100 passos por medicao)

| disparo | ativos | arestas/passo | FRONTIER | **DENSE** | eventos/s |
|---|---|---|---|---|---|
| 0,1% | 146 | 21.130 | 131,7 us | **70,8 us** | 0,30 G/s |
| 1,0% | 1.657 | 259.452 | 360,2 us | **293,4 us** | 0,88 G/s |
| 5,0% | 8.183 | 1.221.782 | 718,0 us | **591,7 us** | 2,06 G/s |
| 10% | 16.496 | 2.595.878 | 1286,4 us | **1034,1 us** | 2,51 G/s |
| 100% (estresse) | 164.451 | 25.550.583 | 6691,4 us | **6440,3 us** | 3,97 G/s |

**DENSE vence em todas as taxas.** A compactacao mais o dispatch indireto custam
mais do que economizam nesta escala. Contra-intuitivo, e e o motivo de as duas
terem sido implementadas em vez de uma escolhida no papel.

Variancia entre repeticoes: 5–10%, com o Editor renderizando junto. Uma passada
anterior deu resultado nao-monotonico (0,5% mais lento que 1,0%) -- era ruido, e
so repetindo deu pra saber.

---

## 3. O numero que decide a arquitetura

`dt` neural = 0,5 ms. Pra tempo real, o passo neural inteiro tem que caber em
**500 us**.

| disparo | LIF | scatter | **total** | orcamento |
|---|---|---|---|---|
| 0,1% | 17,9 us | 70,8 us | **88,7 us** | **5,6x de folga** |
| 1,0% | 17,9 us | 293,4 us | **311,3 us** | **1,6x de folga** |
| 5,0% | 17,9 us | 591,7 us | 609,6 us | 1,2x acima |
| 10% | 17,9 us | 1034,1 us | 1052,0 us | 2,1x acima |
| 100% | 17,9 us | 6440,3 us | 6458,2 us | 12,9x acima |

**Whole-CNS neural em tempo real e possivel na RX 6700 XT ate ~1% de disparo por
passo de 0,5 ms.**

Em atividade fisiologica a fracao que dispara numa janela de 0,5 ms e bem menor
que 1% -- o baseline do modelo de Shiu et al. e 0 Hz, e mesmo sob estimulo a
atividade e esparsa. Os 10% e os 100% sao estresse, nao regime esperado.

E ha folga de engenharia ainda nao explorada: nas taxas baixas o tempo e
dominado por custo fixo, nao por trabalho. A 0,1%, 70,8 us pra 21 mil arestas da
3,4 ns por aresta, contra 0,25 ns/aresta a 100%. O `LimpaAcum` percorre os 164k
neuronios todo passo e podia ser fundido ao kernel de LIF -- exatamente a
"fusao de kernels" que o MJWarp lista como sua otimizacao numero 1.

---

## 4. O que isto NAO estabelece

- **Nao houve validacao numerica do scatter.** O kernel de LIF denso foi
  comparado com a referencia (999/1000 neuronios com contagem de spike
  identica); o de propagacao **ainda nao**. Velocidade sem corretude nao conta,
  e essa e a proxima divida.
- **O padrao de disparo e aleatorio uniforme**, nao atividade real do circuito.
  A distribuicao de GRAU e real; quem dispara, nao. Atividade real e
  correlacionada, e correlacao muda contencao atomica.
- **O atraso nao esta implementado.** O benchmark propaga no mesmo passo. Com
  `T_DELAY = 1,8 ms` e `dt = 0,5 ms` sao ~4 passos de atraso: um ring buffer de 4
  slots de `g`, que custa +7,6 MiB. Barato, mas nao medido.
- **Nao ha laco fechado.** Sem retina, sem fisica, sem telemetria. O passo neural
  medido esta isolado.
- **Medido com o Editor aberto**, o que adiciona ruido e provavelmente custa
  desempenho. Um processo dedicado deve ser mais rapido, nao mais lento.

---

## 5. O que isto muda no plano

1. **Whole-CNS deixa de ser ambicao distante.** 197,5 MiB e 311 us por passo a 1%
   de atividade sao numeros de hoje, nesta placa.
2. **A estrategia e DENSE, nao frontier** -- ao menos ate a fusao de kernels e a
   atividade correlacionada serem medidas.
3. **A proxima divida e corretude do scatter**, nao velocidade.
4. **Nao ha motivo pra fisica em GPU.** O neural cabe; a fisica continua sendo
   problema de geometria na CPU.
