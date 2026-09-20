# Por que o Giant Fiber cala no conectoma inteiro

O runtime integrado deu 4 fugas no escopo de circuito e 0 no Male CNS inteiro,
com o mesmo estimulo. "Nao e bug" e uma afirmacao, e afirmacao precisa de
numero. Este documento tem o numero.

Medido por `benchmarks/neural/trace_giant_fiber.py`, saida em
`benchmarks/neural/giant_fiber_trace.json`.

---

## Como a comparacao foi feita justa

A **mesma realizacao de Poisson** nos dois escopos: 311 LC4/LPLC2 a 20 Hz por
400 ms (800 passos), semente 7, **2433 spikes sensoriais** sorteados uma vez e
reaplicados identicos. Sem isso estariamos comparando dois estimulos diferentes.

Nenhum parametro foi alterado.

---

## O resultado

| | circuito | whole CNS |
|---|---|---|
| neuronios | 1.261 | 164.451 |
| arestas | 105.130 | 25.550.583 |
| **arestas entrando no GF** | **1.455** | **1.455** |
| excitatorias / inibitorias / mudas | 819 / 612 / 24 | 819 / 612 / 24 |
| peso total das entradas | +6.454,8 / −3.608,3 mV | +6.454,8 / −3.608,3 mV |
| **excitacao recebida** | 42.934,9 mV | **37.623,3 mV** |
| **inibicao recebida** | −38.866,8 mV | **−79.796,2 mV** |
| **liquido** | **+4.068,1 mV** | **−42.172,9 mV** |
| v do GF (max / media) | −45,026 / −50,409 | −45,086 / **−298,874** |
| **spikes do GF** | **106** | **35** |

Limiar −45,0 mV, repouso −52,0 mV.

### A resposta, em uma frase

**O GF nao dispara porque recebeu 5.312 mV a MENOS de excitacao e 40.929 mV a
MAIS de inibicao** -- o liquido vai de +4.068 mV para −42.173 mV, troca de sinal.

E o ponto decisivo: **a conectividade e IDENTICA**. Os mesmos 1.455 pares, os
mesmos pesos, o mesmo sinal. O circuito reduzido ja continha todas as entradas
do Giant Fiber.

O que muda e **quanto esses pre-sinapticos disparam**. No circuito eles so
recebem os LC4/LPLC2. No conectoma inteiro eles sao alimentados pelo resto do
cerebro, e passam a disparar muito mais.

### De onde vem a inibicao a mais

| circuito | | whole CNS | |
|---|---|---|---|
| PVLP010 (10074) | −16.166,7 mV | GNG300 (529031) | −14.374,8 mV |
| CL367 (11361) | −6.283,2 mV | CL367 (11361) | −14.361,6 mV |
| PVLP024 (16475) | −2.987,6 mV | SAD073 (13274) | −6.237,0 mV |
| PVLP010 (531985) | −2.940,3 mV | SAD073 (10315) | −5.880,6 mV |
| AN05B006 (14899) | −2.482,7 mV | LHAD1g1 (10297) | −5.800,0 mV |
| PVLP024 (16459) | −1.826,6 mV | LHAD1g1 (10147) | −5.583,6 mV |
| LHAD1g1 (10147) | −1.809,5 mV | IN12B015 (801828) | −5.451,6 mV |

No circuito, a inibicao e dominada por **PVLP010** -- que e parte da propria via
visual. No conectoma inteiro entram **GNG300** (ganglio subesofagico) e
**SAD073**, que no circuito reduzido nao tinham como disparar porque suas
proprias entradas nao estavam la. E **CL367** mais que dobra (−6.283 → −14.362).

---

## Isto confirma o gate que o projeto ja tinha medido

`sim/inhibition_gate.py` mostrou que a inibicao no GF e um **gate**: a 3 Hz de
looming, 0 fugas com inibicao contra 7,1 sem. O achado atual e a versao
completa do mesmo fenomeno -- quanto mais da rede participa, mais forte o gate.

**O resultado zero-fuga fica.** Nao foi ajustado estimulo, limiar, peso nem
inibicao, e nenhum neuronio foi removido.

---

## Uma limitacao do modelo que so apareceu nesta escala

O potencial medio do GF no conectoma inteiro e **−298,9 mV**. O repouso e −52 mV.

O modelo de Shiu et al. e `dv/dt = (-(v - V_rest) + g) / tau` -- **nao tem
potencial de reversao inibitorio**, entao `v` pode descer indefinidamente. Num
circuito reduzido isso nunca importou: a media ficava em −50,4 mV. Com o
conectoma inteiro a inibicao empurra a membrana centenas de mV abaixo de
qualquer faixa fisiologica.

Isso **nao e bug de implementacao** -- e o modelo publicado sendo aplicado numa
escala em que sua simplificacao pesa. Vale registrar como limitacao conhecida:

- um neuronio a −299 mV precisa de muito mais excitacao pra voltar ao limiar do
  que um a −52 mV, entao o gate fica mais forte do que ficaria com um piso de
  reversao
- a magnitude do zero-fuga no whole CNS depende dessa ausencia de piso

Nao vamos adicionar um piso agora: seria mudar o modelo, e o modelo e de Shiu
et al., nao nosso. Mas qualquer conclusao sobre comportamento no conectoma
inteiro tem que carregar esta ressalva.

---

## Inputs ausentes: a rede esta silenciosa ou tem baseline artificial?

Verificado no codigo: o runtime integrado **nao aplica nenhuma entrada tonica**.
Nao ha `TONIC_INHIB_HZ` nem qualquer corrente basal em `sim/drosobot_lab.py` nem
em `sim/neural/`.

As unicas entradas externas sao os 311 LC4/LPLC2 com a taxa derivada da retina.
Todo o resto da atividade -- inclusive os 23.235 neuronios que dispararam na
corrida de 210 ms -- veio pela conectividade, a partir desses 311.

Isso responde a pergunta: **(A)**, as populacoes sem modelo sensorial estao
silenciosas ate receberem input da rede. Nao ha cascata a partir de baseline
artificial.

Vale contrastar com os experimentos de circuito antigos: la
`sim/connectome_model.py` aplica `TONIC_INHIB_HZ = 5.0` nos inibitorios, marcado
como ASSUMPTION. Essa suposicao fazia sentido no circuito reduzido, onde os
inibitorios nao tinham fonte. **No conectoma inteiro ela nao e usada, e nao
deve ser** -- ali eles tem fonte de verdade.

---

## O que continua em aberto

- O trace usa estimulo constante de 20 Hz por 400 ms. No laco fechado o looming
  e transitorio e mais fraco, e ai o whole CNS da 0 fugas em vez de 35 spikes.
  A diferenca entre "35 spikes" e "0 fugas" e o regime de estimulo, e nao foi
  mapeada.
- Nao medimos quantos passos o GF fica abaixo de −100 mV, nem se ha um regime de
  estimulo em que o whole CNS volta a disparar fuga.
- Nao sabemos se com potencial de reversao o resultado mudaria. Isso exigiria
  mudar o modelo.
