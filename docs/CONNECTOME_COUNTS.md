# Contagem do conectoma — a fonte de verdade

Havia dois números circulando: **164.451** e **164.249**. Os dois estão certos e
medem coisas diferentes. Este documento diz qual é qual, e qual é o número que o
runtime, a telemetria, a Unity e a documentação usam.

> **Neurônios simulados: 164.451. Arestas dirigidas: 25.550.583.**
> É isso que aparece em toda parte.

---

## De onde cada número vem

| | valor | o que é |
|---|---|---|
| segmentos do dataset | **88.384.522** | corpos segmentados no volume, antes de qualquer filtro |
| arestas entre segmentos | 151.856.684 | |
| sinapses entre segmentos | 311.833.243 | |
| **neurônios anotados** | **164.451** | segmentos que passam no filtro de anotação |
| **arestas dirigidas** | **25.550.583** | pares (pré, pós) em que os dois lados são neurônios anotados |
| sinapses anatômicas | 123.967.037 | soma dos pesos dessas arestas |
| sinapses por aresta | 4,85 | |
| neurônios **com** pelo menos uma aresta | 164.249 | |
| neurônios **isolados** | **202** | anotados, mas sem nenhuma aresta sobrevivente |

```
164.451 neurônios anotados
  − 202 isolados
= 164.249 neurônios no grafo conectado
```

Medido diretamente no CSR carregado, não deduzido:

```
n no CSR                  164451
sem aresta de SAIDA         1179
sem aresta de ENTRADA        313
ISOLADOS (nem in nem out)    202
n − isolados              164249
```

## Por que a diferença existia

Dois scripts contaram de formas diferentes, e nenhum estava errado:

- `benchmarks/neural/build_csr.py` monta o grafo sobre **todos os neurônios
  anotados** e reporta `164.451`. É esse CSR que o runtime carrega.
- `benchmarks/neural/audit_connectome_graph.py` conta
  `union(unique(pre), unique(post))` — só quem aparece em alguma aresta — e
  reporta `164.249` em `male_cns_graph_stats.json`.

O campo `"neurons": 164249` daquele arquivo de estatísticas significa
**"neurônios no grafo conectado"**. Ele fica como está: é um registro de
medição, com data, e reescrever medição antiga para caber numa narrativa é
pior que explicar a diferença.

## Critério do filtro de anotação

`build_csr.py`, sobre `body-annotations.feather`:

```python
sel  = statusLabel ∈ {"Roughly traced", "Reviewed",
                      "Prelim Roughly traced", "RT Hard to trace"}
sel &= superclass != "glia"
```

Uma aresta sobrevive quando **os dois lados** estão nesse conjunto.

Sinal por neurônio (regra de Dale): `consensus_nt`, caindo para `predicted_nt`
só quando a curada está vazia — a mesma ordem de preferência de
`sim/connectome_model.py`. GABA e glutamato são inibitórios.

| | |
|---|---|
| arestas excitatórias | 15.164.662 |
| arestas inibitórias | 9.799.540 |
| arestas mudas (NT desconhecido) | 586.381 |
| neurônios sem NT | 3.125 |

## Por que os 202 isolados continuam sendo simulados

Eles são integrados pelo LIF a cada passo como qualquer outro: decaem para o
repouso e nunca disparam, porque nada chega neles e eles não alcançam ninguém.

Tirá-los mudaria `n` de 164.451 para 164.249 e **não mudaria uma única
trajetória** — mas mudaria o mapeamento índice↔bodyId, invalidaria os
`gf_roles.json` e os benchmarks já gravados, e economizaria 0,12% da memória de
estado. Não vale.

O número honesto de "neurônios simulados" é o que o integrador de fato
integra: **164.451**.

## Aresta ≠ sinapse, segmento ≠ neurônio

**Aresta ≠ sinapse.** A unidade computacional é a aresta agregada por par de
neurônios. As ~124M sinapses anatômicas estão condensadas no peso dela (4,85 por
aresta, em média). Expandir de volta multiplicaria a memória por cinco e não
mudaria a dinâmica do modelo — o LIF soma peso, não conta sinapse.

**Segmento ≠ neurônio.** 88,4M segmentos, 164.451 neurônios anotados. Dizer
"88 milhões de neurônios" seria errado por um fator de 537.

## Onde estes números aparecem

| lugar | campo |
|---|---|
| runtime | `--info`, e `engine.resumo()["neurons_simulated"]` |
| telemetria | `experiment_info.runtime.neurons_simulated` / `edges_simulated` |
| Unity | painel RUNTIME, linha `Neurons ... simulados` |
| metadados do CSR | `connectome/csr/meta.json` → `neurons`, `edges` |
| docs | `README.md`, `docs/RUNNING_THE_LAB.md`, este arquivo |

O painel RUNTIME mostra, logo abaixo, quantas morfologias existem na tela. É
outro número, com outra procedência (vem do asset da Unity, não do runtime), e
somar os dois não significa nada.

## Fonte

Janelia neuPrint **`male-cns:v1.0`**, via
`connectome-weights-male-cns-v1.0-minconf-0.5.feather` (limiar de confiança de
sinapse 0,5, aplicado a montante por quem publicou o dump).
