# Auditoria dos pares de colisao do NeuroMechFly

87% do passo de fisica e narrowphase de malha
(`NEUROMECHFLY_PHYSICS_FEATURES.md`). Este documento classifica os 2220 pares,
mede quais de fato tocam, e testa o que acontece quando se reduz o conjunto.

Scripts: `benchmarks/physics/collision_pruning/`.

---

## 1. Classificacao

Modelo: `flygym 1.2.1`, `Fly(self_collisions="legs")` (o padrao, e o que o
Drosobot roda), `timestep=1e-4`.

| classe | pares | % |
|---|---|---|
| contralateral (pernas de lados opostos) | 1152 | 51,9% |
| ipsilateral vizinha (pernas do mesmo lado, adjacentes) | 512 | 23,1% |
| ipsilateral distante | 256 | 11,5% |
| mesma perna, segmentos nao adjacentes | 252 | 11,4% |
| perna x chao | 48 | 2,2% |
| **mesma perna, segmentos adjacentes** | **0** | -- |

A ultima linha e informacao util: o FlyGym **ja** exclui segmentos ligados por
junta (`fly.py:720-741` cuida da relacao pai/filho). Essa poda obvia ja foi
feita pelo upstream; nao ha ganho facil ali.

---

## 2. O que de fato toca

Corrida de 2,0 s (20.000 passos), caminhada reta, semente 0. Pra cada par,
registramos contatos reais e uma **cota inferior** da distancia entre
superficies (distancia entre centros menos os dois raios envolventes). Por ser
cota inferior, ela nunca superestima a folga: se diz que ha espaco, ha espaco.

| classe | pares | tocaram |
|---|---|---|
| contralateral | 1152 | **2** |
| ipsilateral vizinha | 512 | **0** |
| ipsilateral distante | 256 | **0** |
| mesma perna distante | 252 | **0** |
| perna x chao | 48 | **17** |

**19 pares de 2220 tocaram.** Os dois nao-chao foram `LHCoxa x RHCoxa` (coxas
traseiras contralaterais), por 6 passos.

E a cota confirma que nao e so "nao foi observado":

| exigencia de folga | pares que nunca chegaram perto | conflitos |
|---|---|---|
| > 0,00 mm | 2082 (93,8%) | **0** |
| > 0,25 mm | 1886 (85,0%) | **0** |
| > 1,00 mm | 1054 (47,5%) | **0** |

Zero conflitos: nenhum par com cota positiva registrou contato. Pra 2082 pares,
contato era **geometricamente impossivel** durante a corrida, nao apenas ausente.

Os 48 pares com o chao aparecem com cota absurda (−99 mm) porque o plano do chao
tem extensao enorme e esfera envolvente nao significa nada pra ele. Chao nunca
entra em poda.

---

## 3. Uma tentativa que falhou, e por que fica registrada

A primeira tentativa desligou pares **depois** do modelo compilado, com
`m.pair_margin[k] = -1e6`. O resultado foi rejeitado por tres sinais:

```
COL_NARROW 12.431 ms (189.4%)      -- impossivel: mais que o passo inteiro
ganho de passo 0.10x               -- ficou 10x MAIS LENTO
ncon medio 11.12 nas tres configs  -- nenhum par foi desligado
```

Duas causas. `margin` **filtra o resultado** do narrowphase, nao impede o
narrowphase de rodar -- entao nao havia ganho a ter por ali. E os timers do
MuJoCo lidos por cima de `sim.step()` nao fecham em 100%, porque o wrapper do
FlyGym faz mais do que um `mj_step`; daí os 189%.

O numero acima de 100% deveria ter sido parada imediata. Ficou aqui pra ninguem
repetir o caminho, e o perfil passou a ser medido sobre `mj_step` puro.

---

## 4. O jeito certo, e o resultado

O botao suportado e a lista `self_collisions`, de onde `Fly.init_self_contacts()`
gera os pares (`fly.py:715`). Tres configuracoes, mesma semente, mesma pose,
mesmo drive, mesma arena, mesmo timestep, 2,0 s:

| config | pares | `mj_step` | COL_NARROW | ganho |
|---|---|---|---|---|
| REFERENCE (`"legs"`) | 2220 | 1,757 ms | 1,521 ms (86,6%) | -- |
| TARSI (`"tarsi"`) | 870 | **0,165 ms** | 0,069 ms (41,8%) | **10,66x** |
| NONE (`"none"`) | 48 | **0,091 ms** | 0,005 ms (5,1%) | **19,40x** |

`NONE` nao e proposta -- e o **limite superior** do ganho por esta via.

### A trajetoria

| config | avanco | desvio pos max | qpos max | orientacao max | ncon medio |
|---|---|---|---|---|---|
| REFERENCE | 25,48 mm | -- | -- | -- | 11,12 |
| TARSI | 25,48 mm | **0,0000 mm** | **0,000000 rad** | **0,000 deg** | 11,11 |
| NONE | 25,48 mm | **0,0000 mm** | **0,000000 rad** | **0,000 deg** | 11,11 |

**Identica.** Nao "dentro da tolerancia": identica nas amostras medidas.

Faz sentido: os pares removidos nunca produziram contato, entao nunca entraram
numa forca. `ncon` cair de 11,12 pra 11,11 e a unica diferenca, e e a media sobre
amostras.

---

## 5. O detalhe que muda a leitura

O `mj_step` ficou **10,66x** mais rapido. O passo completo, so **1,37x**
(2525,9 → 1844,7 us).

Porque o gargalo mudou de lugar. Com a colisao em 165 us, o que domina passa a
ser o **wrapper do FlyGym 1.x** -- a observacao completa montada a cada passo,
com lacos Python sobre contatos e indexacao por nome do `dm_control`, medida em
39% do passo em `FLYGYM2_ARCHITECTURE.md`.

Os dois achados se encaixam exatamente:

```
hoje                2526 us/passo   (colisao domina)
so podando          1845 us/passo   (wrapper domina)
podando + FlyGym 2.x   ~?           (a medir)
```

Nenhum dos dois precisa de GPU.

---

## 6. O que NAO esta provado

- **Um comportamento so.** Caminhada reta, semente 0, 2 s, terreno plano. Curva
  fechada, fuga de ré e campo de obstaculos poem as pernas em poses que esta
  corrida nao visitou. Antes de virar padrao, a auditoria tem que rodar nos tres
  experimentos do Drosobot e em varias sementes.
- **`"tarsi"` nao e a poda da auditoria.** E um preset do FlyGym que calhou de
  cortar bem. A poda por par medido (1886 pares com folga > 0,25 mm) precisaria
  de construcao propria da lista -- e, pelos numeros, `"tarsi"` ja entrega quase
  todo o ganho disponivel com uma opcao suportada.
- **Auto-colisao tem funcao.** Ela impede pernas atravessarem umas as outras em
  poses extremas. Em caminhada reta isso nunca foi exercitado; num tropeco pode
  ser. E por isso que `NONE` fica como limite de medicao, nao como recomendacao.

---

## 7. Recomendacao

Rodar a auditoria nos tres experimentos e em pelo menos tres sementes. Se
`"tarsi"` continuar produzindo trajetoria identica em todos, adotar como padrao
e **mostrar na interface qual conjunto de colisao esta ativo** -- passa a ser
parte da definicao do modelo, nao detalhe de implementacao.

Enquanto isso nao acontece, `"legs"` continua sendo o padrao.
