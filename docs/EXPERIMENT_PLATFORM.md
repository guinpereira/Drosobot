# A plataforma experimental

Como rodar experimentos reproduzíveis no Drosobot: escrever a configuração,
sair, voltar horas depois com resultados estruturados, comparação
circuito × conectoma inteiro, e replay na Unity.

---

## Em um comando

```bat
.venv-flygym2\Scripts\python -m sim.run_experiments sim\baterias\looming_lateral.json
```

Isso expande 3 condições × 2 escopos × 3 sementes = **18 corridas**, roda todas,
grava cada uma, e imprime a tabela final mais a comparação circuito × whole de
cada par.

Para ver o que rodaria sem rodar: `--seco`. Para listar as baterias prontas:
`--lista`.

---

## A receita

Tudo que decide o resultado num objeto só. Vai inteira para o `metadata.json`
de cada corrida, junto com o ambiente medido e o commit do repositório.

| campo | |
|---|---|
| `nome`, `versao` | identificam o experimento |
| `condicao` | o ponto dentro dele (`esquerda`, `rapido`…) |
| `seed` | semente do experimento — a realização de Poisson |
| `physics`, `neural` | backends |
| `escopo` | `whole` ou `circuit` |
| `duracao_s` | segundos de mosca |
| `arena` | `looming`, `optomotor`, `obstaculos`, `flat` |
| `estimulo` | parâmetros da condição, por arena |
| `colisao` | `legs` (padrão científico) |

A receita **recusa cedo** o que não existe: arena desconhecida, escopo
desconhecido, duração não positiva, e campo com nome errado — `semente` em vez
de `seed` levanta em vez de cair no padrão em silêncio. Uma bateria de horas que
morre na décima corrida por erro de digitação é pior que uma que recusa antes de
começar.

### O hash da ciência

`hash_ciencia()` resume, num valor, tudo que mudaria o resultado por razão
científica: as constantes de Shiu et al., o timestep, a escala de ponto fixo, a
transdução retina → taxa. É **lido do código**, não copiado — mudar `V_TH` em
`model.py` muda o hash, e há teste para isso.

Duas corridas com hashes diferentes **não são comparáveis**, e a análise recusa
compará-las. O backend não entra: trocar de GPU não muda resultado, e a
equivalência CPU/OpenCL é verificada em outro teste.

---

## O que cada corrida deixa

```
runs/2026-09-20_151038_looming_whole_centro_seed0/
    metadata.json    receita, ambiente, conectoma, hash, commit
    telemetry.jsonl  toda mensagem que saiu — a fonte do replay
    timeseries.csv   uma linha por janela neural, colunas fixas
    events.jsonl     só o raro: spike do GF, fuga, limitação do modelo
    summary.json     os agregados que uma tabela comparativa lê
```

Cinco arquivos porque cada um responde a uma pergunta diferente. Um spike do
Giant Fiber numa corrida de 600 s está perdido dentro de 60.000 linhas de série
temporal; em `events.jsonl` ele é uma linha.

---

## A análise circuito × whole

```python
from lab.analise import compara, texto
print(texto(compara(pasta_circuito, pasta_whole)))
```

Reporta, da entrada ao desfecho: spikes sensoriais, entrada máxima, excitação,
inibição, líquido, `v` mínimo, spikes do GF, spikes do TTMn, fugas. E a
contribuição no Giant Fiber **quebrada por população pré-sináptica**.

Exemplo real (0,3 s, semente 0, looming central):

```
grandeza                     circuito        whole    razao
spikes sensoriais                   8            8      1.00
entrada max (Hz)                 5.76         5.76      1.00
excitacao no GF (mV)            479.3        378.7      0.79
inibicao no GF (mV)            -320.9     -2,290.5      7.14
liquido no GF (mV)              158.4     -1,911.8    -12.07
spikes do GF                        4            1      0.25

circuito: 19 populacoes contribuem, maior e PVLP010 (-113,8 mV)
whole:    80 populacoes, maior e GNG300 (-1.135,2 mV)
```

**A entrada sensorial é idêntica** (8 spikes, 5,76 Hz) — é isso que torna a
diferença atribuível à rede, e não a estímulos diferentes. `GNG300` não existe
no subgrafo do circuito e sozinho responde pela maior parte da inibição a mais.

---

## Os experimentos

### Looming / Giant Fiber

A via publicada deste projeto: LC4/LPLC2 → DNp01 → TTMn. Esfera de 3 mm se
aproximando de 30 mm até 4 mm.

Condições: **azimute** (`looming_lateral.json`: esquerda / centro / direita) e
**velocidade** (`looming_velocidade.json`: ciclo de 0,4 / 0,8 / 1,6 s).

Com azimute 0 a geometria reduz exatamente ao caso já medido, então as corridas
centrais continuam comparáveis com o que foi publicado antes.

### Campo de obstáculos

Pilares fixos no chão, à esquerda / no centro / à direita. Usa a **mesma via de
looming**: um obstáculo que se aproxima é, para a retina, um estímulo em
expansão. A pergunta é se o reflexo de fuga guia desvio, e não só reação.

**Estes colidem de verdade** — e isso custou uma depuração. O FlyGym 2.x não usa
`contype`/`conaffinity`: todos os 69 geoms da mosca têm `(0, 0)`, e o contato
com o chão vem de 55 `<pair>` declarados um a um. Um pilar com `contype=1`
atravessa a mosca sem um único contato. Medido antes do conserto: 3 s de corrida,
retina vendo o obstáculo o tempo todo, **zero contatos**. Depois: 110 pares de
colisão, 8.270 contatos, e a mosca barrada.

### Optomotor — resultado negativo

Tambor de 12 postes girando em torno da mosca. Via própria: T4/T5 → HS → DNa02,
com os `bodyId` e os lados extraídos do conectoma
(`connectome/opto_roles.json`).

**A transdução funciona.** Fluxo óptico horizontal por mínimos quadrados
(Lucas–Kanade 1D sobre os omatídeos), medido com a mosca parada:

```
tambor +180°/s    fluxo  L -1,48   R -1,48
tambor -180°/s    fluxo  L +4,35   R +4,48
tambor parado     fluxo  L +0,05   R +0,09
```

O sinal inverte com a direção e é ~0 parado. Os dois olhos concordam, que é o
esperado: um tambor girando em torno do animal é rotação do campo (yaw), não
translação.

**A via não propaga.** Com o estímulo a 180°/s, os T4/T5 disparam, mas HS e
DNa02 ficam em **zero**, e não há giro. As duas direções produzem entradas
sensoriais diferentes (2 contra 5 spikes numa corrida de 1 s) e o mesmo desfecho
motor: nenhum.

**Nenhum parâmetro foi ajustado para produzir resposta.** Aumentar o ganho da
transdução até o DNa02 disparar produziria um "reflexo optomotor" fabricado por
nós, não medido no conectoma. O resultado fica como está.

Duas causas prováveis, nenhuma verificada:

1. A taxa sensorial é baixa demais para o limiar do LIF. O caminho tem três
   sinapses, e cada uma precisa somar sobre o decaimento de 20 ms.
2. Com a mosca **andando**, o fluxo óptico próprio abafa o do tambor: o fluxo
   médio cai de ~4,4 para ~0,2. O optomotor real é medido com a mosca presa, e
   este runtime não tem preparação presa.

Fica registrado como o resultado negativo do desvio de obstáculo já registrado
no README: medido, documentado, não corrigido.

---

## Replay na Unity

```bat
.venv-flygym2\Scripts\python -m sim.replay            # a corrida mais recente
.venv-flygym2\Scripts\python -m sim.replay --lista
.venv-flygym2\Scripts\python -m sim.replay <pasta> --velocidade 4
```

Depois, Unity → Play. O replay serve de volta, na mesma porta, **as mesmas
mensagens** que saíram da simulação, na mesma ordem e com o mesmo espaçamento.
Para a interface não há diferença entre ao vivo e gravado — nenhuma linha de C#
nova.

Isso importa cientificamente: o que aparece na tela é literalmente o que a
simulação produziu, não uma reconstrução a partir do resumo.

Validado: uma corrida de whole CNS reaberta mostrando mosca, arena, estímulo,
retina, CNS 3D, gate com `GNG300`, eventos e o aviso de limitação do modelo —
sem simular nada.

---

## Procedência, sempre

| | |
|---|---|
| **DATA** | conectoma: quem liga em quem, com que peso, com que neurotransmissor |
| **MODEL** | LIF de Shiu et al., regra de Dale, potencial de membrana, spikes |
| **ASSUMPTION** | transdução retina → taxa, ganho, teto, mapeamento motor, conjunto de colisão |
| **MODEL LIMITATION** | sem potencial de reversão inibitório: `v` vai a valores não fisiológicos no whole CNS, **sem clamp** |

O escopo aparece em todo `metadata.json` com o nome longo —
**Male CNS whole-connectome simulation** — e com o aviso:

> Whole-connectome simulation is NOT a complete functional brain: only the
> looming pathway (LC4/LPLC2) and the motor pathway (DNp01, TTMn) have modelled
> sensory/motor semantics.

Há teste para isso: o metadata não pode afirmar "cérebro completo" em lugar
nenhum.

---

## Escrever uma bateria nova

```json
{
  "nome": "minha_bateria",
  "duracao_s": 2.0,
  "seeds": [0, 1, 2, 3, 4],
  "escopos": ["circuit", "whole"],
  "condicoes": [
    {"condicao": "forte", "arena": "looming",
     "estimulo": {"ciclo_s": 0.4}, "notas": "por que esta condicao existe"}
  ]
}
```

Produto cartesiano de sementes × escopos × condições. Escrever as receitas à mão
convida a erro de digitação numa delas, e um experimento com um parâmetro errado
no meio é pior que um experimento que não rodou.

Uma corrida que falha é registrada com o erro e a bateria **continua** — horas
de trabalho não podem morrer na décima.
