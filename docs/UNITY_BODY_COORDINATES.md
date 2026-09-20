# Coordenadas do corpo: MuJoCo -> Unity

Como a pose da mosca sai da física e chega na tela, e por que a conversão mora
num arquivo só.

Código: [`MujocoFrame.cs`](../unity/DrosobotLab/Assets/Scripts/Lab/MujocoFrame.cs).
Todo o resto — `FlyBody`, `LabBootstrap` — chama ele. **Nenhum outro arquivo
troca eixo.** Isso não é preferência de estilo: converter em dois lugares foi o
que produziu o bug do deslocamento duplicado descrito no fim.

---

## Os dois sistemas

| | MuJoCo (física) | Unity (tela) |
|---|---|---|
| quiralidade | destro | canhoto |
| "pra cima" | +Z | +Y |
| quatérnio | `(w, x, y, z)` | `(x, y, z, w)` |
| unidade | milímetro | 1 unidade = 1 mm (`FlyBody.escala = 1`) |
| frente da mosca | +X | +X |

Nenhuma escala é aplicada. O modelo do NeuroMechFly já está em milímetros e o
Lab adota 1 unidade Unity = 1 mm, então os números que aparecem no inspetor da
Unity são diretamente comparáveis aos do MuJoCo. Tórax a 2,1 significa 2,1 mm de
altura nos dois.

## Posição

```
Unity(x, y, z) = MuJoCo(x, z, y)
```

Trocar Y e Z faz duas coisas ao mesmo tempo: leva o "pra cima" do Z pro Y, e
inverte a quiralidade — que é exatamente o que a passagem de destro pra canhoto
exige. Não existe um segundo passo de espelhamento; quem adicionar um vai
espelhar a mosca.

## Rotação

```
Unity(x, y, z, w) = (-mx, -mz, -my, mw)      # entrada MuJoCo (mw, mx, my, mz)
```

A parte vetorial sofre a mesma troca Y↔Z da posição, e depois é **negada**. A
negação é necessária porque a troca de eixo é uma reflexão (determinante −1):
sob ela um vetor comum se transforma direto, mas um pseudovetor — que é o que o
eixo de rotação é — troca de sinal. Sem a negação a mosca gira ao contrário.

Isso passa despercebido no tórax, que fica quase alinhado, e aparece nas pernas,
que têm rotações grandes: as tíbias apontam pro lado errado e a mosca parece
"desmontada" em vez de "invertida".

### Por que dá pra compor a bind pose já no espaço da Unity

`FlyBody.CalculaBindPose` acumula a cadeia de corpos depois de converter cada
transform local, em vez de acumular em MuJoCo e converter no fim. As duas coisas
dão o mesmo resultado porque a conversão `M` é uma conjugação:

```
M(R1·R2)M⁻¹ = (M R1 M⁻¹)(M R2 M⁻¹)
M(p + R·q)  = M p + (M R M⁻¹)(M q)
```

Ou seja, converter-depois-compor e compor-depois-converter coincidem. Foi
escolhido converter primeiro pra que `MujocoFrame` continue sendo o único lugar
com conhecimento dos eixos.

## Transforms estáticos das malhas

`tools/export_fly_mesh.py` exporta, pra cada geom, o `pos`/`quat` **local** dele
dentro do corpo, tirado do modelo **compilado** (não do XML fonte — o compilador
do MuJoCo resolve `fromto`, defaults e referências de mesh, e o que a física usa
é o resultado dele).

Na Unity isso vira hierarquia:

```
NeuroMechFly            (raiz, SEMPRE identidade)
  c_thorax              <- pose de MUNDO, da telemetria
    fly/c_thorax        <- offset local do geom, fixo, do JSON
  lf_coxa               <- pose de MUNDO
    fly/lf_coxa         <- offset local do geom, fixo
  ...
```

O segmento recebe a pose viva; o filho carrega o offset da malha e nunca se
mexe. Essa separação é o que faz a mesma hierarquia servir pra pose viva e pra
bind pose sem código duplicado.

## Espaço da pose: MUNDO

A telemetria (`body_pose` no `frame`) manda, pra cada segmento, o transform em
**mundo** — é o que o MuJoCo tem em `data.xpos`/`data.xquat`. Consequência:

> **A raiz visual (`NeuroMechFly`) tem que ficar na identidade.**

Se alguém mover a raiz para a posição global da mosca *e* os segmentos já
estiverem em mundo, o deslocamento entra duas vezes e a mosca some pra longe da
câmera. Aconteceu. Hoje:

- `MujocoFrame.RaizNeutra` confere a raiz todo quadro e loga `LogError` com o
  valor encontrado, em vez de deixar o sintoma como "a mosca está estranha";
- `LabBootstrap` só move `_fly` quando as malhas reais **não** estão montadas
  (o marcador de cápsula), guardado por `temMoscaReal`.

O `enum EspacoPose` existe pra tornar a escolha explícita no código. Hoje só
`Mundo` está em uso; `Local` está declarado porque é a alternativa real (mandar
ângulos de junta em vez de poses globais), e deixar a suposição sem nome foi
justamente o que permitiu o bug.

## Bind pose

O modo **Bind Pose** monta a mosca a partir do campo `bodies` do
`fly_body.json`: o transform local de cada corpo em relação ao pai, acumulado da
raiz pra baixo. É a pose de repouso do **modelo**, com todas as juntas em zero.

**Não é a postura de andar e não é telemetria.** Números medidos dela:

```
tórax z = 2,100 mm          cabeça (olho) z = 2,118 mm
                coxa      fêmur     tíbia    tarso5    y tarso5
  lf            1,870     1,505     0,800    -0,282      0,167
  lm            1,618     1,437     0,653    -0,621      0,124
  lh            1,603     1,404     0,568    -0,814      0,087
  rf/rm/rh      espelhados exatamente (|y_L + y_R| = 0,0000 nos três pares)
```

O que isso confirma: a cadeia desce monotonicamente (coxa > fêmur > tíbia >
tarso) nas seis pernas, a simetria esquerda/direita é exata, e o tórax fica
2,67 mm acima do plano médio dos tarsos. Nessa pose as pernas ficam recolhidas
perto da linha média (|y| < 0,2 mm) porque as juntas estão em zero — é o
esperado pro repouso do modelo, não um defeito de montagem.

Serve como **referência fixa**: se a mosca parece errada durante a corrida mas
certa em Bind Pose, o problema está na telemetria ou na interpolação, não na
montagem nem na conversão de eixos.

## Eixos desenhados no modo Eixos

Vermelho/verde/ciano = X/Y/Z **da Unity**, já convertidos. Não são os eixos do
MuJoCo. Comparar a seta verde ("pra cima" na Unity) com o +Z do MuJoCo é
comparar a mesma direção física com dois nomes.

Os segmentos com eixo são o tórax e a cadeia `coxa → trochanterfemur → tíbia →
tarsus1` das seis pernas. Os 69 juntos viram um novelo ilegível.

## Nomes de segmento

O modelo compilado prefixa os corpos com o nome do agente: `fly/c_thorax`. A
pose chega sem prefixo: `c_thorax`. `FlyBody.Normaliza` corta nos dois lados; se
cortasse só num, nada casava e a mosca ficava parada sem erro nenhum.

Os nomes são minúsculos com underscore — `lf_coxa`, `lf_trochanterfemur`,
`lf_tibia`, `lf_tarsus1..5` — e **não** a forma de artigo (`LFCoxa`). No
NeuroMechFly trocânter e fêmur são um corpo só, daí `trochanterfemur`.

Prefixos: primeira letra `l`/`r` = lado, segunda `f`/`m`/`h` = perna dianteira /
mediana / traseira.
