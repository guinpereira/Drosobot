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
Unity são diretamente comparáveis aos do MuJoCo. Tórax a 1,083 significa
1,083 mm de altura nos dois.

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

### Por que dá pra converter antes de compor

A hierarquia da Unity compõe transforms já convertidos (pai em mundo, filho no
offset local) em vez de compor em MuJoCo e converter no fim. As duas ordens dão
o mesmo resultado porque a conversão `M` é uma conjugação:

```
M(R1·R2)M⁻¹ = (M R1 M⁻¹)(M R2 M⁻¹)
M(p + R·q)  = M p + (M R M⁻¹)(M q)
```

Ou seja, converter-depois-compor e compor-depois-converter coincidem. Foi
escolhido converter primeiro pra que `MujocoFrame` continue sendo o único lugar
com conhecimento dos eixos.

Isso vale pra rotação e para a posição, e é o que permite exportar os vértices
já convertidos (seção abaixo): `M(p + R·q)` com `q` sendo o vértice dá
`M p + (M R M⁻¹)·M q`, que é exatamente pai-convertido vezes vértice-convertido.

Cuidado: a igualdade **não** dispensa converter alguma das partes. Foi o que
aconteceu com os vértices — converter duas das três (posição e rotação) e
deixar a terceira crua não é composição, é mistura de espaços.

## Os vértices das malhas também são convertidos

`tools/export_fly_mesh.py` grava os OBJ **já em eixos da Unity**: cada vértice
sai como `(x, z, y)`, e a ordem dos índices de cada face é invertida junto,
porque a troca de eixo é uma reflexão e inverteria as normais.

Isso não era feito, e foi o bug mais difícil de ver da rodada. As posições e as
rotações eram convertidas; os vértices não. Cada malha ficava pendurada num
transform já convertido mas com a geometria em eixos do MuJoCo — ou seja, girada
90° sobre a própria origem. Num tórax quase isotrópico ninguém nota. Num tarso,
que tem 0,11 mm e é alongado, a peça sai inteira de onde deveria estar e a perna
aparece desmontada no ar.

Como foi diagnosticado, já que os números não pegaram: os tamanhos da malha
importada pela Unity foram comparados com os do MuJoCo. `fly/lf_tibia` dava
`(0,0852, 0,1114, 0,5520)` nos dois — **idênticos**, quando o correto seria
`(0,0852, 0,5520, 0,1114)` com Y e Z trocados. Igualdade era a prova de que a
conversão não tinha acontecido.

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

O modo **Bind Pose** usa o campo `pose_repouso` do `fly_body.json`: os `xpos` e
`xquat` que o MuJoCo tem depois de `mj_forward` na `qpos` padrão do modelo, em
**mundo**. É a mesma grandeza que a telemetria manda por quadro, então a pose de
repouso e a pose viva percorrem exatamente o mesmo caminho de código.

**Não é composto da hierarquia de corpos.** Já foi, e estava errado: `body_pos`
e `body_quat` são a árvore cinemática com as **juntas em zero**, e as 73 `qpos`
do padrão do NeuroMechFly são todas não-nulas (postura de pé). Compor a árvore
ignorando as juntas erra até **1,77 mm** numa mosca de 2,7 mm — as pernas descem
coladas na linha média em vez de abrirem.

Números da pose exportada (mm, eixos do MuJoCo):

```
             coxa z   fêmur z  tíbia z  tarso5 z  tarso5 y  tarso5 x
  lf          0,863    0,539    0,810     0,065     0,997     1,339
  lm          0,634    0,458    0,981     0,062     1,469     0,320
  lh          0,632    0,460    1,058     0,085     1,048    -1,474
  rf/rm/rh    espelhados; |y_L + y_R| ≤ 0,0002

  tórax z = 1,083        seis tarsos em z = 0,071 ± 0,010
  tórax 1,012 mm acima do plano dos tarsos
```

O que isso confirma: os seis tarsos estão **no chão** (desvio de 10 µm entre
eles), o corpo está suspenso 1 mm acima, e as pernas abrem ~1,17 mm de cada
lado, com o tripé na geometria certa — dianteiras à frente (x = +1,34),
medianas ao lado (x = +0,32), traseiras atrás (x = −1,47).

Serve como **referência fixa**: se a mosca parece errada durante a corrida mas
certa em Bind Pose, o problema está na telemetria ou na interpolação, não na
montagem, nas malhas nem na conversão de eixos.

### Validação visual

Só números não bastaram nesta rodada. A composição errada da árvore passava em
todas as verificações numéricas que eu tinha — a cadeia coxa > fêmur > tíbia >
tarso continuava descendo, e a simetria esquerda/direita era exata — e mesmo
assim o render mostrava a mosca desmontada. As duas falhas (árvore sem juntas e
vértices não convertidos) só apareceram na imagem.

`Assets/Editor/CaptureBodyShots.cs` renderiza as quatro vistas em edit mode, sem
Play e sem telemetria:

```
Unity.exe -batchmode -quit -projectPath unity\DrosobotLab ^
  -executeMethod Drosobot.EditorTools.CaptureBodyShots.Capturar ^
  -logFile unity\capture.log
```

Saída em `docs/images/fly_bindpose_{perspectiva,lateral,topo,frente}.png`. A
referência de comparação é o render do próprio MuJoCo pelo `mujoco.Renderer`, do
mesmo modelo compilado.

O OBJ leva normais por vértice, calculadas na exportação por média das faces
ponderada pela área. Sem elas a Unity deduz normais com um ângulo de suavização
fixo e, como as malhas do NeuroMechFly são decimadas (~2.000 faces por peça), o
resultado ficava facetado. A normal vem da **geometria**, não é aparência
inventada — é a mesma superfície que a física usa, só sombreada direito.

### Cor é outra coisa

O modelo que a física roda **não traz cor**: os 69 geoms têm `rgba`
(0,5 0,5 0,5 1) e `matid = -1` — o modelo inteiro tem `nmat = 1`, e esse
material é o `grid` do chão da arena.

Há três esquemas, e o painel CORPO diz qual está ligado e de onde ele veio:

| esquema | origem | procedência |
|---|---|---|
| **Clay** | o cinza 0,5 que o modelo carrega | DATA |
| **Flybody** | valores de `flygym/assets/model/flybody/fruitfly.xml` (`body`, `lower`, `brown`, `membrane`, `red`), atribuídos por anatomia | ASSUMPTION |
| **Drosophila** | escolhido a olho pra leitura no fundo escuro | ASSUMPTION |

O Flybody fica ao lado do Drosophila de propósito: é o único com origem
rastreável, e apagá-lo pra deixar só o que "parece melhor" jogaria fora a única
referência externa que existe.

Os dois esquemas de cor compartilham a mesma **estrutura**: banda do abdômen e
escurecimento distal das pernas são derivados das cores base, não listados um a
um. Só a cor muda.

Nenhuma dessas cores codifica grandeza: não há contato, ativação nem força sendo
pintada no corpo.

Nervuras de asa não foram feitas. No `fruitfly.xml` elas são um geom separado
por cima da membrana; no modelo do NeuroMechFly a asa é uma malha só, sem
nervura na geometria e sem UV. Desenhá-las seria inventar anatomia numa imagem
que as pessoas vão ler como sendo o modelo.

## Eixos desenhados no modo Eixos

As vistas do painel CORPO seguem daí. A câmera fica em
`target + Euler(pitch, yaw, 0) · (0, 0, −d)`, então `yaw = 0` a põe em −Z
olhando para +Z. Como +X é a frente da mosca e +Z é o lado dela, **lateral é
`yaw = 0`, não 90**; frente é `yaw = −90`; topo é `pitch = 89`.

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
