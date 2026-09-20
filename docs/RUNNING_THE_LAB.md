# Rodar o Drosobot Lab

O laboratório são **dois processos**: a simulação em Python e a visualização na
Unity. Eles conversam por dois sockets locais e por mais nada.

```
   Python                                            Unity
   ──────                                            ─────
   sim/drosobot_lab.py  ──── telemetria 8765 ───▶   Drosobot Lab
                        ◀─── controle   8766 ────
```

A telemetria é **mão única**: a Unity só lê. A única coisa que volta é o canal
de controle, e ele escolhe experimento e aperta start/pause/reset — não existe
comando que altere peso, limiar, entrada ou drive.

---

## O comando

```bat
.venv-flygym2\Scripts\python sim\drosobot_lab.py ^
    --physics flygym2 ^
    --neural opencl ^
    --cns whole ^
    --experiment looming ^
    --telemetry ^
    --duracao 600
```

Depois: abrir a Unity em `unity/DrosobotLab` e dar **Play**.

A ordem não importa. A Unity reconecta sozinha enquanto a simulação não estiver
no ar, e o painel CORPO mostra `DESCONECTADO / CONECTANDO / RODANDO / PARADO`
com a idade da última telemetria — se o Python morrer, a Unity **para de
afirmar** que está rodando em vez de continuar desenhando o último quadro.

`--duracao 0` roda até a interface mandar parar. Com `--espera`, a simulação
sobe mas não começa: quem escolhe o experimento é o painel EXPERIMENTO.

### O player standalone

Para rodar sem abrir o Editor (é o que a validação visual usa):

```bat
Unity.exe -batchmode -quit -projectPath unity\DrosobotLab ^
  -executeMethod Drosobot.EditorTools.BuildLabPlayer.Build ^
  -logFile unity\buildplayer.log

unity\DrosobotLab\Build\DrosobotLab.exe
```

Com `-autocapture <segundos> -capturedir <pasta>` ele conduz uma sessão sozinho
— troca de vista, de modo e de paleta — e fotografa. As imagens em
`docs/images/live/` saíram assim, contra a simulação de verdade.

### Só olhar o que seria montado

```bat
.venv-flygym2\Scripts\python sim\drosobot_lab.py --info
.venv-flygym2\Scripts\python sim\drosobot_lab.py --lista
```

`--info` carrega o conectoma, sobe pra GPU, imprime o que achou e sai. É o teste
mais rápido de que a máquina está pronta.

### Opções que importam

| opção | padrão | o que faz |
|---|---|---|
| `--physics` | `flygym2` | `flygym2` é o caminho principal; `flygym1` é a referência |
| `--neural` | `auto` | `opencl`, `cpu`, `d3d12`. `auto` prefere GPU e cai pra CPU |
| `--cns` | `whole` | `whole` = 164.451 neurônios; `circuit` = 1.261 |
| `--experiment` | `looming` | `looming`, `flat`, ou um id de `--lista` |
| `--colisao` | `legs` | padrão científico validado. `tarsi` **reprovou** em curva |
| `--duracao` | `2.0` | segundos de mosca; `0` = até mandarem parar |
| `--seed` | `0` | semente do experimento (realização de Poisson) |
| `--espera` | — | não começa sozinho; espera a interface |
| `--sem-controle` | — | não abre o canal 8766 |

---

## Os dois ambientes Python

Existem dois de propósito, e nenhum substitui o outro.

| | `.venv` | `.venv-flygym2` |
|---|---|---|
| papel | **referência / regressão** | **desenvolvimento** |
| Python | 3.11.9 | 3.14.4 |
| FlyGym | 1.2.1 | 2.1.0 |
| MuJoCo | 3.2.7 | 3.9.0 |
| numpy | 2.0.2 | 2.5.3 |
| Brian2 | 2.9.0 | — |
| pyopencl | 2026.1.4 | 2026.1.4 |

`.venv` tem o Brian2, que é a referência contra a qual o LIF próprio foi
verificado, e o FlyGym 1.x que produziu os resultados publicados no README.
**Não destruir.** `.venv-flygym2` não tem Brian2: o teste
`test_parametros_batem_com_o_modelo` falha lá por isso, e é esperado — ele roda
no `.venv`.

```bat
py -3.11 -m venv .venv
.venv\Scripts\pip install -r requirements.txt

py -3.14 -m venv .venv-flygym2
.venv-flygym2\Scripts\pip install -e research\upstream\flygym pyopencl
```

---

## O que aparece na tela

**Área principal** — a mosca do NeuroMechFly (malhas reais, exportadas do modelo
compilado que a física roda), a arena, e o estímulo de looming.

**Coluna da esquerda**

- **DROSOBOT LAB** — SIM / WALL / RTF, atividade por população, e os eventos
  científicos. `looming_start` e `looming_end` são bordas, não estado: emitir a
  cada janela enchia a lista de 100 linhas por segundo e empurrava para fora
  justamente os eventos raros — spike do GF, fuga, limitação do modelo.
- **ATIVIDADE DE POPULAÇÃO** — quem participa sem ter morfologia individual.
- **GIANT FIBER — GATE** — excitação, inibição, líquido, `v` mínimo, spikes, e
  os maiores contribuintes inibitórios do passo, rotulados por tipo
  (`SAD073`, `CL367`, `LHAD1g1`…). Abaixo, o aviso de limitação do modelo
  quando `v` cruza −150 mV.
- **VISÃO** (segunda coluna) — os 721 omatídeos de cada olho.

**Coluna da direita**

- **EXPERIMENTO** — primeiro, porque é o único painel com que se *age*: escolha
  do experimento, semente, Start / Pause / Reset / Stop, e o estado da corrida.
- **MALE CNS** — o conectoma 3D numa caixa, com as sinapses pulsando. Câmera
  própria: antes ele dividia câmera com a mosca e saía de quadro quando ela
  andava.
- **CORPO** — estado da conexão com a idade da telemetria, modos
  `Normal / Bind Pose / Eixos / Rótulos`, vistas `Persp / Lado / Topo / Frente`,
  botão Focar, e as paletas `Clay / Flybody / Drosophila`.
- **RUNTIME** — OS, CPU, GPU, backend de física, backend neural, pares de
  colisão, neurônios simulados, morfologias na tela, arestas, VRAM, custo por
  etapa e RTF.
- **PROCEDÊNCIA** e **INSPECTOR**.

**Rodapé** — sinais em janela rolante.

As colunas rolam quando não cabem, em vez de cortar o último painel em
silêncio.

### Teclas

`C` liga o fluxo de sinapses · `F` troca o filtro de arestas · `N` liga/desliga
neurônios · `B` liga/desliga a casca · `P` modo apresentação · `Esc` limpa a
seleção · botão esquerdo orbita · roda aproxima · botão do meio empurra.

---

## Desempenho medido

RX 6700 XT (gfx1031, 20 CUs, 12 GiB), Windows 11, FlyGym 2.1.0 / MuJoCo 3.9,
OpenCL, looming, `legs`, whole CNS, 1 s de mosca:

| etapa | baseline | fastpath | hoje |
|---|---|---|---|
| física | 14.405 · 78,9% | 5.167 · 58,3% | **3.328 · 67,8%** |
| neural | 3.363 · 18,4% | 3.068 · 34,6% | **916 · 18,7%** |
| visão | 14 | 14 | 15 |
| leitura | 220 | 174 | 179 |
| telemetria | 191 | 191 | 208 |
| **total** | 18.266 | 8.860 | **4.907** |
| **RTF** | 0,055× | 0,113× | **0,204×** |

Unidade: **ms de relógio por segundo simulado** — a única base em que as etapas
se somam. A seção "por chamada" do profiler é outra grandeza e não soma.

**3,7× no total, sem tocar em ciência**: mesmo timestep, mesmo modelo, mesmo
controlador, mesmo conectoma, mesmos parâmetros. O comportamento é idêntico —
mesmas fugas, mesmos spikes do Giant Fiber, mesma posição final.

O ganho veio de remover trabalho repetido, não de calcular menos:
`sim/physics/fastpath.py` e `native_fastpath.py` no lado da física, e no neural
o scatter por frontier, os argumentos de kernel fixados e a máscara esparsa.
Igualdade bit a bit verificada em `tests/test_fastpath_equivalencia.py`.

Ver [`END_TO_END_PERFORMANCE.md`](research/END_TO_END_PERFORMANCE.md) e
[`PHYSICS_OVERHEAD.md`](research/PHYSICS_OVERHEAD.md).

**O `mj_step` é agora o maior item isolado do laço de física** — 39,5%, contra
11% no começo. Não porque ficou mais lento, mas porque tudo em volta encolheu.
O que sobra é física de verdade.

---

## Números do conectoma

| | |
|---|---|
| dataset | Janelia neuPrint `male-cns:v1.0` |
| neurônios anotados | 164.451 |
| arestas dirigidas agregadas | 25.550.583 |
| sinapses anatômicas | 123.967.037 |
| sinapses por aresta | 4,85 |

**Aresta ≠ sinapse.** A unidade computacional é a aresta agregada por par de
neurônios; as ~124M sinapses anatômicas estão condensadas nela, no peso.
Expandir de volta não mudaria a dinâmica do modelo e multiplicaria a memória
por cinco.

**Segmento ≠ neurônio.** O dataset tem 88,4M segmentos; 164.451 são neurônios
anotados. É deles que o grafo é feito.

---

## Limitações que não são bugs

**O conectoma inteiro simulado não é um cérebro funcional completo.** Só a via
de looming (LC4/LPLC2) e a motora (DNp01, TTMn) têm semântica sensorial/motora
modelada. O resto participa pela conectividade. O runtime reporta as duas coisas
separadas — em `--info`, na telemetria e no painel.

**Potencial de membrana muito negativo no whole CNS.** O LIF de Shiu et al. não
tem potencial de reversão inibitório, então inibição convergente forte leva `v` a
valores não fisiológicos. **Não há clamp**: o valor vai como o modelo produziu, e
a interface mostra um aviso quando cruza −150 mV. Ver
[`docs/research/WHOLE_CNS_GIANT_FIBER.md`](research/WHOLE_CNS_GIANT_FIBER.md).

**O gate do GF não está ligado em t=0.** A supressão da fuga no whole CNS
depende da rede inibitória carregar (~50–100 ms). Antes disso o GF responde como
o circuito isolado responderia. Medido, documentado, não corrigido.

**`tarsi` não é padrão.** A poda de pares de colisão acelera, mas diverge em
2 de 4 comportamentos. Ver
[`docs/research/COLLISION_PAIR_AUDIT.md`](research/COLLISION_PAIR_AUDIT.md).

---

## Quando não funciona

**"canal de controle desligado" no painel EXPERIMENTO**
A simulação subiu com `--sem-controle`, ou outro processo já ocupa a 8766.

**A porta continua ocupada depois de fechar**
`pkill -f` não mata Python no Windows. Use
`Get-Process python | Stop-Process -Force` no PowerShell, ou
`taskkill /IM python.exe /F`. Um runner velho segurando a porta faz o novo cair
em "sem telemetria" **em silêncio**, e a Unity fica lendo dado antigo.

**A Unity mostra a mosca mas ela não anda**
Olhe o painel CORPO. `RODANDO` com idade de telemetria em milissegundos
significa que está chegando pose. Se disser `PARADO`, o Python terminou a
duração — `--duracao 0` evita isso.

**A mosca aparece desmontada ou longe da câmera**
Modo `Bind Pose`: ele congela a pose de repouso do modelo e não lê telemetria
nenhuma. Se ficar certa ali e errada rodando, o problema é telemetria ou
interpolação, não montagem. Ver
[`docs/UNITY_BODY_COORDINATES.md`](UNITY_BODY_COORDINATES.md).

**`BackendIndisponivel` no neural**
`--neural cpu` sempre funciona (é a referência em fp64, e é lenta). Para ver os
dispositivos OpenCL:
`python -c "import sys;sys.path.insert(0,'sim');from neural import dispositivos;print(dispositivos())"`.

**`OverflowError` sobre a escala de ponto fixo**
O conectoma carregado tem convergência maior do que a escala aguenta em int32.
Isso é o guard funcionando: overflow de atômico não dá erro, dá a volta, e uma
inibição enorme viraria excitação enorme. Ver `sim/neural/model.py`.

**As malhas da mosca não existem**
`.venv-flygym2\Scripts\python tools\export_fly_mesh.py` — extrai as 69 malhas do
modelo compilado. Os `.obj` não vão pro git.
