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

**Coluna da direita**

- **MALE CNS** — o conectoma 3D numa caixa, com as sinapses pulsando. Câmera
  própria: antes ele dividia câmera com a mosca e saía de quadro quando ela
  andava.
- **CORPO** — modos `Normal / Bind Pose / Eixos / Rótulos`, vistas
  `Persp / Lado / Topo / Frente`, botão Focar, paletas `Clay / Flybody /
  Drosophila`, e o estado da conexão.
- **PROCEDÊNCIA** — a separação DATA / MODEL / ASSUMPTION, sempre visível.
- **RUNTIME** — OS, CPU, GPU, backend de física, backend neural, pares de
  colisão, neurônios simulados, morfologias na tela, arestas, VRAM, custo por
  etapa e RTF.
- **GIANT FIBER — GATE** — excitação, inibição, líquido, `v` mínimo, spikes, e
  os maiores contribuintes inibitórios do passo, rotulados por tipo.
- **VISÃO** — os 721 omatídeos por olho.
- **INSPECTOR** e **EXPERIMENTO**.

**Rodapé** — sinais em janela rolante.

### Teclas

`C` liga o fluxo de sinapses · `F` troca o filtro de arestas · `N` liga/desliga
neurônios · `B` liga/desliga a casca · `P` modo apresentação · `Esc` limpa a
seleção · botão esquerdo orbita · roda aproxima · botão do meio empurra.

---

## Desempenho medido

RX 6700 XT (gfx1031, 20 CUs, 12 GiB), Windows 11, FlyGym 2.1.0 / MuJoCo 3.9,
OpenCL, looming, `legs`, 1 s de mosca:

| etapa | whole CNS | circuito |
|---|---|---|
| física | 14.405 ms/s · 78,9% | 13.036 ms/s · 86,4% |
| neural | 3.363 ms/s · 18,4% | 1.453 ms/s · 9,6% |
| visão | 14 ms/s | 8 ms/s |
| leitura | 220 ms/s | 159 ms/s |
| telemetria | 0,1 ms/s | 192 ms/s |
| **total** | **18.266 ms/s** | **15.092 ms/s** |
| **RTF** | **0,055×** | **0,066×** |

Unidade: **ms de relógio por segundo simulado** — a única base em que as etapas
se somam. A seção "por chamada" do profiler é outra grandeza e não soma.

**O gargalo é a física**, 79–86% do relógio, mesmo com o conectoma inteiro na
GPU. Trocar 1.261 neurônios por 164.451 — 243× mais arestas — custa 1,9 s de
relógio por segundo simulado. Foi isso que a GPU comprou.

A telemetria custa quase nada sem cliente conectado e ~190 ms/s com a Unity
lendo; as duas linhas acima são de corridas diferentes nesse aspecto.

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
