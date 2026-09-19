# Drosobot

Robô controlado por circuitos **reais** extraídos do conectoma completo do sistema
nervoso da mosca-das-frutas (*Drosophila melanogaster*, dataset **Male CNS**, Janelia).

Nada de rede neural genérica treinada do zero — os pesos sinápticos vêm direto do
mapeamento real do cérebro+cordão nervoso da mosca (microscopia eletrônica + reconstrução
automatizada). A ideia: será que dá pra pegar a fiação de verdade de um circuito de reflexo
da mosca e usar ela pra controlar um robô?

**Fase atual: tudo validado em software, zero hardware físico comprado ainda.**
Circuito só vira compra de peça depois de provar que funciona na simulação.

## Circuito 1 — Giant Fiber (reflexo de fuga)

Pathway: **PVLP** (visual) → **DNp01** (Giant Fiber, o neurônio de fuga mais estudado
da mosca) → **TTMn** (Tergotrochanteral Motor Neuron, motoneurônio real do músculo de pulo).

Confirmado batendo com a literatura: entrada forte vem de área visual (PVLP), saída
vai pro motoneurônio de pulo documentado. Simulação (Brian2, peso sináptico real do
neuPrint) mostra o padrão esperado — silêncio enquanto o estímulo (objeto se
aproximando, simulado) é fraco, disparo esparso e confiável quando cruza o limiar:

![Raster Giant Fiber](docs/images/giant_fiber_raster.png)

Robô digital reagindo ao disparo real (pulo pra trás a cada spike do TTMn):

![Robô reagindo - Giant Fiber](docs/images/digital_robot_reaction.png)

## Circuito 2 — Optomotor (vira em resposta a movimento visual)

Pathway: **T4/T5** (detecção de movimento, ~13.500 neurônios) → **HS** (wide-field
integration, 8 neurônios) → **DNa02** (descending neuron documentado na literatura
como controlador de giro durante caminhada) → **Sternal anterior rotator MN**
(motoneurônio real que gira a coxa da perna).

Importante: verificamos conectividade real antes de montar a rede — **VS e
LPLC2/LC4 não conectam direto no DNa02** nesse dataset (peso zero), só HS conecta
(fraco, mas real). O circuito usado é o que **existe de fato** nos dados, não o
mais "óbvio" da literatura de outras espécies de mosca.

![Raster Optomotor](docs/images/optomotor_raster.png)

Robô digital virando (heading acumula em degraus a cada disparo do motor de perna,
relaxa de volta pro reto entre disparos):

![Robô virando - Optomotor](docs/images/optomotor_robot_reaction.png)

## Simulador ao vivo (pygame) — os dois circuitos rodando ao mesmo tempo

`sim/live_robot_sim.py`: em vez de rodar 300ms e parar, os dois circuitos (Giant
Fiber + Optomotor) ficam ativos continuamente, com peso sináptico real, e você
controla o estímulo pelo teclado (seta cima = objeto se aproxima, seta esq/dir =
movimento visual) enquanto vê o robô reagir na tela em tempo real. Mesma prioridade
do firmware real: escape sempre interrompe um giro em andamento.

Simplificação documentada: o circuito optomotor aqui não separa os dois hemisférios
(pool único dos top-20 T4/T5 por peso) — o que vem do dado real é **se e quando** o
circuito dispara; a **direção** do giro no desenho usa a tecla que você está
segurando no momento, não vem do dado.

```
.venv\Scripts\python sim\live_robot_sim.py
```

Giro isolado (segurando seta esquerda — cada disparo do DNa02 vira 2°, sem limite
artificial, só a taxa de disparo real do circuito):

![Giro ao vivo](docs/images/live_sim_turn_loop.png)

Os dois circuitos coexistindo no mesmo robô — trilha mostra o giro (curva) seguido
de fuga (ponta vermelha, pulo pra trás):

![Escape e giro combinados](docs/images/live_sim_combined.png)

Ajuste de ganho documentado (não escondido): primeira versão usava 6°/spike +
limite de 1 giro a cada 60ms — o robô fechava um loop completo em menos de 1
segundo (rápido demais pra acompanhar) e depois quase não girava quando os dois
freios foram empilhados junto (o cooldown multiplicou o efeito do grau baixo).
Solução: manter só 1 parâmetro (2°/spike) e deixar a velocidade real do circuito
(refratário de 20ms do DNa02) decidir o ritmo, sem freio artificial por cima.

## Visualização 3D — a morfologia real dos neurônios usados

Os dois circuitos acima não são grafo abstrato: cada neurônio tem morfologia 3D
reconstruída por microscopia eletrônica. `blender/render_circuits.py` monta a cena
no Blender com a malha do cérebro+VNC do Male CNS (template `JRCFIB2022M`, via
`navis-flybrains`) e os esqueletos reais dos 54 neurônios das duas simulações,
coloridos por papel no circuito.

![CNS frontal com os dois circuitos](docs/images/cns_circuitos_frontal.png)

| cor | grupo | papel |
|---|---|---|
| vermelho | `gf_dnp01_giantfiber` | Giant Fiber (DNp01) |
| laranja | `gf_ttmn_motor` | motoneurônio de pulo (TTMn) |
| azul | `gf_sensor_visual` | entrada visual (PVLP) |
| azul claro | `om_sensor_t4t5` | detecção de movimento (T4/T5) |
| amarelo | `om_hs_widefield` | integração wide-field (HS) |
| verde | `om_dna02_steering` | comando de giro (DNa02) |
| magenta | `om_leg_motor` | motoneurônio de perna |

A vista frontal é a melhor checagem de sanidade do pipeline: os dois hemisférios
espelham limpo. Escala ou eixo errados quebrariam essa simetria na hora.

A espessura não é decorativa — usa o raio real de cada ponto do SWC. Por isso o
Giant Fiber aparece visivelmente mais grosso que o resto: ele é o axônio de maior
calibre do CNS da mosca (raio até 964 unidades contra mediana 32).

![CNS em perspectiva](docs/images/cns_circuitos_perspectiva.png)

### Como rodar

```
.venv\Scripts\python connectome\fetch_skeletons_for_blender.py   # gera .swc + .obj (uma vez)
"C:\Program Files\Blender Foundation\Blender 5.2\blender.exe" --python blender\render_circuits.py
```

Pra regerar as imagens acima sem abrir janela: `blender.exe --background --python blender\render_doc.py`.

O Python embutido do Blender é isolado (ignora user-site e `PYTHONPATH`), então o
`navis` precisa ser instalado num diretório próprio que o script insere no
`sys.path`:

```
"C:\Program Files\Blender Foundation\Blender 5.2\5.2\python\bin\python.exe" -m pip install --target blender\_pylibs navis
```

`blender/_pylibs/`, os `.swc` e os `.obj` são gerados/vendorizados (~420 MB) e não
sobem pro git, igual os CSV do `connectome/`.

### Percalços reais, documentados pra não repetir

Os neurônios simplesmente **não apareciam** na cena — carregavam sem erro, 54
objetos com nome certo, nada visível. Quatro causas empilhadas:

1. **`navis.interfaces.blender.Handler` tem `scaling=1/10000` por padrão.** Ele
   aplica isso *em cima* de qualquer escala já aplicada no neurônio. O esqueleto
   ficava 10.000x menor que a malha — um ponto preto perto da origem. Toda a
   conversão (voxel → nm → viewport) deve ir no construtor do `Handler`, e só lá.
2. **Unidade.** O esqueleto do neuPrint vem em voxel (`n.units == "8 nanometer"`),
   a malha do `flybrains` já vem em nanômetro puro. Sem o fator 8 o neurônio sai
   8x menor.
3. **`bpy.ops.wm.obj_import` converte eixo por padrão** (OBJ Y-up → Blender Z-up,
   troca Y↔Z e nega um deles). Os esqueletos entram pelo navis sem essa conversão,
   então malha e neurônio ficavam em orientações diferentes. Corrigido com
   `forward_axis='Y', up_axis='Z'`.
4. **Bevel chapado escondia a estrutura.** `bevel_depth` fixo era ~8x a mediana do
   raio real, e os arbores finos inchavam até se fundirem num bloco sólido.
   `use_radii=True` usa o raio real por ponto.

Lição de método: `obj.dimensions` levou a diagnóstico errado por um bom tempo. Com
um bevel grosso numa curva minúscula, ele reporta basicamente o bevel — mudar a
escala dos pontos quase não mexia no número. `blender/diagnose.py` calcula o bbox
percorrendo os pontos da curva com a matriz de mundo, que é o que de fato mede.

## Firmware — testado no Wokwi, sem placa física

Ponte "burra" de propósito (`hardware/motor_bridge`): só lê HC-SR04 e executa o
comando de motor que o PC mandar. Toda decisão (fugir ou virar) vem do PC, rodando
os circuitos reais acima. Drive diferencial, 2 motores (L298N, canal A e B).
Protocolo serial 115200 baud:

- placa → PC: `D:<cm>` (leitura periódica, a cada 50ms)
- PC → placa: `E` (escape, os dois motores em reverso, 150ms)
- PC → placa: `L` / `R` (vira esquerda/direita, pivot 80ms)
- **Prioridade**: escape sempre interrompe um giro em andamento (igual biologia)

Testado no simulador Wokwi (extensão VS Code) antes de qualquer compra:

![Setup Wokwi](docs/images/wokwi_setup.jpg)

Percalços reais no caminho, documentados aqui pra não repetir:
- O chip `chip-l298n` (custom, GitHub) não carrega na extensão VS Code offline — só
  funciona no wokwi.com online. Solução: LED direto nos pinos IN1/IN2 pra validar a
  lógica do firmware sem depender do driver simulado. Não muda nada no firmware real.
- Extensão VS Code não tem painel "Serial Monitor" separado — é o próprio **Terminal**
  do VS Code, e desde a v2.1.0 tem [bug conhecido](https://github.com/wokwi/wokwi-features/issues/698)
  que não mostra o que você digita (mas o envio funciona normal, "às cegas").

![Missing chip breakout](docs/images/wokwi_missing_chip.png)

## Setup

```
py -3.11 -m venv .venv
.venv\Scripts\pip install -r requirements.txt
```

Testado: Python 3.11.9, Brian2 2.9.0, neuprint-python 0.6.3, numpy travado em `<2.1`
(Brian2 2.9.0 quebra com numpy 2.4 — `ndarray.ptp` foi removido).

## Acesso ao neuPrint (dado real do conectoma)

Server: `neuprint.janelia.org` — Dataset: `male-cns:v1.0`

Precisa de token pessoal (login necessário, não dá pra automatizar):
1. Login em https://neuprint.janelia.org (conta Google)
2. Account → copia o Auth Token
3. Salva em `.neuprint_token` na raiz do projeto (já no `.gitignore`, nunca vai pro git)

## Firmware — compilar e testar

```
arduino-cli compile --fqbn arduino:avr:uno hardware/motor_bridge --output-dir hardware/motor_bridge/build
```

No VS Code, com a extensão Wokwi instalada: abre `hardware/motor_bridge`, `F1` →
`Wokwi: Start Simulation`, digita `E` no painel Terminal pra disparar o escape.

`hardware/pc_bridge_test.py` (pyserial) fica pronto pra quando tiver placa física com
COM real — a extensão VS Code do Wokwi não expõe porta COM do host, só terminal interno.

## Estrutura

- `connectome/` — scripts de fetch no neuPrint + CSV de conectividade (dado bruto, não sobe pro git)
- `sim/` — simulações Brian2 (rede spiking com peso sináptico real) + imagens de resultado
- `hardware/` — firmware Arduino/ESP32 + diagrama Wokwi
- `blender/` — cena 3D no Blender (malha do CNS + esqueletos reais); `_pylibs/`,
  `skeletons/` e os `.obj` sao gerados/vendorizados, nao sobem pro git
- `docs/images/` — imagens usadas neste README

## Roadmap / Status

- [x] Ambiente Python validado (Brian2 + neuprint-python)
- [x] Circuito 1 (Giant Fiber) — fetch, rede, robô digital reagindo
- [x] Firmware compilado e testado no Wokwi (sem hardware físico)
- [x] Circuito 2 (Optomotor) — fetch, rede, robô digital virando
- [x] Combinar os dois circuitos num robô só, ao vivo, controlado por teclado (`sim/live_robot_sim.py`), com HUD e prioridade escape > giro validados visualmente
- [x] Firmware com drive diferencial (2 motores, `E`/`L`/`R`, prioridade escape > giro)
- [x] Visualizacao 3D da morfologia real dos 54 neuronios sobre a malha do CNS
      (`blender/render_circuits.py`), validada pela simetria bilateral
- [ ] Comprar kit físico (favorito atual: Kuyshun ESP32-CAM 328P — tem HC-SR04 +
      arquitetura dual-MCU ESP32-CAM/ATmega328P já pronta, resolve o aperto de GPIO)
- [ ] Portar firmware simulado pro hardware real, validar ponta a ponta
