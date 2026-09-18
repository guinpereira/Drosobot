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

## Firmware — testado no Wokwi, sem placa física

Ponte "burra" de propósito (`hardware/motor_bridge`): só lê HC-SR04 e executa pulso
de escape. Toda decisão de quando fugir vem do PC, rodando o circuito real acima.
Protocolo serial 115200 baud:

- placa → PC: `D:<cm>` (leitura periódica, a cada 50ms)
- PC → placa: `E` (dispara pulso de escape, motor reverso 150ms)

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
- `docs/images/` — imagens usadas neste README

## Roadmap / Status

- [x] Ambiente Python validado (Brian2 + neuprint-python)
- [x] Circuito 1 (Giant Fiber) — fetch, rede, robô digital reagindo
- [x] Firmware compilado e testado no Wokwi (sem hardware físico)
- [x] Circuito 2 (Optomotor) — fetch, rede, robô digital virando
- [ ] Combinar os dois circuitos num robô só (foge de objeto E vira por movimento)
- [ ] Comprar kit físico (favorito atual: Kuyshun ESP32-CAM 328P — tem HC-SR04 +
      arquitetura dual-MCU ESP32-CAM/ATmega328P já pronta, resolve o aperto de GPIO)
- [ ] Portar firmware simulado pro hardware real, validar ponta a ponta
