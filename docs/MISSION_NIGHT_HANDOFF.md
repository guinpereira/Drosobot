# Handoff da missão noturna — Drosobot

Estado da missão ao fim da noite de 20/09/2026. Branch `research/gpu-core`.

Para retomar:

> Leia `docs/MISSION_NIGHT_HANDOFF.md` e continue de onde parou.

---

## Etapas

| # | etapa | estado |
|---|---|---|
| 1 | Unificar o runtime: controle, catálogo, retina, máquina de estados | **feito** |
| 2 | Device/GPU na telemetria e no painel | **feito** |
| 3 | GF gate: contribuintes inibitórios rotulados por tipo | **feito** |
| 4 | CNS: `simulated` separado de `visualized morphologies` | **feito** |
| 5 | Guard de overflow no ponto fixo | **feito** |
| 6 | Smoke + regressão científica + ponta a ponta + contrato de telemetria | **feito** |
| 7 | Benchmark ponta a ponta | **feito** |
| 8 | Docs: README, RUNNING_THE_LAB, VULKAN_STRATEGY, THIRD_PARTY_NOTICES | **feito** |
| 9 | Commits + push | **feito** |
| 10 | Relatório final | **feito** |

## O que ficou pendente, e por quê

- **Vulkan.** Decisão registrada em `docs/research/VULKAN_STRATEGY.md`: não
  implementar agora. A física é 79–86% do relógio; o neural é 18% com o
  conectoma inteiro. A interface está pronta, os gatilhos que fariam valer a
  pena estão listados.
- **Sessão Unity ao vivo não foi vista por olho humano nesta rodada.** O que foi
  verificado: o C# compila sem erro, e o teste de contrato confirma que toda
  mensagem e todo campo que os painéis leem chegam no fluxo. Ver o laboratório
  rodando exige Play no Editor.
- **Custo da física.** É o gargalo e não foi atacado. Reduzi-lo é o próximo
  ganho real; qualquer trabalho no neural mexe em menos de um quinto do relógio.
- **`lab_runner.py` continua existindo.** É a regressão dos circuitos que
  produzem as figuras do README. Não foi absorvido de propósito.

## Invariantes que não podem ser tocados

- `legs` é o padrão de colisão; `tarsi` não vira padrão global.
- Sem clamp no potencial de membrana. O valor vai como veio; o aviso é registro.
- Sem entrada tônica global no whole CNS. `TONIC_INHIB_HZ` não volta.
- Escala de ponto fixo fica em 16384; não volta para 1024. O guard confere
  contra o conectoma carregado e levanta se não couber.
- Entrada sensorial é Poisson discreto, nunca corrente média equivalente.
- Conectoma inteiro nunca é chamado de cérebro funcional completo.
- Não ajustar parâmetro para o comportamento "aparecer".

## Onde estão as coisas

| | |
|---|---|
| runtime | `sim/drosobot_lab.py` |
| como rodar | `docs/RUNNING_THE_LAB.md` |
| coordenadas da mosca | `docs/UNITY_BODY_COORDINATES.md` |
| gate do GF | `docs/research/WHOLE_CNS_GIANT_FIBER.md` |
| tabela 4-vias | `docs/research/FOUR_WAY_LOOMING_COMPARISON.md` |
| backends | `docs/research/BACKEND_MATRIX.md`, `VULKAN_STRATEGY.md` |
| testes | `tests/test_drosobot_lab.py`, `tests/test_neural_backend.py` |
