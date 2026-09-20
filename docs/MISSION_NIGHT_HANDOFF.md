# Handoff da missão noturna — Drosobot

Estado vivo da missão. **Atualizado a cada etapa concluída.** Se a sessão cair,
é daqui que se retoma.

Branch: `research/gpu-core`. Não mergear para `master`, não force-push.

Para retomar manualmente, basta colar:

> Retome a missão noturna do Drosobot. Leia `docs/MISSION_NIGHT_HANDOFF.md`,
> continue autônomo até o fim, commit e push em `research/gpu-core`.

---

## Ordem de trabalho

| # | etapa | estado |
|---|---|---|
| 1 | Unificar o runtime: `drosobot_lab.py` ganha canal de controle, catálogo de experimentos, retina e máquina de estados | **a fazer** |
| 2 | Device/GPU na telemetria e no painel (OS, CPU, GPU, backends, escopo, neurônios, arestas, VRAM, RTF) | a fazer |
| 3 | GF gate: contribuintes inibitórios por nome (via `connectome/neuron_properties.csv`) | a fazer |
| 4 | CNS: separar `simulated neurons` de `visualized morphologies` na UI | a fazer |
| 5 | Guard de overflow na acumulação de ponto fixo | a fazer |
| 6 | Smoke test + regressão científica + teste ponta a ponta | a fazer |
| 7 | Um benchmark ponta a ponta (RTF, física, neural, visão, telemetria, gargalo) | a fazer |
| 8 | Docs: README + `docs/` (arquitetura, execução, ambientes, GPU, FlyGym1x2, whole CNS, limitações, troubleshooting) | a fazer |
| 9 | Commits funcionais + `git push origin research/gpu-core` | a fazer |
| 10 | Relatório final (18 seções + WHAT I DID NOT CHANGE) | a fazer |

## Já concluído antes da missão

- Mosca 3D validada: eixos, bind pose, normais com vinco, três paletas
  (Clay/Flybody/Drosophila). Ver `docs/UNITY_BODY_COORDINATES.md`.
- Comparação 4-vias com arena equivalente. Ver
  `docs/research/FOUR_WAY_LOOMING_COMPARISON.md`.
- Achado: o gate do GF não está ligado em t=0 — o whole CNS dá 1 fuga no
  transiente de partida. Ver `docs/research/WHOLE_CNS_GIANT_FIBER.md`.

## Estado do código (para quem retoma)

- `sim/drosobot_lab.py` — runtime novo: PhysicsAdapter + NeuralEngine + whole
  CNS + telemetria. **Não** tem canal de controle nem manda retina.
- `sim/lab_runner.py` — runtime antigo: canal de controle (8766), catálogo de
  experimentos, circuitos de `sim/experiments/`. É o que a Unity conversa hoje.
- A unificação é dar ao `drosobot_lab.py` o que o `lab_runner.py` tem de
  controle, mantendo o `lab_runner.py` como referência/regressão dos
  experimentos de circuito que geram as figuras do README.

## Invariantes que não podem ser tocados

- `legs` é o padrão de colisão; `tarsi` não vira padrão global.
- Sem clamp no potencial de membrana. O valor vai como veio; o aviso é registro.
- Sem entrada tônica global no whole CNS. `TONIC_INHIB_HZ` não volta.
- Escala de ponto fixo fica em 16384; não volta para 1024.
- Entrada sensorial é Poisson discreto, nunca corrente média equivalente.
- Conectoma inteiro nunca é chamado de cérebro funcional completo.
- Não ajustar parâmetro para o comportamento "aparecer".
