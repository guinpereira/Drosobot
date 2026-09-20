# Upstream lock

Commits exatos em que esta pesquisa se baseia. Qualquer afirmacao em
`docs/research/*.md` foi lida NESTES commits -- upstream se move, e uma analise
que diz "o codigo faz X" sem dizer qual codigo nao da pra conferir depois.

Os clones NAO sao versionados (ver `research/upstream/.gitignore`). Pra
reproduzir: `bash research/clone_upstream.sh`.

Data dos clones: **2026-09-20**
Maquina: Windows 11 Pro 10.0.26200, Ryzen 7 5700X, Radeon RX 6700 XT 12 GB

| repositorio | owner | branch | commit | describe | licenca |
|---|---|---|---|---|---|
| flygym | NeLy-EPFL | main | `38c8ec61034cd59bc5ba0de20688d4a3c0000d60` | v2.1.0-7-g38c8ec6 | Apache-2.0 |
| flygym (1.x) | NeLy-EPFL | tag v1.2.1 | `c7affce924cb1c6add16619adf83be5c6b223e89` | v1.2.1 | Apache-2.0 |
| mujoco | google-deepmind | main (depth 50) | `5ceb72b1bbe87612a2859304d2c22a15cd3c180d` | 3.13.0-353-g5ceb72b | Apache-2.0 |
| mujoco_warp | google-deepmind | main | `87e742d31c96f69a70741c51b9ade43bd8d1b60b` | v3.13.0-9-g87e742d | Apache-2.0 |
| warp | NVIDIA | main | `015e5a17827d4409782e81026d3de6f61b9ae1a7` | llvm-sdk-22.1.8-warp.1-272-g015e5a178 | Apache-2.0 |

## Por que cada um

**flygym (2.x)** -- o alvo da migracao. Reescrito: `src/flygym/`, API nova,
caminho GPU em `src/flygym/warp/`. E onde estao os ganhos que queremos entender.

**flygym v1.2.1** -- a versao que o Drosobot roda hoje (`pip show flygym` -> 1.2.1).
Clonado como *worktree* do mesmo repo, nao como segundo clone:
`git worktree add ../flygym-1.2.1 v1.2.1`. Serve pra diferenca 1.x -> 2.x ser
medida, nao lembrada.

**mujoco** -- o motor de fisica que roda por baixo dos dois. Interessa o
narrowphase, o solver e o MJX. Clonado com `--depth 50`: o historico completo
passa de 1 GB e nao precisamos dele.

**mujoco_warp** -- a implementacao GPU do MuJoCo. E a fonte principal de ideias
de paralelizacao, e tambem de onde sai a conclusao sobre o que NAO serve pro
nosso caso.

**warp** -- a linguagem de kernel embaixo do mujoco_warp. Clonado pra responder
uma pergunta especifica, e ela foi respondida: quais backends existem.

## Versoes do ambiente Drosobot atual

Medidas com `.venv/Scripts/python`, mesmo dia:

```
flygym       1.2.1
mujoco       3.2.7
dm_control   1.0.27
gymnasium    1.3.0
numpy        2.0.2
brian2       2.9.0
```

Note a distancia: o FlyGym 2.x pede `mujoco>=3.9,<3.10` e Python >=3.12. Nao e
upgrade de versao menor -- e outra API sobre outra base.
