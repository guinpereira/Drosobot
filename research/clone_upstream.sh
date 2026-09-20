#!/usr/bin/env bash
# Reconstroi research/upstream/ nos commits de research/UPSTREAM_LOCK.md.
#
# Os clones nao entram no nosso repositorio: sao centenas de MB de codigo de
# terceiros com historia propria. O que versionamos e o lock e a analise.
#
#   bash research/clone_upstream.sh
#
# Roda de qualquer diretorio. Nao apaga nada: repo ja clonado e so trazido pro
# commit do lock.
set -euo pipefail

RAIZ="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="$RAIZ/research/upstream"
mkdir -p "$DEST"

# repo|url|commit|profundidade (0 = completo)
REPOS=(
  "flygym|https://github.com/NeLy-EPFL/flygym.git|38c8ec61034cd59bc5ba0de20688d4a3c0000d60|0"
  "mujoco|https://github.com/google-deepmind/mujoco.git|5ceb72b1bbe87612a2859304d2c22a15cd3c180d|50"
  "mujoco_warp|https://github.com/google-deepmind/mujoco_warp.git|87e742d31c96f69a70741c51b9ade43bd8d1b60b|0"
  "warp|https://github.com/NVIDIA/warp.git|015e5a17827d4409782e81026d3de6f61b9ae1a7|0"
)

for entrada in "${REPOS[@]}"; do
  IFS='|' read -r nome url commit prof <<< "$entrada"
  alvo="$DEST/$nome"
  if [ -d "$alvo/.git" ]; then
    echo "== $nome ja existe, fixando no commit do lock"
  else
    echo "== clonando $nome"
    if [ "$prof" = "0" ]; then
      git clone --quiet "$url" "$alvo"
    else
      # Historico raso: o mujoco completo passa de 1 GB e a analise so precisa
      # da arvore. Fundo o bastante pra `git describe` ainda achar uma tag.
      git clone --quiet --depth "$prof" "$url" "$alvo"
    fi
  fi
  git -C "$alvo" fetch --quiet origin "$commit" 2>/dev/null || true
  git -C "$alvo" checkout --quiet "$commit" 2>/dev/null \
    || echo "   AVISO: nao consegui fixar $nome em $commit (clone raso?)"
  echo "   $(git -C "$alvo" rev-parse HEAD)"
done

# O FlyGym 1.x e a versao que o Drosobot roda hoje. Vem do MESMO repo, como
# worktree: um segundo clone so duplicaria 131 MB pra ter outra tag.
WT="$DEST/flygym-1.2.1"
if [ ! -d "$WT" ]; then
  echo "== worktree flygym v1.2.1 (versao que o Drosobot usa hoje)"
  git -C "$DEST/flygym" worktree add --quiet "$WT" v1.2.1
fi
echo "   $(git -C "$WT" rev-parse HEAD)"

echo
echo "pronto. Confira contra research/UPSTREAM_LOCK.md"
