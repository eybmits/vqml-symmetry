#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-python3}"
SHARDS="${SHARDS:-5}"
EPOCHS="${EPOCHS:-100}"
RUN_AUDIT="${RUN_AUDIT:-0}"

cd "${ROOT}"

"${PYTHON}" -m experiments.consistent_paper_experiments --stage prepare-epochs --epochs "${EPOCHS}"
"${PYTHON}" -m experiments.consistent_paper_experiments --stage seed-existing

if [[ "${RUN_AUDIT}" == "1" ]]; then
  for ((i = 0; i < SHARDS; i++)); do
    "${PYTHON}" -m experiments.consistent_paper_experiments --stage audit --shard-index "${i}" --shard-count "${SHARDS}" &
  done
  wait
  "${PYTHON}" -m experiments.consistent_paper_experiments --stage choose-budget
fi

for stage in headline-edge headline-lines ablation random; do
  for ((i = 0; i < SHARDS; i++)); do
    "${PYTHON}" -m experiments.consistent_paper_experiments --stage "${stage}" --epochs "${EPOCHS}" --shard-index "${i}" --shard-count "${SHARDS}" &
  done
  wait
  "${PYTHON}" -m experiments.consistent_paper_experiments --stage merge
done

"${PYTHON}" -m experiments.consistent_paper_experiments --stage validate --epochs "${EPOCHS}"
"${PYTHON}" paper/make_figures.py
(cd paper && latexmk -pdf -interaction=nonstopmode main.tex)
