#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-python3}"
EPOCHS="${EPOCHS:-100}"

cd "${ROOT}"

"${PYTHON}" -m src.sanity_checks --skip-smoke
"${PYTHON}" -m experiments.consistent_paper_experiments --stage validate --epochs "${EPOCHS}"
"${PYTHON}" paper/make_figures.py
(cd paper && make all)
