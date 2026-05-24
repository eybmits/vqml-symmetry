#!/usr/bin/env zsh
set -euo pipefail

cd /Users/markus/Documents/Programmierung/qc_symmetry

PYTHON=/Users/markus/anaconda3/bin/python
LOG=results/logs/line_depth_after_line.log
OUTPUT=results/csv/results_depth_sweep_linezzz_all_groups_draft_L1234_p123.csv

mkdir -p results/logs results/csv

{
  echo "[$(date)] Waiting for qc_line_ansatz_after_depth to finish."
  while (screen -ls || true) | grep -q "qc_line_ansatz_after_depth"; do
    echo "[$(date)] Line ansatz queue still running; waiting 5 minutes."
    sleep 300
  done

  echo "[$(date)] Starting line_zzz depth sweep."
  "${PYTHON}" -u -m experiments.depth_sweep \
    --circuit-family line_zzz \
    --subgroups none,Z2_rot180,Z2_reflection,C4,D2_V4,D4 \
    --L-values 1,2,3,4 \
    --p-values 1,2,3 \
    --train-size 450 \
    --seeds 0,1,2 \
    --epochs 60 \
    --steps-per-epoch 20 \
    --batch-size 15 \
    --test-size 600 \
    --output "${OUTPUT}" \
    --resume \
    --no-plots

  echo "[$(date)] line_zzz depth sweep finished."
} 2>&1 | tee -a "${LOG}"
