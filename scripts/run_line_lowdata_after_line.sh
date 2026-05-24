#!/usr/bin/env zsh
set -euo pipefail

cd /Users/markus/Documents/Programmierung/qc_symmetry

PYTHON=/Users/markus/anaconda3/bin/python
LOG=results/logs/line_lowdata_after_line.log
OUTPUT=results/csv/results_lowdata_linezzz_all_groups_L3p2.csv
SUMMARY=results/csv/table_lowdata_linezzz_all_groups_L3p2.csv

mkdir -p results/logs results/csv

{
  echo "[$(date)] Waiting for qc_line_ansatz_after_depth to finish."
  while (screen -ls || true) | grep -q "qc_line_ansatz_after_depth"; do
    echo "[$(date)] Line ansatz queue still running; waiting 5 minutes."
    sleep 300
  done

  echo "[$(date)] Starting line_zzz low-training-size sweep."
  "${PYTHON}" -u -m experiments.line_ansatz_sweep \
    --circuit-families line_zzz \
    --subgroups none,Z2_rot180,Z2_reflection,C4,D2_V4,D4 \
    --train-sizes 3,6,9,12,15,18,24,30,45,60,90,120 \
    --L 3 \
    --p 2 \
    --seeds 0,1,2 \
    --epochs 60 \
    --steps-per-epoch 20 \
    --batch-size 15 \
    --test-size 600 \
    --output "${OUTPUT}" \
    --summary-output "${SUMMARY}" \
    --resume

  echo "[$(date)] line_zzz low-training-size sweep finished."
} 2>&1 | tee -a "${LOG}"
