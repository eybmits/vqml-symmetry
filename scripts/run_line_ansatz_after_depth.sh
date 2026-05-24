#!/usr/bin/env zsh
set -euo pipefail

cd /Users/markus/Documents/Programmierung/qc_symmetry

PYTHON=/Users/markus/anaconda3/bin/python
LOG=results/logs/line_ansatz_after_depth.log
OUTPUT=results/csv/results_line_ansatz_draft_L3p2.csv
SUMMARY=results/csv/table_line_ansatz_draft_L3p2.csv

mkdir -p results/logs results/csv

{
  echo "[$(date)] Waiting for qc_goldilocks_compression_depth_queue to finish."
  while (screen -ls || true) | grep -q "qc_goldilocks_compression_depth_queue"; do
    echo "[$(date)] Existing queue still running; waiting 5 minutes."
    sleep 300
  done

  echo "[$(date)] Starting winner-line ansatz draft sweep."
  "${PYTHON}" -u -m experiments.line_ansatz_sweep \
    --circuit-families edge,line_zzz \
    --subgroups none,Z2_rot180,Z2_reflection,C4,D2_V4,D4 \
    --train-sizes 30,120,450 \
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

  echo "[$(date)] Winner-line ansatz draft sweep finished."
} 2>&1 | tee -a "${LOG}"
