#!/usr/bin/env zsh
set -euo pipefail

cd /Users/markus/Documents/Programmierung/qc_symmetry

PYTHON=/Users/markus/anaconda3/bin/python
LOG=results/logs/oracle_inspired_robust_C4D4_L3p2_train450_600_750.log
OUTPUT=results/csv/results_oracle_inspired_robust_C4D4_L3p2_train450_600_750.csv
SUMMARY=results/csv/table_oracle_inspired_robust_C4D4_L3p2_train450_600_750.csv

mkdir -p results/logs results/csv

{
  echo "[$(date)] Starting robust oracle-inspired architecture sweep."
  "${PYTHON}" -u -m experiments.line_ansatz_sweep \
    --circuit-families edge,line_zzz,line_ccrz,line_zzz_ccrz \
    --subgroups C4,D4 \
    --train-sizes 450,600,750 \
    --L 3 \
    --p 2 \
    --seeds 0,1,2,3,4 \
    --epochs 100 \
    --steps-per-epoch 30 \
    --batch-size 15 \
    --test-size 600 \
    --output "${OUTPUT}" \
    --summary-output "${SUMMARY}" \
    --resume

  echo "[$(date)] Robust oracle-inspired architecture sweep finished."
} 2>&1 | tee -a "${LOG}"
