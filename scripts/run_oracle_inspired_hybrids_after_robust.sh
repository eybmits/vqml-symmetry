#!/usr/bin/env zsh
set -euo pipefail

cd /Users/markus/Documents/Programmierung/qc_symmetry

PYTHON=/Users/markus/anaconda3/bin/python
LOG=results/logs/oracle_inspired_hybrids_after_robust_train450_600_750.log
OUTPUT=results/csv/results_oracle_inspired_robust_C4D4_L3p2_train450_600_750.csv
SUMMARY=results/csv/table_oracle_inspired_robust_C4D4_L3p2_train450_600_750.csv

mkdir -p results/logs results/csv

{
  echo "[$(date)] Waiting for qc_oracle_inspired_robust to finish."
  while true; do
    SCREEN_LIST="$(screen -ls 2>/dev/null || true)"
    if printf "%s\n" "${SCREEN_LIST}" | grep -q "[[:space:]][0-9][0-9]*\\.qc_oracle_inspired_robust[[:space:]]"; then
      sleep 60
    else
      break
    fi
  done

  echo "[$(date)] Robust base block finished; running sanity checks."
  "${PYTHON}" -m compileall -q src experiments
  "${PYTHON}" -m src.sanity_checks --skip-smoke

  echo "[$(date)] Starting robust oracle-inspired hybrid architecture sweep."
  "${PYTHON}" -u -m experiments.line_ansatz_sweep \
    --circuit-families edge_line_zzz,edge_line_ccrz,edge_line_zzz_ccrz,line_pair_crz \
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

  echo "[$(date)] Robust oracle-inspired hybrid architecture sweep finished."
} 2>&1 | tee -a "${LOG}"
