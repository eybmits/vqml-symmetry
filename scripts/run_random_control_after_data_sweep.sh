#!/usr/bin/env zsh
set -euo pipefail

cd /Users/markus/Documents/Programmierung/qc_symmetry

PARTIAL_CSV="results/csv/results_partial_data_sweep_full_L3p2.csv"
PARTIAL_LOG="results/logs/partial_data_sweep_full_L3p2.log"
CONTROL_CSV="results/csv/results_random_sharing_control_full_L3p2_train450.csv"
CONTROL_LOG="results/logs/random_sharing_control_full_L3p2_train450.log"
QUEUE_LOG="results/logs/random_control_queue.log"

mkdir -p results/csv results/figures results/logs

{
  echo "[$(date)] Queue started."
  echo "[$(date)] Waiting for partial data sweep to finish..."
} >> "$QUEUE_LOG"

while pgrep -f "experiments.partial_equivariance_sweep.*results_partial_data_sweep_full_L3p2" >/dev/null; do
  {
    echo "[$(date)] partial sweep still running."
    if [[ -f "$PARTIAL_CSV" ]]; then
      wc -l "$PARTIAL_CSV"
    fi
    if [[ -f "$PARTIAL_LOG" ]]; then
      tail -n 3 "$PARTIAL_LOG"
    fi
  } >> "$QUEUE_LOG" 2>&1
  sleep 300
done

echo "[$(date)] Partial data sweep finished. Building summary." >> "$QUEUE_LOG"

/Users/markus/anaconda3/bin/python -m experiments.paper_summary \
  "$PARTIAL_CSV" \
  --kind partial \
  --table-name table_partial_data_sweep_full_L3p2 \
  >> "$QUEUE_LOG" 2>&1

echo "[$(date)] Starting random-sharing control." >> "$QUEUE_LOG"

/Users/markus/anaconda3/bin/python -u -m experiments.random_sharing_control \
  --L 3 \
  --p 2 \
  --train-size 450 \
  --seeds 0,1,2,3,4 \
  --epochs 100 \
  --steps-per-epoch 30 \
  --test-size 600 \
  --output "$CONTROL_CSV" \
  --resume \
  --no-plots \
  > "$CONTROL_LOG" 2>&1

echo "[$(date)] Random-sharing control finished. Building control plot." >> "$QUEUE_LOG"

/Users/markus/anaconda3/bin/python -m experiments.paper_summary \
  "$CONTROL_CSV" \
  --kind random \
  >> "$QUEUE_LOG" 2>&1

echo "[$(date)] Queue finished." >> "$QUEUE_LOG"
