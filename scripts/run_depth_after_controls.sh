#!/usr/bin/env zsh
set -euo pipefail

cd /Users/markus/Documents/Programmierung/qc_symmetry

PARTIAL_CSV="results/csv/results_partial_data_sweep_full_L3p2.csv"
CONTROL_CSV="results/csv/results_random_sharing_control_full_L3p2_train450.csv"
LOWDATA_CSV="results/csv/results_lowdata_goldilocks_c4_d4_L3p2.csv"
LOWDATA_LOG="results/logs/lowdata_goldilocks_c4_d4_L3p2.log"
COMPRESSION_CSV="results/csv/results_d4_compression_sweep_L3p2.csv"
COMPRESSION_LOG="results/logs/d4_compression_sweep_L3p2.log"
DEPTH_CSV="results/csv/results_depth_sweep_all_groups_draft_L1234_p123.csv"
DEPTH_LOG="results/logs/depth_sweep_all_groups_draft_L1234_p123.log"
QUEUE_LOG="results/logs/depth_queue.log"

mkdir -p results/csv results/figures results/logs

echo "[$(date)] Depth queue started." >> "$QUEUE_LOG"
echo "[$(date)] Waiting for data sweep and random-sharing control." >> "$QUEUE_LOG"

while true; do
  partial_running=0
  control_running=0
  partial_ready=0
  control_ready=0

  pgrep -f "experiments.partial_equivariance_sweep.*results_partial_data_sweep_full_L3p2" >/dev/null && partial_running=1
  pgrep -f "experiments.random_sharing_control.*results_random_sharing_control_full_L3p2_train450" >/dev/null && control_running=1

  if [[ -f "$PARTIAL_CSV" ]] && [[ $(wc -l < "$PARTIAL_CSV") -ge 151 ]]; then
    partial_ready=1
  fi
  if [[ -f "$CONTROL_CSV" ]] && [[ $(wc -l < "$CONTROL_CSV") -ge 56 ]]; then
    control_ready=1
  fi

  {
    echo "[$(date)] partial_running=$partial_running partial_ready=$partial_ready control_running=$control_running control_ready=$control_ready"
    [[ -f "$PARTIAL_CSV" ]] && wc -l "$PARTIAL_CSV"
    [[ -f "$CONTROL_CSV" ]] && wc -l "$CONTROL_CSV"
  } >> "$QUEUE_LOG" 2>&1

  if [[ "$partial_running" -eq 0 && "$control_running" -eq 0 && "$partial_ready" -eq 1 && "$control_ready" -eq 1 ]]; then
    break
  fi
  sleep 300
done

echo "[$(date)] Starting focused low-data C4-vs-D4 sweep." >> "$QUEUE_LOG"

/Users/markus/anaconda3/bin/python -u -m experiments.partial_equivariance_sweep \
  --subgroups C4,D4 \
  --train-sizes 3,6,9,12,15,18,24,30,45,60,90,120 \
  --L 3 \
  --p 2 \
  --seeds 0,1,2,3,4 \
  --epochs 100 \
  --steps-per-epoch 30 \
  --test-size 600 \
  --output "$LOWDATA_CSV" \
  --resume \
  --no-plots \
  > "$LOWDATA_LOG" 2>&1

echo "[$(date)] Focused low-data sweep finished. Building low-data summary." >> "$QUEUE_LOG"

/Users/markus/anaconda3/bin/python -m experiments.paper_summary \
  "$LOWDATA_CSV" \
  --kind partial \
  --table-name table_lowdata_goldilocks_c4_d4_L3p2 \
  >> "$QUEUE_LOG" 2>&1

echo "[$(date)] Starting beyond-D4 compression sweep." >> "$QUEUE_LOG"

/Users/markus/anaconda3/bin/python -u -m experiments.d4_compression_sweep \
  --train-sizes 30,120,450 \
  --L 3 \
  --p 2 \
  --seeds 0,1,2,3,4 \
  --epochs 100 \
  --steps-per-epoch 30 \
  --test-size 600 \
  --output "$COMPRESSION_CSV" \
  --resume \
  --no-plots \
  > "$COMPRESSION_LOG" 2>&1

echo "[$(date)] Beyond-D4 compression sweep finished. Building compression summary." >> "$QUEUE_LOG"

/Users/markus/anaconda3/bin/python -m experiments.paper_summary \
  "$COMPRESSION_CSV" \
  --kind partial \
  --table-name table_d4_compression_sweep_L3p2 \
  >> "$QUEUE_LOG" 2>&1

echo "[$(date)] Starting depth sweep." >> "$QUEUE_LOG"

/Users/markus/anaconda3/bin/python -u -m experiments.depth_sweep \
  --subgroups none,Z2_rot180,Z2_reflection,C4,D2_V4,D4 \
  --L-values 1,2,3,4 \
  --p-values 1,2,3 \
  --train-size 450 \
  --seeds 0,1,2 \
  --epochs 60 \
  --steps-per-epoch 20 \
  --test-size 600 \
  --output "$DEPTH_CSV" \
  --resume \
  --no-plots \
  > "$DEPTH_LOG" 2>&1

echo "[$(date)] Depth sweep finished. Building plot." >> "$QUEUE_LOG"

/Users/markus/anaconda3/bin/python -m experiments.paper_summary \
  "$DEPTH_CSV" \
  --kind depth \
  >> "$QUEUE_LOG" 2>&1

echo "[$(date)] Depth queue finished." >> "$QUEUE_LOG"
