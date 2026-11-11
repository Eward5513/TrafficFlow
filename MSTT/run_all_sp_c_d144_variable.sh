#!/usr/bin/env bash
# AllSpC2All on the four p10_p20 frequencies.
# GPU 5: 30min, then 1h. GPU 7: 2h, then day.
# The two cards start together. A card starts its second run only after the first exits.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON="${PYTHON:-/home/tzhang174/miniforge3/envs/trafficflow/bin/python3}"
CONFIG="$ROOT/MSTT/configs/all_sp_c_separate_mlp_d144_projected_all.json"
LOG_DIR="/data5/tzhang174/TrafficFlow/xuancheng26/training_results_3s/reports/mstt_all_sp_c_d144"

run_chain() {
  local gpu="$1"
  shift
  local dataset
  cd "$ROOT"
  for dataset in "$@"; do
    echo "[$(date -Is)] gpu=${gpu} start ${dataset}"
    "$PYTHON" -u "$ROOT/MSTT/train.py" \
      --config "$CONFIG" \
      --task AllSpC2All \
      --penetration p10_p20 \
      --dataset-id "$dataset" \
      --gpu "$gpu"
    echo "[$(date -Is)] gpu=${gpu} done ${dataset}"
  done
}

if [[ "${1:-}" == "chain" ]]; then
  shift
  run_chain "$@"
  exit 0
fi

mkdir -p "$LOG_DIR"

launch() {
  local gpu="$1"
  shift
  local pidfile="$LOG_DIR/gpu${gpu}.pid"
  local log="$LOG_DIR/gpu${gpu}.log"
  if [[ -f "$pidfile" ]]; then
    local old
    old="$(cat "$pidfile")"
    if kill -0 "$old" 2>/dev/null; then
      echo "gpu ${gpu} already has chain pid ${old}" >&2
      exit 1
    fi
  fi
  nohup "$0" chain "$gpu" "$@" >"$log" 2>&1 &
  echo $! >"$pidfile"
  echo "gpu ${gpu}: pid $(cat "$pidfile") log ${log}"
}

launch 5 variable_30min_p10_p20 variable_1h_p10_p20
launch 7 variable_2h_p10_p20 variable_day_p10_p20
