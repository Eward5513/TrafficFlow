#!/usr/bin/env bash
# state_aux_gate AllSpC2All on the four p10_p20 frequencies.
# Usage: bash MSTT/run_all_sp_c_state_aux_gate.sh GPU [GPU ...]
# Datasets are dealt in order. Each GPU runs one dataset at a time and starts
# the next only after the previous process exits.
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: $0 GPU [GPU ...]" >&2
  exit 2
fi

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON="${PYTHON:-/home/tzhang174/miniforge3/envs/trafficflow/bin/python3}"
CONFIG="$ROOT/MSTT/configs/all_sp_c_state_aux_gate_d128_projected_all.json"
LOG_DIR="/data5/tzhang174/TrafficFlow/xuancheng26/training_results_3s/reports/mstt_state_aux_gate_d128"
DATASETS=(
  variable_30min_p10_p20
  variable_1h_p10_p20
  variable_2h_p10_p20
  variable_day_p10_p20
)

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

seen=" "
for gpu in "$@"; do
  if [[ ! "$gpu" =~ ^[0-9]+$ ]]; then
    echo "GPU must be an integer, got ${gpu}" >&2
    exit 2
  fi
  if [[ "$seen" == *" ${gpu} "* ]]; then
    echo "duplicate GPU ${gpu}" >&2
    exit 2
  fi
  seen+="${gpu} "
done

mkdir -p "$LOG_DIR"

gpus=("$@")
gpu_count="${#gpus[@]}"
declare -a buckets
for ((slot=0; slot<gpu_count; slot++)); do
  buckets[slot]=""
done
for ((index=0; index<${#DATASETS[@]}; index++)); do
  slot=$((index % gpu_count))
  buckets[slot]+="${DATASETS[index]} "
done

launch() {
  local gpu="$1"
  shift
  if [[ $# -eq 0 ]]; then
    return 0
  fi
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
  echo "gpu ${gpu}: datasets $* pid $(cat "$pidfile") log ${log}"
}

for ((slot=0; slot<gpu_count; slot++)); do
  # shellcheck disable=SC2086
  launch "${gpus[slot]}" ${buckets[slot]}
done
