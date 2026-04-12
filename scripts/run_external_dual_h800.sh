#!/usr/bin/env bash

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
LOG_ROOT="${LOG_ROOT:-$PROJECT_ROOT/outputs/external_runs/logs}"
LONGMEMEVAL_GPU_ID="${LONGMEMEVAL_GPU_ID:-0}"
LOCOMO_GPU_ID="${LOCOMO_GPU_ID:-1}"

mkdir -p "$LOG_ROOT"
cd "$PROJECT_ROOT"

PID_LONG=""
PID_LOCO=""

cleanup() {
  [[ -n "$PID_LONG" ]] && kill "$PID_LONG" 2>/dev/null || true
  [[ -n "$PID_LOCO" ]] && kill "$PID_LOCO" 2>/dev/null || true
}
trap cleanup EXIT

echo "[dual] launching LongMemEval on GPU ${LONGMEMEVAL_GPU_ID}"
GPU_ID="$LONGMEMEVAL_GPU_ID" \
PROJECT_ROOT="$PROJECT_ROOT" \
bash "$PROJECT_ROOT/scripts/run_external_benchmark.sh" longmemeval \
  >"$LOG_ROOT/longmemeval.log" 2>&1 &
PID_LONG=$!

echo "[dual] launching LoCoMo on GPU ${LOCOMO_GPU_ID}"
GPU_ID="$LOCOMO_GPU_ID" \
PROJECT_ROOT="$PROJECT_ROOT" \
bash "$PROJECT_ROOT/scripts/run_external_benchmark.sh" locomo \
  >"$LOG_ROOT/locomo.log" 2>&1 &
PID_LOCO=$!

FAIL=0

if ! wait "$PID_LONG"; then
  echo "[dual] LongMemEval failed. Check $LOG_ROOT/longmemeval.log"
  FAIL=1
else
  echo "[dual] LongMemEval finished."
fi

if ! wait "$PID_LOCO"; then
  echo "[dual] LoCoMo failed. Check $LOG_ROOT/locomo.log"
  FAIL=1
else
  echo "[dual] LoCoMo finished."
fi

if [[ "$FAIL" -ne 0 ]]; then
  exit 1
fi

echo "[dual] all runs finished successfully"
