#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$PROJECT_ROOT"

export PYTHONPATH="$PROJECT_ROOT/src:${PYTHONPATH:-}"
export E5_MODEL_PATH=/gfs/space/private/wujn/Learn/models/intfloat/e5-base-v2
export BGE_M3_MODEL_PATH=/gfs/space/private/wujn/Learn/models/BAAI/bge-m3
export RERANKER_MODEL_PATH=/gfs/space/private/wujn/Learn/models/BAAI/bge-reranker-v2-m3
export LONGMEMEVAL_INPUT=/gfs/space/private/wujn/Learn/datasets/longmemeval/longmemeval_s
export LOCOMO_INPUT=/gfs/space/private/wujn/Learn/datasets/locomo/locomo10.json

run_variant() {
  local bench="$1"
  local gpu="$2"
  local variant="$3"
  local retriever="$4"
  local retriever_model="$5"
  local reranker_model="$6"

  local bench_root="$PROJECT_ROOT/outputs/retrieval_v1/${bench}_results"
  local single_out="$bench_root/$variant"

  FORCE_RERUN=1 \
  RUN_SUITE=single \
  DEVICE=cuda \
  GPU_ID="$gpu" \
  PROJECT_ROOT="$PROJECT_ROOT" \
  LONGMEMEVAL_OUTPUT_ROOT="$PROJECT_ROOT/outputs/retrieval_v1/longmemeval_results" \
  LONGMEMEVAL_SOURCE_JSONL="$PROJECT_ROOT/outputs/retrieval_v1/longmemeval_results/source.jsonl" \
  LONGMEMEVAL_NORMALIZED_DIR="$PROJECT_ROOT/outputs/retrieval_v1/longmemeval_results/normalized" \
  LOCOMO_OUTPUT_ROOT="$PROJECT_ROOT/outputs/retrieval_v1/locomo_results" \
  LOCOMO_SOURCE_JSONL="$PROJECT_ROOT/outputs/retrieval_v1/locomo_results/source.jsonl" \
  LOCOMO_NORMALIZED_DIR="$PROJECT_ROOT/outputs/retrieval_v1/locomo_results/normalized" \
  RETRIEVER="$retriever" \
  RETRIEVER_MODEL_PATH="$retriever_model" \
  RERANKER_MODEL_PATH="$reranker_model" \
  SINGLE_RUN_OUTPUT_DIR="$single_out" \
  bash "$PROJECT_ROOT/scripts/run_external_benchmark.sh" "$bench"
}

run_benchmark() {
  local bench="$1"
  local gpu="$2"
  run_variant "$bench" "$gpu" dense_e5 dense "$E5_MODEL_PATH" ""
  run_variant "$bench" "$gpu" dense_e5_rerank dense "$E5_MODEL_PATH" "$RERANKER_MODEL_PATH"
  run_variant "$bench" "$gpu" hybrid_bge_m3 hybrid "$BGE_M3_MODEL_PATH" ""
  run_variant "$bench" "$gpu" hybrid_bge_m3_rerank hybrid "$BGE_M3_MODEL_PATH" "$RERANKER_MODEL_PATH"
}

(run_benchmark longmemeval 0 > "$PROJECT_ROOT/outputs/retrieval_v1/logs/longmemeval_v1.log" 2>&1) &
PID_LONG=$!
(run_benchmark locomo 1 > "$PROJECT_ROOT/outputs/retrieval_v1/logs/locomo_v1.log" 2>&1) &
PID_LOCO=$!

echo "PID_LONG=$PID_LONG"
echo "PID_LOCO=$PID_LOCO"
wait "$PID_LONG"
wait "$PID_LOCO"
