#!/usr/bin/env bash

set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <benchmark>"
  echo "benchmark: longmemeval | locomo"
  exit 1
fi

BENCHMARK="$1"

if [[ "$BENCHMARK" != "longmemeval" && "$BENCHMARK" != "locomo" ]]; then
  echo "Unsupported benchmark: $BENCHMARK"
  exit 1
fi

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-python}"
DEVICE="${DEVICE:-cuda}"
GPU_ID="${GPU_ID:-0}"
RETRIEVE_TOP_K="${RETRIEVE_TOP_K:-20}"
FINAL_TOP_K="${FINAL_TOP_K:-10}"
BATCH_SIZE="${BATCH_SIZE:-16}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$PROJECT_ROOT/outputs/external_runs}"
RUN_SUITE="${RUN_SUITE:-all}"

case "$BENCHMARK" in
  longmemeval)
    RAW_INPUT_PATH="${LONGMEMEVAL_INPUT:-}"
    BENCHMARK_OUTPUT_ROOT="${LONGMEMEVAL_OUTPUT_ROOT:-$OUTPUT_ROOT/longmemeval}"
    SOURCE_JSONL_PATH="${LONGMEMEVAL_SOURCE_JSONL:-$BENCHMARK_OUTPUT_ROOT/source.jsonl}"
    NORMALIZED_DIR="${LONGMEMEVAL_NORMALIZED_DIR:-$OUTPUT_ROOT/longmemeval/normalized}"
    ;;
  locomo)
    RAW_INPUT_PATH="${LOCOMO_INPUT:-}"
    BENCHMARK_OUTPUT_ROOT="${LOCOMO_OUTPUT_ROOT:-$OUTPUT_ROOT/locomo}"
    SOURCE_JSONL_PATH="${LOCOMO_SOURCE_JSONL:-$BENCHMARK_OUTPUT_ROOT/source.jsonl}"
    NORMALIZED_DIR="${LOCOMO_NORMALIZED_DIR:-$OUTPUT_ROOT/locomo/normalized}"
    ;;
esac

E5_MODEL_PATH="${E5_MODEL_PATH:-${RETRIEVER_MODEL_PATH:-}}"
BGE_M3_MODEL_PATH="${BGE_M3_MODEL_PATH:-}"
RERANKER_MODEL_PATH="${RERANKER_MODEL_PATH:-}"

if [[ -z "$RAW_INPUT_PATH" ]]; then
  echo "Missing raw input path for $BENCHMARK."
  echo "Set LONGMEMEVAL_INPUT or LOCOMO_INPUT."
  exit 1
fi

mkdir -p "$NORMALIZED_DIR" "$BENCHMARK_OUTPUT_ROOT"

cd "$PROJECT_ROOT"

export PYTHONPATH="${PROJECT_ROOT}/src:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="$GPU_ID"

echo "[${BENCHMARK}] convert_raw_external"
if [[ ! -f "$SOURCE_JSONL_PATH" ]]; then
  "$PYTHON_BIN" -m memory_collapse.external_cli convert_raw_external \
    --benchmark "$BENCHMARK" \
    --input-path "$RAW_INPUT_PATH" \
    --output-path "$SOURCE_JSONL_PATH"
else
  echo "[${BENCHMARK}] skip convert_raw_external (already done)"
fi

echo "[${BENCHMARK}] prepare_external"
if [[ ! -f "$NORMALIZED_DIR/queries.jsonl" ]]; then
  "$PYTHON_BIN" -m memory_collapse.external_cli prepare_external \
    --benchmark "$BENCHMARK" \
    --input-path "$SOURCE_JSONL_PATH" \
    --output-dir "$NORMALIZED_DIR"
else
  echo "[${BENCHMARK}] skip prepare_external (already done)"
fi

run_variant() {
  local variant_name="$1"
  local retriever_name="$2"
  local retriever_model_path="${3:-}"
  local reranker_model_path="${4:-}"
  local run_output_dir="$BENCHMARK_OUTPUT_ROOT/$variant_name"

  # Skip if already completed successfully
  if [[ -f "$run_output_dir/retrieval_summary.json" ]]; then
    echo "[${BENCHMARK}] skip variant=${variant_name} (already done)"
    return 0
  fi

  mkdir -p "$run_output_dir"
  echo "[${BENCHMARK}] run_external_retrieval variant=${variant_name}"

  local cmd=(
    "$PYTHON_BIN" -m memory_collapse.external_cli run_external_retrieval
    --normalized-dir "$NORMALIZED_DIR"
    --output-dir "$run_output_dir"
    --retriever "$retriever_name"
    --device "$DEVICE"
    --retrieve-top-k "$RETRIEVE_TOP_K"
    --final-top-k "$FINAL_TOP_K"
    --batch-size "$BATCH_SIZE"
  )

  if [[ -n "$retriever_model_path" ]]; then
    cmd+=(--retriever-model "$retriever_model_path")
  fi

  if [[ -n "$reranker_model_path" ]]; then
    cmd+=(--reranker-model "$reranker_model_path")
  fi

  "${cmd[@]}"
}

if [[ "$RUN_SUITE" == "all" ]]; then
  run_variant "tfidf" "tfidf"

  if [[ -n "$E5_MODEL_PATH" ]]; then
    run_variant "dense_e5" "dense" "$E5_MODEL_PATH"
    if [[ -n "$RERANKER_MODEL_PATH" ]]; then
      run_variant "dense_e5_rerank" "dense" "$E5_MODEL_PATH" "$RERANKER_MODEL_PATH"
    fi
  else
    echo "[${BENCHMARK}] skip dense_e5: E5_MODEL_PATH not set"
  fi

  if [[ -n "$BGE_M3_MODEL_PATH" ]]; then
    run_variant "hybrid_bge_m3" "hybrid" "$BGE_M3_MODEL_PATH"
    if [[ -n "$RERANKER_MODEL_PATH" ]]; then
      run_variant "hybrid_bge_m3_rerank" "hybrid" "$BGE_M3_MODEL_PATH" "$RERANKER_MODEL_PATH"
    fi
  else
    echo "[${BENCHMARK}] skip hybrid_bge_m3: BGE_M3_MODEL_PATH not set"
  fi
else
  RETRIEVER="${RETRIEVER:-tfidf}"
  RETRIEVER_MODEL_PATH="${RETRIEVER_MODEL_PATH:-}"
  SINGLE_RUN_OUTPUT_DIR="${SINGLE_RUN_OUTPUT_DIR:-$BENCHMARK_OUTPUT_ROOT/${RETRIEVER}_manual}"
  mkdir -p "$SINGLE_RUN_OUTPUT_DIR"
  echo "[${BENCHMARK}] run_external_retrieval single retriever=${RETRIEVER}"
  CMD=(
    "$PYTHON_BIN" -m memory_collapse.external_cli run_external_retrieval
    --normalized-dir "$NORMALIZED_DIR"
    --output-dir "$SINGLE_RUN_OUTPUT_DIR"
    --retriever "$RETRIEVER"
    --device "$DEVICE"
    --retrieve-top-k "$RETRIEVE_TOP_K"
    --final-top-k "$FINAL_TOP_K"
    --batch-size "$BATCH_SIZE"
  )
  if [[ -n "$RETRIEVER_MODEL_PATH" ]]; then
    CMD+=(--retriever-model "$RETRIEVER_MODEL_PATH")
  fi
  if [[ -n "$RERANKER_MODEL_PATH" ]]; then
    CMD+=(--reranker-model "$RERANKER_MODEL_PATH")
  fi
  "${CMD[@]}"
fi

echo "[${BENCHMARK}] done"
echo "normalized_dir=$NORMALIZED_DIR"
echo "benchmark_output_root=$BENCHMARK_OUTPUT_ROOT"
