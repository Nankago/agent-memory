# External Retrieval Server Run Guide

This bundle is prepared for running the external retrieval stack on a Linux GPU server.

## What this bundle can run

Per benchmark, the default script runs this sequence:

1. `prepare_external`
2. `tfidf`
3. `dense_e5`
4. `dense_e5_rerank`
5. `hybrid_bge_m3`
6. `hybrid_bge_m3_rerank`

The dual-GPU launcher runs:

- `LongMemEval` on GPU 0
- `LoCoMo` on GPU 1

## Expected input

The scripts now handle raw benchmark files directly.

They first run:

1. `convert_raw_external`
2. `prepare_external`
3. retrieval variants

Supported raw inputs:

- `LongMemEval`: raw JSON or JSONL with fields such as `question`, `answer`, `haystack_sessions`, `answer_session_ids`
- `LoCoMo`: raw `locomo10.json` style data with `conversation` and `qa`

## Install

Use Python 3.12+.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements_external.txt
```

If your server already has a matching PyTorch install, keep that one and only install the remaining packages if needed.

This lightweight package uses the dedicated module entrypoint:

```bash
python -m memory_collapse.external_cli --help
```

## Run everything on two GPUs

```bash
chmod +x scripts/run_external_benchmark.sh scripts/run_external_dual_h800.sh

export PROJECT_ROOT=$(pwd)
export PYTHON_BIN=python

export LONGMEMEVAL_INPUT=/path/to/longmemeval.jsonl
export LOCOMO_INPUT=/path/to/locomo.jsonl

export E5_MODEL_PATH=/path/to/models/intfloat/e5-base-v2
export BGE_M3_MODEL_PATH=/path/to/models/BAAI/bge-m3
export RERANKER_MODEL_PATH=/path/to/models/BAAI/bge-reranker-v2-m3

export RETRIEVE_TOP_K=50
export FINAL_TOP_K=10
export BATCH_SIZE=16

bash scripts/run_external_dual_h800.sh
```

## Output layout

Logs:

- `outputs/external_runs/logs/longmemeval.log`
- `outputs/external_runs/logs/locomo.log`

Per-benchmark outputs:

- `outputs/external_runs/<benchmark>/normalized`
- `outputs/external_runs/<benchmark>/tfidf`
- `outputs/external_runs/<benchmark>/dense_e5`
- `outputs/external_runs/<benchmark>/dense_e5_rerank`
- `outputs/external_runs/<benchmark>/hybrid_bge_m3`
- `outputs/external_runs/<benchmark>/hybrid_bge_m3_rerank`

Each benchmark also writes:

- `source.jsonl`

Each variant writes:

- `retrieval_diagnostics.jsonl`
- `retrieval_summary.json`

## Notes

- The scripts already set `PYTHONPATH=src`.
- Dense retrieval and reranking require `torch` and `transformers`.
- The current `hybrid` mode is an RRF fusion of TF-IDF and dense retrieval.
