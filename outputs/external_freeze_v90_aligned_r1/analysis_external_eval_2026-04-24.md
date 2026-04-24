# External Eval Analysis Notes (2026-04-24)

## Status

- External evaluation is complete under `outputs/external_freeze_v90_aligned_r1`.
- Required result files are present:
  - `summary_longmemeval/external_end_to_end_summary.csv`
  - `summary_locomo/external_end_to_end_summary.csv`
  - `failure_slices/failure_slices.md`
- This note is based on result inspection only. No rerun was started.
- `CURRENT_EXTERNAL_EVAL_STATUS_2026-04-23.md` is historical runtime state and should not be used as the new execution starting point.

## SERVER_RUN_FLOW.md Completion Audit

`SERVER_RUN_FLOW.md` defines Route B as:

1. reuse `outputs/external_runs/`
2. skip retrieval rerun
3. run `scripts/run_external_answer_freeze.sh`
4. produce structured methods, Qwen/Llama reader baselines, summaries, and failure slices

Current output audit:

- Retrieval stage has all expected benchmark/variant outputs under `outputs/external_runs`:
  - benchmarks: `longmemeval`, `locomo`
  - variants: `tfidf`, `dense_e5`, `dense_e5_rerank`, `hybrid_bge_m3`, `hybrid_bge_m3_rerank`
  - each variant has `retrieval_diagnostics.jsonl` and `retrieval_summary.json`
- Answer-freeze stage has all expected detailed outputs:
  - 2 benchmarks x 5 retrieval variants x 5 methods = 50 method directories
  - 50 `metrics.json` files
  - 50 `prediction_diagnostics.jsonl` files
- Diagnostics row counts are complete:
  - every LongMemEval diagnostics file has 500 rows
  - every LoCoMo diagnostics file has 1,986 rows
- Summary consistency check passed:
  - 50 summary rows
  - 50 matching `metrics.json` files
  - no missing metrics files
  - no metric mismatches between summary CSVs and per-method `metrics.json`

Conclusion: the flow does not appear half-complete. There is no missing continuation step to run from `SERVER_RUN_FLOW.md` unless a new rerun with changed settings is explicitly desired.

## Correct Rerun Paths

Use these only if a later rerun is explicitly needed:

```bash
PROJECT_ROOT=/gfs/space/private/wujn/Learn/agent-memory-server-bundle-v90-20260422-final
MODEL_DIR=/gfs/space/private/wujn/Learn/agent_memory/outputs/default_direct_validity_v12/models
QWEN_MODEL_PATH=/gfs/space/private/wujn/Learn/Qwen2.5-7B-instruct
LLAMA_MODEL_PATH=/gfs/space/private/wujn/Learn/Llama-3.1-8B-Instruct
```

Do not use `artifacts/external_answer_models` from the bundle as the structured-method `MODEL_DIR`; it is not the model directory used by the successful run.

## Best Summary Results

### LongMemEval

Best accuracy:

| retrieval_variant | method | accuracy | exact_match | hit@1 | final_support_recall |
| --- | --- | ---: | ---: | ---: | ---: |
| hybrid_bge_m3_rerank | proposed_learned_direct_valid_resolver | 0.168 | 0.106 | 0.862 | 0.943 |
| hybrid_bge_m3_rerank | proposed_learned_direct_valid | 0.164 | 0.106 | 0.862 | 0.943 |
| dense_e5_rerank | proposed_learned_direct_valid | 0.162 | 0.108 | 0.854 | 0.939 |
| tfidf | proposed_learned_direct_valid_resolver | 0.162 | 0.098 | 0.760 | 0.901 |

Main deltas vs `retrieval_only_baseline`:

- `hybrid_bge_m3`: proposed direct valid improves accuracy by +0.018; resolver improves by +0.016.
- `hybrid_bge_m3_rerank`: proposed direct valid improves by +0.006; resolver improves by +0.010.
- `dense_e5`: proposed direct valid improves by +0.002; resolver improves by +0.010.
- Reader baselines are much lower than retrieval-only, typically about -0.108 to -0.124 accuracy.

### LoCoMo

Best accuracy:

| retrieval_variant | method | accuracy | exact_match | hit@1 | final_support_recall |
| --- | --- | ---: | ---: | ---: | ---: |
| hybrid_bge_m3_rerank | proposed_learned_direct_valid | 0.161 | 0.097 | 0.629 | 0.827 |
| dense_e5_rerank | proposed_learned_direct_valid | 0.160 | 0.096 | 0.628 | 0.814 |
| dense_e5_rerank | proposed_learned_direct_valid_resolver | 0.160 | 0.098 | 0.628 | 0.814 |
| hybrid_bge_m3_rerank | proposed_learned_direct_valid_resolver | 0.160 | 0.098 | 0.629 | 0.827 |

Main deltas vs `retrieval_only_baseline`:

- `dense_e5`: proposed direct valid improves accuracy by +0.019; resolver improves by +0.020.
- `hybrid_bge_m3`: proposed direct valid improves by +0.016; resolver improves by +0.015.
- `dense_e5_rerank`: proposed direct valid improves by +0.006; resolver improves by +0.006.
- `hybrid_bge_m3_rerank`: proposed direct valid improves by +0.007; resolver improves by +0.006.
- Reader baselines are much lower than retrieval-only, typically about -0.096 to -0.147 accuracy.

## Failure Pattern Notes

`failure_rows.csv` contains all evaluated rows, including rows with `failure_bucket=correct`. The incorrect-only counts are:

| benchmark | total rows | correct rows | incorrect rows |
| --- | ---: | ---: | ---: |
| longmemeval | 12,500 | 1,322 | 11,178 |
| locomo | 49,650 | 4,331 | 45,319 |

Incorrect-only failure buckets:

| benchmark | supported_but_wrong | confident_selection_error | retrieval_miss | comparative_reasoning | empty_prediction |
| --- | ---: | ---: | ---: | ---: | ---: |
| longmemeval | 6,068 | 3,894 | 438 | 736 | 42 |
| locomo | 24,800 | 14,481 | 5,644 | 85 | 309 |

Top incorrect question buckets:

- LongMemEval: `quantity`, `generic`, `price`, `title_or_name`, `duration`.
- LoCoMo: `generic`, `when`, `item_name`, `title_or_name`, `event_name`.

For proposed methods specifically:

- LongMemEval direct valid: 2,108 incorrect rows; main buckets are `confident_selection_error` 960 and `supported_but_wrong` 941.
- LongMemEval resolver: 2,102 incorrect rows; it shifts errors toward `confident_selection_error` 1,215 and away from `supported_but_wrong` 679.
- LoCoMo direct valid: 8,498 incorrect rows; main buckets are `confident_selection_error` 3,724, `supported_but_wrong` 3,618, and `retrieval_miss` 1,104.
- LoCoMo resolver: 8,504 incorrect rows; it shifts errors toward `confident_selection_error` 4,761 and away from `supported_but_wrong` 2,588.

## Next Analysis Direction

- Prioritize error inspection for `supported_but_wrong` and `confident_selection_error`, because they dominate both benchmarks.
- For LongMemEval, inspect `quantity`, `generic`, and `price` questions first.
- For LoCoMo, inspect `generic`, `when`, and `item_name` questions first.
- Treat open-model reader baselines as weak baselines in this run; they should not drive rerun decisions without first checking prompt/parsing behavior.
