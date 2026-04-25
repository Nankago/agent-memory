# Reader Probe V2

Use this after the first Qwen/LLaMA run produced very low reader scores.

The V2 runner changes are:

- evidence is quoted as snippets, so the model is told not to continue it;
- the final question and JSON instruction are at the end of the prompt;
- tokenizer truncation keeps the end of the prompt;
- each snippet is pre-truncated to reduce prompt overflow;
- default generation is reduced to 24 new tokens;
- output parsing reads `{"answer":"..."}` first.

## What This Probe Runs

Only the strongest retrieval line:

```text
hybrid_bge_m3_rerank
```

Across:

- LongMemEval + Qwen
- LoCoMo + Qwen
- LongMemEval + LLaMA
- LoCoMo + LLaMA

It writes to:

```text
outputs/external_reader_probe_v2/
```

It does not overwrite the previous full freeze.

## Server Commands

From the bundle root:

```bash
export PROJECT_ROOT=$(pwd)
export PYTHON_BIN=python
export PYTHONPATH=$PROJECT_ROOT/src

export RETRIEVAL_ROOT=$PROJECT_ROOT/outputs/external_runs

export OPEN_MODEL_HOME=/path/to/local_hf_models
export QWEN_MODEL_PATH=$OPEN_MODEL_HOME/Qwen2.5-7B-Instruct
export LLAMA_MODEL_PATH=$OPEN_MODEL_HOME/Meta-Llama-3.1-8B-Instruct

chmod +x scripts/run_external_reader_probe_v2.sh
bash scripts/run_external_reader_probe_v2.sh
```

If the model folders have different names:

```bash
export QWEN_MODEL_PATH=/path/to/Qwen2.5-7B-Instruct
export LLAMA_MODEL_PATH=/path/to/Meta-Llama-3.1-8B-Instruct
```

## Files To Bring Back

```text
outputs/external_reader_probe_v2/summary_longmemeval/external_end_to_end_summary.csv
outputs/external_reader_probe_v2/summary_locomo/external_end_to_end_summary.csv
outputs/external_reader_probe_v2/failure_slices/failure_rows.csv
outputs/external_reader_probe_v2/failure_slices/failure_slices.md
outputs/external_reader_probe_v2/<benchmark>/hybrid_bge_m3_rerank/<reader_method>/prediction_diagnostics.jsonl
outputs/external_reader_probe_v2/<benchmark>/hybrid_bge_m3_rerank/<reader_method>/metrics.json
```

## How To Judge The Probe

The first run had abnormal reader outputs:

- LongMemEval reader accuracy around 0.038-0.040 on the strong line;
- LoCoMo reader accuracy around 0.007 on the strong line;
- most predictions were long continuations rather than short answers.

V2 is working if:

- average prediction length drops sharply;
- most predictions look like short answer phrases;
- accuracy moves closer to retrieval-only, even if it does not beat the structured method.

If V2 is still weak, the next baseline should be candidate-constrained:

```text
show extracted candidate answers + evidence, ask the model to select one candidate answer
```
