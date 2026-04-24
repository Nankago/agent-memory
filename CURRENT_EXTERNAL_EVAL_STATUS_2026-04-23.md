# Current External Eval Status (2026-04-23)

> Historical run log only. For a fresh Codex handoff, use `HANDOFF_FOR_NEW_CODEX_2026-04-24.md` instead.

## Decision

- Use the bundle as the execution root:
  - `/gfs/space/private/wujn/Learn/agent-memory-server-bundle-v90-20260422-final`
- Follow `SERVER_RUN_FLOW.md` Route B:
  - reuse existing retrieval outputs
  - run `scripts/run_external_answer_freeze.sh`
- Include both open-model reader baselines requested by collaborator:
  - `Qwen2.5-7B-Instruct`
  - `Llama-3.1-8B-Instruct`

## Why this route

- The local git repo `/gfs/space/private/wujn/Learn/agent_memory` is current with `origin/main`, but it does not contain the newer freeze scripts and reader-baseline pipeline used by the bundle.
- The bundle contains the intended server-run flow:
  - `SERVER_RUN_FLOW.md`
  - `README_SERVER_RUN.md`
  - `scripts/run_external_answer_freeze.sh`
  - `scripts/run_external_open_model_baselines.sh`
  - `src/memory_collapse/external_reader_baselines.py`
  - `src/memory_collapse/external_pipeline_v0.py`

## Data and model state

- Existing retrieval outputs were copied into:
  - `/gfs/space/private/wujn/Learn/agent-memory-server-bundle-v90-20260422-final/outputs/external_runs`
- Structured answer-model artifacts already exist in:
  - `/gfs/space/private/wujn/Learn/agent-memory-server-bundle-v90-20260422-final/artifacts/external_answer_models`
- Qwen model path:
  - `/gfs/space/private/wujn/Learn/Qwen2.5-7B-instruct`
- Llama 3.1 model path:
  - `/gfs/space/private/wujn/Learn/Meta-Llama-3.1-8B-Instruct`
- Convenience symlink for Llama 3.1:
  - `/gfs/space/private/wujn/Learn/Llama-3.1-8B-Instruct`

## Llama 3.1 note

- `/gfs/space/private/wujn/Learn/Llama-3-8b-Instruct` is not `Llama-3.1-8B-Instruct`.
- Official `Llama-3.1-8B-Instruct` was downloaded from ModelScope mirror into:
  - `/gfs/space/private/wujn/Learn/Meta-Llama-3.1-8B-Instruct`
- Verified characteristics from `config.json`:
  - `max_position_embeddings = 131072`
  - `rope_scaling` present

## Runtime checks

- GPU is required for the requested reader baselines in any practical sense.
- Escalated `nvidia-smi` confirmed:
  - 2 x `NVIDIA H800 80GB`
  - GPU 0 already occupied by another Python job
  - GPU 1 idle at check time
- Plan:
  - bind this run to `CUDA_VISIBLE_DEVICES=1`

## Python environment checks

- Global Python is usable for this run:
  - `torch 2.7.0a0+7c8ec84dab.nv25.03`
  - `transformers 5.3.0`
  - `pandas 2.2.3`
  - `scikit-learn 1.6.1`
  - `PyYAML 6.0.2`
- Bundle CLI import check passed:
  - `python -m memory_collapse.cli --help`

## First run issue and fix

- Initial attempt used:
  - `MODEL_DIR=$PROJECT_ROOT/artifacts/external_answer_models`
- That failed during structured-method loading with:
  - `FileNotFoundError: Missing relevance model under .../artifacts/external_answer_models`
- Root cause:
  - bundle `artifacts/external_answer_models` is a minimal subset and does not satisfy the full `models/` layout expected by `run_external_end_to_end`
- Working fix:
  - point `MODEL_DIR` to the complete local model directory instead:
  - `/gfs/space/private/wujn/Learn/agent_memory/outputs/default_direct_validity_v12/models`

## Current run status

- Relaunched with corrected `MODEL_DIR`
- Confirmed active process:
  - `bash scripts/run_external_answer_freeze.sh`
  - `python -m memory_collapse.cli run_external_end_to_end ...`
- Previous foreground terminal session id:
  - `65557`
- Confirmed outputs are being written under:
  - `/gfs/space/private/wujn/Learn/agent-memory-server-bundle-v90-20260422-final/outputs/external_freeze_v90_aligned_r1`
- Example files already created:
  - `longmemeval/tfidf/retrieval_only_baseline/metrics.csv`
  - `longmemeval/tfidf/retrieval_only_baseline/prediction_diagnostics.jsonl`
  - `longmemeval/tfidf/retrieval_only_baseline/metrics.json`

## Background relaunch for SSH disconnect safety

- On `2026-04-24`, the foreground run was intentionally interrupted and relaunched in detached mode so it can survive SSH disconnect.
- Detached launch method:
  - `setsid + nohup`
- Current detached shell PID:
  - `4143725`
- Current detached Python PID:
  - `4143744`
- Latest PID file:
  - `/gfs/space/private/wujn/Learn/agent-memory-server-bundle-v90-20260422-final/logs/external_answer_freeze_latest.pid`
- Current detached log:
  - `/gfs/space/private/wujn/Learn/agent-memory-server-bundle-v90-20260422-final/logs/external_answer_freeze_20260423T160211Z.log`
- Verified detached state:
  - shell process parent is now PID `1`
  - Python worker is running under that detached shell

## How to inspect later

```bash
cd /gfs/space/private/wujn/Learn/agent-memory-server-bundle-v90-20260422-final

cat logs/external_answer_freeze_latest.pid
ps -fp "$(cat logs/external_answer_freeze_latest.pid)"
pgrep -af "memory_collapse.cli run_external_end_to_end|run_external_reader_baseline|run_external_answer_freeze.sh"
tail -f logs/external_answer_freeze_20260423T160211Z.log
```

## GPU usage note

- At `2026-04-23 23:53 UTC`, both GPUs were idle at the instant of check.
- This does not mean the evaluation is GPU-free.
- The currently running step is still the CPU-side structured evaluation:
  - `run_external_end_to_end`
- GPU will be needed later when the flow reaches reader baselines:
  - `reader_qwen25_7b_instruct`
  - `reader_llama31_8b_instruct`
- Practical implication:
  - GPU 1 may look idle now, but it is reserved for the later reader-baseline stage of this same run.

## Recovery plan if GPU stage fails later

- If the current flow later fails specifically at the reader-baseline stage, the earlier structured outputs can be kept.
- In that case, do not rerun the whole freeze flow immediately.
- Resume only the GPU-dependent reader stage with:

```bash
cd /gfs/space/private/wujn/Learn/agent-memory-server-bundle-v90-20260422-final

export PROJECT_ROOT="$PWD"
export PYTHON_BIN=python
export PYTHONPATH="$PROJECT_ROOT/src"
export RETRIEVAL_ROOT="$PROJECT_ROOT/outputs/external_runs"
export FREEZE_ROOT="$PROJECT_ROOT/outputs/external_freeze_v90_aligned_r1"
export QWEN_MODEL_PATH=/gfs/space/private/wujn/Learn/Qwen2.5-7B-instruct
export LLAMA_MODEL_PATH=/gfs/space/private/wujn/Learn/Llama-3.1-8B-Instruct
export CUDA_VISIBLE_DEVICES=1
export READER_DEVICE=cuda

bash scripts/run_external_open_model_baselines.sh

python scripts/analyze_external_failures.py \
  --pred-root "$FREEZE_ROOT" \
  --retrieval-root "$RETRIEVAL_ROOT" \
  --output-dir "$FREEZE_ROOT/failure_slices"
```

## Launch command

```bash
cd /gfs/space/private/wujn/Learn/agent-memory-server-bundle-v90-20260422-final

export PROJECT_ROOT="$PWD"
export PYTHON_BIN=python
export PYTHONPATH="$PROJECT_ROOT/src"
export RETRIEVAL_ROOT="$PROJECT_ROOT/outputs/external_runs"
export FREEZE_ROOT="$PROJECT_ROOT/outputs/external_freeze_v90_aligned_r1"
export MODEL_DIR=/gfs/space/private/wujn/Learn/agent_memory/outputs/default_direct_validity_v12/models
export QWEN_MODEL_PATH=/gfs/space/private/wujn/Learn/Qwen2.5-7B-instruct
export LLAMA_MODEL_PATH=/gfs/space/private/wujn/Learn/Llama-3.1-8B-Instruct
export CUDA_VISIBLE_DEVICES=1
export READER_DEVICE=cuda

bash scripts/run_external_answer_freeze.sh
```

## Expected outputs

- Main output root:
  - `/gfs/space/private/wujn/Learn/agent-memory-server-bundle-v90-20260422-final/outputs/external_freeze_v90_aligned_r1`
- Most important summaries:
  - `summary_longmemeval/external_end_to_end_summary.csv`
  - `summary_locomo/external_end_to_end_summary.csv`
  - `failure_slices/failure_slices.md`
