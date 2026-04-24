# Handoff For New Codex (2026-04-24)

## Current status

- The external evaluation requested in this thread is already finished.
- Do not try to resume the old background process.
- Treat all old PID and session references as historical only.

## Canonical execution root

- Use the bundle directory as the canonical project root:
  - `/gfs/space/private/wujn/Learn/agent-memory-server-bundle-v90-20260422-final`

## Final outputs

- Main output root:
  - `/gfs/space/private/wujn/Learn/agent-memory-server-bundle-v90-20260422-final/outputs/external_freeze_v90_aligned_r1`
- Main summary tables:
  - `/gfs/space/private/wujn/Learn/agent-memory-server-bundle-v90-20260422-final/outputs/external_freeze_v90_aligned_r1/summary_longmemeval/external_end_to_end_summary.csv`
  - `/gfs/space/private/wujn/Learn/agent-memory-server-bundle-v90-20260422-final/outputs/external_freeze_v90_aligned_r1/summary_locomo/external_end_to_end_summary.csv`
- Full metrics tables:
  - `/gfs/space/private/wujn/Learn/agent-memory-server-bundle-v90-20260422-final/outputs/external_freeze_v90_aligned_r1/summary_longmemeval/external_end_to_end_metrics.csv`
  - `/gfs/space/private/wujn/Learn/agent-memory-server-bundle-v90-20260422-final/outputs/external_freeze_v90_aligned_r1/summary_locomo/external_end_to_end_metrics.csv`
- Failure analysis outputs:
  - `/gfs/space/private/wujn/Learn/agent-memory-server-bundle-v90-20260422-final/outputs/external_freeze_v90_aligned_r1/failure_slices/failure_rows.csv`
  - `/gfs/space/private/wujn/Learn/agent-memory-server-bundle-v90-20260422-final/outputs/external_freeze_v90_aligned_r1/failure_slices/failure_slices_by_question.csv`
  - `/gfs/space/private/wujn/Learn/agent-memory-server-bundle-v90-20260422-final/outputs/external_freeze_v90_aligned_r1/failure_slices/failure_slices_by_bucket.csv`
  - `/gfs/space/private/wujn/Learn/agent-memory-server-bundle-v90-20260422-final/outputs/external_freeze_v90_aligned_r1/failure_slices/failure_slices.md`

## What was actually run

- Route used:
  - `SERVER_RUN_FLOW.md` Route B
- Meaning:
  - reuse existing retrieval outputs
  - run the answer-level freeze
  - include both reader baselines requested by collaborator
- Models used for reader baselines:
  - Qwen path:
    - `/gfs/space/private/wujn/Learn/Qwen2.5-7B-instruct`
  - Llama 3.1 path:
    - `/gfs/space/private/wujn/Learn/Llama-3.1-8B-Instruct`

## Important implementation detail

- Do not use the bundle's minimal artifact directory as `MODEL_DIR` for structured methods:
  - `/gfs/space/private/wujn/Learn/agent-memory-server-bundle-v90-20260422-final/artifacts/external_answer_models`
- That subset is insufficient for `run_external_end_to_end`.
- The correct `MODEL_DIR` used for the successful run was:
  - `/gfs/space/private/wujn/Learn/agent_memory/outputs/default_direct_validity_v12/models`

## Why the older handoff doc is confusing

- It mixes historical run-time state with current state.
- It still mentions old session ids, old PIDs, and background-resume mechanics.
- Those details were useful while the job was live, but they are no longer the right starting point for a new Codex.
- A new Codex should start from the completed outputs, not from the stale process state.

## If the new Codex needs to verify completion

Run:

```bash
cd /gfs/space/private/wujn/Learn/agent-memory-server-bundle-v90-20260422-final

ls outputs/external_freeze_v90_aligned_r1/summary_longmemeval
ls outputs/external_freeze_v90_aligned_r1/summary_locomo
ls outputs/external_freeze_v90_aligned_r1/failure_slices
```

Expected files include:

- `external_end_to_end_summary.csv`
- `external_end_to_end_metrics.csv`
- `failure_slices.md`

## If the new Codex needs to rerun later

Use this command pattern:

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

## Recommended starting point for the next Codex

- First read this file.
- Then inspect:
  - `SERVER_RUN_FLOW.md`
  - `outputs/external_freeze_v90_aligned_r1/summary_longmemeval/external_end_to_end_summary.csv`
  - `outputs/external_freeze_v90_aligned_r1/summary_locomo/external_end_to_end_summary.csv`
  - `outputs/external_freeze_v90_aligned_r1/failure_slices/failure_slices.md`
- Only after that decide whether the next task is:
  - analysis of results
  - exporting tables
  - plotting
  - rerunning with different models or settings
