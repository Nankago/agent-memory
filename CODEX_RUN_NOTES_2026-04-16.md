# Codex Run Notes (2026-04-16 UTC)

## Request Summary
User asked to run two full external end-to-end evaluations:
- benchmark: `longmemeval`
- benchmark: `locomo`
- methods: `retrieval_only_baseline`, `proposed_learned_direct_valid`, `proposed_learned_direct_valid_resolver`
- retrieval variants: `tfidf`, `dense_e5`, `dense_e5_rerank`, `hybrid_bge_m3`, `hybrid_bge_m3_rerank`

## What Actually Happened
1. Command syntax check: user-provided `run_external_end_to_end` command format is valid.
2. Multiple run attempts were started.
3. Several Python processes remained stuck in `D (disk sleep)` state and did not make forward progress.
4. No new summary CSV was produced during this session.
5. Existing historical summary files were already complete (30 rows = 2 benchmarks x 5 variants x 3 methods):
   - `outputs/external_runs/external_end_to_end_summary_combined.csv`
   - `outputs/external_runs/external_end_to_end_summary.csv`
   - `outputs/external_runs/external_end_to_end_metrics.csv`

## Failure / Confusion Root Cause
1. Main misunderstanding: `run_external_end_to_end` is answer-level post-processing on existing retrieval diagnostics, not GPU-heavy retrieval inference.
2. Therefore GPU utilization near 0% is expected for this command.
3. GPU-heavy stage is `run_external_retrieval` (dense/hybrid/rerank) with `--device cuda`.
4. Additional operational issue: residual/stuck processes in `D (disk sleep)` made repeated reruns unreliable.

## Code Evidence (for future reference)
- `run_external_end_to_end` parser has no `--device` option:
  - `src/memory_collapse/cli.py`
- `run_external_retrieval` has `--device` and is the stage that uses torch/transformers:
  - `src/memory_collapse/cli.py`
  - `src/memory_collapse/external_retrieval.py`
- End-to-end stage loads retrieval diagnostics and runs scoring/aggregation logic:
  - `src/memory_collapse/external_pipeline.py`

## Final Conclusion
- The user command was syntactically correct, but mismatched with the expectation of GPU utilization.
- To use GPU, run retrieval stage first with `--device cuda`, then run `run_external_end_to_end` for evaluation summary.

## Recommended Next-Session Procedure
1. Verify no stale run process before starting.
2. If goal includes GPU usage, run:
   - `run_external_retrieval --device cuda`
3. Then run:
   - `run_external_end_to_end` for metrics/summary tables.
4. If process enters persistent `D (disk sleep)`, stop and switch node/environment or inspect underlying storage I/O.

## Existing Valid Summary Artifact
Use this as the current valid aggregate table unless a fresh successful rerun is completed:
- `outputs/external_runs/external_end_to_end_summary_combined.csv`
