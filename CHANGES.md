# Optimization Changes

Date: 2026-04-12

## 1. Fix double `_normalize_context` call (`external.py`)

**Before:** `_normalize_context(context)` was called twice per query — once to iterate memories, once to compute `num_memories`.
**After:** Result cached in `normalized_context` and reused.
**Impact:** Eliminates redundant list traversal for every query.

---

## 2. Fix TF-IDF vectorizer fit strategy (`external_retrieval.py`)

**Before:** `fit_transform(memory_texts + [case.prompt])` — the query was included in the corpus, polluting IDF weights with query terms.
**After:** `fit_transform(memory_texts)` then `transform([case.prompt])` — IDF is computed on the memory corpus only, which is the correct retrieval setup.
**Impact:** More accurate TF-IDF scores; query terms no longer inflate their own IDF.

---

## 3. BF16 model loading for dense retriever and reranker (`external_retrieval.py`)

**Before:** Models loaded in default float32.
**After:** `torch_dtype=torch.bfloat16` when `device != "cpu"`.
**Impact:** ~2x reduction in GPU memory usage; faster matrix ops on H800 (native BF16 tensor cores).

---

## 4. `autocast` for dense encoding and reranking (`external_retrieval.py`)

**Before:** Inference ran without mixed-precision context.
**After:** `torch.cuda.amp.autocast(dtype=torch.bfloat16)` wraps all forward passes on GPU. A `_null_context` fallback is used on CPU so the code path is identical.
Logits and embeddings are cast back to float32 before `.numpy()` to avoid downstream precision issues.
**Impact:** Further speedup on H800 with no change to output precision.

---

## 5. Per-query progress logging (`external_retrieval.py`)

**Before:** No output during retrieval loop; runs appeared frozen on large benchmarks.
**After:** `print(f"[retrieval] {i}/{total} query_id=...")` with `flush=True` on every query.
**Impact:** Visible progress in both interactive runs and log files.

---

## 6. Orphan process protection in dual-GPU launcher (`run_external_dual_h800.sh`)

**Before:** If the script exited before reaching `wait`, background subprocesses became orphans.
**After:** `trap cleanup EXIT` kills both PIDs on any exit (normal, error, or signal). PIDs are initialized to empty strings so the trap is safe to call even if a launch failed.
**Impact:** No leaked GPU processes on failure.

---

## 7. Skip-if-done for all pipeline stages (`run_external_benchmark.sh`)

**Before:** Every run re-executed all stages from scratch, even if outputs already existed.
**After:**
- `convert_raw_external` skipped if `source.jsonl` exists.
- `prepare_external` skipped if `normalized/queries.jsonl` exists.
- Each retrieval variant skipped if `retrieval_summary.json` exists in its output dir.

**Impact:** Interrupted runs resume from the last completed stage; no wasted compute on reruns.

---

## 8. Fix `RETRIEVE_TOP_K` default value (`run_external_benchmark.sh`)

**Problem identified after first run:** Both benchmarks have small haystacks (LongMemEval avg=50, LoCoMo avg=28 memories per query). With `RETRIEVE_TOP_K=50`, the retriever was fetching the entire haystack every time, making `recall@retrieve_k` trivially 1.000 for all methods — no retrieval discrimination at all.

**Before:** `RETRIEVE_TOP_K` defaulted to `50`.
**After:** Default changed to `20`, which is genuinely below the haystack size for both benchmarks (~40% of LongMemEval, ~72% of LoCoMo), forcing the retriever to actually filter.

**Impact:** `recall@retrieve_k` now meaningfully differentiates tfidf vs dense vs hybrid. The reranking stage remains at `FINAL_TOP_K=10`.

**Note:** Previous results in `outputs/external_runs/` were generated with `RETRIEVE_TOP_K=50` and should be discarded. Clear the variant output dirs and rerun.

---

## Files changed

| File | Changes |
|------|---------|
| `src/memory_collapse/external.py` | Fix #1 |
| `src/memory_collapse/external_retrieval.py` | Fix #2, #3, #4, #5 |
| `scripts/run_external_benchmark.sh` | Fix #7, Fix #8 |
| `scripts/run_external_dual_h800.sh` | Fix #6 |

---

# Formal External Retrieval Freeze Run

Date: 2026-04-13

## 1. Formal freeze mode for benchmark script (`run_external_benchmark.sh`)

Added two control flags:

- `FORMAL_FREEZE_SETTINGS=1`:
  - Force `RETRIEVE_TOP_K=20`
  - Force `FINAL_TOP_K=10`
  - Force `RUN_SUITE=all` (both benchmarks run the same full variant set)
- `FORCE_RERUN=1`:
  - If `retrieval_summary.json` already exists, delete that variant output dir and rerun instead of skipping.

In formal freeze mode, the script also requires all model paths to be set and runs exactly:

1. `tfidf`
2. `dense_e5`
3. `dense_e5_rerank`
4. `hybrid_bge_m3`
5. `hybrid_bge_m3_rerank`

## 2. Runtime robustness: CUDA availability check with fallback (`run_external_benchmark.sh`)

Added CUDA preflight:

- If `DEVICE=cuda*` and `torch.cuda.is_available()` is true, keep GPU run.
- If CUDA is unavailable and `ALLOW_CPU_FALLBACK=1` (default), switch to `DEVICE=cpu`.
- If CUDA is unavailable and `ALLOW_CPU_FALLBACK=0`, fail fast.

This avoids mid-run crashes when GPU is not visible in the current runtime.

## 3. Retrieval summary aggregation script (`scripts/summarize_external_retrieval.py`)

Added a new summary script to aggregate per-variant `retrieval_summary.json` into:

- `outputs/external_runs/retrieval_summary_table.csv`
- `outputs/external_runs/retrieval_summary_table.md`

It is fixed to the formal variant set and metrics:

- Variants: `tfidf`, `dense_e5`, `dense_e5_rerank`, `hybrid_bge_m3`, `hybrid_bge_m3_rerank`
- Metrics:
  - `support_recall_at_retrieve_k`
  - `support_recall_at_final_k`
  - `support_hit_at_1`
  - `support_mrr`

## 4. Formal rerun completed

A full formal rerun was executed for both benchmarks (`LongMemEval`, `LoCoMo`) with:

- `RETRIEVE_TOP_K=20`
- `FINAL_TOP_K=10`
- same variant set across both benchmarks

All generated variant summaries in this rerun report `retrieve_top_k=20` and `final_top_k=10`.
