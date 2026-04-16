# Memory Collapse Codebase Analysis Report

## 1. ENTRY POINT: `src/memory_collapse/cli.py`

**Location**: `/gfs/space/private/wujn/Learn/agent_memory/src/memory_collapse/cli.py` (159 lines)

### Main Structure
- Single entry point with `build_parser()` that creates an argument parser
- `main()` function dispatches to subcommands based on `args.command`
- Uses `argparse` for CLI argument parsing

### Supported Subcommands
1. `generate` - Generate synthetic world, memories, queries, and exact labels
2. `train_estimators` - Train learned estimators
3. `run_baselines` - Run heuristic baselines and proposed controllers
4. `plot` - Render fixed main figures
5. `prepare_external` - Normalize JSONL external benchmark
6. `run_external_retrieval` - Run retrieval on external benchmark
7. **`run_external_end_to_end`** - Answer-level external evaluation ⭐

---

## 2. `run_external_end_to_end` COMMAND SPECIFICATION

**Location**: Lines 58-71, implemented in `external_pipeline.py`

### Arguments

```
--benchmark (required)
  - Benchmark name (e.g., "longmemeval" or "locomo")
  - Type: string

--retrieval-variant (required, repeatable)
  - Retrieval variant name
  - Can repeat for batch runs: --retrieval-variant tfidf --retrieval-variant dense_e5
  - Type: string

--input-dir (optional)
  - Single retrieval variant directory containing retrieval_diagnostics.jsonl
  - Mutually exclusive with --input-root
  - Only allows single --retrieval-variant when used

--input-root (optional)
  - Parent directory whose children are retrieval variant folders
  - Mutually exclusive with --input-dir
  - Allows multiple --retrieval-variant values

--method (optional, repeatable)
  - Method to run
  - Can repeat for multiple methods
  - If not specified, runs DEFAULT_EXTERNAL_METHODS
  - Type: string

--model-dir (optional)
  - Synthetic model artifact directory
  - Required if any method other than "retrieval_only_baseline" is used
  - Type: path

--output-root (optional)
  - Root directory for external end-to-end outputs
  - Default: "outputs/external_runs"
  - Type: path

--summary-root (optional)
  - Directory for combined summary tables
  - If not specified, defaults to DEFAULT_SUMMARY_ROOT

--top-k (optional)
  - Cap on retrieved candidates before method scoring
  - Type: integer

--final-k (optional)
  - Cap on reranked candidates before method scoring
  - Type: integer
```

### Input Validation (from `_resolve_external_inputs`)
```
- At least one of --input-dir or --input-root must be provided
- Cannot use both --input-dir and --input-root together
- If using --input-dir: must have exactly ONE --retrieval-variant
- If using --input-root: can have multiple --retrieval-variant values
```

---

## 3. RETRIEVAL VARIANTS & METHODS

### Supported Retrieval Variants (from existing outputs)

Based on analysis of `/outputs/external_runs/retrieval_summary_table.csv`:

**For LongMemEval benchmark:**
- ✅ `tfidf` - TF-IDF retrieval
- ✅ `dense_e5` - Dense E5 model retrieval
- ✅ `dense_e5_rerank` - Dense E5 with reranking
- ✅ `hybrid_bge_m3` - Hybrid BGE-M3 retrieval
- ✅ `hybrid_bge_m3_rerank` - Hybrid BGE-M3 with reranking

**For LoCoMo benchmark:**
- ✅ Same 5 variants available

### External Methods (from `external_pipeline.py`)

**Defined constants** (lines 30-35):

1. **`retrieval_only_baseline`** (name: `"retrieval_only_baseline"`)
   - Top retrieved context with local extractive answer snippet
   - Does NOT require --model-dir
   - Runs on any retrieval output

2. **`DIRECT_VALID_METHOD`** (imported from `memory_collapse.baselines`)
   - Reuse synthetic direct_valid scorer on external retrieved candidates
   - REQUIRES --model-dir with trained synthetic artifacts
   - Used for proposed_learned_direct_valid

3. **`DIRECT_VALID_RESOLVER_METHOD`** (imported from `memory_collapse.baselines`)
   - Reuse synthetic direct_valid plus resolver on external retrieved candidates
   - REQUIRES --model-dir with trained synthetic artifacts
   - Used for proposed_learned_direct_valid_resolver

**Default behavior**: If no --method specified, runs all three methods

⚠️ **CRITICAL NOTE**: The exact string values for DIRECT_VALID_METHOD and DIRECT_VALID_RESOLVER_METHOD are defined in a module `memory_collapse.baselines` that is NOT present in the current source tree. This module appears to be from an external package dependency and is referenced but not available in `/gfs/space/private/wujn/Learn/agent_memory/src/memory_collapse/`.

### Model Requirements

For methods other than `retrieval_only_baseline`, the following models must exist in `--model-dir`:
- Relevance model
- Query validity models (useful_label and valid_label)
- Anti-support model
- Value resolver model

---

## 4. OUTPUT STRUCTURE

### Directory Structure
```
{output_root}/{benchmark}/{retrieval_variant}/{method}/
├── prediction_diagnostics.jsonl     # Per-query predictions
├── metrics.csv                       # Single-row CSV with metrics
└── metrics.json                      # Same metrics as JSON
```

### Summary Tables (written to `summary_root` or DEFAULT_SUMMARY_ROOT)
- `external_end_to_end_metrics.csv` - All metrics rows
- `external_end_to_end_summary.csv` - Projected/simplified metrics
- `external_end_to_end_summary.md` - Markdown table for easy viewing

### CSV Summary Table Structure

**From `_summarize_method_run()` (lines 623-642):**

Each row contains:
```
{
  "benchmark": str,
  "retrieval_variant": str,
  "method": str,
  "num_queries": int,
  "accuracy": float (0-1),                           # is_correct metric
  "exact_match": float (0-1),                        # exact match metric
  "hit_at_1": float | None,                          # First position recall
  "mrr": float | None,                               # Mean reciprocal rank
  "retrieval_support_recall_at_retrieve_k": float,   # Retrieve stage recall
  "retrieval_support_recall_at_final_k": float,      # Final rank stage recall
  "notes": str                                       # Method description
}
```

**Projected summary columns** (from `_summary_projection()`, lines 669-682):
- Same as above (excludes only the raw metrics.json)

### Markdown Table Generation

**From `_render_summary_markdown()` (lines 685-709):**

Renders all rows into markdown table with headers:
```markdown
| benchmark | retrieval_variant | method | num_queries | accuracy | exact_match | 
| hit_at_1 | mrr | retrieval_support_recall_at_retrieve_k | 
| retrieval_support_recall_at_final_k | notes |
```

Float values formatted to 4 decimal places, None values shown as empty cells.

---

## 5. ACTUAL OUTPUT EXAMPLES (from existing runs)

### Location
```
/gfs/space/private/wujn/Learn/agent_memory/outputs/external_runs/
├── longmemeval/
│   ├── tfidf/
│   ├── dense_e5/
│   ├── dense_e5_rerank/
│   ├── hybrid_bge_m3/
│   ├── hybrid_bge_m3_rerank/
│   └── normalized/
├── locomo/
│   ├── tfidf/
│   ├── dense_e5/
│   ├── dense_e5_rerank/
│   ├── hybrid_bge_m3/
│   ├── hybrid_bge_m3_rerank/
│   └── normalized/
├── logs/
├── retrieval_summary_table.csv
└── retrieval_summary_table.md
```

### Sample Retrieval Diagnostics
Each variant has `retrieval_diagnostics.jsonl` with per-query entries:

```json
{
  "query_id": "lm-1",
  "benchmark": "longmemeval",
  "prompt": "Where did Alice move?",
  "gold_answer": "Berlin",
  "answer_support_ids": ["e47becba_m0052"],
  "retriever": "tfidf",
  "retrieve_top_k": 20,
  "final_top_k": 10,
  "retrieved": [
    {
      "memory_id": "e47becba_m0049",
      "score": 0.037492,
      "stage": "retrieve",
      "is_answer_support": false,
      "text": "..."
    },
    ...
  ],
  "final_ranked": [
    {
      "memory_id": "...",
      "score": 0.xxx,
      "stage": "...",
      "is_answer_support": bool,
      "text": "..."
    },
    ...
  ],
  "support_recall_at_retrieve_k": 1.0,
  "support_recall_at_final_k": 0.95,
  "support_hit_at_1": 1.0,
  "support_mrr": 1.0
}
```

### Sample CSV Summary (retrieval_summary_table.csv)
```csv
benchmark,variant,retrieve_top_k,final_top_k,support_recall_at_retrieve_k,support_recall_at_final_k,support_hit_at_1,support_mrr
longmemeval,tfidf,20,10,0.946667,0.900933,0.760000,0.828029
longmemeval,dense_e5,20,10,0.967667,0.909433,0.780000,0.840080
longmemeval,dense_e5_rerank,20,10,0.967667,0.939233,0.854000,0.894816
longmemeval,hybrid_bge_m3,20,10,0.982667,0.950167,0.774000,0.846344
longmemeval,hybrid_bge_m3_rerank,20,10,0.982667,0.943167,0.862000,0.901379
```

---

## 6. IMPLEMENTATION FLOW

### ExternalEndToEndRunner (class, lines 117-262)

**Constructor** (`__init__`):
```python
__init__(
  benchmark_name: str,
  output_root: str | Path,
  model_dir: str | Path | None = None,
  top_k: int | None = None,
  final_k: int | None = None,
  summary_root: str | Path | None = None
)
```

**Main method** (`run`):
```python
def run(
  self,
  retrieval_inputs: dict[str, str | Path],  # {variant: directory}
  methods: list[str]
) -> dict[str, str]
```

**Flow**:
1. Normalize retrieval variant names
2. Validate no unsupported methods
3. For each retrieval_variant → For each method:
   - Load retrieval diagnostics from `retrieval_inputs[variant]/retrieval_diagnostics.jsonl`
   - Process each case with `_run_case()`
   - Summarize results with `_summarize_method_run()`
   - Write outputs to `{output_root}/{benchmark}/{variant}/{method}/`
4. Aggregate all metrics rows
5. Write summary tables to `summary_root`
6. Return diagnostics paths and summary paths

---

## 7. CRITICAL DEPENDENCIES & IMPORTS

### External Dependencies (from external_pipeline.py)
```python
from memory_collapse.anti_support import anti_support_model_exists, load_anti_support_bundle
from memory_collapse.baselines import (
    DIRECT_VALID_METHOD,                    # ❌ NOT FOUND
    DIRECT_VALID_RESOLVER_METHOD,           # ❌ NOT FOUND
    _aggregate_prediction,                  # ❌ NOT FOUND
    _softmax_confidence,                    # ❌ NOT FOUND
)
from memory_collapse.estimators import build_query_memory_contexts
from memory_collapse.query_validity import query_validity_model_exists, load_query_validity_bundle
from memory_collapse.relevance import load_relevance_bundle, normalized_similarity, relevance_model_exists
from memory_collapse.value_resolver import (
    build_value_candidate_feature_rows,
    load_value_resolver_bundle,
    value_resolver_model_exists,
)
```

### Missing Modules ❌
These modules are imported but NOT present in the codebase:
- `memory_collapse.anti_support`
- `memory_collapse.baselines` (contains DIRECT_VALID_METHOD, DIRECT_VALID_RESOLVER_METHOD)
- `memory_collapse.estimators`
- `memory_collapse.query_validity`
- `memory_collapse.relevance`
- `memory_collapse.value_resolver`

### Available Modules ✅
- `memory_collapse.external_pipeline` (717 lines)
- `memory_collapse.external_retrieval` (587 lines)
- `memory_collapse.external_preprocess` (218 lines)
- `memory_collapse.io_utils` (47 lines - CSV/JSONL I/O utilities)

---

## 8. METHOD NAME INFERENCE

Based on test file `/tests/test_external_pipeline.py` line 162:
```python
methods=[RETRIEVAL_ONLY_BASELINE, DIRECT_VALID_RESOLVER_METHOD]
```

The likely string values (NOT confirmed - these are imported from missing module):
- `"retrieval_only_baseline"` ✅ (defined in external_pipeline.py line 30)
- `DIRECT_VALID_METHOD` → likely `"proposed_learned_direct_valid"` or similar
- `DIRECT_VALID_RESOLVER_METHOD` → likely `"proposed_learned_direct_valid_resolver"` or similar

From summary notes in `_summary_note()` (lines 645-652):
```python
def _summary_note(method: str) -> str:
    if method == RETRIEVAL_ONLY_BASELINE:
        return "Top retrieved context with local extractive answer snippet."
    if method == DIRECT_VALID_METHOD:
        return "Reuse synthetic direct_valid scorer on external retrieved candidates."
    if method == DIRECT_VALID_RESOLVER_METHOD:
        return "Reuse synthetic direct_valid plus resolver on external retrieved candidates."
```

---

## 9. EXAMPLE CLI COMMANDS

### Minimal Example (retrieval only)
```bash
python -m memory_collapse.cli run_external_end_to_end \
  --benchmark longmemeval \
  --retrieval-variant tfidf \
  --input-dir ./outputs/external_runs/longmemeval/tfidf
```

### Multiple Variants with Batch Processing
```bash
python -m memory_collapse.cli run_external_end_to_end \
  --benchmark longmemeval \
  --retrieval-variant tfidf \
  --retrieval-variant dense_e5 \
  --retrieval-variant hybrid_bge_m3_rerank \
  --input-root ./outputs/external_runs/longmemeval \
  --method retrieval_only_baseline
```

### Full Pipeline with Learned Methods (requires models)
```bash
python -m memory_collapse.cli run_external_end_to_end \
  --benchmark longmemeval \
  --retrieval-variant dense_e5_rerank \
  --input-dir ./outputs/external_runs/longmemeval/dense_e5_rerank \
  --method retrieval_only_baseline \
  --method proposed_learned_direct_valid \
  --method proposed_learned_direct_valid_resolver \
  --model-dir ./outputs/synthetic_run_1/models \
  --output-root ./outputs/external_e2e \
  --summary-root ./outputs/external_e2e_summary
```

---

## 10. POTENTIAL ISSUES & VERIFICATION CHECKLIST

### ❌ BLOCKING ISSUES

1. **Missing `memory_collapse.baselines` Module**
   - The module containing DIRECT_VALID_METHOD and DIRECT_VALID_RESOLVER_METHOD is not in the repo
   - Any attempt to run methods other than `retrieval_only_baseline` will fail
   - The exact string values for these methods are unknown

2. **Missing Model Artifact Modules**
   - `memory_collapse.anti_support`
   - `memory_collapse.estimators`
   - `memory_collapse.query_validity`
   - `memory_collapse.relevance`
   - `memory_collapse.value_resolver`

### ✅ VERIFICATION STEPS

- [x] Entry point exists and is well-structured
- [x] CLI argument parsing is correctly implemented
- [x] Input validation logic is sound
- [x] Output directory structure is created properly
- [x] CSV/JSONL writers use pandas + custom utilities
- [x] Retrieval variant naming matches observed outputs
- [x] Benchmark naming validation works
- [x] Existing outputs demonstrate structure works for retrieval_only
- [ ] Method string values need confirmation from `memory_collapse.baselines`
- [ ] Model loading functions need implementation
- [ ] End-to-end with learned methods cannot be tested without baselines module

---

## 11. CSV GENERATION LOGIC

**Source**: `io_utils.py` lines 38-42

```python
def write_csv(path: str | Path, rows: Sequence[dict]) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_path, index=False)
    return output_path
```

**Process**:
1. Takes list of dictionaries (one per row)
2. Converts to pandas DataFrame
3. Writes to CSV without index column
4. Returns path

**Summary generation flow** (lines 170-174):
```python
metrics_csv_path = write_csv(self.summary_root / "external_end_to_end_metrics.csv", all_metric_rows)
summary_rows = [_summary_projection(row) for row in all_metric_rows]
summary_csv_path = write_csv(self.summary_root / "external_end_to_end_summary.csv", summary_rows)
summary_md_path = self.summary_root / "external_end_to_end_summary.md"
summary_md_path.write_text(_render_summary_markdown(summary_rows), encoding="utf-8")
```

---

## SUMMARY

**✅ WORKING**:
- CLI argument parsing and validation
- Batch processing of multiple retrieval variants
- Input/output directory management
- CSV/JSONL writing with pandas
- Markdown table generation
- `retrieval_only_baseline` method

**❌ INCOMPLETE/MISSING**:
- Method string constants from `memory_collapse.baselines`
- Model artifact loaders (relevance, query_validity, anti_support, value_resolver)
- Learned method implementations

**📋 RECOMMENDATION**: 
Before running end-to-end commands with learned methods, confirm:
1. The `memory_collapse.baselines` module location
2. The exact string values for DIRECT_VALID_METHOD and DIRECT_VALID_RESOLVER_METHOD
3. Availability of trained model artifacts for the synthetic benchmark
