# External Retrieval Benchmark Session Resume (2026-04-14)

## Goal
- Baseline frozen at `outputs/retrieval_v0`
- Re-run with collaborator-updated code:
  - `src/memory_collapse/external_preprocess.py`
  - `src/memory_collapse/external_retrieval.py`
- New outputs to `outputs/retrieval_v1`
- Only 4 variants:
  - `dense_e5`
  - `dense_e5_rerank`
  - `hybrid_bge_m3`
  - `hybrid_bge_m3_rerank`

## Verified Facts
- `outputs/retrieval_v0/longmemeval_results` exists
- `outputs/retrieval_v0/locomo_results` exists
- `outputs/retrieval_v0/BASELINE_STATUS.md` exists and contains:
  - `longmemeval_results: final`
  - `locomo_results: provisional`
- `retrieval_v0` has all required 4 variants for both benchmarks.

## Input Choice Decision (for strict comparability)
- Use `longmemeval_s`, not `longmemeval_m`.
- Evidence: converting `longmemeval_s` with current `convert_raw_external` gives a `source.jsonl` with identical SHA256 to `outputs/retrieval_v0/longmemeval_results/source.jsonl`.
- So switching to `longmemeval_m` would break strict v0 comparability.

## Paths
- `LONGMEMEVAL_INPUT=/gfs/space/private/wujn/Learn/datasets/longmemeval/longmemeval_s`
- `LOCOMO_INPUT=/gfs/space/private/wujn/Learn/datasets/locomo/locomo10.json`
- `E5_MODEL_PATH=/gfs/space/private/wujn/Learn/models/intfloat/e5-base-v2`
- `BGE_M3_MODEL_PATH=/gfs/space/private/wujn/Learn/models/BAAI/bge-m3`
- `RERANKER_MODEL_PATH=/gfs/space/private/wujn/Learn/models/BAAI/bge-reranker-v2-m3`

## Progress Snapshot (2026-04-14 07:01:08 UTC)
- `convert_raw_external + prepare_external` already done for both benchmarks under `outputs/retrieval_v1`.
- Completed summaries currently present:
  - `outputs/retrieval_v1/longmemeval_results/dense_e5/retrieval_summary.json`
  - `outputs/retrieval_v1/longmemeval_results/dense_e5_rerank/retrieval_summary.json`
  - `outputs/retrieval_v1/longmemeval_results/hybrid_bge_m3/retrieval_summary.json`
- Still running at snapshot time:
  - longmemeval `hybrid_bge_m3_rerank`
  - locomo side currently on `dense_e5` (then remaining variants)

## Reusable Commands

### 1) Rebuild `retrieval_v1` sources + normalized data (must run for new chunking logic)
```bash
cd /gfs/space/private/wujn/Learn/agent_memory
export PYTHONPATH=src

python -m memory_collapse.external_cli convert_raw_external \
  --benchmark longmemeval \
  --input-path /gfs/space/private/wujn/Learn/datasets/longmemeval/longmemeval_s \
  --output-path outputs/retrieval_v1/longmemeval_results/source.jsonl
python -m memory_collapse.external_cli prepare_external \
  --benchmark longmemeval \
  --input-path outputs/retrieval_v1/longmemeval_results/source.jsonl \
  --output-dir outputs/retrieval_v1/longmemeval_results/normalized

python -m memory_collapse.external_cli convert_raw_external \
  --benchmark locomo \
  --input-path /gfs/space/private/wujn/Learn/datasets/locomo/locomo10.json \
  --output-path outputs/retrieval_v1/locomo_results/source.jsonl
python -m memory_collapse.external_cli prepare_external \
  --benchmark locomo \
  --input-path outputs/retrieval_v1/locomo_results/source.jsonl \
  --output-dir outputs/retrieval_v1/locomo_results/normalized
```

### 2) Launch only the 4 required variants (both benchmarks, dual GPU)
```bash
cd /gfs/space/private/wujn/Learn/agent_memory
bash outputs/retrieval_v1/run_v1_4variants.sh
```

### 3) Monitor progress
```bash
cd /gfs/space/private/wujn/Learn/agent_memory
find outputs/retrieval_v1 -maxdepth 4 -type f -name 'retrieval_summary.json' | sort

tail -f outputs/retrieval_v1/logs/longmemeval_v1.log
# in another terminal:
tail -f outputs/retrieval_v1/logs/locomo_v1.log

ps -ef | grep -E 'memory_collapse.external_cli run_external_retrieval|run_external_benchmark.sh|run_v1_4variants.sh' | grep -v grep
```

### 4) One-shot completeness check (8 summaries)
```bash
cd /gfs/space/private/wujn/Learn/agent_memory
python - <<'PY'
from pathlib import Path
benches=['longmemeval_results','locomo_results']
variants=['dense_e5','dense_e5_rerank','hybrid_bge_m3','hybrid_bge_m3_rerank']
missing=[]
for b in benches:
  for v in variants:
    p=Path('outputs/retrieval_v1')/b/v/'retrieval_summary.json'
    if not p.exists():
      missing.append(str(p))
print('ALL_DONE' if not missing else 'MISSING')
for m in missing:
  print(m)
PY
```

### 5) Compare `retrieval_v0` vs `retrieval_v1`
```bash
cd /gfs/space/private/wujn/Learn/agent_memory
python - <<'PY'
import json, csv
from pathlib import Path

v0 = Path('outputs/retrieval_v0')
v1 = Path('outputs/retrieval_v1')
benches = ['longmemeval_results', 'locomo_results']
variants = ['dense_e5', 'dense_e5_rerank', 'hybrid_bge_m3', 'hybrid_bge_m3_rerank']
metrics = ['support_recall_at_retrieve_k', 'support_recall_at_final_k', 'support_hit_at_1', 'support_mrr']

rows = []
for b in benches:
    for v in variants:
        p0 = v0 / b / v / 'retrieval_summary.json'
        p1 = v1 / b / v / 'retrieval_summary.json'
        d0 = json.loads(p0.read_text()) if p0.exists() else {}
        d1 = json.loads(p1.read_text()) if p1.exists() else {}
        r = {'benchmark': b, 'variant': v}
        for m in metrics:
            a = d0.get('metrics', {}).get(m)
            c = d1.get('metrics', {}).get(m)
            r[f'v0_{m}'] = a
            r[f'v1_{m}'] = c
            r[f'delta_{m}'] = (None if a is None or c is None else c - a)
        rows.append(r)

out = Path('outputs/retrieval_v1/retrieval_v0_vs_v1.csv')
out.parent.mkdir(parents=True, exist_ok=True)
with out.open('w', newline='', encoding='utf-8') as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader(); w.writerows(rows)
print(out)
PY
```

## Note on `run_external_dual_h800.sh`
- It is fine as a generic launcher, but for this strict-comparison task we must enforce:
  - `longmemeval_s`
  - re-run `convert_raw_external` + `prepare_external`
  - only the 4 required variants
- `outputs/retrieval_v1/run_v1_4variants.sh` is purpose-built for this exact scope.
