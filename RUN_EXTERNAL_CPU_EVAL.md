# External End-to-End CPU 评测运行说明

本说明用于分别生成 `longmemeval` 与 `locomo` 两份独立总表，避免互相覆盖。

## 运行环境

在仓库目录执行：

```bash
cd /gfs/space/private/wujn/Learn/agent_memory
mkdir -p /tmp/mpl

export PYTHONPATH=src
export MPLCONFIGDIR=/tmp/mpl
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
```

## 任务 1：longmemeval

```bash
python -m memory_collapse.cli run_external_end_to_end \
  --benchmark longmemeval \
  --retrieval-variant tfidf \
  --retrieval-variant dense_e5 \
  --retrieval-variant dense_e5_rerank \
  --retrieval-variant hybrid_bge_m3 \
  --retrieval-variant hybrid_bge_m3_rerank \
  --input-root outputs/external_runs/longmemeval \
  --method retrieval_only_baseline \
  --method proposed_learned_direct_valid \
  --method proposed_learned_direct_valid_resolver \
  --model-dir outputs/default_direct_validity_v12 \
  --output-root outputs/external_runs \
  --summary-root outputs/external_runs/cpu_eval_longmemeval
```

主要产出：

- `outputs/external_runs/cpu_eval_longmemeval/external_end_to_end_summary.csv`
- `outputs/external_runs/cpu_eval_longmemeval/external_end_to_end_metrics.csv`

## 任务 2：locomo

```bash
python -m memory_collapse.cli run_external_end_to_end \
  --benchmark locomo \
  --retrieval-variant tfidf \
  --retrieval-variant hybrid_bge_m3 \
  --retrieval-variant hybrid_bge_m3_rerank \
  --input-root outputs/external_runs/locomo \
  --method retrieval_only_baseline \
  --method proposed_learned_direct_valid \
  --method proposed_learned_direct_valid_resolver \
  --model-dir outputs/default_direct_validity_v12 \
  --output-root outputs/external_runs \
  --summary-root outputs/external_runs/cpu_eval_locomo
```

主要产出：

- `outputs/external_runs/cpu_eval_locomo/external_end_to_end_summary.csv`
- `outputs/external_runs/cpu_eval_locomo/external_end_to_end_metrics.csv`

## 查看结果

重点查看这两个总表 CSV：

- `outputs/external_runs/cpu_eval_longmemeval/external_end_to_end_summary.csv`
- `outputs/external_runs/cpu_eval_locomo/external_end_to_end_summary.csv`

## 备注

- 运行中可能出现 `scikit-learn` 模型反序列化的版本告警（例如 `1.5.1 -> 1.6.1`），本次执行未导致任务失败。
