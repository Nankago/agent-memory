# Retrieval v0 vs v1

## longmemeval

| variant | delta_support_recall_at_retrieve_k | delta_support_recall_at_final_k | delta_support_hit_at_1 | delta_support_mrr |
| --- | --- | --- | --- | --- |
| dense_e5 | +0.003000 | -0.000333 | +0.000000 | -0.000800 |
| dense_e5_rerank | +0.003000 | +0.002000 | -0.002000 | -0.001175 |
| hybrid_bge_m3 | -0.000667 | -0.000167 | +0.004000 | +0.000937 |
| hybrid_bge_m3_rerank | -0.000667 | -0.000833 | -0.002000 | -0.001388 |

## locomo

| variant | delta_support_recall_at_retrieve_k | delta_support_recall_at_final_k | delta_support_hit_at_1 | delta_support_mrr |
| --- | --- | --- | --- | --- |
| dense_e5 | +0.057385 | +0.110221 | +0.150581 | +0.143165 |
| dense_e5_rerank | +0.057385 | +0.139878 | +0.201617 | +0.189088 |
| hybrid_bge_m3 | -0.005413 | -0.001669 | -0.003537 | +0.012516 |
| hybrid_bge_m3_rerank | -0.005413 | +0.123216 | +0.198585 | +0.183448 |
