# External Failure Slices

## longmemeval

### reader_qwen25_14b_instruct

| slice | variant | num_queries | num_failures | accuracy | failure_rate |
| --- | --- | --- | --- | --- | --- |
| price | hybrid_bge_m3_rerank | 37 | 35 | 0.054054 | 0.945946 |
| duration | hybrid_bge_m3_rerank | 26 | 24 | 0.076923 | 0.923077 |
| generic | hybrid_bge_m3_rerank | 102 | 94 | 0.078431 | 0.921569 |
| quantity | hybrid_bge_m3_rerank | 166 | 147 | 0.114458 | 0.885542 |
| where | hybrid_bge_m3_rerank | 20 | 17 | 0.150000 | 0.850000 |
| title_or_name | hybrid_bge_m3_rerank | 24 | 18 | 0.250000 | 0.750000 |

| failure_bucket | variant | num_queries | num_failures | accuracy | failure_rate |
| --- | --- | --- | --- | --- | --- |
| confident_selection_error | hybrid_bge_m3_rerank | 233 | 233 | 0.000000 | 1.000000 |
| supported_but_wrong | hybrid_bge_m3_rerank | 166 | 166 | 0.000000 | 1.000000 |
| comparative_reasoning | hybrid_bge_m3_rerank | 25 | 25 | 0.000000 | 1.000000 |
| correct | hybrid_bge_m3_rerank | 62 | 0 | 1.000000 | 0.000000 |

## locomo

### reader_qwen25_14b_instruct

| slice | variant | num_queries | num_failures | accuracy | failure_rate |
| --- | --- | --- | --- | --- | --- |
| generic | hybrid_bge_m3_rerank | 876 | 778 | 0.111872 | 0.888128 |
| activity_list | hybrid_bge_m3_rerank | 44 | 39 | 0.113636 | 0.886364 |
| duration | hybrid_bge_m3_rerank | 35 | 31 | 0.114286 | 0.885714 |
| item_name | hybrid_bge_m3_rerank | 182 | 160 | 0.120879 | 0.879121 |
| title_or_name | hybrid_bge_m3_rerank | 160 | 136 | 0.150000 | 0.850000 |
| group_name | hybrid_bge_m3_rerank | 53 | 45 | 0.150943 | 0.849057 |
| judgment | hybrid_bge_m3_rerank | 44 | 37 | 0.159091 | 0.840909 |
| person_name | hybrid_bge_m3_rerank | 52 | 43 | 0.173077 | 0.826923 |

| failure_bucket | variant | num_queries | num_failures | accuracy | failure_rate |
| --- | --- | --- | --- | --- | --- |
| confident_selection_error | hybrid_bge_m3_rerank | 756 | 756 | 0.000000 | 1.000000 |
| supported_but_wrong | hybrid_bge_m3_rerank | 627 | 627 | 0.000000 | 1.000000 |
| retrieval_miss | hybrid_bge_m3_rerank | 266 | 266 | 0.000000 | 1.000000 |
| correct | hybrid_bge_m3_rerank | 336 | 0 | 1.000000 | 0.000000 |
