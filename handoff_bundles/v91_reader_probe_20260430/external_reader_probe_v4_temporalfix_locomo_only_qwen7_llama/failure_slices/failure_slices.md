# External Failure Slices

## longmemeval

### reader_llama31_8b_instruct

| slice | variant | num_queries | num_failures | accuracy | failure_rate |
| --- | --- | --- | --- | --- | --- |
| price | hybrid_bge_m3_rerank | 37 | 35 | 0.054054 | 0.945946 |
| generic | hybrid_bge_m3_rerank | 102 | 95 | 0.068627 | 0.931373 |
| duration | hybrid_bge_m3_rerank | 26 | 23 | 0.115385 | 0.884615 |
| where | hybrid_bge_m3_rerank | 20 | 17 | 0.150000 | 0.850000 |
| quantity | hybrid_bge_m3_rerank | 166 | 141 | 0.150602 | 0.849398 |
| title_or_name | hybrid_bge_m3_rerank | 24 | 16 | 0.333333 | 0.666667 |

| failure_bucket | variant | num_queries | num_failures | accuracy | failure_rate |
| --- | --- | --- | --- | --- | --- |
| supported_but_wrong | hybrid_bge_m3_rerank | 213 | 213 | 0.000000 | 1.000000 |
| confident_selection_error | hybrid_bge_m3_rerank | 176 | 176 | 0.000000 | 1.000000 |
| comparative_reasoning | hybrid_bge_m3_rerank | 24 | 24 | 0.000000 | 1.000000 |
| correct | hybrid_bge_m3_rerank | 73 | 0 | 1.000000 | 0.000000 |

### reader_qwen25_7b_instruct

| slice | variant | num_queries | num_failures | accuracy | failure_rate |
| --- | --- | --- | --- | --- | --- |
| price | hybrid_bge_m3_rerank | 37 | 37 | 0.000000 | 1.000000 |
| where | hybrid_bge_m3_rerank | 20 | 19 | 0.050000 | 0.950000 |
| generic | hybrid_bge_m3_rerank | 102 | 96 | 0.058824 | 0.941176 |
| duration | hybrid_bge_m3_rerank | 26 | 24 | 0.076923 | 0.923077 |
| quantity | hybrid_bge_m3_rerank | 166 | 139 | 0.162651 | 0.837349 |
| title_or_name | hybrid_bge_m3_rerank | 24 | 17 | 0.291667 | 0.708333 |

| failure_bucket | variant | num_queries | num_failures | accuracy | failure_rate |
| --- | --- | --- | --- | --- | --- |
| confident_selection_error | hybrid_bge_m3_rerank | 233 | 233 | 0.000000 | 1.000000 |
| supported_but_wrong | hybrid_bge_m3_rerank | 163 | 163 | 0.000000 | 1.000000 |
| comparative_reasoning | hybrid_bge_m3_rerank | 26 | 26 | 0.000000 | 1.000000 |
| correct | hybrid_bge_m3_rerank | 64 | 0 | 1.000000 | 0.000000 |

## locomo

### reader_llama31_8b_instruct

| slice | variant | num_queries | num_failures | accuracy | failure_rate |
| --- | --- | --- | --- | --- | --- |
| duration | hybrid_bge_m3_rerank | 35 | 31 | 0.114286 | 0.885714 |
| generic | hybrid_bge_m3_rerank | 876 | 773 | 0.117580 | 0.882420 |
| item_name | hybrid_bge_m3_rerank | 182 | 157 | 0.137363 | 0.862637 |
| title_or_name | hybrid_bge_m3_rerank | 160 | 136 | 0.150000 | 0.850000 |
| event_name | hybrid_bge_m3_rerank | 89 | 73 | 0.179775 | 0.820225 |
| activity_list | hybrid_bge_m3_rerank | 44 | 36 | 0.181818 | 0.818182 |
| group_name | hybrid_bge_m3_rerank | 53 | 42 | 0.207547 | 0.792453 |
| person_name | hybrid_bge_m3_rerank | 52 | 41 | 0.211538 | 0.788462 |

| failure_bucket | variant | num_queries | num_failures | accuracy | failure_rate |
| --- | --- | --- | --- | --- | --- |
| supported_but_wrong | hybrid_bge_m3_rerank | 929 | 929 | 0.000000 | 1.000000 |
| confident_selection_error | hybrid_bge_m3_rerank | 424 | 424 | 0.000000 | 1.000000 |
| retrieval_miss | hybrid_bge_m3_rerank | 263 | 263 | 0.000000 | 1.000000 |
| correct | hybrid_bge_m3_rerank | 368 | 0 | 1.000000 | 0.000000 |

### reader_qwen25_7b_instruct

| slice | variant | num_queries | num_failures | accuracy | failure_rate |
| --- | --- | --- | --- | --- | --- |
| judgment | hybrid_bge_m3_rerank | 44 | 40 | 0.090909 | 0.909091 |
| generic | hybrid_bge_m3_rerank | 876 | 789 | 0.099315 | 0.900685 |
| item_name | hybrid_bge_m3_rerank | 182 | 163 | 0.104396 | 0.895604 |
| activity_list | hybrid_bge_m3_rerank | 44 | 39 | 0.113636 | 0.886364 |
| duration | hybrid_bge_m3_rerank | 35 | 31 | 0.114286 | 0.885714 |
| title_or_name | hybrid_bge_m3_rerank | 160 | 141 | 0.118750 | 0.881250 |
| group_name | hybrid_bge_m3_rerank | 53 | 44 | 0.169811 | 0.830189 |
| person_name | hybrid_bge_m3_rerank | 52 | 43 | 0.173077 | 0.826923 |

| failure_bucket | variant | num_queries | num_failures | accuracy | failure_rate |
| --- | --- | --- | --- | --- | --- |
| confident_selection_error | hybrid_bge_m3_rerank | 731 | 731 | 0.000000 | 1.000000 |
| supported_but_wrong | hybrid_bge_m3_rerank | 672 | 672 | 0.000000 | 1.000000 |
| retrieval_miss | hybrid_bge_m3_rerank | 267 | 267 | 0.000000 | 1.000000 |
| correct | hybrid_bge_m3_rerank | 313 | 0 | 1.000000 | 0.000000 |
