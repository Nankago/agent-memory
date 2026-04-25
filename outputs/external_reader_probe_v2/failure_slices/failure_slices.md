# External Failure Slices

## longmemeval

### reader_llama31_8b_instruct

| slice | variant | num_queries | num_failures | accuracy | failure_rate |
| --- | --- | --- | --- | --- | --- |
| where | hybrid_bge_m3_rerank | 20 | 20 | 0.000000 | 1.000000 |
| price | hybrid_bge_m3_rerank | 37 | 35 | 0.054054 | 0.945946 |
| duration | hybrid_bge_m3_rerank | 26 | 24 | 0.076923 | 0.923077 |
| generic | hybrid_bge_m3_rerank | 102 | 94 | 0.078431 | 0.921569 |
| quantity | hybrid_bge_m3_rerank | 166 | 145 | 0.126506 | 0.873494 |
| title_or_name | hybrid_bge_m3_rerank | 24 | 17 | 0.291667 | 0.708333 |

| failure_bucket | variant | num_queries | num_failures | accuracy | failure_rate |
| --- | --- | --- | --- | --- | --- |
| confident_selection_error | hybrid_bge_m3_rerank | 211 | 211 | 0.000000 | 1.000000 |
| supported_but_wrong | hybrid_bge_m3_rerank | 188 | 188 | 0.000000 | 1.000000 |
| comparative_reasoning | hybrid_bge_m3_rerank | 29 | 29 | 0.000000 | 1.000000 |
| correct | hybrid_bge_m3_rerank | 58 | 0 | 1.000000 | 0.000000 |

### reader_qwen25_7b_instruct

| slice | variant | num_queries | num_failures | accuracy | failure_rate |
| --- | --- | --- | --- | --- | --- |
| where | hybrid_bge_m3_rerank | 20 | 20 | 0.000000 | 1.000000 |
| price | hybrid_bge_m3_rerank | 37 | 35 | 0.054054 | 0.945946 |
| generic | hybrid_bge_m3_rerank | 102 | 95 | 0.068627 | 0.931373 |
| duration | hybrid_bge_m3_rerank | 26 | 24 | 0.076923 | 0.923077 |
| quantity | hybrid_bge_m3_rerank | 166 | 142 | 0.144578 | 0.855422 |
| title_or_name | hybrid_bge_m3_rerank | 24 | 18 | 0.250000 | 0.750000 |

| failure_bucket | variant | num_queries | num_failures | accuracy | failure_rate |
| --- | --- | --- | --- | --- | --- |
| confident_selection_error | hybrid_bge_m3_rerank | 243 | 243 | 0.000000 | 1.000000 |
| supported_but_wrong | hybrid_bge_m3_rerank | 157 | 157 | 0.000000 | 1.000000 |
| comparative_reasoning | hybrid_bge_m3_rerank | 29 | 29 | 0.000000 | 1.000000 |
| correct | hybrid_bge_m3_rerank | 57 | 0 | 1.000000 | 0.000000 |

## locomo

### reader_llama31_8b_instruct

| slice | variant | num_queries | num_failures | accuracy | failure_rate |
| --- | --- | --- | --- | --- | --- |
| when | hybrid_bge_m3_rerank | 271 | 271 | 0.000000 | 1.000000 |
| duration | hybrid_bge_m3_rerank | 35 | 34 | 0.028571 | 0.971429 |
| generic | hybrid_bge_m3_rerank | 876 | 824 | 0.059361 | 0.940639 |
| activity_list | hybrid_bge_m3_rerank | 44 | 41 | 0.068182 | 0.931818 |
| item_name | hybrid_bge_m3_rerank | 182 | 169 | 0.071429 | 0.928571 |
| group_name | hybrid_bge_m3_rerank | 53 | 48 | 0.094340 | 0.905660 |
| title_or_name | hybrid_bge_m3_rerank | 160 | 143 | 0.106250 | 0.893750 |
| event_name | hybrid_bge_m3_rerank | 89 | 75 | 0.157303 | 0.842697 |

| failure_bucket | variant | num_queries | num_failures | accuracy | failure_rate |
| --- | --- | --- | --- | --- | --- |
| supported_but_wrong | hybrid_bge_m3_rerank | 1054 | 1054 | 0.000000 | 1.000000 |
| confident_selection_error | hybrid_bge_m3_rerank | 494 | 494 | 0.000000 | 1.000000 |
| retrieval_miss | hybrid_bge_m3_rerank | 269 | 269 | 0.000000 | 1.000000 |
| correct | hybrid_bge_m3_rerank | 166 | 0 | 1.000000 | 0.000000 |

### reader_qwen25_7b_instruct

| slice | variant | num_queries | num_failures | accuracy | failure_rate |
| --- | --- | --- | --- | --- | --- |
| when | hybrid_bge_m3_rerank | 271 | 271 | 0.000000 | 1.000000 |
| judgment | hybrid_bge_m3_rerank | 44 | 43 | 0.022727 | 0.977273 |
| duration | hybrid_bge_m3_rerank | 35 | 34 | 0.028571 | 0.971429 |
| activity_list | hybrid_bge_m3_rerank | 44 | 42 | 0.045455 | 0.954545 |
| generic | hybrid_bge_m3_rerank | 876 | 822 | 0.061644 | 0.938356 |
| item_name | hybrid_bge_m3_rerank | 182 | 170 | 0.065934 | 0.934066 |
| title_or_name | hybrid_bge_m3_rerank | 160 | 149 | 0.068750 | 0.931250 |
| group_name | hybrid_bge_m3_rerank | 53 | 47 | 0.113208 | 0.886792 |

| failure_bucket | variant | num_queries | num_failures | accuracy | failure_rate |
| --- | --- | --- | --- | --- | --- |
| confident_selection_error | hybrid_bge_m3_rerank | 795 | 795 | 0.000000 | 1.000000 |
| supported_but_wrong | hybrid_bge_m3_rerank | 778 | 778 | 0.000000 | 1.000000 |
| retrieval_miss | hybrid_bge_m3_rerank | 273 | 273 | 0.000000 | 1.000000 |
| correct | hybrid_bge_m3_rerank | 136 | 0 | 1.000000 | 0.000000 |
