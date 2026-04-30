# External Failure Slices

## longmemeval

### reader_llama31_8b_instruct

| slice | variant | num_queries | num_failures | accuracy | failure_rate |
| --- | --- | --- | --- | --- | --- |
| price | hybrid_bge_m3_rerank | 37 | 35 | 0.054054 | 0.945946 |
| generic | hybrid_bge_m3_rerank | 102 | 95 | 0.068627 | 0.931373 |
| where | hybrid_bge_m3_rerank | 20 | 18 | 0.100000 | 0.900000 |
| duration | hybrid_bge_m3_rerank | 26 | 23 | 0.115385 | 0.884615 |
| quantity | hybrid_bge_m3_rerank | 166 | 142 | 0.144578 | 0.855422 |
| title_or_name | hybrid_bge_m3_rerank | 24 | 17 | 0.291667 | 0.708333 |

| failure_bucket | variant | num_queries | num_failures | accuracy | failure_rate |
| --- | --- | --- | --- | --- | --- |
| supported_but_wrong | hybrid_bge_m3_rerank | 208 | 208 | 0.000000 | 1.000000 |
| confident_selection_error | hybrid_bge_m3_rerank | 184 | 184 | 0.000000 | 1.000000 |
| comparative_reasoning | hybrid_bge_m3_rerank | 24 | 24 | 0.000000 | 1.000000 |
| correct | hybrid_bge_m3_rerank | 69 | 0 | 1.000000 | 0.000000 |

### reader_qwen25_14b_instruct

| slice | variant | num_queries | num_failures | accuracy | failure_rate |
| --- | --- | --- | --- | --- | --- |
| price | hybrid_bge_m3_rerank | 37 | 35 | 0.054054 | 0.945946 |
| duration | hybrid_bge_m3_rerank | 26 | 24 | 0.076923 | 0.923077 |
| generic | hybrid_bge_m3_rerank | 102 | 93 | 0.088235 | 0.911765 |
| quantity | hybrid_bge_m3_rerank | 166 | 147 | 0.114458 | 0.885542 |
| where | hybrid_bge_m3_rerank | 20 | 17 | 0.150000 | 0.850000 |
| title_or_name | hybrid_bge_m3_rerank | 24 | 18 | 0.250000 | 0.750000 |

| failure_bucket | variant | num_queries | num_failures | accuracy | failure_rate |
| --- | --- | --- | --- | --- | --- |
| confident_selection_error | hybrid_bge_m3_rerank | 231 | 231 | 0.000000 | 1.000000 |
| supported_but_wrong | hybrid_bge_m3_rerank | 167 | 167 | 0.000000 | 1.000000 |
| comparative_reasoning | hybrid_bge_m3_rerank | 25 | 25 | 0.000000 | 1.000000 |
| correct | hybrid_bge_m3_rerank | 63 | 0 | 1.000000 | 0.000000 |

### reader_qwen25_7b_instruct

| slice | variant | num_queries | num_failures | accuracy | failure_rate |
| --- | --- | --- | --- | --- | --- |
| price | hybrid_bge_m3_rerank | 37 | 37 | 0.000000 | 1.000000 |
| where | hybrid_bge_m3_rerank | 20 | 19 | 0.050000 | 0.950000 |
| generic | hybrid_bge_m3_rerank | 102 | 96 | 0.058824 | 0.941176 |
| duration | hybrid_bge_m3_rerank | 26 | 24 | 0.076923 | 0.923077 |
| quantity | hybrid_bge_m3_rerank | 166 | 140 | 0.156627 | 0.843373 |
| title_or_name | hybrid_bge_m3_rerank | 24 | 17 | 0.291667 | 0.708333 |

| failure_bucket | variant | num_queries | num_failures | accuracy | failure_rate |
| --- | --- | --- | --- | --- | --- |
| confident_selection_error | hybrid_bge_m3_rerank | 233 | 233 | 0.000000 | 1.000000 |
| supported_but_wrong | hybrid_bge_m3_rerank | 164 | 164 | 0.000000 | 1.000000 |
| comparative_reasoning | hybrid_bge_m3_rerank | 26 | 26 | 0.000000 | 1.000000 |
| correct | hybrid_bge_m3_rerank | 63 | 0 | 1.000000 | 0.000000 |

## locomo

### reader_llama31_8b_instruct

| slice | variant | num_queries | num_failures | accuracy | failure_rate |
| --- | --- | --- | --- | --- | --- |
| when | hybrid_bge_m3_rerank | 271 | 270 | 0.003690 | 0.996310 |
| generic | hybrid_bge_m3_rerank | 876 | 777 | 0.113014 | 0.886986 |
| duration | hybrid_bge_m3_rerank | 35 | 31 | 0.114286 | 0.885714 |
| item_name | hybrid_bge_m3_rerank | 182 | 157 | 0.137363 | 0.862637 |
| title_or_name | hybrid_bge_m3_rerank | 160 | 137 | 0.143750 | 0.856250 |
| event_name | hybrid_bge_m3_rerank | 89 | 73 | 0.179775 | 0.820225 |
| activity_list | hybrid_bge_m3_rerank | 44 | 36 | 0.181818 | 0.818182 |
| group_name | hybrid_bge_m3_rerank | 53 | 43 | 0.188679 | 0.811321 |

| failure_bucket | variant | num_queries | num_failures | accuracy | failure_rate |
| --- | --- | --- | --- | --- | --- |
| supported_but_wrong | hybrid_bge_m3_rerank | 1124 | 1124 | 0.000000 | 1.000000 |
| confident_selection_error | hybrid_bge_m3_rerank | 325 | 325 | 0.000000 | 1.000000 |
| retrieval_miss | hybrid_bge_m3_rerank | 264 | 264 | 0.000000 | 1.000000 |
| correct | hybrid_bge_m3_rerank | 270 | 0 | 1.000000 | 0.000000 |

### reader_qwen25_14b_instruct

| slice | variant | num_queries | num_failures | accuracy | failure_rate |
| --- | --- | --- | --- | --- | --- |
| when | hybrid_bge_m3_rerank | 271 | 270 | 0.003690 | 0.996310 |
| generic | hybrid_bge_m3_rerank | 876 | 776 | 0.114155 | 0.885845 |
| duration | hybrid_bge_m3_rerank | 35 | 31 | 0.114286 | 0.885714 |
| item_name | hybrid_bge_m3_rerank | 182 | 160 | 0.120879 | 0.879121 |
| activity_list | hybrid_bge_m3_rerank | 44 | 38 | 0.136364 | 0.863636 |
| group_name | hybrid_bge_m3_rerank | 53 | 45 | 0.150943 | 0.849057 |
| judgment | hybrid_bge_m3_rerank | 44 | 37 | 0.159091 | 0.840909 |
| title_or_name | hybrid_bge_m3_rerank | 160 | 134 | 0.162500 | 0.837500 |

| failure_bucket | variant | num_queries | num_failures | accuracy | failure_rate |
| --- | --- | --- | --- | --- | --- |
| supported_but_wrong | hybrid_bge_m3_rerank | 801 | 801 | 0.000000 | 1.000000 |
| confident_selection_error | hybrid_bge_m3_rerank | 667 | 667 | 0.000000 | 1.000000 |
| retrieval_miss | hybrid_bge_m3_rerank | 268 | 268 | 0.000000 | 1.000000 |
| correct | hybrid_bge_m3_rerank | 249 | 0 | 1.000000 | 0.000000 |

### reader_qwen25_7b_instruct

| slice | variant | num_queries | num_failures | accuracy | failure_rate |
| --- | --- | --- | --- | --- | --- |
| when | hybrid_bge_m3_rerank | 271 | 270 | 0.003690 | 0.996310 |
| judgment | hybrid_bge_m3_rerank | 44 | 40 | 0.090909 | 0.909091 |
| generic | hybrid_bge_m3_rerank | 876 | 782 | 0.107306 | 0.892694 |
| item_name | hybrid_bge_m3_rerank | 182 | 162 | 0.109890 | 0.890110 |
| activity_list | hybrid_bge_m3_rerank | 44 | 39 | 0.113636 | 0.886364 |
| duration | hybrid_bge_m3_rerank | 35 | 31 | 0.114286 | 0.885714 |
| title_or_name | hybrid_bge_m3_rerank | 160 | 139 | 0.131250 | 0.868750 |
| person_name | hybrid_bge_m3_rerank | 52 | 43 | 0.173077 | 0.826923 |

| failure_bucket | variant | num_queries | num_failures | accuracy | failure_rate |
| --- | --- | --- | --- | --- | --- |
| supported_but_wrong | hybrid_bge_m3_rerank | 842 | 842 | 0.000000 | 1.000000 |
| confident_selection_error | hybrid_bge_m3_rerank | 644 | 644 | 0.000000 | 1.000000 |
| retrieval_miss | hybrid_bge_m3_rerank | 268 | 268 | 0.000000 | 1.000000 |
| correct | hybrid_bge_m3_rerank | 229 | 0 | 1.000000 | 0.000000 |
