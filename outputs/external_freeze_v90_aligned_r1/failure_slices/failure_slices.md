# External Failure Slices

## longmemeval

### proposed_learned_direct_valid

| slice | variant | num_queries | num_failures | accuracy | failure_rate |
| --- | --- | --- | --- | --- | --- |
| price | dense_e5_rerank | 37 | 36 | 0.027027 | 0.972973 |
| price | hybrid_bge_m3_rerank | 37 | 36 | 0.027027 | 0.972973 |
| generic | dense_e5_rerank | 102 | 98 | 0.039216 | 0.960784 |
| generic | hybrid_bge_m3_rerank | 102 | 98 | 0.039216 | 0.960784 |
| generic | dense_e5 | 102 | 97 | 0.049020 | 0.950980 |
| generic | hybrid_bge_m3 | 102 | 97 | 0.049020 | 0.950980 |
| price | dense_e5 | 37 | 35 | 0.054054 | 0.945946 |
| price | tfidf | 37 | 35 | 0.054054 | 0.945946 |

| failure_bucket | variant | num_queries | num_failures | accuracy | failure_rate |
| --- | --- | --- | --- | --- | --- |
| supported_but_wrong | hybrid_bge_m3 | 220 | 220 | 0.000000 | 1.000000 |
| confident_selection_error | tfidf | 220 | 220 | 0.000000 | 1.000000 |
| confident_selection_error | dense_e5 | 215 | 215 | 0.000000 | 1.000000 |
| supported_but_wrong | hybrid_bge_m3_rerank | 204 | 204 | 0.000000 | 1.000000 |
| supported_but_wrong | dense_e5_rerank | 200 | 200 | 0.000000 | 1.000000 |
| confident_selection_error | dense_e5_rerank | 182 | 182 | 0.000000 | 1.000000 |

### proposed_learned_direct_valid_resolver

| slice | variant | num_queries | num_failures | accuracy | failure_rate |
| --- | --- | --- | --- | --- | --- |
| price | dense_e5_rerank | 37 | 36 | 0.027027 | 0.972973 |
| price | hybrid_bge_m3_rerank | 37 | 36 | 0.027027 | 0.972973 |
| generic | dense_e5_rerank | 102 | 98 | 0.039216 | 0.960784 |
| generic | dense_e5 | 102 | 97 | 0.049020 | 0.950980 |
| generic | hybrid_bge_m3 | 102 | 97 | 0.049020 | 0.950980 |
| generic | hybrid_bge_m3_rerank | 102 | 97 | 0.049020 | 0.950980 |
| price | dense_e5 | 37 | 35 | 0.054054 | 0.945946 |
| price | hybrid_bge_m3 | 37 | 35 | 0.054054 | 0.945946 |

| failure_bucket | variant | num_queries | num_failures | accuracy | failure_rate |
| --- | --- | --- | --- | --- | --- |
| confident_selection_error | dense_e5 | 274 | 274 | 0.000000 | 1.000000 |
| confident_selection_error | tfidf | 252 | 252 | 0.000000 | 1.000000 |
| confident_selection_error | dense_e5_rerank | 238 | 238 | 0.000000 | 1.000000 |
| confident_selection_error | hybrid_bge_m3 | 238 | 238 | 0.000000 | 1.000000 |
| confident_selection_error | hybrid_bge_m3_rerank | 213 | 213 | 0.000000 | 1.000000 |
| supported_but_wrong | hybrid_bge_m3_rerank | 166 | 166 | 0.000000 | 1.000000 |

### reader_llama31_8b_instruct

| slice | variant | num_queries | num_failures | accuracy | failure_rate |
| --- | --- | --- | --- | --- | --- |
| generic | hybrid_bge_m3 | 102 | 102 | 0.000000 | 1.000000 |
| generic | tfidf | 102 | 102 | 0.000000 | 1.000000 |
| duration | hybrid_bge_m3 | 26 | 26 | 0.000000 | 1.000000 |
| title_or_name | hybrid_bge_m3 | 24 | 24 | 0.000000 | 1.000000 |
| title_or_name | tfidf | 24 | 24 | 0.000000 | 1.000000 |
| where | dense_e5 | 20 | 20 | 0.000000 | 1.000000 |
| where | dense_e5_rerank | 20 | 20 | 0.000000 | 1.000000 |
| where | hybrid_bge_m3 | 20 | 20 | 0.000000 | 1.000000 |

| failure_bucket | variant | num_queries | num_failures | accuracy | failure_rate |
| --- | --- | --- | --- | --- | --- |
| supported_but_wrong | tfidf | 390 | 390 | 0.000000 | 1.000000 |
| supported_but_wrong | hybrid_bge_m3_rerank | 379 | 379 | 0.000000 | 1.000000 |
| supported_but_wrong | dense_e5_rerank | 378 | 378 | 0.000000 | 1.000000 |
| supported_but_wrong | hybrid_bge_m3 | 367 | 367 | 0.000000 | 1.000000 |
| supported_but_wrong | dense_e5 | 357 | 357 | 0.000000 | 1.000000 |
| confident_selection_error | hybrid_bge_m3 | 70 | 70 | 0.000000 | 1.000000 |

### reader_qwen25_7b_instruct

| slice | variant | num_queries | num_failures | accuracy | failure_rate |
| --- | --- | --- | --- | --- | --- |
| generic | tfidf | 102 | 102 | 0.000000 | 1.000000 |
| price | dense_e5 | 37 | 37 | 0.000000 | 1.000000 |
| duration | hybrid_bge_m3 | 26 | 26 | 0.000000 | 1.000000 |
| title_or_name | dense_e5 | 24 | 24 | 0.000000 | 1.000000 |
| title_or_name | dense_e5_rerank | 24 | 24 | 0.000000 | 1.000000 |
| title_or_name | hybrid_bge_m3 | 24 | 24 | 0.000000 | 1.000000 |
| title_or_name | hybrid_bge_m3_rerank | 24 | 24 | 0.000000 | 1.000000 |
| title_or_name | tfidf | 24 | 24 | 0.000000 | 1.000000 |

| failure_bucket | variant | num_queries | num_failures | accuracy | failure_rate |
| --- | --- | --- | --- | --- | --- |
| supported_but_wrong | tfidf | 386 | 386 | 0.000000 | 1.000000 |
| supported_but_wrong | dense_e5_rerank | 376 | 376 | 0.000000 | 1.000000 |
| supported_but_wrong | hybrid_bge_m3_rerank | 372 | 372 | 0.000000 | 1.000000 |
| supported_but_wrong | dense_e5 | 360 | 360 | 0.000000 | 1.000000 |
| supported_but_wrong | hybrid_bge_m3 | 351 | 351 | 0.000000 | 1.000000 |
| confident_selection_error | hybrid_bge_m3 | 80 | 80 | 0.000000 | 1.000000 |

### retrieval_only_baseline

| slice | variant | num_queries | num_failures | accuracy | failure_rate |
| --- | --- | --- | --- | --- | --- |
| price | dense_e5_rerank | 37 | 36 | 0.027027 | 0.972973 |
| price | hybrid_bge_m3_rerank | 37 | 36 | 0.027027 | 0.972973 |
| generic | dense_e5_rerank | 102 | 98 | 0.039216 | 0.960784 |
| generic | hybrid_bge_m3_rerank | 102 | 98 | 0.039216 | 0.960784 |
| generic | dense_e5 | 102 | 97 | 0.049020 | 0.950980 |
| generic | hybrid_bge_m3 | 102 | 97 | 0.049020 | 0.950980 |
| price | dense_e5 | 37 | 35 | 0.054054 | 0.945946 |
| price | hybrid_bge_m3 | 37 | 35 | 0.054054 | 0.945946 |

| failure_bucket | variant | num_queries | num_failures | accuracy | failure_rate |
| --- | --- | --- | --- | --- | --- |
| confident_selection_error | tfidf | 298 | 298 | 0.000000 | 1.000000 |
| confident_selection_error | dense_e5 | 252 | 252 | 0.000000 | 1.000000 |
| confident_selection_error | hybrid_bge_m3_rerank | 247 | 247 | 0.000000 | 1.000000 |
| supported_but_wrong | hybrid_bge_m3 | 245 | 245 | 0.000000 | 1.000000 |
| confident_selection_error | dense_e5_rerank | 240 | 240 | 0.000000 | 1.000000 |
| confident_selection_error | hybrid_bge_m3 | 149 | 149 | 0.000000 | 1.000000 |

## locomo

### proposed_learned_direct_valid

| slice | variant | num_queries | num_failures | accuracy | failure_rate |
| --- | --- | --- | --- | --- | --- |
| generic | dense_e5 | 876 | 848 | 0.031963 | 0.968037 |
| generic | hybrid_bge_m3 | 876 | 843 | 0.037671 | 0.962329 |
| generic | tfidf | 876 | 832 | 0.050228 | 0.949772 |
| generic | dense_e5_rerank | 876 | 829 | 0.053653 | 0.946347 |
| quantity | tfidf | 37 | 35 | 0.054054 | 0.945946 |
| generic | hybrid_bge_m3_rerank | 876 | 828 | 0.054795 | 0.945205 |
| group_name | dense_e5 | 53 | 49 | 0.075472 | 0.924528 |
| group_name | dense_e5_rerank | 53 | 49 | 0.075472 | 0.924528 |

| failure_bucket | variant | num_queries | num_failures | accuracy | failure_rate |
| --- | --- | --- | --- | --- | --- |
| confident_selection_error | tfidf | 961 | 961 | 0.000000 | 1.000000 |
| supported_but_wrong | hybrid_bge_m3 | 883 | 883 | 0.000000 | 1.000000 |
| supported_but_wrong | dense_e5 | 751 | 751 | 0.000000 | 1.000000 |
| confident_selection_error | hybrid_bge_m3_rerank | 707 | 707 | 0.000000 | 1.000000 |
| confident_selection_error | hybrid_bge_m3 | 706 | 706 | 0.000000 | 1.000000 |
| confident_selection_error | dense_e5_rerank | 701 | 701 | 0.000000 | 1.000000 |

### proposed_learned_direct_valid_resolver

| slice | variant | num_queries | num_failures | accuracy | failure_rate |
| --- | --- | --- | --- | --- | --- |
| generic | dense_e5 | 876 | 846 | 0.034247 | 0.965753 |
| generic | hybrid_bge_m3 | 876 | 844 | 0.036530 | 0.963470 |
| generic | tfidf | 876 | 834 | 0.047945 | 0.952055 |
| generic | dense_e5_rerank | 876 | 830 | 0.052511 | 0.947489 |
| generic | hybrid_bge_m3_rerank | 876 | 828 | 0.054795 | 0.945205 |
| group_name | dense_e5_rerank | 53 | 49 | 0.075472 | 0.924528 |
| group_name | hybrid_bge_m3 | 53 | 49 | 0.075472 | 0.924528 |
| group_name | hybrid_bge_m3_rerank | 53 | 49 | 0.075472 | 0.924528 |

| failure_bucket | variant | num_queries | num_failures | accuracy | failure_rate |
| --- | --- | --- | --- | --- | --- |
| confident_selection_error | tfidf | 1117 | 1117 | 0.000000 | 1.000000 |
| confident_selection_error | hybrid_bge_m3 | 955 | 955 | 0.000000 | 1.000000 |
| confident_selection_error | hybrid_bge_m3_rerank | 934 | 934 | 0.000000 | 1.000000 |
| confident_selection_error | dense_e5_rerank | 909 | 909 | 0.000000 | 1.000000 |
| confident_selection_error | dense_e5 | 846 | 846 | 0.000000 | 1.000000 |
| supported_but_wrong | hybrid_bge_m3 | 638 | 638 | 0.000000 | 1.000000 |

### reader_llama31_8b_instruct

| slice | variant | num_queries | num_failures | accuracy | failure_rate |
| --- | --- | --- | --- | --- | --- |
| when | dense_e5 | 271 | 271 | 0.000000 | 1.000000 |
| when | hybrid_bge_m3 | 271 | 271 | 0.000000 | 1.000000 |
| when | hybrid_bge_m3_rerank | 271 | 271 | 0.000000 | 1.000000 |
| when | tfidf | 271 | 271 | 0.000000 | 1.000000 |
| item_name | dense_e5 | 182 | 182 | 0.000000 | 1.000000 |
| item_name | hybrid_bge_m3_rerank | 182 | 182 | 0.000000 | 1.000000 |
| title_or_name | tfidf | 160 | 160 | 0.000000 | 1.000000 |
| where | hybrid_bge_m3_rerank | 91 | 91 | 0.000000 | 1.000000 |

| failure_bucket | variant | num_queries | num_failures | accuracy | failure_rate |
| --- | --- | --- | --- | --- | --- |
| supported_but_wrong | tfidf | 1728 | 1728 | 0.000000 | 1.000000 |
| supported_but_wrong | hybrid_bge_m3 | 1682 | 1682 | 0.000000 | 1.000000 |
| supported_but_wrong | hybrid_bge_m3_rerank | 1576 | 1576 | 0.000000 | 1.000000 |
| supported_but_wrong | dense_e5_rerank | 1532 | 1532 | 0.000000 | 1.000000 |
| supported_but_wrong | dense_e5 | 1488 | 1488 | 0.000000 | 1.000000 |
| retrieval_miss | dense_e5 | 343 | 343 | 0.000000 | 1.000000 |

### reader_qwen25_7b_instruct

| slice | variant | num_queries | num_failures | accuracy | failure_rate |
| --- | --- | --- | --- | --- | --- |
| when | dense_e5 | 271 | 271 | 0.000000 | 1.000000 |
| when | hybrid_bge_m3 | 271 | 271 | 0.000000 | 1.000000 |
| when | hybrid_bge_m3_rerank | 271 | 271 | 0.000000 | 1.000000 |
| when | tfidf | 271 | 271 | 0.000000 | 1.000000 |
| item_name | hybrid_bge_m3_rerank | 182 | 182 | 0.000000 | 1.000000 |
| item_name | tfidf | 182 | 182 | 0.000000 | 1.000000 |
| title_or_name | tfidf | 160 | 160 | 0.000000 | 1.000000 |
| where | dense_e5_rerank | 91 | 91 | 0.000000 | 1.000000 |

| failure_bucket | variant | num_queries | num_failures | accuracy | failure_rate |
| --- | --- | --- | --- | --- | --- |
| supported_but_wrong | tfidf | 1741 | 1741 | 0.000000 | 1.000000 |
| supported_but_wrong | hybrid_bge_m3 | 1704 | 1704 | 0.000000 | 1.000000 |
| supported_but_wrong | hybrid_bge_m3_rerank | 1589 | 1589 | 0.000000 | 1.000000 |
| supported_but_wrong | dense_e5_rerank | 1555 | 1555 | 0.000000 | 1.000000 |
| supported_but_wrong | dense_e5 | 1486 | 1486 | 0.000000 | 1.000000 |
| retrieval_miss | dense_e5 | 344 | 344 | 0.000000 | 1.000000 |

### retrieval_only_baseline

| slice | variant | num_queries | num_failures | accuracy | failure_rate |
| --- | --- | --- | --- | --- | --- |
| generic | dense_e5 | 876 | 849 | 0.030822 | 0.969178 |
| generic | hybrid_bge_m3 | 876 | 843 | 0.037671 | 0.962329 |
| generic | tfidf | 876 | 831 | 0.051370 | 0.948630 |
| generic | dense_e5_rerank | 876 | 829 | 0.053653 | 0.946347 |
| generic | hybrid_bge_m3_rerank | 876 | 829 | 0.053653 | 0.946347 |
| quantity | tfidf | 37 | 35 | 0.054054 | 0.945946 |
| group_name | dense_e5 | 53 | 50 | 0.056604 | 0.943396 |
| group_name | hybrid_bge_m3 | 53 | 50 | 0.056604 | 0.943396 |

| failure_bucket | variant | num_queries | num_failures | accuracy | failure_rate |
| --- | --- | --- | --- | --- | --- |
| confident_selection_error | tfidf | 1259 | 1259 | 0.000000 | 1.000000 |
| confident_selection_error | hybrid_bge_m3_rerank | 965 | 965 | 0.000000 | 1.000000 |
| confident_selection_error | dense_e5_rerank | 959 | 959 | 0.000000 | 1.000000 |
| confident_selection_error | dense_e5 | 895 | 895 | 0.000000 | 1.000000 |
| confident_selection_error | hybrid_bge_m3 | 854 | 854 | 0.000000 | 1.000000 |
| supported_but_wrong | hybrid_bge_m3 | 767 | 767 | 0.000000 | 1.000000 |
