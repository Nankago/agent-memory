| benchmark | retrieval_variant | method | num_queries | accuracy | exact_match | hit_at_1 | mrr | retrieval_support_recall_at_retrieve_k | retrieval_support_recall_at_final_k | notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| locomo | hybrid_bge_m3_rerank | reader_qwen25_14b_instruct | 1986 | 0.1692 | 0.1027 | 0.6291 | 0.6993 | 0.9766 | 0.8267 | Open-model reader baseline using Qwen2.5-14B-Instruct on the retrieved top-k snippets. |
