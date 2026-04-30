| benchmark | retrieval_variant | method | num_queries | accuracy | exact_match | hit_at_1 | mrr | retrieval_support_recall_at_retrieve_k | retrieval_support_recall_at_final_k | notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| longmemeval | hybrid_bge_m3_rerank | reader_qwen25_14b_instruct | 500 | 0.1240 | 0.0860 | 0.8620 | 0.9014 | 0.9827 | 0.9432 | Open-model reader baseline using Qwen2.5-14B-Instruct on the retrieved top-k snippets. |
