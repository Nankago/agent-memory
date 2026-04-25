| benchmark | retrieval_variant | method | num_queries | accuracy | exact_match | hit_at_1 | mrr | retrieval_support_recall_at_retrieve_k | retrieval_support_recall_at_final_k | notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| longmemeval | hybrid_bge_m3_rerank | reader_llama31_8b_instruct | 500 | 0.1160 | 0.0820 | 0.8620 | 0.9014 | 0.9827 | 0.9432 | Open-model reader baseline using Meta-Llama-3.1-8B-Instruct on the retrieved top-k snippets. |
| longmemeval | hybrid_bge_m3_rerank | reader_qwen25_7b_instruct | 500 | 0.114 | 0.072 | 0.862 | 0.901379365079365 | 0.9826666666666666 | 0.9431666666666667 | Open-model reader baseline using Qwen2.5-7B-Instruct on the retrieved top-k snippets. |
