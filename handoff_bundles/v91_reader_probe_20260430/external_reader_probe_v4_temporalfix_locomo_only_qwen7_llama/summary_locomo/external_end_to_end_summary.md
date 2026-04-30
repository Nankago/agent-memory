| benchmark | retrieval_variant | method | num_queries | accuracy | exact_match | hit_at_1 | mrr | retrieval_support_recall_at_retrieve_k | retrieval_support_recall_at_final_k | notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| locomo | hybrid_bge_m3_rerank | reader_llama31_8b_instruct | 1986 | 0.1853 | 0.1037 | 0.6291 | 0.6993 | 0.9766 | 0.8267 | Open-model reader baseline using Meta-Llama-3.1-8B-Instruct on the retrieved top-k snippets. |
| locomo | hybrid_bge_m3_rerank | reader_qwen25_7b_instruct | 1986 | 0.15760322255790535 | 0.09768378650553877 | 0.6291056088933805 | 0.699307811384618 | 0.9766087431993237 | 0.8266673014102924 | Open-model reader baseline using Qwen2.5-7B-instruct on the retrieved top-k snippets. |
