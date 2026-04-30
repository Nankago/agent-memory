# Reader Strict Metrics

| benchmark | retrieval_variant | method | num_queries | relaxed_accuracy | exact_match | unknown_rate | empty_gold_count | nonempty_num_queries | nonempty_relaxed_accuracy | nonempty_exact_match | json_object_rate | prompt_versions |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| locomo | hybrid_bge_m3_rerank | reader_llama31_8b_instruct | 1986 | 0.185297 | 0.103726 | 0.093152 | 444 | 1542 | 0.238651 | 0.133593 | 0.982880 | reader_json_evidence_v3_temporal_nonabstain |
| locomo | hybrid_bge_m3_rerank | reader_qwen25_7b_instruct | 1986 | 0.157603 | 0.097684 | 0.390232 | 444 | 1542 | 0.202983 | 0.125811 | 0.996979 | reader_json_evidence_v3_temporal_nonabstain |
| longmemeval | hybrid_bge_m3_rerank | reader_llama31_8b_instruct | 500 | 0.146000 | 0.102000 | 0.198000 | 0 | 500 | 0.146000 | 0.102000 | 0.992000 | reader_json_evidence_v3_temporal_nonabstain |
| longmemeval | hybrid_bge_m3_rerank | reader_qwen25_7b_instruct | 500 | 0.128000 | 0.084000 | 0.532000 | 0 | 500 | 0.128000 | 0.084000 | 0.992000 | reader_json_evidence_v3_temporal_nonabstain |
