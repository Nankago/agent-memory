# Reader Strict Metrics

| benchmark | retrieval_variant | method | num_queries | relaxed_accuracy | exact_match | unknown_rate | empty_gold_count | nonempty_num_queries | nonempty_relaxed_accuracy | nonempty_exact_match | json_object_rate | prompt_versions |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| locomo | hybrid_bge_m3_rerank | reader_qwen25_14b_instruct | 1986 | 0.169184 | 0.102719 | 0.455186 | 444 | 1542 | 0.217899 | 0.132296 | 0.999496 | reader_json_evidence_v3_temporal_nonabstain |
| longmemeval | hybrid_bge_m3_rerank | reader_qwen25_14b_instruct | 500 | 0.124000 | 0.086000 | 0.510000 | 0 | 500 | 0.124000 | 0.086000 | 0.992000 | reader_json_evidence_v3_temporal_nonabstain |
