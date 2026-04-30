# Reader Strict Metrics

| benchmark | retrieval_variant | method | num_queries | relaxed_accuracy | exact_match | unknown_rate | empty_gold_count | nonempty_num_queries | nonempty_relaxed_accuracy | nonempty_exact_match | json_object_rate | prompt_versions |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| locomo | hybrid_bge_m3_rerank | reader_llama31_8b_instruct | 1986 | 0.135952 | 0.074522 | 0.133434 | 444 | 1542 | 0.175097 | 0.095979 | 0.985901 | reader_json_evidence_v3_temporal_nonabstain |
| locomo | hybrid_bge_m3_rerank | reader_qwen25_14b_instruct | 1986 | 0.125378 | 0.072508 | 0.504532 | 444 | 1542 | 0.161479 | 0.093385 | 0.999496 | reader_json_evidence_v3_temporal_nonabstain |
| locomo | hybrid_bge_m3_rerank | reader_qwen25_7b_instruct | 1986 | 0.115307 | 0.066465 | 0.440584 | 444 | 1542 | 0.148508 | 0.085603 | 0.997986 | reader_json_evidence_v3_temporal_nonabstain |
| longmemeval | hybrid_bge_m3_rerank | reader_llama31_8b_instruct | 500 | 0.138000 | 0.096000 | 0.228000 | 0 | 500 | 0.138000 | 0.096000 | 0.988000 | reader_json_evidence_v3_temporal_nonabstain |
| longmemeval | hybrid_bge_m3_rerank | reader_qwen25_14b_instruct | 500 | 0.126000 | 0.088000 | 0.498000 | 0 | 500 | 0.126000 | 0.088000 | 0.996000 | reader_json_evidence_v3_temporal_nonabstain |
| longmemeval | hybrid_bge_m3_rerank | reader_qwen25_7b_instruct | 500 | 0.126000 | 0.084000 | 0.528000 | 0 | 500 | 0.126000 | 0.084000 | 0.990000 | reader_json_evidence_v3_temporal_nonabstain |
