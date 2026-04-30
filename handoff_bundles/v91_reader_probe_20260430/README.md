# Reader Probe V91 Handoff (2026-04-30)

This bundle contains the server outputs used for collaborator review.

## Included Runs

- `external_reader_probe_v3`  
  Baseline v3 run (Qwen2.5-7B, Llama-3.1-8B, Qwen2.5-14B).

- `external_reader_probe_v4_temporalfix_qwen14`  
  V4 temporal-fallback experiment for Qwen2.5-14B.

- `external_reader_probe_v4_temporalfix_locomo_only_qwen7_llama`  
  V4 temporal-fallback experiment for Qwen2.5-7B and Llama-3.1-8B with `READER_TEMPORAL_FALLBACK=locomo_only`.

- `figures_reader_probe_v4`  
  Comparison plots (`when` accuracy, unknown rate, core metrics, JSON rate).

## Logs

- `reader_probe_v3_20260426_220631.log`
- `reader_probe_v4_temporalfix_qwen14_20260430_105653.log`
- `reader_probe_v4_temporalfix_locomo_only_qwen7_llama_20260430_115446.log`

## Extra Archive

- `external_reader_probe_v3_handoff_20260426_220631.zip`
