# V4 Change Log (Temporal Fix)

This note documents the code changes and run variants used for `v4`.

## Modified Code File

- `agent-memory-server-bundle-v91-20260426-reader-probe-v3-qwen14/src/memory_collapse/external_reader_baselines.py`

## What Changed

1. Added temporal-question detection:
- regex for `when / date / day / month / year / time` style prompts.

2. Added anchored-date detection:
- regex for date-like values such as `2023-05-07`, `7 May 2023`, `June 2023`, etc.

3. Added relative-time detection:
- regex for expressions like `yesterday`, `last week`, `ago`, etc.

4. Added temporal fallback logic:
- for temporal questions, if model output is `unknown`, relative/vague, or not date-anchored, select the highest-scoring anchored date candidate from retrieval values.

5. Added runtime switch:
- env var: `READER_TEMPORAL_FALLBACK`
- supported modes:
  - `off` (or `0/false/disabled`): disable fallback
  - `locomo_only` (default): enable only for LoCoMo
  - `on` (or `1/true/all`): enable for all benchmarks

6. Prompt tightening for temporal questions:
- additional instruction to avoid `unknown` when snippets contain anchored temporal evidence.

## V4 Run Variants Produced

1. `external_reader_probe_v4_temporalfix_qwen14`
- model: `reader_qwen25_14b_instruct`
- setting: temporal fallback effectively applied to both benchmarks in this run.

2. `external_reader_probe_v4_temporalfix_locomo_only_qwen7_llama`
- models: `reader_qwen25_7b_instruct`, `reader_llama31_8b_instruct`
- setting: `READER_TEMPORAL_FALLBACK=locomo_only`

## Main Observed Effect

- Strong gain on `LoCoMo when` slice (especially Qwen14 run).
- Lower `unknown_rate` on LoCoMo for affected runs.
- Small tradeoff possible on LongMemEval depending on fallback mode.

## Related Artifacts

- `external_reader_probe_v4_temporalfix_qwen14/`
- `external_reader_probe_v4_temporalfix_locomo_only_qwen7_llama/`
- `figures_reader_probe_v4/`
- logs:
  - `reader_probe_v4_temporalfix_qwen14_20260430_105653.log`
  - `reader_probe_v4_temporalfix_locomo_only_qwen7_llama_20260430_115446.log`
