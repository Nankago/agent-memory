# Reader Probe V2 Bundle

This branch contains the fixed server bundle for rerunning the Qwen/LLaMA reader probe only.

Package:
- handoff_bundles/agent-memory-server-bundle-v90-20260425-reader-probe-v2-package.zip

Main changes:
- JSON-only reader prompt
- quoted evidence snippets
- prompt truncation preserves final question/instructions
- max_new_tokens default 24
- probe script: scripts/run_external_reader_probe_v2.sh

Run on server:

unzip handoff_bundles/agent-memory-server-bundle-v90-20260425-reader-probe-v2-package.zip
cd agent-memory-server-bundle-v90-20260422-final
bash scripts/run_external_reader_probe_v2.sh
