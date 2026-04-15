from __future__ import annotations

from pathlib import Path

import pandas as pd

from memory_collapse.baselines import DIRECT_VALID_RESOLVER_METHOD
from memory_collapse.external_pipeline import RETRIEVAL_ONLY_BASELINE, run_external_end_to_end
from memory_collapse.io_utils import read_jsonl, write_jsonl
from tests.test_helpers import fresh_run_dir


class _FakeRelevanceBundle:
    def predict_relevance(self, query, memory, tfidf_score, observable_context=None):
        text = str(memory["memory_text"]).lower()
        score = 0.25 + 0.55 * float(tfidf_score)
        if "berlin" in text or "travel refund" in text:
            score += 0.20
        return max(0.02, min(0.99, score))


class _FakeQueryValidityBundle:
    def __init__(self, target_label: str):
        self.target_label = target_label
        self.condition = "relevant_only"

    def predict_query_validity(self, query, memory, tfidf_score, observable_context=None):
        text = str(memory["memory_text"]).lower()
        score = 0.30 + 0.40 * float(tfidf_score)
        if "berlin" in text or "travel refund" in text:
            score += 0.25
        if "apples" in text or "identity" in text:
            score -= 0.10
        return max(0.02, min(0.99, score))


class _FakeAntiSupportBundle:
    def predict_anti_support(self, feature_dict):
        return 0.10


class _FakeValueResolverBundle:
    condition = "top_k_direct_valid_candidates"
    objective = "candidate_binary_classification"

    def predict_candidate_scores(self, feature_rows):
        scores: list[float] = []
        for row in feature_rows:
            scores.append(
                float(row.get("candidate_support_sum", 0.0))
                + 0.10 * float(row.get("candidate_rho_mean", 0.0))
                - 0.10 * float(row.get("candidate_self_anti_sum", 0.0))
            )
        return scores


def test_external_end_to_end_runner_writes_diagnostics_metrics_and_summary(monkeypatch):
    run_dir = fresh_run_dir("external_end_to_end")
    retrieval_dir = run_dir / "longmemeval" / "dense_e5_rerank"
    write_jsonl(
        retrieval_dir / "retrieval_diagnostics.jsonl",
        [
            {
                "query_id": "lm-1",
                "benchmark": "longmemeval",
                "prompt": "Where did Alice move?",
                "gold_answer": "Berlin",
                "answer_support_ids": ["lm-1_m0000"],
                "retrieve_top_k": 2,
                "final_top_k": 2,
                "support_recall_at_retrieve_k": 1.0,
                "support_recall_at_final_k": 1.0,
                "support_hit_at_1": 1.0,
                "support_mrr": 1.0,
                "retrieved": [
                    {
                        "memory_id": "lm-1_m0000",
                        "score": 0.98,
                        "text": "Alice moved to Berlin last spring.",
                    },
                    {
                        "memory_id": "lm-1_m0001",
                        "score": 0.20,
                        "text": "Alice bought apples from the market.",
                    },
                ],
                "final_ranked": [
                    {
                        "memory_id": "lm-1_m0000",
                        "score": 0.98,
                        "text": "Alice moved to Berlin last spring.",
                    },
                    {
                        "memory_id": "lm-1_m0001",
                        "score": 0.20,
                        "text": "Alice bought apples from the market.",
                    },
                ],
            },
            {
                "query_id": "lm-2",
                "benchmark": "longmemeval",
                "prompt": "What refund did the customer ask for?",
                "gold_answer": "travel refund",
                "answer_support_ids": ["lm-2_m0000"],
                "retrieve_top_k": 2,
                "final_top_k": 2,
                "support_recall_at_retrieve_k": 1.0,
                "support_recall_at_final_k": 1.0,
                "support_hit_at_1": 1.0,
                "support_mrr": 1.0,
                "retrieved": [
                    {
                        "memory_id": "lm-2_m0000",
                        "score": 0.91,
                        "text": "The customer asked for a travel refund after the canceled booking.",
                    },
                    {
                        "memory_id": "lm-2_m0001",
                        "score": 0.30,
                        "text": "The agent confirmed identity before continuing.",
                    },
                ],
                "final_ranked": [
                    {
                        "memory_id": "lm-2_m0000",
                        "score": 0.91,
                        "text": "The customer asked for a travel refund after the canceled booking.",
                    },
                    {
                        "memory_id": "lm-2_m0001",
                        "score": 0.30,
                        "text": "The agent confirmed identity before continuing.",
                    },
                ],
            },
        ],
    )

    def _fake_loader(_model_dir):
        return type(
            "_Bundles",
            (),
            {
                "relevance_bundle": _FakeRelevanceBundle(),
                "query_validity_bundles": {
                    "useful_label": _FakeQueryValidityBundle("useful_label"),
                    "valid_label": _FakeQueryValidityBundle("valid_label"),
                },
                "anti_support_bundle": _FakeAntiSupportBundle(),
                "value_resolver_bundle": _FakeValueResolverBundle(),
            },
        )()

    import memory_collapse.external_pipeline as external_pipeline

    monkeypatch.setattr(external_pipeline, "_load_external_method_bundles", _fake_loader)

    outputs = run_external_end_to_end(
        benchmark_name="longmemeval",
        retrieval_inputs={"dense_e5_rerank": retrieval_dir},
        methods=[RETRIEVAL_ONLY_BASELINE, DIRECT_VALID_RESOLVER_METHOD],
        output_root=run_dir / "outputs",
        summary_root=run_dir / "outputs",
        model_dir=run_dir / "fake_models",
        final_k=2,
    )

    summary_csv = Path(outputs["summary_csv"])
    summary_md = Path(outputs["summary_md"])
    metrics_csv = Path(outputs["metrics_csv"])
    retrieval_diag = Path(outputs["diagnostics"]["dense_e5_rerank:retrieval_only_baseline"])
    resolver_diag = Path(outputs["diagnostics"][f"dense_e5_rerank:{DIRECT_VALID_RESOLVER_METHOD}"])

    assert summary_csv.exists()
    assert summary_md.exists()
    assert metrics_csv.exists()
    assert retrieval_diag.exists()
    assert resolver_diag.exists()

    retrieval_rows = read_jsonl(retrieval_diag)
    resolver_rows = read_jsonl(resolver_diag)
    metrics = pd.read_csv(metrics_csv)
    summary = pd.read_csv(summary_csv)

    assert len(retrieval_rows) == 2
    assert len(resolver_rows) == 2
    assert retrieval_rows[0]["candidate_value_scores"]
    assert resolver_rows[0]["direct_valid_metadata"]["component_debug"]
    assert resolver_rows[0]["resolver_metadata"]["used"] is True
    assert set(metrics["method"]) == {RETRIEVAL_ONLY_BASELINE, DIRECT_VALID_RESOLVER_METHOD}
    assert set(summary["method"]) == {RETRIEVAL_ONLY_BASELINE, DIRECT_VALID_RESOLVER_METHOD}
    assert float(metrics["accuracy"].min()) >= 1.0
    assert "dense_e5_rerank" in summary_md.read_text(encoding="utf-8")
    assert DIRECT_VALID_RESOLVER_METHOD in summary_md.read_text(encoding="utf-8")
