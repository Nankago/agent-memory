#!/usr/bin/env python

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

BENCHMARKS = ["longmemeval", "locomo"]
VARIANTS = [
    "tfidf",
    "dense_e5",
    "dense_e5_rerank",
    "hybrid_bge_m3",
    "hybrid_bge_m3_rerank",
]
METRICS = [
    "support_recall_at_retrieve_k",
    "support_recall_at_final_k",
    "support_hit_at_1",
    "support_mrr",
]


def load_rows(output_root: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for benchmark in BENCHMARKS:
        for variant in VARIANTS:
            summary_path = output_root / benchmark / variant / "retrieval_summary.json"
            row = {"benchmark": benchmark, "variant": variant}
            if summary_path.exists():
                data = json.loads(summary_path.read_text())
                row["retrieve_top_k"] = str(data.get("retrieve_top_k", ""))
                row["final_top_k"] = str(data.get("final_top_k", ""))
                metrics = data.get("metrics", {})
                for metric in METRICS:
                    value = metrics.get(metric, "")
                    row[metric] = f"{float(value):.6f}" if value != "" else ""
            else:
                row["retrieve_top_k"] = ""
                row["final_top_k"] = ""
                for metric in METRICS:
                    row[metric] = ""
            rows.append(row)
    return rows


def write_csv(rows: list[dict[str, str]], path: Path) -> None:
    fieldnames = ["benchmark", "variant", "retrieve_top_k", "final_top_k", *METRICS]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(rows: list[dict[str, str]], path: Path) -> None:
    lines: list[str] = []
    title_map = {"longmemeval": "LongMemEval", "locomo": "LoCoMo"}
    for benchmark in BENCHMARKS:
        lines.append(f"## {title_map[benchmark]}")
        lines.append("")
        header = [
            "variant",
            "retrieve_top_k",
            "final_top_k",
            "support_recall_at_retrieve_k",
            "support_recall_at_final_k",
            "support_hit_at_1",
            "support_mrr",
        ]
        lines.append("| " + " | ".join(header) + " |")
        lines.append("| " + " | ".join(["---"] * len(header)) + " |")
        for row in rows:
            if row["benchmark"] != benchmark:
                continue
            lines.append(
                "| "
                + " | ".join(
                    [
                        row["variant"],
                        row["retrieve_top_k"],
                        row["final_top_k"],
                        row["support_recall_at_retrieve_k"],
                        row["support_recall_at_final_k"],
                        row["support_hit_at_1"],
                        row["support_mrr"],
                    ]
                )
                + " |"
            )
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize external retrieval results.")
    parser.add_argument(
        "--output-root",
        default="outputs/external_runs",
        help="Root directory containing benchmark outputs.",
    )
    parser.add_argument(
        "--csv-path",
        default="outputs/external_runs/retrieval_summary_table.csv",
        help="Output CSV path.",
    )
    parser.add_argument(
        "--md-path",
        default="outputs/external_runs/retrieval_summary_table.md",
        help="Output Markdown table path.",
    )
    args = parser.parse_args()

    output_root = Path(args.output_root)
    rows = load_rows(output_root)

    csv_path = Path(args.csv_path)
    md_path = Path(args.md_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.parent.mkdir(parents=True, exist_ok=True)

    write_csv(rows, csv_path)
    write_markdown(rows, md_path)
    print(f"wrote {csv_path}")
    print(f"wrote {md_path}")


if __name__ == "__main__":
    main()
