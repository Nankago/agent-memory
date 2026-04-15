from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from memory_collapse.io_utils import ensure_dir, read_csv, read_jsonl


BASELINE_PLOT_METHODS = [
    "latest_write",
    "tfidf_only",
    "tfidf_plus_recency",
    "proposed_heuristic",
    "oracle_valid",
]


def _bin_series(series: pd.Series, bins: list[float], labels: list[str]) -> pd.Series:
    return pd.cut(series, bins=bins, labels=labels, include_lowest=True)


def _primary_proposed_method(stress_metrics: pd.DataFrame) -> str:
    proposed = stress_metrics[stress_metrics["method"].str.startswith("proposed_")].copy()
    if proposed.empty:
        return "proposed_heuristic"
    summary = (
        proposed.groupby("method", as_index=False)
        .agg(mean_accuracy=("accuracy", "mean"), mean_collapse=("collapse_rate", "mean"))
        .sort_values(["mean_accuracy", "mean_collapse", "method"], ascending=[False, True, True])
    )
    return str(summary.iloc[0]["method"])


def plot_main_figures(run_dir: str | Path) -> dict[str, str]:
    root = Path(run_dir)
    figures_dir = ensure_dir(root / "figures")
    metrics = read_csv(root / "results" / "metrics_by_method.csv")
    diagnostics = pd.DataFrame(read_jsonl(root / "results" / "query_diagnostics.jsonl"))

    stress_metrics = metrics[metrics["stress_name"] != "overall"].copy()
    stress_metrics = stress_metrics.sort_values(["stress_value", "method"])
    proposed_method = _primary_proposed_method(stress_metrics)

    sns.set_theme(style="whitegrid")

    collapse_path = figures_dir / "collapse_curve.png"
    plt.figure(figsize=(9, 5))
    plot_methods = [*BASELINE_PLOT_METHODS[:-1], proposed_method, BASELINE_PLOT_METHODS[-1]]
    plot_frame = stress_metrics[stress_metrics["method"].isin(plot_methods)]
    sns.lineplot(data=plot_frame, x="stress_value", y="collapse_rate", hue="method", marker="o")
    plt.title("Collapse Curve Along Composite Stress Path")
    plt.xlabel("Composite stress")
    plt.ylabel("Collapse rate")
    plt.tight_layout()
    plt.savefig(collapse_path, dpi=200)
    plt.close()

    decomp_path = figures_dir / "failure_decomposition.png"
    plt.figure(figsize=(9, 5))
    proposed = stress_metrics[stress_metrics["method"] == proposed_method].sort_values("stress_value")
    x = proposed["stress_value"].to_numpy()
    forgetting = proposed["forgetting_rate"].to_numpy()
    stale = proposed["stale_dominance_rate"].to_numpy()
    residual = proposed["residual_rate"].to_numpy()
    plt.stackplot(
        x,
        forgetting,
        stale,
        residual,
        labels=["forgetting", "stale dominance", "residual"],
        alpha=0.85,
    )
    plt.legend(loc="upper left")
    plt.title(f"Failure Decomposition for {proposed_method}")
    plt.xlabel("Composite stress")
    plt.ylabel("Rate")
    plt.tight_layout()
    plt.savefig(decomp_path, dpi=200)
    plt.close()

    oracle_gap_path = figures_dir / "oracle_gap.png"
    plt.figure(figsize=(9, 5))
    oracle_frame = stress_metrics[
        stress_metrics["method"].isin(["latest_write", proposed_method, "oracle_latest", "oracle_valid"])
    ]
    sns.lineplot(data=oracle_frame, x="stress_value", y="accuracy", hue="method", marker="o")
    plt.title("Oracle Gap vs Proposed Controller")
    plt.xlabel("Composite stress")
    plt.ylabel("Accuracy")
    plt.tight_layout()
    plt.savefig(oracle_gap_path, dpi=200)
    plt.close()

    heatmap_path = figures_dir / "conflict_heatmap.png"
    plt.figure(figsize=(8, 6))
    conflict = diagnostics[
        (diagnostics["method"] == proposed_method)
        & (diagnostics["conflict_present"] == True)
        & diagnostics["age_gap"].notna()
        & diagnostics["reliability_gap"].notna()
    ].copy()
    if conflict.empty:
        heatmap = pd.DataFrame([[0.0]], index=["n/a"], columns=["n/a"])
    else:
        conflict["age_gap_bin"] = _bin_series(
            conflict["age_gap"],
            bins=[-10.0, -2.0, 0.0, 2.0, 10.0],
            labels=["much older gold", "older gold", "older wrong", "much older wrong"],
        )
        conflict["rel_gap_bin"] = _bin_series(
            conflict["reliability_gap"],
            bins=[-1.0, -0.1, 0.1, 1.0],
            labels=["wrong stronger", "balanced", "gold stronger"],
        )
        conflict["correct"] = (~conflict["is_error"]).astype(float)
        heatmap = conflict.pivot_table(
            index="age_gap_bin",
            columns="rel_gap_bin",
            values="correct",
            aggfunc="mean",
            fill_value=0.0,
        )
    sns.heatmap(heatmap, annot=True, fmt=".2f", cmap="viridis", vmin=0.0, vmax=1.0)
    plt.title(f"Conflict Robustness Heatmap ({proposed_method})")
    plt.xlabel("Reliability gap")
    plt.ylabel("Age gap")
    plt.tight_layout()
    plt.savefig(heatmap_path, dpi=200)
    plt.close()

    return {
        "collapse_curve": str(collapse_path),
        "failure_decomposition": str(decomp_path),
        "oracle_gap": str(oracle_gap_path),
        "conflict_heatmap": str(heatmap_path),
    }
