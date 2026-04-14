from __future__ import annotations

import argparse
import json
from pathlib import Path

from memory_collapse.baselines import run_baselines
from memory_collapse.config import load_config
from memory_collapse.estimators import train_estimators
from memory_collapse.external import JsonlExternalAdapter
from memory_collapse.external_pipeline import run_external_end_to_end
from memory_collapse.external_retrieval import run_external_retrieval
from memory_collapse.generator import generate_artifacts
from memory_collapse.plots import plot_main_figures


def _default_run_dir(config: dict) -> Path:
    return Path(config["output_root"]) / config["run_name"]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Memory collapse synthetic benchmark CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    generate_parser = subparsers.add_parser("generate", help="Generate synthetic world, memories, queries, and exact labels.")
    generate_parser.add_argument("--config", required=True, help="Path to YAML config.")
    generate_parser.add_argument("--output-dir", help="Optional explicit run directory.")

    train_parser = subparsers.add_parser("train_estimators", help="Train learned write, survival, and relevance estimators.")
    train_parser.add_argument("--run-dir", required=True, help="Generated run directory with synthetic artifacts.")
    train_parser.add_argument("--output-dir", help="Optional directory for model artifacts. Defaults to <run-dir>/models.")

    baseline_parser = subparsers.add_parser("run_baselines", help="Run heuristic baselines and proposed controllers.")
    baseline_parser.add_argument("--run-dir", required=True, help="Generated run directory.")
    baseline_parser.add_argument("--estimator-dir", help="Optional run/models directory to load learned estimators from.")
    baseline_parser.add_argument("--force-train", action="store_true", help="Retrain estimators before running baselines.")
    baseline_parser.add_argument("--skip-train", action="store_true", help="Fail if estimators are missing instead of training them.")

    plot_parser = subparsers.add_parser("plot", help="Render the fixed main figures.")
    plot_parser.add_argument("--run-dir", required=True, help="Run directory with metrics and diagnostics.")

    external_parser = subparsers.add_parser("prepare_external", help="Normalize a JSONL external benchmark into a common adapter format.")
    external_parser.add_argument("--benchmark", required=True, help="Benchmark name, for example longmemeval or locomo.")
    external_parser.add_argument("--input-path", required=True, help="Path to source JSONL file.")
    external_parser.add_argument("--output-dir", required=True, help="Directory for normalized adapter artifacts.")

    retrieval_parser = subparsers.add_parser("run_external_retrieval", help="Run retrieval and optional reranking on normalized external benchmark data.")
    retrieval_parser.add_argument("--normalized-dir", required=True, help="Directory created by prepare_external.")
    retrieval_parser.add_argument("--output-dir", required=True, help="Directory for retrieval diagnostics and summary.")
    retrieval_parser.add_argument("--retriever", default="tfidf", choices=["tfidf", "dense", "hybrid"], help="Retriever backend.")
    retrieval_parser.add_argument("--retriever-model", help="Local path or model id for dense/hybrid retrieval.")
    retrieval_parser.add_argument("--reranker-model", help="Optional local path or model id for cross-encoder reranking.")
    retrieval_parser.add_argument("--device", default="cpu", help="Torch device, for example cpu or cuda.")
    retrieval_parser.add_argument("--retrieve-top-k", type=int, default=20, help="How many items to keep after first-stage retrieval.")
    retrieval_parser.add_argument("--final-top-k", type=int, default=10, help="How many items to keep after reranking.")
    retrieval_parser.add_argument("--batch-size", type=int, default=16, help="Batch size for dense retrieval and reranking.")

    end_to_end_parser = subparsers.add_parser(
        "run_external_end_to_end",
        help="Run answer-level external evaluation on top of existing retrieval diagnostics.",
    )
    end_to_end_parser.add_argument("--benchmark", required=True, help="Benchmark name, for example longmemeval or locomo.")
    end_to_end_parser.add_argument("--retrieval-variant", action="append", required=True, help="Retrieval variant name. Repeat for batch runs.")
    end_to_end_parser.add_argument("--input-dir", help="Single retrieval variant directory containing retrieval_diagnostics.jsonl.")
    end_to_end_parser.add_argument("--input-root", help="Parent directory whose children are retrieval variant folders.")
    end_to_end_parser.add_argument("--method", action="append", help="Method to run. Repeat for multiple methods.")
    end_to_end_parser.add_argument("--model-dir", help="Synthetic model artifact directory used by direct_valid / resolver methods.")
    end_to_end_parser.add_argument("--output-root", default="outputs/external_runs", help="Root directory for external end-to-end outputs.")
    end_to_end_parser.add_argument("--summary-root", help="Optional directory for the combined summary tables.")
    end_to_end_parser.add_argument("--top-k", type=int, help="Optional cap on retrieved candidates before method scoring.")
    end_to_end_parser.add_argument("--final-k", type=int, help="Optional cap on reranked candidates before method scoring.")
    return parser


def _resolve_external_inputs(args: argparse.Namespace) -> dict[str, str]:
    variants = [variant.strip() for variant in args.retrieval_variant if variant and variant.strip()]
    if args.input_dir and args.input_root:
        raise ValueError("--input-dir and --input-root cannot be used together.")
    if args.input_dir:
        if len(variants) != 1:
            raise ValueError("--input-dir only supports a single --retrieval-variant.")
        return {variants[0]: args.input_dir}
    if args.input_root:
        root = Path(args.input_root)
        return {variant: str(root / variant) for variant in variants}
    raise ValueError("Either --input-dir or --input-root is required for run_external_end_to_end.")


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "generate":
        config = load_config(args.config)
        run_dir = Path(args.output_dir) if args.output_dir else _default_run_dir(config)
        manifest = generate_artifacts(config, run_dir)
        print(json.dumps({"run_dir": str(run_dir), "manifest": manifest}, indent=2))
        return

    if args.command == "train_estimators":
        outputs = train_estimators(args.run_dir, output_dir=args.output_dir)
        print(json.dumps(outputs, indent=2))
        return

    if args.command == "run_baselines":
        outputs = run_baselines(
            args.run_dir,
            estimator_dir=args.estimator_dir,
            force_train=args.force_train,
            skip_train=args.skip_train,
        )
        print(json.dumps(outputs, indent=2))
        return

    if args.command == "plot":
        outputs = plot_main_figures(args.run_dir)
        print(json.dumps(outputs, indent=2))
        return

    if args.command == "prepare_external":
        adapter = JsonlExternalAdapter(args.benchmark)
        outputs = adapter.adapt(args.input_path, args.output_dir)
        print(json.dumps(outputs, indent=2))
        return

    if args.command == "run_external_retrieval":
        outputs = run_external_retrieval(
            args.normalized_dir,
            args.output_dir,
            retriever=args.retriever,
            retriever_model=args.retriever_model,
            reranker_model=args.reranker_model,
            device=args.device,
            retrieve_top_k=args.retrieve_top_k,
            final_top_k=args.final_top_k,
            batch_size=args.batch_size,
        )
        print(json.dumps(outputs, indent=2))
        return

    if args.command == "run_external_end_to_end":
        outputs = run_external_end_to_end(
            benchmark_name=args.benchmark,
            retrieval_inputs=_resolve_external_inputs(args),
            methods=args.method or [],
            output_root=args.output_root,
            model_dir=args.model_dir,
            top_k=args.top_k,
            final_k=args.final_k,
            summary_root=args.summary_root,
        )
        print(json.dumps(outputs, indent=2))
        return

    raise ValueError(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    main()
