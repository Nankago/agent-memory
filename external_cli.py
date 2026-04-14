from __future__ import annotations

import argparse
import json

from memory_collapse.external import JsonlExternalAdapter
from memory_collapse.external_preprocess import convert_raw_external
from memory_collapse.external_retrieval import run_external_retrieval


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Memory collapse external retrieval CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    convert_parser = subparsers.add_parser("convert_raw_external", help="Convert raw LongMemEval or LoCoMo benchmark files into the JSONL shape expected by prepare_external.")
    convert_parser.add_argument("--benchmark", required=True, help="Benchmark name, for example longmemeval or locomo.")
    convert_parser.add_argument("--input-path", required=True, help="Path to raw benchmark file.")
    convert_parser.add_argument("--output-path", required=True, help="Path for converted JSONL.")

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
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "convert_raw_external":
        outputs = convert_raw_external(args.benchmark, args.input_path, args.output_path)
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

    raise ValueError(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    main()
