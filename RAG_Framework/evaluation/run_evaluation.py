"""CLI entry point for RAGAS evaluation.

Usage:
    # Quick benchmark (amnesty_qa, 30 questions)
    python -m evaluation.run_evaluation --mode benchmark --dataset amnesty_qa

    # Benchmark with limit
    python -m evaluation.run_evaluation --mode benchmark --dataset scifact --max-questions 50

    # Reference-free on own documents
    python -m evaluation.run_evaluation --mode reference-free --queries "Q1?" "Q2?"
    python -m evaluation.run_evaluation --mode reference-free --queries-file queries.txt
"""

import argparse
import os
import sys

# Ensure project root is on the path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
THESIS_ROOT = os.path.dirname(PROJECT_ROOT)
if THESIS_ROOT not in sys.path:
    sys.path.insert(0, THESIS_ROOT)

from mlx_lm import load

from RAG_Framework.core.config import (
    EMBEDDING_MODEL_NAME,
    MODEL_PATH,
    RERANKER_MODEL_NAME,
)
from RAG_Framework.components.retrievers import Retriever
from RAG_Framework.evaluation.mlx_llm_wrapper import MLXRagasLLM
from RAG_Framework.evaluation.embeddings_wrapper import E5RagasEmbeddings


def _build_components():
    """Load LLM, build retriever, and create RAGAS wrappers."""
    print("Loading LLM...")
    llm_model, llm_tokenizer = load(MODEL_PATH)

    print("Initialising retriever...")
    retriever = Retriever(
        llm_model=llm_model,
        llm_tokenizer=llm_tokenizer,
        embedding_model_name=EMBEDDING_MODEL_NAME,
        reranker_model_name=RERANKER_MODEL_NAME,
    )

    # Build RAGAS wrappers (reuse the retriever's lazily-loaded embedding model)
    ragas_llm = MLXRagasLLM(model=llm_model, tokenizer=llm_tokenizer)
    ragas_embeddings = E5RagasEmbeddings(retriever.embedding_model)

    return llm_model, llm_tokenizer, retriever, ragas_llm, ragas_embeddings


def main():
    parser = argparse.ArgumentParser(description="RAGAS Evaluation for the RAG Framework")
    parser.add_argument(
        "--mode",
        choices=["benchmark", "reference-free"],
        required=True,
        help="Evaluation mode.",
    )
    parser.add_argument(
        "--dataset",
        choices=["amnesty_qa", "scifact"],
        default="amnesty_qa",
        help="Benchmark dataset (only for --mode benchmark).",
    )
    parser.add_argument(
        "--max-questions",
        type=int,
        default=None,
        help="Limit number of questions to evaluate.",
    )
    parser.add_argument(
        "--queries",
        nargs="*",
        help="Queries for reference-free mode.",
    )
    parser.add_argument(
        "--queries-file",
        type=str,
        default=None,
        help="Path to a text file with one query per line.",
    )
    args = parser.parse_args()

    # Disable async to avoid issues with local model
    os.environ["RAGAS_RUN_ASYNC"] = "false"

    llm_model, llm_tokenizer, retriever, ragas_llm, ragas_embeddings = _build_components()

    if args.mode == "benchmark":
        from RAG_Framework.evaluation.benchmark import run_benchmark_evaluation

        run_benchmark_evaluation(
            dataset_name=args.dataset,
            llm_model=llm_model,
            llm_tokenizer=llm_tokenizer,
            retriever=retriever,
            ragas_llm=ragas_llm,
            ragas_embeddings=ragas_embeddings,
            max_questions=args.max_questions,
        )

    elif args.mode == "reference-free":
        from RAG_Framework.evaluation.reference_free import run_reference_free_evaluation

        queries = args.queries or []
        if args.queries_file:
            with open(args.queries_file) as f:
                queries.extend(line.strip() for line in f if line.strip())
        if not queries:
            parser.error("Provide --queries or --queries-file for reference-free mode.")

        run_reference_free_evaluation(
            queries=queries,
            retriever=retriever,
            llm_model=llm_model,
            llm_tokenizer=llm_tokenizer,
            ragas_llm=ragas_llm,
            ragas_embeddings=ragas_embeddings,
        )


if __name__ == "__main__":
    main()
