"""Benchmark evaluation against standard datasets with ground truth.

Supported datasets:
  - amnesty_qa (explodinggradients/amnesty_qa): 20 Qs, comes with contexts + ground truth.
  - scifact (mteb/scifact): 1109 queries, ~5K passages. Requires corpus indexing.
"""

import json
import math
import os
import time
from datetime import datetime

from datasets import load_dataset
from huggingface_hub import hf_hub_download
from mlx_lm import generate as mlx_generate
from mlx_lm.sample_utils import make_sampler
from ragas.metrics.collections import (
    AnswerRelevancy,
    ContextPrecisionWithReference,
    ContextRecall,
    FactualCorrectness,
    Faithfulness,
)

from RAG_Framework.core.config import (
    EVAL_BENCHMARK_DATASET,
    EVAL_BENCHMARK_DATASET_SUBSET,
    EVAL_LLM_MAX_TOKENS,
    EVAL_LLM_TEMPERATURE,
    EVAL_MAX_QUESTIONS,
    EVAL_OUTPUT_DIR,
)


def _generate_answer(query, context, llm_model, llm_tokenizer):
    """Generate an answer for a query given retrieved context."""
    system_prompt = (
        "You are a helpful assistant. Answer the question based on the provided context. "
        "If the context doesn't contain enough information, say so."
    )
    user_message = f"Context:\n{context}\n\nQuestion: {query}"
    conversation = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message},
    ]
    prompt = llm_tokenizer.apply_chat_template(
        conversation, add_generation_prompt=True, tokenize=False
    )
    sampler = make_sampler(temp=EVAL_LLM_TEMPERATURE)
    return mlx_generate(
        llm_model,
        llm_tokenizer,
        prompt=prompt,
        max_tokens=EVAL_LLM_MAX_TOKENS,
        sampler=sampler,
        verbose=False,
    )


# ---------------------------------------------------------------------------
# amnesty_qa helpers
# ---------------------------------------------------------------------------

def _load_amnesty_qa(max_questions=None):
    """Load amnesty_qa dataset. Returns list of dicts with question, contexts, ground_truth."""
    ds = load_dataset(EVAL_BENCHMARK_DATASET, EVAL_BENCHMARK_DATASET_SUBSET, split="eval")
    samples = []
    for row in ds:
        samples.append({
            "question": row["user_input"],
            "contexts": row["retrieved_contexts"],
            "ground_truth": row["reference"],
        })
        if max_questions and len(samples) >= max_questions:
            break
    return samples


# ---------------------------------------------------------------------------
# scifact helpers
# ---------------------------------------------------------------------------

def _load_scifact(max_questions=None):
    """Load scifact queries + corpus from mteb/scifact JSONL files."""
    qpath = hf_hub_download("mteb/scifact", "queries.jsonl", repo_type="dataset")
    cpath = hf_hub_download("mteb/scifact", "corpus.jsonl", repo_type="dataset")

    queries = []
    with open(qpath) as f:
        for line in f:
            row = json.loads(line)
            queries.append({
                "query_id": str(row["_id"]),
                "question": row["text"],
                "ground_truth": row["text"],
            })
            if max_questions and len(queries) >= max_questions:
                break

    corpus = {}
    with open(cpath) as f:
        for line in f:
            row = json.loads(line)
            title = row.get("title", "")
            text = row.get("text", "")
            corpus[str(row["_id"])] = f"{title}. {text}" if title else text

    return queries, corpus


def _index_scifact_corpus(corpus, retriever, embedding_model):
    """Index the scifact corpus into the retriever using existing Indexer methods."""
    from RAG_Framework.components.indexer import Indexer

    texts = list(corpus.values())
    doc_ids = list(corpus.keys())
    metadata = [{"document_name": f"scifact_{did}", "page": 0, "chunk_idx": i}
                 for i, did in enumerate(doc_ids)]

    indexer = Indexer()
    print(f"Indexing {len(texts)} scifact passages...")
    embeddings = indexer.get_embeddings(texts, model=embedding_model)
    multi_vector_index = indexer.build_multi_vector_index(embeddings, metadata, texts)
    bm25 = indexer.build_bm25_index(texts)
    faiss_index = indexer.build_faiss_index(multi_vector_index)
    metadata_index = indexer.build_metadata_index(metadata)

    retriever.update_indices(multi_vector_index, bm25, metadata_index, faiss_index)
    print("Scifact corpus indexed.")


# ---------------------------------------------------------------------------
# Metric scoring (calls collections metrics directly via batch_score)
# ---------------------------------------------------------------------------

def _run_metrics(samples, ragas_llm, ragas_embeddings, has_reference):
    """Score all samples with RAGAS collections metrics.

    Each metric's batch_score() expects a list of dicts whose keys match the
    metric's ascore() parameter names.
    """
    metric_instances = {
        "faithfulness": Faithfulness(llm=ragas_llm),
        "answer_relevancy": AnswerRelevancy(llm=ragas_llm, embeddings=ragas_embeddings),
    }
    if has_reference:
        metric_instances.update({
            "context_recall": ContextRecall(llm=ragas_llm),
            "context_precision": ContextPrecisionWithReference(llm=ragas_llm),
            "factual_correctness": FactualCorrectness(llm=ragas_llm),
        })

    # Prepare input dicts for each metric (different kwargs per metric)
    inputs_map = {
        "faithfulness": [
            {"user_input": s["user_input"], "response": s["response"],
             "retrieved_contexts": s["retrieved_contexts"]}
            for s in samples
        ],
        "answer_relevancy": [
            {"user_input": s["user_input"], "response": s["response"]}
            for s in samples
        ],
    }
    if has_reference:
        inputs_map["context_recall"] = [
            {"user_input": s["user_input"], "retrieved_contexts": s["retrieved_contexts"],
             "reference": s["reference"]}
            for s in samples
        ]
        inputs_map["context_precision"] = [
            {"user_input": s["user_input"], "reference": s["reference"],
             "retrieved_contexts": s["retrieved_contexts"]}
            for s in samples
        ]
        inputs_map["factual_correctness"] = [
            {"response": s["response"], "reference": s["reference"]}
            for s in samples
        ]

    # Run each metric
    all_scores = {name: [] for name in metric_instances}
    for name, metric in metric_instances.items():
        print(f"  Scoring {name}...")
        try:
            results = metric.batch_score(inputs_map[name])
            all_scores[name] = [r.value for r in results]
        except Exception as e:
            print(f"    WARNING: {name} failed: {e}")
            all_scores[name] = [float("nan")] * len(samples)

    return all_scores


# ---------------------------------------------------------------------------
# Main benchmark entry point
# ---------------------------------------------------------------------------

def run_benchmark_evaluation(
    dataset_name,
    llm_model,
    llm_tokenizer,
    retriever,
    ragas_llm,
    ragas_embeddings,
    max_questions=None,
):
    """Run benchmark evaluation with a standard dataset."""
    if max_questions is None:
        max_questions = EVAL_MAX_QUESTIONS

    print(f"\n{'='*60}")
    print(f"Benchmark Evaluation — dataset: {dataset_name}")
    print(f"{'='*60}\n")

    # ---- Load dataset ----
    if dataset_name == "amnesty_qa":
        data = _load_amnesty_qa(max_questions)
        has_contexts = True
    elif dataset_name == "scifact":
        queries, corpus = _load_scifact(max_questions)
        _index_scifact_corpus(corpus, retriever, retriever.embedding_model)
        data = queries
        has_contexts = False
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}. Use 'amnesty_qa' or 'scifact'.")

    n = len(data)
    print(f"Loaded {n} samples.\n")

    # ---- Generate answers ----
    samples = []
    for i, item in enumerate(data):
        question = item["question"]
        ground_truth = item.get("ground_truth", "")
        print(f"[{i+1}/{n}] {question[:80]}...")

        if has_contexts:
            contexts = item["contexts"]
            context_str = "\n---\n".join(contexts)
        else:
            context_str, contexts = retriever.combined_retrieval_with_chunks(question)

        t0 = time.time()
        response = _generate_answer(question, context_str, llm_model, llm_tokenizer)
        gen_time = time.time() - t0
        print(f"  Generated in {gen_time:.1f}s")

        samples.append({
            "user_input": question,
            "response": response,
            "retrieved_contexts": contexts,
            "reference": ground_truth,
        })

    # ---- Save generated answers (checkpoint) ----
    os.makedirs(EVAL_OUTPUT_DIR, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    checkpoint_path = os.path.join(EVAL_OUTPUT_DIR, f"_checkpoint_{dataset_name}_{ts}.json")
    with open(checkpoint_path, "w") as f:
        json.dump(samples, f, indent=2, default=str)
    print(f"\nAnswers checkpointed to {checkpoint_path}")

    # ---- RAGAS evaluation ----
    print("Running RAGAS evaluation...")
    has_reference = any(s.get("reference") for s in samples)
    all_scores = _run_metrics(samples, ragas_llm, ragas_embeddings, has_reference)

    # ---- Merge scores per sample ----
    per_sample_results = []
    for i, s in enumerate(samples):
        scores = {}
        for metric_name, score_list in all_scores.items():
            val = score_list[i]
            scores[metric_name] = val if not (isinstance(val, float) and math.isnan(val)) else None
        per_sample_results.append({**s, "scores": scores})

    # ---- Aggregate ----
    aggregate = {}
    for metric_name, score_list in all_scores.items():
        valid = [v for v in score_list if isinstance(v, (int, float)) and not math.isnan(v)]
        if valid:
            aggregate[metric_name] = sum(valid) / len(valid)

    # ---- Output ----
    output = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "mode": "benchmark",
            "dataset": dataset_name,
            "num_samples": n,
        },
        "aggregate_scores": aggregate,
        "per_sample_results": per_sample_results,
    }

    out_path = os.path.join(EVAL_OUTPUT_DIR, f"benchmark_{dataset_name}_{ts}.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)

    # Clean up checkpoint
    os.remove(checkpoint_path)

    print(f"\n{'='*60}")
    print(f"Aggregate Scores ({dataset_name}, {n} samples):")
    for metric, score in aggregate.items():
        print(f"  {metric}: {score:.4f}")
    print(f"\nResults saved to: {out_path}")
    print(f"{'='*60}\n")

    return output
