"""Reference-free evaluation: runs queries through the RAG pipeline and evaluates
with Faithfulness and AnswerRelevancy (no ground truth needed)."""

import json
import math
import os
import time
from datetime import datetime

from mlx_lm import generate as mlx_generate
from mlx_lm.sample_utils import make_sampler
from ragas.metrics.collections import AnswerRelevancy, Faithfulness

from RAG_Framework.core.config import (
    EVAL_LLM_MAX_TOKENS,
    EVAL_LLM_TEMPERATURE,
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


def run_reference_free_evaluation(queries, retriever, llm_model, llm_tokenizer,
                                  ragas_llm, ragas_embeddings):
    """Run reference-free evaluation on a list of queries using the live RAG pipeline."""
    print(f"\n{'='*60}")
    print(f"Reference-Free Evaluation — {len(queries)} queries")
    print(f"{'='*60}\n")

    samples = []
    for i, query in enumerate(queries):
        print(f"[{i+1}/{len(queries)}] Processing: {query[:80]}...")

        context_string, chunks = retriever.combined_retrieval_with_chunks(query)

        t0 = time.time()
        response = _generate_answer(query, context_string, llm_model, llm_tokenizer)
        gen_time = time.time() - t0
        print(f"  Answer generated in {gen_time:.1f}s ({len(response)} chars)")

        samples.append({
            "user_input": query,
            "response": response,
            "retrieved_contexts": chunks,
        })

    # ---- Save checkpoint ----
    os.makedirs(EVAL_OUTPUT_DIR, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    checkpoint_path = os.path.join(EVAL_OUTPUT_DIR, f"_checkpoint_reffree_{ts}.json")
    with open(checkpoint_path, "w") as f:
        json.dump(samples, f, indent=2, default=str)

    # ---- Score with RAGAS collections metrics ----
    print("\nRunning RAGAS evaluation...")

    metrics = {
        "faithfulness": Faithfulness(llm=ragas_llm),
        "answer_relevancy": AnswerRelevancy(llm=ragas_llm, embeddings=ragas_embeddings),
    }

    faith_inputs = [
        {"user_input": s["user_input"], "response": s["response"],
         "retrieved_contexts": s["retrieved_contexts"]}
        for s in samples
    ]
    relevancy_inputs = [
        {"user_input": s["user_input"], "response": s["response"]}
        for s in samples
    ]

    all_scores = {}
    for name, (metric, inputs) in {
        "faithfulness": (metrics["faithfulness"], faith_inputs),
        "answer_relevancy": (metrics["answer_relevancy"], relevancy_inputs),
    }.items():
        print(f"  Scoring {name}...")
        try:
            results = metric.batch_score(inputs)
            all_scores[name] = [r.value for r in results]
        except Exception as e:
            print(f"    WARNING: {name} failed: {e}")
            all_scores[name] = [float("nan")] * len(samples)

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
            "mode": "reference-free",
            "num_samples": len(queries),
        },
        "aggregate_scores": aggregate,
        "per_sample_results": per_sample_results,
    }

    out_path = os.path.join(EVAL_OUTPUT_DIR, f"reference_free_{ts}.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)

    os.remove(checkpoint_path)

    print(f"\n{'='*60}")
    print("Aggregate Scores:")
    for metric, score in aggregate.items():
        print(f"  {metric}: {score:.4f}")
    print(f"\nResults saved to: {out_path}")
    print(f"{'='*60}\n")

    return output
