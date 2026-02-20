from mlx_lm.sample_utils import make_sampler

from RAG_Framework.agents.planner import parse_json_safely
from RAG_Framework.core.config import MAX_EVAL_CHARS_PER_SOURCE

_EVALUATOR_SAMPLER = make_sampler(temp=0.3)


class AgenticEvaluator:

    @staticmethod
    def batch_evaluate_goals(goals, retrieved_contexts, all_previous_contexts,
                              llm_model, llm_tokenizer,
                              stream_callback=None, goal_indices=None):
        """Evaluate all goals in parallel using BatchGenerator.

        Emits reasoning_evaluation stream events immediately as each goal
        finishes. Returns a list of evaluation dicts in goals order.
        """
        from RAG_Framework.agents.batch_generator import run_batch_generate

        if goal_indices is None:
            goal_indices = list(range(len(goals)))

        system_prompt = (
            "Evaluate if the retrieved context satisfies the research goal.\n\n"
            "OUTPUT (JSON only, no explanation outside the JSON):\n"
            "{\n"
            '    "is_complete": true,\n'
            '    "confidence": 0.8,\n'
            '    "information_gain": 0.7,\n'
            '    "sparse_results": false,\n'
            '    "contradictory_info": false,\n'
            '    "missing_aspects": [],\n'
            '    "reasoning": "one sentence max"\n'
            "}"
        )

        prompts = []
        for goal, retrieved_context in zip(goals, retrieved_contexts):
            # Truncate per source so the evaluator sees a representative sample
            # from every source rather than the full verbatim content.
            # Synthesis uses goal.retrieved_info which retains the full content.
            eval_context = "\n---\n".join(
                s[:MAX_EVAL_CHARS_PER_SOURCE]
                for s in retrieved_context.split("\n---\n")
            )

            prev_summary = ""
            if all_previous_contexts:
                for i, ctx in enumerate(all_previous_contexts[-3:]):
                    prev_summary += f"\n--- Previous finding {i+1} ---\n{ctx[:500]}\n"

            user_prompt = (
                f"Goal: {goal.description}\n\n"
                f"Retrieved context:\n{eval_context}\n\n"
                f"Previous findings:{prev_summary if prev_summary else ' (none)'}\n\n"
                "Evaluate quality and novelty. Output JSON only."
            )

            conversation = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
            prompts.append(llm_tokenizer.apply_chat_template(
                conversation, add_generation_prompt=True, tokenize=False
            ))

        evaluations = [None] * len(goals)

        def on_eval_complete(idx, response):
            retrieved_context = retrieved_contexts[idx]
            print(f"\n[BATCH_EVAL] Goal {idx} done ({len(response)} chars): {repr(response)}")

            eval_result = parse_json_safely(response)
            if eval_result and "is_complete" in eval_result:
                eval_result.setdefault("information_gain", 0.5)
                eval_result.setdefault("sparse_results", False)
                eval_result.setdefault("contradictory_info", False)
                eval_result.setdefault("missing_aspects", [])
                eval_result.setdefault("reasoning", "")
            else:
                print(f"[BATCH_EVAL] Goal {idx}: parse failed, using fallback")
                eval_result = {
                    "is_complete": len(retrieved_context) > 100 if retrieved_context else False,
                    "confidence": 0.5,
                    "information_gain": 0.5,
                    "sparse_results": len(retrieved_context) < 50 if retrieved_context else True,
                    "contradictory_info": False,
                    "missing_aspects": [],
                    "reasoning": "Fallback evaluation"
                }

            evaluations[idx] = eval_result

            if stream_callback:
                stream_callback('reasoning_evaluation', {
                    'confidence': eval_result.get('confidence', 0.5),
                    'information_gain': eval_result.get('information_gain', 0.5),
                    'is_complete': eval_result.get('is_complete', False),
                    'sparse_results': eval_result.get('sparse_results', False),
                    'contradictory_info': eval_result.get('contradictory_info', False),
                    'reasoning': eval_result.get('reasoning', ''),
                    'goal_index': goal_indices[idx]
                })

        run_batch_generate(
            llm_model, llm_tokenizer, prompts,
            max_tokens=300, label="BATCH_EVAL",
            on_complete=on_eval_complete,
            sampler=_EVALUATOR_SAMPLER
        )

        return evaluations
