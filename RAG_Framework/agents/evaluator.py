from typing import Dict, Any, List

from RAG_Framework.agents.planner import ReasoningGoal, ReasoningPlan, parse_json_safely, generate_fixed


class AgenticEvaluator:
    """Evaluates completeness and quality of retrieved information"""

    @staticmethod
    def evaluate_goal_completion(
        goal: ReasoningGoal,
        retrieved_context: str,
        llm_model,
        llm_tokenizer
    ) -> Dict[str, Any]:
        """Evaluate if a goal has been satisfactorily completed"""
        system_prompt = """Evaluate if retrieved context satisfies the search goal.

SCORING (0.0-1.0):
- Direct answer present: +0.4
- Complete coverage: +0.3
- Specific facts/data: +0.2
- Authoritative source: +0.1

OUTPUT (JSON only):
{
    "is_complete": true/false,
    "confidence": 0.0-1.0,
    "missing_aspects": ["aspect1", "aspect2"],
    "reasoning": "brief assessment"
}"""

        user_prompt = f"""Goal: {goal.description}

Retrieved context:
{retrieved_context[:1000]}...

Is this sufficient? Output JSON only."""

        conversation = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        prompt = llm_tokenizer.apply_chat_template(
            conversation,
            add_generation_prompt=True,
            tokenize=False
        )

        print(f"\n[EVALUATOR] Formatted prompt:\n{prompt}")

        response = generate_fixed(
            llm_model,
            llm_tokenizer,
            prompt=prompt,
            max_tokens=600,
            verbose=True
        )

        print(f"\n[EVALUATOR] Raw LLM response:\n{repr(response)}")

        eval_result = parse_json_safely(response)
        print(f"[EVALUATOR] Parsed JSON: {eval_result}")

        if eval_result and "is_complete" in eval_result:
            return eval_result
        else:
            print(f"[EVALUATOR] Evaluation parsing failed: Could not parse JSON from LLM response")
            # Fallback evaluation
            return {
                "is_complete": len(retrieved_context) > 100 if retrieved_context else False,
                "confidence": 0.5,
                "missing_aspects": [],
                "reasoning": "Fallback evaluation"
            }

    @staticmethod
    def evaluate_goal_completion_with_gain(
        goal: ReasoningGoal,
        retrieved_context: str,
        previous_contexts: List[str],
        llm_model,
        llm_tokenizer
    ) -> Dict[str, Any]:
        """Evaluate goal completion with information-gain and quality flags.

        Adds novelty scoring and flags for sparse/contradictory results
        compared to the basic evaluate_goal_completion().
        """
        system_prompt = """Evaluate retrieved information quality and novelty.

SCORING (0.0-1.0):
- Direct answer: +0.4, Complete coverage: +0.3, Specific facts: +0.2, Source quality: +0.1

NOVELTY (information_gain 0.0-1.0):
- 1.0 = entirely new facts not in previous findings
- 0.0 = complete duplicate of what we already have

FLAGS:
- sparse_results: true if retrieval returned very little useful content
- contradictory_info: true if new findings contradict previous ones

OUTPUT (JSON only):
{
    "is_complete": true,
    "confidence": 0.8,
    "information_gain": 0.7,
    "sparse_results": false,
    "contradictory_info": false,
    "missing_aspects": [],
    "reasoning": "brief"
}"""

        # Build summary of previous findings (last 3, 500 chars each)
        prev_summary = ""
        if previous_contexts:
            for i, ctx in enumerate(previous_contexts[-3:]):
                prev_summary += f"\n--- Previous finding {i+1} ---\n{ctx[:500]}\n"

        user_prompt = f"""Goal: {goal.description}

Retrieved context:
{retrieved_context[:1000]}

Previous findings:{prev_summary if prev_summary else " (none)"}

Evaluate quality and novelty. Output JSON only."""

        conversation = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        prompt = llm_tokenizer.apply_chat_template(
            conversation,
            add_generation_prompt=True,
            tokenize=False
        )

        print(f"\n[EVAL+GAIN] Formatted prompt:\n{prompt}")

        response = generate_fixed(
            llm_model,
            llm_tokenizer,
            prompt=prompt,
            max_tokens=600,
            verbose=True
        )

        print(f"\n[EVAL+GAIN] Raw LLM response:\n{repr(response)}")

        eval_result = parse_json_safely(response)
        print(f"[EVAL+GAIN] Parsed JSON: {eval_result}")

        if eval_result and "is_complete" in eval_result:
            # Ensure all expected fields have defaults
            eval_result.setdefault("information_gain", 0.5)
            eval_result.setdefault("sparse_results", False)
            eval_result.setdefault("contradictory_info", False)
            eval_result.setdefault("missing_aspects", [])
            eval_result.setdefault("reasoning", "")
            return eval_result
        else:
            print(f"[EVAL+GAIN] Parsing failed: Could not parse JSON from LLM response")
            return {
                "is_complete": len(retrieved_context) > 100 if retrieved_context else False,
                "confidence": 0.5,
                "information_gain": 0.5,
                "sparse_results": len(retrieved_context) < 50 if retrieved_context else True,
                "contradictory_info": False,
                "missing_aspects": [],
                "reasoning": "Fallback evaluation"
            }

    @staticmethod
    def evaluate_overall_completeness(
        plan: ReasoningPlan,
        llm_model,
        llm_tokenizer
    ) -> Dict[str, Any]:
        """Evaluate overall completeness of the reasoning process"""
        all_info = []
        for goal in plan.goals:
            if goal.retrieved_info:
                for info in goal.retrieved_info:
                    # Ensure each item is a string
                    if isinstance(info, str):
                        all_info.append(info)
                    elif isinstance(info, tuple):
                        # Handle tuple case (e.g., from available_tools returning tuple)
                        all_info.append(str(info[0]) if info else "")
                    else:
                        all_info.append(str(info))

        system_prompt = """Final assessment: Can we comprehensively answer the original query?

THRESHOLDS:
- Confident answer: confidence >= 0.7, all key aspects covered
- Partial answer: confidence 0.4-0.7, main aspects covered
- Cannot answer: confidence < 0.4, critical gaps remain

OUTPUT (JSON only):
{
    "can_answer": true/false,
    "overall_confidence": 0.0-1.0,
    "coverage_assessment": "what we know vs what's missing",
    "needs_more_search": true/false
}"""

        # Limit to ~8000 characters total, not list items
        combined_info = "\n".join(all_info)
        if len(combined_info) > 8000:
            combined_info = combined_info[:8000] + "..."

        user_prompt = f"""Original query: {plan.main_query}

            Completed goals: {plan.get_completion_rate()*100:.0f}%

            Retrieved information:
            {combined_info}

            Can we answer comprehensively? Output JSON only."""

        conversation = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        prompt = llm_tokenizer.apply_chat_template(
            conversation,
            add_generation_prompt=True,
            tokenize=False
        )

        print(f"\n[OVERALL_EVAL] Formatted prompt:\n{prompt}")

        response = generate_fixed(
            llm_model,
            llm_tokenizer,
            prompt=prompt,
            max_tokens=600,
            verbose=True
        )

        print(f"\n[OVERALL_EVAL] Raw LLM response:\n{repr(response)}")

        eval_result = parse_json_safely(response)
        print(f"[OVERALL_EVAL] Parsed JSON: {eval_result}")

        if eval_result and "can_answer" in eval_result:
            return eval_result
        else:
            print(f"[OVERALL_EVAL] Parsing failed: Could not parse JSON from LLM response")
            return {
                "can_answer": plan.get_completion_rate() > 0.6,
                "overall_confidence": plan.get_completion_rate(),
                "coverage_assessment": "Fallback assessment",
                "needs_more_search": plan.get_completion_rate() < 0.6
            }
