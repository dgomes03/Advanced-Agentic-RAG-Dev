from typing import Dict, Any
from mlx_lm import generate

from RAG_Framework.agents.planner import ReasoningGoal, ReasoningPlan, parse_json_safely


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
        system_prompt = """You are an information completeness evaluator. Assess if the retrieved context adequately addresses the goal.

            Output ONLY valid JSON:
            {
                "is_complete": true/false,
                "confidence": 0.0-1.0,
                "missing_aspects": ["aspect1", "aspect2"],
                "reasoning": "brief explanation"
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
        
        response = generate(
            llm_model,
            llm_tokenizer,
            prompt=prompt,
            max_tokens=300,
            verbose=False
        )
        
        eval_result = parse_json_safely(response)

        if eval_result and "is_complete" in eval_result:
            return eval_result
        else:
            print(f"Evaluation parsing failed: Could not parse JSON from LLM response")
            # Fallback evaluation
            return {
                "is_complete": len(retrieved_context) > 100 if retrieved_context else False,
                "confidence": 0.5,
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

        system_prompt = """You are a final completeness evaluator. Assess if we have enough information to answer the original query comprehensively.

            Output ONLY valid JSON:
            {
                "can_answer": true/false,
                "overall_confidence": 0.0-1.0,
                "coverage_assessment": "brief assessment",
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
        
        response = generate(
            llm_model,
            llm_tokenizer,
            prompt=prompt,
            max_tokens=300,
            verbose=False
        )
        
        eval_result = parse_json_safely(response)

        if eval_result and "can_answer" in eval_result:
            return eval_result
        else:
            print(f"Overall evaluation parsing failed: Could not parse JSON from LLM response")
            return {
                "can_answer": plan.get_completion_rate() > 0.6,
                "overall_confidence": plan.get_completion_rate(),
                "coverage_assessment": "Fallback assessment",
                "needs_more_search": plan.get_completion_rate() < 0.6
            }
