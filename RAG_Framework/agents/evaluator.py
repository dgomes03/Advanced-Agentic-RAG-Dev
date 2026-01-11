import json
from typing import Dict, Any
from mlx_lm import generate

from RAG_Framework.agents.planner import ReasoningGoal, ReasoningPlan


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
        
        try:
            # Clean and parse
            response = response.strip()
            if '```json' in response:
                response = response.split('```json')[1].split('```')[0].strip()
            elif '```' in response:
                response = response.split('```')[1].split('```')[0].strip()
            
            eval_result = json.loads(response)
            return eval_result
        except Exception as e:
            print(f"Evaluation parsing failed: {e}")
            # Fallback evaluation
            return {
                "is_complete": len(retrieved_context) > 100,
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
                all_info.extend(goal.retrieved_info)
        
        system_prompt = """You are a final completeness evaluator. Assess if we have enough information to answer the original query comprehensively.

            Output ONLY valid JSON:
            {
                "can_answer": true/false,
                "overall_confidence": 0.0-1.0,
                "coverage_assessment": "brief assessment",
                "needs_more_search": true/false
            }"""
        
        # adicionar KV-Cache a isto !!!!!!!!!
        combined_info = "\n".join(all_info[:8000])  # Limit context
        
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
        
        try:
            response = response.strip()
            if '```json' in response:
                response = response.split('```json')[1].split('```')[0].strip()
            elif '```' in response:
                response = response.split('```')[1].split('```')[0].strip()
            
            eval_result = json.loads(response)
            return eval_result
        except Exception as e:
            print(f"⚠️  Overall evaluation parsing failed: {e}")
            return {
                "can_answer": plan.get_completion_rate() > 0.6,
                "overall_confidence": plan.get_completion_rate(),
                "coverage_assessment": "Fallback assessment",
                "needs_more_search": plan.get_completion_rate() < 0.6
            }
