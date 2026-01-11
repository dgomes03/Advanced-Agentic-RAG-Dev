import json
from dataclasses import dataclass, field
from typing import List, Dict, Optional
from enum import Enum
from mlx_lm import generate


class ReasoningState(Enum):
    """States in the reasoning process"""
    INITIAL_QUERY = "initial_query"
    PLANNING = "planning"
    SEARCHING = "searching"
    EVALUATING = "evaluating"
    REPLANNING = "replanning"
    ANSWERING = "answering"
    COMPLETE = "complete"

@dataclass
class ReasoningGoal:
    """Represents a sub-goal in multi-step reasoning"""
    description: str
    priority: int
    status: str = "pending"  # pending, in_progress, completed, failed
    retrieved_info: List[str] = field(default_factory=list)
    confidence: float = 0.0
    
@dataclass
class ReasoningPlan:
    """Complete reasoning plan with multiple goals"""
    main_query: str
    goals: List[ReasoningGoal] = field(default_factory=list)
    current_step: int = 0
    total_confidence: float = 0.0
    
    def add_goal(self, goal: ReasoningGoal):
        self.goals.append(goal)
    
    def get_next_goal(self) -> Optional[ReasoningGoal]:
        """Get next pending or in_progress goal"""
        for goal in sorted(self.goals, key=lambda g: g.priority):
            if goal.status in ["pending", "in_progress"]:
                return goal
        return None
    
    def is_complete(self) -> bool:
        """Check if all goals are completed"""
        return all(g.status == "completed" for g in self.goals)
    
    def get_completion_rate(self) -> float:
        """Get percentage of completed goals"""
        if not self.goals:
            return 0.0
        completed = sum(1 for g in self.goals if g.status == "completed")
        return completed / len(self.goals)


class AgenticPlanner:
    """Handles planning and replanning of reasoning steps"""
    
    @staticmethod
    def create_initial_plan(query: str, llm_model, llm_tokenizer) -> ReasoningPlan:
        """Create initial reasoning plan by decomposing the query"""

        system_prompt = """You are a query decomposition expert. Break down queries into sub-goals.
            For each sub-goal:
            1. Describe what information is needed
            2. Assign priority (1=highest, 5=lowest)
            3. Keep descriptions clear and searchable

            Output ONLY valid JSON in this exact format:
            {
                "goals": [
                    {"description": "goal description", "priority": 1},
                    {"description": "another goal", "priority": 2}
                ]
            }"""
        
        user_prompt = f"""Query: {query}

            Decompose this into 2-3 searchable sub-goals. Output JSON only."""
        
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
            max_tokens=500,
            verbose=False
        )
        
        # Parse the response
        plan = ReasoningPlan(main_query=query)
        try:
            # Clean response and extract JSON
            response = response.strip()
            if '```json' in response:
                response = response.split('```json')[1].split('```')[0].strip()
            elif '```' in response:
                response = response.split('```')[1].split('```')[0].strip()
            
            plan_data = json.loads(response)
            for goal_data in plan_data.get("goals", []):
                goal = ReasoningGoal(
                    description=goal_data["description"],
                    priority=goal_data.get("priority", 3)
                )
                plan.add_goal(goal)
            
            print(f"\nCreated plan with {len(plan.goals)} goals:")
            for i, g in enumerate(plan.goals, 1):
                print(f"  {i}. [{g.priority}] {g.description}")
                
        except Exception as e:
            print(f"Plan parsing failed: {e}")
            # Fallback: treat entire query as single goal
            plan.add_goal(ReasoningGoal(
                description=query,
                priority=1
            ))
        
        return plan
    
    @staticmethod
    def replan(plan: ReasoningPlan, evaluation: Dict, llm_model, llm_tokenizer) -> ReasoningPlan:

        """Replan based on evaluation results"""

        system_prompt = """You are a reasoning coordinator. Based on the evaluation, decide if we need additional search goals.

            Output ONLY valid JSON:
            {
                "needs_replanning": true/false,
                "new_goals": [
                    {"description": "new goal if needed", "priority": 1}
                ],
                "reasoning": "brief explanation"
            }"""
        
        current_state = {
            "completed_goals": [g.description for g in plan.goals if g.status == "completed"],
            "pending_goals": [g.description for g in plan.goals if g.status != "completed"],
            "evaluation": evaluation
        }
        
        user_prompt = f"""Current state: {json.dumps(current_state, indent=2)}

            Should we add more search goals? Output JSON only."""
        
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
            max_tokens=400,
            verbose=False
        )
        
        try:
            # Clean and parse response
            response = response.strip()
            if '```json' in response:
                response = response.split('```json')[1].split('```')[0].strip()
            elif '```' in response:
                response = response.split('```')[1].split('```')[0].strip()
            
            replan_data = json.loads(response)
            
            if replan_data.get("needs_replanning", False):
                print(f"\nReplanning: {replan_data.get('reasoning', 'Adding new goals')}")
                for goal_data in replan_data.get("new_goals", []):
                    new_goal = ReasoningGoal(
                        description=goal_data["description"],
                        priority=goal_data.get("priority", 3)
                    )
                    plan.add_goal(new_goal)
                    print(f"  + Added: {new_goal.description}")
        except Exception as e:
            print(f"Replan parsing failed: {e}")
        
        return plan
