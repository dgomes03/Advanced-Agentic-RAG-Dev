import json
import re
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
from enum import Enum
from mlx_lm import generate


def parse_json_safely(response: str) -> Optional[Dict[str, Any]]:
    """
    Robustly parse JSON from LLM response, handling common issues:
    - Control characters in strings
    - Markdown code blocks
    - Text before/after JSON
    """
    if not response:
        return None

    response = response.strip()

    # Remove markdown code blocks
    if '```json' in response:
        response = response.split('```json')[1].split('```')[0].strip()
    elif '```' in response:
        parts = response.split('```')
        if len(parts) >= 2:
            response = parts[1].strip()

    # Try to extract JSON object using regex (handles text before/after)
    json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', response, re.DOTALL)
    if json_match:
        response = json_match.group()

    # Remove control characters that break JSON parsing (except valid whitespace)
    # Replace problematic control chars with spaces
    response = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', ' ', response)

    # Try parsing
    try:
        return json.loads(response)
    except json.JSONDecodeError:
        # Try fixing common issues: unescaped newlines in strings
        # Replace actual newlines inside strings with escaped versions
        try:
            # More aggressive cleanup: normalize whitespace
            response = ' '.join(response.split())
            return json.loads(response)
        except json.JSONDecodeError:
            return None


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
    strategy: str = "hybrid"  # "hybrid" | "bm25_only" | "dense_only"

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

        system_prompt = """You are a research planning assistant. Decompose queries into searchable sub-goals.

RULES:
- Create 2-4 focused sub-goals (no more)
- Each goal = one searchable question
- Priority: 1=essential, 2=helpful, 3=supplementary
- Order from foundational to dependent
- Strategy: "hybrid" (default), "bm25_only" (keyword-heavy queries), "dense_only" (semantic queries)

OUTPUT (JSON only):
{
    "goals": [
        {"description": "specific searchable question", "priority": 1, "strategy": "hybrid"},
        {"description": "another question", "priority": 2, "strategy": "hybrid"}
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

        print(f"\n[PLANNER] Formatted prompt:\n{prompt}")

        response = generate(
            llm_model,
            llm_tokenizer,
            prompt=prompt,
            max_tokens=500,
            verbose=True
        )

        print(f"\n[PLANNER] Raw LLM response:\n{repr(response)}")

        # Parse the response
        plan = ReasoningPlan(main_query=query)

        plan_data = parse_json_safely(response)
        print(f"[PLANNER] Parsed JSON: {plan_data}")

        if plan_data and "goals" in plan_data:
            for goal_data in plan_data.get("goals", []):
                if isinstance(goal_data, dict) and "description" in goal_data:
                    goal = ReasoningGoal(
                        description=goal_data["description"],
                        priority=goal_data.get("priority", 3),
                        strategy=goal_data.get("strategy", "hybrid")
                    )
                    plan.add_goal(goal)

            if plan.goals:
                print(f"\n[PLANNER] Created plan with {len(plan.goals)} goals:")
                for i, g in enumerate(plan.goals, 1):
                    print(f"  {i}. [P{g.priority}] ({g.strategy}) {g.description}")
            else:
                print(f"[PLANNER] Plan parsing failed: No valid goals found in response")
                plan.add_goal(ReasoningGoal(description=query, priority=1))
        else:
            print(f"[PLANNER] Plan parsing failed: Could not parse JSON from LLM response")
            # Fallback: treat entire query as single goal
            plan.add_goal(ReasoningGoal(description=query, priority=1))

        return plan

    @staticmethod
    def replan(plan: ReasoningPlan, evaluation: Dict, llm_model, llm_tokenizer) -> ReasoningPlan:
        """Replan based on evaluation results. Supports strategy switching, reformulation, and stopping."""

        system_prompt = """You are a research coordinator. Decide next action.

OPTIONS:
1. ADD_GOALS: New sub-questions to fill gaps
2. REFORMULATE: Rewrite a query (too vague or contradictory results)
3. SWITCH_STRATEGY: Change from semantic to keyword search or vice versa
4. STOP: Sufficient information gathered

DECISION CRITERIA:
- Are there unanswered aspects of the original query?
- Did searches reveal new information needs?
- Is confidence below 0.7 for key facts?
- Were results sparse or contradictory? (consider SWITCH_STRATEGY)
- Was the query too vague? (consider REFORMULATE)

OUTPUT (JSON only):
{
    "action": "ADD_GOALS",
    "new_goals": [{"description": "...", "priority": 1, "strategy": "hybrid"}],
    "reasoning": "one sentence"
}"""

        current_state = {
            "original_query": plan.main_query,
            "completed_goals": [
                {"description": g.description, "confidence": g.confidence, "strategy": g.strategy}
                for g in plan.goals if g.status == "completed"
            ],
            "pending_goals": [g.description for g in plan.goals if g.status == "pending"],
            "evaluation": evaluation
        }

        user_prompt = f"""Current state: {json.dumps(current_state, indent=2)}

What should we do next? Output JSON only."""

        conversation = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        prompt = llm_tokenizer.apply_chat_template(
            conversation,
            add_generation_prompt=True,
            tokenize=False
        )

        print(f"\n[REPLANNER] Formatted prompt:\n{prompt}")

        response = generate(
            llm_model,
            llm_tokenizer,
            prompt=prompt,
            max_tokens=400,
            verbose=True
        )

        print(f"\n[REPLANNER] Raw LLM response:\n{repr(response)}")

        replan_data = parse_json_safely(response)
        print(f"[REPLANNER] Parsed JSON: {replan_data}")

        if replan_data is None:
            print(f"[REPLANNER] Replan parsing failed: Could not parse JSON from LLM response")
            return plan

        action = replan_data.get("action", "STOP")
        reasoning = replan_data.get("reasoning", "")
        print(f"\n[REPLANNER] Action: {action} — {reasoning}")

        if action == "ADD_GOALS":
            for goal_data in replan_data.get("new_goals", []):
                if isinstance(goal_data, dict) and "description" in goal_data:
                    new_goal = ReasoningGoal(
                        description=goal_data["description"],
                        priority=goal_data.get("priority", 3),
                        strategy=goal_data.get("strategy", "hybrid")
                    )
                    plan.add_goal(new_goal)
                    print(f"  + Added goal: {new_goal.description} (strategy={new_goal.strategy})")

        elif action == "REFORMULATE":
            for goal_data in replan_data.get("new_goals", []):
                if isinstance(goal_data, dict) and "description" in goal_data:
                    new_goal = ReasoningGoal(
                        description=goal_data["description"],
                        priority=goal_data.get("priority", 1),
                        strategy=goal_data.get("strategy", "hybrid")
                    )
                    plan.add_goal(new_goal)
                    print(f"  + Reformulated goal: {new_goal.description}")

        elif action == "SWITCH_STRATEGY":
            # Switch remaining pending goals to bm25_only
            for goal in plan.goals:
                if goal.status == "pending":
                    goal.strategy = "bm25_only"
                    print(f"  ~ Switched strategy to bm25_only: {goal.description}")
            # Also add any new goals from the replan
            for goal_data in replan_data.get("new_goals", []):
                if isinstance(goal_data, dict) and "description" in goal_data:
                    new_goal = ReasoningGoal(
                        description=goal_data["description"],
                        priority=goal_data.get("priority", 2),
                        strategy="bm25_only"
                    )
                    plan.add_goal(new_goal)
                    print(f"  + Added bm25 goal: {new_goal.description}")

        elif action == "STOP":
            print("  Replanner decided to stop — sufficient information gathered.")

        return plan
