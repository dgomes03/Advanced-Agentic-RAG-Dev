import io
import json
import re
import sys
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Dict, Optional, Any
from enum import Enum
from mlx_lm import generate


def generate_fixed(model, tokenizer, prompt, max_tokens, verbose=True):
    """Wrapper around mlx_lm.generate() that fixes BPE decoding artifacts.

    Captures the verbose stdout output, applies the GPT-2 byte decoder to
    strip Ġ/Ċ artifacts, re-prints clean text, and returns the fixed string.
    """
    from RAG_Framework.components.generators.standard import Generator

    old_stdout = sys.stdout
    sys.stdout = _cap = io.StringIO()
    try:
        result = generate(model, tokenizer, prompt=prompt, max_tokens=max_tokens, verbose=verbose)
    finally:
        sys.stdout = old_stdout

    if verbose:
        print(Generator._fix_bpe_artifacts(_cap.getvalue()), end='')

    return Generator._fix_bpe_artifacts(result)


def parse_json_safely(response: str) -> Optional[Dict[str, Any]]:
    """
    Robustly parse JSON from LLM response, handling common issues:
    - BPE decoding artifacts (Ġ, Ċ, etc.) from broken tokenizer.decode()
    - Control characters in strings
    - Markdown code blocks
    - Text before/after JSON
    """
    if not response:
        return None

    # Fix BPE artifacts before any parsing (e.g. Ministral tokenizer Ġ→space, Ċ→newline)
    try:
        from RAG_Framework.components.generators.standard import Generator
        response = Generator._fix_bpe_artifacts(response)
    except ImportError:
        pass

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

        current_datetime = datetime.now().strftime("%A, %B %d, %Y at %H:%M")
        system_prompt = f"""Current date and time: {current_datetime}

You are a research planning assistant. Decompose queries into web search queries.

RULES:
- Create 2-4 focused sub-goals (no more)
- Each description = a SHORT search query: 3-6 keywords only, NO full sentences or questions
- Write like a Google search, not a research question (e.g. "ColBERT reranking benchmark 2025" not "What are the benchmark results for ColBERT in reranking tasks?")
- Priority: 1=essential, 2=helpful, 3=supplementary
- Order from foundational to dependent
- Strategy: "hybrid" (default), "bm25_only" (keyword-heavy queries), "dense_only" (semantic queries)

OUTPUT (JSON only):
{{
    "goals": [
        {{"description": "short keyword query here", "priority": 1, "strategy": "hybrid"}},
        {{"description": "another keyword query", "priority": 2, "strategy": "hybrid"}}
    ]
}}"""

        user_prompt = f"""Query: {query}

Decompose this into 2-3 short keyword search queries. Output JSON only."""

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

        response = generate_fixed(
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

        current_datetime = datetime.now().strftime("%A, %B %d, %Y at %H:%M")
        is_sparse = evaluation.get("sparse_results", False)
        sparse_directive = (
            "\n⚠️  sparse_results=True detected. You MUST choose REFORMULATE.\n"
            "Rewrite the failed_query from the evaluation with 2-4 DIFFERENT shorter keywords.\n"
            "Use a completely different angle — synonyms, broader/narrower terms, or related concepts.\n"
            "Do NOT choose ADD_GOALS or STOP when results were sparse.\n"
        ) if is_sparse else ""

        system_prompt = f"""Current date and time: {current_datetime}

You are a research coordinator. Decide next action.
{sparse_directive}
OPTIONS:
1. REFORMULATE: Rewrite a failed/sparse query with 2-4 shorter or different keywords (USE THIS when sparse_results=True)
2. ADD_GOALS: Add new sub-questions only if there are genuine unanswered information gaps
3. SWITCH_STRATEGY: Change search strategy (bm25/dense) for existing pending goals
4. STOP: Sufficient information already gathered to answer the query

DECISION CRITERIA:
- sparse_results=True → MUST REFORMULATE with different keywords (not ADD_GOALS, not STOP)
- contradictory_info=True → consider SWITCH_STRATEGY or REFORMULATE
- Confidence < 0.7 and unanswered aspects → ADD_GOALS for specific missing pieces
- All key aspects covered with confidence ≥ 0.7 → STOP

CRITICAL: All goal descriptions must be SHORT keyword search queries (3-6 words, no full sentences).
Write like a Google search: "ColBERT E5 benchmark 2025" NOT "What are the performance benchmarks for ColBERT and E5 models?"

OUTPUT (JSON only):
{{
    "action": "REFORMULATE",
    "new_goals": [{{"description": "shorter different keyword query", "priority": 1, "strategy": "hybrid"}}],
    "reasoning": "one sentence"
}}"""

        current_state = {
            "original_query": plan.main_query,
            "completed_goals": [
                {"description": g.description, "confidence": g.confidence, "strategy": g.strategy}
                for g in plan.goals if g.status == "completed"
            ],
            "failed_goals": [g.description for g in plan.goals if g.status == "failed"],
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

        response = generate_fixed(
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
