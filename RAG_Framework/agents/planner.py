import io
import json
import re
import sys
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Dict, Optional, Any
from mlx_lm import generate
from mlx_lm.sample_utils import make_sampler

from RAG_Framework.core.config import MAX_REASONING_GOALS
from RAG_Framework.core.BPE_decode import BPEDecoder

_PLANNER_SAMPLER = make_sampler(temp=0.1)


def generate_fixed(model, tokenizer, prompt, max_tokens, verbose=True, sampler=None):
    """Wrapper around mlx_lm.generate() that fixes BPE decoding artifacts.

    Captures the verbose stdout output, applies the GPT-2 byte decoder to
    strip Ġ/Ċ artifacts, re-prints clean text, and returns the fixed string.
    """
    kwargs = dict(prompt=prompt, max_tokens=max_tokens, verbose=verbose)
    if sampler is not None:
        kwargs["sampler"] = sampler

    old_stdout = sys.stdout
    sys.stdout = _cap = io.StringIO()
    try:
        result = generate(model, tokenizer, **kwargs)
    finally:
        sys.stdout = old_stdout

    if verbose:
        print(BPEDecoder.fix_bpe_artifacts(_cap.getvalue()), end='')

    return BPEDecoder.fix_bpe_artifacts(result)


def parse_json_safely(response: str) -> Optional[Dict[str, Any]]:
    """Robustly parse JSON from LLM response, handling common issues:
    - BPE decoding artifacts (Ġ, Ċ, etc.) from broken tokenizer.decode()
    - Control characters in strings
    - Markdown code blocks
    - Text before/after JSON
    """
    if not response:
        return None

    response = BPEDecoder.fix_bpe_artifacts(response)

    response = response.strip()

    if '```json' in response:
        response = response.split('```json')[1].split('```')[0].strip()
    elif '```' in response:
        parts = response.split('```')
        if len(parts) >= 2:
            response = parts[1].strip()

    json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', response, re.DOTALL)
    if json_match:
        response = json_match.group()

    response = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', ' ', response)

    try:
        return json.loads(response)
    except json.JSONDecodeError:
        try:
            response = ' '.join(response.split())
            return json.loads(response)
        except json.JSONDecodeError:
            return None


@dataclass
class ReasoningGoal:
    """Represents a sub-goal in multi-step reasoning"""
    description: str
    status: str = "pending"  # pending, in_progress, completed, failed, cancelled
    retrieved_info: List[str] = field(default_factory=list)
    confidence: float = 0.0
    replan_count: int = 0


@dataclass
class ReasoningPlan:
    """Complete reasoning plan with multiple goals"""
    main_query: str
    goals: List[ReasoningGoal] = field(default_factory=list)

    def add_goal(self, goal: ReasoningGoal):
        self.goals.append(goal)

    def is_complete(self) -> bool:
        return all(g.status in ("completed", "failed", "cancelled") for g in self.goals)

    def get_completion_rate(self) -> float:
        if not self.goals:
            return 0.0
        terminal = sum(1 for g in self.goals if g.status in ("completed", "failed", "cancelled"))
        return terminal / len(self.goals)


class AgenticPlanner:

    @staticmethod
    def create_initial_plan(query: str, llm_model, llm_tokenizer) -> ReasoningPlan:
        """Decompose query into a set of web search sub-goals."""
        current_datetime = datetime.now().strftime("%A, %B %d, %Y at %H:%M")
        system_prompt = f"""Current date and time: {current_datetime}

You are a research planning assistant. Decompose queries into web search queries.

RULES:
- Use the MINIMUM number of goals needed, MAXIMUM {MAX_REASONING_GOALS}
- Most queries need only 1-2 goals. Only use more if the query has clearly distinct sub-topics
- Each description = a SHORT search query: 3-6 keywords only, NO full sentences or questions
- Write like a Google search: "BM25 RAG replacement 2025" not "What replaces BM25 in RAG?"
- CRITICAL: Do NOT invent goals about technologies, methods, or subtopics not explicitly mentioned in the query. Only search for what the user literally asked.

EXAMPLE — Query: "Given that BM25 is obsolete for RAG, what should replace it?"
CORRECT (2 goals):
  1. "BM25 obsolete RAG 2025 alternatives"
  2. "modern retrieval models RAG BM25 replacement"
WRONG — do NOT add a third goal like these, they were not asked:
  ✗ "dense retrieval RAG vs sparse RAG comparison"
  ✗ "ColBERT SPLADE hybrid benchmarks"
  ✗ "hybrid retrieval RAG performance"
Stop at 2. Do not add comparison or analysis goals the user did not request.

OUTPUT (JSON only):
{{
    "goals": [
        {{"description": "short keyword query here"}},
        {{"description": "another keyword query"}}
    ]
}}"""

        user_prompt = f"""Query: {query}

Decompose this into the minimum number of keyword search queries needed (max {MAX_REASONING_GOALS}). Output JSON only."""

        conversation = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        prompt = llm_tokenizer.apply_chat_template(
            conversation, add_generation_prompt=True, tokenize=False
        )

        print(f"\n[PLANNER] Formatted prompt:\n{prompt}")

        response = generate_fixed(
            llm_model, llm_tokenizer, prompt=prompt, max_tokens=500, verbose=True,
            sampler=_PLANNER_SAMPLER
        )

        print(f"\n[PLANNER] Raw LLM response:\n{repr(response)}")

        plan = ReasoningPlan(main_query=query)
        plan_data = parse_json_safely(response)
        print(f"[PLANNER] Parsed JSON: {plan_data}")

        if plan_data and "goals" in plan_data:
            for goal_data in plan_data.get("goals", [])[:MAX_REASONING_GOALS]:
                if isinstance(goal_data, dict) and "description" in goal_data:
                    plan.add_goal(ReasoningGoal(description=goal_data["description"]))

            if plan.goals:
                print(f"\n[PLANNER] Created plan with {len(plan.goals)} goals:")
                for i, g in enumerate(plan.goals, 1):
                    print(f"  {i}. {g.description}")
            else:
                print(f"[PLANNER] No valid goals found, falling back to single goal")
                plan.add_goal(ReasoningGoal(description=query))
        else:
            print(f"[PLANNER] JSON parse failed, falling back to single goal")
            plan.add_goal(ReasoningGoal(description=query))

        return plan

    @staticmethod
    def replan_goal(goal: ReasoningGoal, evaluation: Dict, llm_model, llm_tokenizer) -> ReasoningGoal:
        """Reformulate one sparse goal into a new ReasoningGoal with updated keywords."""
        current_datetime = datetime.now().strftime("%A, %B %d, %Y at %H:%M")
        system_prompt = f"""Current date and time: {current_datetime}

You are a search query optimizer. A web search returned sparse results for a research goal.
Rewrite the goal as a new, different web search query using different keywords.

RULES:
- Use 3-6 different keywords (synonyms, broader/narrower terms, related concepts)
- Write like a Google search, not a question

OUTPUT (JSON only):
{{"new_description": "new keyword query here", "reasoning": "one sentence"}}"""

        user_prompt = f"""Failed goal: {goal.description}
Evaluation: {evaluation.get('reasoning', 'sparse results')}

Rewrite with different keywords. Output JSON only."""

        conversation = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        prompt = llm_tokenizer.apply_chat_template(
            conversation, add_generation_prompt=True, tokenize=False
        )

        print(f"\n[REPLAN] Formatted prompt:\n{prompt}")

        response = generate_fixed(
            llm_model, llm_tokenizer, prompt=prompt, max_tokens=200, verbose=True,
            sampler=_PLANNER_SAMPLER
        )

        print(f"\n[REPLAN] Raw LLM response:\n{repr(response)}")

        replan_data = parse_json_safely(response)
        print(f"[REPLAN] Parsed JSON: {replan_data}")

        if replan_data and "new_description" in replan_data:
            new_desc = replan_data["new_description"]
            print(f"[REPLAN] New description: {new_desc}")
        else:
            print(f"[REPLAN] Parse failed — falling back to original description")
            new_desc = goal.description

        return ReasoningGoal(
            description=new_desc,
            replan_count=goal.replan_count + 1
        )
