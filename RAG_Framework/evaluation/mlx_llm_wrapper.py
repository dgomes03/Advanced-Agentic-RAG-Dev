"""Custom RAGAS LLM wrapper for local MLX-LM models.

RAGAS 0.4+ requires InstructorBaseRagasLLM which returns structured Pydantic
objects.  We prompt the local model to output JSON matching the expected schema,
then parse it into the requested Pydantic model.
"""

import asyncio
import json
import re
import typing as t

from mlx_lm import generate
from mlx_lm.sample_utils import make_sampler
from pydantic import BaseModel
from ragas.llms import InstructorBaseRagasLLM

from RAG_Framework.core.config import EVAL_LLM_MAX_TOKENS, EVAL_LLM_TEMPERATURE

T = t.TypeVar("T", bound=BaseModel)


def _repair_json(text: str) -> str:
    """Try to repair truncated JSON by removing the last incomplete element
    and closing open brackets/braces."""
    # Strip trailing whitespace and commas
    text = text.rstrip().rstrip(",")

    # If it already parses, return as-is
    try:
        json.loads(text)
        return text
    except json.JSONDecodeError:
        pass

    # Try progressively removing trailing content to find a valid state
    # Strategy: find the last complete object/value before the truncation
    # by looking for the last "}," or "}" that leaves valid JSON after repair
    for i in range(len(text) - 1, 0, -1):
        if text[i] in ('}', ']', '"') or text[i].isdigit():
            candidate = text[:i + 1]
            # Count and close open brackets
            open_braces = candidate.count("{") - candidate.count("}")
            open_brackets = candidate.count("[") - candidate.count("]")
            if open_braces >= 0 and open_brackets >= 0:
                repaired = candidate + "]" * open_brackets + "}" * open_braces
                try:
                    json.loads(repaired)
                    return repaired
                except json.JSONDecodeError:
                    continue

    return text


def _extract_json(text: str) -> str:
    """Pull the first JSON object/array from LLM output."""
    # Try ```json ... ``` fences first
    m = re.search(r"```(?:json)?\s*(\{.*?\}|\[.*?\])\s*```", text, re.DOTALL)
    if m:
        return m.group(1)
    # Bare complete JSON object
    m = re.search(r"(\{.*\})", text, re.DOTALL)
    if m:
        return m.group(1)
    # Bare complete JSON array
    m = re.search(r"(\[.*\])", text, re.DOTALL)
    if m:
        return m.group(1)
    # Truncated — find start of JSON and try to repair
    m = re.search(r"(\{.*)", text, re.DOTALL)
    if m:
        return _repair_json(m.group(1))
    return text


class MLXRagasLLM(InstructorBaseRagasLLM):
    """Wraps an already-loaded MLX-LM model/tokenizer for RAGAS 0.4+ evaluation."""

    def __init__(self, model, tokenizer, *,
                 max_tokens: int = EVAL_LLM_MAX_TOKENS,
                 temperature: float = EVAL_LLM_TEMPERATURE):
        self.model = model
        self.tokenizer = tokenizer
        self.max_tokens = max_tokens
        self.temperature = temperature

    def _call_llm(self, prompt: str) -> str:
        conversation = [{"role": "user", "content": prompt}]
        formatted = self.tokenizer.apply_chat_template(
            conversation, add_generation_prompt=True, tokenize=False,
        )
        sampler = make_sampler(temp=self.temperature)
        return generate(
            self.model, self.tokenizer,
            prompt=formatted,
            max_tokens=self.max_tokens,
            sampler=sampler,
            verbose=False,
        )

    def _generate_structured(self, prompt: str, response_model: t.Type[T]) -> T:
        schema = response_model.model_json_schema()
        full_prompt = (
            f"{prompt}\n\n"
            f"Respond with ONLY a valid JSON object (no markdown, no explanation). "
            f"Keep it concise. Schema:\n{json.dumps(schema)}"
        )
        raw = self._call_llm(full_prompt)
        extracted = _extract_json(raw)
        try:
            return response_model.model_validate_json(extracted)
        except Exception:
            # Try repair
            repaired = _repair_json(extracted)
            try:
                return response_model.model_validate_json(repaired)
            except Exception:
                pass
            # Retry once with stricter instruction
            retry_prompt = (
                f"{prompt}\n\n"
                f"CRITICAL: Output ONLY valid JSON, nothing else. Be very concise. Schema:\n"
                f"{json.dumps(schema)}"
            )
            raw = self._call_llm(retry_prompt)
            extracted = _extract_json(raw)
            repaired = _repair_json(extracted)
            return response_model.model_validate_json(repaired)

    def generate(self, prompt: str, response_model: t.Type[T]) -> T:
        return self._generate_structured(prompt, response_model)

    async def agenerate(self, prompt: str, response_model: t.Type[T]) -> T:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, self._generate_structured, prompt, response_model,
        )
