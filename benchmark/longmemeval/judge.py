"""LLM-as-judge for LongMemEval answer scoring.

``GPT4oJudge`` calls the OpenAI API to evaluate whether a model answer
correctly answers a question given the reference ground truth.

``MockJudge`` is a deterministic stub for CPU / dry-run testing: it scores
non-empty answers as 1.0 and empty answers as 0.0.
"""

from __future__ import annotations

import json
import os
from abc import ABC, abstractmethod
from typing import List


class BaseJudge(ABC):
    """Abstract base for answer judges."""

    @abstractmethod
    def score(self, question: str, reference: str, answer: str) -> float:
        """Return a score in [0.0, 1.0] for *answer* given *question* and *reference*."""

    def score_batch(self, items: List[dict]) -> List[float]:
        """Score a batch of items.

        Each item must have keys: ``question``, ``reference``, ``answer``.
        Default implementation calls :meth:`score` sequentially; override for
        parallelism.
        """
        return [
            self.score(item["question"], item["reference"], item["answer"])
            for item in items
        ]


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

_JUDGE_SYSTEM_PROMPT = """\
You are an impartial judge evaluating whether a model answer is correct.
Given a question, a reference (ground-truth) answer, and a model answer,
determine whether the model answer correctly addresses the question.

Respond ONLY with valid JSON: {"score": 1} if the model answer is correct,
{"score": 0} if it is incorrect or does not answer the question.
Do not add any other text outside the JSON object.
"""

_JUDGE_USER_TEMPLATE = """\
Question: {question}

Reference answer: {reference}

Model answer: {answer}

Is the model answer correct?"""


def _build_messages(question: str, reference: str, answer: str) -> list:
    return [
        {"role": "system", "content": _JUDGE_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": _JUDGE_USER_TEMPLATE.format(
                question=question, reference=reference, answer=answer
            ),
        },
    ]


# ---------------------------------------------------------------------------
# GPT-4o judge
# ---------------------------------------------------------------------------

class GPT4oJudge(BaseJudge):
    """Score answers with GPT-4o (or any model configurable via ``JUDGE_MODEL``)."""

    def __init__(self) -> None:
        api_key = os.environ.get("OPENAI_API_KEY", "")
        if not api_key:
            raise RuntimeError(
                "OPENAI_API_KEY is not set. "
                "Export it before running the judge:\n"
                "  export OPENAI_API_KEY=sk-..."
            )
        # Import lazily so the module is importable without `openai` installed
        try:
            import openai  # noqa: F401
        except ImportError as exc:
            raise RuntimeError(
                "The 'openai' package is required for GPT4oJudge. "
                "Install it with: pip install openai"
            ) from exc

        import openai as _openai

        self._client = _openai.OpenAI(api_key=api_key)
        self._model = os.environ.get("JUDGE_MODEL", "gpt-4o")

    def score(self, question: str, reference: str, answer: str) -> float:
        """Return 0.0 or 1.0 based on GPT-4o judgement."""
        messages = _build_messages(question, reference, answer)
        response = self._client.chat.completions.create(
            model=self._model,
            messages=messages,
            temperature=0.0,
            max_tokens=16,
            response_format={"type": "json_object"},
        )
        content = response.choices[0].message.content or "{}"
        try:
            parsed = json.loads(content)
            return float(parsed.get("score", 0))
        except (json.JSONDecodeError, TypeError, ValueError):
            return 0.0

    def score_batch(self, items: List[dict]) -> List[float]:
        """Score items sequentially (API rate limits make parallelism fragile)."""
        return [
            self.score(item["question"], item["reference"], item["answer"])
            for item in items
        ]


# ---------------------------------------------------------------------------
# Mock judge (CPU / no-API dry-run)
# ---------------------------------------------------------------------------

class MockJudge(BaseJudge):
    """Deterministic stub judge for CPU dry-run testing.

    Returns 1.0 for any non-empty answer, 0.0 for empty answers.
    No API calls are made.
    """

    def score(self, question: str, reference: str, answer: str) -> float:  # noqa: ARG002
        return 1.0 if answer.strip() else 0.0

    def score_batch(self, items: List[dict]) -> List[float]:
        return [self.score(i["question"], i["reference"], i["answer"]) for i in items]
