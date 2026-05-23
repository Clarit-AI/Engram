"""LLM-as-judge for LongMemEval answer scoring.

``LLMJudge`` calls an OpenAI-compatible API to evaluate whether a model answer
correctly answers a question given the reference ground truth.  The judge model
and endpoint are fully configurable via environment variables (see
``LLMJudge`` docstring).

``MockJudge`` is a deterministic stub for CPU / dry-run testing: it scores
non-empty answers as 1.0 and empty answers as 0.0.
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from typing import List

# ---------------------------------------------------------------------------
# Judge configuration (env-var driven)
# ---------------------------------------------------------------------------

# Official judge per LongMemEval paper §3.3 is gpt-4o-2024-08-06.
# Override via JUDGE_MODEL env var to use any OpenAI-compatible model.
JUDGE_MODEL = os.environ.get("JUDGE_MODEL", "gpt-4o-2024-08-06")

# Base URL for the judge API.  None means use the OpenAI default endpoint.
# Set JUDGE_BASE_URL to point at any OpenAI-compatible server
# (e.g. a local vLLM/SGLang instance, Azure OpenAI, etc.).
JUDGE_BASE_URL = os.environ.get("JUDGE_BASE_URL", None)

# The env var name that must hold the API key for the judge.
JUDGE_API_KEY_ENV = "OPENAI_API_KEY"


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
# Prompt — matches official xiaowu0162/LongMemEval evaluate_qa.py
# ---------------------------------------------------------------------------

_JUDGE_SYSTEM_PROMPT = """\
You are an impartial judge evaluating whether a model answer is correct.
Given a question, a reference (ground-truth) answer, and a model answer,
determine whether the model answer correctly addresses the question.

Respond with a single word: yes if the model answer is correct, no if it is
incorrect or does not answer the question.
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
# LLM judge (model-agnostic, env-var configured)
# ---------------------------------------------------------------------------

class LLMJudge(BaseJudge):
    """Score answers using an LLM judge via an OpenAI-compatible API.

    Official judge per LongMemEval paper §3.3 is ``gpt-4o-2024-08-06``; override
    via the ``JUDGE_MODEL`` env var.

    Scoring method (verified against xiaowu0162/LongMemEval ``evaluate_qa.py``):
    - Binary accuracy: judge prompt returns yes/no.
    - score = 1.0 if "yes" appears in the response (case-insensitive), else 0.0.

    Environment variables
    ---------------------
    JUDGE_MODEL
        Judge model name (default: ``gpt-4o-2024-08-06``).
    JUDGE_BASE_URL
        Base URL for the judge API (default: OpenAI; set to any
        OpenAI-compatible endpoint such as a local vLLM/SGLang server).
    OPENAI_API_KEY
        Required; must be set before calling the judge.
    """

    def __init__(self) -> None:
        api_key = os.environ.get(JUDGE_API_KEY_ENV, "")
        if not api_key:
            raise RuntimeError(
                f"{JUDGE_API_KEY_ENV} is not set. "
                "Export it before running the judge:\n"
                f"  export {JUDGE_API_KEY_ENV}=sk-..."
            )
        # Import lazily so the module is importable without `openai` installed.
        try:
            import openai  # noqa: F401
        except ImportError as exc:
            raise RuntimeError(
                "The 'openai' package is required for LLMJudge. "
                "Install it with: pip install openai"
            ) from exc

        import openai as _openai

        client_kwargs: dict = {"api_key": api_key}
        if JUDGE_BASE_URL is not None:
            client_kwargs["base_url"] = JUDGE_BASE_URL

        self._client = _openai.OpenAI(**client_kwargs)
        self._model = JUDGE_MODEL

    def score(self, question: str, reference: str, answer: str) -> float:
        """Return 0.0 or 1.0 using official LongMemEval binary yes/no scoring.

        Scoring follows ``xiaowu0162/LongMemEval`` ``evaluate_qa.py``:
        score = 1.0 if "yes" (case-insensitive) appears in the judge response,
        else 0.0.
        """
        messages = _build_messages(question, reference, answer)
        response = self._client.chat.completions.create(
            model=self._model,
            messages=messages,
            temperature=0.0,
            max_tokens=16,
        )
        content = response.choices[0].message.content or ""
        # Official scoring: binary yes/no, case-insensitive substring match.
        return 1.0 if "yes" in content.lower() else 0.0

    def score_batch(self, items: List[dict]) -> List[float]:
        """Score items sequentially (API rate limits make parallelism fragile)."""
        return [
            self.score(item["question"], item["reference"], item["answer"])
            for item in items
        ]


# Backwards-compatible alias; prefer LLMJudge in new code.
GPT4oJudge = LLMJudge


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
