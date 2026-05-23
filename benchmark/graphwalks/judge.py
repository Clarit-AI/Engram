"""Scoring / judging for GraphWalks benchmark.

GraphWalks has exact ground-truth answers (node names along a path), so
``ExactMatchScorer`` is the primary judge.  ``GPT4oJudge`` is available for
borderline or free-form cases.  ``MockJudge`` is used in dry-run / CI contexts
where no API key is available.
"""

from __future__ import annotations

import os
import re
import string


class ExactMatchScorer:
    """Score answers by exact match after normalisation.

    Normalisation:
    - Strip leading / trailing whitespace.
    - Lower-case.
    - Remove punctuation (``string.punctuation``).
    - Collapse internal whitespace to a single space.
    """

    @staticmethod
    def _normalize(text: str) -> str:
        text = text.strip().lower()
        text = text.translate(str.maketrans("", "", string.punctuation))
        text = re.sub(r"\s+", " ", text)
        return text

    def score(self, answer: str, reference: str) -> float:
        """Return 1.0 if normalised *answer* matches normalised *reference*, else 0.0."""
        return 1.0 if self._normalize(answer) == self._normalize(reference) else 0.0


class GPT4oJudge:
    """LLM-as-judge for borderline or free-form answers.

    Reads ``OPENAI_API_KEY`` from the environment.  Raises ``RuntimeError`` at
    construction time if the key is absent so that callers fail fast rather than
    at the first actual judging call.

    The model is controlled by the ``JUDGE_MODEL`` env var, defaulting to
    ``gpt-4o``.
    """

    SYSTEM_PROMPT = (
        "You are a precise evaluator for a graph-traversal QA benchmark. "
        "The user will provide a ground-truth answer and a model answer. "
        "Reply with only '1' if the model answer is correct, or '0' if incorrect."
    )

    def __init__(self) -> None:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "OPENAI_API_KEY is not set. "
                "Export it before using GPT4oJudge, or use MockJudge for dry runs."
            )
        self._api_key = api_key
        self._model = os.environ.get("JUDGE_MODEL", "gpt-4o")

    def score(self, answer: str, reference: str) -> float:
        """Ask the judge model whether *answer* matches *reference*.

        Returns 1.0 for correct, 0.0 for incorrect.
        """
        try:
            import openai  # noqa: PLC0415
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "openai package is required for GPT4oJudge. Install it with: "
                "pip install openai"
            ) from exc

        client = openai.OpenAI(api_key=self._api_key)
        user_content = (
            f"Ground truth: {reference}\n"
            f"Model answer: {answer}\n"
            "Is the model answer correct? Reply with only '1' or '0'."
        )
        response = client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": self.SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ],
            max_tokens=4,
            temperature=0.0,
        )
        verdict = response.choices[0].message.content.strip()
        return 1.0 if verdict.startswith("1") else 0.0


class MockJudge:
    """Deterministic judge for CPU dry-runs and CI (no API calls).

    Returns 1.0 for any non-empty answer, 0.0 for empty.
    """

    def score(self, answer: str, reference: str) -> float:  # noqa: ARG002
        return 1.0 if answer.strip() else 0.0
