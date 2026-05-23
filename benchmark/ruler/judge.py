"""
Scoring / judging module for RULER.

RULER is synthetic with exact ground truth, so the primary scorer is
ExactMatchScorer.  GPT4oJudge is provided for tasks where exact match
is too strict (e.g. paraphrase answers in qa_1 / qa_2).  MockJudge is
used in dry-run / CI environments.
"""

from __future__ import annotations

import os
from typing import Union


class ExactMatchScorer:
    """Exact-match scorer for RULER tasks.

    Returns 1.0 if the answer string contains any of the reference strings
    (case-insensitive substring match), else 0.0.

    RULER answers are typically short identifiers (needle strings, word
    lists, variable values), so substring containment is the appropriate
    check rather than full-string equality.
    """

    def score(
        self,
        answer: str,
        reference: Union[str, list],
    ) -> float:
        """Score an answer against one or more references.

        Args:
            answer: The model-generated answer string.
            reference: A single reference string or a list of reference strings.
                       Scoring passes (1.0) if the answer contains ANY reference.

        Returns:
            1.0 if matched, 0.0 otherwise.
        """
        if isinstance(reference, str):
            references = [reference]
        else:
            references = list(reference)

        answer_lower = answer.strip().lower()
        for ref in references:
            if ref.strip().lower() in answer_lower:
                return 1.0
        return 0.0


class GPT4oJudge:
    """LLM-based judge backed by the OpenAI API.

    Reads OPENAI_API_KEY from the environment.  The model is controlled
    by the JUDGE_MODEL env var (default: gpt-4o).

    Raises RuntimeError on construction if OPENAI_API_KEY is absent.
    """

    def __init__(self) -> None:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "OPENAI_API_KEY environment variable is not set. "
                "Export it before using GPT4oJudge."
            )
        self._api_key = api_key
        self._model = os.environ.get("JUDGE_MODEL", "gpt-4o")
        # Import lazily so the module can be imported without openai installed
        try:
            import openai  # noqa: F401
        except ImportError as exc:
            raise RuntimeError(
                "openai package is required for GPT4oJudge. "
                "Install it with: pip install openai"
            ) from exc

    def score(
        self,
        answer: str,
        reference: Union[str, list],
        context: str = "",
    ) -> float:
        """Score an answer using GPT-4o as a judge.

        Args:
            answer: Model-generated answer.
            reference: Ground-truth reference(s).
            context: Optional context/question for the judge prompt.

        Returns:
            Score between 0.0 and 1.0.
        """
        import openai

        client = openai.OpenAI(api_key=self._api_key)

        if isinstance(reference, list):
            ref_text = "\n".join(f"- {r}" for r in reference)
        else:
            ref_text = reference

        system_prompt = (
            "You are an evaluator for a reading-comprehension benchmark. "
            "Given a model answer and the ground-truth reference(s), respond "
            "with exactly one of: CORRECT or INCORRECT."
        )
        user_prompt = (
            f"Reference answer(s):\n{ref_text}\n\n"
            f"Model answer:\n{answer}\n\n"
            "Is the model answer correct? Reply CORRECT or INCORRECT only."
        )
        if context:
            user_prompt = f"Context/Question:\n{context}\n\n" + user_prompt

        response = client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=5,
            temperature=0.0,
        )
        verdict = response.choices[0].message.content.strip().upper()
        return 1.0 if "CORRECT" in verdict else 0.0


class MockJudge:
    """Deterministic mock judge for CI / dry-run environments.

    Returns 1.0 for any non-empty answer, 0.0 for empty answers.
    Does not call any external API.
    """

    def score(
        self,
        answer: str,
        reference: Union[str, list],
        context: str = "",
    ) -> float:
        return 1.0 if answer.strip() else 0.0
