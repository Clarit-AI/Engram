"""
RULER scoring — purely programmatic, no LLM judge.
Verified against hsiehjackson/RULER/scripts/eval/synthetic/constants.py.

RULER is a synthetic benchmark with exact ground truth. Scoring is done by
substring matching, not by an LLM judge. Two scoring functions are used:

- string_match_all: recall-based. For each (prediction, references) pair,
  computes the fraction of reference strings found as substrings in the
  prediction. Used for NIAH variants, VT, CWE, FWE.

- string_match_part: any-match. For each pair, returns 1.0 if any reference
  is found as a substring in the prediction, else 0.0. Used for QA_1
  (SQuAD) and QA_2 (HotPotQA).

No OPENAI_API_KEY is required for RULER scoring.
"""

from __future__ import annotations

from typing import Union


class StringMatchAllScorer:
    """Recall-based substring scorer for RULER.

    For a single (prediction, references) pair: computes the fraction of
    reference strings that appear as substrings in the prediction
    (case-insensitive). This matches `string_match_all` in the official
    hsiehjackson/RULER harness.

    Returns a value in [0.0, 1.0] (NOT 0–100).

    Used for: NIAH variants, VT, CWE, FWE.
    """

    def score(
        self,
        prediction: str,
        references: Union[str, list],
    ) -> float:
        """Score a prediction against a list of reference strings.

        Args:
            prediction: The model-generated response string.
            references: A single reference string or a list of valid
                        reference strings. ALL must be found for score=1.0.

        Returns:
            Fraction of references found as substrings in the prediction,
            in [0.0, 1.0].
        """
        if isinstance(references, str):
            ref_list = [references]
        else:
            ref_list = list(references)

        if not ref_list:
            return 0.0

        pred_lower = prediction.lower()
        hits = sum(1.0 if r.lower() in pred_lower else 0.0 for r in ref_list)
        return hits / len(ref_list)


class StringMatchPartScorer:
    """Any-match substring scorer for RULER QA tasks.

    For a single (prediction, references) pair: returns 1.0 if ANY reference
    string appears as a substring in the prediction (case-insensitive),
    else 0.0. This matches `string_match_part` in the official
    hsiehjackson/RULER harness.

    Returns a value in [0.0, 1.0] (NOT 0–100).

    Used for: qa_1 (SQuAD), qa_2 (HotPotQA).
    """

    def score(
        self,
        prediction: str,
        references: Union[str, list],
    ) -> float:
        """Score a prediction against a list of reference strings.

        Args:
            prediction: The model-generated response string.
            references: A single reference string or a list of valid
                        reference strings. ANY match yields score=1.0.

        Returns:
            1.0 if any reference is found as a substring in the prediction,
            else 0.0.
        """
        if isinstance(references, str):
            ref_list = [references]
        else:
            ref_list = list(references)

        if not ref_list:
            return 0.0

        pred_lower = prediction.lower()
        return max(1.0 if r.lower() in pred_lower else 0.0 for r in ref_list)


# QA tasks that use string_match_part (any-match)
_QA_TASKS = {"qa_1", "qa_2"}


def get_scorer(task_name: str):
    """Return the correct scorer for the given RULER task name.

    Args:
        task_name: RULER task name (e.g. "niah_single_1", "qa_1").

    Returns:
        StringMatchPartScorer for qa_1/qa_2; StringMatchAllScorer otherwise.
    """
    if task_name in _QA_TASKS:
        return StringMatchPartScorer()
    return StringMatchAllScorer()


class MockJudge:
    """Deterministic mock judge for CI / dry-run environments.

    Returns 1.0 for any non-empty prediction, 0.0 for empty predictions.
    Does not call any external API. Compatible with the scorer interface.
    """

    def score(
        self,
        prediction: str,
        references: Union[str, list],
    ) -> float:
        return 1.0 if prediction.strip() else 0.0
