"""GraphWalks scoring — purely programmatic set-overlap F1.

Verified against openai/graphwalks HF dataset card.

GraphWalks is purely programmatic — no LLM judge required.  The official
scoring method uses set-overlap F1 over node sets.  Models must end responses
with ``Final Answer: [node1, node2, ...]``; responses not in this format
score 0.
"""

from __future__ import annotations

import re


def parse_answer(response: str) -> list[str]:
    """Extract node list from ``Final Answer: [node1, node2, ...]`` on the last line.

    Returns an empty list if the expected format is absent.
    """
    line = response.split("\n")[-1]
    if "Final Answer:" not in line:
        return []
    match = re.search(r"Final Answer: ?\[.*\]", line)
    if not match:
        return []
    inner = match.group(0).removeprefix("Final Answer:").strip()
    inner = inner.strip("[]")
    return [x.strip() for x in inner.split(",") if x.strip()]


class SetF1Scorer:
    """Set-overlap F1 scorer for GraphWalks.

    Computes F1 between the predicted node set (parsed from the model's
    ``Final Answer: [...]`` line) and the ground-truth node set.

    Both task types (BFS, Parents) use this scorer.
    """

    def score(self, prediction: str, reference: list[str]) -> float:
        """Return set-overlap F1 in [0.0, 1.0].

        Parameters
        ----------
        prediction:
            Raw model response string.  Parsed via :func:`parse_answer`.
        reference:
            Ground-truth node list.
        """
        sampled_set = set(parse_answer(prediction))
        truth_set = set(reference)
        n_sampled = len(sampled_set)
        n_golden = len(truth_set)
        n_overlap = len(sampled_set & truth_set)
        recall = n_overlap / n_golden if n_golden > 0 else 0
        precision = n_overlap / n_sampled if n_sampled > 0 else 0
        if recall + precision == 0:
            return 1.0 if n_golden == 0 else 0.0
        return 2 * (recall * precision) / (recall + precision)


class MockJudge:
    """Deterministic judge for CPU dry-runs and CI (no API calls).

    Returns 1.0 for any non-empty answer, 0.0 for empty.
    """

    def score(self, answer: str, reference: list[str]) -> float:  # noqa: ARG002
        return 1.0 if answer.strip() else 0.0
