"""GraphWalks benchmark runner.

Two-phase execution:
1. ``run_baseline`` — sends full graph context + question to the model every
   time (simulating a stateless deployment).
2. ``run_engram`` — checks for a pre-saved snapshot; if present (warm), sends
   only the question; if absent (cold), sends full context, then saves a stub
   snapshot.

Usage
-----
python -m benchmark.graphwalks.runner \\
    --model-url http://localhost:30000/v1 \\
    --snapshot-dir /tmp/snapshots \\
    --output results/graphwalks.jsonl \\
    --num-questions 100 \\
    --mode both
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Literal, Optional

from .judge import MockJudge, SetF1Scorer
from .results import QuestionResult, RunSummary

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass
class Question:
    """A single GraphWalks question."""

    question_id: str
    graph_text: str  # Textual description of the graph
    question: str    # Natural-language question to answer
    answer: list     # Ground-truth answer node list
    num_hops: int
    num_nodes: int


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------


def _post_completion(
    model_url: str,
    prompt: str,
    model: str,
    timeout: float = 120.0,
    http_client=None,
) -> tuple[str, float]:
    """Send a chat completion request; return (answer_text, ttft_seconds).

    If *http_client* is provided (for testing), calls
    ``http_client.post(url, json=payload)`` and expects a dict response.
    Otherwise uses the ``requests`` library.
    """
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 256,
        "temperature": 0.0,
    }
    url = model_url.rstrip("/") + "/chat/completions"

    t0 = time.perf_counter()

    if http_client is not None:
        response_data = http_client.post(url, json=payload)
    else:
        import requests  # noqa: PLC0415

        resp = requests.post(url, json=payload, timeout=timeout)
        resp.raise_for_status()
        response_data = resp.json()

    ttft = time.perf_counter() - t0

    try:
        text = response_data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        logger.warning("Unexpected response shape: %s — %s", response_data, exc)
        text = ""

    return text, ttft


def _count_tokens(text: str) -> int:
    """Token count proxy: split on whitespace."""
    return len(text.split())


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


class GraphWalksRunner:
    """Orchestrates baseline and Engram runs for a list of questions.

    Parameters
    ----------
    model_url:
        Base URL of the OpenAI-compatible inference server
        (e.g. ``http://localhost:30000/v1``).
    snapshot_dir:
        Directory where Engram snapshots are stored.  Each snapshot lives at
        ``<snapshot_dir>/<question_id>/snapshot.json``.
    model:
        Model name to pass in the ``"model"`` field of each request.
    scorer:
        Scoring object with a ``.score(answer, reference) -> float`` method.
        Defaults to ``ExactMatchScorer``.
    http_client:
        Optional test double for the HTTP layer.  Must expose a
        ``.post(url, json=...) -> dict`` interface.
    """

    def __init__(
        self,
        model_url: str = "http://localhost:30000/v1",
        snapshot_dir: "str | Path" = "/tmp/engram_snapshots",
        model: str = "default",
        scorer=None,
        http_client=None,
    ) -> None:
        self.model_url = model_url
        self.snapshot_dir = Path(snapshot_dir)
        self.model = model
        self.scorer = scorer if scorer is not None else SetF1Scorer()
        self.http_client = http_client

    # ------------------------------------------------------------------
    # Baseline
    # ------------------------------------------------------------------

    def run_baseline(self, questions: List[Question]) -> List[QuestionResult]:
        """Run full-context baseline for every question."""
        results: List[QuestionResult] = []
        for q in questions:
            prompt = (
                f"{q.graph_text}\n\n{q.question}\n\n"
                "End your response with exactly: Final Answer: [node1, node2, ...]"
            )
            input_tokens = _count_tokens(prompt)

            answer, ttft = _post_completion(
                self.model_url,
                prompt,
                self.model,
                http_client=self.http_client,
            )

            output_tokens = _count_tokens(answer)
            score = self.scorer.score(answer, q.answer)

            # Baseline always runs the full context; it has no warm/cold
            # distinction.  We label it "cold" to reflect that no state was
            # reused.
            results.append(
                QuestionResult(
                    question_id=q.question_id,
                    graph_size=q.num_nodes,
                    num_hops=q.num_hops,
                    restore_mode="cold",
                    baseline_ttft_s=ttft,
                    baseline_input_tokens=input_tokens,
                    baseline_output_tokens=output_tokens,
                    baseline_answer=answer,
                    baseline_score=score,
                    # Engram fields are placeholders until run_engram fills them.
                    engram_ttft_s=0.0,
                    engram_input_tokens=0,
                    engram_output_tokens=0,
                    engram_answer="",
                    engram_score=0.0,
                )
            )
            logger.debug(
                "Baseline %s: ttft=%.3fs tokens=%d score=%.1f",
                q.question_id,
                ttft,
                input_tokens,
                score,
            )
        return results

    # ------------------------------------------------------------------
    # Engram (snapshot-aware)
    # ------------------------------------------------------------------

    def _snapshot_path(self, question_id: str) -> Path:
        return self.snapshot_dir / question_id / "snapshot.json"

    def _snapshot_exists(self, question_id: str) -> bool:
        return self._snapshot_path(question_id).exists()

    def _save_stub_snapshot(self, question_id: str, graph_text: str) -> None:
        """Write a minimal stub snapshot so subsequent runs are warm."""
        snap_path = self._snapshot_path(question_id)
        snap_path.parent.mkdir(parents=True, exist_ok=True)
        stub = {
            "question_id": question_id,
            "graph_text_hash": hash(graph_text),
            "snapshot_version": 1,
        }
        snap_path.write_text(json.dumps(stub), encoding="utf-8")

    def run_engram(self, questions: List[Question]) -> List[QuestionResult]:
        """Run Engram (snapshot-restored) inference for every question.

        Warm questions send only the question prompt (graph state already in
        snapshot).  Cold questions send the full context and then save a stub
        snapshot for future warm runs.
        """
        results: List[QuestionResult] = []
        for q in questions:
            is_warm = self._snapshot_exists(q.question_id)
            restore_mode: Literal["warm", "cold"] = "warm" if is_warm else "cold"

            answer_instruction = (
                "\n\nEnd your response with exactly: Final Answer: [node1, node2, ...]"
            )
            if is_warm:
                # Warm: only send the question — graph context is in the snapshot.
                prompt = q.question + answer_instruction
            else:
                # Cold: send full context, then persist a stub snapshot.
                prompt = f"{q.graph_text}\n\n{q.question}" + answer_instruction
                self._save_stub_snapshot(q.question_id, q.graph_text)

            input_tokens = _count_tokens(prompt)

            answer, ttft = _post_completion(
                self.model_url,
                prompt,
                self.model,
                http_client=self.http_client,
            )

            output_tokens = _count_tokens(answer)
            score = self.scorer.score(answer, q.answer)

            results.append(
                QuestionResult(
                    question_id=q.question_id,
                    graph_size=q.num_nodes,
                    num_hops=q.num_hops,
                    restore_mode=restore_mode,
                    # Baseline fields are placeholders; run_baseline fills them.
                    baseline_ttft_s=0.0,
                    baseline_input_tokens=0,
                    baseline_output_tokens=0,
                    baseline_answer="",
                    baseline_score=0.0,
                    engram_ttft_s=ttft,
                    engram_input_tokens=input_tokens,
                    engram_output_tokens=output_tokens,
                    engram_answer=answer,
                    engram_score=score,
                )
            )
            logger.debug(
                "Engram %s (%s): ttft=%.3fs tokens=%d score=%.1f",
                q.question_id,
                restore_mode,
                ttft,
                input_tokens,
                score,
            )
        return results

    # ------------------------------------------------------------------
    # Combined run helpers
    # ------------------------------------------------------------------

    def merge_results(
        self,
        baseline: List[QuestionResult],
        engram: List[QuestionResult],
    ) -> List[QuestionResult]:
        """Merge baseline and Engram results into a unified list.

        The Engram ``restore_mode`` label is authoritative (warm/cold).
        Baseline measurements are copied into the combined record.
        """
        baseline_by_id = {r.question_id: r for r in baseline}
        merged: List[QuestionResult] = []
        for er in engram:
            br = baseline_by_id.get(er.question_id)
            if br is None:
                logger.warning("No baseline result for %s", er.question_id)
                merged.append(er)
                continue
            merged.append(
                QuestionResult(
                    question_id=er.question_id,
                    graph_size=er.graph_size,
                    num_hops=er.num_hops,
                    restore_mode=er.restore_mode,
                    baseline_ttft_s=br.baseline_ttft_s,
                    baseline_input_tokens=br.baseline_input_tokens,
                    baseline_output_tokens=br.baseline_output_tokens,
                    baseline_answer=br.baseline_answer,
                    baseline_score=br.baseline_score,
                    engram_ttft_s=er.engram_ttft_s,
                    engram_input_tokens=er.engram_input_tokens,
                    engram_output_tokens=er.engram_output_tokens,
                    engram_answer=er.engram_answer,
                    engram_score=er.engram_score,
                )
            )
        return merged


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="GraphWalks benchmark runner for Engram.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--model-url",
        default="http://localhost:30000/v1",
        help="Base URL of the OpenAI-compatible inference server.",
    )
    p.add_argument(
        "--model",
        default="default",
        help='Model name to pass in the "model" field.',
    )
    p.add_argument(
        "--snapshot-dir",
        default="/tmp/engram_snapshots",
        help="Directory for Engram snapshot files.",
    )
    p.add_argument(
        "--output",
        default="results/graphwalks.jsonl",
        help="Output JSONL path.",
    )
    p.add_argument(
        "--num-questions",
        type=int,
        default=100,
        help="Number of questions to run (0 = all).",
    )
    p.add_argument(
        "--mode",
        choices=["baseline", "engram", "both"],
        default="both",
        help="Which phases to run.",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Use synthetic questions and MockJudge; no live server required.",
    )
    p.add_argument(
        "--data-path",
        default=None,
        help="Path to a JSONL file of questions. Defaults to the downloaded dataset.",
    )
    return p


def _load_questions(data_path: Optional[str], num_questions: int) -> List[Question]:
    """Load questions from a JSONL file."""
    if data_path is None:
        default = (
            Path(__file__).parent / "data" / "graphwalks" / "test_dry_run.jsonl"
        )
        data_path = str(default)

    path = Path(data_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}. "
            "Run: python -m benchmark.graphwalks.download"
        )

    questions: List[Question] = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            questions.append(
                Question(
                    question_id=d["question_id"],
                    graph_text=d["graph_text"],
                    question=d["question"],
                    answer=d["answer"],
                    num_hops=d["num_hops"],
                    num_nodes=d["num_nodes"],
                )
            )
            if num_questions and len(questions) >= num_questions:
                break
    return questions


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = _build_parser()
    args = parser.parse_args()

    scorer = MockJudge() if args.dry_run else SetF1Scorer()

    if args.dry_run:
        from .download import generate_synthetic_question  # noqa: PLC0415

        raw = [generate_synthetic_question(num_nodes=5, num_hops=3, seed=i) for i in range(5)]
        questions = [
            Question(
                question_id=d["question_id"],
                graph_text=d["graph_text"],
                question=d["question"],
                answer=d["answer"],
                num_hops=d["num_hops"],
                num_nodes=d["num_nodes"],
            )
            for d in raw
        ]
    else:
        questions = _load_questions(args.data_path, args.num_questions)

    runner = GraphWalksRunner(
        model_url=args.model_url,
        snapshot_dir=args.snapshot_dir,
        model=args.model,
        scorer=scorer,
    )

    baseline_results: List[QuestionResult] = []
    engram_results: List[QuestionResult] = []

    if args.mode in ("baseline", "both"):
        logger.info("Running baseline phase (%d questions)…", len(questions))
        baseline_results = runner.run_baseline(questions)

    if args.mode in ("engram", "both"):
        logger.info("Running Engram phase (%d questions)…", len(questions))
        engram_results = runner.run_engram(questions)

    if args.mode == "both":
        merged = runner.merge_results(baseline_results, engram_results)
    elif args.mode == "baseline":
        merged = baseline_results
    else:
        merged = engram_results

    summary = RunSummary(model=args.model, results=merged)
    summary.to_jsonl(args.output)
    logger.info("Results written to %s", args.output)

    # Print a brief summary to stdout.
    print("\n=== GraphWalks Run Summary ===")
    print(f"  Model        : {summary.model}")
    print(f"  Questions    : {len(merged)}")
    print(f"  Warm         : {len(summary.warm_results)}")
    print(f"  Cold         : {len(summary.cold_results)}")
    if summary.token_reduction is not None:
        print(f"  Token Reduction (all)  : {summary.token_reduction:.1%}")
    if summary.warm_token_reduction is not None:
        print(f"  Token Reduction (warm) : {summary.warm_token_reduction:.1%}")
    if summary.ttft_speedup is not None:
        print(f"  TTFT Speedup           : {summary.ttft_speedup:.2f}x")


if __name__ == "__main__":
    main()
