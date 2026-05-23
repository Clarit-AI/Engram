"""
RULER harness runner — two-phase execution (baseline vs Engram).

Baseline: sends full context + question on every request.
Engram: checks for a saved snapshot; if present (warm), sends only the
question; if absent (cold), sends full context + question and saves a
stub snapshot.

Usage (CLI):
    python -m benchmark.ruler.runner \\
        --model-url http://localhost:30000 \\
        --context-lengths 4096,8192 \\
        --tasks niah_single_1,vt \\
        --output /tmp/ruler_results.jsonl \\
        --mode both \\
        --snapshot-dir /tmp/ruler_snapshots \\
        --num-samples 5
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Callable, List, Optional, Tuple

from .judge import ExactMatchScorer
from .results import RestoreMode, RunSummary, TaskResult
from .tasks import (
    RULER_TASKS,
    TaskInstance,
    generate_synthetic_task,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Token counting
# ---------------------------------------------------------------------------


def _count_tokens(text: str) -> int:
    """Count tokens using a tokenizer if available, else whitespace-split proxy."""
    try:

        # Use a fast generic tokenizer; cache it in a module-level variable
        tok = _get_cached_tokenizer()
        if tok is not None:
            return len(tok.encode(text, add_special_tokens=False))
    except Exception:
        pass
    # Fallback: whitespace split (≈1 token/word for English)
    return len(text.split())


_cached_tokenizer = None
_tokenizer_tried = False


def _get_cached_tokenizer():
    global _cached_tokenizer, _tokenizer_tried
    if _tokenizer_tried:
        return _cached_tokenizer
    _tokenizer_tried = True
    try:
        from transformers import AutoTokenizer  # type: ignore

        _cached_tokenizer = AutoTokenizer.from_pretrained(
            "gpt2", use_fast=True, local_files_only=False
        )
    except Exception:
        _cached_tokenizer = None
    return _cached_tokenizer


# ---------------------------------------------------------------------------
# HTTP client abstraction (swap for real client in integration tests)
# ---------------------------------------------------------------------------

HttpClient = Callable[[str, str, str], Tuple[str, float]]
"""Type alias: (model_url, prompt, model_name) -> (response_text, latency_s)"""


def _real_http_client(
    model_url: str,
    prompt: str,
    model_name: str,
    max_new_tokens: int = 256,
    timeout: int = 120,
) -> Tuple[str, float]:
    """Call an OpenAI-compatible /v1/completions endpoint."""
    import requests  # type: ignore

    payload = {
        "model": model_name,
        "prompt": prompt,
        "max_tokens": max_new_tokens,
        "temperature": 0.0,
    }
    url = model_url.rstrip("/") + "/v1/completions"
    t0 = time.perf_counter()
    resp = requests.post(url, json=payload, timeout=timeout)
    latency = time.perf_counter() - t0
    resp.raise_for_status()
    data = resp.json()
    text = data["choices"][0]["text"]
    return text, latency


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


class RULERRunner:
    """Two-phase RULER evaluation runner.

    Args:
        model_url: Base URL of the serving endpoint (e.g. http://localhost:30000).
        model_name: Model name passed in the API payload.
        snapshot_dir: Directory for Engram snapshots.  Each task gets its own
            subdirectory keyed by task_id.
        http_client: Optional injectable HTTP client (for testing).
        scorer: Optional injectable scorer (default: ExactMatchScorer).
        max_new_tokens: Max tokens to generate per request.
    """

    def __init__(
        self,
        model_url: str = "http://localhost:30000",
        model_name: str = "model",
        snapshot_dir: Optional[str | Path] = None,
        http_client: Optional[HttpClient] = None,
        scorer=None,
        max_new_tokens: int = 256,
    ) -> None:
        self.model_url = model_url.rstrip("/")
        self.model_name = model_name
        self.snapshot_dir = Path(snapshot_dir) if snapshot_dir else Path("/tmp/ruler_snapshots")
        self.snapshot_dir.mkdir(parents=True, exist_ok=True)
        self._http_client = http_client or _real_http_client
        self.scorer = scorer if scorer is not None else ExactMatchScorer()
        self.max_new_tokens = max_new_tokens

    # ------------------------------------------------------------------ #
    # Baseline phase                                                       #
    # ------------------------------------------------------------------ #

    def run_baseline(self, tasks: List[TaskInstance]) -> List[TaskResult]:
        """Run baseline evaluation: always sends full context + question.

        Args:
            tasks: List of TaskInstance objects to evaluate.

        Returns:
            List of TaskResult with restore_mode="cold" (baseline never warms).
        """
        results: List[TaskResult] = []
        for task in tasks:
            prompt = self._build_full_prompt(task)
            input_tokens = _count_tokens(prompt)

            logger.info(
                "Baseline | %s | ctx=%d | tokens=%d",
                task.task_name,
                task.context_length,
                input_tokens,
            )

            answer, latency = self._call(prompt)
            output_tokens = _count_tokens(answer)
            score = self._score(answer, task.answer)

            result = TaskResult(
                task_id=task.task_id or task.task_name,
                task_name=task.task_name,
                context_length=task.context_length,
                restore_mode="cold",  # Baseline is always cold (no snapshots)
                baseline_ttft_s=latency,
                baseline_input_tokens=input_tokens,
                baseline_output_tokens=output_tokens,
                baseline_answer=answer,
                baseline_score=score,
                # Engram fields not applicable in baseline-only mode; use sentinels
                engram_ttft_s=0.0,
                engram_input_tokens=0,
                engram_output_tokens=0,
                engram_answer="",
                engram_score=0.0,
            )
            results.append(result)

        return results

    # ------------------------------------------------------------------ #
    # Engram phase                                                         #
    # ------------------------------------------------------------------ #

    def run_engram(self, tasks: List[TaskInstance]) -> List[TaskResult]:
        """Run Engram evaluation with warm/cold snapshot logic.

        Warm path (snapshot present): send only the question.
        Cold path (no snapshot): send full context + question, then save
        a stub snapshot so the next call with the same task is warm.

        Args:
            tasks: List of TaskInstance objects to evaluate.

        Returns:
            List of TaskResult with restore_mode set to "warm" or "cold".
        """
        results: List[TaskResult] = []
        for task in tasks:
            restore_mode, prompt, input_tokens = self._engram_prompt(task)

            logger.info(
                "Engram | %s | ctx=%d | mode=%s | tokens=%d",
                task.task_name,
                task.context_length,
                restore_mode,
                input_tokens,
            )

            answer, latency = self._call(prompt)
            output_tokens = _count_tokens(answer)
            score = self._score(answer, task.answer)

            # If cold, save snapshot so subsequent runs are warm
            if restore_mode == "cold":
                self._save_snapshot(task)

            result = TaskResult(
                task_id=task.task_id or task.task_name,
                task_name=task.task_name,
                context_length=task.context_length,
                restore_mode=restore_mode,
                # Baseline fields filled with sentinels (Engram-only run)
                baseline_ttft_s=0.0,
                baseline_input_tokens=_count_tokens(self._build_full_prompt(task)),
                baseline_output_tokens=0,
                baseline_answer="",
                baseline_score=0.0,
                engram_ttft_s=latency,
                engram_input_tokens=input_tokens,
                engram_output_tokens=output_tokens,
                engram_answer=answer,
                engram_score=score,
            )
            results.append(result)

        return results

    # ------------------------------------------------------------------ #
    # Combined phase (both baseline and Engram in one pass)                #
    # ------------------------------------------------------------------ #

    def run_both(self, tasks: List[TaskInstance]) -> List[TaskResult]:
        """Run both baseline and Engram for each task, merging into one result.

        First runs baseline (always cold), then Engram (warm/cold based on
        snapshot state).  Merges baseline and Engram fields into a single
        TaskResult per task.
        """
        results: List[TaskResult] = []

        for task in tasks:
            # --- Baseline ---
            baseline_prompt = self._build_full_prompt(task)
            baseline_input_tokens = _count_tokens(baseline_prompt)
            baseline_answer, baseline_latency = self._call(baseline_prompt)
            baseline_output_tokens = _count_tokens(baseline_answer)
            baseline_score = self._score(baseline_answer, task.answer)

            # --- Engram ---
            restore_mode, engram_prompt, engram_input_tokens = self._engram_prompt(task)
            engram_answer, engram_latency = self._call(engram_prompt)
            engram_output_tokens = _count_tokens(engram_answer)
            engram_score = self._score(engram_answer, task.answer)

            if restore_mode == "cold":
                self._save_snapshot(task)

            logger.info(
                "Both | %s | ctx=%d | mode=%s | baseline_tok=%d | engram_tok=%d",
                task.task_name,
                task.context_length,
                restore_mode,
                baseline_input_tokens,
                engram_input_tokens,
            )

            result = TaskResult(
                task_id=task.task_id or task.task_name,
                task_name=task.task_name,
                context_length=task.context_length,
                restore_mode=restore_mode,
                baseline_ttft_s=baseline_latency,
                baseline_input_tokens=baseline_input_tokens,
                baseline_output_tokens=baseline_output_tokens,
                baseline_answer=baseline_answer,
                baseline_score=baseline_score,
                engram_ttft_s=engram_latency,
                engram_input_tokens=engram_input_tokens,
                engram_output_tokens=engram_output_tokens,
                engram_answer=engram_answer,
                engram_score=engram_score,
            )
            results.append(result)

        return results

    # ------------------------------------------------------------------ #
    # Internal helpers                                                     #
    # ------------------------------------------------------------------ #

    def _snapshot_path(self, task: TaskInstance) -> Path:
        task_id = task.task_id or task.task_name
        return self.snapshot_dir / task_id / "snapshot.json"

    def _is_warm(self, task: TaskInstance) -> bool:
        return self._snapshot_path(task).exists()

    def _save_snapshot(self, task: TaskInstance) -> None:
        snapshot_path = self._snapshot_path(task)
        snapshot_path.parent.mkdir(parents=True, exist_ok=True)
        stub = {
            "task_id": task.task_id,
            "task_name": task.task_name,
            "context_length": task.context_length,
            "saved_at": time.time(),
            "_note": "Stub snapshot — replace with real Engram state blob for GPU runs.",
        }
        snapshot_path.write_text(json.dumps(stub, indent=2), encoding="utf-8")
        logger.debug("Saved snapshot: %s", snapshot_path)

    def _engram_prompt(
        self, task: TaskInstance
    ) -> Tuple[RestoreMode, str, int]:
        """Return (restore_mode, prompt_text, input_token_count)."""
        if self._is_warm(task):
            # Warm: snapshot exists — send only the question
            prompt = f"Question: {task.question}\nAnswer:"
            return "warm", prompt, _count_tokens(prompt)
        else:
            # Cold: no snapshot — send full context + question
            prompt = self._build_full_prompt(task)
            return "cold", prompt, _count_tokens(prompt)

    def _build_full_prompt(self, task: TaskInstance) -> str:
        return (
            f"{task.context_text}\n\nQuestion: {task.question}\nAnswer:"
        )

    def _call(self, prompt: str) -> Tuple[str, float]:
        """Invoke the HTTP client, returning (answer_text, latency_s)."""
        return self._http_client(self.model_url, prompt, self.model_name)

    def _score(self, answer: str, reference) -> float:
        return self.scorer.score(answer, reference)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m benchmark.ruler.runner",
        description="Run RULER benchmark against an Engram-compatible endpoint.",
    )
    parser.add_argument(
        "--model-url",
        default="http://localhost:30000",
        help="Base URL of the serving endpoint.",
    )
    parser.add_argument(
        "--model-name",
        default="model",
        help="Model name passed in the API payload.",
    )
    parser.add_argument(
        "--context-lengths",
        default="4096,8192",
        help="Comma-separated list of context lengths to benchmark.",
    )
    parser.add_argument(
        "--tasks",
        default=",".join(RULER_TASKS),
        help="Comma-separated list of RULER task names.",
    )
    parser.add_argument(
        "--output",
        default="ruler_results.jsonl",
        help="Output JSONL path.",
    )
    parser.add_argument(
        "--snapshot-dir",
        default="/tmp/ruler_snapshots",
        help="Directory for Engram snapshots.",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=5,
        help="Number of samples per (task, context_length) combination.",
    )
    parser.add_argument(
        "--mode",
        choices=["baseline", "engram", "both"],
        default="both",
        help="Evaluation mode.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=256,
        help="Max tokens to generate per request.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    )

    context_lengths = [int(x) for x in args.context_lengths.split(",")]
    task_names = [t.strip() for t in args.tasks.split(",") if t.strip()]

    # Validate task names
    unknown = [t for t in task_names if t not in RULER_TASKS]
    if unknown:
        logger.error("Unknown task names: %s", unknown)
        return 1

    # Generate task instances
    logger.info(
        "Generating %d task × %d context_length × %d samples instances…",
        len(task_names),
        len(context_lengths),
        args.num_samples,
    )
    task_instances: List[TaskInstance] = []
    for task_name in task_names:
        for ctx_len in context_lengths:
            for sample_idx in range(args.num_samples):
                task_instances.append(
                    generate_synthetic_task(task_name, ctx_len, sample_idx)
                )
    logger.info("Total instances: %d", len(task_instances))

    runner = RULERRunner(
        model_url=args.model_url,
        model_name=args.model_name,
        snapshot_dir=args.snapshot_dir,
        max_new_tokens=args.max_new_tokens,
    )

    summary = RunSummary(model=args.model_name)

    if args.mode == "baseline":
        results = runner.run_baseline(task_instances)
        summary.results = results
    elif args.mode == "engram":
        results = runner.run_engram(task_instances)
        summary.results = results
    else:  # both
        results = runner.run_both(task_instances)
        summary.results = results

    summary.to_jsonl(args.output)
    logger.info("Results written to %s", args.output)
    logger.info(
        "token_reduction=%.3f  ttft_speedup=%.3fx  compute_amortization=%.0f",
        summary.token_reduction,
        summary.ttft_speedup,
        summary.compute_amortization,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
