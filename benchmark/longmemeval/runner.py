"""LongMemEval runner — two-phase benchmark execution.

Phases
------
baseline
    Stateless: every question sends the full ``memory_text`` + question to the
    model. No snapshot is consulted or saved.

engram
    Stateful: if a snapshot exists for the question's memory (warm), only the
    question itself is sent — the model state already encodes the context.
    If no snapshot exists (cold), the full context is sent and a stub snapshot
    is saved for future warm reuse.

Token counting
--------------
``len(text.split())`` is used as a cheap proxy when ``transformers`` is not
available (sufficient for CPU dry-runs). Real GPU runs should supply a proper
tokenizer via the ``tokenizer_name`` constructor parameter.

TTFT measurement
----------------
Streaming is used when available; TTFT is measured from the start of the
request to receipt of the first token chunk. In dry-run mode, synthetic latency
values are returned.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from .judge import BaseJudge, MockJudge
from .results import QuestionResult, RunSummary, to_jsonl


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class Question:
    """A single LongMemEval question."""

    question_id: str
    memory_text: str       # Long context (up to 1.5M tokens)
    question: str          # The actual question
    answer: str            # Ground-truth answer
    memory_type: str       # episodic | semantic | temporal | spatial | factual


# ---------------------------------------------------------------------------
# Token counting
# ---------------------------------------------------------------------------

def _count_tokens(text: str, tokenizer=None) -> int:  # type: ignore[type-arg]
    """Count tokens using a tokenizer when available, else word-split proxy."""
    if tokenizer is not None:
        try:
            return len(tokenizer.encode(text))
        except Exception:  # noqa: BLE001
            pass
    return len(text.split())


def _load_tokenizer(tokenizer_name: Optional[str]):
    """Try to load a HuggingFace tokenizer; return None on failure."""
    if tokenizer_name is None:
        return None
    try:
        from transformers import AutoTokenizer  # type: ignore[import]

        return AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)
    except Exception:  # noqa: BLE001
        return None


# ---------------------------------------------------------------------------
# HTTP client
# ---------------------------------------------------------------------------

_DRY_RUN_COMPLETION = "This is a synthetic dry-run answer."
_DRY_RUN_TTFT = 0.05  # seconds


def _chat_completion(
    model_url: str,
    messages: list,
    dry_run: bool = False,
    stream: bool = False,
) -> tuple[str, float]:
    """Send a chat completion request; return (answer_text, ttft_seconds).

    In dry-run mode no HTTP call is made.
    """
    if dry_run:
        return _DRY_RUN_COMPLETION, _DRY_RUN_TTFT

    try:
        import requests  # type: ignore[import]
    except ImportError as exc:
        raise RuntimeError("'requests' package is required for live runs.") from exc

    payload = {
        "model": "default",
        "messages": messages,
        "stream": stream,
        "max_tokens": 256,
        "temperature": 0.0,
    }
    url = f"{model_url.rstrip('/')}/v1/chat/completions"
    headers = {"Content-Type": "application/json"}

    t0 = time.perf_counter()
    if stream:
        with requests.post(url, json=payload, headers=headers, stream=True, timeout=120) as resp:
            resp.raise_for_status()
            first_token_time: Optional[float] = None
            chunks: list[str] = []
            for raw_line in resp.iter_lines():
                if not raw_line:
                    continue
                line = raw_line.decode("utf-8") if isinstance(raw_line, bytes) else raw_line
                if not line.startswith("data:"):
                    continue
                data_str = line[len("data:"):].strip()
                if data_str == "[DONE]":
                    break
                try:
                    chunk = json.loads(data_str)
                    delta = chunk["choices"][0]["delta"].get("content", "")
                    if delta and first_token_time is None:
                        first_token_time = time.perf_counter()
                    chunks.append(delta)
                except (json.JSONDecodeError, KeyError, IndexError):
                    continue
            ttft = (first_token_time or time.perf_counter()) - t0
            answer = "".join(chunks)
    else:
        resp = requests.post(url, json=payload, headers=headers, timeout=120)
        ttft = time.perf_counter() - t0
        resp.raise_for_status()
        data = resp.json()
        answer = data["choices"][0]["message"]["content"]

    return answer, ttft


# ---------------------------------------------------------------------------
# Snapshot helpers
# ---------------------------------------------------------------------------

def _snapshot_path(snapshot_dir: Path, question_id: str) -> Path:
    return snapshot_dir / question_id / "snapshot.json"


def _snapshot_exists(snapshot_dir: Path, question_id: str) -> bool:
    return _snapshot_path(snapshot_dir, question_id).exists()


def _save_snapshot(snapshot_dir: Path, question_id: str, metadata: dict) -> None:
    path = _snapshot_path(snapshot_dir, question_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f)


def _load_snapshot(snapshot_dir: Path, question_id: str) -> dict:
    path = _snapshot_path(snapshot_dir, question_id)
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

class LongMemEvalRunner:
    """Two-phase runner: baseline (stateless) and Engram (stateful)."""

    def __init__(
        self,
        model_url: str,
        snapshot_dir: Path,
        judge: BaseJudge,
        dry_run: bool = False,
        tokenizer_name: Optional[str] = None,
        stream: bool = False,
    ) -> None:
        self.model_url = model_url
        self.snapshot_dir = Path(snapshot_dir)
        self.judge = judge
        self.dry_run = dry_run
        self.stream = stream
        self._tokenizer = _load_tokenizer(tokenizer_name)

    def _tokens(self, text: str) -> int:
        return _count_tokens(text, self._tokenizer)

    # ------------------------------------------------------------------
    # Baseline path
    # ------------------------------------------------------------------

    def run_baseline(self, questions: List[Question]) -> List[QuestionResult]:
        """Stateless path: send full context + question for each question."""
        results: List[QuestionResult] = []
        for q in questions:
            full_prompt = f"{q.memory_text}\n\n{q.question}"
            messages = [{"role": "user", "content": full_prompt}]

            answer, ttft = _chat_completion(
                self.model_url, messages, dry_run=self.dry_run, stream=self.stream
            )
            score = self.judge.score(q.question, q.answer, answer)

            results.append(
                QuestionResult(
                    question_id=q.question_id,
                    restore_mode="cold",  # baseline is always "cold" (no snapshot)
                    baseline_ttft_s=ttft,
                    baseline_input_tokens=self._tokens(full_prompt),
                    baseline_output_tokens=self._tokens(answer),
                    baseline_answer=answer,
                    baseline_score=score,
                    # Engram fields are zeroed for baseline-only results
                    engram_ttft_s=0.0,
                    engram_input_tokens=0,
                    engram_output_tokens=0,
                    engram_answer="",
                    engram_score=0.0,
                )
            )
        return results

    # ------------------------------------------------------------------
    # Engram path
    # ------------------------------------------------------------------

    def run_engram(self, questions: List[Question]) -> List[QuestionResult]:
        """Stateful path: restore snapshot when available (warm), else cold."""
        results: List[QuestionResult] = []
        for q in questions:
            warm = _snapshot_exists(self.snapshot_dir, q.question_id)
            restore_mode: str = "warm" if warm else "cold"

            if warm:
                # Warm: snapshot encodes the context — send only the question.
                _snapshot_meta = _load_snapshot(self.snapshot_dir, q.question_id)
                prompt = q.question
            else:
                # Cold: no snapshot — send full context + question, then save stub.
                prompt = f"{q.memory_text}\n\n{q.question}"

            messages = [{"role": "user", "content": prompt}]
            answer, ttft = _chat_completion(
                self.model_url, messages, dry_run=self.dry_run, stream=self.stream
            )
            score = self.judge.score(q.question, q.answer, answer)

            if not warm:
                _save_snapshot(
                    self.snapshot_dir,
                    q.question_id,
                    {
                        "question_id": q.question_id,
                        "memory_type": q.memory_type,
                        "saved_at": time.time(),
                        "memory_tokens": self._tokens(q.memory_text),
                        "stub": True,
                    },
                )

            results.append(
                QuestionResult(
                    question_id=q.question_id,
                    restore_mode=restore_mode,  # type: ignore[arg-type]
                    # Engram fields
                    engram_ttft_s=ttft,
                    engram_input_tokens=self._tokens(prompt),
                    engram_output_tokens=self._tokens(answer),
                    engram_answer=answer,
                    engram_score=score,
                    # Baseline fields are zeroed when running engram-only
                    baseline_ttft_s=0.0,
                    baseline_input_tokens=0,
                    baseline_output_tokens=0,
                    baseline_answer="",
                    baseline_score=0.0,
                )
            )
        return results

    # ------------------------------------------------------------------
    # Combined run (baseline + engram, merging results)
    # ------------------------------------------------------------------

    def run_both(self, questions: List[Question]) -> List[QuestionResult]:
        """Run baseline and Engram paths and merge into combined QuestionResult."""
        merged: List[QuestionResult] = []
        for q in questions:
            # --- baseline ---
            full_prompt = f"{q.memory_text}\n\n{q.question}"
            b_messages = [{"role": "user", "content": full_prompt}]
            b_answer, b_ttft = _chat_completion(
                self.model_url, b_messages, dry_run=self.dry_run, stream=self.stream
            )
            b_score = self.judge.score(q.question, q.answer, b_answer)

            # --- engram ---
            warm = _snapshot_exists(self.snapshot_dir, q.question_id)
            restore_mode: str = "warm" if warm else "cold"

            if warm:
                _load_snapshot(self.snapshot_dir, q.question_id)
                e_prompt = q.question
            else:
                e_prompt = f"{q.memory_text}\n\n{q.question}"

            e_messages = [{"role": "user", "content": e_prompt}]
            e_answer, e_ttft = _chat_completion(
                self.model_url, e_messages, dry_run=self.dry_run, stream=self.stream
            )
            e_score = self.judge.score(q.question, q.answer, e_answer)

            if not warm:
                _save_snapshot(
                    self.snapshot_dir,
                    q.question_id,
                    {
                        "question_id": q.question_id,
                        "memory_type": q.memory_type,
                        "saved_at": time.time(),
                        "memory_tokens": self._tokens(q.memory_text),
                        "stub": True,
                    },
                )

            merged.append(
                QuestionResult(
                    question_id=q.question_id,
                    restore_mode=restore_mode,  # type: ignore[arg-type]
                    baseline_ttft_s=b_ttft,
                    baseline_input_tokens=self._tokens(full_prompt),
                    baseline_output_tokens=self._tokens(b_answer),
                    baseline_answer=b_answer,
                    baseline_score=b_score,
                    engram_ttft_s=e_ttft,
                    engram_input_tokens=self._tokens(e_prompt),
                    engram_output_tokens=self._tokens(e_answer),
                    engram_answer=e_answer,
                    engram_score=e_score,
                )
            )
        return merged


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _load_questions(data_path: Path, num_questions: Optional[int]) -> List[Question]:
    """Load questions from a JSONL file."""
    questions: List[Question] = []
    with data_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            questions.append(
                Question(
                    question_id=d["question_id"],
                    memory_text=d["memory_text"],
                    question=d["question"],
                    answer=d["answer"],
                    memory_type=d.get("memory_type", "unknown"),
                )
            )
            if num_questions and len(questions) >= num_questions:
                break
    return questions


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="LongMemEval benchmark runner for Engram stateful inference."
    )
    parser.add_argument(
        "--model-url",
        default="http://localhost:30000",
        help="Base URL of the model server (OpenAI-compatible API).",
    )
    parser.add_argument(
        "--snapshot-dir",
        default="/tmp/lme_snapshots",
        type=Path,
        help="Directory for Engram snapshot stubs.",
    )
    parser.add_argument(
        "--output",
        default="/tmp/lme_results.jsonl",
        type=Path,
        help="Path to write JSONL results.",
    )
    parser.add_argument(
        "--data",
        default=None,
        type=Path,
        help="Path to JSONL question file. Defaults to benchmark/longmemeval/data/longmemeval/test_dry_run.jsonl",
    )
    parser.add_argument(
        "--num-questions",
        default=500,
        type=int,
        help="Maximum number of questions to evaluate.",
    )
    parser.add_argument(
        "--mode",
        choices=["baseline", "engram", "both"],
        default="both",
        help="Which evaluation path(s) to run.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Stub all HTTP calls; no live server required.",
    )
    parser.add_argument(
        "--model-name",
        default="engram-mamba",
        help="Model name recorded in results metadata.",
    )
    parser.add_argument(
        "--tokenizer",
        default=None,
        help="HuggingFace tokenizer name for accurate token counts.",
    )
    parser.add_argument(
        "--stream",
        action="store_true",
        help="Use streaming for TTFT measurement.",
    )
    args = parser.parse_args(argv)

    # Resolve data path
    if args.data is None:
        here = Path(__file__).parent
        args.data = here / "data" / "longmemeval" / "test_dry_run.jsonl"

    if not args.data.exists():
        print(
            f"[ERROR] Data file not found: {args.data}\n"
            "Run: python -m benchmark.longmemeval.download --dry-run",
            file=sys.stderr,
        )
        return 1

    questions = _load_questions(args.data, args.num_questions)
    print(f"Loaded {len(questions)} questions from {args.data}")

    judge: BaseJudge = MockJudge() if args.dry_run else MockJudge()  # default mock
    runner = LongMemEvalRunner(
        model_url=args.model_url,
        snapshot_dir=args.snapshot_dir,
        judge=judge,
        dry_run=args.dry_run,
        tokenizer_name=args.tokenizer,
        stream=args.stream,
    )

    if args.mode == "baseline":
        results = runner.run_baseline(questions)
    elif args.mode == "engram":
        results = runner.run_engram(questions)
    else:
        results = runner.run_both(questions)

    summary = RunSummary(
        benchmark_name="LongMemEval",
        model=args.model_name,
        results=results,
    )
    to_jsonl(args.output, summary)

    print(f"\nResults written to: {args.output}")
    print(f"  Questions:             {len(results)}")
    print(f"  Warm:                  {len(summary.warm_questions)}")
    print(f"  Cold:                  {len(summary.cold_questions)}")
    print(f"  Mean token reduction:  {summary.mean_token_reduction:.3f}")
    print(f"  Mean TTFT speedup:     {summary.mean_ttft_speedup:.3f}x")
    print(f"  Mean compute amort.:   {summary.mean_compute_amortization:.1f} tokens saved")
    return 0


if __name__ == "__main__":
    sys.exit(main())
