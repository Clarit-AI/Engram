"""Results schema for the LongMemEval benchmark harness.

Every question result stores ``restore_mode`` (``"warm"`` | ``"cold"``),
reflecting the CS-161 warm/cold gate used by the Engram path.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Literal, Optional


@dataclass
class QuestionResult:
    """Per-question result comparing baseline vs Engram paths."""

    question_id: str
    restore_mode: Literal["warm", "cold"]

    # Baseline (stateless, full context every time)
    baseline_ttft_s: float
    baseline_input_tokens: int
    baseline_output_tokens: int
    baseline_answer: str
    baseline_score: float  # 0.0–1.0

    # Engram (stateful, warm restore skips context re-processing)
    engram_ttft_s: float
    engram_input_tokens: int
    engram_output_tokens: int
    engram_answer: str
    engram_score: float  # 0.0–1.0

    # --- per-question metrics ---

    @property
    def token_reduction(self) -> float:
        """Fraction of tokens saved by Engram vs baseline.

        Defined as ``(baseline_input - engram_input) / baseline_input``.
        Returns 0.0 when baseline_input_tokens is 0.
        """
        if self.baseline_input_tokens == 0:
            return 0.0
        return (self.baseline_input_tokens - self.engram_input_tokens) / self.baseline_input_tokens

    @property
    def ttft_speedup(self) -> float:
        """TTFT speedup ratio: ``baseline_ttft / engram_ttft``.

        Returns 1.0 when engram_ttft_s is 0 (no speedup measured).
        """
        if self.engram_ttft_s == 0.0:
            return 1.0
        return self.baseline_ttft_s / self.engram_ttft_s

    @property
    def compute_amortization(self) -> float:
        """Proxy for FLOPs-saved ratio.

        Defined as ``tokens_saved * 1.0`` (linear proxy for compute).
        Tokens saved = ``baseline_input_tokens - engram_input_tokens``.
        """
        return float(max(0, self.baseline_input_tokens - self.engram_input_tokens))

    def to_dict(self) -> dict:
        d = asdict(self)
        # Append computed metrics for convenience
        d["token_reduction"] = self.token_reduction
        d["ttft_speedup"] = self.ttft_speedup
        d["compute_amortization"] = self.compute_amortization
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "QuestionResult":
        # Strip computed fields that are not constructor params
        for key in ("token_reduction", "ttft_speedup", "compute_amortization"):
            d.pop(key, None)
        return cls(**d)


@dataclass
class RunSummary:
    """Aggregated run results across all questions."""

    benchmark_name: str
    model: str
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    results: List[QuestionResult] = field(default_factory=list)

    # --- aggregate properties ---

    @property
    def warm_questions(self) -> List[QuestionResult]:
        """Questions answered via warm snapshot restore."""
        return [r for r in self.results if r.restore_mode == "warm"]

    @property
    def cold_questions(self) -> List[QuestionResult]:
        """Questions answered via cold (full context) path."""
        return [r for r in self.results if r.restore_mode == "cold"]

    @property
    def mean_token_reduction(self) -> float:
        """Mean token_reduction across all questions (warm + cold).

        Denominator is ``len(results)`` regardless of warm/cold split.
        """
        if not self.results:
            return 0.0
        return sum(r.token_reduction for r in self.results) / len(self.results)

    @property
    def mean_ttft_speedup(self) -> float:
        """Mean TTFT speedup across all questions."""
        if not self.results:
            return 1.0
        return sum(r.ttft_speedup for r in self.results) / len(self.results)

    @property
    def mean_compute_amortization(self) -> float:
        """Mean compute_amortization (tokens saved proxy) across all questions."""
        if not self.results:
            return 0.0
        return sum(r.compute_amortization for r in self.results) / len(self.results)

    def to_dict(self) -> dict:
        return {
            "benchmark_name": self.benchmark_name,
            "model": self.model,
            "timestamp": self.timestamp,
            "num_results": len(self.results),
            "num_warm": len(self.warm_questions),
            "num_cold": len(self.cold_questions),
            "mean_token_reduction": self.mean_token_reduction,
            "mean_ttft_speedup": self.mean_ttft_speedup,
            "mean_compute_amortization": self.mean_compute_amortization,
            "results": [r.to_dict() for r in self.results],
        }

    @classmethod
    def from_dict(cls, d: dict) -> "RunSummary":
        results = [QuestionResult.from_dict(r) for r in d.get("results", [])]
        return cls(
            benchmark_name=d["benchmark_name"],
            model=d["model"],
            timestamp=d.get("timestamp", ""),
            results=results,
        )


# ---------------------------------------------------------------------------
# Serialization helpers
# ---------------------------------------------------------------------------

def to_jsonl(path: Path, summary: RunSummary) -> None:
    """Write a RunSummary as newline-delimited JSON.

    First line: summary header (no ``results`` list).
    Subsequent lines: one QuestionResult per line.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        # Header line (summary metadata, no nested results list)
        header = summary.to_dict()
        header.pop("results", None)
        f.write(json.dumps(header) + "\n")
        for result in summary.results:
            f.write(json.dumps(result.to_dict()) + "\n")


def from_jsonl(path: Path) -> RunSummary:
    """Read a RunSummary previously written by ``to_jsonl``."""
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

    if not lines:
        raise ValueError(f"Empty JSONL file: {path}")

    header = json.loads(lines[0])
    results: List[QuestionResult] = []
    for line in lines[1:]:
        results.append(QuestionResult.from_dict(json.loads(line)))

    return RunSummary(
        benchmark_name=header["benchmark_name"],
        model=header["model"],
        timestamp=header.get("timestamp", ""),
        results=results,
    )
