"""
Results schema for the RULER benchmark harness.

Every question result stores `restore_mode: Literal["warm", "cold"]`.
A token-reduction number with no warm/cold label is not publishable.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Literal, Optional


RestoreMode = Literal["warm", "cold"]


@dataclass
class TaskResult:
    """Per-task-instance result capturing both baseline and Engram measurements."""

    # Identity
    task_id: str
    task_name: str
    context_length: int
    restore_mode: RestoreMode  # warm = snapshot hit, cold = snapshot miss

    # Baseline (full context re-processed every turn)
    baseline_ttft_s: float
    baseline_input_tokens: int
    baseline_output_tokens: int
    baseline_answer: str
    baseline_score: float  # 0.0 – 1.0

    # Engram (snapshot-aware path)
    engram_ttft_s: float
    engram_input_tokens: int
    engram_output_tokens: int
    engram_answer: str
    engram_score: float  # 0.0 – 1.0

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "TaskResult":
        return cls(**d)


@dataclass
class RunSummary:
    """Aggregated run-level results across all tasks and context lengths."""

    benchmark: str = "ruler"
    model: str = ""
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    results: List[TaskResult] = field(default_factory=list)

    # ------------------------------------------------------------------ #
    # Core publishable metrics                                             #
    # ------------------------------------------------------------------ #

    @property
    def token_reduction(self) -> float:
        """1 - (engram_tokens / baseline_tokens) across all results.

        Positive means Engram sent fewer tokens.
        """
        baseline_total = sum(r.baseline_input_tokens for r in self.results)
        engram_total = sum(r.engram_input_tokens for r in self.results)
        if baseline_total == 0:
            return 0.0
        return 1.0 - (engram_total / baseline_total)

    @property
    def ttft_speedup(self) -> float:
        """Geometric mean of per-result (baseline_ttft / engram_ttft) ratios.

        >1.0 means Engram is faster.  Returns NaN if any engram_ttft is 0.
        """
        ratios = []
        for r in self.results:
            if r.engram_ttft_s <= 0:
                return math.nan
            ratios.append(r.baseline_ttft_s / r.engram_ttft_s)
        if not ratios:
            return 0.0
        log_sum = sum(math.log(x) for x in ratios if x > 0)
        return math.exp(log_sum / len(ratios))

    @property
    def compute_amortization(self) -> float:
        """Total tokens saved across all results (proxy for FLOPs amortized)."""
        return float(
            sum(
                max(0, r.baseline_input_tokens - r.engram_input_tokens)
                for r in self.results
            )
        )

    # ------------------------------------------------------------------ #
    # Breakdowns                                                           #
    # ------------------------------------------------------------------ #

    def by_task(self) -> Dict[str, List[TaskResult]]:
        """Group results by task_name."""
        groups: Dict[str, List[TaskResult]] = {}
        for r in self.results:
            groups.setdefault(r.task_name, []).append(r)
        return groups

    def by_context_length(self) -> Dict[int, List[TaskResult]]:
        """Group results by context_length."""
        groups: Dict[int, List[TaskResult]] = {}
        for r in self.results:
            groups.setdefault(r.context_length, []).append(r)
        return groups

    def by_restore_mode(self) -> Dict[RestoreMode, List[TaskResult]]:
        """Group results by restore_mode (warm vs cold)."""
        groups: Dict[RestoreMode, List[TaskResult]] = {"warm": [], "cold": []}
        for r in self.results:
            groups[r.restore_mode].append(r)
        return groups

    def warm_token_reduction(self) -> float:
        """token_reduction computed over warm hits only."""
        warm = self.by_restore_mode()["warm"]
        if not warm:
            return 0.0
        baseline_total = sum(r.baseline_input_tokens for r in warm)
        engram_total = sum(r.engram_input_tokens for r in warm)
        if baseline_total == 0:
            return 0.0
        return 1.0 - (engram_total / baseline_total)

    def cold_token_reduction(self) -> float:
        """token_reduction computed over cold misses only.

        Expected to be ~0 or slightly negative (overhead) for cold paths.
        """
        cold = self.by_restore_mode()["cold"]
        if not cold:
            return 0.0
        baseline_total = sum(r.baseline_input_tokens for r in cold)
        engram_total = sum(r.engram_input_tokens for r in cold)
        if baseline_total == 0:
            return 0.0
        return 1.0 - (engram_total / baseline_total)

    def task_score_summary(self) -> Dict[str, Dict[str, float]]:
        """Per-task average baseline and engram scores."""
        summary = {}
        for task_name, task_results in self.by_task().items():
            summary[task_name] = {
                "baseline_score": sum(r.baseline_score for r in task_results)
                / len(task_results),
                "engram_score": sum(r.engram_score for r in task_results)
                / len(task_results),
                "n": len(task_results),
            }
        return summary

    # ------------------------------------------------------------------ #
    # Serialization                                                        #
    # ------------------------------------------------------------------ #

    def to_dict(self) -> dict:
        return {
            "benchmark": self.benchmark,
            "model": self.model,
            "timestamp": self.timestamp,
            "results": [r.to_dict() for r in self.results],
            # Bake metrics into the summary dict for easy inspection
            "metrics": {
                "token_reduction": self.token_reduction,
                "ttft_speedup": self.ttft_speedup,
                "compute_amortization": self.compute_amortization,
                "warm_token_reduction": self.warm_token_reduction(),
                "cold_token_reduction": self.cold_token_reduction(),
            },
        }

    def to_jsonl(self, path: str | Path) -> None:
        """Write one JSON line per TaskResult, plus a summary line at the end."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            for r in self.results:
                f.write(json.dumps(r.to_dict()) + "\n")
            # Final line is the aggregated summary (no "results" list to avoid duplication)
            summary_line = {
                "benchmark": self.benchmark,
                "model": self.model,
                "timestamp": self.timestamp,
                "_record_type": "summary",
                "metrics": {
                    "token_reduction": self.token_reduction,
                    "ttft_speedup": self.ttft_speedup,
                    "compute_amortization": self.compute_amortization,
                    "warm_token_reduction": self.warm_token_reduction(),
                    "cold_token_reduction": self.cold_token_reduction(),
                },
                "task_score_summary": self.task_score_summary(),
            }
            f.write(json.dumps(summary_line) + "\n")

    @classmethod
    def from_jsonl(cls, path: str | Path) -> "RunSummary":
        """Load a RunSummary from a JSONL file written by to_jsonl()."""
        path = Path(path)
        results: List[TaskResult] = []
        summary_meta: Optional[dict] = None

        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                if record.get("_record_type") == "summary":
                    summary_meta = record
                else:
                    results.append(TaskResult.from_dict(record))

        obj = cls(results=results)
        if summary_meta:
            obj.benchmark = summary_meta.get("benchmark", "ruler")
            obj.model = summary_meta.get("model", "")
            obj.timestamp = summary_meta.get("timestamp", obj.timestamp)
        return obj
