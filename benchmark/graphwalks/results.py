"""Results schema for the GraphWalks benchmark.

Every question result stores ``restore_mode`` as a Literal["warm", "cold"] so
that warm/cold splits are always preserved in serialised output and are never
inferred after the fact.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Literal, Optional


@dataclass
class QuestionResult:
    """Per-question benchmark result."""

    question_id: str
    graph_size: int  # number of nodes
    num_hops: int
    restore_mode: Literal["warm", "cold"]

    # Baseline (full context every turn, no snapshot)
    baseline_ttft_s: float
    baseline_input_tokens: int
    baseline_output_tokens: int
    baseline_answer: str
    baseline_score: float

    # Engram (snapshot-restored)
    engram_ttft_s: float
    engram_input_tokens: int
    engram_output_tokens: int
    engram_answer: str
    engram_score: float

    # ------------------------------------------------------------------
    # Derived metrics (read-only properties)
    # ------------------------------------------------------------------

    @property
    def token_reduction(self) -> float:
        """Fraction of input tokens saved versus baseline.

        token_reduction = 1 - engram_input_tokens / baseline_input_tokens

        A value of 0.80 means Engram sent 80% fewer tokens.
        Returns 0.0 when baseline_input_tokens is zero (degenerate case).
        """
        if self.baseline_input_tokens == 0:
            return 0.0
        return 1.0 - self.engram_input_tokens / self.baseline_input_tokens

    @property
    def ttft_speedup(self) -> float:
        """Ratio of baseline TTFT to Engram TTFT (>1 means Engram is faster).

        Returns 1.0 when engram_ttft_s is zero (degenerate case).
        """
        if self.engram_ttft_s == 0.0:
            return 1.0
        return self.baseline_ttft_s / self.engram_ttft_s

    @property
    def compute_amortization(self) -> float:
        """Fraction of prefill compute amortised across warm restores.

        Defined as token_reduction for warm results; 0.0 for cold (no
        amortisation possible if the snapshot was not already available).
        """
        if self.restore_mode == "warm":
            return self.token_reduction
        return 0.0

    # ------------------------------------------------------------------
    # Serialisation helpers
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        d = asdict(self)
        # Append computed metrics so they appear in serialised output.
        d["token_reduction"] = self.token_reduction
        d["ttft_speedup"] = self.ttft_speedup
        d["compute_amortization"] = self.compute_amortization
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "QuestionResult":
        # Drop computed keys that may be present in stored JSON.
        keep = {
            k for k in d
            if k not in ("token_reduction", "ttft_speedup", "compute_amortization")
        }
        return cls(**{k: d[k] for k in keep})


def _safe_mean(values: List[float]) -> Optional[float]:
    return sum(values) / len(values) if values else None


@dataclass
class RunSummary:
    """Aggregated results for a single benchmark run."""

    benchmark: str = "graphwalks"
    model: str = ""
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    results: List[QuestionResult] = field(default_factory=list)

    # ------------------------------------------------------------------
    # Warm / cold split helpers
    # ------------------------------------------------------------------

    @property
    def warm_results(self) -> List[QuestionResult]:
        return [r for r in self.results if r.restore_mode == "warm"]

    @property
    def cold_results(self) -> List[QuestionResult]:
        return [r for r in self.results if r.restore_mode == "cold"]

    # ------------------------------------------------------------------
    # Aggregate metrics (all results)
    # ------------------------------------------------------------------

    @property
    def token_reduction(self) -> Optional[float]:
        vals = [r.token_reduction for r in self.results]
        return _safe_mean(vals)

    @property
    def ttft_speedup(self) -> Optional[float]:
        vals = [r.ttft_speedup for r in self.results]
        return _safe_mean(vals)

    @property
    def compute_amortization(self) -> Optional[float]:
        vals = [r.compute_amortization for r in self.results]
        return _safe_mean(vals)

    # Warm-only aggregates
    @property
    def warm_token_reduction(self) -> Optional[float]:
        vals = [r.token_reduction for r in self.warm_results]
        return _safe_mean(vals)

    @property
    def warm_ttft_speedup(self) -> Optional[float]:
        vals = [r.ttft_speedup for r in self.warm_results]
        return _safe_mean(vals)

    # Cold-only aggregates
    @property
    def cold_token_reduction(self) -> Optional[float]:
        vals = [r.token_reduction for r in self.cold_results]
        return _safe_mean(vals)

    @property
    def cold_ttft_speedup(self) -> Optional[float]:
        vals = [r.ttft_speedup for r in self.cold_results]
        return _safe_mean(vals)

    # ------------------------------------------------------------------
    # Per-hop breakdown
    # ------------------------------------------------------------------

    def by_hop_count(self) -> Dict[int, List[QuestionResult]]:
        """Return results grouped by num_hops."""
        groups: Dict[int, List[QuestionResult]] = {}
        for r in self.results:
            groups.setdefault(r.num_hops, []).append(r)
        return groups

    def metrics_by_hop(self) -> Dict[int, dict]:
        """Aggregate metrics broken down by hop count."""
        breakdown: Dict[int, dict] = {}
        for hop, group in self.by_hop_count().items():
            breakdown[hop] = {
                "n": len(group),
                "token_reduction": _safe_mean([r.token_reduction for r in group]),
                "ttft_speedup": _safe_mean([r.ttft_speedup for r in group]),
                "compute_amortization": _safe_mean(
                    [r.compute_amortization for r in group]
                ),
                "baseline_score": _safe_mean([r.baseline_score for r in group]),
                "engram_score": _safe_mean([r.engram_score for r in group]),
            }
        return breakdown

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        return {
            "benchmark": self.benchmark,
            "model": self.model,
            "timestamp": self.timestamp,
            "n_total": len(self.results),
            "n_warm": len(self.warm_results),
            "n_cold": len(self.cold_results),
            "token_reduction": self.token_reduction,
            "ttft_speedup": self.ttft_speedup,
            "compute_amortization": self.compute_amortization,
            "warm_token_reduction": self.warm_token_reduction,
            "warm_ttft_speedup": self.warm_ttft_speedup,
            "cold_token_reduction": self.cold_token_reduction,
            "cold_ttft_speedup": self.cold_ttft_speedup,
            "metrics_by_hop": self.metrics_by_hop(),
            "results": [r.to_dict() for r in self.results],
        }

    def to_jsonl(self, path: "str | Path") -> None:
        """Write results to a JSONL file (one JSON object per line).

        The first line is the run-level summary header; subsequent lines are
        per-question results.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as fh:
            header = {k: v for k, v in self.to_dict().items() if k != "results"}
            fh.write(json.dumps(header) + "\n")
            for r in self.results:
                fh.write(json.dumps(r.to_dict()) + "\n")

    @classmethod
    def from_jsonl(cls, path: "str | Path") -> "RunSummary":
        """Load a RunSummary from a JSONL file written by ``to_jsonl``."""
        path = Path(path)
        lines = path.read_text(encoding="utf-8").splitlines()
        if not lines:
            raise ValueError(f"Empty JSONL file: {path}")
        header = json.loads(lines[0])
        results = [QuestionResult.from_dict(json.loads(ln)) for ln in lines[1:]]
        return cls(
            benchmark=header.get("benchmark", "graphwalks"),
            model=header.get("model", ""),
            timestamp=header.get("timestamp", ""),
            results=results,
        )
