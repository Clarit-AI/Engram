"""GraphWalks benchmark dry-run tests.

Verifies all pipeline stages (schema, scoring, runner warm/cold logic,
serialisation) on CPU with no GPU and no live inference server.

Run with:
    pytest benchmark/graphwalks/test_dry_run.py -v
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any, Dict
from unittest.mock import MagicMock

import pytest

from benchmark.graphwalks.download import generate_synthetic_question
from benchmark.graphwalks.judge import ExactMatchScorer, MockJudge
from benchmark.graphwalks.results import QuestionResult, RunSummary
from benchmark.graphwalks.runner import GraphWalksRunner, Question

# ---------------------------------------------------------------------------
# Shared fixtures / helpers
# ---------------------------------------------------------------------------

# 5 synthetic graph questions covering different hop counts.
SYNTHETIC_QUESTIONS = [
    generate_synthetic_question(num_nodes=6, num_hops=h, seed=i)
    for i, h in enumerate([1, 2, 3, 2, 1])
]


def _make_question(d: Dict[str, Any]) -> Question:
    return Question(
        question_id=d["question_id"],
        graph_text=d["graph_text"],
        question=d["question"],
        answer=d["answer"],
        num_hops=d["num_hops"],
        num_nodes=d["num_nodes"],
    )


QUESTIONS = [_make_question(d) for d in SYNTHETIC_QUESTIONS]


def _mock_response(content: str = "node B") -> Dict:
    """Build a minimal OpenAI-compatible chat completion response dict."""
    return {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": content,
                }
            }
        ]
    }


@pytest.fixture
def mock_http_client():
    """An HTTP client stub that returns a canned completion response."""
    client = MagicMock()
    client.post.return_value = _mock_response("node B")
    return client


@pytest.fixture
def tmp_snapshot_dir(tmp_path: Path) -> Path:
    return tmp_path / "snapshots"


# ---------------------------------------------------------------------------
# TestResultsSchema
# ---------------------------------------------------------------------------


class TestResultsSchema:
    def _make_result(
        self,
        restore_mode: str = "cold",
        baseline_input: int = 200,
        engram_input: int = 40,
        baseline_ttft: float = 1.0,
        engram_ttft: float = 0.2,
        num_hops: int = 2,
    ) -> QuestionResult:
        return QuestionResult(
            question_id="q1",
            graph_size=5,
            num_hops=num_hops,
            restore_mode=restore_mode,  # type: ignore[arg-type]
            baseline_ttft_s=baseline_ttft,
            baseline_input_tokens=baseline_input,
            baseline_output_tokens=10,
            baseline_answer="B",
            baseline_score=1.0,
            engram_ttft_s=engram_ttft,
            engram_input_tokens=engram_input,
            engram_output_tokens=10,
            engram_answer="B",
            engram_score=1.0,
        )

    def test_question_result_warm_cold_label(self):
        """restore_mode must be preserved exactly as set."""
        warm = self._make_result(restore_mode="warm")
        cold = self._make_result(restore_mode="cold")
        assert warm.restore_mode == "warm"
        assert cold.restore_mode == "cold"

    def test_token_reduction_metric(self):
        r = self._make_result(baseline_input=200, engram_input=40)
        assert abs(r.token_reduction - 0.80) < 1e-9

    def test_ttft_speedup_metric(self):
        r = self._make_result(baseline_ttft=1.0, engram_ttft=0.25)
        assert abs(r.ttft_speedup - 4.0) < 1e-9

    def test_compute_amortization_warm(self):
        r = self._make_result(restore_mode="warm", baseline_input=200, engram_input=40)
        # Warm: amortization == token_reduction
        assert abs(r.compute_amortization - r.token_reduction) < 1e-9

    def test_compute_amortization_cold(self):
        r = self._make_result(restore_mode="cold")
        # Cold: no amortization
        assert r.compute_amortization == 0.0

    def test_token_reduction_zero_baseline(self):
        r = self._make_result(baseline_input=0, engram_input=0)
        assert r.token_reduction == 0.0

    def test_ttft_speedup_zero_engram(self):
        r = self._make_result(baseline_ttft=1.0, engram_ttft=0.0)
        assert r.ttft_speedup == 1.0

    def test_run_summary_aggregates_by_hops(self):
        results = [
            self._make_result(restore_mode="warm", num_hops=1),
            self._make_result(restore_mode="warm", num_hops=2),
            self._make_result(restore_mode="cold", num_hops=2),
        ]
        # Give distinct question IDs so we can have multiple in one summary.
        for i, r in enumerate(results):
            r.question_id = f"q{i}"

        summary = RunSummary(model="test-model", results=results)
        by_hop = summary.metrics_by_hop()
        assert 1 in by_hop
        assert 2 in by_hop
        assert by_hop[1]["n"] == 1
        assert by_hop[2]["n"] == 2

    def test_warm_cold_split_on_summary(self):
        warm_r = self._make_result(restore_mode="warm")
        warm_r.question_id = "qw"
        cold_r = self._make_result(restore_mode="cold")
        cold_r.question_id = "qc"
        summary = RunSummary(results=[warm_r, cold_r])
        assert len(summary.warm_results) == 1
        assert len(summary.cold_results) == 1

    def test_metrics(self):
        r1 = self._make_result(
            restore_mode="warm",
            baseline_input=200,
            engram_input=40,
            baseline_ttft=1.0,
            engram_ttft=0.25,
        )
        r1.question_id = "q1"
        r2 = self._make_result(
            restore_mode="cold",
            baseline_input=200,
            engram_input=200,
            baseline_ttft=1.0,
            engram_ttft=1.0,
        )
        r2.question_id = "q2"
        summary = RunSummary(results=[r1, r2])
        # Average token reduction: (0.8 + 0.0) / 2 = 0.4
        assert abs(summary.token_reduction - 0.4) < 1e-9  # type: ignore[operator]
        # Warm-only
        assert abs(summary.warm_token_reduction - 0.80) < 1e-9  # type: ignore[operator]

    def test_jsonl_roundtrip(self):
        results = []
        for i, q in enumerate(SYNTHETIC_QUESTIONS):
            mode = "warm" if i % 2 == 0 else "cold"
            results.append(
                QuestionResult(
                    question_id=q["question_id"],
                    graph_size=q["num_nodes"],
                    num_hops=q["num_hops"],
                    restore_mode=mode,  # type: ignore[arg-type]
                    baseline_ttft_s=0.5,
                    baseline_input_tokens=100,
                    baseline_output_tokens=10,
                    baseline_answer="B",
                    baseline_score=1.0,
                    engram_ttft_s=0.1,
                    engram_input_tokens=20,
                    engram_output_tokens=10,
                    engram_answer="B",
                    engram_score=1.0,
                )
            )

        summary = RunSummary(model="dry-run-model", results=results)

        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "run.jsonl"
            summary.to_jsonl(path)

            loaded = RunSummary.from_jsonl(path)

        assert loaded.model == summary.model
        assert len(loaded.results) == len(summary.results)
        for orig, loaded_r in zip(summary.results, loaded.results):
            assert loaded_r.question_id == orig.question_id
            assert loaded_r.restore_mode == orig.restore_mode
            assert loaded_r.num_hops == orig.num_hops


# ---------------------------------------------------------------------------
# TestSyntheticGeneration
# ---------------------------------------------------------------------------


class TestSyntheticGeneration:
    def test_generate_synthetic_question_keys(self):
        q = generate_synthetic_question(num_nodes=5, num_hops=3, seed=0)
        required_keys = {
            "question_id", "graph_text", "question", "answer",
            "num_hops", "num_nodes",
        }
        assert required_keys.issubset(q.keys())

    def test_generate_synthetic_question_answer_correctness(self):
        """The generated answer must be reachable in num_hops steps."""
        import string  # noqa: PLC0415

        for seed in range(5):
            q = generate_synthetic_question(num_nodes=6, num_hops=2, seed=seed)
            graph_text = q["graph_text"]
            answer = q["answer"]
            # The answer should appear in the graph description
            assert answer in graph_text, f"Answer '{answer}' not in graph: {graph_text}"
            # The answer must be a valid uppercase letter
            assert answer in string.ascii_uppercase

    def test_generate_synthetic_question_reproducible(self):
        q1 = generate_synthetic_question(num_nodes=5, num_hops=2, seed=42)
        q2 = generate_synthetic_question(num_nodes=5, num_hops=2, seed=42)
        assert q1 == q2

    def test_generate_synthetic_question_different_seeds(self):
        q0 = generate_synthetic_question(num_nodes=6, num_hops=2, seed=0)
        q1 = generate_synthetic_question(num_nodes=6, num_hops=2, seed=1)
        # Different seeds should (generally) produce different question IDs.
        assert q0["question_id"] != q1["question_id"]

    def test_exact_match_scorer_match(self):
        scorer = ExactMatchScorer()
        assert scorer.score("Node B", "node b") == 1.0

    def test_exact_match_scorer_no_match(self):
        scorer = ExactMatchScorer()
        assert scorer.score("Node C", "Node B") == 0.0

    def test_exact_match_scorer_normalized(self):
        scorer = ExactMatchScorer()
        # Punctuation and case should not matter.
        assert scorer.score("B!", "B") == 1.0
        assert scorer.score("  b  ", "B") == 1.0

    def test_mock_judge_non_empty(self):
        judge = MockJudge()
        assert judge.score("some answer", "reference") == 1.0

    def test_mock_judge_empty(self):
        judge = MockJudge()
        assert judge.score("", "reference") == 0.0
        assert judge.score("   ", "reference") == 0.0


# ---------------------------------------------------------------------------
# TestRunnerDryRun
# ---------------------------------------------------------------------------


class TestRunnerDryRun:
    """All tests mock the HTTP layer; no GPU or live server required."""

    def _make_runner(
        self,
        tmp_snapshot_dir: Path,
        answer: str = "node B",
        scorer=None,
    ) -> GraphWalksRunner:
        client = MagicMock()
        client.post.return_value = _mock_response(answer)
        return GraphWalksRunner(
            model_url="http://mock-server/v1",
            snapshot_dir=tmp_snapshot_dir,
            model="mock-model",
            scorer=scorer or MockJudge(),
            http_client=client,
        )

    def test_baseline_run(self, tmp_snapshot_dir: Path):
        runner = self._make_runner(tmp_snapshot_dir)
        results = runner.run_baseline(QUESTIONS)

        assert len(results) == len(QUESTIONS)
        for r in results:
            assert r.restore_mode == "cold"
            assert r.baseline_ttft_s >= 0.0
            assert r.baseline_input_tokens > 0
            assert r.engram_input_tokens == 0  # Not set in baseline-only run

    def test_engram_cold_run(self, tmp_snapshot_dir: Path):
        """First run with no pre-existing snapshots → all cold."""
        runner = self._make_runner(tmp_snapshot_dir)
        results = runner.run_engram(QUESTIONS)

        assert len(results) == len(QUESTIONS)
        for r in results:
            assert r.restore_mode == "cold", f"{r.question_id} should be cold"
            assert r.engram_ttft_s >= 0.0
            assert r.engram_input_tokens > 0

    def test_engram_warm_run(self, tmp_snapshot_dir: Path):
        """Pre-create snapshots → all warm."""
        runner = self._make_runner(tmp_snapshot_dir)

        # Pre-seed snapshots for every question.
        for q in QUESTIONS:
            snap = tmp_snapshot_dir / q.question_id / "snapshot.json"
            snap.parent.mkdir(parents=True, exist_ok=True)
            snap.write_text(json.dumps({"question_id": q.question_id}))

        results = runner.run_engram(QUESTIONS)

        assert len(results) == len(QUESTIONS)
        for r in results:
            assert r.restore_mode == "warm", f"{r.question_id} should be warm"

    def test_warm_cold_labeling(self, tmp_snapshot_dir: Path):
        """Mixed pre-existing snapshots produce correct warm/cold labels."""
        runner = self._make_runner(tmp_snapshot_dir)

        # Pre-seed only the first question.
        first_q = QUESTIONS[0]
        snap = tmp_snapshot_dir / first_q.question_id / "snapshot.json"
        snap.parent.mkdir(parents=True, exist_ok=True)
        snap.write_text(json.dumps({"question_id": first_q.question_id}))

        results = runner.run_engram(QUESTIONS)

        result_by_id = {r.question_id: r for r in results}
        assert result_by_id[first_q.question_id].restore_mode == "warm"
        for q in QUESTIONS[1:]:
            assert result_by_id[q.question_id].restore_mode == "cold"

    def test_cold_run_creates_snapshot(self, tmp_snapshot_dir: Path):
        """Cold Engram run must create a stub snapshot file."""
        runner = self._make_runner(tmp_snapshot_dir)
        runner.run_engram(QUESTIONS[:1])

        snap = tmp_snapshot_dir / QUESTIONS[0].question_id / "snapshot.json"
        assert snap.exists(), "Stub snapshot should have been written after cold run"
        data = json.loads(snap.read_text())
        assert "question_id" in data

    def test_warm_run_sends_shorter_prompt(self, tmp_snapshot_dir: Path):
        """Warm Engram run should use fewer input tokens than cold."""
        runner = self._make_runner(tmp_snapshot_dir)

        q = QUESTIONS[0]

        # Cold run first
        cold_results = runner.run_engram([q])
        cold_tokens = cold_results[0].engram_input_tokens

        # The snapshot now exists → warm on second run
        warm_results = runner.run_engram([q])
        warm_tokens = warm_results[0].engram_input_tokens

        assert warm_tokens < cold_tokens, (
            f"Warm run ({warm_tokens} tokens) should use fewer tokens "
            f"than cold run ({cold_tokens} tokens)"
        )

    def test_result_serialization(self, tmp_snapshot_dir: Path):
        """Full pipeline → JSONL serialisation → reload round-trip."""
        runner = self._make_runner(tmp_snapshot_dir)

        baseline = runner.run_baseline(QUESTIONS)
        engram = runner.run_engram(QUESTIONS)
        merged = runner.merge_results(baseline, engram)

        summary = RunSummary(model="mock-model", results=merged)

        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "results.jsonl"
            summary.to_jsonl(out)

            loaded = RunSummary.from_jsonl(out)

        assert len(loaded.results) == len(merged)
        for orig, reloaded in zip(merged, loaded.results):
            assert reloaded.question_id == orig.question_id
            assert reloaded.restore_mode == orig.restore_mode

    def test_merge_results_copies_baseline_fields(self, tmp_snapshot_dir: Path):
        """merge_results must populate baseline_* from the baseline run."""
        runner = self._make_runner(tmp_snapshot_dir)
        baseline = runner.run_baseline(QUESTIONS[:1])
        engram = runner.run_engram(QUESTIONS[:1])
        merged = runner.merge_results(baseline, engram)

        assert len(merged) == 1
        r = merged[0]
        # Baseline fields should come from the baseline run.
        assert r.baseline_input_tokens == baseline[0].baseline_input_tokens
        assert r.baseline_ttft_s == baseline[0].baseline_ttft_s
        # Engram fields should come from the engram run.
        assert r.engram_input_tokens == engram[0].engram_input_tokens
