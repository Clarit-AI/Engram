"""CPU / mock dry-run test suite for the LongMemEval harness.

Runs with no GPU, no live server, and no OpenAI API key.
Execute with:
    pytest benchmark/longmemeval/test_dry_run.py -v
"""

from __future__ import annotations

import json
import tempfile
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from benchmark.longmemeval.judge import MockJudge
from benchmark.longmemeval.results import (
    QuestionResult,
    RunSummary,
    from_jsonl,
    to_jsonl,
)
from benchmark.longmemeval.runner import LongMemEvalRunner, Question


# ---------------------------------------------------------------------------
# Synthetic dataset — 5 questions, no download needed
# ---------------------------------------------------------------------------

SYNTHETIC_QUESTIONS = [
    Question(
        question_id="syn_001",
        memory_text=(
            "On Monday, Alice told Bob that she had visited Paris last summer. "
            "She described the Eiffel Tower as magnificent and mentioned eating "
            "a croissant at a small café near the Louvre."
        ),
        question="What city did Alice visit last summer?",
        answer="Paris",
        memory_type="episodic",
    ),
    Question(
        question_id="syn_002",
        memory_text=(
            "The capital of France is Paris. The city is home to approximately "
            "2.1 million people within its city limits."
        ),
        question="What is the approximate population of Paris within its city limits?",
        answer="2.1 million",
        memory_type="factual",
    ),
    Question(
        question_id="syn_003",
        memory_text=(
            "The project kicked off on January 15, 2024. The final release "
            "was shipped on June 1, 2024."
        ),
        question="When was the final release shipped?",
        answer="June 1, 2024",
        memory_type="temporal",
    ),
    Question(
        question_id="syn_004",
        memory_text=(
            "The office building has four floors. Engineering occupies floor 2."
        ),
        question="Which floor does the Engineering team occupy?",
        answer="Floor 2",
        memory_type="spatial",
    ),
    Question(
        question_id="syn_005",
        memory_text=(
            "Mamba is a state-space model architecture that processes sequences "
            "in linear time."
        ),
        question="What is the time complexity of Mamba's sequence processing?",
        answer="Linear time",
        memory_type="semantic",
    ),
]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def tmp_snapshot_dir(tmp_path: Path) -> Path:
    return tmp_path / "snapshots"


@pytest.fixture
def dry_run_runner(tmp_snapshot_dir: Path) -> LongMemEvalRunner:
    return LongMemEvalRunner(
        model_url="http://localhost:30000",
        snapshot_dir=tmp_snapshot_dir,
        judge=MockJudge(),
        dry_run=True,
    )


def _make_result(
    question_id: str = "q1",
    restore_mode: str = "cold",
    baseline_ttft: float = 1.0,
    baseline_input: int = 1000,
    baseline_output: int = 50,
    engram_ttft: float = 0.5,
    engram_input: int = 10,
    engram_output: int = 50,
    baseline_score: float = 1.0,
    engram_score: float = 1.0,
) -> QuestionResult:
    return QuestionResult(
        question_id=question_id,
        restore_mode=restore_mode,  # type: ignore[arg-type]
        baseline_ttft_s=baseline_ttft,
        baseline_input_tokens=baseline_input,
        baseline_output_tokens=baseline_output,
        baseline_answer="baseline answer",
        baseline_score=baseline_score,
        engram_ttft_s=engram_ttft,
        engram_input_tokens=engram_input,
        engram_output_tokens=engram_output,
        engram_answer="engram answer",
        engram_score=engram_score,
    )


# ---------------------------------------------------------------------------
# TestResultsSchema
# ---------------------------------------------------------------------------

class TestResultsSchema:
    def test_question_result_warm_cold_label(self) -> None:
        r_warm = _make_result(restore_mode="warm")
        r_cold = _make_result(restore_mode="cold")
        assert r_warm.restore_mode == "warm"
        assert r_cold.restore_mode == "cold"

    def test_run_summary_aggregates(self) -> None:
        results = [
            _make_result("q1", "warm", baseline_input=1000, engram_input=10),
            _make_result("q2", "cold", baseline_input=500, engram_input=500),
        ]
        summary = RunSummary(benchmark_name="LongMemEval", model="test-model", results=results)

        assert len(summary.warm_questions) == 1
        assert len(summary.cold_questions) == 1

        # mean_token_reduction denominator is len(results) = 2
        # q1 reduction = (1000 - 10) / 1000 = 0.99
        # q2 reduction = (500 - 500) / 500 = 0.0
        expected_mean = (0.99 + 0.0) / 2
        assert abs(summary.mean_token_reduction - expected_mean) < 1e-6

    def test_jsonl_roundtrip(self, tmp_path: Path) -> None:
        results = [_make_result("q1", "warm"), _make_result("q2", "cold")]
        summary = RunSummary(benchmark_name="LongMemEval", model="m1", results=results)
        out = tmp_path / "out.jsonl"
        to_jsonl(out, summary)
        loaded = from_jsonl(out)

        assert loaded.benchmark_name == summary.benchmark_name
        assert loaded.model == summary.model
        assert len(loaded.results) == 2
        assert loaded.results[0].question_id == "q1"
        assert loaded.results[0].restore_mode == "warm"
        assert loaded.results[1].restore_mode == "cold"

    def test_token_reduction_metric(self) -> None:
        r = _make_result(baseline_input=1000, engram_input=100)
        assert abs(r.token_reduction - 0.9) < 1e-9

    def test_token_reduction_zero_baseline(self) -> None:
        r = _make_result(baseline_input=0, engram_input=0)
        assert r.token_reduction == 0.0

    def test_ttft_speedup_metric(self) -> None:
        r = _make_result(baseline_ttft=2.0, engram_ttft=0.5)
        assert abs(r.ttft_speedup - 4.0) < 1e-9

    def test_ttft_speedup_zero_engram(self) -> None:
        r = _make_result(baseline_ttft=1.0, engram_ttft=0.0)
        assert r.ttft_speedup == 1.0

    def test_compute_amortization_metric(self) -> None:
        r = _make_result(baseline_input=1000, engram_input=100)
        assert r.compute_amortization == 900.0

    def test_compute_amortization_no_saving(self) -> None:
        r = _make_result(baseline_input=500, engram_input=500)
        assert r.compute_amortization == 0.0


# ---------------------------------------------------------------------------
# TestMockJudge
# ---------------------------------------------------------------------------

class TestMockJudge:
    def test_mock_judge_scores_nonempty_answer(self) -> None:
        judge = MockJudge()
        score = judge.score("What is 2+2?", "4", "The answer is 4.")
        assert score == 1.0

    def test_mock_judge_scores_empty_answer_zero(self) -> None:
        judge = MockJudge()
        score = judge.score("What is 2+2?", "4", "")
        assert score == 0.0

    def test_mock_judge_scores_whitespace_answer_zero(self) -> None:
        judge = MockJudge()
        score = judge.score("What is 2+2?", "4", "   ")
        assert score == 0.0

    def test_mock_judge_batch(self) -> None:
        judge = MockJudge()
        items = [
            {"question": "q1", "reference": "r1", "answer": "non-empty"},
            {"question": "q2", "reference": "r2", "answer": ""},
        ]
        scores = judge.score_batch(items)
        assert scores == [1.0, 0.0]


# ---------------------------------------------------------------------------
# TestRunnerDryRun
# ---------------------------------------------------------------------------

class TestRunnerDryRun:
    def test_baseline_run(self, dry_run_runner: LongMemEvalRunner) -> None:
        results = dry_run_runner.run_baseline(SYNTHETIC_QUESTIONS)
        assert len(results) == len(SYNTHETIC_QUESTIONS)
        for r in results:
            assert r.restore_mode == "cold"
            assert r.baseline_ttft_s > 0
            assert r.baseline_input_tokens > 0
            assert r.baseline_answer != ""
            assert r.baseline_score == 1.0

    def test_engram_cold_run(
        self, dry_run_runner: LongMemEvalRunner, tmp_snapshot_dir: Path
    ) -> None:
        # No snapshots pre-exist → all cold
        results = dry_run_runner.run_engram(SYNTHETIC_QUESTIONS)
        assert len(results) == len(SYNTHETIC_QUESTIONS)
        for r in results:
            assert r.restore_mode == "cold"
            assert r.engram_ttft_s > 0
            assert r.engram_answer != ""

    def test_engram_warm_run(
        self, dry_run_runner: LongMemEvalRunner, tmp_snapshot_dir: Path
    ) -> None:
        # Pre-populate snapshot for syn_001
        snap_path = tmp_snapshot_dir / "syn_001" / "snapshot.json"
        snap_path.parent.mkdir(parents=True, exist_ok=True)
        snap_path.write_text(
            json.dumps({"question_id": "syn_001", "stub": True, "memory_tokens": 50})
        )

        q = SYNTHETIC_QUESTIONS[0]  # syn_001
        results = dry_run_runner.run_engram([q])
        assert results[0].restore_mode == "warm"
        # Warm: only the question is sent — fewer tokens than full context
        full_prompt_tokens = len((q.memory_text + "\n\n" + q.question).split())
        question_only_tokens = len(q.question.split())
        assert results[0].engram_input_tokens == question_only_tokens
        assert results[0].engram_input_tokens < full_prompt_tokens

    def test_warm_cold_labeling(
        self, dry_run_runner: LongMemEvalRunner, tmp_snapshot_dir: Path
    ) -> None:
        # Pre-populate snapshots for first 2 questions
        for q in SYNTHETIC_QUESTIONS[:2]:
            snap_path = tmp_snapshot_dir / q.question_id / "snapshot.json"
            snap_path.parent.mkdir(parents=True, exist_ok=True)
            snap_path.write_text(json.dumps({"question_id": q.question_id, "stub": True}))

        results = dry_run_runner.run_engram(SYNTHETIC_QUESTIONS)
        modes = [r.restore_mode for r in results]
        assert modes[:2] == ["warm", "warm"]
        assert all(m == "cold" for m in modes[2:])

    def test_result_serialization(
        self,
        dry_run_runner: LongMemEvalRunner,
        tmp_snapshot_dir: Path,
        tmp_path: Path,
    ) -> None:
        results = dry_run_runner.run_both(SYNTHETIC_QUESTIONS)
        summary = RunSummary(
            benchmark_name="LongMemEval",
            model="test-model",
            results=results,
        )
        out = tmp_path / "results.jsonl"
        to_jsonl(out, summary)
        loaded = from_jsonl(out)

        assert len(loaded.results) == len(SYNTHETIC_QUESTIONS)
        for orig, loaded_r in zip(results, loaded.results):
            assert orig.question_id == loaded_r.question_id
            assert orig.restore_mode == loaded_r.restore_mode
            assert abs(orig.baseline_ttft_s - loaded_r.baseline_ttft_s) < 1e-9
            assert abs(orig.engram_ttft_s - loaded_r.engram_ttft_s) < 1e-9

    def test_both_run_produces_merged_results(
        self, dry_run_runner: LongMemEvalRunner
    ) -> None:
        results = dry_run_runner.run_both(SYNTHETIC_QUESTIONS)
        assert len(results) == len(SYNTHETIC_QUESTIONS)
        for r in results:
            # Both baseline and engram fields must be populated
            assert r.baseline_ttft_s > 0
            assert r.engram_ttft_s > 0
            assert r.baseline_answer != ""
            assert r.engram_answer != ""

    def test_run_summary_metrics(self, dry_run_runner: LongMemEvalRunner) -> None:
        results = dry_run_runner.run_both(SYNTHETIC_QUESTIONS)
        summary = RunSummary(
            benchmark_name="LongMemEval",
            model="test-model",
            results=results,
        )
        # Dry-run: all cold (no snapshots), same mock answer for both paths
        # baseline_input > engram_input when cold (same full prompt sent)
        # so token_reduction should be 0 (inputs are equal)
        assert summary.mean_token_reduction >= 0.0
        assert summary.mean_ttft_speedup > 0.0
        assert summary.mean_compute_amortization >= 0.0
        assert len(summary.cold_questions) == len(SYNTHETIC_QUESTIONS)

    def test_snapshot_saved_after_cold(
        self, dry_run_runner: LongMemEvalRunner, tmp_snapshot_dir: Path
    ) -> None:
        q = SYNTHETIC_QUESTIONS[0]
        dry_run_runner.run_engram([q])
        snap = tmp_snapshot_dir / q.question_id / "snapshot.json"
        assert snap.exists()
        meta = json.loads(snap.read_text())
        assert meta["question_id"] == q.question_id
        assert meta["stub"] is True
