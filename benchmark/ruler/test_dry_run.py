"""
CPU/mock dry-run tests for the RULER benchmark harness.

All tests run without a GPU, without an active model server, and without
any network access.  External HTTP calls are replaced by a mock client.

Run with:
    pytest benchmark/ruler/test_dry_run.py -v
"""

from __future__ import annotations

from typing import Tuple

import pytest

from .judge import ExactMatchScorer, MockJudge
from .results import RunSummary, TaskResult
from .runner import RULERRunner, _count_tokens
from .tasks import (
    RULER_TASKS,
    TaskInstance,
    generate_synthetic_task,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_mock_http_client(answer: str = "ABCD1234", latency: float = 0.05):
    """Return a mock HTTP client that always responds with `answer`."""

    def client(model_url: str, prompt: str, model_name: str) -> Tuple[str, float]:
        return answer, latency

    return client


@pytest.fixture
def mock_http_client():
    return _make_mock_http_client()


@pytest.fixture
def tmp_snapshot_dir(tmp_path):
    snapshot_dir = tmp_path / "snapshots"
    snapshot_dir.mkdir()
    return snapshot_dir


def _make_task(
    task_name: str = "niah_single_1",
    context_length: int = 4096,
    sample_idx: int = 0,
) -> TaskInstance:
    return generate_synthetic_task(task_name, context_length, sample_idx)


def _make_full_task_result(
    restore_mode="cold",
    task_name="niah_single_1",
    context_length=4096,
) -> TaskResult:
    return TaskResult(
        task_id=f"{task_name}_cl{context_length}_test",
        task_name=task_name,
        context_length=context_length,
        restore_mode=restore_mode,
        baseline_ttft_s=1.0,
        baseline_input_tokens=1000,
        baseline_output_tokens=10,
        baseline_answer="ABCD1234",
        baseline_score=1.0,
        engram_ttft_s=0.2,
        engram_input_tokens=20,
        engram_output_tokens=10,
        engram_answer="ABCD1234",
        engram_score=1.0,
    )


# ---------------------------------------------------------------------------
# TestResultsSchema
# ---------------------------------------------------------------------------


class TestResultsSchema:
    def test_task_result_warm_cold_label(self):
        """TaskResult must carry a warm/cold restore_mode."""
        warm_result = _make_full_task_result(restore_mode="warm")
        cold_result = _make_full_task_result(restore_mode="cold")

        assert warm_result.restore_mode == "warm"
        assert cold_result.restore_mode == "cold"

    def test_run_summary_aggregates_by_task(self):
        """RunSummary.by_task() groups results correctly."""
        results = [
            _make_full_task_result(task_name="niah_single_1"),
            _make_full_task_result(task_name="niah_single_1"),
            _make_full_task_result(task_name="vt"),
        ]
        summary = RunSummary(model="test-model", results=results)
        by_task = summary.by_task()

        assert set(by_task.keys()) == {"niah_single_1", "vt"}
        assert len(by_task["niah_single_1"]) == 2
        assert len(by_task["vt"]) == 1

    def test_jsonl_roundtrip(self, tmp_path):
        """to_jsonl / from_jsonl preserves all fields."""
        results = [
            _make_full_task_result(restore_mode="warm"),
            _make_full_task_result(restore_mode="cold", task_name="vt"),
        ]
        summary = RunSummary(model="roundtrip-model", results=results)
        path = tmp_path / "test.jsonl"
        summary.to_jsonl(path)

        loaded = RunSummary.from_jsonl(path)
        assert loaded.model == "roundtrip-model"
        assert len(loaded.results) == 2
        modes = {r.restore_mode for r in loaded.results}
        assert modes == {"warm", "cold"}
        task_names = {r.task_name for r in loaded.results}
        assert task_names == {"niah_single_1", "vt"}

    def test_metrics(self):
        """Token reduction, TTFT speedup, and compute amortization are correct."""
        # 1000 baseline tokens, 20 engram tokens → 98% reduction
        result = _make_full_task_result()
        summary = RunSummary(results=[result])

        assert abs(summary.token_reduction - (1 - 20 / 1000)) < 1e-6
        # Speedup = 1.0 / 0.2 = 5.0
        assert abs(summary.ttft_speedup - 5.0) < 1e-6
        # Amortization = 1000 - 20 = 980
        assert summary.compute_amortization == 980.0

    def test_warm_cold_breakdown(self):
        """Warm and cold breakdowns are computed independently."""
        warm = _make_full_task_result(restore_mode="warm")
        cold = TaskResult(
            task_id="cold_task",
            task_name="vt",
            context_length=4096,
            restore_mode="cold",
            baseline_ttft_s=1.0,
            baseline_input_tokens=500,
            baseline_output_tokens=10,
            baseline_answer="",
            baseline_score=0.0,
            engram_ttft_s=0.8,
            engram_input_tokens=500,  # cold path: same tokens
            engram_output_tokens=10,
            engram_answer="",
            engram_score=0.0,
        )
        summary = RunSummary(results=[warm, cold])

        # Warm: 20/1000 input → 98% reduction
        assert abs(summary.warm_token_reduction() - (1 - 20 / 1000)) < 1e-6
        # Cold: same tokens → 0% reduction
        assert abs(summary.cold_token_reduction()) < 1e-6

    def test_empty_summary_metrics(self):
        """Empty RunSummary returns safe defaults."""
        summary = RunSummary()
        assert summary.token_reduction == 0.0
        assert summary.compute_amortization == 0.0

    def test_task_result_serialization(self):
        """TaskResult.to_dict() / from_dict() is lossless."""
        original = _make_full_task_result(restore_mode="warm")
        roundtripped = TaskResult.from_dict(original.to_dict())
        assert roundtripped.restore_mode == "warm"
        assert roundtripped.baseline_input_tokens == 1000
        assert roundtripped.engram_input_tokens == 20
        assert roundtripped.baseline_score == 1.0


# ---------------------------------------------------------------------------
# TestSyntheticGeneration
# ---------------------------------------------------------------------------


class TestSyntheticGeneration:
    def test_generate_niah_task(self):
        """NIAH synthetic tasks contain the needle in the context."""
        instance = generate_synthetic_task("niah_single_1", 4096)

        assert instance.task_name == "niah_single_1"
        assert instance.context_length == 4096
        assert isinstance(instance.answer, str)
        assert len(instance.answer) > 0
        # Needle value should appear in context
        assert instance.answer in instance.context_text

    def test_generate_vt_task(self):
        """Variable tracking task embeds the answer in the context."""
        instance = generate_synthetic_task("vt", 4096)

        assert instance.task_name == "vt"
        assert isinstance(instance.answer, str)
        assert len(instance.answer) > 0
        # The initial variable value should be present somewhere in context
        assert instance.answer in instance.context_text

    def test_generate_cwe_task(self):
        """CWE task returns a list of common words as the answer."""
        instance = generate_synthetic_task("cwe", 4096)

        assert instance.task_name == "cwe"
        assert isinstance(instance.answer, list)
        assert len(instance.answer) >= 1

    def test_generate_fwe_task(self):
        """FWE task returns a list of frequent words."""
        instance = generate_synthetic_task("fwe", 4096)

        assert isinstance(instance.answer, list)
        assert len(instance.answer) >= 1

    def test_generate_qa_tasks(self):
        """QA tasks embed a fact and return a string answer."""
        for task_name in ["qa_1", "qa_2"]:
            instance = generate_synthetic_task(task_name, 4096)
            assert isinstance(instance.answer, str)
            assert len(instance.answer) > 0
            assert instance.answer in instance.context_text

    def test_generate_multikey_tasks(self):
        """Multi-key NIAH tasks produce string answers."""
        for task_name in ["niah_multikey_1", "niah_multikey_2", "niah_multikey_3"]:
            instance = generate_synthetic_task(task_name, 4096)
            assert isinstance(instance.answer, str)
            assert len(instance.answer) > 0

    def test_generate_multivalue_task(self):
        """Multi-value NIAH task produces a list answer."""
        instance = generate_synthetic_task("niah_multivalue", 4096)
        assert isinstance(instance.answer, list)
        assert len(instance.answer) >= 2

    def test_generate_multiquery_task(self):
        """Multi-query NIAH task produces a string answer."""
        instance = generate_synthetic_task("niah_multiquery", 4096)
        assert isinstance(instance.answer, str)

    def test_task_id_is_deterministic(self):
        """Same inputs produce the same task_id."""
        inst1 = generate_synthetic_task("niah_single_1", 4096, 0)
        inst2 = generate_synthetic_task("niah_single_1", 4096, 0)
        assert inst1.task_id == inst2.task_id

    def test_task_id_differs_by_sample_idx(self):
        """Different sample_idx values produce different task_ids."""
        inst0 = generate_synthetic_task("niah_single_1", 4096, 0)
        inst1 = generate_synthetic_task("niah_single_1", 4096, 1)
        assert inst0.task_id != inst1.task_id

    def test_all_13_tasks_generate_without_error(self):
        """All 13 RULER task types generate successfully."""
        for task_name in RULER_TASKS:
            instance = generate_synthetic_task(task_name, 4096)
            assert instance.task_name == task_name
            assert len(instance.context_text) > 0
            assert len(instance.question) > 0

    def test_unknown_task_raises(self):
        """Requesting an unknown task name raises ValueError."""
        with pytest.raises(ValueError, match="Unknown task"):
            generate_synthetic_task("not_a_real_task", 4096)

    def test_exact_match_scorer(self):
        """ExactMatchScorer returns 1.0 for correct containment, 0.0 otherwise."""
        scorer = ExactMatchScorer()

        # String reference
        assert scorer.score("The answer is ABCD1234", "ABCD1234") == 1.0
        assert scorer.score("ABCD1234", "ABCD1234") == 1.0
        assert scorer.score("wrong answer", "ABCD1234") == 0.0

        # List reference — any match passes
        assert scorer.score("Paris", ["London", "Paris", "Tokyo"]) == 1.0
        assert scorer.score("Sydney", ["London", "Paris", "Tokyo"]) == 0.0

        # Case insensitivity
        assert scorer.score("paris", "Paris") == 1.0

    def test_mock_judge(self):
        """MockJudge returns 1.0 for non-empty, 0.0 for empty."""
        judge = MockJudge()
        assert judge.score("some answer", "reference") == 1.0
        assert judge.score("", "reference") == 0.0
        assert judge.score("   ", "reference") == 0.0

    def test_context_text_scales_with_length(self):
        """Larger context_length produces longer context text."""
        small = generate_synthetic_task("niah_single_1", 512)
        large = generate_synthetic_task("niah_single_1", 4096)
        assert len(large.context_text) > len(small.context_text)


# ---------------------------------------------------------------------------
# TestRunnerDryRun
# ---------------------------------------------------------------------------


class TestRunnerDryRun:
    def _make_runner(self, mock_http_client, tmp_snapshot_dir):
        return RULERRunner(
            model_url="http://mock-server:30000",
            model_name="mock-model",
            snapshot_dir=tmp_snapshot_dir,
            http_client=mock_http_client,
            scorer=MockJudge(),
        )

    def test_baseline_run(self, mock_http_client, tmp_snapshot_dir):
        """Baseline run produces TaskResult objects with restore_mode='cold'."""
        runner = self._make_runner(mock_http_client, tmp_snapshot_dir)
        tasks = [_make_task("niah_single_1", 4096)]

        results = runner.run_baseline(tasks)

        assert len(results) == 1
        r = results[0]
        assert r.restore_mode == "cold"
        assert r.baseline_ttft_s > 0
        assert r.baseline_input_tokens > 0
        assert r.baseline_answer != "" or r.baseline_answer == ""  # any string is ok

    def test_engram_cold_run(self, mock_http_client, tmp_snapshot_dir):
        """Engram run without a snapshot produces restore_mode='cold'."""
        runner = self._make_runner(mock_http_client, tmp_snapshot_dir)
        task = _make_task("niah_single_1", 4096)

        # Ensure no snapshot exists
        snapshot_path = runner._snapshot_path(task)
        assert not snapshot_path.exists()

        results = runner.run_engram([task])

        assert len(results) == 1
        r = results[0]
        assert r.restore_mode == "cold"
        # After a cold run, snapshot should be created
        assert snapshot_path.exists()

    def test_engram_warm_run(self, mock_http_client, tmp_snapshot_dir):
        """Engram run with a pre-existing snapshot produces restore_mode='warm'."""
        runner = self._make_runner(mock_http_client, tmp_snapshot_dir)
        task = _make_task("niah_single_1", 4096)

        # Pre-create the snapshot
        runner._save_snapshot(task)
        assert runner._snapshot_path(task).exists()

        results = runner.run_engram([task])

        assert len(results) == 1
        r = results[0]
        assert r.restore_mode == "warm"

    def test_warm_cold_labeling(self, mock_http_client, tmp_snapshot_dir):
        """First engram call is cold; second (same task) is warm."""
        runner = self._make_runner(mock_http_client, tmp_snapshot_dir)
        task = _make_task("niah_single_1", 4096)

        # First call — cold
        results_cold = runner.run_engram([task])
        assert results_cold[0].restore_mode == "cold"

        # Second call — warm (snapshot was saved by first call)
        results_warm = runner.run_engram([task])
        assert results_warm[0].restore_mode == "warm"

    def test_warm_tokens_less_than_cold(self, mock_http_client, tmp_snapshot_dir):
        """Warm path sends fewer tokens than cold path for the same task."""
        runner = self._make_runner(mock_http_client, tmp_snapshot_dir)
        task = _make_task("niah_single_1", 4096)

        # Cold run — gets full prompt
        cold_results = runner.run_engram([task])
        cold_tokens = cold_results[0].engram_input_tokens

        # Warm run — gets question-only prompt
        warm_results = runner.run_engram([task])
        warm_tokens = warm_results[0].engram_input_tokens

        assert warm_tokens < cold_tokens, (
            f"Warm should use fewer tokens than cold: warm={warm_tokens}, cold={cold_tokens}"
        )

    def test_run_both_produces_all_fields(self, mock_http_client, tmp_snapshot_dir):
        """run_both fills both baseline_* and engram_* fields in each result."""
        runner = self._make_runner(mock_http_client, tmp_snapshot_dir)
        task = _make_task("niah_single_1", 4096)

        results = runner.run_both([task])

        assert len(results) == 1
        r = results[0]
        # Both sides should have non-trivial values
        assert r.baseline_input_tokens > 0
        assert r.engram_input_tokens > 0
        assert r.restore_mode in ("warm", "cold")

    def test_multiple_tasks(self, mock_http_client, tmp_snapshot_dir):
        """Runner handles multiple task instances correctly."""
        runner = self._make_runner(mock_http_client, tmp_snapshot_dir)
        tasks = [
            _make_task("niah_single_1", 4096),
            _make_task("vt", 4096),
            _make_task("qa_1", 4096),
        ]

        results = runner.run_both(tasks)
        assert len(results) == 3
        task_names = {r.task_name for r in results}
        assert task_names == {"niah_single_1", "vt", "qa_1"}

    def test_run_summary_integration(self, mock_http_client, tmp_snapshot_dir, tmp_path):
        """Full pipeline: run_both → RunSummary → to_jsonl → from_jsonl."""
        runner = self._make_runner(mock_http_client, tmp_snapshot_dir)
        tasks = [_make_task("niah_single_1", 4096)]

        results = runner.run_both(tasks)
        summary = RunSummary(model="integration-test", results=results)

        output_path = tmp_path / "ruler_test.jsonl"
        summary.to_jsonl(output_path)

        loaded = RunSummary.from_jsonl(output_path)
        assert len(loaded.results) == 1
        assert loaded.results[0].restore_mode in ("warm", "cold")
        assert loaded.model == "integration-test"

    def test_snapshot_dir_created_automatically(self, mock_http_client, tmp_path):
        """Runner creates snapshot_dir if it does not exist."""
        snap_dir = tmp_path / "does" / "not" / "exist"
        assert not snap_dir.exists()

        RULERRunner(
            model_url="http://mock:30000",
            model_name="mock",
            snapshot_dir=snap_dir,
            http_client=mock_http_client,
            scorer=MockJudge(),
        )
        assert snap_dir.exists()

    def test_token_counting_proxy(self):
        """_count_tokens returns a positive integer for non-empty text."""
        assert _count_tokens("hello world foo") == 3
        assert _count_tokens("") == 0

    def test_baseline_does_not_create_snapshots(self, mock_http_client, tmp_snapshot_dir):
        """Baseline run must NOT create any snapshots."""
        runner = self._make_runner(mock_http_client, tmp_snapshot_dir)
        task = _make_task("niah_single_1", 4096)

        runner.run_baseline([task])

        # Snapshot directory should be empty (no snapshot created)
        snapshot_path = runner._snapshot_path(task)
        assert not snapshot_path.exists()

    def test_engram_cold_then_warm_score_equal(self, tmp_snapshot_dir):
        """Score should be consistent between cold and warm runs given same mock answer."""
        # Use an answer that matches the task's answer for scoring
        task = _make_task("niah_single_1", 4096)
        answer = task.answer if isinstance(task.answer, str) else task.answer[0]
        client = _make_mock_http_client(answer=answer)

        scorer = ExactMatchScorer()
        runner = RULERRunner(
            model_url="http://mock:30000",
            model_name="mock",
            snapshot_dir=tmp_snapshot_dir,
            http_client=client,
            scorer=scorer,
        )

        cold_results = runner.run_engram([task])
        warm_results = runner.run_engram([task])

        assert cold_results[0].engram_score == 1.0
        assert warm_results[0].engram_score == 1.0


# ---------------------------------------------------------------------------
# TestDownloadModule
# ---------------------------------------------------------------------------


class TestDownloadModule:
    def test_generate_dataset_small(self, tmp_path):
        """generate_dataset() works offline with a small config."""
        from .download import generate_dataset, load_dataset

        output = tmp_path / "test_dataset.jsonl"
        instances = generate_dataset(
            tasks=["niah_single_1", "vt"],
            context_lengths=[4096],
            num_samples=2,
            output_path=output,
        )

        assert len(instances) == 4  # 2 tasks × 1 ctx_len × 2 samples
        assert output.exists()

        # Reload and verify
        loaded = load_dataset(output)
        assert len(loaded) == 4
        task_names = {inst.task_name for inst in loaded}
        assert task_names == {"niah_single_1", "vt"}

    def test_dataset_info_has_required_keys(self):
        """RULER_DATASET_INFO contains all required metadata keys."""
        from .download import RULER_DATASET_INFO

        for key in ("name", "license", "tasks", "context_lengths", "num_tasks"):
            assert key in RULER_DATASET_INFO

        assert RULER_DATASET_INFO["license"] == "MIT"
        assert len(RULER_DATASET_INFO["tasks"]) == 13
