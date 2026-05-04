"""
Unit tests for scripts/benchmarks/locomo_runner.py.

All tests run on CPU with no model server (dry_run=True or mocked calls).
Run: pytest tests/benchmarks/test_locomo_runner.py -v
"""

import json
import os
import sys
import tempfile
from pathlib import Path

import pytest

# Ensure the scripts directory is importable
sys.path.insert(0, str(Path(__file__).parents[2] / "scripts" / "benchmarks"))

from locomo_runner import (
    _normalize,
    _token_f1,
    _partial_f1,
    build_context_text,
    flatten_conversation,
    load_dataset,
    main,
    parse_args,
    run_engram,
    run_stateless,
    score_qa,
    write_outputs,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_conversation(num_sessions: int = 2, turns_per_session: int = 3) -> dict:
    conv = {
        "speaker_a": "Alice",
        "speaker_b": "Bob",
    }
    dia = 0
    for s in range(1, num_sessions + 1):
        conv[f"session_{s}_date_time"] = f"2023-0{s}-01T10:00:00"
        turns = []
        for t in range(1, turns_per_session + 1):
            dia += 1
            turns.append({
                "speaker": "Alice" if t % 2 == 1 else "Bob",
                "dia_id": f"D{s}:{t}",
                "text": f"Session {s} turn {t} content.",
            })
        conv[f"session_{s}"] = turns
    return conv


def _make_sample(sample_id: str = "conv-test", qa: list | None = None) -> dict:
    if qa is None:
        qa = [
            {"question": "What did Alice say first?", "answer": "Session 1 turn 1 content.",
             "evidence": ["D1:1"], "category": 4},
            {"question": "When was session 1?", "answer": "2023-01-01",
             "evidence": ["D1:1"], "category": 2},
            {"question": "Did Alice discuss invisible things?", "adversarial_answer": "yes",
             "evidence": [], "category": 5},
        ]
    return {
        "sample_id": sample_id,
        "conversation": _make_conversation(),
        "qa": qa,
        "event_summary": {},
        "observation": {},
        "session_summary": {},
    }


def _write_dataset(path: str, samples: list) -> None:
    with open(path, "w") as f:
        json.dump(samples, f)


# ---------------------------------------------------------------------------
# Argparse
# ---------------------------------------------------------------------------

class TestArgparse:
    def test_required_mode(self):
        with pytest.raises(SystemExit):
            parse_args(["--model", "m", "--dataset-path", "x", "--output-dir", "y"])

    def test_stateless_mode(self):
        args = parse_args([
            "--mode", "stateless", "--model", "m",
            "--dataset-path", "x", "--output-dir", "y",
        ])
        assert args.mode == "stateless"
        assert args.dry_run is False

    def test_engram_mode(self):
        args = parse_args([
            "--mode", "engram", "--model", "m",
            "--dataset-path", "x", "--output-dir", "y",
            "--dry-run",
        ])
        assert args.mode == "engram"
        assert args.dry_run is True

    def test_conversations_all(self):
        args = parse_args([
            "--mode", "stateless", "--model", "m",
            "--dataset-path", "x", "--output-dir", "y",
        ])
        assert args.conversations == "all"

    def test_conversations_int(self):
        args = parse_args([
            "--mode", "stateless", "--model", "m",
            "--dataset-path", "x", "--output-dir", "y",
            "--conversations", "3",
        ])
        assert args.conversations == "3"


# ---------------------------------------------------------------------------
# Dataset loader
# ---------------------------------------------------------------------------

class TestDatasetLoader:
    def test_load_basic(self, tmp_path):
        f = tmp_path / "test.json"
        samples = [_make_sample("conv-1"), _make_sample("conv-2")]
        _write_dataset(str(f), samples)
        data = load_dataset(str(f))
        assert len(data) == 2
        assert data[0]["sample_id"] == "conv-1"

    def test_load_max_conversations(self, tmp_path):
        f = tmp_path / "test.json"
        samples = [_make_sample(f"conv-{i}") for i in range(5)]
        _write_dataset(str(f), samples)
        data = load_dataset(str(f), max_conversations=2)
        assert len(data) == 2

    def test_missing_file_raises(self):
        with pytest.raises(FileNotFoundError):
            load_dataset("/nonexistent/path.json")

    def test_non_list_raises(self, tmp_path):
        f = tmp_path / "bad.json"
        f.write_text('{"not": "a list"}')
        with pytest.raises(ValueError):
            load_dataset(str(f))


# ---------------------------------------------------------------------------
# Conversation flattening
# ---------------------------------------------------------------------------

class TestFlattenConversation:
    def test_turn_count(self):
        conv = _make_conversation(num_sessions=2, turns_per_session=3)
        turns = flatten_conversation(conv)
        assert len(turns) == 6

    def test_session_order(self):
        conv = _make_conversation(num_sessions=3, turns_per_session=1)
        turns = flatten_conversation(conv)
        assert [t["session"] for t in turns] == [1, 2, 3]

    def test_dia_id_preserved(self):
        conv = _make_conversation(num_sessions=1, turns_per_session=2)
        turns = flatten_conversation(conv)
        assert turns[0]["dia_id"] == "D1:1"
        assert turns[1]["dia_id"] == "D1:2"

    def test_blip_caption_included(self):
        conv = _make_conversation(num_sessions=1, turns_per_session=1)
        conv["session_1"][0]["blip_caption"] = "a photo of a cat"
        turns = flatten_conversation(conv)
        assert turns[0]["blip_caption"] == "a photo of a cat"

    def test_build_context_text(self):
        turns = [
            {"speaker": "Alice", "text": "Hello.", "blip_caption": ""},
            {"speaker": "Bob", "text": "Hi!", "blip_caption": "a sunset"},
        ]
        ctx = build_context_text(turns)
        assert "Alice: Hello." in ctx
        assert "Bob: Hi! [image: a sunset]" in ctx


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

class TestScoring:
    def test_cat4_exact(self):
        qa = {"category": 4, "answer": "adoption agencies"}
        assert score_qa("adoption agencies", qa) == pytest.approx(1.0)

    def test_cat4_partial(self):
        qa = {"category": 4, "answer": "the quick brown fox"}
        score = score_qa("the quick fox", qa)
        assert 0 < score < 1.0

    def test_cat4_no_match(self):
        qa = {"category": 4, "answer": "xyz"}
        assert score_qa("completely different", qa) == 0.0

    def test_cat2_temporal(self):
        qa = {"category": 2, "answer": "7 May 2023"}
        assert score_qa("7 May 2023", qa) == pytest.approx(1.0)

    def test_cat3_semicolon(self):
        qa = {"category": 3, "answer": "psychology; counseling"}
        score = score_qa("psychology", qa)
        assert score == pytest.approx(1.0)

    def test_cat1_multi_hop(self):
        qa = {"category": 1, "answer": "cats, dogs"}
        assert score_qa("cats and dogs", qa) >= 0.5

    def test_cat5_adversarial_no_info(self):
        qa = {"category": 5, "adversarial_answer": "self-care"}
        assert score_qa("No information available in the conversation", qa) == 1.0

    def test_cat5_adversarial_wrong(self):
        qa = {"category": 5, "adversarial_answer": "self-care"}
        assert score_qa("self-care is important", qa) == 0.0

    def test_cat5_not_mentioned(self):
        qa = {"category": 5, "adversarial_answer": "x"}
        assert score_qa("That's not mentioned in the conversation.", qa) == 1.0

    def test_normalize_strips_articles(self):
        assert _normalize("the quick brown fox") == "quick brown fox"

    def test_token_f1_symmetry(self):
        a, b = "hello world", "world hello"
        assert _token_f1(a, b) == pytest.approx(_token_f1(b, a))


# ---------------------------------------------------------------------------
# Run modes (dry-run)
# ---------------------------------------------------------------------------

class TestStatelessMode:
    def test_returns_one_result_per_qa(self):
        sample = _make_sample()
        results = run_stateless(sample, "http://unused", "test-model", dry_run=True)
        assert len(results) == len(sample["qa"])

    def test_result_schema(self):
        sample = _make_sample()
        results = run_stateless(sample, "http://unused", "test-model", dry_run=True)
        r = results[0]
        assert "question" in r
        assert "prediction" in r
        assert "score" in r
        assert "latency_s" in r
        assert r["save_latency_s"] is None
        assert r["restore_latency_s"] is None

    def test_score_is_float_in_range(self):
        sample = _make_sample()
        results = run_stateless(sample, "http://unused", "test-model", dry_run=True)
        for r in results:
            assert 0.0 <= r["score"] <= 1.0


class TestEngramMode:
    def test_returns_one_result_per_qa(self):
        sample = _make_sample()
        results = run_engram(sample, "http://unused", "test-model", dry_run=True)
        assert len(results) == len(sample["qa"])

    def test_result_has_save_restore_latency(self):
        sample = _make_sample()
        results = run_engram(sample, "http://unused", "test-model", dry_run=True)
        r = results[0]
        assert r["save_latency_s"] is not None
        assert r["restore_latency_s"] is not None
        assert r["save_latency_s"] > 0
        assert r["restore_latency_s"] > 0

    def test_score_is_float_in_range(self):
        sample = _make_sample()
        results = run_engram(sample, "http://unused", "test-model", dry_run=True)
        for r in results:
            assert 0.0 <= r["score"] <= 1.0


# ---------------------------------------------------------------------------
# Output schema
# ---------------------------------------------------------------------------

class TestOutputSchema:
    def test_json_output_valid(self, tmp_path):
        sample = _make_sample()
        results = run_stateless(sample, "http://unused", "model", dry_run=True)
        json_path, csv_path = write_outputs(
            {"conv-test": results}, str(tmp_path), "stateless", "model"
        )
        with open(json_path) as f:
            data = json.load(f)
        assert data["mode"] == "stateless"
        assert data["model"] == "model"
        assert "overall_f1" in data
        assert "f1_by_category" in data
        assert "total_questions" in data
        assert data["total_questions"] == len(sample["qa"])

    def test_csv_output_valid(self, tmp_path):
        import csv as _csv
        sample = _make_sample()
        results = run_stateless(sample, "http://unused", "model", dry_run=True)
        _, csv_path = write_outputs(
            {"conv-test": results}, str(tmp_path), "stateless", "model"
        )
        with open(csv_path) as f:
            rows = list(_csv.DictReader(f))
        assert len(rows) == len(sample["qa"])
        assert "score" in rows[0]
        assert "prediction" in rows[0]

    def test_overall_f1_in_range(self, tmp_path):
        sample = _make_sample()
        results = run_stateless(sample, "http://unused", "model", dry_run=True)
        json_path, _ = write_outputs(
            {"conv-test": results}, str(tmp_path), "stateless", "model"
        )
        with open(json_path) as f:
            data = json.load(f)
        assert 0.0 <= data["overall_f1"] <= 1.0


# ---------------------------------------------------------------------------
# End-to-end dry-run (main entry point)
# ---------------------------------------------------------------------------

class TestDryRunEndToEnd:
    def test_main_stateless_dry_run(self, tmp_path):
        dataset_path = str(tmp_path / "dataset.json")
        _write_dataset(dataset_path, [_make_sample("conv-1")])
        main([
            "--mode", "stateless",
            "--model", "test-model",
            "--dataset-path", dataset_path,
            "--conversations", "1",
            "--output-dir", str(tmp_path / "out"),
            "--dry-run",
        ])
        out_dir = tmp_path / "out"
        json_files = list(out_dir.glob("*.json"))
        csv_files = list(out_dir.glob("*.csv"))
        assert len(json_files) == 1
        assert len(csv_files) == 1

    def test_main_engram_dry_run(self, tmp_path):
        dataset_path = str(tmp_path / "dataset.json")
        _write_dataset(dataset_path, [_make_sample("conv-1")])
        main([
            "--mode", "engram",
            "--model", "test-model",
            "--dataset-path", dataset_path,
            "--conversations", "1",
            "--output-dir", str(tmp_path / "out"),
            "--dry-run",
        ])
        out_dir = tmp_path / "out"
        json_files = list(out_dir.glob("*.json"))
        assert len(json_files) == 1
        with open(json_files[0]) as f:
            data = json.load(f)
        assert data["mode"] == "engram"
        assert data["mean_restore_latency_s"] is not None

    def test_no_network_calls_in_dry_run(self, tmp_path, monkeypatch):
        """Confirm dry-run doesn't touch urllib."""
        import urllib.request

        def _fail(*_a, **_kw):
            raise RuntimeError("Network call made in dry-run mode")

        monkeypatch.setattr(urllib.request, "urlopen", _fail)
        dataset_path = str(tmp_path / "dataset.json")
        _write_dataset(dataset_path, [_make_sample("conv-1")])
        main([
            "--mode", "engram",
            "--model", "test-model",
            "--dataset-path", dataset_path,
            "--conversations", "1",
            "--output-dir", str(tmp_path / "out"),
            "--dry-run",
        ])
