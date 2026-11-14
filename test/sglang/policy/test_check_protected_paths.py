"""
Unit tests for the protected-path guard's local-edit check.

Covers which files `check-local-edits` asks an acknowledgement for, with and
without a git ref to compare against. Needs git and the repo checkout only.
"""

import argparse
import importlib.util
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT = REPO_ROOT / "scripts" / "policy" / "check_protected_paths.py"
MISSING_REF = "refs/engram-test/does-not-exist"

# In the static list on every branch.
STATIC_PROTECTED = "AGENTS.md"
# Not in the static list, and not a local-only file.
UNPROTECTED = "python/sglang/engram_policy_test_placeholder.py"


@pytest.fixture(scope="module")
def guard():
    spec = importlib.util.spec_from_file_location("check_protected_paths", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(autouse=True)
def no_ack(guard, monkeypatch):
    ack_env = guard.load_policy()["enforcement"]["local_ack_env"]
    monkeypatch.delenv(ack_env, raising=False)
    return ack_env


def run_check(guard, files, upstream_ref):
    args = argparse.Namespace(files=files, fork_ref="HEAD", upstream_ref=upstream_ref)
    return guard.cmd_check_local_edits(args)


class TestMissingUpstreamRef:
    """Without the upstream ref, the static list still applies."""

    def test_static_file_needs_ack(self, guard, capsys):
        assert run_check(guard, [STATIC_PROTECTED], MISSING_REF) == 1
        out = capsys.readouterr().out
        assert "static list only" in out
        assert f"- {STATIC_PROTECTED}" in out

    def test_unprotected_file_passes(self, guard, capsys):
        assert run_check(guard, [UNPROTECTED], MISSING_REF) == 0
        assert "static list only" in capsys.readouterr().out

    def test_only_static_files_are_named(self, guard, capsys):
        assert run_check(guard, [UNPROTECTED, STATIC_PROTECTED], MISSING_REF) == 1
        out = capsys.readouterr().out
        assert f"- {STATIC_PROTECTED}" in out
        assert f"- {UNPROTECTED}" not in out

    def test_ack_env_allows_static_file(self, guard, monkeypatch, no_ack, capsys):
        monkeypatch.setenv(no_ack, "1")
        assert run_check(guard, [STATIC_PROTECTED], MISSING_REF) == 0
        assert "acknowledged" in capsys.readouterr().out

    def test_local_only_file_is_ignored(self, guard, capsys):
        local_only = guard.load_policy()["local_only_globs"][0]
        assert run_check(guard, [local_only], MISSING_REF) == 0
        assert capsys.readouterr().out == ""


class TestWithRef:
    """HEAD...HEAD is an empty comparison, so only the static list matches."""

    def test_static_file_needs_ack(self, guard, capsys):
        assert run_check(guard, [STATIC_PROTECTED], "HEAD") == 1
        out = capsys.readouterr().out
        assert "static list only" not in out
        assert f"- {STATIC_PROTECTED}" in out

    def test_unprotected_file_passes(self, guard):
        assert run_check(guard, [UNPROTECTED], "HEAD") == 0
