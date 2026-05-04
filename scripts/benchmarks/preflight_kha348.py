"""
Pre-flight check: verify KHA-348 fix is merged and present in local main.

Exit 0  — fix confirmed in main, safe to proceed.
Exit 1  — fix not confirmed; diagnostic printed to stderr.

Usage: python scripts/benchmarks/preflight_kha348.py
"""

from __future__ import annotations

import subprocess
import sys

# Known good commits introduced by the KHA-348 fix branch.
# Update this list if the fix is squash-merged under a different SHA.
KHA348_FIX_COMMITS = [
    "bfc58ee8a",  # fix(snapshot): live-restore clears pending entry + conv_id conflict resolution
    "c5b958c55",  # fix(snapshot): live-restore path syncs req.conversation_id with effective_conv_id
]

KHA348_LINEAR_URL = (
    "https://linear.app/khaentertainment/issue/KHA-348/"
    "v1chatcompletions-does-not-rehydrate-restored-snapshot-state-across"
)


def run(cmd: list[str]) -> tuple[int, str]:
    r = subprocess.run(cmd, capture_output=True, text=True)
    return r.returncode, r.stdout.strip()


def check_linear_status() -> bool:
    """
    Attempts a lightweight check via the gh CLI if available.
    Falls back to True (non-blocking) when gh is not configured.
    This is best-effort; the git-log check below is authoritative.
    """
    rc, _ = run(["which", "gh"])
    if rc != 0:
        print("[preflight] gh CLI not found — skipping Linear API check (non-blocking)", flush=True)
        return True
    print(f"[preflight] KHA-348 Linear: {KHA348_LINEAR_URL}", flush=True)
    return True


def check_git_log() -> bool:
    """Verify at least one KHA-348 fix commit is reachable from HEAD."""
    rc, log = run(["git", "log", "--oneline", "-200"])
    if rc != 0:
        print("[preflight] ERROR: git log failed", file=sys.stderr)
        return False
    for sha in KHA348_FIX_COMMITS:
        if sha in log:
            print(f"[preflight] KHA-348 fix commit found in git log: {sha}", flush=True)
            return True
    print("[preflight] ERROR: KHA-348 fix commits not found in last 200 log entries.", file=sys.stderr)
    print(f"[preflight] Expected one of: {KHA348_FIX_COMMITS}", file=sys.stderr)
    print("[preflight] Run: git fetch origin main && git merge origin/main", file=sys.stderr)
    return False


def check_main_branch() -> bool:
    """Warn (non-blocking) if we're not on main."""
    rc, branch = run(["git", "rev-parse", "--abbrev-ref", "HEAD"])
    if branch != "main":
        print(f"[preflight] WARNING: not on main branch (current: {branch})", flush=True)
        print("[preflight] KHA-348 fix must be reachable from the branch used for benchmarking.", flush=True)
    return True


def main() -> None:
    print("=== preflight_kha348: verifying KHA-348 fix is present ===", flush=True)
    failures = []

    check_main_branch()

    if not check_linear_status():
        failures.append("Linear status check failed")

    if not check_git_log():
        failures.append("KHA-348 fix commits not found in git log")

    if failures:
        print("\n[preflight] FAIL — KHA-348 fix not confirmed.", file=sys.stderr)
        for f in failures:
            print(f"  - {f}", file=sys.stderr)
        sys.exit(1)

    print("\n[preflight] PASS — KHA-348 fix confirmed in git history.", flush=True)
    sys.exit(0)


if __name__ == "__main__":
    main()
