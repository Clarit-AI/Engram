#!/usr/bin/env python3
"""Shared protected-path policy for upstream sync and local agent guardrails."""

from __future__ import annotations

import argparse
import fnmatch
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Iterable

REPO_ROOT = Path(__file__).resolve().parents[2]
POLICY_PATH = REPO_ROOT / ".engram" / "policy" / "protected-paths.json"


def run_git(args: list[str]) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout


def load_policy() -> dict:
    with POLICY_PATH.open("r", encoding="utf-8") as fh:
        policy = json.load(fh)

    if not isinstance(policy, dict):
        raise ValueError("Policy file must be a JSON object.")

    for key in ("static_protected_globs", "local_only_globs", "enforcement"):
        if key not in policy:
            raise ValueError(f"Policy file missing required key: {key}")

    return policy


def normalize_path(path: str) -> str:
    normalized = path.strip().replace("\\", "/")
    while normalized.startswith("./"):
        normalized = normalized[2:]
    return normalized


def split_nonempty_lines(text: str) -> list[str]:
    return [normalize_path(line) for line in text.splitlines() if line.strip()]


def split_pipe_list(text: str) -> list[str]:
    if not text.strip():
        return []
    return [normalize_path(item) for item in text.split("|") if item.strip()]


def unique_sorted(paths: Iterable[str]) -> list[str]:
    return sorted({normalize_path(path) for path in paths if normalize_path(path)})


def matches_any(path: str, patterns: Iterable[str]) -> bool:
    normalized = normalize_path(path)
    return any(fnmatch.fnmatchcase(normalized, pattern) for pattern in patterns)


def filter_matching(paths: Iterable[str], patterns: Iterable[str]) -> list[str]:
    pattern_list = list(patterns)
    return unique_sorted(path for path in paths if matches_any(path, pattern_list))


def git_changed_files(refspec: str) -> list[str]:
    return split_nonempty_lines(run_git(["diff", "--name-only", refspec]))


def git_ref_exists(ref: str) -> bool:
    result = subprocess.run(
        ["git", "rev-parse", "--verify", "--quiet", f"{ref}^{{commit}}"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    return result.returncode == 0


def compute_sync_policy(
    policy: dict,
    *,
    fork_ref: str,
    upstream_ref: str,
    conflicts: str,
) -> dict:
    static_globs = policy["static_protected_globs"]

    fork_only_files = set(git_changed_files(f"{upstream_ref}...{fork_ref}"))
    upstream_incoming_files = set(git_changed_files(f"{fork_ref}..{upstream_ref}"))
    conflict_files = set(split_pipe_list(conflicts))

    static_touches = set(filter_matching(upstream_incoming_files, static_globs))
    dynamic_touches = fork_only_files & upstream_incoming_files
    protected_upstream_touches = unique_sorted(static_touches | dynamic_touches)
    protected_conflicts = unique_sorted(conflict_files & set(protected_upstream_touches))

    if conflict_files:
        if protected_conflicts:
            risk_level = "high"
            risk_reason = "Merge conflicts in protected fork files"
        elif protected_upstream_touches:
            risk_level = "high"
            risk_reason = "Merge is blocked and upstream touched protected fork files"
        else:
            risk_level = "medium"
            risk_reason = "Merge conflicts in upstream-only files (likely resolvable)"
    else:
        if protected_upstream_touches:
            risk_level = "high"
            risk_reason = "Clean merge, but upstream touched protected fork files"
        else:
            risk_level = "low"
            risk_reason = "Clean merge, no protected fork files affected"

    return {
        "risk_level": risk_level,
        "risk_reason": risk_reason,
        "fork_only_files": unique_sorted(fork_only_files),
        "upstream_incoming_files": unique_sorted(upstream_incoming_files),
        "protected_upstream_touches": protected_upstream_touches,
        "protected_conflicts": protected_conflicts,
    }


def write_github_output(path: str, values: dict[str, str | list[str]]) -> None:
    with open(path, "a", encoding="utf-8") as fh:
        for key, value in values.items():
            if isinstance(value, list):
                fh.write(f"{key}<<EOF\n")
                fh.write("\n".join(value))
                fh.write("\nEOF\n")
            else:
                fh.write(f"{key}={value}\n")


def cmd_validate_policy(_: argparse.Namespace) -> int:
    policy = load_policy()
    for key in ("static_protected_globs", "local_only_globs"):
        if not isinstance(policy[key], list) or not all(
            isinstance(item, str) for item in policy[key]
        ):
            raise ValueError(f"{key} must be a list of strings")
    if not isinstance(policy["enforcement"], dict):
        raise ValueError("enforcement must be an object")
    print(f"Policy ok: {POLICY_PATH.relative_to(REPO_ROOT)}")
    return 0


def cmd_classify_upstream_sync(args: argparse.Namespace) -> int:
    policy = load_policy()
    result = compute_sync_policy(
        policy,
        fork_ref=args.fork_ref,
        upstream_ref=args.upstream_ref,
        conflicts=args.conflicts,
    )

    if args.github_output:
        write_github_output(
            args.github_output,
            {
                "risk_level": result["risk_level"],
                "risk_reason": result["risk_reason"],
                "protected_upstream_touches": result["protected_upstream_touches"],
                "protected_conflicts": result["protected_conflicts"],
            },
        )
    else:
        print(json.dumps(result, indent=2))
    return 0


def cmd_check_local_edits(args: argparse.Namespace) -> int:
    policy = load_policy()
    static_globs = policy["static_protected_globs"]
    local_only_globs = policy["local_only_globs"]
    ack_env = policy["enforcement"].get("local_ack_env", "ENGRAM_ALLOW_PROTECTED_EDITS")

    candidate_files = unique_sorted(args.files)
    candidate_files = [
        path for path in candidate_files if not matches_any(path, local_only_globs)
    ]

    if not candidate_files:
        return 0

    missing_refs = [
        ref for ref in (args.upstream_ref, args.fork_ref) if not git_ref_exists(ref)
    ]
    if missing_refs:
        print(
            "Protected-path policy fallback: missing git ref(s) "
            + ", ".join(missing_refs)
            + ". Treating all candidate files as fork-owned because "
            + f"{args.upstream_ref}...{args.fork_ref} cannot be compared."
        )
        fork_owned_files = set(candidate_files)
    else:
        try:
            fork_owned_files = set(
                git_changed_files(f"{args.upstream_ref}...{args.fork_ref}")
            )
        except subprocess.CalledProcessError:
            print(
                "Protected-path policy fallback: unable to diff "
                + f"{args.upstream_ref}...{args.fork_ref}. "
                + "Treating all candidate files as fork-owned."
            )
            fork_owned_files = set(candidate_files)

    protected_files = unique_sorted(
        path
        for path in candidate_files
        if matches_any(path, static_globs) or path in fork_owned_files
    )

    if not protected_files:
        return 0

    if os.environ.get(ack_env) == "1":
        print(
            f"{ack_env}=1 acknowledged protected path edits:\n"
            + "\n".join(f"- {path}" for path in protected_files)
        )
        return 0

    print(
        "Protected-path policy requires review acknowledgement before committing "
        "or pushing changes to these files:\n"
        + "\n".join(f"- {path}" for path in protected_files)
        + f"\n\nIf this is intentional, rerun with {ack_env}=1."
    )
    return 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate = subparsers.add_parser("validate-policy")
    validate.set_defaults(func=cmd_validate_policy)

    classify = subparsers.add_parser("classify-upstream-sync")
    classify.add_argument("--fork-ref", required=True)
    classify.add_argument("--upstream-ref", required=True)
    classify.add_argument("--conflicts", default="")
    classify.add_argument("--github-output")
    classify.set_defaults(func=cmd_classify_upstream_sync)

    local = subparsers.add_parser("check-local-edits")
    local.add_argument("--fork-ref", default="HEAD")
    local.add_argument("--upstream-ref", default="upstream/main")
    local.add_argument("files", nargs="*")
    local.set_defaults(func=cmd_check_local_edits)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
