# CLAUDE.md — Engram

This file contains the public, repo-safe guidance for Claude-style agents.
Use [`AGENTS.md`](AGENTS.md) as the canonical project instructions.

## Project Identity

**Engram** (formerly **sglang-mamba**) is a fork of
[SGLang](https://github.com/sgl-project/sglang) that adds snapshot/statefulness
support for Mamba-family and related recurrent-state models.

Legacy references to `sglang-mamba` may still appear in paths, tests, and git
history. They refer to this same project.

## Start Here

- Read [`AGENTS.md`](AGENTS.md)
- Use the current code and active docs as the source of truth
- Prefer [`docs/stateful_mamba/api_guide.md`](docs/stateful_mamba/api_guide.md)
  for user-facing API guidance
- Use [`docs/stateful_mamba/http_api_spec.md`](docs/stateful_mamba/http_api_spec.md)
  for the technical HTTP contract
- Treat [`docs/stateful_mamba/.archive/`](docs/stateful_mamba/.archive/) as
  historical only

## Local Notes

If `.claude/local-notes.md` exists, it may contain local maintainer preferences
or environment details. Treat it as local-only context, not public project
policy.

## Git Hooks

Git hooks live in `.engram/hooks/` and activate with one command per checkout:

```bash
git config core.hooksPath .engram/hooks
```

The `pre-push` hook blocks direct pushes to `main` and runs ruff + CPU-only
tests before any push. `start-feature.sh` creates properly-based branches.
See `.engram/hooks/README.md` for full usage.

## Agent Skills

Skills live in `.claude/skills/` and are invoked via the `Skill` tool.

- **`engram-brief-contract`** — Enforces the canonical 5-part brief format and
  report-back contracts from the Linear runbooks. Applies universally to feature
  work (6-item report-back) and upstream merge work (9-item report-back). See
  `.claude/skills/engram-brief-contract/SKILL.md`.

## Slash Commands

Explicit-invocation shortcuts live in `.claude/commands/` and fire only when you
type `/<name>` (never auto-triggered). They wrap the runbook lifecycle:

- **`/engram-hooks-on`** / **`/engram-hooks-off [--restore-husky]`** — toggle the
  Engram git hooks (`core.hooksPath`).
- **`/engram-feature <type>/<slug> [--worktree]`** — start a branch (or worktree)
  off latest `origin/main`.
- **`/engram-status`** — branch, distance from `origin/main`, hook state, base
  SHA, and open PR.
- **`/engram-pr <title>`** — open a PR with the canonical body template; type
  inferred from the branch prefix.
- **`/engram-gate`** — run the pre-push gate (ruff `F401`/`F821` +
  `pytest test/srt/cpu/`) on demand.
- **`/engram-runbook`** / **`/engram-merge-runbook`** — fetch and display the
  Linear runbooks inline.

## Protected Paths

Protected-path policy lives in `.engram/policy/protected-paths.json`.
The shared helper is `scripts/policy/check_protected_paths.py`.

Agents should treat protected-path edits and upstream touches as review-required
work, and should update the policy file rather than hardcoding new path rules in
the workflow.
