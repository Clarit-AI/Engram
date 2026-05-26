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

## Protected Paths

Protected-path policy lives in `.engram/policy/protected-paths.json`.
The shared helper is `scripts/policy/check_protected_paths.py`.

Agents should treat protected-path edits and upstream touches as review-required
work, and should update the policy file rather than hardcoding new path rules in
the workflow.
