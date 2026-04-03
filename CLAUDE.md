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
