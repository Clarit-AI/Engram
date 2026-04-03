# AGENTS.md — Engram

**Engram** (formerly **sglang-mamba**) is a fork of
[SGLang](https://github.com/sgl-project/sglang) that adds snapshot/statefulness
support for Mamba-family and related recurrent-state models.

> Legacy references to `sglang-mamba` may still appear in paths, tests, and git
> history. They refer to this same project.

## Source Of Truth

When working in this repo, prefer these sources in order:

1. Current code
2. [`docs/stateful_mamba/api_guide.md`](docs/stateful_mamba/api_guide.md)
3. [`docs/stateful_mamba/http_api_spec.md`](docs/stateful_mamba/http_api_spec.md)
4. [`test/phases/results/INDEX.md`](test/phases/results/INDEX.md)

Treat [`docs/stateful_mamba/.archive/`](docs/stateful_mamba/.archive/) as
historical reference only.

## Public API Orientation

- Engram keeps the standard SGLang and OpenAI-compatible serving flows
- Snapshot/statefulness support is additive, not a separate platform
- The snapshot HTTP routes are:
  - `POST /save_snapshot`
  - `POST /list_snapshots`
  - `POST /get_snapshot_info`
  - `POST /restore_snapshot`
  - `POST /delete_snapshot`

## Key Code Areas

- `python/sglang/srt/entrypoints/http_server.py`: snapshot HTTP routes
- `python/sglang/srt/managers/io_struct.py`: snapshot request/response types
- `python/sglang/srt/managers/scheduler.py`: snapshot hooks and restore behavior
- `python/sglang/srt/snapshot/`: snapshot, tiering, and health-monitoring code
- `python/sglang/lang/interpreter.py`: `ProgramState` snapshot helpers
- `python/sglang/snapshot.py`: public `SnapshotManager(runtime.endpoint)` API

## Repo Guidance

- Prefer repo-relative paths and placeholders over personal usernames, machine
  names, account IDs, or local absolute paths in tracked docs
- Keep tracked instructions repo-generic; maintainer-specific workflow notes
  belong in local-only files, not the public repo
- If `.claude/local-notes.md` exists, treat it as local maintainer context, not
  project policy
- Project skill files under `.claude/skills/` are intended to help agents work
  inside this repository
