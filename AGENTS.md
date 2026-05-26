<!-- ENGRAM_MODIFIED — Engram agent configuration and multi-agent framework -->
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

## Agent Skills

Skills live in `.claude/skills/`. Invoke them via the `Skill` tool.

- **`engram-brief-contract`** (`.claude/skills/engram-brief-contract/SKILL.md`)
  — Enforces the canonical 5-part brief format and report-back contracts from
  the Linear runbooks. Applies to feature/issue work (6-item report-back) and
  upstream merge work (9-item report-back). Triggers when an agent receives a
  structured brief or when the user references the runbook contract.

## Git Hooks

Mechanical enforcement is provided by the hooks in `.engram/hooks/`. Activate
once per checkout:

```bash
git config core.hooksPath .engram/hooks
```

- **`pre-push`** — blocks direct pushes to `main`; runs ruff lint and
  `pytest test/srt/cpu/` before every push.
- **`start-feature.sh`** — creates a branch off `origin/main` with slug
  validation and clobber protection. Usage: `.engram/hooks/start-feature.sh feat/my-feature`

See `.engram/hooks/README.md` for full details and the `--no-verify` escape hatch.

## Protected Path Policy

- The source of truth for protected files is
  `.engram/policy/protected-paths.json`
- Upstream sync automation and local agent guardrails both use
  `scripts/policy/check_protected_paths.py`
- Static protected globs cover permanently sensitive Engram surfaces
- Dynamic protection also applies to fork-owned files detected relative to
  `upstream/main`
- If upstream touches any protected file, automation must not auto-merge
- If you intentionally edit protected files locally, acknowledge that review is
  required by rerunning with `ENGRAM_ALLOW_PROTECTED_EDITS=1`
- Do not add new workflow-only path rules in `.github/workflows/upstream-sync.yml`;
  update `.engram/policy/protected-paths.json` instead
