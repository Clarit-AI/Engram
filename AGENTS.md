<!-- ENGRAM_MODIFIED — Engram agent configuration and multi-agent framework -->
# AGENTS.md: Engram

**Engram** (formerly **sglang-mamba**) is a fork of
[SGLang](https://github.com/sgl-project/sglang) that adds snapshot/statefulness
support for Mamba-family and related recurrent-state models.

> Legacy references to `sglang-mamba` may still appear in paths, tests, and git
> history. They refer to this same project.

This file is the canonical agent instruction file for the repo.
[`CLAUDE.md`](CLAUDE.md) imports it and adds only Claude-specific notes.

## Rules That Come First

1. **Engram pins to SGLang release tags.** Do not merge `upstream/main`, and do
   not add a schedule back to `.github/workflows/upstream-sync.yml`. Engram takes
   an upstream change only by cherry-pick, for a needed model, a security fix or
   a benchmark need. Moving to a newer SGLang release is planned work that the
   maintainer starts.
2. **Never push to `sgl-project/sglang` and never open a PR there.** Pass
   `--repo Clarit-AI/Engram` to every `gh` command that reads or writes a PR.
   In a fork checkout `gh` can resolve its default repo to upstream. Check with
   `gh repo set-default --view`.
3. **Status comes from a test run.** A claim that something works, passes or is
   current cites a row in the release-test harness's `results.tsv`: commit SHA,
   tag, model, GPU and log folder. No row, no claim. Do not take release status
   from a memory system or from an older document, and do not write release
   status into a memory system.
4. **No tag without a row.** A release tag is cut only after the harness passes
   on that exact commit. Agents do not push tags.
5. **Work on a branch and open a PR.** The maintainer merges to `main`.

The release-test harness lives outside this repo, in `Clarit-AI/engram-harness`
(maintainers only). It installs a ref, starts a server on a GPU, and checks that
a restored snapshot continues with the same tokens as a straight-through run.

## Source Of Truth

When working in this repo, prefer these sources in order:

1. Current code
2. [`docs/stateful_mamba/api_guide.md`](docs/stateful_mamba/api_guide.md)
3. [`docs/stateful_mamba/http_api_spec.md`](docs/stateful_mamba/http_api_spec.md)
4. `results.tsv` in the release-test harness, for what passes on which commit

[`test/phases/results/INDEX.md`](test/phases/results/INDEX.md) records the
March 2026 test program. It is history, not current status. Treat
[`docs/stateful_mamba/.archive/`](docs/stateful_mamba/.archive/) the same way.

## Things That Are Easy To Get Wrong

- **Tag names.** Engram releases are tagged `engram-vX.Y.Z`. A bare `vX.Y.Z`
  tag in this repo is an upstream SGLang tag. `v0.5.19`, for example, is an
  SGLang release and not an Engram one.
- **The enabling flag** is `--enable-snapshot-persistence`.
  `--enable-mamba-snapshots` does not exist.
- **PR #91 (KHA-390)** made a plain `/generate` sent after `/restore_snapshot`
  start from the restored state. It merged on 2026-05-23 as `f0b617c56c`. It is
  in `main` and not in the `engram-v0.5.2` tag.
- **PR #101**, the 158-commit upstream sync of 2026-05-25, is an unmerged draft.
  Its branch `upstream-sync-20260525` is not `main`. Do not base work on it and
  do not copy skills or instruction files from it.
- **The snapshot code moved out of `scheduler.py`** on 2026-05-20. See Key Code
  Areas.

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
- `python/sglang/srt/managers/scheduler_snapshot_handlers.py`: the save, list,
  info, restore and delete handlers, and the pending-restore registry
- `python/sglang/srt/managers/scheduler_snapshot_system.py`: snapshot system
  startup and the callback that takes automatic snapshots
- `python/sglang/srt/managers/scheduler_components/batch_result_processor.py`:
  where a finished request triggers that callback
- `python/sglang/srt/managers/scheduler.py`: thin delegators to the files above
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
  inside this repository. Load them from a checkout of `main`. Copies installed
  at user level on a machine go stale and can come from the wrong branch.

## Agent Skills

Skills live in `.claude/skills/`. Invoke them via the `Skill` tool.

- **`engram-sglang`** (`.claude/skills/engram-sglang/SKILL.md`): the snapshot
  APIs, server flags, where the code lives, and how to check what passes.
- **`engram-brief-contract`** (`.claude/skills/engram-brief-contract/SKILL.md`):
  the 5-part brief format and the 6-item report-back for feature, fix and chore
  work. Triggers when an agent receives a structured brief or when the user
  references the runbook contract.

The other skills in that folder come from upstream SGLang and are not
Engram-specific.

## Slash Commands

Explicit-invocation shortcuts live in `.claude/commands/` and fire only when you
type `/<name>`. They never auto-trigger.

- **`/engram-hooks-on`**: activate the Engram git hooks (`core.hooksPath` to
  `.engram/hooks`); reports any prior value it overrode.
- **`/engram-hooks-off [--restore-husky]`**: unset `core.hooksPath`, or restore
  `.husky` with the flag.
- **`/engram-feature <type>/<slug> [--worktree]`**: branch off latest
  `origin/main` via `start-feature.sh`, or create an isolated worktree.
- **`/engram-status`**: branch, ahead/behind vs `origin/main`, hook state,
  base SHA, and any open PR, as a key/value block.
- **`/engram-pr <title>`**: push if needed and open a PR against
  `Clarit-AI/Engram` with the canonical body template (type inferred from branch).
- **`/engram-gate`**: run the same lint + CPU-test gate as the pre-push hook
  (ruff `F401`/`F821` + `pytest test/srt/cpu/`), on demand.
- **`/engram-runbook`**: fetch and display the Feature & Issue Runbook from
  Linear.

## Git Hooks

Mechanical enforcement is provided by the hooks in `.engram/hooks/`. They are
off until you activate them, once per checkout:

```bash
git config core.hooksPath .engram/hooks
```

- **`pre-push`** refuses pushes to upstream and direct pushes to `main`, then
  runs ruff lint and `pytest test/srt/cpu/`. The lint and test gates need an
  environment with ruff, pytest and sglang; see the README for `ENGRAM_PYTHON`
  and `ENGRAM_SKIP_GATE`.
- **`start-feature.sh`** creates a branch off `origin/main` with slug
  validation and clobber protection. Usage: `.engram/hooks/start-feature.sh feat/my-feature`

See `.engram/hooks/README.md` for full details.

## Protected Path Policy

- The source of truth for protected files is
  `.engram/policy/protected-paths.json`
- Local agent guardrails and the manually started upstream-sync workflow both
  use `scripts/policy/check_protected_paths.py`
- Static protected globs cover permanently sensitive Engram surfaces
- Dynamic protection also applies to fork-owned files detected relative to
  `upstream/main`
- If upstream touches any protected file, automation must not auto-merge
- If you intentionally edit protected files locally, acknowledge that review is
  required by rerunning with `ENGRAM_ALLOW_PROTECTED_EDITS=1`
- Do not add new workflow-only path rules in `.github/workflows/upstream-sync.yml`;
  update `.engram/policy/protected-paths.json` instead
