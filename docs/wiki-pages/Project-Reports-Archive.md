# Project Reports Archive

> This page consolidates project reports and resync documents originally at the repo root.

## Test Results Summary

### Nemotron 3 Super 120B FP8
- **Date:** 2026-04-01
- **Model:** nemotron-3-super-120b-fp8
- **Status:** See `test/phases/results/compat-nemotron-3-super-120b-fp8-20260401.md`

### Qwen3 Coder Next FP8
- **Date:** 2026-04-01
- **Model:** qwen3-coder-next-fp8
- **Status:** See `test/phases/results/compat-qwen3-coder-next-fp8-2026-04-01.md`

## Project Resync (2026-03-28)

### Executive Summary

Sync-mode audit of the Engram project. All 10 Linear issues (KHA-5 through KHA-14) reviewed against codebase reality.

**Key findings:**
- KHA-5 is done in code but still in Backlog — PR #6 merged the startup snapshot warm restore into `main`
- The current working branch (`fix/snapshot-restore-state-sync`) is 8 commits behind `main` and lacks that implementation
- CLAUDE.md has drifted from reality after rapid merge activity (PRs #4 and #6)
- Five PERF issues (KHA-7–KHA-11) are premature until server-phase testing completes on sm75+ hardware

## Test Phase Status

| Phase | Description | Result |
|-------|-------------|--------|
| 0 | Environment verification | **PASS** |
| 1 | Stateless inference baseline | INCOMPLETE — run first on A100 |
| 2 | MambaPool unit tests | **PASS** (5/5) |
| 3 | MambaRadixCache gauntlet | **PASS** (16/16) |
| 4 | Live server — no_buffer strategy | INCOMPLETE |
| 5 | Mamba2Metadata integrity | **PASS** (5/5) |
| 6 | extra_buffer strategy | INCOMPLETE |
| 7 | Snapshot system e2e | INCOMPLETE — validates Gap fixes PRs #4 #6 |
| 8 | Gauntlet stress tests | INCOMPLETE |

**Resume order:** 1 → 4 → 7 → 6 → 8. Stop at first failure and diagnose.

## Related Documentation

- [Stateful Mamba Guide](./Stateful-Mamba-Guide.md)
- [Upstream Sync](./Upstream-Sync-Q1-2026.md)
- [Agent Framework](./Agent-Framework.md)
