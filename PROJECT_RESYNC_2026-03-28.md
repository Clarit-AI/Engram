# PROJECT RESYNC — 2026-03-28

## Executive Summary

Sync-mode audit of the sglang-mamba project. All 10 Linear issues (KHA-5 through KHA-14) reviewed against codebase reality. **KHA-5 is done in code but still in Backlog** — PR #6 merged the startup snapshot warm restore into `main`. The current working branch (`fix/snapshot-restore-state-sync`) is 8 commits behind `main` and lacks that implementation. CLAUDE.md has drifted from reality after rapid merge activity (PRs #4 and #6). AGENTS.md does not exist. Five PERF issues (KHA-7–KHA-11) are premature until server-phase testing completes on sm75+ hardware.

---

## Gap Analysis — Status Matrix

### Complete (built and working)

| Feature | Code Location | Notes |
|---------|--------------|-------|
| Snapshot save/list/get/delete | `python/sglang/srt/snapshot/` + scheduler + HTTP endpoints | 7 source files, fully wired |
| Snapshot restore (in-place + create_new_request) | `scheduler.py:1294+` | PR #4, merged 2026-03-28 |
| Startup snapshot warm restore (Gap 3) | `tier_manager.py:restore_latest_snapshots_to_warm_tier()` | PR #6, merged to `main` |
| MambaRadixCache | `python/sglang/srt/mem_cache/mamba_radix_cache.py` (1239 lines) | Dual LRU, tombstones, COW |
| Agent Framework | `python/sglang/srt/agents/` (10 files) | 4 built-in tools, REST + WebSocket |
| 3-Tier Memory | `python/sglang/srt/snapshot/tier_manager.py` (468 lines) | VRAM/RAM/Disk with LRU |
| SnapshotManager API | `python/sglang/snapshot.py` (303 lines) | Public API exported from `__init__.py` |
| ConversationTracker | `python/sglang/srt/snapshot/conversation_tracker.py` | Tier state tracking |
| Snapshot hooks + policy | `snapshot_hooks.py`, `snapshot_policy.py` | Branching, retention, trigger policies |
| Fork-specific unit tests | `test/sglang/snapshot/` (4 files), `test/sglang/agents/` (3 files) | 46 pass, 1 skip (snapshot); 37 pass (agents) |
| Test phase infrastructure | `test/phases/` (9 plans, config.sh, codemap, results/) | Phases 0/2/3/5 PASS |
| Radix cache gauntlet tests | `test/registered/radix_cache/` (2 files) | Comprehensive + gauntlet |
| Startup restore test | `python/sglang/test/srt/test_startup_snapshot_restore.py` | 202 lines, added in PR #6 |
| Skills for Gemini | `skills/mamba-sglang/` (SKILL.md + 3 references) | Not linked from CLAUDE.md |
| Beads tracking | `.beads/` | Init'd on `main`, not on current branch |
| Cache diagnostics | `mamba_radix_cache.py` hit_tokens/miss_tokens | Added in recent commits |

### In Progress / Partially Done

| Item | Status | Notes |
|------|--------|-------|
| Server-phase testing (1/4/6/7/8) | **Blocked** | All need sm75+ GPU; V100 is sm70 |
| Phase 3.4 Final Audit (KHA-6) | Pending | Scope needs updating to reflect test phases |
| Model documentation (KHA-12, KHA-13) | Not started | References `phase3/docs/pending_updates.json` |

### Planned but Not Built

| Item | Plan Source | Status |
|------|-------------|--------|
| Phase test files (1,4,6,7,8) | `test/phases/` plans | Plans exist, no `.py` test files created |
| Automatic snapshot triggers | `MAMBA_SNAPSHOT_RESTORATION_PLAN.md` Phase 3 Task 3 | Deferred |
| Bulk snapshot operations | `MAMBA_SNAPSHOT_RESTORATION_PLAN.md` Phase 3 Task 5 | Not started |
| Performance optimizations | KHA-7 through KHA-11 | Identified via static analysis, not profiled on running system |

### Undocumented (built but not in user-facing docs)

- ~14 server configuration flags (`snapshot_retention_count`, `snapshot_trigger_policy`, `max_warm_conversations`, `conversation_*_timeout`, `enable_cross_session_refs`, `agent_tool_timeout`, etc.)
- `skills/mamba-sglang/` directory
- `docs/migration-prep/` operational docs
- `test/phases/config.sh` test configuration
- `.beads/` issue tracking
- `test/registered/radix_cache/` gauntlet tests

### Broken / Problematic

| Issue | Severity |
|-------|----------|
| `docs/stateful_mamba/README.md` says Phase 2 "Coming Soon" but it's done | Medium (stale docs) |
| License mismatch: SKILL.md says MIT; upstream is Apache 2.0 | Low |
| Author mismatch: SKILL.md says "Orchestra Research"; repo is KHAEntertainment | Low |
| Model name mismatch: SKILL.md uses `granite-4.0-h-small`; CLAUDE.md uses `granite-4.0-h-tiny` | Low |
| `--snapshot-auto-restore` flag (default True) gives false impression on branches without PR #6 | Low |

---

## Changes Made to Agent Instructions

### CLAUDE.md — needs update (not yet modified)

Identified drift:
1. **Missing `tier_manager.py`** from architecture section — now a core component with `restore_latest_snapshots_to_warm_tier()`
2. **Missing test phase infrastructure** — `test/phases/` is the primary testing approach but not mentioned
3. **Missing `.beads/`** tracking
4. **Missing core-memory Linear access pattern** — required for session start
5. **KHA-5 still in Active Issues** — it's done on `main`
6. **Missing sm75+ hardware blocker** from Known Issues
7. **Missing `test/registered/radix_cache/`** from test section
8. **Missing `snapshot_hooks.py`** from architecture
9. **Missing `conversation_tracker.py`** from architecture

### AGENTS.md — does not exist, needs creation

Required peer file to CLAUDE.md for non-Claude agent tools (Gemini CLI, Codex, etc.). Must stand alone with same core information.

### skills/mamba-sglang/SKILL.md — exists, minor issues

- Model name: `ibm-granite/granite-4.0-h-small` should be `granite-4.0-h-tiny`
- Author: "Orchestra Research" should probably reference KHAEntertainment
- Not linked from CLAUDE.md or any index

---

## Linear Audit Report

### Issues Reviewed: 10 (KHA-5 through KHA-14)

### Appears Done (still open — flag for manual close)

- **KHA-5**: Implement restore_snapshots_on_startup (Gap 3)
  - Evidence: `restore_latest_snapshots_to_warm_tier()` implemented in `tier_manager.py`, merged via PR #6 to `main`. Scheduler calls it on startup. Description still says "only logs found snapshots."
  - **Note**: Only done on `main`. Current branch (`fix/snapshot-restore-state-sync`) is 8 commits behind and lacks this. Merge or rebase needed.

### Stale (needs update — flag for human review)

- **KHA-6**: Phase 3.4 — Final Audit
  - Drift: Doesn't reference `test/phases/` infrastructure (9 phases, config.sh, codemap). Scope should include Phase 7 as critical validation. KHA-14 dependency not noted.
- **KHA-7**: [PERF] Remove setattr/getattr overhead in LRUList
  - Drift: Still Urgent, but premature until server phases pass. Micro-optimizing code that hasn't been validated end-to-end on real hardware.
- **KHA-8**: [PERF] Optimize tensor cloning — same premature concern
- **KHA-9**: [PERF] Cache tree depth — same premature concern
- **KHA-10**: [PERF] Optimize LRU traversal — same premature concern
- **KHA-11**: [PERF] Batch lock operations — same premature concern
- **KHA-12**: Document sglang.srt.models.mamba — references `phase3/docs/pending_updates.json`, verify path exists
- **KHA-13**: Document sglang.srt.layers.mamba — same path verification needed

### Accurate (no action needed)

- **KHA-14**: Clean up phase3 docs after Phase 3.4 completes — accurate and correctly Low priority

### Missing Issues (new gaps)

---

**Create AGENTS.md — peer file to CLAUDE.md**
- Type: Chore | Priority: High | Effort: S
- Why not covered: No existing issue covers this. AGENTS.md doesn't exist.

**Description:**
AGENTS.md is required for non-Claude agent tools (Gemini CLI, Codex, OpenCode). The `skills/mamba-sglang/SKILL.md` exists but serves a different purpose. AGENTS.md must mirror CLAUDE.md's core content and include the core-memory Linear access pattern.

**Context & Where to Start:**
- Read `CLAUDE.md` as canonical source
- Read `skills/mamba-sglang/SKILL.md` for Gemini-specific content
- Follow v1.3 cleanup prompt template structure

**Approach:**
1. Read CLAUDE.md fully
2. Create AGENTS.md with identical core sections
3. Add core-memory Linear access pattern (accountId: `0b4764e3-a793-4537-89b7-b26eff7b7675`)
4. Verify no contradictions between the two files

**Acceptance Criteria:**
- [ ] AGENTS.md exists at project root
- [ ] Stands alone — no "see CLAUDE.md" pointers
- [ ] Includes core-memory Linear session-start pattern

**Guardrails:**
- Do not reduce CLAUDE.md or SKILL.md

---

**Close KHA-5 and update CLAUDE.md Active Issues table**
- Type: Chore | Priority: High | Effort: XS
- Why not covered: KHA-5 is still Backlog but code is merged to main.

**Description:**
KHA-5 is implemented (PR #6 merged to main). Both Linear and CLAUDE.md need updating.

**Context & Where to Start:**
- Close KHA-5 in Linear
- Update CLAUDE.md lines 166 (Active Issues table)

**Acceptance Criteria:**
- [ ] KHA-5 closed in Linear
- [ ] CLAUDE.md reflects current state

---

**Update CLAUDE.md for current project reality**
- Type: Chore | Priority: Medium | Effort: S
- Why not covered: Multiple drifts after PRs #4 and #6.

**Description:**
CLAUDE.md is missing: tier_manager.py in architecture, test phase infrastructure, .beads/, core-memory Linear pattern, sm75+ blocker, gauntlet tests, and has stale KHA-5 reference.

**Context & Where to Start:**
- Current CLAUDE.md at project root
- `test/phases/README.md` and `codemap.md` for test infrastructure
- `git diff d7750d0..f9077a28b` for what changed

**Approach:**
1. Add tier_manager.py, snapshot_hooks.py, conversation_tracker.py to architecture
2. Add test phase infrastructure section
3. Add core-memory Linear access pattern
4. Add sm75+ blocker to Known Issues
5. Remove KHA-5 from Active Issues
6. Add gauntlet test reference

**Acceptance Criteria:**
- [ ] All new components from PRs #4 and #6 documented
- [ ] Core-memory Linear pattern included
- [ ] Hardware blocker documented
- [ ] Active Issues table current

---

**Merge main into current branch or switch to main**
- Type: Chore | Priority: Medium | Effort: XS
- Why not covered: Current branch is 8 commits behind main, missing PR #6 implementation.

**Description:**
`fix/snapshot-restore-state-sync` is behind `main` by 8 commits including PR #6 (startup restore), test fixes, and VM setup docs. The work this branch was created for (PR #4) is merged. Consider merging main in or switching to main for future work.

---

### Summary

The Linear backlog is **reasonably healthy** but stale. KHA-5 should be closed — the work is done on main. KHA-6 needs a scope update to reflect the expanded test infrastructure. All 5 PERF issues (KHA-7–KHA-11) are premature without server-phase validation on sm75+ hardware — they should be explicitly blocked or deferred. The biggest gaps are: (1) AGENTS.md doesn't exist, (2) CLAUDE.md has drifted from reality, (3) the current working branch is behind main. The recommended order is: close KHA-5 → merge main into current branch → update CLAUDE.md → create AGENTS.md → defer PERF issues.

---

## Recommended Next Session Starting Point

1. **Close KHA-5** in Linear (takes 30 seconds)
2. **Merge `main` into `fix/snapshot-restore-state-sync`** or switch to `main` for future work
3. **Update CLAUDE.md** with the corrections identified above
4. **Create AGENTS.md** as a peer to CLAUDE.md
5. **Add "Blocked by sm75+ validation" note** to KHA-7 through KHA-11
6. **Provision sm75+ GPU instance** and run Phase 7 (validates all gap fixes)

---

*Resync performed: 2026-03-28 | Mode: Sync | Agent: Claude Code (glm-5.1) | Linear project: SGLang - Mamba (e14f2152be8d)*
