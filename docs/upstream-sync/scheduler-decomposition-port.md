# Scheduler Decomposition Port Plan

**Status:** Draft, pre-execution. Awaiting review.
**Created:** 2026-05-19
**Driving sync window:** upstream `b67400702` → upstream/main HEAD (`b911fd167`), 168 commits, includes the v0.5.12 tag.
**Workstream type:** Multi-day forward port. Sequenced AFTER PR #71 (engram-v0.4.0, 509-commit Phase 3b window) merges. Blocks the next upstream sync.

## 1. Context

The 168-commit window includes a coherent upstream refactor wave decomposing the `Scheduler` god-class into ~15 dedicated component classes. A mechanical merge resolution would restore ENGRAM logic onto a Scheduler shell that no longer owns the relevant call sites — silent semantic corruption that would not surface until H200 runtime.

This document plans the forward port: re-home each of our 18 ENGRAM blocks (15 in `scheduler.py` + 3 in the to-be-deleted `scheduler_output_processor_mixin.py`) onto upstream's component architecture, on a clean tree, one commit per re-homing. Once landed on `main`, the next sync should encounter dramatically fewer conflicts because the architectural surface will already match upstream's component shape.

## 2. Upstream refactor — PR series

All from the 168-commit window. The core decomposition lives in PRs `#25516`–`#25756`. New subdirectory: `python/sglang/srt/managers/scheduler_components/`.

| Component file (new) | Replaces (deleted/folded) | Concern |
|---|---|---|
| `scheduler_components/batch_result_processor.py` | `scheduler_output_processor_mixin.py` (DELETED, PR #25637) | Batch result handling, output finalization |
| `scheduler_components/output_streamer.py` | inline `scheduler.py` streaming code | Per-token output streaming |
| `scheduler_components/output_sender.py` | inline `SenderWrapper` in `scheduler.py` | ZMQ output transport |
| `scheduler_components/request_receiver.py` | inline request ingress in `scheduler.py` | Incoming `Req` lifecycle |
| `scheduler_components/logprob_result_processor.py` | inline logprob state in `scheduler.py` | Logprob accumulation |
| `scheduler_components/metrics_reporter.py` | `SchedulerMetricsMixin` (DELETED) | Metrics emission |
| `scheduler_components/profiler_manager.py` | `SchedulerProfilerMixin` (DELETED) | Profiler controls |
| `scheduler_components/weight_updater.py` | `SchedulerUpdateWeightsMixin` (DELETED) | Weight-update RPC |
| `scheduler_components/invariant_checker.py` | `scheduler_runtime_checker_mixin` (DELETED) | Runtime invariant checks |
| `scheduler_components/ipc_channels.py` | inline ZMQ socket setup in `scheduler.py` | IPC socket container |
| `scheduler_components/idle_sleeper.py` | inline `IdleSleeper` in `scheduler.py` | Idle-mode sleep coordination |
| `scheduler_components/load_inquirer.py` | inline queue-load code in `scheduler.py` | Queue-load reporting |
| `scheduler_components/kv_events_publisher.py` | inline KV-event code in `scheduler.py` | KV-cache event emission |
| `scheduler_components/pool_stats_observer.py` | inline pool-stats in `scheduler.py` | Pool-stats sampling |
| `scheduler_components/dp_attn.py` | inline DP-attention methods in `scheduler.py` | DP-attention adapter |
| `scheduler_components/flush_wrapper.py` | inline pending-flush state in `scheduler.py` | Pending-flush bookkeeping |
| `scheduler_components/new_token_ratio_tracker.py` | inline new-token-ratio state in `scheduler.py` | New-token-ratio state |

Additional structural changes affecting fork surface:

- `init_cache_with_memory_pool()` extracted to `mem_cache.kv_cache_builder.build_kv_cache()` (PRs #25601–#25608).
- Module-level helpers (incl. `EmbeddingBatchResult`) moved out of `scheduler.py` (PR #25638).
- `ModelWorkerBatch` indirection removed (PR #25516).

## 3. ENGRAM block inventory

### 3a. `scheduler_output_processor_mixin.py` (3 blocks, file slated for deletion)

| # | Lines | Header | Purpose | Target component |
|---|---|---|---|---|
| M1 | 399–439 | diffusion LLM batch result processing | Routes diffusion-model batch results through DLLM-specific path before the standard result handler | `scheduler_components/batch_result_processor.py` — open question whether `SchedulerDllmMixin` (existing) absorbs this instead; needs upstream batch_result_processor.py read during execution |
| M2 | 619–644 | snapshot finished-request state before cache release | Captures Mamba snapshot state when a `Req` finishes, before its cache slot is released | `scheduler_components/batch_result_processor.py` (hook in result finalization path) |
| M3 | 1083–1120 | restore_snapshot stateful-generate output routing | Diverts output for stateful-generate requests when servicing a restore | **Open question:** `scheduler_components/batch_result_processor.py` OR `scheduler_components/output_streamer.py` — depends on which class owns the actual output dispatch site post-refactor |

### 3b. `scheduler.py` (15 BEGIN markers across 14 logical blocks)

| # | Lines | Header | Purpose | Target |
|---|---|---|---|---|
| S1 | 25–28 | snapshot restore request IDs | `import uuid` for restore RID minting | **Stays on `scheduler.py`** (or moves with whichever block consumes `uuid` — currently the registry/handlers) |
| S2 | 30–(unclosed) | pending restore registry | Imports `OrderedDict` for the registry. **Malformed: no END before L92's inline pair.** | Cleanup needed — re-scope to wrap only `OrderedDict` then close. After cleanup, stays on `scheduler.py` imports |
| S3 | 92 (inline) | snapshot request routing | Wraps a small group of fork-added `io_struct` imports inline | Stays on `scheduler.py` imports |
| S4 | 318–340 | pending restore registry entry + capacity | Module-level `PendingRestoreEntry` dataclass + `PENDING_RESTORE_REGISTRY_MAX` const | **NEW fork file:** `python/sglang/srt/managers/scheduler_pending_restore.py`. Aligns with upstream's PR #25638 pattern of moving module-level helpers out of `scheduler.py` |
| S5 | 425–430 | scheduler metrics configuration | Fork-only metrics config tweak in `Scheduler.__init__` | `scheduler_components/metrics_reporter.py` (config passed at construction) |
| S6 | 485–491 | snapshot and agent subsystems | `self.init_snapshot_system()` + `self.init_agent_system()` calls in `__init__` | **Stays on `Scheduler.__init__`** — these are top-level subsystem inits |
| S7 | 1021–1351 | snapshot system initialization and startup restore | ~330 lines: `init_snapshot_system`, `restore_snapshots_on_startup`, supporting methods | **NEW fork file:** `python/sglang/srt/managers/scheduler_snapshot_system.py` (lifecycle surface). Scheduler holds a thin `self.snapshot_system = SchedulerSnapshotSystem(...)` reference, constructed with explicit args (see §7 R5) |
| S7a | 1354–1382 | restore_snapshots_on_startup implementation | Nested inside S7. Implementation of startup-time restore loop | Co-located with S7 in `scheduler_snapshot_system.py` |
| S8 | 1353–2459 | snapshot save, list, info, restore, and eviction handlers | RPC handler methods for snapshot operations | **NEW fork file:** `python/sglang/srt/managers/scheduler_snapshot_handlers.py` (RPC handler surface). Split from S7's lifecycle file to keep both under ~700 lines (see §7 Q3 resolution) |
| S9 | 2461–2531 | agent tool framework initialization | `init_agent_system()` and related setup | **NEW fork file:** `python/sglang/srt/managers/scheduler_agent_system.py` |
| S10 | 3009–3015 | snapshot request handlers | Dispatch into the recv-from-rpc loop for snapshot-typed requests | **Open question:** stays on `Scheduler.recv_requests` OR moves to `scheduler_components/request_receiver.py` — depends on where dispatch landed in `request_receiver.py` |
| S11 | 3575–3577 | conversation tracking on requests | Small hook setting `conversation_id` on incoming `Req` | **Open question:** `scheduler_components/request_receiver.py` if request-level field assignment moved there; else stays on Scheduler |
| S12 | 3677–3698 | pending-restore registry hydration | Hook on incoming request to consume a staged pending-restore entry | **Open question:** `scheduler_components/request_receiver.py` OR stays on Scheduler depending on where the ingress path lives. Same call-site cluster as S10/S11 |
| S13 | 4691–4693 | automatic snapshot hook triggering | Tiny hook in batch-result handling that decides whether to fire a snapshot | `scheduler_components/batch_result_processor.py` |
| S14 | 4700–4737 | post-forward snapshot hook runner | Runs after each forward pass to drive snapshot side-effects | `scheduler_components/batch_result_processor.py` (most likely) — confirm against upstream BRP API during execution |

## 4. Pre-existing marker hygiene

Found during the merge investigation, pre-dates this batch — fix as part of the port, not a separate cleanup PR:

- **L29 orphan END** in `scheduler.py` imports area — duplicate `# --- END ENGRAM ---` immediately after the legitimate closer at L28. Origin: likely from earlier hygiene churn (post-PR #70).
- **L30 unclosed BEGIN** — `# --- BEGIN ENGRAM: pending restore registry ---` opens but no matching END appears before L92's inline pair. Currently mis-scopes the entire fork-added stdlib import block.

Cleanup approach: re-scope the L30 BEGIN to wrap only the `from collections import OrderedDict` line (since `OrderedDict` is the fork-added symbol; `deque` was pre-existing), close immediately, and delete the L29 orphan.

## 5. Porting order

**Revised 2026-05-20: split into extraction PR (now) and re-homing PR (post-sync).** The port's component-class targets — `batch_result_processor.py`, `metrics_reporter.py`, `request_receiver.py`, etc. — do not exist in our fork at v0.5.0. They live in the 168-commit window past v0.5.0 (upstream commit `99ad2b089` and the surrounding PR cluster). Plumbing-confirmed: those paths are absent on `main`, present on `upstream/main`. Doing the entire port on a single branch off v0.5.0 is therefore impossible as originally specified.

The fix is to reorder: do the **fork-only extractions now** (Phases B + C + E + F — files that exist in our fork), let the 168-commit sync arrive the new component files in their upstream shape, then do the **component re-homing as a follow-up workstream** (Phase A + Phase D component parts) against the merged tree.

### 5a. Extraction PR — now, off v0.5.0

Phases B + C + E + F. Pure semantics-preserving extraction: move code into new fork-only files, leave thin references on `Scheduler`, no behavior change. DRAFT PR pending GPU smoke (same draft-then-validate pattern as PR #71).

**Phase B — Module-level fork helpers (1 commit).** Pure extraction, no upstream conflict surface.

1. Extract S4 (`PendingRestoreEntry` + capacity const) into new `scheduler_pending_restore.py`. Update `scheduler.py` imports.

**Phase C — Fork-only subsystems (3 commits).** Bulk extraction, mechanical; thin references left on Scheduler. Split between lifecycle and handler surfaces to keep each new file navigable (§7 Q3).

2. Extract S7 + S7a (snapshot system lifecycle: `init_snapshot_system`, `restore_snapshots_on_startup`, supporting methods) into `scheduler_snapshot_system.py`. ~330 lines.
3. Extract S8 (snapshot RPC handlers: save/list/info/restore/eviction) into `scheduler_snapshot_handlers.py`. ~1100 lines.
4. Extract S9 (agent tool framework) into `scheduler_agent_system.py`. ~70 lines.

**Phase E — Scheduler.__init__ wiring (1 commit).** Wire `self.snapshot_system`, `self.snapshot_handlers`, `self.agent_system`, `self.pending_restore_registry` into `__init__`. S6's call sites become thin delegations. Explicit-args construction per §7 R5.

5. Update `Scheduler.__init__` for the new subsystem references.

**Phase F — Marker hygiene (1 commit).** Final pass.

6. Fix L29 orphan END + L30 unclosed BEGIN in `scheduler.py`. Re-scope L30 to wrap only `OrderedDict`. Verify global block balance after all extraction commits.

**Extraction PR total: 6 commits** (one per phase listed above). Each commit is independently AST-parseable and block-balanced. Per-commit CI gates per §6.

**What stays in `scheduler.py` / mixin through the extraction (semantics-preserving):** S1/S2/S3 imports (re-pointed at new fork files), S5 metrics config, S6 init delegations (now thin), S10/S11/S12 request-path hooks, S13/S14 batch-result hooks. M1/M2/M3 stay in `scheduler_output_processor_mixin.py`. These are the small surgical hooks that the re-homing PR will move later, once their target component files exist in the fork.

### 5b. 168-commit upstream sync — between extraction and re-homing

Not part of this port. The sync runs as its own workstream after the extraction PR merges. The extraction's payoff is that `scheduler.py`'s 47-commit upstream refactor lands against a `scheduler.py` that is ~1500 lines slimmer (snapshot/agent/pending-restore content gone), shrinking the conflict surface dramatically. The sync arrives `scheduler_components/batch_result_processor.py`, `metrics_reporter.py`, `request_receiver.py`, `output_streamer.py`, and the rest.

### 5c. Re-homing PR — follow-up, post-sync

Phase A + Phase D component parts. Now-tractable because target files exist in the merged fork.

7. Phase A: M1 → `scheduler_components/batch_result_processor.py`.
8. Phase A: M2 → `scheduler_components/batch_result_processor.py`.
9. Phase A: M3 → `batch_result_processor.py` first attempt; relocate to `output_streamer.py` if read shows per-token dispatch (§7 OQ1). Delete `scheduler_output_processor_mixin.py` in the same commit (file's last block).
10. Phase D: S13 → `scheduler_components/batch_result_processor.py`.
11. Phase D: S14 → `scheduler_components/batch_result_processor.py`.
12. Phase D: S5 → `scheduler_components/metrics_reporter.py`.
13. Phase D: S10/S11/S12 → `scheduler_components/request_receiver.py` (1 commit if bundled, up to 3 if the read shows they split per §7 OQ2).
14. Phase D (contingent): M3 relocation to `output_streamer.py` if Phase A first attempt was wrong.

**Re-homing PR total: 7–10 commits** depending on S10/S11/S12 bundling and M3 contingent relocation.

### 5d. Combined totals

Extraction (6) + re-homing (7–10) = **13–16 commits across two PRs**, same total as the original single-PR plan. Each commit independently AST-parseable and block-balanced. CI gates per commit: `python -m py_compile` on touched files + `scripts/policy/check_protected_paths.py` invocation + `engram-markers` block balance check (need to extract from existing tooling; if not standalone, run on the full repo).

## 6. Validation strategy

Per the no-local-test rule in effect across the sync workstream:

- **Per commit:** `python -m py_compile` on every touched `.py` file. ENGRAM block balance check on every touched file (BEGIN count == END count). `scripts/policy/check_protected_paths.py` invocation if any protected paths are touched.
- **End of port (pre-PR):** Full-repo block balance check. Full-repo `python -m py_compile` smoke (catch import cycles introduced by new fork files). `engram-markers --strict` if such a tool exists; otherwise `grep -c` based check.
- **PR review gate:** Same draft-PR pattern as PR #71. Pytest remains env-blocked locally; CI runs the full suite on push. PR description must enumerate the 13–16 commits with their target files and surface explicit attention to the open-question resolutions in Phase D.
- **Post-merge (H200):** Re-run KHA-360 diagnostic harness (already scaffolded on `feat/kha-360-diagnostic-harness`) against the ported tree before the next upstream sync attempts. Snapshot save/restore is the load-bearing fork capability; if any of M2, M3, S7, S8, S12, or S13/S14 was re-homed incorrectly, the diagnostic surfaces it. **Confirm KHA-360 harness readiness before Phase A starts; flag any gaps as a blocking parallel workstream.**

### Rollback strategy

The per-commit structure of Phases A–F enables clean rollback if a later phase surfaces a porting mistake from an earlier one. Use `git revert <SHA>` on the specific bad commit rather than amending or rebasing — the port PR's commit history is the audit trail for the re-homing decisions, and rewriting it loses that signal. If revert reveals the mistake spans multiple commits, revert the cluster and append corrected commits at the tip.

### Progress checkpoints

After each Phase (A, B, C, D, E, F) lands, post a status comment on the port PR with:

- Commits landed in the phase (SHA via `git rev-parse HEAD` + one-line subject via `git show -s --format=%s <sha>`; per the verification protocol below)
- Any deviations from this plan
- Open questions resolved during execution + how
- Block balance after the phase

Enables overseer review without re-reading the full diff at each checkpoint.

### Verification protocol (rtk-safe)

`rtk` (the token-optimized CLI proxy active in this environment) caches verbose porcelain summaries and can serve stale or falsified values. Plumbing / machine-readable forms pass through raw (diagnostic-confirmed 2026-05-20; widened 2026-05-20-pm after a −2169 vs −1524 mismatch surfaced from `git diff --stat`). The cache surface is **broader than `git log` alone** — any verbose porcelain summary is suspect (`git log`, `git diff --stat`, `git show --stat`, and likely `git status` long-form). Therefore:

**HARD RULE: porcelain summary forms are BANNED for any verification or decision-gating check during the port.** "Prefer plumbing" relies on remembering every time — across 13–16 commits and six phases, that's a slip waiting to happen. State inspection uses plumbing / machine-readable forms ONLY:

| Need | Use (safe) | Banned (rtk-suspect) |
|---|---|---|
| Current HEAD | `git rev-parse HEAD` | `git log -1 --format=…` |
| Confirm a commit's contents | `git cat-file -p HEAD` (or `-p <sha>`) | `git show <sha>` (porcelain summary) |
| Current branch ref | `git symbolic-ref HEAD` | — |
| Tag / ref resolution | `git show-ref --tags` ; `git rev-parse <ref>` | — |
| Commit count in a range | `git rev-list --count <range>` | `git log --oneline … \| wc -l` |
| File contents at a ref | `git show <ref>:<path>` | — |
| Diff magnitudes (additions/deletions per file) | `git diff --numstat <range>` | `git diff --stat <range>` / `git show --stat <sha>` |
| Files changed in a range | `git diff --name-only <range>` | `git diff --stat <range>` |

For human-readable history or summaries ONLY (never to gate a decision or propagate as a figure): `rtk proxy git log …` / `rtk proxy git diff --stat …` bypass the cache. Any number quoted in a PR description, SESSION_STATE entry, commit message, or status report must come from a `--numstat` / `--name-only` / `--format` / plumbing source — never from the visual `--stat` summary line.

**Per-phase backstop (memory-independent):**

- At each phase boundary, capture HEAD via `git rev-parse HEAD`. Record the SHA in the PR comment AND `s3://engram/coordination/SESSION_STATE.md`.
- Before the next phase starts, re-confirm `git rev-parse HEAD` equals the last recorded SHA. Mismatch = STOP and escalate (real divergence in the worktree, not a display artifact).
- Block-balance checks use `grep` on files directly (not `git`) — unaffected by rtk, keep as-is.

This protocol is not optional. The rtk caching issue is tracked separately as an out-of-band tooling bug; for the port window we work around it with this discipline.

## 7. Resolved decisions and remaining open questions

### Phase applicability audit (added 2026-05-20 mid-extraction)

After the §5 re-architecture split the port into extraction PR + re-homing PR, every resolved decision and open question was audited and tagged for scope. Tagging prevents a later phase from hitting an inapplicable instruction. Tags:

- **[EXTRACTION]** — applies to this PR. Files involved exist on v0.5.0; the work is achievable now.
- **[RE-HOMING/SYNC]** — deferred. Involves upstream component files or sync-introduced changes that don't exist in our fork yet.

| Item | Tag | Notes |
|---|---|---|
| **Q3 — S8 file split** (Resolved) | [EXTRACTION] | `scheduler_snapshot_handlers.py` is created in Phase C.2. Split decision is load-bearing for this PR. |
| **Q4 — Marker policy on fully fork-authored files** (Resolved) | [EXTRACTION] | All four new fork files (`scheduler_pending_restore.py`, `scheduler_snapshot_system.py`, `scheduler_snapshot_handlers.py`, `scheduler_agent_system.py`) created in this PR follow the header-only pattern. Already applied to files committed so far (Phase B + Phase C.1). |
| **R5 — Extraction style** (Resolved + re-scoped 2026-05-20) | [EXTRACTION] *(module-level relocation part)* + [RE-HOMING/SYNC] *(explicit-args attribute migration part)* | Extraction PR uses module-level function relocation (already applied in Phase C.1). The original "explicit-args, no back-ref" intent applies to the re-homing PR alongside component integration. R5 is the only decision that legitimately spans both scopes. |
| **OQ1 — M3 target ambiguity (BRP vs output_streamer)** | [RE-HOMING/SYNC] | Both `batch_result_processor.py` and `output_streamer.py` live in upstream's `scheduler_components/` subdirectory — neither exists in our fork at v0.5.0. M3 itself stays in the mixin through this PR. |
| **OQ2 — S10/S11/S12 target ambiguity (request_receiver)** | [RE-HOMING/SYNC] | `request_receiver.py` is an upstream component that doesn't exist in our fork yet. The three blocks stay on `Scheduler` through this PR. |
| **Risk 3 — KV-cache builder relocation** | [RE-HOMING/SYNC] | `init_cache_with_memory_pool → build_kv_cache` is an upstream PR #25601–#25608 change in the 168-commit window past v0.5.0. Not in our fork yet. Audit happens post-sync. |
| **Risk 4 — EmbeddingBatchResult relocation** | [RE-HOMING/SYNC] | Upstream PR #25638 moved this dataclass out of `scheduler.py`. The destination doesn't exist in our fork yet. Our `PendingRestoreEntry` (Phase B) is already in its own fork file regardless of what upstream does with `EmbeddingBatchResult`. |

**§5 audit:**

- **§5a (extraction PR)** — [EXTRACTION], the entire active workstream.
- **§5b (168-commit sync)** — [RE-HOMING/SYNC] context; runs as its own workstream between #3 and #5 in §9.
- **§5c (re-homing PR)** — [RE-HOMING/SYNC] entirely; out of scope for this PR.
- **§5d (combined totals)** — bookkeeping; both scopes.

**§6 audit:**

- Per-commit / end-of-port / PR review gate / post-merge GPU smoke / rollback / progress checkpoints — all [EXTRACTION]-applicable as written. The rtk-safe verification protocol is mandatory for this PR.
- The "post-merge (H200)" line references KHA-360 harness readiness — that gating is for the **eventual port completion (re-homing PR)**, not the extraction PR. Extraction PR's smoke is its own thing (semantics-preservation check, not a full snapshot save/restore cycle). Flagging this as a soft mismatch worth noting: §6 originally framed the H200 gate as one event; with the split, the extraction PR's smoke and the re-homing PR's KHA-360 diagnostic are two distinct events. No doc edit needed beyond this audit row — the §9 sequencing already implies the split.

**Items that did not bucket cleanly:**

- **R5** spans both scopes (noted above) — flagged in the table, intentional. Today's re-scoping is what made R5 split.
- **§6 post-merge gate** spans both scopes (noted above) — flagged in the audit, no edit needed.

### Resolved decisions

- **Q3 — S8 file split (resolved: split).** S7+S7a+S8 combined is ~1430 lines, past the navigability threshold for a single file. Split at the lifecycle/handler seam: `scheduler_snapshot_system.py` (lifecycle, ~330 lines) and `scheduler_snapshot_handlers.py` (RPC handlers, ~1100 lines). Reflected in §3b targets and §5 Phase C (now 3 commits).
- **Q4 — Marker policy on fully fork-authored files (resolved: header only).** Precedent: `python/sglang/snapshot.py` (added 2026-04-24) uses a top-of-file `# ENGRAM_MODIFIED — <description>` header and no internal `# --- BEGIN ENGRAM ---` / `# --- END ENGRAM ---` blocks. Rationale: when the entire file is fork-authored, internal blocks would wrap everything and be redundant. **Apply this pattern to all four new fork files added in this port:** `scheduler_pending_restore.py`, `scheduler_snapshot_system.py`, `scheduler_snapshot_handlers.py`, `scheduler_agent_system.py`. Top-of-file header only.
- **R5 — Snapshot system extraction style (re-scoped 2026-05-20: applies to RE-HOMING PR, not extraction PR).** Originally resolved as "explicit-args, no self.scheduler back-ref" when the port was conceived as a single-PR end-to-end effort. With the §5 re-architecture splitting the work into extraction-first (§5a) and re-homing (§5c), R5's "explicit-args" decision belongs to the re-homing PR — that's where attribute-ownership migration aligns naturally with the call-site re-homing onto upstream's `scheduler_components/` classes, and where the next-sync resilience payoff actually applies. The extraction PR uses **module-level function relocation** instead: new fork files expose `init_X(scheduler)` / `restore_X(scheduler)` module functions; the `Scheduler` keeps the method names as thin delegators; the ~10 subsystem attributes stay on the `Scheduler` object; the ~50 cross-subsystem call sites elsewhere in `scheduler.py` are UNCHANGED. This is pure logic relocation, no attribute-ownership migration, no behavior change. The re-homing PR then performs the explicit-args migration informed by upstream's component shape. R5 is relocated, not deleted.

### Remaining open questions

1. **M3 target ambiguity.** "restore_snapshot stateful-generate output routing" could live in `batch_result_processor.py` (if it's a result-finalization hook) or `output_streamer.py` (if it's a per-token dispatch hook). Decided during Phase A by reading both upstream files. Worst case: the hook genuinely spans both, requires splitting into two ports. **Resolution: first port attempted in Phase A to BRP; relocation to `output_streamer.py` is a contingent Phase D step (§5 step 12) if the Phase A read shows it's a per-token dispatch hook.**
2. **S10/S11/S12 target ambiguity.** Three blocks all sit in the same call-site cluster (request ingress / req-creation path). Whether they relocate to `request_receiver.py` or stay on `Scheduler` depends on how thinly `request_receiver.py` was carved out. Decided during Phase D by reading `request_receiver.py`.
3. **Risk: KV-cache builder relocation.** `init_cache_with_memory_pool` → `build_kv_cache` may have moved fork-relevant initialization off the Scheduler that we expected to be there. Not directly an ENGRAM-block port, but the snapshot system's interaction with the cache surface is load-bearing. Audit during Phase C.
4. **Risk: `EmbeddingBatchResult` relocation.** Upstream moved this dataclass out of `scheduler.py` (PR #25638). Our `PendingRestoreEntry` lives adjacent in the current scheduler.py. Confirm during Phase B that the new location upstream chose for `EmbeddingBatchResult` is compatible with also housing `PendingRestoreEntry`, OR keep them in separate fork-only files (the resolved Q4 marker policy means a fork-only file is cheap to add).

## 8. Out of scope for this plan

- Actual code changes. This doc only.
- Validation of the broader 168-commit window's non-scheduler conflicts. Those are tractable mechanically once the scheduler-decomposition port lands and a fresh sync branch is attempted.
- Re-evaluation of the snapshot system's design. The port preserves current fork semantics; no behavioral changes.

## 9. Sequencing

**Revised 2026-05-20: sync sits BETWEEN extraction and re-homing.** Original sequencing placed the entire port before the next sync. Plumbing-confirmed reality: the port's component-class targets don't exist in our fork at v0.5.0 — they arrive with the 168-commit sync. The extraction-first split (§5) lets the load-bearing slim-down of `scheduler.py` land before the sync, then re-homing onto the merged component files becomes a clean follow-up.

| Order | Workstream | Status |
|---|---|---|
| 1 | PR #71 (engram-v0.5.0, Phase 3b 509-commit window) | MERGED 2026-05-20; tagged `engram-v0.5.0` at `65c1ecf6e` |
| 2 | This planning doc (PR #74 + PR #75 §6 amendment) | MERGED 2026-05-20 |
| 3 | **Extraction PR** (PR off `port/scheduler-decomposition`, branched from `main` at `03ff82942`). Phases B + C + E + F. DRAFT pending GPU smoke. ~6 commits. | **In progress — this is the current workstream** |
| 4 | 168-commit upstream sync (+ accumulated drift since 2026-05-15). Lands against a slimmed `scheduler.py`; arrives `scheduler_components/` to the fork. | Blocked on #3 merge |
| 5 | **Re-homing PR** (post-sync). Phase A (M1/M2/M3 + mixin delete) + Phase D component parts (S5, S13/S14, S10/S11/S12). ~7–10 commits. | Blocked on #4 merge |

**Why this ordering pays off:** the extraction in #3 moves ~1500 lines of fork-only snapshot/agent/pending-restore content out of `scheduler.py` into files upstream doesn't touch. When the 47-commit upstream refactor of `scheduler.py` arrives in #4, it lands against a much slimmer file. The small surgical hooks that remain in `scheduler.py` (S5, S10/S11/S12, S13/S14) and the mixin (M1/M2/M3) wait for #5, when their target component files exist in the merged fork.
