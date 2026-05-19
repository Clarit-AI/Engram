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

Order minimizes cross-commit churn and front-loads the lowest-uncertainty work.

**Phase A — Mixin retirement (3 commits).** Clearest destinations; the file is being deleted upstream anyway, forcing the issue.

1. Port M1 (diffusion LLM batch result processing) → `batch_result_processor.py`.
2. Port M2 (snapshot finished-request state) → `batch_result_processor.py`.
3. Port M3 (restore_snapshot output routing) → `batch_result_processor.py`, and delete `scheduler_output_processor_mixin.py` in the same commit (the file's last remaining block). (Deletion bundles with M3's port since M3 is the file's last block.) If execution reveals M3 belongs in `output_streamer.py` instead of BRP, port it there instead — the BRP-vs-output_streamer read happens here, in Phase A.

**Phase B — Module-level fork helpers (1 commit).** Pure extraction, no upstream conflict surface.

5. Extract S4 (`PendingRestoreEntry` + capacity const) into new `scheduler_pending_restore.py`. Update scheduler.py imports.

**Phase C — Fork-only subsystems (3 commits).** Bulk extraction, mechanical; thin references left on Scheduler. Split between lifecycle and handler surfaces to keep each new file navigable (§7 Q3).

6. Extract S7 + S7a (snapshot system lifecycle: `init_snapshot_system`, `restore_snapshots_on_startup`, supporting methods) into `scheduler_snapshot_system.py`. ~330 lines.
6b. Extract S8 (snapshot RPC handlers: save/list/info/restore/eviction) into `scheduler_snapshot_handlers.py`. ~1100 lines.
7. Extract S9 (agent tool framework) into `scheduler_agent_system.py`. ~70 lines.

**Phase D — Component-class re-homing (4–7 commits, depending on open-question resolution).**

8. Port S13 (auto-snapshot trigger) → `batch_result_processor.py`.
9. Port S14 (post-forward snapshot hook) → `batch_result_processor.py`.
10. Port S5 (metrics config) → `metrics_reporter.py`.
11. Resolve open questions for S10, S11, S12 by reading `request_receiver.py`. Port each to its determined home — likely all three to `request_receiver.py`, bundled into one commit since they're a tight call-site cluster; up to 3 commits if the read shows they split.
12. (Contingent) If Phase A determined M3 belongs in `output_streamer.py` but couldn't fully relocate it without the Phase D request-path context, complete the relocation here. No-op if M3's Phase A home was correct.

**Phase E — Scheduler.__init__ wiring (1 commit).** Update `Scheduler` to instantiate the new fork subsystems alongside upstream's component instantiations.

13. Wire `self.snapshot_system`, `self.agent_system`, `self.pending_restore_registry` into `__init__`. S6's call sites become thin delegations.

**Phase F — Marker hygiene (1 commit).** Final pass.

14. Fix L29 orphan END + L30 unclosed BEGIN. Re-scope L30 to wrap only `OrderedDict`. Verify global block balance.

**Total: 13–16 commits.** Each commit is independently AST-parseable and block-balanced. CI gates per commit: `python -m py_compile` on touched files + `scripts/policy/check_protected_paths.py` invocation + `engram-markers` block balance check (need to extract from existing tooling; if not standalone, run on the full repo).

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

- Commits landed in the phase (SHA + one-line subject each)
- Any deviations from this plan
- Open questions resolved during execution + how
- Block balance after the phase

Enables overseer review without re-reading the full diff at each checkpoint.

## 7. Resolved decisions and remaining open questions

### Resolved decisions

- **Q3 — S8 file split (resolved: split).** S7+S7a+S8 combined is ~1430 lines, past the navigability threshold for a single file. Split at the lifecycle/handler seam: `scheduler_snapshot_system.py` (lifecycle, ~330 lines) and `scheduler_snapshot_handlers.py` (RPC handlers, ~1100 lines). Reflected in §3b targets and §5 Phase C (now 3 commits).
- **Q4 — Marker policy on fully fork-authored files (resolved: header only).** Precedent: `python/sglang/snapshot.py` (added 2026-04-24) uses a top-of-file `# ENGRAM_MODIFIED — <description>` header and no internal `# --- BEGIN ENGRAM ---` / `# --- END ENGRAM ---` blocks. Rationale: when the entire file is fork-authored, internal blocks would wrap everything and be redundant. **Apply this pattern to all four new fork files added in this port:** `scheduler_pending_restore.py`, `scheduler_snapshot_system.py`, `scheduler_snapshot_handlers.py`, `scheduler_agent_system.py`. Top-of-file header only.
- **R5 — Snapshot system extraction style (resolved: explicit-args).** Construct the new fork subsystem classes with explicit constructor args, not a `self.scheduler` back-reference. Adds work in Phase C but matches upstream's component discipline and pays back in next-sync resilience. Any cross-cutting Scheduler state that the snapshot/agent systems need is passed at construction time, not pulled through a back-ref.

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

| Order | Workstream | Status |
|---|---|---|
| 1 | PR #71 (engram-v0.4.0, Phase 3b 509-commit window) | In review, H100 validation pending |
| 2 | This planning doc (PR off `docs/scheduler-decomposition-port-plan`) | Draft — this file |
| 3 | Execution of the port (PR off `port/scheduler-decomposition`, branched from `main` AFTER #1 merges) | Not started — blocked on review of this doc + #1 merge |
| 4 | Next upstream sync (168 commits + accumulated drift) | Blocked on #3 |
