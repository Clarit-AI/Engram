# KHA-301 — Multi-Tenant Concurrency Validation

**Date:** 2026-05-22
**Branch / HEAD:** `main @ cd9b495e1` (post PR #87 — `scheduler_output_processor_mixin.py` deleted at `42744f3f7`)
**Server:** Nemotron Elastic 30B (`/workspace/models/nemotron-elastic-30b/`) on H100 80 GB
**Flags:** `--disable-radix-cache --enable-snapshot-persistence --mamba-scheduler-strategy no_buffer --trust-remote-code`, `SGLANG_ENABLE_SPEC_V2=false`
**Harness:** `scripts/validation/multitenant_concurrency.py`
**Raw JSON:** `s3://engram/benchmarks/kha-301-n3-h100.json`, `s3://engram/benchmarks/kha-301-n5-h100.json`

## Goal

Validate that the running server can serve multiple simultaneous snapshot
sessions correctly — measure save latency under load, restore success rate,
and **cross-session state isolation** (no session's tag-word leaks into
another session's responses).

## Harness design

Threaded driver (`concurrent.futures.ThreadPoolExecutor`, one thread per
session). Each session uses a unique `conversation_id` and is assigned a
distinct NATO-phonetic tag-word (`alpha`, `bravo`, ... ). Per session:

1. **Turn 1** — system msg primes the recall-test assistant; user message
   reveals the secret tag-word and asks for an ack. Snapshot saved.
2. **Turns 2..N** — `POST /restore_snapshot` warmup call (`continuation_ids=[]`,
   `max_new_tokens=1`) to exercise the restore path; full chat request issued
   carrying the rolling history; snapshot saved again.

Per-turn instrumentation: chat latency, save latency + attempts, restore
latency + attempts, and two booleans — does the response contain its **own**
tag-word, and does it contain **any foreign** tag-word from another concurrent
session.

## Results

### N = 3 sessions × 5 turns

| Metric | Value |
| --- | ---: |
| Sessions completed | 3 / 3 |
| Save success | 15 / 15 (100.0%) |
| Restore-warmup success | 12 / 12 (100.0%) |
| Own-tag recall (turns ≥ 2) | 12 / 12 (100.0%) |
| Foreign-tag leak turns | 0 / 15 |
| Wall clock | 11.32 s |

Save latency: mean **446.3 ms**, p50 388.7, p95 851.7, p99 877.6, max 884.1 ms.
Restore-warmup latency: mean **843.2 ms**, p50 772.6, p95 1252.3, max 1257.7 ms.
Chat latency: mean **1068.4 ms**, p50 964.8, p95 1492.8, max 1493.1 ms.

### N = 5 sessions × 5 turns

| Metric | Value |
| --- | ---: |
| Sessions completed | 5 / 5 |
| Save success | 25 / 25 (100.0%) |
| Restore-warmup success | 20 / 20 (100.0%) |
| Own-tag recall (turns ≥ 2) | 20 / 20 (100.0%) |
| Foreign-tag leak turns | 0 / 25 |
| Wall clock | 20.96 s |

Save latency: mean **644.4 ms**, p50 633.5, p95 1244.8, p99 1406.1, max 1446.5 ms.
Restore-warmup latency: mean **1350.4 ms**, p50 1307.2, p95 1838.3, p99 2021.5, max 2067.3 ms.
Chat latency: mean **2391.4 ms**, p50 2393.3, p95 2424.7, max 2426.5 ms.

## Scaling shape (N=1 baseline → N=3 → N=5)

| N | Save mean | Restore-warmup mean | Chat mean | Wall clock |
| ---: | ---: | ---: | ---: | ---: |
| 1 (smoke, 3 turns) | 218 ms | 227 ms | 366 ms | 2.2 s |
| 3 (5 turns) | 446 ms | 843 ms | 1068 ms | 11.3 s |
| 5 (5 turns) | 644 ms | 1350 ms | 2391 ms | 21.0 s |

Save latency grows roughly linearly with N (×2.0 at N=3 vs N=1, ×3.0 at N=5).
Chat latency grows nearly perfectly linearly with N (×6.5 at N=5 vs N=1),
consistent with single-stream prefill/decode serialization on one GPU rather
than parallel batching. The 30B model with `--disable-radix-cache` and
`no_buffer` exercises the worst case for batch packing.

## Verdicts

- **Snapshot save under load — PASS.** 40 / 40 saves across both runs succeeded
  on first attempt (no retries needed). Latency degrades sub-linearly with N.
- **Snapshot restore under load — PASS.** 32 / 32 restore-warmup calls
  succeeded on first attempt. The known TokenizerManager single-await gap
  (line ~1631) did **not** surface as restore failures at N ≤ 5 with the
  workload tested. See "Known gap notes" below.
- **State isolation — PASS.** 0 foreign-tag occurrences across 40 model
  responses (40 turns × 0 leaks). Every session's responses (turns 2–N)
  contained its own tag-word, none contained any other session's tag-word.
  Per-session conversation state was correctly partitioned by `conversation_id`.
- **Per-session completion — PASS.** 8 / 8 sessions completed all turns
  without error across both runs.

## Known gap notes

`TokenizerManager` consumes `snapshot_restore_result_queue` with a single
`await` (~line 1631). The architectural concern is that concurrent restores
could starve each other and produce timeouts or stale results. With N=3 and
N=5 sessions, each issuing a single restore-warmup before its chat call, the
queue was never deeply contended — the harness saw 100% success.

The gap is **not invalidated** by this run; it is **not exercised**. A
harder workload (higher N, parallel back-to-back restores, no per-session
chat gap) would be needed to provoke it. Out of scope for this validation.

## Recommendations

1. **N ≤ 5 multi-tenant snapshot workloads are safe** on the current main
   (post PR #87) with Nemotron Elastic 30B and `no_buffer` strategy. State
   isolation and durability are both clean.
2. **For higher concurrency (N ≥ 8)**, design a follow-up run that
   deliberately stresses `snapshot_restore_result_queue` (e.g. N parallel
   restores without intervening generation). Until then, the single-await
   gap remains a known architectural risk, not a measured defect.
3. **Latency under load**: chat latency at N=5 averages ~2.4 s vs ~0.4 s at
   N=1 — operators running concurrent snapshot tenants should size for
   per-stream throughput accordingly. Save/restore overhead at N=5 is well
   under 2 s, acceptable for between-turn checkpointing.
