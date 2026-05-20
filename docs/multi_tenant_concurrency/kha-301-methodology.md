# KHA-301 — Multi-Tenant Concurrency Validation Methodology

Scope: SSM state allocation under concurrent session load on a single
deployment. Orthogonal to the multi-GPU / tensor-parallelism axis covered
by KHA-189; this ticket exercises *many sessions sharing one deployment*,
each with independent state.

Companion script: [`scripts/concurrency/concurrent-smoke.sh`](../../scripts/concurrency/concurrent-smoke.sh)

## Why this exists

SGLang's transformer-concurrency story (RadixAttention prefix sharing,
continuous batching, KV eviction) is transformer-specific. The SSM path is
structurally different in three ways that bite under multi-tenancy:

1. **SSM state is per-session and non-shareable.** No RadixAttention
   analogue for hidden state — two users asking the same first question
   produce two distinct SSM states.
2. **Snapshot restore is a VRAM allocation event.** `/restore_snapshot`
   places state into an active slot. On large hidden-state models
   (Qwen3-Coder-Next 23.7 GB, Nemotron-3-Super 5.3 GB) this arithmetic
   gets tight quickly under concurrent restore.
3. **The 3-tier `MambaHostPool` was not designed for heavy multi-tenant
   workloads.** All Engram benchmarks to date have been single-session.

## Five dimensions to cover (target benchmark surface)

The full ticket scope is five dimensions; this methodology doc names them
so subsequent commits can land each as its own runnable harness. The
smoke harness landing with this doc covers part of Dimension 1 + the
isolation half of Dimension 5 — first signal, not certification.

| # | Dimension | What it answers | Scope here |
| - | --- | --- | --- |
| 1 | Concurrent session scaling | tokens/s per session and p50/p95/p99 at 1/4/16/64/256/1024 concurrent sessions on the same deployment | smoke covers N=4 only |
| 2 | Snapshot restore under concurrency | restore-latency delta when 16 sessions are mid-generation vs cold-deployment baseline | out of scope for smoke |
| 3 | Tier-system behaviour under pressure | warm/demote/re-warm fairness; mixed-lifetime LRU correctness; cold→active promotion latency | out of scope for smoke |
| 4 | Memory fragmentation / leak detection | 24h sustained churn; orphan tier state; fragmentation accumulation | out of scope for smoke |
| 5 | Fairness and isolation | one hot session does not starve others; misbehaving sessions don't degrade the deployment | smoke probes cross-session crosstalk only |

## Smoke (this commit)

The script `concurrent-smoke.sh` runs four parallel chat sessions against
the active server, each with a distinct `conversation_id`. Each session:

1. Establishes a session-specific secret token via turn 1.
2. Saves a snapshot for its `rid`.
3. Asks the recall question in turn 2 (within the same `rid`).

Two checks:

* **Cross-talk isolation** — each session's turn-2 response must contain
  its own secret, never any of the other three. If two sessions interleave
  state, this catches it.
* **Snapshot-allocation feasibility** — four concurrent `/save_snapshot`
  + `/restore_snapshot` cycles must all return 200 within the same window.
  If the `MambaHostPool` thrashes or starves under the load, this catches
  it.

This is a "does it not crash and not crosstalk" smoke, not a benchmark.
Per-request throughput, p99 latency, and VRAM-tier behaviour all need
their own dedicated dimension-1 harness with a load generator and
percentile capture. The smoke is the floor — if the floor cracks, dimension
1 doesn't run.

### Targets

The smoke is target-agnostic — operator passes `--model <path>` and the
server is launched separately. Recommended initial targets:

* **Granite-4.0-h-tiny** — unconstrained on H100 80 GB (~56 MB state per
  session); useful for establishing the "snapshot-allocation feasibility"
  baseline without VRAM pressure.
* **Nemotron-3-Nano-Omni-30B-A3B-Reasoning BF16** — at
  `--mem-fraction-static 0.92` this leaves ~6 GB free; concurrent SSM
  allocation lives entirely in that headroom. Four concurrent sessions is
  feasible; sixteen would not be without tier demotion involved.

### What this smoke does NOT do

* No load generator. No percentile capture. No tier-eviction stress.
* No baseline-vs-stress restore-latency comparison.
* No long-running churn.
* No fairness scoring. (Smoke only asserts no crosstalk; it does not
  measure latency-per-session under contention.)
* No certification of the snapshot-allocation arithmetic in the ticket's
  "Theoretical VRAM concurrent ceiling" section. Closing that requires
  Dimension 1's full harness.

## Prerequisite: attention-KV gap

The KHA-360 diagnostic on 2026-05-20 confirmed the attention-KV
restoration gap on hybrid models. Until that gap is closed, snapshot
restore in this smoke will save/restore SSM state mechanically (which is
what the smoke tests) but will **not** carry attention-derived recall
across the save/restore boundary. That's expected here — the smoke is
about *allocation* and *isolation*, not *behavioural recall*. The recall
test surface is covered by KHA-360's harness.

## Next steps (after the smoke)

1. **Dimension 1 harness:** load generator + percentile capture, sweeping
   N ∈ {1, 4, 16, 64} against Granite-4.0-h-tiny first (no VRAM ceiling),
   then Nemotron-3-Super for the 12-session-class regime, then
   Qwen3-Coder-Next for the 3-4-session-class regime.
2. **Dimension 2 harness:** 16 sessions actively generating + 1 cold
   restore; measure restore-latency p99 under contention vs cold-baseline
   p99. Connects to KHA-370's credibility-figure question (2 ms is the
   cold-baseline; what's the under-contention figure?).
3. **Dimension 3+4 harnesses:** require sustained GPU time; not
   single-session-doable.

## Out of scope (per ticket framing)

* Multi-model-on-one-GPU (weight-memory competition is a different axis).
* MLPerf alignment (a separate methodology call).
* Engineering work on the tier-system itself (separate impl tickets — this
  ticket is *validation*, not *fix*).
