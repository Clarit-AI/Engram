# KHA-360 — Hybrid-Class Stateful Recall Diagnostic Methodology

Disambiguates two confounds in post-registry recall failures observed on
hybrid (Mamba SSM + attention KV) models after the KHA-348 pending-restore
registry shipped.

Companion script: [`scripts/kha-360-diagnostic.sh`](../../scripts/kha-360-diagnostic.sh)

## Why this exists

KHA-348 closed the structural gap where snapshots were loaded but never
staged in any rid-indexed registry, leaving restored state unreachable to
`/v1/chat/completions` consumers. The fix was verified end-to-end on
pure-Mamba2 Codestral 7B v0.1.

A subsequent `validate-sync.sh` run on Granite-4.0-h-tiny (hybrid) produced
4/5 PASS with Test 4 (stateful recall) failing. Two confounds are present
and cannot be separated by the existing test:

- **(A) Surface-form / model-capability confound.** The recall question may
  pattern-match against refusal templates regardless of whether the model
  has the relevant state.
- **(B) Attention-KV restoration gap.** The registry restores SSM state,
  but hybrid models also have attention layers with KV cache. The snapshot
  system has no path for restoring that cache.

## Disambiguation logic

The diagnostic runs in two phases. The phases are not interchangeable; the
order is load-bearing.

### Phase 1 — Baseline (always first)

Establishes the fact and asks the recall question in a **single prompt**,
no role markers, no snapshot involvement. This isolates (A): if the model
cannot recall the fact even when the fact is unambiguously in the current
context, the recall-question surface form is the dominant signal and no
conclusion about the snapshot path is possible from this model under this
methodology.

If baseline fails, the diagnostic exits 3. Per the KHA-360 design note: *“If
baseline fails, the diagnostic methodology is on a model incapable of
demonstrating recall and the whole investigation needs a different
target.”* In that case, the next step is to design a hybrid-friendly recall
test surface form before retrying.

### Phase 2 — KV-restoration probe (only if Phase 1 passed)

Establishes the fact in turn 1, saves the snapshot, restores the snapshot,
and asks the recall question in a **separate** turn 2 with the fact absent
from the current prompt. If a pure-Mamba2 control (Codestral) passes but
the hybrid target fails under identical methodology, (B) is confirmed and
attention-KV restoration becomes a required snapshot-system capability.

## Target model

**Primary:** `nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16` — 23
Mamba-2 layers + 23 MoE layers + 6 GQA attention layers per KHA-338. Also
gates the CAI-8 humanoid demo, so a clean diagnostic here is dual-purpose.

**Control:** `mistralai/Codestral-Mamba-7B-v0.1` — pure Mamba-2, no
attention layers, so no KV restoration is possible or required. Acts as
the "snapshot path works" reference. Run with `--control`.

## Verdict matrix

| Baseline (target) | KV probe (control) | KV probe (target) | Verdict                                                                  |
| ----------------- | ------------------ | ----------------- | ------------------------------------------------------------------------ |
| FAIL              | n/a                | n/a               | Methodology redesign. Surface form dominates; pick a different test.     |
| PASS              | PASS               | PASS              | SSM-only restore is sufficient on hybrid; original 4/5 was surface-form noise. Close KHA-360. |
| PASS              | PASS               | FAIL              | **KV restoration gap CONFIRMED.** File implementation ticket for attention-KV snapshot support. |
| PASS              | FAIL               | any               | Control regression — investigate snapshot path itself before drawing any hybrid conclusion. |

## Cost envelope

Per the KHA-360 estimate: 2–4 H100/H200 hours total. The script's two-phase
design lets the operator abort cleanly after Phase 1 if baseline fails
without wasting Phase 2 GPU time.

## Acceptance-criteria mapping

KHA-360 lists four acceptance criteria. This diagnostic addresses:

1. *Granite baseline diagnostic completed; result determines whether surface form or KV restoration is the dominant factor.* — Phase 1 against the target. (KHA-338 noted Granite-4.0-h-tiny as an earlier target; the diagnostic accepts `--model` for either Granite or Nemotron-3-Nano-Omni.)
2. *If KV restoration is the issue: design proposal for extending the snapshot system to capture and restore attention KV alongside SSM state.* — Implementation work; out of scope for the diagnostic itself. The verdict matrix names this outcome explicitly.
3. *If surface form dominates: snapshot test methodology for hybrid models documented in SYNC_PLAYBOOK or test docs.* — This document.
4. *Nemotron-3-Nano-Omni recall behavior also re-tested under the chosen disambiguation methodology.* — Default target.

## Out of scope

- The registry implementation itself (closed by KHA-348 / PR #65).
- Attention-KV snapshot implementation (will be filed as a separate ticket if Phase 2 confirms the gap).
- Multi-tenant concurrency, cross-model warm-restore latency — see CS-146 workstreams 3 and 4.
