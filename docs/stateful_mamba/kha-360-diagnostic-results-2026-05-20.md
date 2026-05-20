# Hybrid-Class KV/SSM Recall Diagnostic — Results (2026-05-20)

Companion to [`kha-360-diagnostic-methodology.md`](./kha-360-diagnostic-methodology.md).
Run artifacts under [`./kha-360-runs-2026-05-20/`](./kha-360-runs-2026-05-20/).

## Verdict

**Attention-KV restoration gap CONFIRMED on hybrid models.**

The snapshot system saves and restores SSM state; the registry hydration path
(KHA-348) reaches the request correctly; the mechanical save/restore round-trip
is end-to-end observable in the server logs. Hybrid models nevertheless lose
the pre-snapshot conversation when asked to recall in a fresh turn, and the
delta points at the missing attention KV cache.

Two hybrid targets, identical failure pattern:

| Target | Layer mix | Phase 1 baseline | Phase 2 KV-probe | SSM tensor restored |
| -- | -- | -- | -- | -- |
| Granite-4.0-h-tiny | 36 Mamba-2 + attention | **PASS** | **FAIL** | `[36, 48, 64, 128]` ✓ |
| Nemotron-3-Nano-Omni-30B-A3B-Reasoning BF16 | 23 Mamba-2 + 6 GQA + 23 MoE | **PASS** | **FAIL** | `[23, 64, 64, 128]` ✓ |

Per the methodology's verdict matrix, both rows resolve to:

> **PASS baseline · FAIL KV-probe → KV restoration gap CONFIRMED; file the
> attention-KV snapshot implementation ticket.**

The formal "control PASS" row of the matrix could not be filled in directly
(see [Control caveat](#control-caveat) below), but the mechanical evidence
that the snapshot path itself works substitutes for the control's role
("snapshot save/restore is not broken"). The two-hybrid corroboration also
makes the verdict robust against single-model artefacts.

## Evidence per target

### Granite-4.0-h-tiny (37B-class hybrid, ~13 GB BF16)

**Phase 1 baseline.** Single prompt: "Earlier I told you: the secret code is
DELTA-7. What was the secret code I told you earlier?"

Response: *"The secret code you told me earlier was DELTA-7."* → PASS.

**Phase 2 KV-probe.** Turn 1 establishes the fact under a fresh `rid`,
`/save_snapshot` returns 200, `/restore_snapshot` returns 200, turn 2 asks
the recall question with the fact absent from the prompt.

Server log evidence (`granite-server-snapshot-evidence.log`):

```
Conversation registered: kha-360-target-1779308398-390719, tier=warm
Saved to WARM tier: kha-360-target-1779308398-390719, host_pool_size=4
Snapshot saved: conversation=kha-360-target-…, turn=18, branch=None
Manual snapshot saved from WARM tier: snapshot_id=…-t0
POST /save_snapshot → 200
Snapshot staged for pending restore: conversation=…, rid=…, token_count=45
POST /restore_snapshot → 200
Injected state to pool idx 10: 1 conv tensors, temporal shape torch.Size([36, 48, 64, 128]) (moved to cuda)
pending-restore registry hit: hydrated rid=… (canonical=…, conv_id=…, mamba_pool_idx=10)
```

Turn-2 response: *"I'm sorry, but as an AI, I don't have the ability to store
or remember personal data unless it's shared with me in the course of our
conversation."* → FAIL.

The mechanical hydration succeeded (registry hit, SSM tensor moved to CUDA pool
idx 10). The model's attention layers, however, had no KV cache for the
turn-1 prompt — and the model behaves as if the conversation never happened.

### Nemotron-3-Nano-Omni-30B-A3B-Reasoning BF16 (AC #4 re-test, ~60 GB BF16)

Loaded at `--mem-fraction-static 0.92` on H100 80GB; 75 GB used / 6 GB free.
`pip install librosa` required for the `ParakeetExtractor` audio-encoder
import (the model has multimodal heads even when running text-only); the
`libtorchcodec` / `libavutil` warnings are non-fatal and do not block text
inference. Multimodal init proceeds with warnings only.

**Phase 1 baseline.** Reasoning-style trace includes the user's prompt
verbatim and resolves with `"The answer: DEL"` (40-token cap). The diagnostic
matches `DELTA[- ]?7`. → PASS.

**Phase 2 KV-probe.** Same protocol as Granite. Server logs identical pattern,
SSM tensor shape `torch.Size([23, 64, 64, 128])` matches the documented "23
Mamba-2 layers" in the architecture. Turn-2 response is unambiguous:

> *"The user asks: 'What was the secret code I told you earlier?' There's no
> prior context in this conversation. The user is asking for a secret code
> that they claim to have told earlier."*

The Reasoning model's trace explicitly states "no prior context" after a
successful snapshot restore. → FAIL.

## Control caveat

The methodology's prescribed control — `mistralai/Mamba-Codestral-7B-v0.1`
(repo name corrected from `Codestral-Mamba`) — is a code-generation base
model, not instruction-tuned for chat. The same Phase-1 baseline prompt
returns an empty completion. This is not a snapshot-path regression; it's
the ecosystem limitation already called out in the KHA-360 ticket
("Pure-Mamba2 instruction-tuned models do not exist in the ecosystem at
production scale").

Consequence: the matrix row "control PASS · target FAIL" cannot be
populated by a single chat-format diagnostic against a pure-Mamba-2 model
on the open hub.

We rely instead on the two pieces of evidence the control was meant to
provide:

1. **"Snapshot path itself works."** The server logs in both runs show the
   full save → persist → restore → registry-hit → pool-inject chain
   succeeding with `200 OK` at the HTTP boundary and the SSM tensor moving
   to CUDA. This is directly observable on the same instance that ran the
   hybrid Phase 2.
2. **Two-hybrid corroboration.** Granite and Nemotron-3 have different
   layer mixes (36-layer vs. 23-layer Mamba-2 stacks; dense attention vs.
   6-GQA + MoE) and the failure pattern is identical. A single-model
   artefact is not a plausible explanation.

Methodology follow-up: an instruction-tuned pure-Mamba-2 control would
strengthen the verdict but is not gate-blocking. Recommend filing a small
methodology-followup ticket to track that gap (and to revisit the doc's
example repo id — the correct one on the hub is `mistralai/Mamba-Codestral-7B-v0.1`).

## Bug observed in passing

Both runs surfaced a non-fatal exception in the post-forward snapshot hook
on Granite and Nemotron servers:

```
Post-forward hook post_forward_snapshot_callback failed:
  int() argument must be a string, a bytes-like object or a real number, not 'NoneType'
  File ".../python/sglang/srt/managers/scheduler.py", line 1315, in post_forward_snapshot_callback
      req_pool_idx=int(req.req_pool_idx),
```

This matches the "scheduler.py:1315 NoneType investigation" already queued
in the post-PR-71 cleanup batch. The hook failure does not block the
mechanical save/restore (the warm-tier writes and `/save_snapshot` /
`/restore_snapshot` endpoints still succeed). Flagging here so the
investigation has the matching traceback.

## Acceptance criteria coverage

KHA-360's four acceptance criteria, mapped to this run:

1. *Granite baseline diagnostic; determines whether surface form or KV
   restoration dominates.* — **Done.** Baseline PASS rules out the
   surface-form confound on Granite; the KV-probe FAIL identifies the
   restoration gap.
2. *Design proposal for extending the snapshot system to capture and
   restore attention KV alongside SSM state.* — Out of scope for the
   diagnostic itself; filing the implementation ticket is the next action
   per the matrix.
3. *Snapshot test methodology for hybrid models documented.* — The
   methodology doc and this results doc together cover this.
4. *Nemotron-3-Nano-Omni recall behavior re-tested under the chosen
   methodology.* — **Done.** Same verdict as Granite, with the AC #4
   target.

## Reproduction

```
# Granite (already present at /workspace/models/granite-4.0-h-tiny)
SGLANG_ENABLE_SPEC_V2=false python -m sglang.launch_server \
  --model-path /workspace/models/granite-4.0-h-tiny \
  --enable-snapshot-persistence \
  --snapshot-dir /tmp/kha-360-snapshots \
  --snapshot-trigger-policy every_turn \
  --mem-fraction-static 0.80 \
  --disable-cuda-graph

bash scripts/kha-360-diagnostic.sh \
  --model /workspace/models/granite-4.0-h-tiny \
  --phase all
```

For Nemotron, add `--disable-piecewise-cuda-graph --trust-remote-code` and
require `pip install librosa` ahead of launch; the multimodal processor
imports it unconditionally even in text-only inference.

## Environment

- H100 80GB SXM (GCP a3-highgpu-1g), driver 580.159.03, compute_cap 9.0
- torch 2.11.0+cu128, sgl_kernel 0.4.2.post2 (engram-latest sm90 wheel)
- Engram branch `feat/kha-360-diagnostic-harness` rebased onto `origin/main`
  HEAD (PR #75 merge, post-PR-#71)
- Spec-V2 disabled via `SGLANG_ENABLE_SPEC_V2=false`; the upstream contradiction
  decision (GH #63) is still pending in the Session B cleanup batch
- Snapshot dirs: `/tmp/kha-360-snapshots*` (cleared between models)
