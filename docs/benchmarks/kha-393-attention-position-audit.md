# KHA-393 — Attention and Position-Handling Audit on Snapshot Restore

**Scope:** the five models validated for Engram snapshot/restore.
**Audit date:** 2026-05-21.
**Repo HEAD:** `a7e9057ec` (post-v0.5.2).
**Method:** static inspection of the model implementations under
`python/sglang/srt/models/`, the snapshot RPC handlers in
`python/sglang/srt/managers/scheduler_snapshot_handlers.py`, and the
snapshot serialization in `python/sglang/srt/snapshot/`.

## Background

Engram snapshots the recurrent SSM hidden state (Mamba `conv_state` +
`temporal_state`) and deliberately does **not** snapshot the attention
KV cache. For pure-Mamba models this is a complete save. For hybrid
models with attention layers, the attention KV cache restarts empty on
restore while the SSM layers pick up their restored state.

## Per-model summary

| Model | Class | Pos encoding (attn) | Attn / SSM / MoE layers | Restore position behavior | Snapshot size invariant |
|---|---|---|---|---|---|
| Granite 4.0-H | `granitemoehybrid.GraniteMoeHybridForCausalLM` | RoPE if `config.position_embedding_type == "rope"`, else NoPE. Granite-4.0-H ships with NoPE. | Hybrid; per layer-type config | N/A for attn (NoPE); SSM restored verbatim | Yes |
| Nemotron-Cascade-2-30B | `nemotron_h.NemotronHForCausalLM` (model_type `nemotron_h`) | **NoPE** — `NemotronHAttention.forward` calls `RadixAttention` directly with no rotary embedding (`models/nemotron_h.py:515-521`) | Per-layer mix declared in HF config | N/A for attn (NoPE); SSM restored verbatim | Yes |
| Nemotron-3-Super-120B-A12B (Elastic 30B active) | `nemotron_h.NemotronHForCausalLM` | **NoPE** (same code path as Cascade-2; class-level, not config-driven) | 6 attn / 23 Mamba-2 / 23 MoE per HF model card | N/A for attn (NoPE); SSM restored verbatim | Yes |
| Qwen3-Coder-Next | `qwen3_next.Qwen3NextForCausalLM` | **RoPE** — `rotary_emb = get_rope(...)` applied to q/k before attention (`models/qwen3_next.py:638-773`) | Hybrid (linear-attn + full-attn + Mamba blocks) | See findings §1 — KV restart from empty; positions for new tokens start at 0 in the pending-restore hydrate path | Yes |
| Codestral Mamba 7B | `mamba2.Mamba2ForCausalLM` | None — pure Mamba2, no attention layers | All SSM | N/A | Yes |

## Code-path notes

### Snapshot contents (size invariance)

`python/sglang/srt/snapshot/mamba_host_pool.py:43-44` defines the
serialized state as `conv_states: List[torch.Tensor]` and
`temporal_states: torch.Tensor`. No attention-KV fields appear in
`scheduler_snapshot_system.py` (`grep -nE "attn|kv_cache|kv_pool"`
returns no matches). SSM state shapes are determined by per-layer
config (`d_conv`, `d_state`, `d_inner`), not by sequence length, so
snapshot size is constant in context length for all validated models.

No model in the validated set uses sliding-window attention whose
window state would otherwise need to enter the snapshot path.

### Restore path 1 — pending-restore hydrate

`scheduler_snapshot_handlers.py:244-371`
(`_maybe_hydrate_from_pending_restore`). The new request's
`origin_input_ids` is left as the user's new turn only; the SSM pool
slot is loaded with the prior state via
`SnapshotManager.inject_state_to_pool`. The scheduler then prefills
only the new turn's tokens on top of the injected SSM state, and
`position_ids` for those new tokens are computed starting at 0.

- NoPE attention models (Granite-4.0-H, Nemotron-H/Cascade/Super): no
  positional dependence in attention, so the position-0 restart is a
  no-op for attention. SSM continuation is correct.
- Pure-SSM (Codestral): same — no attention layer, no positional
  concern. SSM continuation is correct.
- RoPE attention models (Qwen3-Coder-Next): RoPE is applied to q/k of
  the new tokens at positions `0..M-1`, and the attention KV cache is
  empty (snapshot did not capture it). The new tokens attend only to
  themselves, so RoPE-position-vs-original-position is internally
  consistent within the new step but the attention layer has lost all
  long-range KV context across the restore boundary. This is
  consistent with the documented snapshot design (attention KV is not
  preserved), not a positional bug.

### Restore path 2 — `restore_snapshot` RPC with `continuation_ids`

`scheduler_snapshot_handlers.py:700-817`. Builds a fresh `Req` with
`origin_input_ids = metadata.fill_ids + continuation_ids` (where
`metadata.fill_ids` is the full token history at snapshot time, set in
`scheduler_snapshot_system.py:304-312`) and assigns
`new_req.fill_ids = torch.tensor(origin_input_ids, ...)`.

The new request's `fill_ids` length equals `origin_input_ids` length,
so under the normal scheduler contract no tokens would be prefilled.
This interaction with the injected SSM state — and the resulting
`position_ids` seen by the attention layers in hybrid models — is not
exercised by static inspection alone and **requires live validation**
to confirm the encoded continuation matches the original (see
findings §2).

## Findings

1. **Hybrid attention long-range context loss across restore** —
   architectural property of the design, not a bug. For NoPE hybrids
   (Granite-4.0-H, Nemotron-H/Cascade/Super) the impact is bounded by
   what attention contributed in the original conversation; with NoPE
   attention there is no positional drift, only KV-content loss. For
   RoPE hybrids (Qwen3-Coder-Next) the attention layer is effectively
   restarted from the most-recent turn onward. Recommendation: add a
   one-paragraph note to `docs/stateful_mamba/api_guide.md` under a
   "Hybrid models" subsection making this explicit so reviewers do not
   assume attention-state preservation.

2. **Path 2 (`restore_snapshot` + `continuation_ids`) prefill
   semantics** — `requires live validation`. The handler constructs a
   request whose `fill_ids` equals its `origin_input_ids`, which on
   the surface would leave continuation tokens un-prefilled. Either
   (a) the scheduler treats this case specially when
   `mamba_pool_idx` is pre-assigned, or (b) the path relies on a
   convention not visible from static read. Run a continuation through
   path 2 on Granite-4.0-H and Qwen3-Coder-Next and confirm the model
   sees the expected token stream (compare next-token logprobs against
   a non-snapshot continuation of the same prompt). If divergent, file
   a follow-up issue.

3. **Nemotron-H attention is NoPE in the Engram implementation** —
   `NemotronHAttention.forward` (`models/nemotron_h.py:515-521`)
   passes q/k straight to `RadixAttention` without `rotary_emb`. This
   is the actual inference behavior regardless of whatever the HF
   config might claim. Worth a comment in
   `models/nemotron_h.py` near the class so future maintainers do not
   "fix" the absent rotary embedding. Recommendation: file as a small
   code-comment follow-up.

4. **Granite-4.0-H positional encoding is config-gated** —
   `models/granitemoehybrid.py:224` uses RoPE only when
   `config.position_embedding_type == "rope"`. Granite-4.0-H ships
   with NoPE, but a future Granite checkpoint flipping that config
   field would silently change the restore behavior on hybrid layers.
   Recommendation: capture `position_embedding_type` into snapshot
   metadata and reject restore on mismatch (defer as follow-up, not
   blocking).

5. **Snapshot-size invariance confirmed** for all five models. No
   attention-layer state, no sliding-window cache, no per-token field
   enters the snapshot serialization path.

## Issue candidates (flagged for overseer; not opened)

- Doc note: hybrid-attention KV is not preserved across restore
  (finding 1).
- Code comment in `models/nemotron_h.py` flagging deliberate NoPE
  attention (finding 3).
- Snapshot-metadata field for `position_embedding_type` with restore
  mismatch check (finding 4).
- Live validation of path-2 (`restore_snapshot` + `continuation_ids`)
  prefill semantics (finding 2).

## Out of scope (per brief)

- CAI-12 (conv state mismatch on Codestral) — known open.
- CAI-14 (piecewise CUDA graph on Codestral) — known open.
