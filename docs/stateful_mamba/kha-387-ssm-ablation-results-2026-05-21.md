# KHA-387 SSM Ablation Results — 2026-05-21

## Verdict

**Restored SSM state is not being consumed by the pending-restore + next-/generate path.**

Five SSM conditions on pure-Mamba2 Codestral produced **byte-identical**
continuation prefixes for the first 24 emitted token-ids. Tampering the
on-disk safetensors to all-zero or to Gaussian noise — confirmed by
server-side validation warnings — did not change the model output. This
makes the bug structurally observable: the recurrent state lands in the
mamba pool slot the registry chose, but the request's forward pass
reads from a different slot or starts cold.

The result refutes the broader claim that hybrid recall failure on
KHA-360 is an attention-KV scope problem. Pure-Mamba2 Codestral has no
attention KV path; if it still cannot recall via this path, the
explanation is not attention-KV.

## Environment

| Item | Value |
| --- | --- |
| Date | 2026-05-21 |
| Host | H100 80GB HBM3 (GCP a3-highgpu-1g) |
| Branch | `feat/kha-387-ssm-ablation` |
| Base SHA | `6b95159aa` (`origin/main`, post-PR-#76) |
| Model | `mistralai/Mamba-Codestral-7B-v0.1` (local mirror) |
| Server flags | `--enable-snapshot-persistence --snapshot-trigger-policy every_turn --mamba-scheduler-strategy no_buffer --disable-radix-cache --disable-cuda-graph --disable-piecewise-cuda-graph` |
| Env | `TORCHDYNAMO_DISABLE=1 SGLANG_ENABLE_SPEC_V2=false` |

## Conditions and outcomes

Seed prompt establishes a `stored_pair()` definition that names a
sentinel `ZEPHYR_KHA387:271828`. Continuation prompt is just `    return ` (4 tokens).
At temperature 0, the model is deterministic; any difference in
behavior across cases reflects a difference in state going into the
forward pass.

| Case | Tamper | Restore call | Server log evidence | First 24 token ids of continuation |
| --- | --- | --- | --- | --- |
| no_restore | none | none | none | `[29502, 29513, 781, 29520, 781, 781, 781, 18269, 29574, 29473, 29508, 29502, 29502, 29502, 29501, 29508, 29542, 29542, 29542, 29516, 29508, 29502, 29502, 29508]` |
| real_warm | none | rid only | `Snapshot staged ... Injected state to pool idx 7 ... pending-restore registry hit ... mamba_pool_idx=7` | same 24 ids |
| real_disk | none | rid + turn_number=1 | `Snapshot loaded ... Injected state to pool idx 9 ... pending-restore registry hit ... mamba_pool_idx=9` | same 24 ids |
| zero_disk | conv+temporal → 0 | rid + turn_number=1 | `validation warning (load): all-zero tensor` × 2 ... `validation warning (inject): all-zero tensor` × 2 ... `Injected state to pool idx 11` ... `pending-restore registry hit ... mamba_pool_idx=11` | same 24 ids |
| random_disk | conv+temporal → Gaussian | rid + turn_number=1 | `Snapshot loaded ... Injected state to pool idx 13 ... pending-restore registry hit ... mamba_pool_idx=13` | same 24 ids |

Tamper landed on disk:

- `zero_disk`: pre `conv_abs_sum=6.781e+05, temporal_abs_sum=3.733e+05` → post `0, 0`
- `random_disk`: pre `conv_abs_sum=6.781e+05, temporal_abs_sum=3.733e+05` → post `7.715e+05, 1.500e+07`

The `validation warning (inject): all-zero tensor` line is dispositive:
it fires inside `inject_state_to_pool` after the conv and temporal
tensors are moved to CUDA, immediately before the
`mamba_pool.mamba_cache.{conv,temporal}[:, mamba_pool_idx] = ...`
writes. The zero tensors reached the GPU pool slot. The model still
produced the same output.

## What this rules out

- **Attention-KV scope as the root cause of KHA-360 hybrid failure.**
  Codestral pure-Mamba2 has no attention KV path. If it cannot recall
  via this restore path, attention-KV is not the explanation. A separate
  hybrid-specific KV question may exist, but it is not what KHA-360 was
  diagnosing.
- **Snapshot save/load/inject mechanics.** All three steps fire correctly,
  with the expected validation warnings on tampered input. The bug is
  past `inject_state_to_pool`.

## What this points at

The forward pass for the request that hydrated from the pending-restore
registry is not reading the slot that hydration wrote to. Candidate
sites to investigate, in roughly priority order:

1. **req-vs-mamba pool slot allocation in the prefill path.** The
   registry assigns `req.mamba_pool_idx` after `mamba_pool.alloc(1)`
   (scheduler.py:1635–1671). The prefill scheduler then independently
   chooses how to feed mamba state for this request — verify the
   prefill path reads `req.mamba_pool_idx` rather than allocating its
   own slot.
2. **MambaPool state-reset between alloc and forward.** If the runtime
   re-initializes the slot (zero-out) after alloc but before forward,
   the injected state would be overwritten by zeros and produce
   indistinguishable behavior from no-restore. The all-five-identical
   pattern matches this hypothesis exactly.
3. **mamba_scheduler_strategy=no_buffer interaction.** This run used
   `no_buffer`, the same strategy as KHA-385 and KHA-386. Try
   `extra_buffer` to see if the slot mapping differs.
4. **continuation_ids vs free-text contract.** The historical-passing
   path uses `create_new_request=true` with `continuation_ids`
   (KHA-385 historical PASS). The failing path is `/restore_snapshot`
   then a plain `/generate` with new text. The two paths likely diverge
   in how the request acquires its mamba slot.

## Implications for downstream tickets

- **KHA-360 revised framing.** The May 20 verdict comment on KHA-360
  attributed the hybrid recall failure to an attention-KV gap. That
  attribution does not survive this run. Replace with: "the pending-
  restore + next-/generate path does not consume restored SSM state;
  the hybrid surface form is incidental, the bug is not attention-KV."
- **KHA-389 decision gate.** Direction shifts. KV-cache implementation
  should remain research-only (KHA-253) until the consumption-path bug
  is fixed. Filing a follow-up implementation ticket for "wire
  pending-restore'd mamba slot into the next prefill" is now the next
  step, not KV scope.
- **KHA-255 / CAI-8.** The previous "blocked on attention-KV impl"
  framing was over-scoped. Unblock when the consumption-path fix lands;
  the SSM snapshot system itself is healthy under both historical
  semantics (KHA-385 v2 + KHA-386 historical) and structural
  injection (this run).
- **KHA-370.** The 2 ms restore figure remains a credibility risk
  independent of this; this run does not move that needle.

## Reproduction

```bash
git checkout feat/kha-387-ssm-ablation
scripts/kha-387-ssm-ablation.sh
```

Artifacts: report.md, results.json, log.txt, server.log under
`/workspace/runs/kha-387-ssm-ablation-<timestamp>/`; archived copies
under `docs/stateful_mamba/kha-387-runs-2026-05-21/`.

## Bug not surfaced here

The `scheduler.py:1315` `NoneType` traceback in
`post_forward_snapshot_callback` did **not** reproduce on this run,
unlike on KHA-360. Possibly the simpler /generate continuation
exercises a different lifecycle. Worth checking on the next hybrid run
because it remains on the cleanup queue.
