# KHA-387 SSM Ablation Methodology

## Purpose

Distinguish whether the recurrent SSM state restored by `/restore_snapshot`
is actually being consumed by the next-/generate forward pass.

The KHA-360 chat-turn diagnostic established that hybrid models fail
behavioral recall after pending-restore + question-only chat prefill, even
though server logs show:

- `/save_snapshot` and `/restore_snapshot` return 200
- `Snapshot staged for pending restore`
- `Injected state to pool idx K`
- `pending-restore registry hit: hydrated`

Structural evidence is not consumption evidence. A snapshot can be loaded,
injected into the Mamba pool, and registered as hydrated while the next
forward pass never reads the slot it landed in. KHA-385's v2 probe already
observed that, for pure-Mamba2 Codestral, the byte-level continuation of
restore-only + next /generate is **identical** to the no-restore baseline.
This harness formalizes and extends that observation.

## Conditions

Five conditions are run on the same seed prompt and the same
continuation prompt. Each uses its own conversation_id to keep the WARM
tier in-memory caches isolated.

| Case          | /restore_snapshot? | Source of restored state            |
| ------------- | ------------------ | ----------------------------------- |
| `no_restore`  | skipped            | none                                |
| `real_warm`   | yes (no turn_num)  | WARM tier (in-memory, real state)   |
| `real_disk`   | yes, turn_number=1 | safetensors file on disk, untampered |
| `zero_disk`   | yes, turn_number=1 | safetensors file on disk, zeroed     |
| `random_disk` | yes, turn_number=1 | safetensors file on disk, Gaussian   |

The disk path is exercised by passing `turn_number` to
`/restore_snapshot`. In `scheduler._load_snapshot_for_pending_restore`,
the WARM tier is only consulted when `turn_number is None and
branch_name is None`; setting either forces
`snapshot_manager.load_snapshot()` which reads the safetensors file.

Tampering is done after `/save_snapshot` returns (which promotes the WARM
copy to disk) and before `/restore_snapshot`. Mutations target both
`conv_layer_*` and `temporal` keys.

`validate_state_tensors` (mamba_snapshot.py:101) treats all-zero as a
**warning** at non-strict load and inject, so case `zero_disk` loads
cleanly. The random sampler uses per-tensor mean+std with a 6-sigma
clamp to keep load-time NaN/Inf guards quiet.

## Decision rules

The continuation request emits up to 96 tokens at temperature=0. The
harness compares the first 32 output_ids across cases.

| Observation                                                 | Conclusion                                                                                                    |
| ----------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------- |
| All five prefixes identical                                  | Restored SSM is **not consumed** by the pending-restore + next-/generate path. Bug is in request lifecycle.   |
| real_warm ≡ no_restore but disk cases differ                 | WARM-tier injection is silently dropped; disk path partially works.                                           |
| real_warm differs from no_restore AND zero_disk ≠ random_disk | SSM **is** consumed; tampered tensors produce distinguishable trajectories. Recall failure is then about state sufficiency or surface form, not consumption. |
| zero_disk ≡ random_disk ≡ no_restore but real differs        | Validation strips tampered tensors before injection; rerun with logs to inspect.                              |

## Out of scope

- Whether the SSM state itself is *sufficient* for full hybrid recall —
  that is the attention-KV question. This harness only answers whether
  the SSM state reaches the forward pass at all.
- Hybrid models. Codestral pure-Mamba2 is the cleanest signal because
  there is no attention KV path to confound the observation. Hybrid
  ablation is a strict superset; if SSM is not consumed for pure-Mamba2,
  KV scope is moot until the consumption path is fixed.

## Inputs required

- H100 (or any single GPU with ≥ 16 GB free for a 7 B model)
- `mistralai/Mamba-Codestral-7B-v0.1` available locally
- The same env that KHA-385 used:
  - `TORCHDYNAMO_DISABLE=1`
  - `SGLANG_ENABLE_SPEC_V2=false`
  - `--disable-piecewise-cuda-graph`
  - `--mamba-scheduler-strategy no_buffer`

## How to run

```bash
scripts/kha-387-ssm-ablation.sh
```

Output: `report.md`, `results.json`, `log.txt`, `server.log` under
`/workspace/runs/kha-387-ssm-ablation-<timestamp>/`.
