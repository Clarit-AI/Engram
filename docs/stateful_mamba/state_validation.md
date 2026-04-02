# State Validation & Hidden State Poisoning Guards

## The Risk

Mamba SSM state is a compact fixed-size representation of all tokens processed in a conversation. Unlike attention KV caches, it cannot be reconstructed by reprocessing — once overwritten, the prior state is gone.

In stateless inference this property is harmless: each request starts from a zero state, so any corruption is automatically discarded at request end. In a persistent snapshot system like sglang-mamba, the situation is qualitatively different. A corrupted state written to disk will be restored on every future turn for that conversation, and on every server restart that pre-loads the WARM tier. Corruption is sticky.

Three failure modes produce corrupted hidden states:

- **Adversarial or malformed input.** Carefully constructed token sequences can push SSM recurrences toward NaN or Inf. This is sometimes called "hidden state poisoning" in SSM security literature.
- **Hardware faults.** Transient GPU ECC errors or memory bit flips produce silent NaN/Inf values in tensor computations.
- **Model divergence.** A model loaded from a different checkpoint, or state restored into a pool with a mismatched layer configuration, produces nonsensical but numerically valid-looking tensors.

No other Mamba serving framework has addressed this class of risk because no other framework persists SSM state across requests. The validation layer described here is specific to the persistent snapshot architecture.

Related research context: the "inability to forget" property of SSMs (fixed-size state that cannot selectively erase past context) makes poisoned state especially impactful — unlike transformers where a corrupted KV entry competes with many valid entries in attention, a corrupted SSM state contaminates the entire recurrence.

---

## What We Check

`validate_state_tensors()` runs the following checks on every `conv_state` tensor and on `temporal_states`:

| Check | Trigger | Severity | Notes |
|---|---|---|---|
| Empty conv_states list | `len(conv_states) == 0` | Error (hard fail) | Aborts immediately; remaining checks skipped |
| Empty tensor | `t.numel() == 0` | Error | Per-tensor; skips numeric checks for that tensor |
| Invalid dtype | dtype not in `ALLOWED_DTYPES` | Error | Skips numeric checks (cannot safely call isnan/isinf) |
| NaN | `torch.isnan(t).any()` | Error | FP8 tensors upcasted to float16 before check |
| Inf | `torch.isinf(t).any()` | Error | FP8 tensors upcasted to float16 before check |
| All-zeros | `t.abs().sum() == 0` | Warning (default) / Error (strict mode) | Uninitialized state passes silently in default mode |

**Allowed dtypes:** `float32`, `bfloat16`, `float16`, and `float8_e4m3fn` (when available, for H200/Nemotron compatibility). SSM state is typically bfloat16 even in quantized models; FP8 support is forward-looking.

**Strict mode** (`strict=True`) promotes the all-zeros warning to an error. This is not used in the hot path because a freshly initialized state is legitimately all-zeros before the first forward pass.

---

## Where We Check

Validation runs at four points in the data path, plus one additional gate during startup.

### `save_snapshot()` — before writing to disk

Validation runs before any I/O. If it fails, no files are written and a `ValueError` is raised. This is the primary guard preventing poisoned state from entering persistent storage.

```
extract from GPU → validate → [GATE] → atomic write to disk
                                ^
                          hard fail here
```

### `load_snapshot()` — after loading from disk

Validation runs after `load_file()` deserializes the safetensors archive. If it fails, a `ValueError` is raised and the snapshot is not returned to the caller. This catches corruption introduced at the storage layer (bit rot, partial writes that survived the atomic rename).

### `inject_state_to_pool()` — before writing to GPU

The last checkpoint before restored state enters live inference. If validation fails at this point, the pool slot is not modified and a `ValueError` is raised. Any in-flight request waiting for this slot will see an error rather than receiving poisoned hidden state.

### `extract_state_from_pool()` — after GPU-to-CPU clone

Runs immediately after `.clone().cpu()` on the pool tensors, before any serialization. Catches hardware-induced corruption that occurred during the preceding forward passes. If validation fails, the caller cannot proceed to `save_snapshot()`.

### Startup restore — quarantine gate

`restore_snapshots_on_startup()` in `tier_manager.py` calls `restore_conversation()` for each snapshot found on disk. If that call raises `ValueError` (which includes validation failures from `load_snapshot()` and `inject_state_to_pool()`), the snapshot is quarantined:

```python
except ValueError as e:
    active_logger.warning(
        "Quarantined snapshot for conversation %s: state validation "
        "failed during startup restore: %s",
        conv_id,
        e,
    )
```

The server continues startup normally. Remaining valid snapshots are loaded to the WARM tier. The quarantined snapshot files remain on disk and are not deleted automatically.

### Model compatibility check

`MambaSnapshotMetadata` stores `model_name` (e.g., `"ibm-granite/granite-4.0-h-small"`) at save time. Before injecting restored state, callers should verify `metadata.model_name` matches the running model. Injecting state from a different model into a mismatched pool produces incorrect inference without triggering numeric checks, because the values are numerically valid — they are simply wrong. Shape mismatches are caught by `inject_state_to_pool()` as a `ValueError`; semantic mismatches (same shape, different model) require the caller to enforce the model name check.

---

## Quarantine Behavior

A quarantined snapshot:

- Logs a `WARNING`-level message with the conversation ID and validation error
- Is not loaded into the WARM tier
- Stays on disk unchanged
- Does not block server startup or affect other conversations
- Is not retried on the next startup (same `ValueError` will be raised again)

To recover a quarantined snapshot, either delete it manually from the snapshot directory or fix the underlying data corruption and replace the file.

---

## Performance

All Tier 1 validation checks operate on CPU tensors. In `extract_state_from_pool()` the clone to CPU happens before validation, so there is no additional device transfer cost. In `save_snapshot()` and `load_snapshot()` the tensors are already on CPU.

Typical overhead for a single-conversation Mamba state (~56 MB in bfloat16):

| Operation | Approximate cost |
|---|---|
| `isnan` + `isinf` sweep | 10-25ms |
| `abs().sum()` (all-zeros check) | 5-20ms |
| Total per validation call | 15-45ms |

This overhead is incurred once per `save_snapshot()` and once per `load_snapshot()` call, not per token. There is no GPU pipeline impact.

---

## API Reference

### `validate_state_tensors()`

```python
def validate_state_tensors(
    conv_states: List[torch.Tensor],
    temporal_states: torch.Tensor,
    metadata: Optional[MambaSnapshotMetadata] = None,
    strict: bool = False,
) -> ValidationResult:
```

Validates SSM state tensors for corruption indicators. All checks run on CPU tensors. The `metadata` argument is used only for including the conversation ID in error messages.

Returns a `ValidationResult`. Does not raise — the caller decides whether to raise on `not result.is_valid`.

### `ValidationResult`

```python
@dataclass
class ValidationResult:
    is_valid: bool
    warnings: List[str]
    errors: List[str]

    def __bool__(self) -> bool: ...  # returns is_valid
```

`errors` are populated for conditions that indicate definite corruption (NaN, Inf, empty tensor, bad dtype). `warnings` are populated for conditions that may indicate a problem but are not necessarily corrupt (all-zeros in non-strict mode).

The four pipeline integration points (`save_snapshot`, `load_snapshot`, `inject_state_to_pool`, `extract_state_from_pool`) all treat `not result.is_valid` as a hard failure and raise `ValueError`.

---

## Future: Tier 2 Monitoring

The current checks (Tier 1) are binary pass/fail. A planned Tier 2 layer would add statistical monitoring:

- **Tensor norm tracking.** Compute L2 norm of `temporal_states` after each forward pass and maintain a rolling baseline per conversation.
- **Anomaly detection.** Flag states where the norm deviates more than 3 standard deviations from the baseline. This catches drift that does not produce NaN/Inf but is nonetheless pathological.
- **Configurable interval.** Controlled via `--snapshot-health-check-interval N` (check every N turns). Default: disabled.
- **Failure policy.** Two modes under consideration: `log_and_continue` (warn but proceed) vs `kill_session` (invalidate the conversation's WARM entry and force cold restore on next turn). This is a product decision — `kill_session` is safer but may surprise users with unexpected context resets.

Tier 2 monitoring has no implementation in the current codebase. The `ValidationResult.warnings` list is the intended channel for surfacing Tier 2 findings without blocking inference.
