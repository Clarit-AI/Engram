# Phase 8 — True Stateful Inference

## Purpose

Prove that a multi-turn conversation can be held without the client resending full
conversation history on every turn.  The server reconstructs prior context from a
saved Mamba SSM snapshot; the client sends only new tokens per turn.

This is the core novel capability of the SGLang-Mamba system: snapshot persistence
enables multi-turn inference where token transmission per turn is O(new_tokens),
not O(total_conversation).

## Background

Phases 0–7 validated the snapshot infrastructure: save/restore mechanics, tier
management, cache routing, and metadata integrity. However, every test so far
uses `/v1/chat/completions` which sends the **full conversation history** on every
turn (standard stateless pattern). No test has proven that the server can generate
a correct response from restored Mamba state + new tokens alone.

### Architectural Gap Found

With `--disable-radix-cache` (required for current snapshot testing):
- `has_initial_states = context_lens_tensor > 0`
- `context_lens_tensor = extend_prefix_lens = [len(r.prefix_indices)]`
- `prefix_indices = []` always (no radix cache)
- Therefore `has_initial_states = False` always

This means restored SSM state is **overwritten** by re-prefill. Phase 7's
equivalence test passed because of determinism (same tokens → same output),
not because Mamba state was being used.

### Implementation Required

Two issues required new code (~80 lines, 5 files):

1. **No output channel for `create_new_request=True`**: The HTTP connection gets
   the new rid but generated output is orphaned. Fix: add `continuation_ids` +
   `max_new_tokens` to `RestoreSnapshotReqInput`, block until generation completes,
   return output via `RestoreSnapshotReqOutput`.

2. **Output routing**: Stateful-generate Reqs are intercepted in
   `stream_output_generation` and routed through the snapshot result channel
   instead of normal `BatchTokenIDOut`.

## Server Configuration

```bash
python -m sglang.launch_server \
  --model-path $MODEL_PATH \
  --port 30000 \
  --disable-radix-cache \
  --enable-snapshot-persistence \
  --snapshot-dir /tmp/mamba_snapshots \
  --no-buffer
```

## Test File

`test/registered/radix_cache/test_mamba_stateful_inference.py`

## Tests

### 1. `test_stateful_recall_semantic`
- Turn 1: Establish "secret number is 42" via full `/v1/chat/completions`
- Save snapshot
- Turn 2: Restore snapshot + send only "What is the secret number?" as `continuation_ids`
- **Assert**: Response contains "42"

### 2. `test_stateful_vs_full_resend_equivalence`
- Turn 1: Establish "favorite color is blue"
- Save snapshot
- Baseline: Full resend via `/v1/chat/completions` asking "What is my favorite color?"
- Test: Stateful generate with same question
- **Assert**: Both responses contain "blue"

### 3. `test_multi_turn_stateful_chain`
- Turn 1: Establish color (full chat), save snapshot with `rid1`
- Turn 2: Establish number (stateful, new tokens only), save snapshot with `rid1`
- Turn 3: Ask about both facts (stateful, new tokens only)
- **Assert**: Response contains both "green" and "7"
- **Key**: `rid1` is the conversation handle across all turns

### 4. `test_token_savings_quantification`
- Quantify token savings: stateful should send O(new_tokens) vs O(full_history)
- **Assert**: `full_resend_tokens > stateful_tokens`
- Print savings percentage

## Pass Criteria

- All 4 tests pass
- `test_stateful_recall_semantic` proves the model recalls facts from prior turns
  using only snapshot state (no history resent)
- `test_multi_turn_stateful_chain` proves chained stateful turns work beyond 2 turns
- Token savings > 50% on typical conversations

## Known Issues

- `/v1/tokenize` endpoint has a pre-existing serialization bug; tests load
  `AutoTokenizer` directly via `MODEL_PATH` env var as a workaround
- Chain test requires using `rid1` (original conversation rid) as the handle
  for saves and restores across all turns, not the `restored-*` rids returned
  by stateful generate

## Files Modified

| File | Change |
|------|--------|
| `python/sglang/srt/managers/io_struct.py` | Add continuation_ids, max_new_tokens, output_ids, output_text |
| `python/sglang/srt/managers/scheduler.py` | Append continuation_ids, set `_stateful_generate`, defer output |
| `python/sglang/srt/managers/scheduler_output_processor_mixin.py` | Route stateful output through snapshot channel |
| `python/sglang/srt/managers/tokenizer_manager.py` | Detokenize output_ids |
| `python/sglang/srt/entrypoints/http_server.py` | Expose output_ids, output_text in response |

## Branch

`phase-08-true-stateful-inference` from `main` at `6a17c47fa`

## Depends On

- Phase 7 complete (snapshot E2E passing)
- Server flags: `--disable-radix-cache --enable-snapshot-persistence --no-buffer`
