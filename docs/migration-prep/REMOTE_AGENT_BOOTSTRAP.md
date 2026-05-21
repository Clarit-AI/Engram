# Remote Agent Bootstrap Context
**Date**: March 29, 2026
**Current Branch**: `phase-08-true-stateful-inference`

## What Happened
The previous testing VM crashed due to lack of funds, and the remote `.claude` memory directory was permanently lost. A local agent (Antigravity) reconstructed the exact Phase 8 logic from terminal transcripts, extracted the Python code patches, and pushed them to the `phase-08-true-stateful-inference` branch. You are the new agent picking up the torch!

## The State of the Codebase
We just finished implementing **True Stateful Inference** (Phase 8):
1. **`io_struct.py`**: Added `continuation_ids` and `max_new_tokens` inputs, plus `output_ids` and `output_text` to the response.
2. **`scheduler.py`**: Intercepts stateful generations natively by appending new turn tokens to the snapshot's base context and flags them with `_stateful_generate`.
3. **`scheduler_output_processor_mixin.py`**: Intercepts `_stateful_generate` outputs natively so they don't get orphaned by the `create_new_request=True` flow. Output is routed right into the snapshot return queue!
4. **`test_mamba_stateful_inference.py`**: Completely rewritten multi-turn test, forcing native stateful inference without radix cache, correctly utilizing `rid1` as the root snapshot conversation state.

## Your Immediate Next Steps

1. **Verify Git State**: Ensure you are checked out on `origin/phase-08-true-stateful-inference`.
2. **Start the DEV Server**:
   ```bash
   python -m sglang.launch_server \
     --model-path /mnt/models/granite-4.0-h-tiny \
     --port 30000 \
     --disable-radix-cache \
     --enable-snapshot-persistence \
     --snapshot-dir /tmp/mamba_snapshots \
     --mamba-scheduler-strategy no_buffer
   ```
   *(Note: Adjust model path to match reality on this new VM).*
3. **Run the Reconstructed Test Suite**:
   ```bash
   pytest test/registered/radix_cache/test_mamba_stateful_inference.py -v -s
   ```
4. **Validation**: Ensure all 4 stateful tests pass, especially the deeply nested `test_multi_turn_stateful_chain` using native Snapshot ID continuation.
5. **Next Phase**: Once Phase 8 succeeds, move on to `test/phases/phase-09-gauntlet-stress-tests.md` to stress test the memory pool transitions.

*You have all the context now. You may proceed autonomously.*
