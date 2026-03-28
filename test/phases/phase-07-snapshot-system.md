# Phase 7 — Snapshot System

## Purpose

Verify the `POST /save_snapshot` and `POST /restore_snapshot` HTTP endpoints end-to-end. The snapshot system persists per-request Mamba conv and temporal states to disk as `.safetensors` files, enabling branching conversation trees and state restoration. The key invariant is **state fidelity**: restoring a snapshot and re-running the same prompt at `temperature=0` must produce byte-identical output to the original run. The tier management system (`TierManager`, `MambaHostPool`) is also exercised directly.

## Prerequisites

- Phase 0 complete
- Phase 1 complete (baseline inference verified)
- Phase 4 complete (live server with radix cache working)
- Model checkpoint available at `$MODEL_PATH`
- Sufficient disk space for snapshot files (at least 1 GB free at `$SNAPSHOT_DIR`)
- No other server instances running on port 30000

## Key Files

- `python/sglang/srt/snapshot/mamba_snapshot.py` — `MambaSnapshotManager`, `POST /save_snapshot`, `POST /restore_snapshot`
- `python/sglang/srt/snapshot/tier_manager.py` — `TierManager` (GPU → host → disk tier promotion/demotion)
- `python/sglang/srt/snapshot/mamba_host_pool.py` — `MambaHostPool` (pinned host memory buffer)
- `python/sglang/srt/entrypoints/http_server.py` — HTTP server with snapshot endpoints
- **New**: `test/registered/radix_cache/test_mamba_snapshot_e2e.py`

## Environment Setup

```bash
cd /home/bbrenner/sglang-mamba

source test/phases/config.sh   # sets MODEL_PATH, SERVER_PORT, SERVER_URL, SNAPSHOT_DIR, RESULTS_DIR

# Launch server with snapshot persistence enabled
# Flags confirmed from python/sglang/srt/server_args.py lines 525-526, 4255-4266
python -m sglang.launch_server \
    --model-path $MODEL_PATH \
    --port $SERVER_PORT \
    --mamba-scheduler-strategy no_buffer \
    --enable-snapshot-persistence \
    --snapshot-dir $SNAPSHOT_DIR \
    > /tmp/phase7_server.log 2>&1 &

export SERVER_PID=$!

# Wait for server ready
python -c "
import time, requests
for i in range(60):
    try:
        r = requests.get('http://localhost:30000/health')
        if r.status_code == 200:
            print('Server ready (snapshot mode)')
            break
    except:
        pass
    time.sleep(2)
else:
    print('ERROR: Server did not start')
    exit(1)
"

# Verify snapshot endpoints are registered
curl -s http://localhost:$SERVER_PORT/save_snapshot -X POST -H 'Content-Type: application/json' \
    -d '{"rid": "probe"}' | python -m json.tool || echo "Endpoint check done"
```

> **Important**: Before launching, inspect `python/sglang/srt/server_args.py` to find the correct flag names for enabling snapshots. Update the launch command above accordingly.

## API Reference

### POST /restore_snapshot

Restores a previously saved Mamba snapshot. Supports two modes:

**In-place restore (default)**: The snapshot state is injected into the existing `rid`. Subsequent requests using the same `rid` continue from the restored state.

**New request restore (`create_new_request: true`)**: The snapshot is loaded into a fresh Mamba pool slot and a **new `rid`** is returned. The caller can use this new `rid` independently, enabling a stateless client pattern where the original `rid` can be discarded.

#### Request

```
POST /restore_snapshot
Content-Type: application/json
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `rid` | string | Yes | Request ID of the snapshot to restore |
| `create_new_request` | boolean | No | If `true`, allocates a new Mamba pool slot and returns a new `rid` backed by the restored state. Default: `false` |
| `conversation_id` | string | No | Alternative identifier for the snapshot (mutually exclusive with `rid`) |
| `turn_number` | integer | No | Specific turn to restore when using `conversation_id` |

#### Response (in-place restore)

```json
{
  "success": true,
  "rid": "original-rid-here",
  "mamba_pool_idx": 3,
  "message": "Snapshot restored successfully"
}
```

#### Response (create_new_request: true)

```json
{
  "success": true,
  "rid": "restored-a1b2c3d4",
  "mamba_pool_idx": 7,
  "message": "Created new request from snapshot"
}
```

#### Error Response

```json
{
  "success": false,
  "rid": null,
  "message": "Snapshot not found for rid: nonexistent-rid"
}
```

#### Usage Example (stateless client pattern)

```bash
# 1. Save a snapshot from an existing conversation
curl -X POST http://localhost:30000/save_snapshot \
  -H 'Content-Type: application/json' \
  -d '{"rid": "conv-123-turn-5"}'

# 2. Restore as a NEW request — get a fresh rid
curl -X POST http://localhost:30000/restore_snapshot \
  -H 'Content-Type: application/json' \
  -d '{"rid": "conv-123-turn-5", "create_new_request": true}'
# Returns: {"success": true, "rid": "restored-a1b2c3d4", "mamba_pool_idx": 7, ...}

# 3. Use the new rid for subsequent generation — original rid can be discarded
curl -X POST http://localhost:30000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model": "default", "messages": [...], "rid": "restored-a1b2c3d4"}'
```

## Tasks

### Task 1: Write the snapshot E2E test file

Create `test/registered/radix_cache/test_mamba_snapshot_e2e.py` with the content in **New Test File(s) to Write** below.

### Task 2: Run the automated tests

```bash
SERVER_URL=http://localhost:$SERVER_PORT \
SNAPSHOT_DIR=$SNAPSHOT_DIR \
python -m pytest test/registered/radix_cache/test_mamba_snapshot_e2e.py -v \
    --timeout=120 2>&1 | tee /tmp/phase7_tests.log
```

### Task 3: HITL — snapshot branch and restore (API-direct)

> **Why not the web UI**: The chatbot at http://localhost:3000 loads the full Postgres message history and re-sends it with every request (`getMessagesByChatId` + `convertToUIMessages`, `route.ts` lines 110–156). After a snapshot restore, if you continue in the same chat thread the app sends the DB history including turn 3 — the model would appear to "forget" turn 3 only if the history is stripped, which the app never does. The web UI **cannot** isolate server-side snapshot state from client-side history. Use the API directly.

Run this script which controls exactly what messages are sent at each step:

```bash
python3 - << 'PYEOF'
import requests, json, uuid, time

BASE = "http://localhost:30000"
rid = f"hitl-snap-{uuid.uuid4().hex[:8]}"
print(f"Using rid: {rid}\n")

def chat(messages, label, rid_to_use=None):
    r = requests.post(f"{BASE}/v1/chat/completions",
        json={"model":"default","messages":messages,"rid":rid_to_use or rid,
              "temperature":0,"max_tokens":60}, timeout=60)
    content = r.json()["choices"][0]["message"]["content"]
    print(f"[{label}]\nUser: {messages[-1]['content']}\nAssistant: {content}\n")
    return content

def save(req_rid=None):
    r = requests.post(f"{BASE}/save_snapshot", json={"rid": req_rid or rid}, timeout=30)
    result = r.json()
    print(f"[SAVE] {result}\n")
    return result.get("success", False)

def restore(req_rid=None):
    r = requests.post(f"{BASE}/restore_snapshot", json={"rid": req_rid or rid}, timeout=30)
    result = r.json()
    print(f"[RESTORE] {result}\n")
    return result

def restore_new_request(req_rid=None):
    """Restore with create_new_request=True — returns a NEW rid backed by restored state."""
    r = requests.post(f"{BASE}/restore_snapshot",
        json={"rid": req_rid or rid, "create_new_request": True}, timeout=30)
    result = r.json()
    print(f"[RESTORE_NEW_REQUEST] {result}\n")
    return result

# Turn 1 — establish context (send only this message, no history)
t1_q = [{"role":"user","content":"My secret animal is a narwhal."}]
t1_a = chat(t1_q, "Turn 1")

# Turn 2 — confirm recall (send turns 1+2 only)
t2_q = t1_q + [{"role":"assistant","content":t1_a},
               {"role":"user","content":"What is my secret animal?"}]
t2_a = chat(t2_q, "Turn 2")

# Save snapshot here (post-turn-2 state)
assert save(), "Snapshot save failed — stop"

# Turn 3 — diverge (send turns 1+2+3)
t3_q = t2_q + [{"role":"assistant","content":t2_a},
               {"role":"user","content":"Actually, forget the narwhal. My new secret animal is a capybara."}]
t3_a = chat(t3_q, "Turn 3 (diverge)")

# Restore snapshot to post-turn-2 state (in-place restore)
restore_result = restore()
assert restore_result.get("success", False), "Snapshot restore failed — stop"

# Turn 3 re-run — send ONLY turns 1+2 (no turn 3 in history)
# If restore worked, model should NOT know about the capybara
t3_rerun_q = t2_q + [{"role":"assistant","content":t2_a},
                     {"role":"user","content":"What is my secret animal?"}]
t3_rerun_a = chat(t3_rerun_q, "Turn 3 after restore (no capybara in history)")

# Evaluate
narwhal_recalled = "narwhal" in t3_rerun_a.lower()
capybara_leaked  = "capybara" in t3_rerun_a.lower()
print(f"Narwhal recalled: {narwhal_recalled}")
print(f"Capybara leaked (should be False): {capybara_leaked}")
print(f"\nHITL RESULT (in-place restore): {'PASS' if narwhal_recalled and not capybara_leaked else 'FAIL'}")

# =============================================================================
# Part 2: Test create_new_request=True (stateless client pattern)
# =============================================================================
print("\n" + "="*60)
print("PART 2: create_new_request=True (stateless pattern)")
print("="*60 + "\n")

# Save a snapshot from the current (restored, post-turn-2) state
save_result = save()
assert save_result, "Snapshot save failed — stop"

# Restore with create_new_request=True — returns a NEW rid
new_req_result = restore_new_request()
assert new_req_result.get("success", False), f"create_new_request restore failed: {new_req_result}"

new_rid = new_req_result.get("rid")
mamba_pool_idx = new_req_result.get("mamba_pool_idx")
print(f"Got new rid: {new_rid} (pool idx: {mamba_pool_idx})")
assert new_rid is not None, "Expected rid in response"
assert new_rid != rid, "New rid must differ from original"

# Use the new rid to continue conversation — the restored state is already injected
# Send only turn 1 + turn 2 (same as before), but now using the NEW rid
new_rid_q = [{"role":"user","content":"My secret animal is a narwhal."},
             {"role":"assistant","content":t1_a},
             {"role":"user","content":"What is my secret animal?"}]
new_rid_a = chat(new_rid_q, f"Turn 2 via new_rid={new_rid}")

narwhal_via_new_rid = "narwhal" in new_rid_a.lower()
capybara_via_new_rid = "capybara" in new_rid_a.lower()
print(f"Narwhal recalled via new rid: {narwhal_via_new_rid}")
print(f"Capybara leaked via new rid (should be False): {capybara_via_new_rid}")

print(f"\nHITL RESULT (create_new_request): {'PASS' if narwhal_via_new_rid and not capybara_via_new_rid else 'FAIL'}")
PYEOF
# Tee output to log
2>&1 | tee /tmp/phase7_hitl_log.txt
```

> **What this proves**: By sending only turns 1+2 in the history after restore, if the model answers "narwhal" it could be from either client history or server state. The stronger proof is that "capybara" does **not** appear — because the only way it could appear is if the server-side Mamba state from turn 3 leaked through despite the restore. The automated test `test_restore_snapshot_state_equivalence` provides the byte-identical output proof; this HITL validates the qualitative end-to-end flow.
>
> **Part 2 (create_new_request) proves** that the stateless client pattern works: the original `rid` can be discarded entirely after restore, and the returned `new_rid` can be used independently to continue from the restored state. This is essential for multi-client or stateless API gateway scenarios.

### Task 4: Verify snapshot files on disk

```bash
ls -lh $SNAPSHOT_DIR/
# Expect: .safetensors file(s) and .json metadata file(s)

# Inspect keys inside a safetensors file
python -c "
from safetensors import safe_open
import os, glob
files = glob.glob('$SNAPSHOT_DIR/*.safetensors')
if files:
    with safe_open(files[0], framework='pt') as f:
        print('Keys:', list(f.keys()))
else:
    print('No safetensors files found')
"
```

### Task 5: Check server logs for snapshot activity

```bash
grep -i "snapshot\|save\|restore\|tier\|host_pool\|safetensors" /tmp/phase7_server.log | tail -30
```

### Task 6: Shut down server

```bash
kill $SERVER_PID
wait $SERVER_PID 2>/dev/null
echo "Server stopped"
```

## New Test File(s) to Write

**Path**: `test/registered/radix_cache/test_mamba_snapshot_e2e.py`

```python
from sglang.test.ci.ci_register import register_cuda_ci, register_amd_ci

register_cuda_ci(est_time=150, suite="stage-b-test-small-1-gpu")
register_amd_ci(est_time=150, suite="stage-b-test-small-1-gpu-amd")

import glob
import json
import os
import time
import unittest

import requests

SERVER_URL = os.environ.get("SERVER_URL", "http://localhost:30000")
SNAPSHOT_DIR = os.environ.get("SNAPSHOT_DIR", "/tmp/mamba_snapshots")


class TestMambaSnapshotE2E(unittest.TestCase):

    def setUp(self):
        try:
            r = requests.get(f"{SERVER_URL}/health", timeout=5)
            if r.status_code != 200:
                self.skipTest("Server not available")
        except Exception:
            self.skipTest("Server not available")

    def _chat(self, messages, rid=None, **kwargs):
        kwargs.setdefault("temperature", 0)
        kwargs.setdefault("max_tokens", 80)
        payload = {"model": "default", "messages": messages, **kwargs}
        if rid:
            payload["rid"] = rid
        r = requests.post(f"{SERVER_URL}/v1/chat/completions", json=payload, timeout=60)
        r.raise_for_status()
        return r.json()

    def _save_snapshot(self, rid):
        r = requests.post(f"{SERVER_URL}/save_snapshot",
                          json={"rid": rid}, timeout=30)
        r.raise_for_status()
        return r.json()

    def _restore_snapshot(self, rid):
        r = requests.post(f"{SERVER_URL}/restore_snapshot",
                          json={"rid": rid}, timeout=30)
        r.raise_for_status()
        return r.json()

    def _restore_snapshot_new_request(self, rid):
        """Restore with create_new_request=True — returns a new rid."""
        r = requests.post(f"{SERVER_URL}/restore_snapshot",
                          json={"rid": rid, "create_new_request": True}, timeout=30)
        r.raise_for_status()
        return r.json()

    def test_save_snapshot_returns_success(self):
        """After generating tokens for a request, POST /save_snapshot returns success=True."""
        import uuid
        rid = f"test-save-{uuid.uuid4().hex[:8]}"
        self._chat([{"role": "user", "content": "Hello, what is 1+1?"}], rid=rid)
        time.sleep(0.5)  # allow request to settle

        result = self._save_snapshot(rid)
        self.assertTrue(
            result.get("success", False),
            f"save_snapshot returned: {result}"
        )

    def test_restore_snapshot_state_equivalence(self):
        """Save after turn N, restore, re-generate turn N+1 — output must match original at temperature=0."""
        import uuid
        rid = f"test-restore-{uuid.uuid4().hex[:8]}"
        messages = []

        # Turn 1
        messages.append({"role": "user", "content": "My secret word is: ALPHA42."})
        resp1 = self._chat(messages, rid=rid)
        messages.append({"role": "assistant", "content": resp1["choices"][0]["message"]["content"]})

        # Save snapshot after turn 1
        save_result = self._save_snapshot(rid)
        self.assertTrue(save_result.get("success", False), f"Snapshot save failed: {save_result}")

        # Turn 2 — generate original output
        turn2_prompt = "What is my secret word?"
        messages_with_t2 = messages + [{"role": "user", "content": turn2_prompt}]
        resp2_original = self._chat(messages_with_t2, rid=rid)
        original_output = resp2_original["choices"][0]["message"]["content"]

        # Restore snapshot (back to post-turn-1 state)
        restore_result = self._restore_snapshot(rid)
        self.assertTrue(restore_result.get("success", False), f"Snapshot restore failed: {restore_result}")

        # Re-generate turn 2 from restored state
        resp2_restored = self._chat(messages_with_t2, rid=rid)
        restored_output = resp2_restored["choices"][0]["message"]["content"]

        self.assertEqual(
            original_output.strip(), restored_output.strip(),
            f"Output mismatch after restore.\nOriginal: {original_output}\nRestored: {restored_output}"
        )

    def test_restore_requires_idle_request(self):
        """Restoring a snapshot for an active/unknown request returns success=False gracefully."""
        result = self._restore_snapshot("nonexistent-rid-xyz-999")
        # Should not raise — must return a response indicating failure
        self.assertFalse(
            result.get("success", True),
            f"Expected success=False for nonexistent rid, got: {result}"
        )

    def test_snapshot_disk_format(self):
        """After save, .safetensors and .json files exist on disk with expected keys."""
        import uuid
        rid = f"test-disk-{uuid.uuid4().hex[:8]}"
        self._chat([{"role": "user", "content": "Save this state."}], rid=rid)
        time.sleep(0.5)
        save_result = self._save_snapshot(rid)
        self.assertTrue(save_result.get("success", False))
        time.sleep(1.0)  # allow disk flush

        # Find safetensors files
        safetensors_files = glob.glob(os.path.join(SNAPSHOT_DIR, "**/*.safetensors"), recursive=True)
        self.assertGreater(len(safetensors_files), 0, f"No .safetensors files found in {SNAPSHOT_DIR}")

        # Verify keys
        try:
            from safetensors import safe_open
            with safe_open(safetensors_files[0], framework="pt") as f:
                keys = list(f.keys())
            key_str = str(keys).lower()
            self.assertTrue(
                any(k in key_str for k in ("conv", "ssm", "temporal", "state")),
                f"Expected conv/ssm/temporal/state keys, got: {keys}"
            )
        except ImportError:
            self.skipTest("safetensors not installed")

    def test_snapshot_manager_tier_consistency(self):
        """TierManager transitions between GPU, host (MambaHostPool), and disk without data corruption."""
        # Direct unit test of TierManager — no server needed
        # Import and exercise tier promotion/demotion with mock tensors
        try:
            from sglang.srt.snapshot.tier_manager import TierManager
        except ImportError:
            self.skipTest("TierManager not importable — check snapshot module structure")

        # Construct TierManager with minimal config
        # Exercise: store tensor on GPU → promote to host → demote to disk → reload → verify equal
        # Implementation depends on TierManager API — inspect tier_manager.py before writing
        self.skipTest("Implement after inspecting TierManager API in python/sglang/srt/snapshot/tier_manager.py")

    def test_create_new_request_returns_new_rid(self):
        """create_new_request=True returns a new rid different from the original."""
        import uuid
        rid = f"test-new-rid-{uuid.uuid4().hex[:8]}"
        self._chat([{"role": "user", "content": "Hello, what is 1+1?"}], rid=rid)
        time.sleep(0.5)  # allow request to settle

        save_result = self._save_snapshot(rid)
        self.assertTrue(save_result.get("success", False), f"Snapshot save failed: {save_result}")

        restore_result = self._restore_snapshot_new_request(rid)
        self.assertTrue(restore_result.get("success", False), f"create_new_request restore failed: {restore_result}")

        new_rid = restore_result.get("rid")
        self.assertIsNotNone(new_rid, "Expected rid in response")
        self.assertNotEqual(new_rid, rid, "New rid must differ from original")


if __name__ == "__main__":
    unittest.main()
```

> **Note to implementing agent**: `test_snapshot_manager_tier_consistency` is stubbed — inspect `python/sglang/srt/snapshot/tier_manager.py` and `mamba_host_pool.py` to understand the API, then implement the tier transition test. Remove the `skipTest` once implemented.

## Pass Criteria

- Server starts with snapshot endpoints registered
- `test_save_snapshot_returns_success` passes
- `test_restore_snapshot_state_equivalence` passes — restored output is **identical** to original
- `test_restore_requires_idle_request` returns `success=False` without crashing
- `test_snapshot_disk_format` finds `.safetensors` files with conv/ssm/temporal keys
- `test_snapshot_manager_tier_consistency` either passes or is documented with a clear stub
- `test_create_new_request_returns_new_rid` passes — `create_new_request=True` returns a new rid different from the original
- HITL (in-place restore): restored snapshot reflects only turns 1–2, not turn 3 context
- HITL (create_new_request): new rid can be used independently with restored state; original rid can be discarded

## Write Report

```bash
REPORT="$RESULTS_DIR/phase-07-${MODEL_NAME}-$(date +%Y%m%d-%H%M).md"
cat > "$REPORT" << 'EOF'
# Phase 7 — Snapshot System
**Model**: granite-4.0-h-tiny
**Date**: <date>
**Result**: PASS | FAIL

## Server
- Snapshot endpoints registered: YES | NO
- CUDA errors: NONE | <summary>

## Test Results
| Test | Result |
|------|--------|
| test_save_snapshot_returns_success | PASS/FAIL |
| test_restore_snapshot_state_equivalence | PASS/FAIL |
| test_restore_requires_idle_request | PASS/FAIL |
| test_snapshot_disk_format | PASS/FAIL |
| test_snapshot_manager_tier_consistency | PASS/FAIL/SKIP |
| test_create_new_request_returns_new_rid | PASS/FAIL |

## State Equivalence
Original turn N+1 output: <output>
Restored turn N+1 output: <output>
Identical: YES | NO

## Disk Format
Safetensors keys found: <list>
Files at: <path>

## HITL (snapshot branch test — API-direct, no client history re-injection)
**Result**: PASS | FAIL

rid used: <rid>
Turn 1: My secret animal is a narwhal. → <response>
Turn 2: What is my secret animal? → <response>  [snapshot saved here]
Turn 3: Actually forget narwhal, it's capybara. → <response>  [diverge]
[snapshot restored]
Turn 3 re-run (turns 1+2 history only): What is my secret animal? → <response>
Narwhal recalled: YES | NO
Capybara leaked (must be NO): YES | NO

## HITL (create_new_request — stateless pattern)
**Result**: PASS | FAIL

Original rid: <rid>
New rid returned: <new_rid>
New rid used to query: What is my secret animal? → <response>
Narwhal recalled via new rid: YES | NO
Capybara leaked via new rid (must be NO): YES | NO

## Failures & Tracebacks
<paste here or "None">

## TierManager Notes
<API observations, what was implemented vs skipped>
EOF
echo "Report written to $REPORT"
```

## Reporting

```
PHASE 7 RESULT: PASS | FAIL
Tests run: 6  Passed: X  Failed: Y  Skipped: Z
State equivalence verified: YES | NO
Disk format verified: YES | NO (keys: <list>)
HITL: PASS (context isolation confirmed) | FAIL
Report: $RESULTS_DIR/phase-07-<model>-<date>.md
```