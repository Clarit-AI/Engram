# Phase 4 — Live Server Integration: MambaRadixCache + no_buffer Strategy

## Purpose

Verify that `MambaRadixCache` is correctly populated and queried during actual server inference, using the default `mamba_scheduler_strategy=no_buffer`. This is the first phase where the radix cache is live during real model inference. Key invariants under test: cache hits reduce prefill cost on repeated prefixes, concurrent requests sharing a prefix do not contaminate each other's Mamba states (the `mamba_lock_ref` protection invariant), multi-turn conversations maintain state continuity, and the server handles eviction pressure gracefully.

## Prerequisites

- Phase 0 complete
- Phase 1 complete (baseline inference verified without radix cache)
- Phase 3 complete (radix cache component behavior verified in isolation)
- Model checkpoint available at `$MODEL_PATH`
- No other server instances running on port 30000

## Key Files

- `python/sglang/srt/mem_cache/mamba_radix_cache.py` — `MambaRadixCache` (now active)
- `python/sglang/srt/server_args.py` — `mamba_scheduler_strategy=no_buffer` (default)
- `python/sglang/srt/entrypoints/http_server.py` — HTTP server
- **New**: `test/registered/radix_cache/test_mamba_radix_cache_server_integration.py`

## Environment Setup

```bash
REPO_ROOT=$(git rev-parse --show-toplevel)
cd "$REPO_ROOT"

source test/phases/config.sh   # sets MODEL_PATH, SERVER_PORT, SERVER_URL, RESULTS_DIR

# Launch server WITH radix cache (do NOT pass --disable-radix-cache)
# --enable-cache-report exposes cached_tokens in each response's usage object
python -m sglang.launch_server \
    --model-path $MODEL_PATH \
    --port $SERVER_PORT \
    --mamba-scheduler-strategy no_buffer \
    --enable-cache-report \
    > /tmp/phase4_server.log 2>&1 &

export SERVER_PID=$!

# Wait for server ready
python -c "
import time, requests, os
port = os.environ.get('SERVER_PORT', '30000')
for i in range(60):
    try:
        r = requests.get(f'http://localhost:{port}/health')
        if r.status_code == 200:
            print('Server ready')
            break
    except:
        pass
    time.sleep(2)
else:
    print('ERROR: Server did not start')
    exit(1)
"

# Confirm MambaRadixCache is active
grep -i "MambaRadixCache\|radix.*mamba\|mamba.*radix" /tmp/phase4_server.log | head -10
```

## Tasks

### Task 1: Write the integration test file

Create `test/registered/radix_cache/test_mamba_radix_cache_server_integration.py` with the content in **New Test File(s) to Write** below.

### Task 2: Run the integration tests

```bash
SERVER_URL=http://localhost:$SERVER_PORT \
python -m pytest test/registered/radix_cache/test_mamba_radix_cache_server_integration.py \
    -v --timeout=120 2>&1 | tee /tmp/phase4_tests.log
```

### Task 3: HITL — Server-side state isolation check

> **Warning — history re-injection**: The web UI at http://localhost:3000 loads the full Postgres conversation history and prepends it to every request (`getMessagesByChatId` → `convertToUIMessages`, `route.ts` lines 110–156). A multi-turn chat through the UI would appear to "remember" context due to client-side history re-injection, **not** Mamba server state. Do not use the web UI to prove state persistence.

**Part A — API-direct prefix cache check** (proves MambaRadixCache is working):

```bash
# Send two requests that share a long common prefix.
# The second request's shorter prefill time (visible in server logs) proves a cache hit.
LONG_PREFIX='[{"role":"system","content":"You are a helpful assistant. You are a helpful assistant. You are a helpful assistant. You are a helpful assistant. You are a helpful assistant. You are a helpful assistant. You are a helpful assistant. You are a helpful assistant."}]'

# Request A
curl -s http://localhost:$SERVER_PORT/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d "{\"model\":\"default\",\"messages\":$(echo $LONG_PREFIX | python3 -c \"import sys,json; m=json.load(sys.stdin); m.append({'role':'user','content':'What is 1+1?'}); print(json.dumps(m))\"),\"temperature\":0,\"max_tokens\":10}" \
  2>&1 | tee /tmp/phase4_req_a.json

# Request B (same prefix, different question — should hit radix cache)
curl -s http://localhost:$SERVER_PORT/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d "{\"model\":\"default\",\"messages\":$(echo $LONG_PREFIX | python3 -c \"import sys,json; m=json.load(sys.stdin); m.append({'role':'user','content':'What is 2+2?'}); print(json.dumps(m))\"),\"temperature\":0,\"max_tokens\":10}" \
  2>&1 | tee /tmp/phase4_req_b.json

# Verify cache hit via cached_tokens counter in the response usage object.
# Request B shares the long system prefix with A, so its cached_tokens must be > 0.
python3 -c "
import json, sys

def cached_tokens(path):
    data = json.load(open(path))
    details = data.get('usage', {}).get('prompt_tokens_details', {})
    return details.get('cached_tokens', 0)

ct_a = cached_tokens('/tmp/phase4_req_a.json')
ct_b = cached_tokens('/tmp/phase4_req_b.json')
print(f'Request A cached_tokens: {ct_a}')
print(f'Request B cached_tokens: {ct_b}')
assert ct_b > 0, f'FAIL: Request B had no cached tokens (expected >0 for shared prefix). Got {ct_b}.'
print('PASS: Cache hit confirmed — request B cached_tokens > 0.')
"
```

**Part B — Web UI smoke check** (functional end-to-end only, not a state isolation proof):

Open **http://localhost:3000**, select **SGLang Local**. Have a 3-turn conversation on any topic. Confirm the UI works without errors, streaming functions, and responses are non-empty. This validates the full request path (browser → Next.js → SGLang server) but context here comes from client history, not server state.

```bash
# Log to /tmp/phase4_hitl_log.txt
# First line: "NOTE: Web UI smoke check only — context is client-side history re-injection"
# Format: Turn N | User: <prompt> | Assistant: <response>
```

### Task 4: Inspect server logs for eviction and cache activity

```bash
grep -i "evict\|radix\|mamba_lock\|oom\|cuda error" /tmp/phase4_server.log | tail -50
```

### Task 5: Shut down server

```bash
kill $SERVER_PID
wait $SERVER_PID 2>/dev/null
echo "Server stopped"
```

## New Test File(s) to Write

**Path**: `test/registered/radix_cache/test_mamba_radix_cache_server_integration.py`

```python
from sglang.test.ci.ci_register import register_cuda_ci, register_amd_ci

register_cuda_ci(est_time=180, suite="stage-b-test-small-1-gpu")
register_amd_ci(est_time=180, suite="stage-b-test-small-1-gpu-amd")

import concurrent.futures
import os
import time
import unittest

import requests

SERVER_URL = os.environ.get("SERVER_URL", "http://localhost:30000")
LONG_SYSTEM = "You are a concise assistant. " * 80   # ~500 tokens shared prefix


class TestMambaRadixCacheServerIntegration(unittest.TestCase):

    def _chat(self, messages, **kwargs):
        kwargs.setdefault("temperature", 0)
        kwargs.setdefault("max_tokens", 50)
        payload = {"model": "default", "messages": messages, **kwargs}
        r = requests.post(f"{SERVER_URL}/v1/chat/completions", json=payload, timeout=60)
        r.raise_for_status()
        return r.json()

    def test_cache_hit_on_repeated_prefix(self):
        """Second request sharing a long prefix has shorter prefill (cache hit)."""
        # Request A: long system prompt + question 1
        resp_a = self._chat([
            {"role": "system", "content": LONG_SYSTEM},
            {"role": "user", "content": "What is the capital of France?"},
        ])
        self.assertGreater(len(resp_a["choices"][0]["message"]["content"]), 0)

        # Request B: same system prompt + different question (should hit cache)
        resp_b = self._chat([
            {"role": "system", "content": LONG_SYSTEM},
            {"role": "user", "content": "What is the capital of Germany?"},
        ])
        self.assertGreater(len(resp_b["choices"][0]["message"]["content"]), 0)
        self.assertGreater(
            resp_b["usage"]["prompt_tokens_details"]["cached_tokens"],
            0,
            f"Expected cached_tokens > 0, got: {resp_b['usage']}",
        )

    def test_cache_miss_fallback(self):
        """Unique prefix (never seen before) generates correct output without corruption."""
        import uuid
        unique_prefix = f"Unique context {uuid.uuid4().hex}: "
        resp = self._chat([
            {"role": "user", "content": unique_prefix + "Reply with the word: correct"},
        ])
        content = resp["choices"][0]["message"]["content"].lower()
        self.assertIn("correct", content)

    def test_concurrent_shared_prefix(self):
        """4 concurrent requests sharing the same long system prompt all complete; outputs are independent."""
        messages_base = [{"role": "system", "content": LONG_SYSTEM}]
        questions = [
            "Name one fruit.",
            "Name one planet.",
            "Name one color.",
            "Name one animal.",
        ]

        def send(q):
            return self._chat(messages_base + [{"role": "user", "content": q}])["choices"][0]["message"]["content"].strip()

        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as ex:
            results = list(ex.map(send, questions))

        # All 4 must complete
        self.assertEqual(len(results), 4)
        # All must be non-empty
        for r in results:
            self.assertGreater(len(r), 0)
        # All must be distinct (independent outputs)
        self.assertEqual(len(set(results)), 4, f"Some results were identical: {results}")

    def test_multi_turn_conversation_state_continuity(self):
        """5-turn conversation: each turn relies on server-side state, not replayed history."""
        rid = "continuity-test-rid"

        def turn(user_msg):
            resp = self._chat(
                [{"role": "user", "content": user_msg}],
                rid=rid,
                max_tokens=80,
            )
            return resp["choices"][0]["message"]["content"]

        t1 = turn("My name is Alex and I like the number 42.")
        t2 = turn("What is my name?")
        t3 = turn("What number do I like?")
        t4 = turn("What would you add to 42 to get 100?")
        t5 = turn("Summarize what you know about me in one sentence.")

        # Basic coherence checks
        self.assertIn("alex", t2.lower(), f"Turn 2 forgot the name: {t2}")
        self.assertIn("42", t3, f"Turn 3 forgot the number: {t3}")
        self.assertIn("58", t4, f"Turn 4 arithmetic wrong: {t4}")
        self.assertGreater(len(t5), 10, f"Turn 5 summary too short: {t5}")

    def test_eviction_under_pressure(self):
        """Fill Mamba cache near-capacity with distinct requests; new requests still succeed (eviction works)."""
        # Send many short, unique requests to fill the cache
        for i in range(30):
            resp = self._chat([
                {"role": "user", "content": f"Request number {i}. Reply with: ok{i}"},
            ], max_tokens=10)
            # Each must succeed — eviction must not cause errors
            self.assertEqual(resp["choices"][0]["finish_reason"] in ("stop", "length"), True)
            time.sleep(0.1)  # small delay to avoid rate-limiting

        # Final request must still work
        resp = self._chat([{"role": "user", "content": "Reply with: final_ok"}], max_tokens=10)
        self.assertIn(resp["choices"][0]["finish_reason"], ("stop", "length"))


if __name__ == "__main__":
    unittest.main()
```

## Pass Criteria

- Server starts with `MambaRadixCache` active (confirmed in logs)
- All 5 automated tests pass green
- `test_multi_turn_conversation_state_continuity`: name, number, and arithmetic are correct across all 5 turns
- `test_eviction_under_pressure`: all 30 requests succeed without server error or OOM
- No `CUDA error`, `mamba_lock_ref` assertion failures, or eviction-related exceptions in server log
- HITL Part A: API-direct prefix cache check — second request shows cache hit evidence in server logs
- HITL Part B: Web UI smoke check — 3-turn conversation completes without errors (note: context here is client-side history, not server state proof)

## Write Report

```bash
REPORT="$RESULTS_DIR/phase-04-${MODEL_NAME}-$(date +%Y%m%d-%H%M).md"
cat > "$REPORT" << 'EOF'
# Phase 4 — Live Server Integration: MambaRadixCache + no_buffer
**Model**: granite-4.0-h-tiny
**Date**: <date>
**Result**: PASS | FAIL

## Server
- MambaRadixCache confirmed active in logs: YES | NO
- CUDA errors during run: NONE | <summary>
- Server log: /tmp/phase4_server.log

## Test Results
| Test | Result |
|------|--------|
| test_cache_hit_on_repeated_prefix | PASS/FAIL |
| test_cache_miss_fallback | PASS/FAIL |
| test_concurrent_shared_prefix | PASS/FAIL |
| test_multi_turn_conversation_state_continuity | PASS/FAIL |
| test_eviction_under_pressure | PASS/FAIL |

## HITL
**Part A — API prefix cache check**: PASS | FAIL
Cache hit evidence in server logs: YES | NO
Server log excerpt: <paste relevant lines>

**Part B — Web UI smoke check** (functional only, not state isolation proof):
Result: PASS | FAIL
Turns completed: 3 / errors: 0

## Failures & Tracebacks
<paste here or "None">

## Notes
<eviction behavior, cache hit log evidence, unexpected behavior>
EOF
echo "Report written to $REPORT"
```

## Reporting

```
PHASE 4 RESULT: PASS | FAIL
Tests run: 5  Passed: X  Failed: Y
MambaRadixCache confirmed active: YES | NO
CUDA errors in server log: NONE | <summary>
HITL: PASS (3-turn Web UI smoke check; no server-side context persistence — client history only) | FAIL
Report: $RESULTS_DIR/phase-04-<model>-<date>.md
```
