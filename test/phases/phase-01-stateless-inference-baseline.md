# Phase 1 — Stateless Inference Baseline

## Purpose

Verify the SGLang-Mamba server boots and serves correct outputs **without activating `MambaRadixCache` or snapshot machinery**. The `--disable-radix-cache` flag bypasses the `MambaRadixCache` prefix-avoidance code path, but `HybridReqToTokenPool` (the Mamba-aware memory pool) remains active and will emit its log lines on startup.

## Prerequisites

- Phase 0 complete (all 13 existing tests pass, environment healthy)
- Model checkpoint available (set `MODEL_PATH` before launching)
- No other server instances running on the default port (30000)

## Key Files

- `python/sglang/srt/entrypoints/http_server.py` — HTTP server entry point
- `python/sglang/srt/mem_cache/memory_pool.py` — `HybridReqToTokenPool` (confirm log lines on startup)
- `python/sglang/srt/server_args.py` — `--disable-radix-cache`, `--mamba-scheduler-strategy`
- **New**: `test/registered/radix_cache/test_mamba_baseline_inference.py`

## Environment Setup

```bash
REPO_ROOT=$(git rev-parse --show-toplevel)
cd "$REPO_ROOT"

# Set model path
source test/phases/config.sh   # sets MODEL_PATH, SERVER_PORT, SERVER_URL, RESULTS_DIR

# Launch server (background) — disable radix cache, use default no_buffer strategy
python -m sglang.launch_server \
    --model-path $MODEL_PATH \
    --port $SERVER_PORT \
    --disable-radix-cache \
    --mamba-scheduler-strategy no_buffer \
    > /tmp/phase1_server.log 2>&1 &

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
    print('ERROR: Server did not start in 120s')
    exit(1)
"

# Confirm HybridReqToTokenPool is active (not plain ReqToTokenPool)
grep -i "HybridReqToTokenPool\|mamba" /tmp/phase1_server.log | head -20
```

## Tasks

### Task 1: Write the test file

Create `test/registered/radix_cache/test_mamba_baseline_inference.py` with the content specified in **New Test File(s) to Write** below.

### Task 2: Run the test suite

```bash
# Server must be running before executing this
python -m pytest test/registered/radix_cache/test_mamba_baseline_inference.py -v \
    --server-url http://localhost:$SERVER_PORT \
    2>&1 | tee /tmp/phase1_tests.log
```

If the test file uses `unittest` instead of pytest fixtures, run:

```bash
python test/registered/radix_cache/test_mamba_baseline_inference.py
```

### Task 3: HITL smoke check

Open the chat web UI at **http://localhost:3000** (Next.js chatbot, running under `jeanclawdai`, default model: `sglang/local` → routes to `http://localhost:30000`). Select the **SGLang Local** model if not already selected. Conduct a 3-turn conversation manually and log the exchange. Confirm responses are coherent and contextually connected across turns.

```bash
# Log the HITL exchange to:
# /tmp/phase1_hitl_log.txt
# Format: Turn N — User: <prompt> | Assistant: <response>
```

### Task 4: Check server logs for errors

```bash
grep -i "cuda error\|traceback\|exception\|oom\|out of memory" /tmp/phase1_server.log
# Expected: no matches
```

### Task 5: Shut down server

```bash
kill $SERVER_PID
wait $SERVER_PID 2>/dev/null
echo "Server stopped"
```

## New Test File(s) to Write

**Path**: `test/registered/radix_cache/test_mamba_baseline_inference.py`

```python
from sglang.test.ci.ci_register import register_cuda_ci, register_amd_ci

register_cuda_ci(est_time=120, suite="stage-b-test-small-1-gpu")
register_amd_ci(est_time=120, suite="stage-b-test-small-1-gpu-amd")

import concurrent.futures
import unittest
import requests
import os

SERVER_URL = os.environ.get("SERVER_URL", "http://localhost:30000")


class TestMambaBaselineInference(unittest.TestCase):

    def _chat(self, messages, stream=False, **kwargs):
        payload = {"model": "default", "messages": messages, "stream": stream, **kwargs}
        return requests.post(f"{SERVER_URL}/v1/chat/completions", json=payload, stream=stream, timeout=60)

    def test_health_endpoint(self):
        """GET /health returns 200."""
        r = requests.get(f"{SERVER_URL}/health", timeout=10)
        self.assertEqual(r.status_code, 200)

    def test_single_turn_completion(self):
        """Single /v1/chat/completions request returns non-empty response with correct finish_reason."""
        r = self._chat([{"role": "user", "content": "What is 2+2?"}], temperature=0, max_tokens=50)
        self.assertEqual(r.status_code, 200)
        data = r.json()
        choice = data["choices"][0]
        self.assertIn(choice["finish_reason"], ("stop", "length"))
        self.assertGreater(len(choice["message"]["content"]), 0)

    def test_streaming_completion(self):
        """stream=True: all SSE chunks arrive and final chunk has finish_reason."""
        r = self._chat([{"role": "user", "content": "Count to 5."}], stream=True, temperature=0, max_tokens=50)
        self.assertEqual(r.status_code, 200)
        chunks = list(r.iter_lines())
        # Filter data lines
        data_lines = [l for l in chunks if l.startswith(b"data:") and l != b"data: [DONE]"]
        self.assertGreater(len(data_lines), 0)
        # Last data chunk should contain finish_reason
        import json
        last = json.loads(data_lines[-1][len(b"data:"):])
        self.assertIn(last["choices"][0]["finish_reason"], ("stop", "length"))

    def test_batch_inference_independence(self):
        """N=4 identical prompts at temperature=0 produce identical responses (state isolation)."""
        messages = [{"role": "user", "content": "Reply with exactly the word: apple"}]
        def send(_):
            r = self._chat(messages, temperature=0, max_tokens=10)
            return r.json()["choices"][0]["message"]["content"].strip()

        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as ex:
            results = list(ex.map(send, range(4)))

        self.assertEqual(len(set(results)), 1, f"Responses differed: {results}")

    def test_batch_inference_different_prompts(self):
        """4 different prompts produce semantically distinct responses."""
        prompts = [
            "Name a fruit.",
            "Name a planet.",
            "Name a color.",
            "Name an animal.",
        ]
        def send(p):
            r = self._chat([{"role": "user", "content": p}], temperature=0, max_tokens=20)
            return r.json()["choices"][0]["message"]["content"].strip().lower()

        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as ex:
            results = list(ex.map(send, prompts))

        # All 4 responses should be unique
        self.assertEqual(len(set(results)), 4, f"Some responses were identical: {results}")

    def test_long_context(self):
        """Long system prompt (>512 tokens) does not cause OOM or truncation error."""
        system = "You are a helpful assistant. " * 100  # ~500+ tokens
        r = self._chat([
            {"role": "system", "content": system},
            {"role": "user", "content": "Summarize your role in one sentence."}
        ], temperature=0, max_tokens=50)
        self.assertEqual(r.status_code, 200)
        data = r.json()
        self.assertNotIn("error", data)
        self.assertGreater(len(data["choices"][0]["message"]["content"]), 0)

    def test_sampling_params(self):
        """Varying temperature, top_p, max_new_tokens are respected."""
        # max_new_tokens=5 should produce a short response
        r = self._chat([{"role": "user", "content": "Tell me a long story."}],
                       temperature=0, max_tokens=5)
        self.assertEqual(r.status_code, 200)
        data = r.json()
        # finish_reason should be "length" since we cut short
        self.assertEqual(data["choices"][0]["finish_reason"], "length")
        # Response tokens should be <= 5
        usage = data.get("usage", {})
        if "completion_tokens" in usage:
            self.assertLessEqual(usage["completion_tokens"], 5)


if __name__ == "__main__":
    unittest.main()
```

## Pass Criteria

- Server starts and shows `HybridReqToTokenPool` in logs (confirms Mamba pool active)
- `GET /health` returns 200
- All 7 automated tests pass green
- No CUDA errors or OOM in server logs during or after test run
- HITL: 3-turn conversation produces coherent, contextually connected responses
- Server shuts down cleanly after tests

## Write Report

```bash
REPORT="$RESULTS_DIR/phase-01-${MODEL_NAME}-$(date +%Y%m%d-%H%M).md"
cat > "$REPORT" << EOF
# Phase 1 — Stateless Inference Baseline
**Model**: ${MODEL_NAME}
**Date**: <date>
**Result**: PASS | FAIL

## Server
- HybridReqToTokenPool confirmed in logs: YES | NO
- CUDA errors during run: NONE | <summary>
- Server log: /tmp/phase1_server.log

## Test Results
| Test | Result |
|------|--------|
| test_health_endpoint | PASS/FAIL |
| test_single_turn_completion | PASS/FAIL |
| test_streaming_completion | PASS/FAIL |
| test_batch_inference_independence | PASS/FAIL |
| test_batch_inference_different_prompts | PASS/FAIL |
| test_long_context | PASS/FAIL |
| test_sampling_params | PASS/FAIL |

## HITL (3-turn conversation)
**Result**: PASS | FAIL

| Turn | User | Assistant |
|------|------|-----------|
| 1 | <prompt> | <response> |
| 2 | <prompt> | <response> |
| 3 | <prompt> | <response> |

## Failures & Tracebacks
<paste here or "None">

## Notes
<unexpected behavior, timing, warnings>
EOF
echo "Report written to $REPORT"
```

## Reporting

```
PHASE 1 RESULT: PASS | FAIL
Tests run: 7  Passed: X  Failed: Y
HybridReqToTokenPool confirmed in logs: YES | NO
CUDA errors in server log: NONE | <summary>
HITL: PASS (3 turns coherent) | FAIL
Report: $RESULTS_DIR/phase-01-<model>-<date>.md
```