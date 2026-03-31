# Phase 9 — Gauntlet / Stress Tests

## Purpose

Break things under load to surface race conditions, memory leaks, eviction policy failures, and state contamination that only appear under sustained pressure. This phase runs the server with concurrent request floods, rapidly cycling sequences, mixed request types, and near-OOM conditions. All earlier invariants (state isolation, eviction correctness, LRU integrity, no CUDA errors) must hold under stress.

> **Note**: The original agent prompt was truncated before the task list for this phase was provided. The test structure below is reconstructed from context (radix cache internals, stress test patterns, and Phase 3 test plan). The implementing agent should inspect `phase3/test/test_plan.md` and `phase3/tests/test_coverage_report.md` for any additional stress test requirements documented there, and add them to this phase.

## Prerequisites

- **All prior phases complete** (Phases 0–8 must pass before running stress tests)
- Model checkpoint available at `$MODEL_PATH`
- At least 20 GB VRAM free (stress tests may push near memory limits)
- At least 50 GB disk space (for snapshot stress tests)
- No other server instances running on port 30000

## Key Files

- `python/sglang/srt/mem_cache/mamba_radix_cache.py` — `sanity_check()`, eviction methods, lock refs
- `python/sglang/srt/mem_cache/memory_pool.py` — `MambaPool`, available_size tracking
- `python/sglang/srt/entrypoints/http_server.py` — HTTP server
- `phase3/test/test_plan.md` — check for any additional planned stress tests
- **New**: `test/registered/radix_cache/test_mamba_gauntlet_stress.py`

## Environment Setup

```bash
REPO_ROOT=$(git rev-parse --show-toplevel)
cd "$REPO_ROOT"

source test/phases/config.sh   # sets MODEL_PATH, SERVER_PORT, SERVER_URL, SNAPSHOT_DIR, RESULTS_DIR

# Launch server with all features enabled for maximum stress coverage
python -m sglang.launch_server \
    --model-path $MODEL_PATH \
    --port $SERVER_PORT \
    --mamba-scheduler-strategy no_buffer \
    > /tmp/phase9_server.log 2>&1 &

export SERVER_PID=$!

# Wait for server ready
python -c "
import time, requests, os
server_url = os.environ.get('SERVER_URL', f\"http://localhost:{os.environ.get('SERVER_PORT', '30000')}\")
for i in range(90):
    try:
        r = requests.get(f'{server_url}/health')
        if r.status_code == 200:
            print('Server ready (stress mode)')
            break
    except:
        pass
    time.sleep(2)
else:
    print('ERROR: Server did not start in 180s')
    exit(1)
"
```

## Tasks

### Task 0: Review phase3 test plan for additional stress requirements

```bash
cat phase3/test/test_plan.md | grep -A 3 "stress\|gauntlet\|concurrent\|race\|leak" -i
```

Incorporate any discovered planned-but-unwritten stress tests into the test file.

### Task 1: Write the stress test file

Create `test/registered/radix_cache/test_mamba_gauntlet_stress.py` using the structure in **New Test File(s) to Write** below.

### Task 2: Run the stress tests

```bash
# Stress tests may take 10–30 minutes
SERVER_URL=http://localhost:$SERVER_PORT \
python -m pytest test/registered/radix_cache/test_mamba_gauntlet_stress.py \
    -v --timeout=600 2>&1 | tee /tmp/phase9_stress.log
```

### Task 3: Monitor server health during stress

In a separate terminal, poll server health every 30 seconds:

```bash
for i in $(seq 1 60); do
    echo -n "[$i] "; curl -s "$SERVER_URL/health" | python -m json.tool --no-indent || echo "UNHEALTHY"
    sleep 30
done
```

### Task 4: Collect post-stress diagnostics

```bash
# Check for CUDA errors, OOM, assertion failures, lock ref violations
grep -i "cuda error\|out of memory\|oom\|assertion\|traceback\|lock_ref\|sanity" \
    /tmp/phase9_server.log | head -100

# Check if server is still responsive after stress
curl -s "$SERVER_URL/health"
```

### Task 5: Shut down server

```bash
kill $SERVER_PID
wait $SERVER_PID 2>/dev/null
echo "Server stopped"
```

## New Test File(s) to Write

**Path**: `test/registered/radix_cache/test_mamba_gauntlet_stress.py`

```python
from sglang.test.ci.ci_register import register_cuda_ci, register_amd_ci

# Stress tests: longer timeout, same suite (or consider a nightly suite)
register_cuda_ci(est_time=300, suite="nightly-1-gpu", nightly=True)
register_amd_ci(est_time=300, suite="nightly-amd-1-gpu", nightly=True)

import concurrent.futures
import json
import os
import re
import time
import unittest
import uuid

import requests

SERVER_URL = os.environ.get("SERVER_URL", "http://localhost:30000")
LONG_SYSTEM = "You are a helpful assistant. " * 60


def strip_markdown_json(content: str) -> str:
    cleaned = content.strip()
    fenced = re.match(r"^```(?:json)?\s*(.*?)\s*```$", cleaned, re.DOTALL)
    if fenced:
        return fenced.group(1).strip()
    if cleaned.startswith("`") and cleaned.endswith("`"):
        return cleaned[1:-1].strip()
    return cleaned


class TestMambaGauntletStress(unittest.TestCase):
    """Stress tests — require a running server. All tests use skipTest if server unavailable."""

    def setUp(self):
        try:
            r = requests.get(f"{SERVER_URL}/health", timeout=5)
            if r.status_code != 200:
                self.skipTest("Server not available")
        except Exception:
            self.skipTest("Server not available")

    def _chat(self, messages, **kwargs):
        kwargs.setdefault("temperature", 0)
        kwargs.setdefault("max_tokens", 30)
        payload = {"model": "default", "messages": messages, **kwargs}
        r = requests.post(f"{SERVER_URL}/v1/chat/completions", json=payload, timeout=120)
        r.raise_for_status()
        return r.json()

    def test_high_concurrency_shared_prefix(self):
        """32 concurrent requests sharing the same long prefix all complete without error or state contamination."""
        N = 32
        base_messages = [{"role": "system", "content": LONG_SYSTEM}]
        questions = [f"What is {i} + {i}?" for i in range(N)]

        def send(q):
            resp = self._chat(base_messages + [{"role": "user", "content": q}])
            return resp["choices"][0]["message"]["content"].strip()

        with concurrent.futures.ThreadPoolExecutor(max_workers=N) as ex:
            results = list(ex.map(send, questions))

        # All N must complete
        self.assertEqual(len(results), N)
        # All must be non-empty
        for r in results:
            self.assertGreater(len(r), 0, f"Empty response: {r}")

    def test_rapid_distinct_requests_eviction_pressure(self):
        """100 rapid requests with unique prefixes — server stays healthy, eviction doesn't crash."""
        errors = []
        for i in range(100):
            try:
                resp = self._chat([
                    {"role": "user", "content": f"Unique-{uuid.uuid4().hex}: say ok{i}"},
                ], max_tokens=5)
                self.assertIn(resp["choices"][0]["finish_reason"], ("stop", "length"))
            except Exception as e:
                errors.append(str(e))

        self.assertEqual(errors, [], f"Errors during eviction stress: {errors}")

    def test_repeated_same_request_cache_stability(self):
        """The same request sent 50 times in sequence produces consistent outputs; no crash or corruption."""
        messages = [{"role": "user", "content": "Reply with exactly: STABLE"}]
        outputs = []
        for _ in range(50):
            resp = self._chat(messages, max_tokens=10)
            outputs.append(resp["choices"][0]["message"]["content"].strip())

        # All responses must be non-empty and successful
        for o in outputs:
            self.assertGreater(len(o), 0, f"Empty output in repetition run: {outputs}")

        # At temperature=0, all 50 outputs must be identical — any divergence indicates
        # state corruption or nondeterminism that constitutes a test failure.
        unique = set(outputs)
        self.assertEqual(len(unique), 1,
            f"Outputs diverged across runs — expected all 50 identical at temperature=0: {unique}")

    def test_alternating_long_and_short_requests(self):
        """Interleave long-context and short requests 20 times; verify no cross-contamination."""
        long_msgs = [
            {"role": "system", "content": LONG_SYSTEM},
            {"role": "user", "content": "Summarize your role."},
        ]
        short_msgs = [{"role": "user", "content": "Say: short"}]

        for i in range(20):
            long_resp = self._chat(long_msgs, max_tokens=20)
            short_resp = self._chat(short_msgs, max_tokens=5)
            self.assertGreater(len(long_resp["choices"][0]["message"]["content"]), 0)
            self.assertGreater(len(short_resp["choices"][0]["message"]["content"]), 0)

    def test_server_health_after_stress(self):
        """After all stress tests run, server is still responsive and returns 200 on /health."""
        r = requests.get(f"{SERVER_URL}/health", timeout=10)
        self.assertEqual(r.status_code, 200, "Server became unhealthy after stress tests")

    def test_concurrent_multi_turn_conversations(self):
        """8 concurrent 5-turn conversations, each with a unique persona, all stay coherent."""
        personas = [f"User{i}" for i in range(8)]

        def run_conversation(persona):
            history = []
            history.append({"role": "user", "content": f"My name is {persona}. Reply with JSON: {{\"name\":\"{persona}\"}}"})
            resp = self._chat(history, max_tokens=60)
            history.append({"role": "assistant", "content": resp["choices"][0]["message"]["content"]})
            try:
                parsed = json.loads(
                    strip_markdown_json(resp["choices"][0]["message"]["content"])
                )
            except (json.JSONDecodeError, ValueError):
                return f"FAIL: {persona} gave non-JSON response: {resp['choices'][0]['message']['content']}"
            if parsed.get("name") != persona:
                return f"FAIL: {persona} name mismatch: {parsed.get('name')}"

            for turn in range(4):
                history.append({"role": "user", "content": f"Turn {turn+2}: what is my name? Reply with JSON: {{\"name\":\"{persona}\"}}"})
                resp = self._chat(history, max_tokens=60)
                content = resp["choices"][0]["message"]["content"]
                history.append({"role": "assistant", "content": content})
                try:
                    parsed = json.loads(strip_markdown_json(content))
                except (json.JSONDecodeError, ValueError):
                    return f"FAIL: {persona} turn {turn+2} non-JSON: {content}"
                if parsed.get("name") != persona:
                    return f"FAIL: {persona} turn {turn+2} name mismatch: {parsed.get('name')}"
            return f"PASS: {persona}"

        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as ex:
            results = list(ex.map(run_conversation, personas))

        failures = [r for r in results if r.startswith("FAIL")]
        self.assertEqual(failures, [], f"Conversation coherence failures: {failures}")


# For any additional finalized gauntlet scenarios, consult phase3/test/test_plan.md
# and add only the completed test designs here before execution.


if __name__ == "__main__":
    unittest.main()
```

## Pass Criteria

- All 6 implemented stress tests pass
- `test_high_concurrency_shared_prefix`: all 32 requests complete, non-empty, no errors
- `test_rapid_distinct_requests_eviction_pressure`: 100 requests complete, zero errors
- `test_repeated_same_request_cache_stability`: all 50 outputs identical (temperature=0 — any divergence is a failure)
- `test_concurrent_multi_turn_conversations`: all 8 conversations maintain persona across 5 turns
- `test_server_health_after_stress`: server still returns 200 after full gauntlet
- No `CUDA error`, `mamba_lock_ref` assertion, or `sanity_check` failure in server logs during or after run
- Server does not OOM or crash at any point

## Write Report

```bash
REPORT="$RESULTS_DIR/phase-09-${MODEL_NAME}-$(date +%Y%m%d-%H%M).md"
cat > "$REPORT" << 'EOF'
# Phase 9 — Gauntlet / Stress Tests
**Model**: granite-4.0-h-tiny
**Date**: <date>
**Duration**: <total runtime>
**Result**: PASS | FAIL

## Server Health
- Healthy post-stress (/health returns 200): YES | NO
- CUDA errors logged: NONE | <count and excerpts>
- Eviction errors: NONE | <details>
- mamba_lock_ref violations: NONE | <details>
- Server log size: <bytes>

## Test Results
| Test | Requests | Pass | Errors |
|------|----------|------|--------|
| test_high_concurrency_shared_prefix | 32 | X | Y |
| test_rapid_distinct_requests_eviction_pressure | 100 | X | Y |
| test_repeated_same_request_cache_stability | 50 | X | Y |
| test_alternating_long_and_short_requests | 40 | X | Y |
| test_concurrent_multi_turn_conversations | 8×5 turns | X | Y |
| test_server_health_after_stress | 1 | X | Y |

## Additional Tests from phase3/test/test_plan.md
<list tests added, or "none found">

## Observed Anomalies
<race conditions, unexpected slowdowns, warnings worth investigating>

## Failures & Tracebacks
<paste here or "None">
EOF
echo "Report written to $REPORT"
```

## Reporting

```text
PHASE 9 RESULT: PASS | FAIL
Tests run: 6+  Passed: X  Failed: Y
Server healthy post-stress: YES | NO
CUDA errors: NONE | <count>
Report: $RESULTS_DIR/phase-08-<model>-<date>.md
```
