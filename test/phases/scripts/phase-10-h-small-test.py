#!/usr/bin/env python3
"""
Phase 10: granite-4.0-h-small scaling tests.

Uses /v1/completions (no chat template) and snapshot persistence API.
Records detailed results to test/phases/results/phase-10-logs/.
"""

import json
import os
import subprocess
import time
from datetime import datetime
from pathlib import Path

import requests

SERVER_URL = os.environ.get("SERVER_URL", "http://localhost:30000")
SNAPSHOT_DIR = os.environ.get("SNAPSHOT_DIR", "/tmp/mamba_snapshots")
RESULTS_DIR = Path("test/phases/results/phase-10-logs")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
RESULTS_FILE = RESULTS_DIR / f"h-small-test-{TIMESTAMP}.json"


def get_gpu_vram():
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=2,
        )
        return int(r.stdout.strip()) if r.returncode == 0 else 0
    except Exception:
        return 0


def get_proc_rss():
    try:
        r = subprocess.run(["ps", "aux"], capture_output=True, text=True, timeout=2)
        total_kb = 0
        for line in r.stdout.split("\n"):
            if "sglang::" in line or "launch_server" in line:
                parts = line.split()
                if len(parts) >= 6:
                    try:
                        total_kb += int(parts[5])
                    except ValueError:
                        pass
        return total_kb // 1024
    except Exception:
        return 0


def completion(prompt: str, max_tokens: int = 50, temperature: float = 0.0) -> dict:
    payload = {
        "model": "default",
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    r = requests.post(f"{SERVER_URL}/v1/completions", json=payload, timeout=120)
    r.raise_for_status()
    return r.json()


def snapshot_save(rid: str) -> dict:
    r = requests.post(f"{SERVER_URL}/save_snapshot", json={"rid": rid}, timeout=30)
    return r.json()


def snapshot_restore(
    snapshot_id: str, prompt: str = "Continue: ", max_tokens: int = 30
) -> dict:
    payload = {
        "snapshot_id": snapshot_id,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0.0,
    }
    r = requests.post(f"{SERVER_URL}/restore_snapshot", json=payload, timeout=60)
    return r.json()


def check_health() -> bool:
    try:
        r = requests.get(f"{SERVER_URL}/v1/models", timeout=5)
        return r.status_code == 200
    except Exception:
        return False


def run_tests():
    results = {
        "model": "granite-4.0-h-small",
        "timestamp": TIMESTAMP,
        "tests": [],
        "resource_snapshots": [],
    }

    if not check_health():
        print("ERROR: Server not available")
        return results

    gpu_start = get_gpu_vram()
    rss_start = get_proc_rss()
    print(f"Initial state: GPU={gpu_start}MB, RSS={rss_start}MB")

    # --- Test 1: Basic inference ---
    print("\n=== Test 1: Basic Inference ===")
    t0 = time.time()
    resp = completion("What is 2+2? The answer is", max_tokens=10)
    elapsed = time.time() - t0
    text = resp["choices"][0]["text"].strip()
    test1 = {
        "name": "basic_inference",
        "prompt": "What is 2+2? The answer is",
        "response": text,
        "tokens": resp["usage"]["completion_tokens"],
        "latency_s": round(elapsed, 3),
        "pass": len(text) > 0,
    }
    results["tests"].append(test1)
    print(
        f"  Response: '{text}' ({resp['usage']['completion_tokens']} tokens, {elapsed:.3f}s)"
    )

    # --- Test 2: Multi-turn without snapshots ---
    print("\n=== Test 2: Multi-turn without snapshots (10 turns) ===")
    turns = []
    conversation = "User: My name is Alice and I like cats.\nAssistant: "
    for i in range(10):
        t0 = time.time()
        resp = completion(conversation, max_tokens=40)
        elapsed = time.time() - t0
        reply = resp["choices"][0]["text"].strip()
        turns.append(
            {
                "turn": i + 1,
                "response": reply,
                "latency_s": round(elapsed, 3),
                "tokens": resp["usage"]["completion_tokens"],
            }
        )
        conversation += reply + f"\nUser: Turn {i+2}: What is my name?\nAssistant: "
        print(f"  Turn {i+1}: {reply[:50]}... ({elapsed:.3f}s)")

    test2 = {
        "name": "multi_turn_no_snapshot",
        "turns": turns,
        "pass": all(t["tokens"] > 0 for t in turns),
    }
    results["tests"].append(test2)

    # --- Test 3: Snapshot save/restore ---
    print("\n=== Test 3: Snapshot Save & Restore ===")
    # 3a: Establish conversation
    resp = completion(
        "User: My favorite number is 42 and my color is green. Remember these.\nAssistant: ",
        max_tokens=40,
    )
    rid = resp.get("id", "unknown")
    initial_reply = resp["choices"][0]["text"].strip()
    print(f"  Initial response: {initial_reply[:80]}...")

    # 3b: Save snapshot
    t0 = time.time()
    save_result = snapshot_save(rid)
    save_latency = time.time() - t0
    snapshot_id = save_result.get("snapshot_id")
    print(
        f"  Save: id={snapshot_id}, success={save_result.get('success')}, latency={save_latency:.3f}s"
    )

    test3a = {
        "name": "snapshot_save",
        "snapshot_id": snapshot_id,
        "success": save_result.get("success", False),
        "latency_s": round(save_latency, 3),
        "pass": save_result.get("success", False) and snapshot_id is not None,
    }
    results["tests"].append(test3a)

    if snapshot_id:
        # 3c: Restore snapshot
        t0 = time.time()
        restore_result = snapshot_restore(
            snapshot_id,
            prompt="User: What is my favorite number?\nAssistant: ",
            max_tokens=30,
        )
        restore_latency = time.time() - t0
        restored_reply = ""
        if "choices" in restore_result:
            restored_reply = restore_result["choices"][0]["text"].strip()
        print(f"  Restore: '{restored_reply[:80]}' ({restore_latency:.3f}s)")

        test3b = {
            "name": "snapshot_restore",
            "snapshot_id": snapshot_id,
            "response": restored_reply,
            "latency_s": round(restore_latency, 3),
            "success": "choices" in restore_result,
            "pass": len(restored_reply) > 0,
        }
        results["tests"].append(test3b)

        # 3d: Multiple saves to test snapshot retention
        print("\n=== Test 3d: Multiple Snapshot Saves ===")
        snap_ids = [snapshot_id]
        for i in range(5):
            resp = completion(
                f"User: Fact {i+1}: The capital of France is Paris.\nAssistant: ",
                max_tokens=20,
            )
            rid = resp.get("id", f"turn-{i}")
            save = snapshot_save(rid)
            if save.get("success"):
                snap_ids.append(save["snapshot_id"])
                print(f"  Snapshot {i+1}: {save['snapshot_id'][:20]}...")
            else:
                print(f"  Snapshot {i+1}: FAILED - {save}")

        test3d = {
            "name": "multiple_snapshots",
            "snapshot_count": len(snap_ids),
            "pass": len(snap_ids) >= 4,
        }
        results["tests"].append(test3d)

    # --- Test 4: Rapid fire requests ---
    print("\n=== Test 4: Rapid Fire (50 requests) ===")
    errors = 0
    latencies = []
    for i in range(50):
        t0 = time.time()
        try:
            resp = completion(f"Request {i}: Say OK.", max_tokens=5)
            lat = time.time() - t0
            latencies.append(lat)
        except Exception as e:
            errors += 1
            latencies.append(-1)

    test4 = {
        "name": "rapid_fire",
        "total": 50,
        "errors": errors,
        "avg_latency_s": round(
            sum(l for l in latencies if l > 0)
            / max(1, sum(1 for l in latencies if l > 0)),
            3,
        ),
        "max_latency_s": round(
            max(l for l in latencies if l > 0) if any(l > 0 for l in latencies) else 0,
            3,
        ),
        "min_latency_s": round(
            min(l for l in latencies if l > 0) if any(l > 0 for l in latencies) else 0,
            3,
        ),
        "pass": errors == 0,
    }
    results["tests"].append(test4)
    print(
        f"  {50 - errors}/50 successful, avg={test4['avg_latency_s']}s, max={test4['max_latency_s']}s"
    )

    # --- Test 5: Long context ---
    print("\n=== Test 5: Long Context (2K tokens) ===")
    long_prompt = "This is a test of long context handling. " * 100  # ~600 tokens
    long_prompt += "Q: What was this text about? A: "
    t0 = time.time()
    resp = completion(long_prompt, max_tokens=50)
    elapsed = time.time() - t0
    long_reply = resp["choices"][0]["text"].strip()
    prompt_tokens = resp["usage"]["prompt_tokens"]

    test5 = {
        "name": "long_context_2k",
        "prompt_tokens": prompt_tokens,
        "response": long_reply[:100],
        "latency_s": round(elapsed, 3),
        "pass": len(long_reply) > 0 and prompt_tokens > 500,
    }
    results["tests"].append(test5)
    print(
        f"  {prompt_tokens} prompt tokens, response: '{long_reply[:60]}...', {elapsed:.3f}s"
    )

    # --- Final resource snapshot ---
    gpu_end = get_gpu_vram()
    rss_end = get_proc_rss()
    results["resource_snapshots"] = {
        "gpu_start_mb": gpu_start,
        "gpu_end_mb": gpu_end,
        "gpu_delta_mb": gpu_end - gpu_start,
        "rss_start_mb": rss_start,
        "rss_end_mb": rss_end,
        "rss_delta_mb": rss_end - rss_start,
    }
    print(f"\n=== Resources ===")
    print(f"  GPU: {gpu_start} -> {gpu_end} MB (delta: {gpu_end - gpu_start:+d})")
    print(f"  RSS: {rss_start} -> {rss_end} MB (delta: {rss_end - rss_start:+d})")

    # --- Health check after stress ---
    health = check_health()
    results["post_test_health"] = health
    print(f"  Post-test health: {'OK' if health else 'FAIL'}")

    # --- Summary ---
    total = len(results["tests"])
    passed = sum(1 for t in results["tests"] if t.get("pass"))
    results["summary"] = {"total": total, "passed": passed, "failed": total - passed}
    print(f"\n=== Summary: {passed}/{total} tests passed ===")

    # Save results
    with open(RESULTS_FILE, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to: {RESULTS_FILE}")

    return results


if __name__ == "__main__":
    run_tests()
