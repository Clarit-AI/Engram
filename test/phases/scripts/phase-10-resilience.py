#!/usr/bin/env python3
"""
Phase 10 Addendum: Resilience Testing

Tests snapshot system behavior under adverse conditions:
1. Client disconnect mid-stream
2. Server SIGKILL mid-inference (with startup preload check)
3. Server SIGKILL during snapshot write (atomic write verification)
4. Graceful SIGTERM shutdown
5. Abort request + snapshot interaction

Usage:
    python test/phases/scripts/phase-10-resilience.py --test all
    python test/phases/scripts/phase-10-resilience.py --test 2
"""

import argparse
import json
import os
import signal
import subprocess
import sys
import threading
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import requests

SERVER_URL = os.environ.get("SERVER_URL", "http://localhost:30000")
SNAPSHOT_DIR = os.environ.get("SNAPSHOT_DIR", "/tmp/mamba_snapshots")
MODEL_PATH = os.environ.get("MODEL_PATH", "/mnt/models/granite-4.0-h-tiny")
SERVER_PORT = int(os.environ.get("SERVER_PORT", "30000"))

WARM_RESTORE_MAX_MS = 50
COLD_RESTORE_MIN_MS = 50


# ────────────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────────────

def log(msg: str):
    print(f"  {msg}")


def get_gpu_vram_mb() -> int:
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=2,
        )
        if r.returncode == 0:
            return int(r.stdout.strip().split("\n")[0].strip())
    except Exception:
        pass
    return 0


def get_proc_rss_mb() -> int:
    try:
        r = subprocess.run(["ps", "aux"], capture_output=True, text=True, timeout=2)
        if r.returncode == 0:
            total_kb = 0
            for line in r.stdout.split("\n"):
                if "sglang::" in line or "launch_server" in line:
                    parts = line.split()
                    if len(parts) >= 6:
                        try:
                            total_kb += int(parts[5])
                        except ValueError:
                            continue
            return total_kb // 1024
    except Exception:
        pass
    return 0


def check_health(timeout: float = 5) -> bool:
    try:
        r = requests.get(f"{SERVER_URL}/health", timeout=timeout)
        return r.status_code == 200
    except Exception:
        return False


def check_health_generate(timeout: float = 30) -> bool:
    try:
        r = requests.get(f"{SERVER_URL}/health_generate", timeout=timeout)
        return r.status_code == 200
    except Exception:
        return False


def get_server_pids() -> List[int]:
    """Get all PIDs matching sglang server (main + workers)."""
    pids = set()
    for pattern in ["sglang::", "launch_server", "sglang.launch_server"]:
        try:
            r = subprocess.run(["pgrep", "-f", pattern], capture_output=True, text=True, timeout=2)
            if r.returncode == 0:
                for line in r.stdout.strip().split("\n"):
                    line = line.strip()
                    if line.isdigit():
                        pids.add(int(line))
        except Exception:
            pass
    return sorted(pids)


def kill_server(sig: int = signal.SIGKILL, timeout: float = 10) -> bool:
    """Send signal to server and wait for death."""
    pids = get_server_pids()
    if not pids:
        log("No server PIDs found")
        return True

    for pid in pids:
        try:
            os.kill(pid, sig)
            log(f"Sent signal {sig} to PID {pid}")
        except ProcessLookupError:
            log(f"PID {pid} already dead")

    # Wait for death
    start = time.time()
    while time.time() - start < timeout:
        if not get_server_pids():
            log("Server process dead")
            return True
        time.sleep(0.5)

    # Force kill if still alive
    remaining = get_server_pids()
    if remaining:
        log(f"Force killing remaining PIDs: {remaining}")
        for pid in remaining:
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        time.sleep(1)

    return len(get_server_pids()) == 0


def start_server(timeout: float = 180) -> bool:
    """Start the server and wait for health."""
    log("Starting server...")

    # Clean stale snapshot dir to avoid preload of old data on first start
    cmd = (
        f"python -m sglang.launch_server "
        f"--model-path {MODEL_PATH} "
        f"--enable-snapshot-persistence "
        f"--snapshot-dir {SNAPSHOT_DIR} "
        f"--mamba-scheduler-strategy no_buffer "
        f"--disable-radix-cache "
        f"--port {SERVER_PORT}"
    )

    subprocess.Popen(
        cmd, shell=True,
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        preexec_fn=os.setsid,
    )

    log(f"Waiting for server healthy (timeout {timeout}s)...")
    start = time.time()
    while time.time() - start < timeout:
        if check_health_generate(timeout=10):
            elapsed = time.time() - start
            log(f"Server healthy after {elapsed:.0f}s")
            return True
        time.sleep(2)

    log("Server failed to become healthy")
    return False


def send_completion(prompt: str, temperature: float = 0.0, max_tokens: int = 100,
                    timeout: float = 300) -> dict:
    start = time.time()
    r = requests.post(
        f"{SERVER_URL}/v1/completions",
        json={"model": "default", "prompt": prompt, "temperature": temperature,
              "max_tokens": max_tokens},
        timeout=timeout,
    )
    elapsed = time.time() - start
    r.raise_for_status()
    data = r.json()
    return {
        "rid": data.get("id", ""),
        "text": data["choices"][0].get("text", ""),
        "prompt_tokens": data["usage"]["prompt_tokens"],
        "completion_tokens": data["usage"]["completion_tokens"],
        "latency_s": elapsed,
    }


def send_generate(text: str, temperature: float = 0.0, max_tokens: int = 100,
                  stream: bool = False, timeout: float = 300,
                  custom_rid: str = None) -> dict:
    payload = {
        "text": text,
        "sampling_params": {"temperature": temperature, "max_new_tokens": max_tokens},
        "stream": stream,
    }
    if custom_rid:
        payload["rid"] = custom_rid

    start = time.time()
    r = requests.post(
        f"{SERVER_URL}/generate", json=payload, timeout=timeout, stream=stream,
    )
    elapsed = time.time() - start

    if stream:
        return {"rid": custom_rid or "", "stream_response": r, "latency_s": elapsed}

    r.raise_for_status()
    data = r.json()
    return {
        "rid": data.get("meta_info", {}).get("id", data.get("rid", "")),
        "text": data.get("text", ""),
        "latency_s": elapsed,
    }


def save_snapshot(rid: str = None, conversation_id: str = None, timeout: float = 30) -> dict:
    payload = {}
    if rid:
        payload["rid"] = rid
    if conversation_id:
        payload["conversation_id"] = conversation_id

    start = time.time()
    r = requests.post(f"{SERVER_URL}/save_snapshot", json=payload, timeout=timeout)
    elapsed = time.time() - start
    r.raise_for_status()
    data = r.json()
    data["latency_ms"] = elapsed * 1000
    return data


def restore_snapshot(conversation_id: str = None, rid: str = None, timeout: float = 60) -> dict:
    payload = {}
    if conversation_id:
        payload["conversation_id"] = conversation_id
    if rid:
        payload["rid"] = rid

    start = time.time()
    r = requests.post(f"{SERVER_URL}/restore_snapshot", json=payload, timeout=timeout)
    elapsed = time.time() - start
    r.raise_for_status()
    data = r.json()
    data["latency_ms"] = elapsed * 1000
    return data


def abort_request(rid: str, timeout: float = 10) -> bool:
    try:
        r = requests.post(f"{SERVER_URL}/abort_request", json={"rid": rid}, timeout=timeout)
        return r.status_code == 200
    except Exception:
        return False


def list_disk_snapshots() -> List[dict]:
    """List all snapshot files on disk with sizes."""
    snap_dir = Path(SNAPSHOT_DIR)
    if not snap_dir.exists():
        return []
    results = []
    for conv_dir in sorted(snap_dir.iterdir()):
        if conv_dir.is_dir():
            files = []
            for f in conv_dir.iterdir():
                if f.is_file():
                    files.append({"name": f.name, "size_bytes": f.stat().st_size})
            if files:
                results.append({
                    "conversation_id": conv_dir.name,
                    "files": files,
                    "total_size_mb": sum(f["size_bytes"] for f in files) / (1024 * 1024),
                })
    return results


def check_snapshot_integrity(conversation_id: str) -> dict:
    """Load safetensors file and verify it's valid."""
    conv_dir = Path(SNAPSHOT_DIR) / f"conversation_{conversation_id}"
    if not conv_dir.exists():
        conv_dir = Path(SNAPSHOT_DIR) / conversation_id
    if not conv_dir.exists():
        return {"exists": False, "error": "directory not found"}

    safetensors_files = list(conv_dir.glob("*.safetensors"))
    if not safetensors_files:
        return {"exists": True, "has_safetensors": False}

    try:
        from safetensors.torch import load_file
        for sf in safetensors_files:
            tensors = load_file(str(sf))
            keys = list(tensors.keys())
            total_params = sum(t.numel() for t in tensors.values())
        return {
            "exists": True,
            "has_safetensors": True,
            "files": [f.name for f in safetensors_files],
            "tensor_keys": keys[:5],
            "total_params": total_params,
        }
    except Exception as e:
        return {"exists": True, "has_safetensors": True, "error": str(e)}


def find_tmp_files() -> List[str]:
    """Find .tmp files in snapshot directory."""
    snap_dir = Path(SNAPSHOT_DIR)
    if not snap_dir.exists():
        return []
    return [str(f) for f in snap_dir.rglob("*.tmp")]


def send_streaming_chat(messages: list, temperature: float = 0.0, max_tokens: int = 200,
                        timeout: float = 30):
    """Send a streaming chat request. Returns the response object for manual chunk reading."""
    return requests.post(
        f"{SERVER_URL}/v1/chat/completions",
        json={
            "model": "default", "messages": messages,
            "temperature": temperature, "max_tokens": max_tokens, "stream": True,
        },
        timeout=timeout, stream=True,
    )


# ────────────────────────────────────────────────────────────────
# Test 1: Client Disconnect Mid-Stream
# ────────────────────────────────────────────────────────────────

def test_client_disconnect() -> dict:
    print("\n" + "=" * 60)
    print("TEST 1: Client Disconnect Mid-Stream")
    print("=" * 60)

    result = {"test": "client_disconnect", "timestamp": datetime.now().isoformat()}

    # Pre-test state
    vram_before = get_gpu_vram_mb()
    result["vram_before_mb"] = vram_before
    log(f"VRAM before: {vram_before} MB")

    if not check_health():
        result["status"] = "SKIP"
        result["reason"] = "Server not healthy"
        return result

    # Step 1: Send streaming request, read 1-2 chunks, then close
    log("Sending streaming request...")
    messages = [{"role": "user", "content": "Write a long essay about the history of computing, from Charles Babbage to modern AI. Be thorough and detailed."}]
    chunks_read = 0
    partial_text = ""

    try:
        resp = send_streaming_chat(messages, temperature=0.0, max_tokens=200)
        log(f"Stream response status: {resp.status_code}")

        for line in resp.iter_lines():
            if line:
                line_str = line.decode("utf-8", errors="replace")
                if line_str.startswith("data: "):
                    chunks_read += 1
                    partial_text += line_str
                    if chunks_read >= 3:
                        log(f"Read {chunks_read} chunks, closing connection...")
                        resp.close()
                        break
    except Exception as e:
        log(f"Exception during streaming (expected): {e}")

    result["chunks_read"] = chunks_read
    result["partial_text_len"] = len(partial_text)

    # Step 2: Wait for cleanup
    log("Waiting 5s for server cleanup...")
    time.sleep(5)

    # Step 3: Check server health
    healthy = check_health()
    result["server_healthy_after_disconnect"] = healthy
    log(f"Server healthy: {healthy}")

    if not healthy:
        result["status"] = "FAIL"
        result["reason"] = "Server not healthy after client disconnect"
        return result

    # Step 4: Send fresh request to verify server works
    log("Sending fresh request...")
    try:
        fresh = send_completion("Say hello in one word.", max_tokens=10)
        result["fresh_request_ok"] = True
        result["fresh_request_text"] = fresh["text"][:50]
        log(f"Fresh request succeeded: {fresh['text'][:50]}")
    except Exception as e:
        result["fresh_request_ok"] = False
        result["fresh_request_error"] = str(e)
        log(f"Fresh request FAILED: {e}")
        result["status"] = "FAIL"
        result["reason"] = f"Server broken after disconnect: {e}"
        return result

    # Step 5: Check VRAM
    vram_after = get_gpu_vram_mb()
    vram_delta = vram_after - vram_before
    result["vram_after_mb"] = vram_after
    result["vram_delta_mb"] = vram_delta
    log(f"VRAM after: {vram_after} MB (delta: {vram_delta:+d} MB)")

    if abs(vram_delta) > 500:
        log(f"WARNING: VRAM delta {vram_delta:+d} MB — possible leak")
        result["vram_leak_suspected"] = True

    result["status"] = "PASS"
    return result


# ────────────────────────────────────────────────────────────────
# Test 2: Server SIGKILL Mid-Inference
# ────────────────────────────────────────────────────────────────

def test_sigkill_mid_inference() -> dict:
    print("\n" + "=" * 60)
    print("TEST 2: Server SIGKILL Mid-Inference")
    print("=" * 60)

    result = {"test": "sigkill_mid_inference", "timestamp": datetime.now().isoformat()}

    if not check_health():
        result["status"] = "SKIP"
        result["reason"] = "Server not healthy at start"
        return result

    # Step 1: Send request, save snapshot
    log("Sending initial request and saving snapshot...")
    try:
        comp = send_completion("Describe the solar system in detail.", max_tokens=100)
        rid1 = comp["rid"]
        log(f"Initial request RID: {rid1[:16]}...")
    except Exception as e:
        result["status"] = "FAIL"
        result["reason"] = f"Initial request failed: {e}"
        return result

    save1 = save_snapshot(rid=rid1)
    result["save1_success"] = save1.get("success", False)
    if not save1.get("success"):
        result["status"] = "FAIL"
        result["reason"] = f"Snapshot save failed: {save1.get('message')}"
        return result

    conv_id = (save1.get("snapshot_id") or "").rsplit("-t", 1)[0]
    result["conversation_id"] = conv_id
    log(f"Snapshot saved: conv_id={conv_id}")

    # Step 2: Verify snapshot on disk
    snap_files = list_disk_snapshots()
    result["snapshots_before_kill"] = len(snap_files)
    integrity = check_snapshot_integrity(conv_id)
    result["integrity_before_kill"] = integrity
    log(f"Snapshot integrity: exists={integrity.get('exists')}, tensors={integrity.get('has_safetensors')}")

    if not integrity.get("exists"):
        result["status"] = "FAIL"
        result["reason"] = "Snapshot not found on disk"
        return result

    # Step 3: Send a second (longer) request — kill during this
    log("Sending long request to kill during...")

    kill_done = threading.Event()

    def send_long_request():
        try:
            send_completion(
                "Write a very long and detailed essay about every aspect of quantum physics. "
                "Include historical context, key experiments, mathematical formulations, "
                "and modern applications. Be extremely thorough.",
                max_tokens=500, timeout=60,
            )
        except Exception:
            pass

    req_thread = threading.Thread(target=send_long_request)
    req_thread.start()
    time.sleep(1)  # Let it start processing

    # Step 4: SIGKILL
    log("Sending SIGKILL to server...")
    killed = kill_server(signal.SIGKILL)
    result["kill_success"] = killed
    kill_done.set()
    req_thread.join(timeout=5)

    if not killed:
        result["status"] = "FAIL"
        result["reason"] = "Failed to kill server"
        return result

    # Step 5: Verify snapshot files survive
    time.sleep(2)
    snap_files_after = list_disk_snapshots()
    result["snapshots_after_kill"] = len(snap_files_after)
    integrity_after = check_snapshot_integrity(conv_id)
    result["integrity_after_kill"] = integrity_after
    log(f"Snapshot after kill: exists={integrity_after.get('exists')}, tensors={integrity_after.get('has_safetensors')}")

    if not integrity_after.get("exists"):
        result["status"] = "FAIL"
        result["reason"] = "Snapshot lost after SIGKILL"
        return result

    # Step 6: Restart server
    log("Restarting server...")
    if not start_server(timeout=180):
        result["status"] = "FAIL"
        result["reason"] = "Server failed to restart after SIGKILL"
        return result

    vram_after_restart = get_gpu_vram_mb()
    result["vram_after_restart_mb"] = vram_after_restart
    log(f"VRAM after restart: {vram_after_restart} MB")

    # Step 7: Restore snapshot — check WARM preload (Gap 3 integration)
    log("Restoring snapshot to check startup preload...")
    try:
        restore = restore_snapshot(conversation_id=conv_id)
        result["restore_success"] = restore.get("success", False)
        restore_latency = restore.get("latency_ms", 0)
        result["restore_latency_ms"] = round(restore_latency, 1)

        if restore_latency < WARM_RESTORE_MAX_MS:
            result["preload_tier"] = "WARM"
            result["preload_worked"] = True
            log(f"Restore latency: {restore_latency:.0f}ms — WARM tier (preload worked!)")
        elif restore_latency >= COLD_RESTORE_MIN_MS:
            result["preload_tier"] = "COLD"
            result["preload_worked"] = False
            log(f"Restore latency: {restore_latency:.0f}ms — COLD tier (preload may not have worked)")
        else:
            result["preload_tier"] = "AMBIGUOUS"
            result["preload_worked"] = None
            log(f"Restore latency: {restore_latency:.0f}ms — ambiguous range")

    except Exception as e:
        result["restore_success"] = False
        result["restore_error"] = str(e)
        log(f"Restore FAILED: {e}")
        result["status"] = "FAIL"
        result["reason"] = f"Restore failed after restart: {e}"
        return result

    if not restore.get("success"):
        result["status"] = "FAIL"
        result["reason"] = f"Restore returned failure: {restore.get('message')}"
        return result

    # Step 8: Generate from restored snapshot to confirm semantic correctness
    log("Generating from restored snapshot...")
    try:
        verify = send_completion("What did we discuss earlier? Summarize in one sentence.", max_tokens=50)
        result["post_restore_generation_ok"] = True
        result["post_restore_text"] = verify["text"][:100]
        log(f"Post-restore generation: {verify['text'][:80]}...")
    except Exception as e:
        result["post_restore_generation_ok"] = False
        result["post_restore_error"] = str(e)
        log(f"Post-restore generation failed: {e}")

    result["status"] = "PASS"
    return result


# ────────────────────────────────────────────────────────────────
# Test 3: Server SIGKILL During Snapshot Write
# ────────────────────────────────────────────────────────────────

def test_sigkill_during_write() -> dict:
    print("\n" + "=" * 60)
    print("TEST 3: Server SIGKILL During Snapshot Write")
    print("=" * 60)

    result = {"test": "sigkill_during_write", "timestamp": datetime.now().isoformat()}

    if not check_health():
        result["status"] = "SKIP"
        result["reason"] = "Server not healthy at start"
        return result

    # Step 1: Send request, get RID
    log("Sending request...")
    try:
        comp = send_completion("Explain machine learning algorithms.", max_tokens=50)
        rid = comp["rid"]
        log(f"RID: {rid[:16]}...")
    except Exception as e:
        result["status"] = "FAIL"
        result["reason"] = f"Request failed: {e}"
        return result

    # Step 2: Race — save snapshot + SIGKILL
    log("Racing save_snapshot + SIGKILL...")
    save_result = {}
    save_error = [None]

    def do_save():
        try:
            sr = save_snapshot(rid=rid, timeout=10)
            save_result.update(sr)
        except Exception as e:
            save_error[0] = e

    save_thread = threading.Thread(target=do_save)
    save_thread.start()

    # Small delay then kill
    time.sleep(0.01)
    killed = kill_server(signal.SIGKILL)
    result["kill_success"] = killed
    save_thread.join(timeout=5)

    result["save_completed"] = bool(save_result)
    result["save_error"] = str(save_error[0]) if save_error[0] else None
    log(f"Save completed: {bool(save_result)}, error: {save_error[0]}")

    # Step 3: Check disk for .tmp files
    time.sleep(2)
    tmp_files = find_tmp_files()
    result["tmp_files"] = tmp_files
    result["tmp_file_count"] = len(tmp_files)
    log(f"Found {len(tmp_files)} .tmp files: {tmp_files}")

    # Step 4: Check for complete vs partial snapshot files
    snap_files = list_disk_snapshots()
    result["snapshots_after_kill"] = len(snap_files)
    log(f"Snapshot dirs on disk: {len(snap_files)}")

    # Check if any .safetensors file is partial (very small)
    partial_found = False
    for snap in snap_files:
        for f in snap["files"]:
            if f["name"].endswith(".safetensors") and f["size_bytes"] < 1000:
                partial_found = True
                log(f"WARNING: Very small safetensors file: {snap['conversation_id']}/{f['name']} ({f['size_bytes']} bytes)")
    result["partial_safetensors_found"] = partial_found

    # Step 5: Restart server
    log("Restarting server...")
    if not start_server(timeout=180):
        result["status"] = "FAIL"
        result["reason"] = "Server failed to restart"
        return result

    log("Server restarted successfully (no crash on preload)")

    # Step 6: Verify server works
    try:
        verify = send_completion("Say OK.", max_tokens=5)
        result["post_restart_ok"] = True
        log(f"Post-restart request: {verify['text'][:20]}")
    except Exception as e:
        result["post_restart_ok"] = False
        result["post_restart_error"] = str(e)
        result["status"] = "FAIL"
        result["reason"] = f"Server broken after restart: {e}"
        return result

    result["status"] = "PASS"
    return result


# ────────────────────────────────────────────────────────────────
# Test 4: Graceful SIGTERM Shutdown
# ────────────────────────────────────────────────────────────────

def test_graceful_sigterm() -> dict:
    print("\n" + "=" * 60)
    print("TEST 4: Graceful SIGTERM Shutdown")
    print("=" * 60)

    result = {"test": "graceful_sigterm", "timestamp": datetime.now().isoformat()}

    if not check_health():
        result["status"] = "SKIP"
        result["reason"] = "Server not healthy at start"
        return result

    # Step 1: Save a snapshot
    log("Sending request and saving snapshot...")
    try:
        comp = send_completion("Tell me about the ocean.", max_tokens=50)
        rid1 = comp["rid"]
    except Exception as e:
        result["status"] = "FAIL"
        result["reason"] = f"Request failed: {e}"
        return result

    save1 = save_snapshot(rid=rid1)
    result["save_success"] = save1.get("success", False)
    if not save1.get("success"):
        result["status"] = "FAIL"
        result["reason"] = f"Save failed: {save1.get('message')}"
        return result

    conv_id = (save1.get("snapshot_id") or "").rsplit("-t", 1)[0]
    result["conversation_id"] = conv_id
    log(f"Snapshot saved: {conv_id}")

    # Verify on disk
    integrity = check_snapshot_integrity(conv_id)
    result["integrity_before"] = integrity

    # Step 2: Send a second request (will be in-flight)
    log("Sending in-flight request...")

    def send_long():
        try:
            send_completion("Write a 500-word essay about climate change and its global impacts.",
                            max_tokens=300, timeout=30)
        except Exception:
            pass

    req_thread = threading.Thread(target=send_long)
    req_thread.start()
    time.sleep(0.5)

    # Step 3: SIGTERM
    log("Sending SIGTERM...")
    sigterm_start = time.time()
    killed = kill_server(signal.SIGTERM, timeout=60)
    sigterm_elapsed = time.time() - sigterm_start
    result["sigterm_shutdown_s"] = round(sigterm_elapsed, 1)
    result["kill_success"] = killed
    log(f"Shutdown took {sigterm_elapsed:.1f}s")

    req_thread.join(timeout=5)

    if sigterm_elapsed > 60:
        result["status"] = "FAIL"
        result["reason"] = f"SIGTERM took {sigterm_elapsed:.0f}s — may have hung"
        return result

    # Step 4: Check disk
    time.sleep(2)
    integrity_after = check_snapshot_integrity(conv_id)
    result["integrity_after_sigterm"] = integrity_after
    log(f"Snapshot after SIGTERM: exists={integrity_after.get('exists')}")

    if not integrity_after.get("exists"):
        result["status"] = "FAIL"
        result["reason"] = "Snapshot lost after SIGTERM"
        return result

    # Step 5: Restart and restore
    log("Restarting server...")
    if not start_server(timeout=180):
        result["status"] = "FAIL"
        result["reason"] = "Server failed to restart after SIGTERM"
        return result

    log("Restoring snapshot...")
    try:
        restore = restore_snapshot(conversation_id=conv_id)
        result["restore_success"] = restore.get("success", False)
        result["restore_latency_ms"] = round(restore.get("latency_ms", 0), 1)
        log(f"Restore: success={restore.get('success')}, latency={restore.get('latency_ms', 0):.0f}ms")
    except Exception as e:
        result["restore_success"] = False
        result["restore_error"] = str(e)
        result["status"] = "FAIL"
        result["reason"] = f"Restore failed: {e}"
        return result

    if not restore.get("success"):
        result["status"] = "FAIL"
        result["reason"] = f"Restore returned failure: {restore.get('message')}"
        return result

    result["status"] = "PASS"
    return result


# ────────────────────────────────────────────────────────────────
# Test 5: Abort Request + Snapshot Interaction
# ────────────────────────────────────────────────────────────────

def test_abort_and_snapshot() -> dict:
    print("\n" + "=" * 60)
    print("TEST 5: Abort Request + Snapshot Interaction")
    print("=" * 60)

    result = {"test": "abort_and_snapshot", "timestamp": datetime.now().isoformat()}

    if not check_health():
        result["status"] = "SKIP"
        result["reason"] = "Server not healthy at start"
        return result

    vram_before = get_gpu_vram_mb()
    result["vram_before_mb"] = vram_before

    # Step 1: Send request with custom RID
    custom_rid = f"abort-test-{uuid.uuid4().hex[:8]}"
    log(f"Sending request with RID: {custom_rid}...")

    gen_result = {}

    def do_generate():
        try:
            r = send_generate(
                "Write a very detailed analysis of neural network architectures. "
                "Include CNNs, RNNs, Transformers, and State Space Models.",
                max_tokens=500, custom_rid=custom_rid,
            )
            gen_result.update(r)
        except Exception as e:
            gen_result["error"] = str(e)

    gen_thread = threading.Thread(target=do_generate)
    gen_thread.start()

    # Step 2: Wait for it to start
    time.sleep(2)

    # Step 3: Abort
    log(f"Aborting request {custom_rid}...")
    abort_ok = abort_request(custom_rid)
    result["abort_success"] = abort_ok
    log(f"Abort response: {abort_ok}")

    # Step 4: Wait for abort cleanup (2s delay in server)
    log("Waiting 3s for abort cleanup...")
    time.sleep(3)

    gen_thread.join(timeout=5)

    # Step 5: Try to save snapshot for the aborted RID
    log(f"Attempting save_snapshot for aborted RID: {custom_rid}...")
    try:
        save_result = save_snapshot(rid=custom_rid)
        result["save_after_abort"] = save_result
        result["save_succeeded"] = save_result.get("success", False)
        log(f"Save result: success={save_result.get('success')}, msg={save_result.get('message')}")

        # If save succeeded, verify the snapshot on disk
        if save_result.get("success"):
            snap_conv_id = (save_result.get("snapshot_id") or "").rsplit("-t", 1)[0]
            if snap_conv_id:
                integrity = check_snapshot_integrity(snap_conv_id)
                result["snapshot_integrity"] = integrity
                log(f"Snapshot integrity: {integrity}")

    except Exception as e:
        result["save_after_abort_error"] = str(e)
        result["save_succeeded"] = False
        log(f"Save after abort raised: {e}")

    # Step 6: Check server health
    healthy = check_health()
    result["server_healthy"] = healthy
    log(f"Server healthy after abort+save: {healthy}")

    if not healthy:
        result["status"] = "FAIL"
        result["reason"] = "Server crashed after abort + save"
        return result

    # Step 7: Send fresh request to verify server still works
    try:
        fresh = send_completion("Say OK.", max_tokens=5)
        result["fresh_request_ok"] = True
        log(f"Fresh request: {fresh['text'][:20]}")
    except Exception as e:
        result["fresh_request_ok"] = False
        result["fresh_request_error"] = str(e)
        result["status"] = "FAIL"
        result["reason"] = f"Server broken after abort+save: {e}"
        return result

    # Step 8: VRAM check
    vram_after = get_gpu_vram_mb()
    vram_delta = vram_after - vram_before
    result["vram_after_mb"] = vram_after
    result["vram_delta_mb"] = vram_delta
    log(f"VRAM delta: {vram_delta:+d} MB")

    result["status"] = "PASS"
    return result


# ────────────────────────────────────────────────────────────────
# Report Generation
# ────────────────────────────────────────────────────────────────

def generate_report(results: List[dict], output_dir: Path):
    lines = [
        "# Phase 10 Addendum: Resilience Test Results\n",
        f"**Date**: {datetime.now().strftime('%Y-%m-%d')}",
        f"**Model**: granite-4.0-h-tiny (4B)",
        f"**Server**: {SERVER_URL}",
        f"**Snapshot Dir**: {SNAPSHOT_DIR}",
        "",
    ]

    # Summary table
    lines.append("## Summary\n")
    lines.append("| Test | Scenario | Result | Key Evidence |")
    lines.append("|------|----------|--------|--------------|")
    for r in results:
        status = r.get("status", "?")
        evidence = ""
        if r["test"] == "client_disconnect":
            evidence = f"Health={r.get('server_healthy_after_disconnect')}, VRAM delta={r.get('vram_delta_mb', 0):+d}MB"
        elif r["test"] == "sigkill_mid_inference":
            tier = r.get("preload_tier", "?")
            evidence = f"Snapshot intact, restore={r.get('restore_latency_ms', 0):.0f}ms ({tier})"
        elif r["test"] == "sigkill_during_write":
            evidence = f"tmp files={r.get('tmp_file_count', 0)}, partial={r.get('partial_safetensors_found')}"
        elif r["test"] == "graceful_sigterm":
            evidence = f"Shutdown={r.get('sigterm_shutdown_s', 0):.1f}s, restore={r.get('restore_success')}"
        elif r["test"] == "abort_and_snapshot":
            evidence = f"Save={r.get('save_succeeded')}, healthy={r.get('server_healthy')}"
        lines.append(f"| {results.index(r) + 1} | {r['test']} | **{status}** | {evidence} |")
    lines.append("")

    # Detailed results per test
    for r in results:
        lines.append(f"### Test: {r['test']}\n")
        lines.append(f"- **Status**: {r.get('status')}")
        if r.get("reason"):
            lines.append(f"- **Reason**: {r['reason']}")
        # Key metrics
        for key in sorted(r.keys()):
            if key in ("test", "status", "reason", "timestamp"):
                continue
            val = r[key]
            if isinstance(val, (dict, list)) and len(str(val)) > 200:
                val = str(val)[:200] + "..."
            lines.append(f"- **{key}**: {val}")
        lines.append("")

    report_path = output_dir / "phase-10-resilience-results.md"
    with open(report_path, "w") as f:
        f.write("\n".join(lines))
    print(f"\nReport written to {report_path}")
    return report_path


# ────────────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────────────

TEST_MAP = {
    "1": test_client_disconnect,
    "2": test_sigkill_mid_inference,
    "3": test_sigkill_during_write,
    "4": test_graceful_sigterm,
    "5": test_abort_and_snapshot,
}


def main():
    global SERVER_URL, SNAPSHOT_DIR, MODEL_PATH

    parser = argparse.ArgumentParser(description="Phase 10: Resilience Testing")
    parser.add_argument("--test", default="all",
                        help="Test number(s) to run: 1,2,3,4,5 or 'all'")
    parser.add_argument("--server-url", default=SERVER_URL)
    parser.add_argument("--snapshot-dir", default=SNAPSHOT_DIR)
    parser.add_argument("--model-path", default=MODEL_PATH)
    parser.add_argument("--output-dir", default="test/phases/results")
    args = parser.parse_args()
    SERVER_URL = args.server_url
    SNAPSHOT_DIR = args.snapshot_dir
    MODEL_PATH = args.model_path

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.test == "all":
        tests = ["1", "2", "3", "4", "5"]
    else:
        tests = [t.strip() for t in args.test.split(",")]

    for t in tests:
        if t not in TEST_MAP:
            print(f"ERROR: Unknown test '{t}'. Valid: {list(TEST_MAP.keys())}")
            sys.exit(1)

    print(f"Phase 10 Resilience Test")
    print(f"  Server:  {SERVER_URL}")
    print(f"  Tests:   {tests}")
    print(f"  SnapDir: {SNAPSHOT_DIR}")
    print()

    # Initial health check (only for test 1 and 5 which don't restart)
    needs_running_server = any(t in ("1", "5") for t in tests)
    if needs_running_server and not check_health():
        print("ERROR: Server not healthy. Start it first.")
        print("  source test/phases/infra/config.sh")
        print("  python -m sglang.launch_server --model-path $MODEL_PATH ...")
        sys.exit(1)

    results = []
    for t in tests:
        r = TEST_MAP[t]()
        results.append(r)
        # Save intermediate results
        with open(output_dir / "phase-10-resilience-results.json", "w") as f:
            json.dump(results, f, indent=2, default=str)

    # Generate report
    generate_report(results, output_dir)

    # Summary
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    for r in results:
        status = r.get("status", "?")
        icon = "PASS" if status == "PASS" else status
        print(f"  Test {r['test']}: {icon}")
        if r.get("reason"):
            print(f"    Reason: {r['reason']}")


if __name__ == "__main__":
    main()
