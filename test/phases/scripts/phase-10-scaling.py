#!/usr/bin/env python3
"""
Phase 10: Scaling & Resource Monitoring

Tests memory behavior, context window scaling, and resource utilization
for Mamba snapshot persistence with built-in monitoring.
"""

import argparse
import asyncio
import json
import os
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import requests

SERVER_URL = os.environ.get("SERVER_URL", "http://localhost:30000")
SNAPSHOT_DIR = os.environ.get("SNAPSHOT_DIR", "/tmp/mamba_snapshots")
LOG_DIR = Path("test/phases/results/phase-10-logs")
LOG_DIR.mkdir(parents=True, exist_ok=True)


class ResourceMonitor:
    """Continuously monitor GPU, RAM, and process resources."""

    def __init__(self, log_file: str, poll_interval: float = 5.0):
        self.log_file = log_file
        self.poll_interval = poll_interval
        self.running = False
        self.metrics: List[Dict] = []

    async def start(self, duration_seconds: Optional[int] = None):
        """Start monitoring. If duration is None, runs until stop() is called."""
        self.running = True
        start_time = time.time()

        # Write CSV header
        with open(self.log_file, "w") as f:
            f.write(
                "timestamp,elapsed_s,gpu_vram_mb,gpu_util_pct,ram_available_mb,ram_mb,proc_rss_mb,fd_count,snapshots_count\n"
            )

        print(f"[Monitor] Started logging to {self.log_file}")

        while self.running:
            elapsed = time.time() - start_time
            timestamp = datetime.now().isoformat()

            # Collect all metrics
            metrics = {
                "timestamp": timestamp,
                "elapsed_s": round(elapsed, 1),
                "gpu_vram_mb": self._get_gpu_vram(),
                "gpu_util_pct": self._get_gpu_util(),
                "ram_available_mb": self._get_ram_available(),
                "ram_mb": self._get_ram_total(),
                "proc_rss_mb": self._get_proc_rss(),
                "fd_count": self._get_fd_count(),
                "snapshots_count": self._get_snapshots_count(),
            }

            self.metrics.append(metrics)

            # Write to CSV
            with open(self.log_file, "a") as f:
                f.write(
                    f"{timestamp},{metrics['elapsed_s']},"
                    f"{metrics['gpu_vram_mb']},{metrics['gpu_util_pct']},"
                    f"{metrics['ram_available_mb']},{metrics['ram_mb']},"
                    f"{metrics['proc_rss_mb']},{metrics['fd_count']},"
                    f"{metrics['snapshots_count']}\n"
                )

            # Print summary
            print(
                f"[{metrics['elapsed_s']:6.1f}s] "
                f"GPU: {metrics['gpu_vram_mb']:5}MB {metrics['gpu_util_pct']:3}% | "
                f"RAM: {metrics['ram_available_mb']:5}MB avail / {metrics['ram_mb']:5}MB total | "
                f"Proc: {metrics['proc_rss_mb']:5}MB | "
                f"FDs: {metrics['fd_count']:4} | "
                f"Snaps: {metrics['snapshots_count']:3}"
            )

            # Check duration limit
            if duration_seconds and elapsed >= duration_seconds:
                break

            await asyncio.sleep(self.poll_interval)

    def stop(self):
        """Stop monitoring."""
        self.running = False
        print(f"[Monitor] Stopped. Collected {len(self.metrics)} data points.")

    def _get_gpu_vram(self) -> int:
        """Get GPU VRAM usage in MB."""
        try:
            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=memory.used",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                timeout=2,
            )
            if result.returncode == 0:
                return int(result.stdout.strip())
        except Exception:
            pass
        return 0

    def _get_gpu_util(self) -> int:
        """Get GPU utilization percentage."""
        try:
            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=utilization.gpu",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                timeout=2,
            )
            if result.returncode == 0:
                return int(result.stdout.strip())
        except Exception:
            pass
        return 0

    def _get_ram_available(self) -> int:
        """Get available RAM in MB."""
        try:
            result = subprocess.run(
                ["free", "-m"], capture_output=True, text=True, timeout=2
            )
            if result.returncode == 0:
                return int(result.stdout.split("\n")[1].split()[6])
        except Exception:
            pass
        return 0

    def _get_ram_total(self) -> int:
        """Get total RAM in MB."""
        try:
            result = subprocess.run(
                ["free", "-m"], capture_output=True, text=True, timeout=2
            )
            if result.returncode == 0:
                return int(result.stdout.split("\n")[1].split()[1])
        except Exception:
            pass
        return 0

    def _get_proc_rss(self) -> int:
        """Get sglang process RSS in MB."""
        try:
            # Sum up all sglang-related processes
            result = subprocess.run(
                ["ps", "aux"], capture_output=True, text=True, timeout=2
            )
            if result.returncode == 0:
                total_rss_kb = 0
                for line in result.stdout.split("\n"):
                    # Look for sglang::scheduler, sglang::detokenizer, etc.
                    if "sglang::" in line or "launch_server" in line:
                        parts = line.split()
                        if len(parts) >= 6:
                            try:
                                total_rss_kb += int(parts[5])
                            except ValueError:
                                continue
                return total_rss_kb // 1024  # Convert to MB
        except Exception:
            pass
        return 0

    def _get_fd_count(self) -> int:
        """Get file descriptor count for sglang processes."""
        try:
            result = subprocess.run(
                ["lsof", "-a", "-p", "$(pgrep -f sglang | head -1)"],
                shell=True,
                capture_output=True,
                text=True,
                timeout=2,
            )
            if result.returncode == 0:
                return len([l for l in result.stdout.split("\n") if l.strip()])
        except Exception:
            pass
        return 0

    def _get_snapshots_count(self) -> int:
        """Count snapshot files in SNAPSHOT_DIR."""
        try:
            if os.path.exists(SNAPSHOT_DIR):
                # Count both .bin and .safetensors files
                return sum(1 for _ in Path(SNAPSHOT_DIR).rglob("*.bin")) + sum(
                    1 for _ in Path(SNAPSHOT_DIR).rglob("*.safetensors")
                )
        except Exception:
            pass
        return 0

    def get_summary(self) -> Dict:
        """Get summary statistics."""
        if not self.metrics:
            return {}

        gpu_vrams = [m["gpu_vram_mb"] for m in self.metrics if m["gpu_vram_mb"] > 0]
        proc_rsss = [m["proc_rss_mb"] for m in self.metrics if m["proc_rss_mb"] > 0]

        return {
            "duration_s": self.metrics[-1]["elapsed_s"],
            "samples": len(self.metrics),
            "gpu_vram_start_mb": gpu_vrams[0] if gpu_vrams else 0,
            "gpu_vram_end_mb": gpu_vrams[-1] if gpu_vrams else 0,
            "gpu_vram_delta_mb": (
                (gpu_vrams[-1] - gpu_vrams[0]) if len(gpu_vrams) > 1 else 0
            ),
            "gpu_vram_max_mb": max(gpu_vrams) if gpu_vrams else 0,
            "proc_rss_start_mb": proc_rsss[0] if proc_rsss else 0,
            "proc_rss_end_mb": proc_rsss[-1] if proc_rsss else 0,
            "proc_rss_delta_mb": (
                (proc_rsss[-1] - proc_rsss[0]) if len(proc_rsss) > 1 else 0
            ),
            "proc_rss_max_mb": max(proc_rsss) if proc_rsss else 0,
            "snapshots_end": self.metrics[-1]["snapshots_count"],
        }


class LoadTestRunner:
    """Run various load test scenarios."""

    def __init__(self, server_url: str = SERVER_URL):
        self.server_url = server_url

    def check_server(self) -> bool:
        """Check if server is running."""
        try:
            # Try v1/models endpoint (SGLang OpenAI-compatible API)
            r = requests.get(f"{self.server_url}/v1/models", timeout=5)
            return r.status_code == 200
        except Exception as e:
            print(f"[Health Check] Failed: {e}")
            return False

    def _chat_completion(self, messages: List[Dict], max_tokens: int = 50) -> Dict:
        """Send chat completion request."""
        payload = {
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": 0.0,
        }
        r = requests.post(
            f"{self.server_url}/v1/chat/completions", json=payload, timeout=120
        )
        r.raise_for_status()
        return r.json()

    async def scenario_small_context(self, turns: int = 10):
        """Scenario: Small context conversations (baseline)."""
        print(f"\n[Scenario] Small Context - {turns} turns")

        messages = [{"role": "system", "content": "You are a helpful assistant."}]

        for i in range(turns):
            messages.append(
                {"role": "user", "content": f"Turn {i+1}: Say something brief."}
            )
            try:
                response = self._chat_completion(messages, max_tokens=30)
                content = response["choices"][0]["message"]["content"]
                messages.append({"role": "assistant", "content": content})
                print(f"  Turn {i+1}: OK")
            except Exception as e:
                print(f"  Turn {i+1}: FAILED - {e}")
                break

            await asyncio.sleep(1)

    async def scenario_medium_context(self, turns: int = 5):
        """Scenario: Medium context (~500 tokens)."""
        print(f"\n[Scenario] Medium Context - {turns} turns with ~500 token context")

        # Build up a medium context
        long_context = "This is a test. " * 100  # ~500 tokens
        messages = [
            {
                "role": "system",
                "content": f"You are a helpful assistant. Context: {long_context}",
            }
        ]

        for i in range(turns):
            messages.append(
                {"role": "user", "content": f"Turn {i+1}: What was that context about?"}
            )
            try:
                response = self._chat_completion(messages, max_tokens=50)
                content = response["choices"][0]["message"]["content"]
                messages.append({"role": "assistant", "content": content})
                print(f"  Turn {i+1}: OK")
            except Exception as e:
                print(f"  Turn {i+1}: FAILED - {e}")
                break

            await asyncio.sleep(2)

    async def scenario_multi_turn_with_snapshots(self, turns: int = 8):
        """Scenario: Multi-turn with explicit save/restore."""
        print(f"\n[Scenario] Multi-turn with Snapshots - {turns} turns")

        # Start conversation
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {
                "role": "user",
                "content": "My favorite color is blue and my lucky number is 7. Remember this.",
            },
        ]

        try:
            # Turn 1: Establish context
            response = self._chat_completion(messages, max_tokens=30)
            rid = response.get("id", "unknown")
            messages.append(
                {
                    "role": "assistant",
                    "content": response["choices"][0]["message"]["content"],
                }
            )

            # Save snapshot
            save_result = requests.post(
                f"{self.server_url}/save_snapshot", json={"rid": rid}, timeout=30
            )
            if save_result.json().get("success"):
                print(f"  Snapshot saved: {save_result.json().get('snapshot_id')}")
            else:
                print(f"  Snapshot save failed: {save_result.json()}")
                return

            # Continue with stateful turns
            for i in range(1, turns):
                messages.append(
                    {
                        "role": "user",
                        "content": f"Turn {i+1}: What's my favorite color?",
                    }
                )
                response = self._chat_completion(messages, max_tokens=30)
                content = response["choices"][0]["message"]["content"]
                messages.append({"role": "assistant", "content": content})
                print(f"  Turn {i+1}: OK - {content[:50]}...")

                # Save snapshot after each turn
                save_result = requests.post(
                    f"{self.server_url}/save_snapshot", json={"rid": rid}, timeout=30
                )
                if not save_result.json().get("success"):
                    print(f"  Snapshot save failed: {save_result.json()}")

                await asyncio.sleep(1)

        except Exception as e:
            print(f"  Scenario failed: {e}")

    async def scenario_continuous_load(self, duration_seconds: int = 60):
        """Scenario: Continuous load for specified duration."""
        print(f"\n[Scenario] Continuous Load - {duration_seconds}s")

        start_time = time.time()
        requests_sent = 0
        errors = 0

        while (time.time() - start_time) < duration_seconds:
            try:
                messages = [
                    {
                        "role": "user",
                        "content": f"Request {requests_sent + 1}: Say hello.",
                    }
                ]
                self._chat_completion(messages, max_tokens=10)
                requests_sent += 1
            except Exception as e:
                errors += 1

            await asyncio.sleep(0.5)

        print(f"  Completed: {requests_sent} requests, {errors} errors")


async def main():
    parser = argparse.ArgumentParser(description="Phase 10 Scaling Tests")
    parser.add_argument(
        "--scenario",
        choices=["baseline", "small", "medium", "snapshots", "continuous", "all"],
        default="baseline",
        help="Which scenario to run",
    )
    parser.add_argument(
        "--duration",
        type=int,
        default=60,
        help="Duration for baseline/continuous (seconds)",
    )
    parser.add_argument(
        "--no-monitor", action="store_true", help="Skip resource monitoring"
    )
    args = parser.parse_args()

    # Check server
    runner = LoadTestRunner()
    if not runner.check_server():
        print(f"ERROR: Server not responding at {SERVER_URL}")
        print("Start the server first:")
        print(
            "  python -m sglang.launch_server --model-path $MODEL_PATH --enable-snapshot-persistence"
        )
        return

    # Set up monitoring
    monitor = None
    if not args.no_monitor:
        log_file = (
            LOG_DIR
            / f"metrics_{args.scenario}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        )
        monitor = ResourceMonitor(str(log_file))

    # Run scenarios
    print("=" * 60)
    print("Phase 10: Scaling & Resource Monitoring")
    print("=" * 60)
    print(f"Server: {SERVER_URL}")
    print(f"Snapshot Dir: {SNAPSHOT_DIR}")
    print(f"Log Dir: {LOG_DIR}")
    print("=" * 60)

    try:
        # Start monitor in background
        if monitor:
            monitor_task = asyncio.create_task(
                monitor.start(duration_seconds=args.duration * 2)
            )
            await asyncio.sleep(2)  # Let monitor establish baseline

        # Run requested scenario(s)
        if args.scenario in ["baseline", "all"]:
            print("\n>>> BASELINE: Idle monitoring")
            if monitor:
                await asyncio.sleep(args.duration)

        if args.scenario in ["small", "all"]:
            await runner.scenario_small_context(turns=10)

        if args.scenario in ["medium", "all"]:
            await runner.scenario_medium_context(turns=5)

        if args.scenario in ["snapshots", "all"]:
            await runner.scenario_multi_turn_with_snapshots(turns=8)

        if args.scenario in ["continuous", "all"]:
            await runner.scenario_continuous_load(duration_seconds=args.duration)

    finally:
        # Stop monitor and print summary
        if monitor:
            monitor.stop()
            await asyncio.sleep(1)

            summary = monitor.get_summary()
            print("\n" + "=" * 60)
            print("RESOURCE MONITORING SUMMARY")
            print("=" * 60)
            print(f"Duration: {summary.get('duration_s', 0):.1f} seconds")
            print(f"Samples: {summary.get('samples', 0)}")
            print(f"\nGPU VRAM:")
            print(f"  Start: {summary.get('gpu_vram_start_mb', 0)} MB")
            print(f"  End:   {summary.get('gpu_vram_end_mb', 0)} MB")
            print(f"  Delta: {summary.get('gpu_vram_delta_mb', 0):+d} MB")
            print(f"  Max:   {summary.get('gpu_vram_max_mb', 0)} MB")
            print(f"\nProcess RSS:")
            print(f"  Start: {summary.get('proc_rss_start_mb', 0)} MB")
            print(f"  End:   {summary.get('proc_rss_end_mb', 0)} MB")
            print(f"  Delta: {summary.get('proc_rss_delta_mb', 0):+d} MB")
            print(f"  Max:   {summary.get('proc_rss_max_mb', 0)} MB")
            print(f"\nSnapshots on disk: {summary.get('snapshots_end', 0)}")

            # Alert on potential leaks
            if abs(summary.get("gpu_vram_delta_mb", 0)) > 500:
                print("\n⚠️  WARNING: GPU VRAM delta > 500MB (possible leak)")
            if abs(summary.get("proc_rss_delta_mb", 0)) > 1000:
                print("\n⚠️  WARNING: Process RSS delta > 1GB (possible leak)")

            # Save summary JSON
            summary_file = (
                LOG_DIR
                / f"summary_{args.scenario}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            )
            with open(summary_file, "w") as f:
                json.dump(summary, f, indent=2)
            print(f"\nSummary saved to: {summary_file}")


if __name__ == "__main__":
    asyncio.run(main())
