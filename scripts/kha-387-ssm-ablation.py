#!/usr/bin/env python3
# ENGRAM_DIAGNOSTIC — KHA-387 SSM ablation
#
# Distinguishes whether restored SSM state is actually being consumed by the
# pending-restore + next /generate request path, by comparing identical
# continuation requests under five SSM conditions:
#
#   A. no_restore       — no /restore_snapshot before the continuation
#   B. real_warm        — /restore_snapshot, WARM tier (in-memory, real state)
#   C. real_disk        — /restore_snapshot with turn_number set, forcing
#                         disk reload of the untampered on-disk safetensors
#   D. zero_disk        — same as C, but on-disk tensors zeroed before restore
#   E. random_disk      — same as C, but on-disk tensors replaced with Gaussian
#
# Decision rules:
#   - A ≡ B ≡ C ≡ D ≡ E   → SSM state is not consumed by this path at all.
#                            Bug is in request lifecycle, not in KV gap.
#   - B ≠ A, but C ≡ D ≡ E → WARM-tier injection differs from disk path
#                            (in-memory hydration is working, disk path is not).
#   - C ≠ D ≠ E and all ≠ A → SSM is consumed; tampered tensors produce
#                              distinguishable behavior; this would refute case 1.
#
# Premise: the WARM in-memory tier serves /restore_snapshot when no
# turn_number is given (scheduler.py:1453-1466). Setting turn_number forces
# snapshot_manager.load_snapshot() to reload from the safetensors file on
# disk, which we may have mutated between save and restore.
#
# Validation note: validate_state_tensors treats all-zero as a WARNING
# (not an error) at non-strict load and inject, so case D loads cleanly.
# Random Gaussian with stats close to the natural distribution avoids
# NaN/Inf rejections in case E.

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests
import torch
from safetensors import safe_open
from safetensors.torch import save_file


# --- Configuration defaults ---

DEFAULT_MODEL = os.environ.get(
    "KHA387_MODEL", "/workspace/models/mamba-codestral-7b"
)
DEFAULT_PORT = int(os.environ.get("KHA387_PORT", "30200"))
DEFAULT_SERVED_NAME = os.environ.get("KHA387_SERVED_NAME", "mamba-codestral-7b")

# Two-segment Codestral prompt: a "seed" that primes a recognizable code
# structure, and a "tail" that without the SSM state would produce
# unrelated content. Modeled after KHA-385 v2's python_constant case
# which already showed restore-and-generate recall of a sentinel.
SEED_PROMPT = (
    "import unittest\n"
    "# regression marker: ZEPHYR_KHA387:271828\n"
    "# this constant is the canonical sentinel used by all KHA-387 tests below\n"
    "# stored_pair returns the sentinel as a tuple of (1, 2)\n"
    "def stored_pair():\n"
    "    # canonical sentinel string defined above is\n"
)
# Continuation prompt — only the immediate tail. Without state, the model
# should fall back to its prior on continuation. With consumed state, it
# should recall the sentinel substring "ZEPHYR_KHA387:271828".
TAIL_PROMPT = "    return "

SENTINEL = "ZEPHYR_KHA387:271828"

MAX_TOKENS = 96
SAMPLING = {
    "temperature": 0.0,
    "top_p": 1.0,
    "top_k": -1,
    "frequency_penalty": 0.0,
    "presence_penalty": 0.0,
    "n": 1,
    "max_new_tokens": MAX_TOKENS,
}


# --- HTTP helpers ---


def http_post(url: str, payload: dict, timeout: float = 120.0) -> Tuple[int, dict]:
    r = requests.post(url, json=payload, timeout=timeout)
    try:
        body = r.json()
    except Exception:
        body = {"_raw": r.text}
    return r.status_code, body


def wait_for_health(server_url: str, max_wait_s: int = 600) -> None:
    deadline = time.time() + max_wait_s
    while time.time() < deadline:
        try:
            r = requests.get(f"{server_url}/health", timeout=5)
            if r.status_code == 200:
                return
        except requests.RequestException:
            pass
        time.sleep(5)
    raise RuntimeError(f"Server at {server_url} did not become healthy in {max_wait_s}s")


# --- Snapshot tamper ---


def read_snapshot_tensors(state_path: Path) -> Dict[str, torch.Tensor]:
    out: Dict[str, torch.Tensor] = {}
    with safe_open(state_path, framework="pt") as f:
        for k in f.keys():
            out[k] = f.get_tensor(k)
    return out


def write_snapshot_tensors(state_path: Path, tensors: Dict[str, torch.Tensor]) -> None:
    # save_file writes atomically by overwriting; matching the format the
    # snapshot manager wrote in the first place.
    tmp = state_path.with_suffix(state_path.suffix + ".kha387tmp")
    save_file(tensors, str(tmp))
    os.replace(tmp, state_path)


def zero_tensors(orig: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    return {k: torch.zeros_like(v) for k, v in orig.items()}


def randomize_tensors(orig: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    # Gaussian with per-tensor stats matched to the original. The natural
    # state distribution is roughly zero-mean; scale by observed std so we
    # don't trip the load-time NaN/Inf guards. validate_state_tensors only
    # flags exact all-zero and NaN/Inf; this passes both.
    out: Dict[str, torch.Tensor] = {}
    for k, v in orig.items():
        std = max(float(v.float().std().item()), 1e-3)
        mean = float(v.float().mean().item())
        # Cap absolute values to 6 sigma to avoid any infinity in bf16.
        sample = torch.randn_like(v.float()) * std + mean
        sample = sample.clamp(min=mean - 6 * std, max=mean + 6 * std)
        out[k] = sample.to(v.dtype)
    return out


# --- Snapshot directory + ID conventions ---


def conv_dir(snap_root: Path, conv_id: str) -> Path:
    return snap_root / f"conversation_{conv_id}"


def turn_state_path(snap_root: Path, conv_id: str, turn: int) -> Path:
    return conv_dir(snap_root, conv_id) / f"turn_{turn}_state.safetensors"


# --- Single case execution ---


def run_case(
    *,
    case: str,
    server_url: str,
    served_name: str,
    snap_root: Path,
    log: List[str],
) -> Dict[str, Any]:
    """Execute one ablation condition end-to-end and return a result dict.

    Each case uses its own conversation_id so tampering one disk file
    does not poison another case's WARM tier.
    """
    conv_id = f"kha387-{case}-{uuid.uuid4().hex[:8]}"
    rid_seed = f"{conv_id}-seed"
    rid_save = f"{conv_id}-save"  # not actually used distinctly; documenting intent
    rid_tail = f"{conv_id}-tail"

    log.append(f"\n==== CASE {case} (conv_id={conv_id}) ====")

    # --- Step 1: seed /generate (establishes state on the seed rid) ---
    seed_payload = {
        "text": SEED_PROMPT,
        "sampling_params": SAMPLING,
        "rid": rid_seed,
        "conversation_id": conv_id,
        "model": served_name,
    }
    code, body = http_post(f"{server_url}/generate", seed_payload)
    seed_text = body.get("text", "")
    log.append(f"seed /generate HTTP {code}; text_head={seed_text[:80]!r}")

    # --- Step 2: /save_snapshot (promote WARM → COLD/disk) ---
    save_payload = {
        "rid": rid_seed,
        "conversation_id": conv_id,
        "snapshot_id": f"{conv_id}-t1",
        "turn_number": 1,
    }
    code, body = http_post(f"{server_url}/save_snapshot", save_payload)
    log.append(f"/save_snapshot HTTP {code} body={json.dumps(body)[:240]}")
    if not body.get("success"):
        return {
            "case": case,
            "conv_id": conv_id,
            "save_ok": False,
            "tail_text": None,
            "tail_output_ids": [],
            "recall_sentinel_present": False,
            "error": "save_snapshot did not return success=true",
        }

    state_path = turn_state_path(snap_root, conv_id, 1)
    if case in ("zero_disk", "random_disk", "real_disk") and not state_path.exists():
        return {
            "case": case,
            "conv_id": conv_id,
            "save_ok": True,
            "tail_text": None,
            "tail_output_ids": [],
            "recall_sentinel_present": False,
            "error": f"expected state file {state_path} after save, not found",
        }

    # --- Step 3: optional disk tamper (cases zero_disk / random_disk only) ---
    pre_stats = None
    post_stats = None
    if case in ("zero_disk", "random_disk"):
        orig = read_snapshot_tensors(state_path)
        pre_stats = {
            k: {
                "shape": tuple(v.shape),
                "dtype": str(v.dtype),
                "mean": float(v.float().mean().item()),
                "std": float(v.float().std().item()),
                "abs_sum": float(v.float().abs().sum().item()),
            }
            for k, v in orig.items()
        }
        mutated = zero_tensors(orig) if case == "zero_disk" else randomize_tensors(orig)
        post_stats = {
            k: {
                "mean": float(v.float().mean().item()),
                "std": float(v.float().std().item()),
                "abs_sum": float(v.float().abs().sum().item()),
            }
            for k, v in mutated.items()
        }
        write_snapshot_tensors(state_path, mutated)
        log.append(
            f"tampered {state_path}: pre conv_abs_sum={pre_stats['conv_layer_0']['abs_sum']:.3e} "
            f"temporal_abs_sum={pre_stats['temporal']['abs_sum']:.3e} → "
            f"post conv_abs_sum={post_stats['conv_layer_0']['abs_sum']:.3e} "
            f"temporal_abs_sum={post_stats['temporal']['abs_sum']:.3e}"
        )

    # --- Step 4: /restore_snapshot (skipped only for no_restore) ---
    if case != "no_restore":
        restore_payload: Dict[str, Any] = {
            "rid": rid_tail,
            "conversation_id": conv_id,
        }
        if case in ("real_disk", "zero_disk", "random_disk"):
            # Setting turn_number forces snapshot_manager.load_snapshot()
            # disk path, bypassing WARM tier in-memory cache.
            restore_payload["turn_number"] = 1
        code, body = http_post(f"{server_url}/restore_snapshot", restore_payload)
        log.append(f"/restore_snapshot HTTP {code} body={json.dumps(body)[:300]}")
        if not body.get("success"):
            return {
                "case": case,
                "conv_id": conv_id,
                "save_ok": True,
                "tail_text": None,
                "tail_output_ids": [],
                "recall_sentinel_present": False,
                "error": f"restore_snapshot did not return success=true: {body}",
                "tampered_pre": pre_stats,
                "tampered_post": post_stats,
            }

    # --- Step 5: continuation /generate ---
    tail_payload = {
        "text": TAIL_PROMPT,
        "sampling_params": SAMPLING,
        "rid": rid_tail,
        "conversation_id": conv_id,
        "model": served_name,
    }
    code, body = http_post(f"{server_url}/generate", tail_payload)
    tail_text = body.get("text", "") or ""
    tail_ids = (body.get("meta_info") or {}).get("output_ids") or body.get("output_ids") or []
    log.append(
        f"tail /generate HTTP {code} text={tail_text[:160]!r} "
        f"first_ids={tail_ids[:24] if isinstance(tail_ids, list) else 'N/A'}"
    )

    return {
        "case": case,
        "conv_id": conv_id,
        "save_ok": True,
        "tail_text": tail_text,
        "tail_output_ids": tail_ids if isinstance(tail_ids, list) else [],
        "recall_sentinel_present": SENTINEL in tail_text,
        "error": None,
        "tampered_pre": pre_stats,
        "tampered_post": post_stats,
    }


# --- Verdict ---


def summarize(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compare the per-case continuation outputs and label the case it supports."""
    by_case = {r["case"]: r for r in results}
    cases = ["no_restore", "real_warm", "real_disk", "zero_disk", "random_disk"]
    present = {c: by_case.get(c, {}) for c in cases}

    # Token-id prefix comparison is more robust than text equality, since
    # whitespace and detokenization may differ but ids will not.
    def head(r: Dict[str, Any], n: int = 32) -> Tuple[int, ...]:
        ids = r.get("tail_output_ids") or []
        return tuple(ids[:n])

    heads = {c: head(present[c]) for c in cases if present[c]}
    all_same = len({h for h in heads.values()}) == 1 and bool(heads)
    real_warm_eq_no_restore = heads.get("real_warm") == heads.get("no_restore")
    zero_eq_random = heads.get("zero_disk") == heads.get("random_disk")
    real_disk_eq_real_warm = heads.get("real_disk") == heads.get("real_warm")
    distinct_count = len(set(heads.values()))

    # Categorize
    if all_same:
        verdict = (
            "Case 1 — restored SSM not consumed. All five conditions produced "
            "the same continuation prefix. The pending-restore + next-/generate "
            "path is not using the injected recurrent state. Bug is in request "
            "lifecycle, not in attention-KV scope."
        )
    elif distinct_count == 2 and real_warm_eq_no_restore and zero_eq_random:
        verdict = (
            "Mixed — WARM hydration path produces no-restore behavior, but the "
            "disk path produces a different (possibly tampered-state) trajectory. "
            "Investigate per-path injection differences."
        )
    elif heads.get("real_warm") != heads.get("no_restore") and heads.get("real_disk") != heads.get("zero_disk"):
        verdict = (
            "Case 2/4 — SSM is being consumed. Real-state continuation differs "
            "from no-restore, and tampered state produces a third trajectory. "
            "Recall failure (if any) is then about state sufficiency or model "
            "surface, not about whether state reaches the forward pass."
        )
    else:
        verdict = (
            "Inconclusive — case-pattern does not match a clean rule. See per-case "
            "outputs and decide manually."
        )

    return {
        "all_same": all_same,
        "real_warm_eq_no_restore": real_warm_eq_no_restore,
        "real_disk_eq_real_warm": real_disk_eq_real_warm,
        "zero_eq_random": zero_eq_random,
        "distinct_prefix_count": distinct_count,
        "verdict": verdict,
        "heads": {c: list(h) for c, h in heads.items()},
    }


# --- Top-level ---


def main() -> int:
    ap = argparse.ArgumentParser(description="KHA-387 SSM ablation harness")
    ap.add_argument("--server-url", default=f"http://localhost:{DEFAULT_PORT}")
    ap.add_argument("--served-name", default=DEFAULT_SERVED_NAME)
    ap.add_argument("--snapshot-dir", required=True,
                    help="Path the running server was launched with as --snapshot-dir")
    ap.add_argument("--run-dir", required=True,
                    help="Directory for this run's report + per-case logs")
    ap.add_argument("--cases", nargs="+",
                    default=["no_restore", "real_warm", "real_disk", "zero_disk", "random_disk"],
                    help="Subset of cases to run")
    args = ap.parse_args()

    snap_root = Path(args.snapshot_dir).resolve()
    run_dir = Path(args.run_dir).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    if not snap_root.exists():
        print(f"ERROR: snapshot dir does not exist: {snap_root}", file=sys.stderr)
        return 2

    wait_for_health(args.server_url)

    results: List[Dict[str, Any]] = []
    log: List[str] = []
    for case in args.cases:
        try:
            res = run_case(
                case=case,
                server_url=args.server_url,
                served_name=args.served_name,
                snap_root=snap_root,
                log=log,
            )
        except Exception as e:
            res = {
                "case": case,
                "error": f"exception during case: {e}",
                "tail_text": None,
                "tail_output_ids": [],
                "recall_sentinel_present": False,
            }
            log.append(f"CASE {case} raised: {e}")
        results.append(res)

    summary = summarize(results)

    (run_dir / "log.txt").write_text("\n".join(log) + "\n")
    (run_dir / "results.json").write_text(
        json.dumps({"results": results, "summary": summary}, indent=2)
    )

    report_lines: List[str] = []
    report_lines.append(f"# KHA-387 SSM Ablation Report — {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}")
    report_lines.append("")
    report_lines.append(f"Server: {args.server_url}  Model name: {args.served_name}")
    report_lines.append(f"Snapshot dir: {snap_root}")
    report_lines.append(f"Run dir: {run_dir}")
    report_lines.append("")
    report_lines.append("## Per-case continuations")
    report_lines.append("")
    report_lines.append("| Case | save_ok | restore_ok | sentinel? | first 16 output ids | text head |")
    report_lines.append("|---|---|---|---|---|---|")
    for r in results:
        ids_head = (r.get("tail_output_ids") or [])[:16]
        text_head = (r.get("tail_text") or "")[:80].replace("|", "\\|").replace("\n", "\\n")
        sent = "YES" if r.get("recall_sentinel_present") else "no"
        err = r.get("error") or ""
        restore_ok = "n/a" if r["case"] == "no_restore" else (
            "fail" if err and "restore" in err else "ok"
        )
        report_lines.append(
            f"| {r['case']} | {bool(r.get('save_ok'))} | {restore_ok} | {sent} | "
            f"`{ids_head}` | `{text_head}` |"
        )
    report_lines.append("")
    report_lines.append("## Summary")
    report_lines.append("")
    report_lines.append(f"- all_same_prefix: **{summary['all_same']}**")
    report_lines.append(f"- real_warm ≡ no_restore: **{summary['real_warm_eq_no_restore']}**")
    report_lines.append(f"- real_disk ≡ real_warm: **{summary['real_disk_eq_real_warm']}**")
    report_lines.append(f"- zero_disk ≡ random_disk: **{summary['zero_eq_random']}**")
    report_lines.append(f"- distinct prefixes: **{summary['distinct_prefix_count']}**")
    report_lines.append("")
    report_lines.append(f"**Verdict:** {summary['verdict']}")
    report_lines.append("")
    report_lines.append("Server-log evidence for restoration events should be cross-checked from")
    report_lines.append("the server.log written alongside this run; this harness records HTTP-")
    report_lines.append("level evidence only.")

    (run_dir / "report.md").write_text("\n".join(report_lines) + "\n")
    print((run_dir / "report.md").read_text())
    return 0


if __name__ == "__main__":
    sys.exit(main())
