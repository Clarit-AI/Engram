#!/usr/bin/env python3
# ENGRAM_DIAGNOSTIC — KHA-398 KV probe measurement harness
#
# Measures whether SSM+KV warm restore recovers hybrid recall vs SSM-only,
# and at what token cost.
#
# Two steps:
#   Step 1 — Binary planted-fact gate
#     Turn 1: "The secret code is ZEPHYR-7749" → /save_snapshot
#     Turn 2: fresh warm restore + "What is the secret code?"
#     PASS = response contains "ZEPHYR-7749"
#     Also confirms kv_state file exists (~67 MB) on disk.
#
#   Step 2 — Three-arm LoCoMo measurement on conv-26 (~3126 tokens)
#     Arm A — stateless ceiling   : full context + question, no snapshot
#     Arm B — SSM-only baseline   : restore from a server on main branch
#                                   (pass --ssm-url; skip arm B if not provided)
#     Arm C — SSM+KV restore      : restore from server on this branch
#     Metrics: token-F1, input_tokens per arm
#     Headline: F1-gap-closed = (C_f1 - B_f1) / (A_f1 - B_f1)
#               token-saving  = 1 - C_tokens / A_tokens
#
# Usage (GPU pod):
#   python3 kha-398-kv-probe.py \
#     --kv-url http://localhost:30000 \
#     [--ssm-url http://localhost:30001] \
#     [--snap-dir /workspace/snapshots] \
#     [--conv-index 2] \
#     [--s3-upload]
#
# The script confirms at exit that:
#   - it ran on exp/kha-398-kv-probe
#   - nothing was merged

import argparse
import json
import logging
import os
import re
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests

logger = logging.getLogger("kha-398")
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------

def _post(url: str, payload: dict, timeout: float = 120.0) -> Tuple[int, dict]:
    r = requests.post(url, json=payload, timeout=timeout)
    try:
        body = r.json()
    except Exception:
        body = {"_raw": r.text[:2000]}
    return r.status_code, body


def wait_healthy(base_url: str, max_s: int = 600, label: str = "server") -> None:
    deadline = time.time() + max_s
    while time.time() < deadline:
        try:
            r = requests.get(f"{base_url}/health", timeout=5)
            if r.status_code == 200:
                logger.info("%s healthy at %s", label, base_url)
                return
        except requests.RequestException:
            pass
        time.sleep(5)
    raise RuntimeError(f"{label} at {base_url} did not become healthy in {max_s}s")


# ---------------------------------------------------------------------------
# Token-F1 scorer (replicated inline — no benchmark dep on GPU)
# ---------------------------------------------------------------------------

def _normalize(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9 ]", " ", text)
    return text.split()


def token_f1(prediction: str, reference: str) -> float:
    pred_tokens = _normalize(prediction)
    ref_tokens = _normalize(reference)
    if not pred_tokens and not ref_tokens:
        return 1.0
    if not pred_tokens or not ref_tokens:
        return 0.0
    common = set(pred_tokens) & set(ref_tokens)
    n_common = sum(min(pred_tokens.count(t), ref_tokens.count(t)) for t in common)
    if n_common == 0:
        return 0.0
    precision = n_common / len(pred_tokens)
    recall = n_common / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)


def score_answer(prediction: str, reference: Any) -> float:
    if isinstance(reference, list):
        return max(token_f1(prediction, r) for r in reference)
    return token_f1(prediction, str(reference))


# ---------------------------------------------------------------------------
# LoCoMo data loading
# ---------------------------------------------------------------------------

GITHUB_LOCOMO_URL = (
    "https://raw.githubusercontent.com/snap-research/locomo/main/data/locomo10.json"
)


def fetch_locomo10(cache_path: Optional[Path] = None) -> list:
    """Return locomo10 as a list of conversation dicts. Caches locally."""
    if cache_path and cache_path.exists():
        logger.info("Loading LoCoMo from cache: %s", cache_path)
        return json.loads(cache_path.read_text())

    logger.info("Fetching locomo10.json from GitHub...")
    r = requests.get(GITHUB_LOCOMO_URL, timeout=30)
    r.raise_for_status()
    data = r.json()
    convs = data if isinstance(data, list) else data.get("data", [data])
    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(convs))
        logger.info("Cached locomo10 to %s", cache_path)
    return convs


def build_conversation_text(conv: dict) -> str:
    """Flatten session_1, session_2, … turns into a single string."""
    parts = []
    session_num = 1
    while True:
        key = f"session_{session_num}"
        if key not in conv:
            break
        parts.append(f"[Session {session_num}]")
        for turn in conv[key]:
            speaker = turn.get("speaker_role", turn.get("speaker", "Speaker"))
            content = turn.get("content", turn.get("text", ""))
            parts.append(f"{speaker}: {content}")
        session_num += 1
    return "\n".join(parts)


def _rough_token_count(text: str) -> int:
    return len(text.split())


def select_conversation(convs: list, conv_index: int, max_tokens: int = 4096) -> dict:
    """Return the conversation at conv_index; warn if it exceeds max_tokens."""
    conv = convs[conv_index]
    text = build_conversation_text(conv)
    n = _rough_token_count(text)
    logger.info(
        "Conv index=%d id=%s  ~%d words in context (cap=%d)",
        conv_index,
        conv.get("dialogue_id", conv_index),
        n,
        max_tokens,
    )
    if n > max_tokens:
        logger.warning(
            "Conv exceeds KV cap (%d > %d) — KV will be partially captured",
            n, max_tokens,
        )
    return conv


# ---------------------------------------------------------------------------
# Snapshot RPCs
# ---------------------------------------------------------------------------

def save_snapshot(base_url: str, rid: str, conv_id: str, branch_name: str) -> bool:
    code, body = _post(f"{base_url}/save_snapshot", {
        "rid": rid,
        "conversation_id": conv_id,
        "branch_name": branch_name,
    })
    ok = body.get("success", False)
    logger.info("/save_snapshot HTTP %d success=%s branch=%s", code, ok, branch_name)
    if not ok:
        logger.warning("save_snapshot body: %s", json.dumps(body)[:300])
    return ok


def restore_snapshot(base_url: str, conv_id: str, branch_name: str) -> bool:
    code, body = _post(f"{base_url}/restore_snapshot", {
        "conversation_id": conv_id,
        "branch_name": branch_name,
    })
    ok = body.get("success", False)
    logger.info("/restore_snapshot HTTP %d success=%s branch=%s", code, ok, branch_name)
    if not ok:
        logger.warning("restore_snapshot body: %s", json.dumps(body)[:300])
    return ok


def chat(
    base_url: str,
    model: str,
    prompt: str,
    rid: Optional[str] = None,
    conv_id: Optional[str] = None,
    max_tokens: int = 128,
) -> dict:
    """OpenAI-compatible chat completion. Returns {text, input_tokens, ttft_s}."""
    messages = [{"role": "user", "content": prompt}]
    payload: dict = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": False,
    }
    if rid:
        payload["rid"] = rid
    if conv_id:
        payload["conversation_id"] = conv_id

    t0 = time.monotonic()
    code, body = _post(f"{base_url}/v1/chat/completions", payload, timeout=180.0)
    elapsed = time.monotonic() - t0

    if code != 200:
        logger.error("chat HTTP %d body=%s", code, json.dumps(body)[:400])
        return {"text": "", "input_tokens": 0, "ttft_s": elapsed, "error": True}

    choice = body.get("choices", [{}])[0]
    text = choice.get("message", {}).get("content", "")
    usage = body.get("usage", {})
    return {
        "text": text,
        "input_tokens": usage.get("prompt_tokens", _rough_token_count(prompt)),
        "output_tokens": usage.get("completion_tokens", 0),
        "ttft_s": elapsed,
        "error": False,
    }


# ---------------------------------------------------------------------------
# Step 1 — Planted-fact probe
# ---------------------------------------------------------------------------

PLANTED_FACT = "The secret code is ZEPHYR-7749."
PROBE_QUESTION = "What is the secret code?"
PROBE_SENTINEL = "ZEPHYR-7749"
PROBE_BRANCH = "kha398-probe-planted"


def step1_planted_fact_probe(
    kv_url: str,
    model: str,
    snap_dir: Optional[Path],
) -> dict:
    logger.info("=== STEP 1: Planted-fact probe ===")
    conv_id = f"kha398-probe-{uuid.uuid4().hex[:8]}"
    rid = f"{conv_id}-turn1"

    # Turn 1: send planted fact (establishes SSM + KV state)
    logger.info("Turn 1: planting fact with rid=%s", rid)
    r1 = chat(kv_url, model, PLANTED_FACT, rid=rid, conv_id=conv_id, max_tokens=32)
    if r1.get("error"):
        return {"step": 1, "pass": False, "error": "Turn 1 chat failed"}

    # Save snapshot with pinned branch_name
    saved = save_snapshot(kv_url, rid, conv_id, PROBE_BRANCH)
    if not saved:
        return {"step": 1, "pass": False, "error": "save_snapshot failed"}

    # Check KV file exists on disk
    kv_file_size_mb = None
    if snap_dir:
        # Look for kv_state file
        kv_candidates = list(snap_dir.rglob(f"*{PROBE_BRANCH}*kv_state*"))
        if kv_candidates:
            kv_file_size_mb = kv_candidates[0].stat().st_size / 1e6
            logger.info(
                "KV file found: %s (%.1f MB)", kv_candidates[0], kv_file_size_mb
            )
        else:
            logger.warning("No kv_state file found under %s", snap_dir)

    # Wait briefly for snapshot to settle
    time.sleep(0.5)

    # Turn 2: fresh restore + question-only
    conv_id2 = f"kha398-warm-{uuid.uuid4().hex[:8]}"
    restored = restore_snapshot(kv_url, conv_id2, PROBE_BRANCH)
    if not restored:
        return {"step": 1, "pass": False, "error": "restore_snapshot failed"}

    r2 = chat(kv_url, model, PROBE_QUESTION, conv_id=conv_id2, max_tokens=64)
    answer = r2.get("text", "")
    passed = PROBE_SENTINEL in answer

    logger.info(
        "Probe answer: %r  sentinel_found=%s", answer[:120], passed
    )

    result = {
        "step": 1,
        "pass": passed,
        "answer": answer,
        "sentinel": PROBE_SENTINEL,
        "kv_file_mb": kv_file_size_mb,
    }
    if passed:
        logger.info("STEP 1 PASS — KV warm restore recalled the planted fact.")
    else:
        logger.error("STEP 1 FAIL — KV warm restore did NOT recall '%s'.", PROBE_SENTINEL)
    return result


# ---------------------------------------------------------------------------
# Step 2 — Three-arm LoCoMo measurement
# ---------------------------------------------------------------------------

BRANCH_PREFIX = "kha398-locomo"
MAX_ANSWER_TOKENS = 128


def _run_arm_stateless(
    base_url: str,
    model: str,
    full_prompt: str,
    question: str,
    ref_answer: Any,
    label: str = "A_ceiling",
) -> dict:
    """Arm A: full context + question, no snapshot."""
    prompt = f"{full_prompt}\n\nQuestion: {question}"
    r = chat(base_url, model, prompt, max_tokens=MAX_ANSWER_TOKENS)
    f1 = score_answer(r["text"], ref_answer)
    logger.info(
        "Arm %s | f1=%.3f input_tokens=%d ans=%r",
        label, f1, r["input_tokens"], r["text"][:80],
    )
    return {
        "arm": label,
        "f1": f1,
        "input_tokens": r["input_tokens"],
        "answer": r["text"],
        "error": r.get("error", False),
    }


def _run_arm_restore(
    base_url: str,
    model: str,
    full_prompt: str,
    question: str,
    ref_answer: Any,
    branch_suffix: str,
    arm_label: str,
) -> dict:
    """Arms B/C: cold pass → save → restore → warm pass (question only)."""
    conv_id = f"kha398-{branch_suffix}-{uuid.uuid4().hex[:8]}"
    branch = f"{BRANCH_PREFIX}-{branch_suffix}-{conv_id[-8:]}"
    rid = f"{conv_id}-cold"

    # Cold pass: establish server-side state
    full_with_question = f"{full_prompt}\n\nQuestion: {question}"
    r_cold = chat(
        base_url, model, full_with_question, rid=rid,
        conv_id=conv_id, max_tokens=MAX_ANSWER_TOKENS,
    )
    if r_cold.get("error"):
        return {"arm": arm_label, "error": True, "f1": 0.0, "input_tokens": 0}

    # Save snapshot
    saved = save_snapshot(base_url, rid, conv_id, branch)
    if not saved:
        logger.warning("Arm %s: save_snapshot failed; skipping", arm_label)
        return {"arm": arm_label, "error": True, "f1": 0.0, "input_tokens": 0}

    time.sleep(0.3)

    # Restore snapshot into a fresh conversation
    conv_warm = f"kha398-warm-{uuid.uuid4().hex[:8]}"
    restored = restore_snapshot(base_url, conv_warm, branch)
    if not restored:
        logger.warning("Arm %s: restore_snapshot failed; skipping", arm_label)
        return {"arm": arm_label, "error": True, "f1": 0.0, "input_tokens": 0}

    # Warm pass: question only
    r_warm = chat(
        base_url, model, f"Question: {question}",
        conv_id=conv_warm, max_tokens=MAX_ANSWER_TOKENS,
    )
    f1 = score_answer(r_warm["text"], ref_answer)
    logger.info(
        "Arm %s | f1=%.3f input_tokens=%d ans=%r",
        arm_label, f1, r_warm["input_tokens"], r_warm["text"][:80],
    )
    return {
        "arm": arm_label,
        "f1": f1,
        "input_tokens": r_warm["input_tokens"],
        "cold_input_tokens": r_cold["input_tokens"],
        "answer": r_warm["text"],
        "error": False,
    }


def step2_three_arm_measurement(
    kv_url: str,
    model: str,
    ssm_url: Optional[str],
    conv: dict,
    conv_index: int,
) -> dict:
    logger.info("=== STEP 2: Three-arm LoCoMo measurement ===")
    conv_text = build_conversation_text(conv)
    qa_list = conv.get("qa", [])
    conv_id_label = str(conv.get("dialogue_id", conv_index))
    logger.info(
        "Conv %s | %d QA pairs | ~%d context words",
        conv_id_label, len(qa_list), _rough_token_count(conv_text),
    )

    if not qa_list:
        return {"step": 2, "error": "no QA pairs in selected conversation"}

    per_q: List[dict] = []
    for qi, qa in enumerate(qa_list):
        question = qa.get("question", "")
        ref_answer = qa.get("answer", "")
        if not question:
            continue
        logger.info("--- Q%d: %s", qi, question[:80])

        arm_a = _run_arm_stateless(kv_url, model, conv_text, question, ref_answer, "A_ceiling")

        arm_b = None
        if ssm_url:
            arm_b = _run_arm_restore(
                ssm_url, model, conv_text, question, ref_answer,
                branch_suffix="ssm", arm_label="B_ssm_only",
            )

        arm_c = _run_arm_restore(
            kv_url, model, conv_text, question, ref_answer,
            branch_suffix="kv", arm_label="C_ssm_kv",
        )

        per_q.append({
            "question_index": qi,
            "question": question,
            "reference": ref_answer,
            "arm_A": arm_a,
            "arm_B": arm_b,
            "arm_C": arm_c,
        })

    # Aggregate
    def _avg_f1(arm_key: str) -> Optional[float]:
        vals = [
            q[arm_key]["f1"]
            for q in per_q
            if q.get(arm_key) and not q[arm_key].get("error")
        ]
        return sum(vals) / len(vals) if vals else None

    def _avg_tokens(arm_key: str) -> Optional[float]:
        vals = [
            q[arm_key]["input_tokens"]
            for q in per_q
            if q.get(arm_key) and not q[arm_key].get("error")
        ]
        return sum(vals) / len(vals) if vals else None

    a_f1 = _avg_f1("arm_A")
    b_f1 = _avg_f1("arm_B") if ssm_url else None
    c_f1 = _avg_f1("arm_C")

    a_tok = _avg_tokens("arm_A")
    b_tok = _avg_tokens("arm_B") if ssm_url else None
    c_tok = _avg_tokens("arm_C")

    # F1-gap-closed = (C - B) / (A - B); defined only when B is available
    gap_closed = None
    if a_f1 is not None and b_f1 is not None and c_f1 is not None:
        denom = a_f1 - b_f1
        gap_closed = (c_f1 - b_f1) / denom if abs(denom) > 1e-6 else None

    # Token saving = 1 - C_tokens / A_tokens
    token_saving = None
    if a_tok and c_tok:
        token_saving = 1.0 - c_tok / a_tok

    logger.info("")
    logger.info("=== Three-arm summary ===")
    logger.info("  Arm A (ceiling)   f1=%-6s  input_tokens=%-6s", f"{a_f1:.3f}" if a_f1 is not None else "N/A", f"{a_tok:.0f}" if a_tok else "N/A")
    if ssm_url:
        logger.info("  Arm B (SSM-only)  f1=%-6s  input_tokens=%-6s", f"{b_f1:.3f}" if b_f1 is not None else "N/A", f"{b_tok:.0f}" if b_tok else "N/A")
    else:
        logger.info("  Arm B (SSM-only)  [skipped — --ssm-url not provided]")
    logger.info("  Arm C (SSM+KV)    f1=%-6s  input_tokens=%-6s", f"{c_f1:.3f}" if c_f1 is not None else "N/A", f"{c_tok:.0f}" if c_tok else "N/A")
    if gap_closed is not None:
        logger.info("  F1-gap-closed     %.1f%%", gap_closed * 100)
    if token_saving is not None:
        logger.info("  Token saving      %.1f%%", token_saving * 100)
    logger.info("")

    return {
        "step": 2,
        "conv_id": conv_id_label,
        "n_questions": len(per_q),
        "arm_A": {"avg_f1": a_f1, "avg_input_tokens": a_tok},
        "arm_B": {"avg_f1": b_f1, "avg_input_tokens": b_tok} if ssm_url else None,
        "arm_C": {"avg_f1": c_f1, "avg_input_tokens": c_tok},
        "f1_gap_closed": gap_closed,
        "token_saving": token_saving,
        "per_question": per_q,
    }


# ---------------------------------------------------------------------------
# S3 upload
# ---------------------------------------------------------------------------

def upload_to_s3(local_path: Path, s3_key: str) -> bool:
    """Upload file to DigitalOcean Spaces (s3://engram/...) via s3cmd."""
    cmd = [
        "s3cmd", "put", str(local_path),
        f"s3://engram/{s3_key}",
        "--host=sfo3.digitaloceanspaces.com",
        "--host-bucket=%(bucket)s.sfo3.digitaloceanspaces.com",
    ]
    logger.info("Uploading: %s", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        logger.error("s3cmd failed: %s", result.stderr[:500])
        return False
    logger.info("Uploaded to s3://engram/%s", s3_key)
    return True


# ---------------------------------------------------------------------------
# Branch guard
# ---------------------------------------------------------------------------

def check_branch() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            capture_output=True, text=True, cwd=Path(__file__).parent.parent,
        )
        branch = result.stdout.strip()
    except Exception:
        branch = "unknown"
    return branch


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="KHA-398 KV-probe measurement harness",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples (GPU pod):

  # Step 1 only (planted-fact gate):
  python3 kha-398-kv-probe.py --kv-url http://localhost:30000 --step 1

  # Both steps, SSM-only arm from a second server on main:
  python3 kha-398-kv-probe.py \\
    --kv-url http://localhost:30000 \\
    --ssm-url http://localhost:30001 \\
    --snap-dir /workspace/snapshots \\
    --s3-upload

  # Both steps, skip SSM-only arm (C vs A comparison only):
  python3 kha-398-kv-probe.py \\
    --kv-url http://localhost:30000 \\
    --snap-dir /workspace/snapshots \\
    --s3-upload
""",
    )
    p.add_argument(
        "--kv-url", required=True,
        help="Base URL of the server running exp/kha-398-kv-probe branch",
    )
    p.add_argument(
        "--ssm-url", default=None,
        help="Base URL of a server running main (SSM-only). If omitted, Arm B is skipped.",
    )
    p.add_argument(
        "--model", default="default",
        help="Model name sent in chat requests (default: 'default')",
    )
    p.add_argument(
        "--snap-dir", type=Path, default=None,
        help="Snapshot directory on the local machine (for KV file size check in Step 1)",
    )
    p.add_argument(
        "--conv-index", type=int, default=2,
        help="Index into locomo10.json to use for Step 2 (default: 2 = conv-26)",
    )
    p.add_argument(
        "--locomo-cache", type=Path,
        default=Path("/tmp/locomo10_kha398.json"),
        help="Local cache path for locomo10.json download",
    )
    p.add_argument(
        "--step", type=int, choices=[0, 1, 2], default=0,
        help="0=both steps (default), 1=Step 1 only, 2=Step 2 only",
    )
    p.add_argument(
        "--s3-upload", action="store_true",
        help="Upload results JSON to s3://engram/benchmarks/kha-398-kv-probe/",
    )
    p.add_argument(
        "--out-dir", type=Path, default=Path("/tmp/kha398"),
        help="Local directory for result JSON (default: /tmp/kha398)",
    )
    p.add_argument(
        "--skip-branch-check", action="store_true",
        help="Skip the branch guard (for debugging outside git repo)",
    )
    return p


def main() -> None:
    args = build_parser().parse_args()

    # Branch guard
    branch = check_branch()
    if not args.skip_branch_check:
        if branch != "exp/kha-398-kv-probe":
            logger.error(
                "Branch guard: expected exp/kha-398-kv-probe, got '%s'. "
                "Pass --skip-branch-check to override (non-merge risk only).",
                branch,
            )
            sys.exit(1)
    logger.info("Branch: %s", branch)

    # Health checks
    wait_healthy(args.kv_url, label="kv-server")
    if args.ssm_url:
        wait_healthy(args.ssm_url, label="ssm-server")

    run_id = f"kha398-{int(time.time())}"
    results: dict = {"run_id": run_id, "branch": branch, "model": args.model}

    # Step 1
    if args.step in (0, 1):
        s1 = step1_planted_fact_probe(
            kv_url=args.kv_url,
            model=args.model,
            snap_dir=args.snap_dir,
        )
        results["step1"] = s1
        if not s1.get("pass") and args.step == 1:
            logger.error("Step 1 FAILED — aborting.")
            _save_and_maybe_upload(results, args, run_id)
            sys.exit(2)

    # Step 2
    if args.step in (0, 2):
        convs = fetch_locomo10(cache_path=args.locomo_cache)
        if args.conv_index >= len(convs):
            logger.error(
                "conv_index=%d out of range (dataset has %d conversations).",
                args.conv_index, len(convs),
            )
            sys.exit(1)
        conv = select_conversation(convs, args.conv_index)
        s2 = step2_three_arm_measurement(
            kv_url=args.kv_url,
            model=args.model,
            ssm_url=args.ssm_url,
            conv=conv,
            conv_index=args.conv_index,
        )
        results["step2"] = s2

    _save_and_maybe_upload(results, args, run_id)

    # Final summary
    logger.info("")
    logger.info("=== KHA-398 RUN COMPLETE ===")
    logger.info("branch  : %s", branch)
    logger.info("run_id  : %s", run_id)
    if "step1" in results:
        s1 = results["step1"]
        logger.info("Step 1  : %s  kv_file=%.1f MB",
                    "PASS" if s1.get("pass") else "FAIL",
                    s1.get("kv_file_mb") or 0)
    if "step2" in results:
        s2 = results["step2"]
        a = s2.get("arm_A") or {}
        b = s2.get("arm_B") or {}
        c = s2.get("arm_C") or {}
        logger.info("Step 2 results:")
        logger.info("  Arm A ceiling  f1=%.3f  tok=%s", a.get("avg_f1") or 0, a.get("avg_input_tokens") or "N/A")
        if s2.get("arm_B"):
            logger.info("  Arm B SSM-only f1=%.3f  tok=%s", b.get("avg_f1") or 0, b.get("avg_input_tokens") or "N/A")
        logger.info("  Arm C SSM+KV   f1=%.3f  tok=%s", c.get("avg_f1") or 0, c.get("avg_input_tokens") or "N/A")
        if s2.get("f1_gap_closed") is not None:
            logger.info("  F1-gap-closed  %.1f%%", s2["f1_gap_closed"] * 100)
        if s2.get("token_saving") is not None:
            logger.info("  Token saving   %.1f%%", s2["token_saving"] * 100)
    logger.info("")
    logger.info("MERGE GUARD: nothing merged. Branch = %s", branch)


def _save_and_maybe_upload(results: dict, args: argparse.Namespace, run_id: str) -> None:
    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"{run_id}.json"
    out_file.write_text(json.dumps(results, indent=2, default=str))
    logger.info("Results saved to %s", out_file)

    if args.s3_upload:
        s3_key = f"benchmarks/kha-398-kv-probe/{run_id}.json"
        upload_to_s3(out_file, s3_key)


if __name__ == "__main__":
    main()
