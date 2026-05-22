"""KHA-397 Addendum 2 — Restore cost measurement (warm vs cold).

Answers: does cold restore deliver Engram's token-reduction / fast-restore
benefit, or does it silently re-prefill the full history?

Metrics captured per request:
  - Extend (prefill) token count, scraped from the server log's
    "Prefill batch, #new-token: N" lines. This is the authoritative
    source from `metrics_reporter.py:491` and reflects the tokens the
    scheduler actually pushed through prefill.
  - TTFT proxy: total wall time of a request with max_new_tokens=1
    (decode of 1 token is negligible vs prefill of hundreds-thousands).
  - Restore round-trip latency (for restore conditions).
  - Snapshot size on disk (du -sb).

Modes:
  --mode warm-suite     run N iterations on an already-running server:
                          for each iter:
                            chat(system+P_salted) -> R, save_snap
                            restore(continuation=M, max_new_tokens=1)
                          then N iters of baseline chat(system+P_salted,M).
                          Writes warm-{Lhint}.json.
  --mode cold-one       on a freshly-started server, restore one already-
                          existing snapshot. Appends to cold-{Lhint}.json.
  --mode summarize      merge warm-{L}.json + cold-{L}.json, compute
                          medians, write summary-{L}.json.

Radix cache should be ENABLED for this experiment (do NOT pass
--disable-radix-cache when launching the server) — the warm-vs-cold
hypothesis is about radix-tree state.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Any

import requests
from transformers import AutoTokenizer


SYSTEM_MSG = (
    "You are a terse assistant. Follow the user's most recent instruction "
    "exactly. Reply with only what is asked, nothing more."
)
M_CONTENT = "Now reply with only the word: BANANA"
# Short ack so R is deterministic and tiny across all iterations.
SHORT_R_PROMPT_SUFFIX = " Just say 'OK'."

# Raw-mode delimiters — used only when --raw-mode is set, for tokenizers
# without a chat template (e.g. base-model Codestral Mamba). The exact
# choice of separator doesn't matter for the cost question (which is
# about server-side prefill behaviour), only that P and M are
# concatenated deterministically and tokenize to a sensible length.
RAW_SYSTEM_PREFIX = "System: "
RAW_USER_PREFIX = "\n\nUser: "
RAW_ASSISTANT_PROMPT = "\n\nAssistant:"

# Regex for "Prefill batch, #new-seq: K, #new-token: N, ..."
PREFILL_RE = re.compile(
    r"Prefill batch.*?#new-seq:\s*(\d+).*?#new-token:\s*(\d+).*?#cached-token:\s*(\d+)"
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=30002)
    p.add_argument("--model-path", default="/workspace/models/granite-4.0-h-tiny")
    p.add_argument(
        "--mode",
        choices=["warm-suite", "cold-one", "summarize"],
        required=True,
    )
    p.add_argument(
        "--history-tokens",
        type=int,
        default=512,
        help="Target token count for P (the history). Used to size the "
             "padding text. The actual fill_ids length will be reported.",
    )
    p.add_argument(
        "--iterations",
        type=int,
        default=5,
        help="Iterations per condition (warm-suite mode only).",
    )
    p.add_argument(
        "--iter",
        type=int,
        default=0,
        help="Cold iteration index (cold-one mode).",
    )
    p.add_argument(
        "--server-log",
        required=False,
        help="Path to the server's log file (for scraping Prefill batch).",
    )
    p.add_argument(
        "--out-dir",
        default="/tmp/kha397-cost",
    )
    p.add_argument(
        "--snapshot-dir",
        default="/workspace/snapshots",
        help="Directory the server writes conversation_<id> snapshots to "
             "(for du -sb size measurement).",
    )
    p.add_argument("--request-timeout", type=float, default=180.0)
    p.add_argument(
        "--raw-mode",
        action="store_true",
        help="Skip apply_chat_template — use raw text via /generate. "
             "Required for tokenizers without a chat template (e.g. base "
             "Codestral Mamba).",
    )
    return p.parse_args()


# ---------------------------------------------------------------------
# Server log scraping
# ---------------------------------------------------------------------


def log_offset(path: str) -> int:
    try:
        return os.path.getsize(path)
    except FileNotFoundError:
        return 0


def scrape_extend_tokens(log_path: str, start_offset: int,
                         poll_seconds: float = 2.0) -> dict[str, Any]:
    """Read server log from start_offset, return prefill-batch info from
    the last "Prefill batch" line emitted since. Polls briefly because
    the server's logger may flush slightly after the HTTP response
    returns."""
    none = lambda raw=None: {"new_token": None, "new_seq": None,
                             "cached_token": None, "raw_line": raw,
                             "found": False}
    deadline = time.time() + poll_seconds
    while True:
        try:
            size = os.path.getsize(log_path)
        except FileNotFoundError:
            size = 0
        if size > start_offset:
            with open(log_path, "rb") as f:
                f.seek(start_offset)
                chunk = f.read(size - start_offset).decode("utf-8", errors="replace")
            lines = [ln for ln in chunk.splitlines() if "Prefill batch" in ln]
            if lines:
                last = lines[-1]
                m = PREFILL_RE.search(last)
                if not m:
                    return {**none(last), "found": True}
                return {
                    "new_seq": int(m.group(1)),
                    "new_token": int(m.group(2)),
                    "cached_token": int(m.group(3)),
                    "raw_line": last,
                    "found": True,
                }
        if time.time() >= deadline:
            return none()
        time.sleep(0.05)


# ---------------------------------------------------------------------
# Prompt generation
# ---------------------------------------------------------------------


_PARAGRAPH = (
    "The cellulose-based parchment carried the cartographer's faded "
    "annotations, brittle now from decades of dry storage. Each margin "
    "bore a different hand: the ink of the original surveyor, the "
    "blue-pencil corrections of a later editor, and the smudged "
    "graphite of an apprentice who, by the look of it, had attempted "
    "to redraw the southern coastline three or four times before "
    "abandoning the effort. A wind-rose decorated the upper corner, "
    "its petals labelled in a half-remembered nautical dialect, and "
    "below it the legend explained symbols for fords, ruined towers, "
    "and the meeting-places of two roads that no longer ran where the "
    "map insisted they should. "
)


def build_padded_p(salt: str, target_tokens: int, tok: AutoTokenizer,
                   raw_mode: bool = False) -> tuple[str, int]:
    """Return a user-content string whose formatted tokenization
    approaches target_tokens. Pads with copies of _PARAGRAPH until the
    target is met, then adds SHORT_R_PROMPT_SUFFIX so R is deterministic.

    In raw_mode (no chat template), the user-content alone is what gets
    sent; in chat-template mode, system+user is wrapped with the
    tokenizer's chat template.
    """
    body = f"[salt={salt}] Please ignore this filler text. "
    while True:
        candidate = body + SHORT_R_PROMPT_SUFFIX
        if raw_mode:
            full = (RAW_SYSTEM_PREFIX + SYSTEM_MSG +
                    RAW_USER_PREFIX + candidate + RAW_ASSISTANT_PROMPT)
            ids = tok.encode(full, add_special_tokens=True)
        else:
            messages = [
                {"role": "system", "content": SYSTEM_MSG},
                {"role": "user", "content": candidate},
            ]
            formatted = tok.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True)
            ids = tok.encode(formatted, add_special_tokens=False)
        if len(ids) >= target_tokens:
            return candidate, len(ids)
        body += _PARAGRAPH


def encode_continuation(tok: AutoTokenizer, user_content: str,
                        raw_mode: bool = False) -> list[int]:
    if raw_mode:
        # Mirror the raw_mode user-turn shape used during the initial
        # chat: the continuation is "what the user would have said next",
        # so we wrap with the same User:/Assistant: scaffold and
        # ask the tokenizer to add no special tokens (the BOS is already
        # at position 0 from the original prefill).
        full = (RAW_USER_PREFIX + user_content + RAW_ASSISTANT_PROMPT)
        return tok.encode(full, add_special_tokens=False)
    formatted = tok.apply_chat_template(
        [{"role": "user", "content": user_content}],
        tokenize=False, add_generation_prompt=True,
    )
    return tok.encode(formatted, add_special_tokens=False)


# ---------------------------------------------------------------------
# HTTP wrappers
# ---------------------------------------------------------------------


def chat(url, model, messages, rid, max_tokens, timeout):
    payload = {
        "model": model, "messages": messages, "temperature": 0,
        "max_tokens": max_tokens, "stream": False, "rid": rid,
    }
    t0 = time.perf_counter()
    r = requests.post(f"{url}/v1/chat/completions", json=payload, timeout=timeout)
    elapsed = (time.perf_counter() - t0) * 1000
    r.raise_for_status()
    body = r.json()
    usage = body.get("usage", {}) or {}
    return {
        "text": body["choices"][0]["message"]["content"],
        "latency_ms": elapsed,
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
    }


def generate_raw(url, text, rid, conv_id, max_tokens, timeout):
    """Raw-mode equivalent of chat(): uses SGLang's native /generate
    endpoint with a plain-text input (no chat template applied). Returns
    the same shape as chat() so the warm-suite code path is identical.
    """
    payload = {
        "text": text,
        "sampling_params": {
            "temperature": 0,
            "max_new_tokens": max_tokens,
        },
        "stream": False,
        "rid": rid,
        "conversation_id": conv_id,
    }
    t0 = time.perf_counter()
    r = requests.post(f"{url}/generate", json=payload, timeout=timeout)
    elapsed = (time.perf_counter() - t0) * 1000
    r.raise_for_status()
    body = r.json()
    if isinstance(body, list):
        body = body[0]
    meta = body.get("meta_info", {}) or {}
    return {
        "text": body.get("text", ""),
        "latency_ms": elapsed,
        "prompt_tokens": meta.get("prompt_tokens"),
        "completion_tokens": meta.get("completion_tokens"),
    }


def save_snap(url, conv_id, timeout):
    t0 = time.perf_counter()
    r = requests.post(f"{url}/save_snapshot",
                      json={"conversation_id": conv_id}, timeout=timeout)
    elapsed = (time.perf_counter() - t0) * 1000
    r.raise_for_status()
    body = r.json()
    body["client_latency_ms"] = elapsed
    return body


def restore(url, conv_id, continuation_ids, max_new_tokens, timeout):
    t0 = time.perf_counter()
    r = requests.post(
        f"{url}/restore_snapshot",
        json={
            "conversation_id": conv_id,
            "create_new_request": True,
            "continuation_ids": continuation_ids,
            "max_new_tokens": max_new_tokens,
        },
        timeout=timeout,
    )
    elapsed = (time.perf_counter() - t0) * 1000
    r.raise_for_status()
    body = r.json()
    body["client_latency_ms"] = elapsed
    return body


def snapshot_size_bytes(snapshot_dir: str, conv_id: str) -> int | None:
    """du -sb on the conversation's snapshot dir."""
    p = Path(snapshot_dir) / f"conversation_{conv_id}"
    if not p.exists():
        return None
    try:
        out = subprocess.check_output(
            ["du", "-sb", str(p)], text=True, timeout=10)
        return int(out.strip().split()[0])
    except Exception:
        return None


# ---------------------------------------------------------------------
# Mode: warm-suite
# ---------------------------------------------------------------------


def run_warm_suite(args, tok, model_name, url) -> int:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = args.server_log
    raw = args.raw_mode

    print(f"[warm-suite] target_history_tokens={args.history_tokens} "
          f"iters={args.iterations} raw_mode={raw}")
    warm_records: list[dict[str, Any]] = []
    baseline_records: list[dict[str, Any]] = []

    # Pre-compute the M continuation tokens (shared across iters).
    continuation_ids = encode_continuation(tok, M_CONTENT, raw_mode=raw)
    print(f"[warm-suite] M continuation_ids len={len(continuation_ids)}")

    for i in range(1, args.iterations + 1):
        salt = uuid.uuid4().hex[:8]
        p_text, expected_len = build_padded_p(
            salt, args.history_tokens, tok, raw_mode=raw)
        conv_id = f"kha397-cost-warm-{salt}"

        # ---- Setup: chat to establish R + save snapshot
        off = log_offset(log_path)
        if raw:
            raw_text = (RAW_SYSTEM_PREFIX + SYSTEM_MSG +
                        RAW_USER_PREFIX + p_text + RAW_ASSISTANT_PROMPT)
            setup = generate_raw(url, raw_text, conv_id, conv_id,
                                 max_tokens=4, timeout=args.request_timeout)
        else:
            messages = [
                {"role": "system", "content": SYSTEM_MSG},
                {"role": "user", "content": p_text},
            ]
            setup = chat(url, model_name, messages, conv_id,
                         max_tokens=4, timeout=args.request_timeout)
        setup_prefill = scrape_extend_tokens(log_path, off)
        snap = save_snap(url, conv_id, args.request_timeout)
        snap_size = snapshot_size_bytes(args.snapshot_dir, conv_id)
        if not snap.get("success"):
            print(f"  [iter {i}] save_snap FAIL: {snap}")
            continue

        # ---- WARM restore: continuation=M, max_new_tokens=1 (TTFT proxy)
        off = log_offset(log_path)
        r_warm = restore(url, conv_id, continuation_ids,
                         max_new_tokens=1, timeout=args.request_timeout)
        warm_prefill = scrape_extend_tokens(log_path, off)
        if not r_warm.get("success"):
            print(f"  [iter {i}] warm restore FAIL: {r_warm}")
            continue

        rec = {
            "iter": i, "conv_id": conv_id, "salt": salt,
            "p_text_chars": len(p_text),
            "setup_prompt_tokens_chat_api": setup.get("prompt_tokens"),
            "setup_chat_latency_ms": setup["latency_ms"],
            "setup_extend_tokens_log": setup_prefill.get("new_token"),
            "setup_cached_tokens_log": setup_prefill.get("cached_token"),
            "save_latency_ms": snap["client_latency_ms"],
            "snapshot_size_bytes": snap_size,
            "warm_restore_latency_ms": r_warm["client_latency_ms"],
            "warm_extend_tokens_log": warm_prefill.get("new_token"),
            "warm_cached_tokens_log": warm_prefill.get("cached_token"),
            "warm_output_text": r_warm.get("output_text"),
        }
        warm_records.append(rec)
        print(f"  [warm iter {i}] setup_extend={setup_prefill.get('new_token')} "
              f"warm_extend={warm_prefill.get('new_token')} "
              f"warm_cached={warm_prefill.get('cached_token')} "
              f"ttft={r_warm['client_latency_ms']:.0f}ms "
              f"snap_size={snap_size}B")

    # ---- BASELINE: cold-start baseline = chat with [system + user:P+M],
    # max_tokens=1. Uses unique salts to avoid radix accumulation.
    print(f"\n[warm-suite] running {args.iterations} baseline chats")
    for i in range(1, args.iterations + 1):
        salt = uuid.uuid4().hex[:8]
        p_text, _ = build_padded_p(
            salt, args.history_tokens, tok, raw_mode=raw)
        baseline_user = p_text + "\n\n" + M_CONTENT
        rid = f"kha397-cost-baseline-{salt}"
        off = log_offset(log_path)
        if raw:
            raw_text = (RAW_SYSTEM_PREFIX + SYSTEM_MSG +
                        RAW_USER_PREFIX + baseline_user + RAW_ASSISTANT_PROMPT)
            b = generate_raw(url, raw_text, rid, rid,
                             max_tokens=1, timeout=args.request_timeout)
        else:
            messages = [
                {"role": "system", "content": SYSTEM_MSG},
                {"role": "user", "content": baseline_user},
            ]
            b = chat(url, model_name, messages, rid,
                     max_tokens=1, timeout=args.request_timeout)
        bp = scrape_extend_tokens(log_path, off)
        rec = {
            "iter": i, "salt": salt,
            "ttft_ms": b["latency_ms"],
            "prompt_tokens_chat_api": b.get("prompt_tokens"),
            "extend_tokens_log": bp.get("new_token"),
            "cached_tokens_log": bp.get("cached_token"),
        }
        baseline_records.append(rec)
        print(f"  [baseline iter {i}] extend={bp.get('new_token')} "
              f"cached={bp.get('cached_token')} ttft={b['latency_ms']:.0f}ms")

    out = {
        "history_tokens_target": args.history_tokens,
        "iterations": args.iterations,
        "warm": warm_records,
        "baseline": baseline_records,
        # Manifest of conv_ids for cold-one runs to reference:
        "snapshot_manifest": [{"iter": r["iter"], "conv_id": r["conv_id"],
                               "snapshot_size_bytes": r["snapshot_size_bytes"]}
                              for r in warm_records],
    }
    out_path = out_dir / f"warm-{args.history_tokens}.json"
    out_path.write_text(json.dumps(out, indent=2, default=str))
    print(f"\n[OUT] {out_path}")
    return 0


# ---------------------------------------------------------------------
# Mode: cold-one
# ---------------------------------------------------------------------


def run_cold_one(args, tok, model_name, url) -> int:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = args.server_log

    warm_path = out_dir / f"warm-{args.history_tokens}.json"
    if not warm_path.exists():
        print(f"[FAIL] warm manifest not found: {warm_path}")
        return 1
    warm = json.loads(warm_path.read_text())
    manifest = warm["snapshot_manifest"]
    if args.iter < 1 or args.iter > len(manifest):
        print(f"[FAIL] --iter {args.iter} out of range "
              f"(manifest has {len(manifest)} entries)")
        return 1
    target = next(m for m in manifest if m["iter"] == args.iter)
    conv_id = target["conv_id"]

    # Sanity: server should be fresh — verify snapshot is loadable.
    info = requests.post(f"{url}/list_snapshots",
                         json={"conversation_id": conv_id},
                         timeout=args.request_timeout).json()
    if not info.get("snapshots"):
        print(f"[FAIL] snapshot {conv_id} not visible on fresh server")
        return 1

    continuation_ids = encode_continuation(
        tok, M_CONTENT, raw_mode=args.raw_mode)
    off = log_offset(log_path)
    t0 = time.perf_counter()
    r_cold = restore(url, conv_id, continuation_ids,
                     max_new_tokens=1, timeout=args.request_timeout)
    elapsed = (time.perf_counter() - t0) * 1000
    cold_prefill = scrape_extend_tokens(log_path, off)
    if not r_cold.get("success"):
        print(f"[FAIL] cold restore: {r_cold}")
        return 1
    rec = {
        "iter": args.iter, "conv_id": conv_id,
        "cold_restore_latency_ms": r_cold["client_latency_ms"],
        "wall_ms": elapsed,
        "cold_extend_tokens_log": cold_prefill.get("new_token"),
        "cold_cached_tokens_log": cold_prefill.get("cached_token"),
        "cold_output_text": r_cold.get("output_text"),
    }
    cold_path = out_dir / f"cold-{args.history_tokens}.json"
    if cold_path.exists():
        existing = json.loads(cold_path.read_text())
    else:
        existing = {"history_tokens_target": args.history_tokens, "cold": []}
    existing["cold"].append(rec)
    cold_path.write_text(json.dumps(existing, indent=2, default=str))
    print(f"[cold iter {args.iter}] extend={cold_prefill.get('new_token')} "
          f"cached={cold_prefill.get('cached_token')} "
          f"ttft={r_cold['client_latency_ms']:.0f}ms")
    return 0


# ---------------------------------------------------------------------
# Mode: summarize
# ---------------------------------------------------------------------


def _median(xs):
    xs = [x for x in xs if x is not None]
    if not xs:
        return None
    return statistics.median(xs)


def run_summarize(args) -> int:
    out_dir = Path(args.out_dir)
    warm_path = out_dir / f"warm-{args.history_tokens}.json"
    cold_path = out_dir / f"cold-{args.history_tokens}.json"
    if not warm_path.exists():
        print(f"[FAIL] no warm file: {warm_path}")
        return 1
    warm = json.loads(warm_path.read_text())
    cold = json.loads(cold_path.read_text()) if cold_path.exists() else {"cold": []}

    warm_iters = warm["warm"]
    base_iters = warm["baseline"]
    cold_iters = cold["cold"]

    def stats(name, xs):
        xs = [x for x in xs if x is not None]
        if not xs:
            return {"n": 0}
        return {
            "n": len(xs),
            "median": statistics.median(xs),
            "min": min(xs),
            "max": max(xs),
            "mean": round(statistics.mean(xs), 2),
        }

    summary = {
        "history_tokens_target": args.history_tokens,
        "history_tokens_actual_median": _median(
            [w.get("setup_extend_tokens_log") for w in warm_iters]),
        "snapshot_size_bytes_median": _median(
            [w.get("snapshot_size_bytes") for w in warm_iters]),
        "baseline": {
            "n": len(base_iters),
            "extend_tokens": stats("baseline_extend",
                                   [b.get("extend_tokens_log") for b in base_iters]),
            "prompt_tokens_chat_api": stats("baseline_prompt",
                                            [b.get("prompt_tokens_chat_api") for b in base_iters]),
            "ttft_ms": stats("baseline_ttft",
                             [b.get("ttft_ms") for b in base_iters]),
        },
        "warm_rc": {
            "n": len(warm_iters),
            "extend_tokens": stats("warm_extend",
                                   [w.get("warm_extend_tokens_log") for w in warm_iters]),
            "cached_tokens": stats("warm_cached",
                                   [w.get("warm_cached_tokens_log") for w in warm_iters]),
            "ttft_ms": stats("warm_ttft",
                             [w.get("warm_restore_latency_ms") for w in warm_iters]),
        },
        "cold_rc": {
            "n": len(cold_iters),
            "extend_tokens": stats("cold_extend",
                                   [c.get("cold_extend_tokens_log") for c in cold_iters]),
            "cached_tokens": stats("cold_cached",
                                   [c.get("cold_cached_tokens_log") for c in cold_iters]),
            "ttft_ms": stats("cold_ttft",
                             [c.get("cold_restore_latency_ms") for c in cold_iters]),
        },
        "warm_records": warm_iters,
        "baseline_records": base_iters,
        "cold_records": cold_iters,
    }
    out_path = out_dir / f"summary-{args.history_tokens}.json"
    out_path.write_text(json.dumps(summary, indent=2, default=str))
    # Print key results to stdout
    print(f"\n=== SUMMARY  history_tokens_target={args.history_tokens}  "
          f"actual_median={summary['history_tokens_actual_median']}  "
          f"snapshot_size_med={summary['snapshot_size_bytes_median']}B ===")
    for cond, body in [("baseline", summary["baseline"]),
                       ("warm_rc",  summary["warm_rc"]),
                       ("cold_rc",  summary["cold_rc"])]:
        et = body["extend_tokens"].get("median")
        ct = body.get("cached_tokens", {}).get("median")
        tt = body["ttft_ms"].get("median")
        n = body["n"]
        line = (f"  {cond:>10}  n={n}  "
                f"extend_med={et}  cached_med={ct}  ttft_med={tt}ms")
        print(line)
    print(f"[OUT] {out_path}")
    return 0


# ---------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------


def main() -> int:
    args = parse_args()
    if args.mode == "summarize":
        return run_summarize(args)
    if args.server_log is None:
        print("[FAIL] --server-log required for warm-suite/cold-one")
        return 1
    url = f"http://{args.host}:{args.port}"
    tok = AutoTokenizer.from_pretrained(args.model_path)
    model_info = requests.get(f"{url}/get_model_info", timeout=10).json()
    model_name = model_info["model_path"]
    print(f"[setup] server reports model_path={model_name}  mode={args.mode}")
    if args.mode == "warm-suite":
        return run_warm_suite(args, tok, model_name, url)
    return run_cold_one(args, tok, model_name, url)


if __name__ == "__main__":
    sys.exit(main())
