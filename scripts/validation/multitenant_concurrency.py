"""KHA-301 multi-tenant concurrency validation harness.

Exercises N simultaneous snapshot sessions against a running SGLang server.
Each session runs a multi-turn conversation, saves/restores snapshots between
turns, and embeds a unique tag-word that later turns must recall. Aggregated
results cover per-session throughput, save latency under load, restore success
rate, and cross-session state isolation.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import dataclasses
import json
import statistics
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests


PHONETIC_TAGS = [
    "alpha", "bravo", "charlie", "delta", "echo", "foxtrot", "golf",
    "hotel", "india", "juliet", "kilo", "lima", "mike", "november",
    "oscar", "papa", "quebec", "romeo", "sierra", "tango", "uniform",
    "victor", "whiskey", "xray", "yankee", "zulu",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Multi-tenant concurrency harness")
    p.add_argument("--port", type=int, default=30001)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--sessions", type=int, default=5)
    p.add_argument("--turns", type=int, default=5)
    p.add_argument("--max-new-tokens", type=int, default=48)
    p.add_argument("--request-timeout", type=float, default=120.0)
    p.add_argument("--save-retries", type=int, default=20)
    p.add_argument("--restore-retries", type=int, default=3)
    p.add_argument("--stagger-seconds", type=float, default=0.0,
                   help="Optional per-session startup stagger to reduce thundering-herd on first prefill.")
    p.add_argument("--output-json", type=Path, required=True)
    return p.parse_args()


def server_url(args: argparse.Namespace) -> str:
    return f"http://{args.host}:{args.port}"


def health(url: str, timeout: float = 10.0) -> None:
    r = requests.get(f"{url}/health", timeout=timeout)
    r.raise_for_status()


def model_info(url: str, timeout: float = 10.0) -> dict[str, Any]:
    r = requests.get(f"{url}/get_model_info", timeout=timeout)
    r.raise_for_status()
    return r.json()


def chat(url: str, model: str, messages: list[dict[str, str]], rid: str,
         max_new_tokens: int, timeout: float) -> dict[str, Any]:
    payload = {
        "model": model,
        "messages": messages,
        "temperature": 0,
        "max_tokens": max_new_tokens,
        "stream": False,
        "rid": rid,
    }
    start = time.perf_counter()
    r = requests.post(f"{url}/v1/chat/completions", json=payload, timeout=timeout)
    elapsed_ms = (time.perf_counter() - start) * 1000
    r.raise_for_status()
    body = r.json()
    text = body["choices"][0]["message"]["content"]
    usage = body.get("usage", {}) or {}
    return {
        "text": text,
        "latency_ms": elapsed_ms,
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
    }


def save_snapshot(url: str, conversation_id: str, timeout: float,
                  retries: int) -> dict[str, Any]:
    last: dict[str, Any] | None = None
    for attempt in range(retries):
        start = time.perf_counter()
        try:
            r = requests.post(
                f"{url}/save_snapshot",
                json={"conversation_id": conversation_id},
                timeout=timeout,
            )
            elapsed_ms = (time.perf_counter() - start) * 1000
            r.raise_for_status()
            body = r.json()
            body["client_latency_ms"] = elapsed_ms
            body["attempts"] = attempt + 1
            if body.get("success") is True:
                return body
            last = body
        except (requests.exceptions.RequestException, json.JSONDecodeError) as e:
            elapsed_ms = (time.perf_counter() - start) * 1000
            last = {
                "success": False,
                "error": str(e),
                "client_latency_ms": elapsed_ms,
                "attempts": attempt + 1,
            }
        time.sleep(0.5)
    return last or {"success": False, "error": "no_response", "attempts": retries}


def restore_snapshot(url: str, conversation_id: str, continuation_ids: list[int],
                     timeout: float, retries: int,
                     max_new_tokens: int = 1) -> dict[str, Any]:
    last: dict[str, Any] | None = None
    for attempt in range(retries):
        start = time.perf_counter()
        try:
            r = requests.post(
                f"{url}/restore_snapshot",
                json={
                    "conversation_id": conversation_id,
                    "create_new_request": True,
                    "continuation_ids": continuation_ids,
                    "max_new_tokens": max_new_tokens,
                },
                timeout=timeout,
            )
            elapsed_ms = (time.perf_counter() - start) * 1000
            r.raise_for_status()
            body = r.json()
            body["client_latency_ms"] = elapsed_ms
            body["attempts"] = attempt + 1
            if body.get("success") is True:
                return body
            last = body
        except (requests.exceptions.RequestException, json.JSONDecodeError) as e:
            elapsed_ms = (time.perf_counter() - start) * 1000
            last = {
                "success": False,
                "error": str(e),
                "client_latency_ms": elapsed_ms,
                "attempts": attempt + 1,
            }
        time.sleep(0.5)
    return last or {"success": False, "error": "no_response", "attempts": retries}


def tokenize(url: str, text: str, timeout: float = 30.0) -> list[int]:
    """Use the server's tokenizer via /tokenize for restore continuation_ids."""
    r = requests.post(f"{url}/tokenize", json={"text": text}, timeout=timeout)
    if r.status_code == 404:
        return []
    r.raise_for_status()
    body = r.json()
    if isinstance(body, list) and body and isinstance(body[0], int):
        return body
    if isinstance(body, dict):
        for k in ("input_ids", "tokens", "token_ids"):
            if k in body and isinstance(body[k], list):
                return body[k]
    return []


@dataclasses.dataclass
class TurnRecord:
    turn_index: int
    chat_latency_ms: float | None
    save_latency_ms: float | None
    save_success: bool
    save_attempts: int
    restore_latency_ms: float | None
    restore_success: bool | None
    restore_attempts: int | None
    response_text: str
    contains_own_tag: bool
    foreign_tags_seen: list[str]


@dataclasses.dataclass
class SessionResult:
    session_id: int
    conversation_id: str
    tag_word: str
    started_at: float
    finished_at: float
    turns: list[TurnRecord]
    completed: bool
    error: str | None


def run_session(args: argparse.Namespace, session_id: int, tag_word: str,
                all_tags: list[str], model: str) -> SessionResult:
    url = server_url(args)
    conv_id = f"kha301-s{session_id}-{uuid.uuid4().hex[:8]}"
    started = time.perf_counter()
    turns: list[TurnRecord] = []
    completed = False
    error: str | None = None

    system_msg = {
        "role": "system",
        "content": (
            "You are a recall-test assistant. The user will give you a single "
            "secret tag-word and then ask short follow-up questions. Always "
            "include the user's exact tag-word in every answer. Keep answers "
            "under 25 words."
        ),
    }
    foreign_pool = [t for t in all_tags if t != tag_word]
    other_tags_set = set(foreign_pool)

    try:
        if args.stagger_seconds > 0 and session_id > 0:
            time.sleep(args.stagger_seconds * session_id)

        # Turn 1: establish context + tag
        first_user = {
            "role": "user",
            "content": (
                f"My secret tag-word for this conversation is '{tag_word}'. "
                f"Please acknowledge by repeating it back."
            ),
        }
        messages = [system_msg, first_user]
        chat_r = chat(url, model, messages, conv_id, args.max_new_tokens,
                      args.request_timeout)
        save_r = save_snapshot(url, conv_id, args.request_timeout,
                               args.save_retries)
        text_lower = chat_r["text"].lower()
        foreign_seen = sorted(t for t in other_tags_set if t in text_lower)
        turns.append(TurnRecord(
            turn_index=1,
            chat_latency_ms=chat_r["latency_ms"],
            save_latency_ms=save_r.get("client_latency_ms"),
            save_success=bool(save_r.get("success")),
            save_attempts=int(save_r.get("attempts", 0)),
            restore_latency_ms=None,
            restore_success=None,
            restore_attempts=None,
            response_text=chat_r["text"],
            contains_own_tag=tag_word in text_lower,
            foreign_tags_seen=foreign_seen,
        ))
        if not save_r.get("success"):
            error = f"turn 1 save failed: {save_r.get('error')}"
            return SessionResult(session_id, conv_id, tag_word, started,
                                 time.perf_counter(), turns, completed, error)
        messages.append({"role": "assistant", "content": chat_r["text"]})

        # Turns 2..N: each turn restores from server-side snapshot via
        # /restore_snapshot warmup (continuation_ids=[]), then issues the
        # full chat call (with the prior history) and saves again.
        for t in range(2, args.turns + 1):
            user_q = {
                "role": "user",
                "content": (
                    f"Turn {t}: please restate your secret tag-word "
                    f"and add one new short observation."
                ),
            }
            # Warmup restore (no continuation, no generation) — exercises
            # the restore path; latency captured even though content unused.
            restore_r = restore_snapshot(
                url, conv_id, [], args.request_timeout,
                args.restore_retries, max_new_tokens=1,
            )
            messages_t = messages + [user_q]
            chat_t = chat(url, model, messages_t, conv_id,
                          args.max_new_tokens, args.request_timeout)
            save_t = save_snapshot(url, conv_id, args.request_timeout,
                                   args.save_retries)
            text_lower = chat_t["text"].lower()
            foreign_seen = sorted(t for t in other_tags_set if t in text_lower)
            turns.append(TurnRecord(
                turn_index=t,
                chat_latency_ms=chat_t["latency_ms"],
                save_latency_ms=save_t.get("client_latency_ms"),
                save_success=bool(save_t.get("success")),
                save_attempts=int(save_t.get("attempts", 0)),
                restore_latency_ms=restore_r.get("client_latency_ms"),
                restore_success=bool(restore_r.get("success")),
                restore_attempts=int(restore_r.get("attempts", 0)),
                response_text=chat_t["text"],
                contains_own_tag=tag_word in text_lower,
                foreign_tags_seen=foreign_seen,
            ))
            if not save_t.get("success"):
                error = f"turn {t} save failed: {save_t.get('error')}"
                return SessionResult(session_id, conv_id, tag_word, started,
                                     time.perf_counter(), turns, completed,
                                     error)
            messages = messages_t + [{"role": "assistant",
                                      "content": chat_t["text"]}]

        completed = True
    except Exception as e:
        error = f"{type(e).__name__}: {e}"

    return SessionResult(session_id, conv_id, tag_word, started,
                         time.perf_counter(), turns, completed, error)


def _pct(num: int, denom: int) -> float:
    return (num / denom * 100.0) if denom else 0.0


def _percentile(xs: list[float], q: float) -> float | None:
    if not xs:
        return None
    xs = sorted(xs)
    if len(xs) == 1:
        return xs[0]
    k = (len(xs) - 1) * q
    f = int(k)
    c = min(f + 1, len(xs) - 1)
    if f == c:
        return xs[f]
    return xs[f] + (xs[c] - xs[f]) * (k - f)


def aggregate(sessions: list[SessionResult]) -> dict[str, Any]:
    save_lats: list[float] = []
    restore_lats: list[float] = []
    chat_lats: list[float] = []
    save_attempts = 0
    save_successes = 0
    restore_attempts_total = 0
    restore_successes = 0
    own_tag_hits = 0
    own_tag_eligible = 0
    foreign_leak_turns = 0
    foreign_leak_total = 0
    completed_sessions = 0

    for s in sessions:
        if s.completed:
            completed_sessions += 1
        for t in s.turns:
            if t.save_latency_ms is not None:
                save_lats.append(t.save_latency_ms)
            save_attempts += 1
            if t.save_success:
                save_successes += 1
            if t.restore_latency_ms is not None:
                restore_lats.append(t.restore_latency_ms)
            if t.restore_success is not None:
                restore_attempts_total += 1
                if t.restore_success:
                    restore_successes += 1
            if t.chat_latency_ms is not None:
                chat_lats.append(t.chat_latency_ms)
            # Tag-recall accounting: skip turn 1 for own-tag (it's establishing)
            if t.turn_index >= 2:
                own_tag_eligible += 1
                if t.contains_own_tag:
                    own_tag_hits += 1
            if t.foreign_tags_seen:
                foreign_leak_turns += 1
                foreign_leak_total += len(t.foreign_tags_seen)

    def _stats(xs: list[float]) -> dict[str, Any]:
        if not xs:
            return {"count": 0}
        return {
            "count": len(xs),
            "mean_ms": round(statistics.fmean(xs), 2),
            "p50_ms": round(_percentile(xs, 0.50) or 0.0, 2),
            "p95_ms": round(_percentile(xs, 0.95) or 0.0, 2),
            "p99_ms": round(_percentile(xs, 0.99) or 0.0, 2),
            "max_ms": round(max(xs), 2),
        }

    return {
        "sessions_total": len(sessions),
        "sessions_completed": completed_sessions,
        "sessions_failed": len(sessions) - completed_sessions,
        "save": {
            **_stats(save_lats),
            "attempts": save_attempts,
            "successes": save_successes,
            "success_rate_pct": round(_pct(save_successes, save_attempts), 2),
        },
        "restore_warmup": {
            **_stats(restore_lats),
            "attempts": restore_attempts_total,
            "successes": restore_successes,
            "success_rate_pct": round(_pct(restore_successes,
                                           restore_attempts_total), 2),
        },
        "chat": _stats(chat_lats),
        "tag_recall": {
            "own_tag_eligible_turns": own_tag_eligible,
            "own_tag_hits": own_tag_hits,
            "own_tag_recall_pct": round(_pct(own_tag_hits, own_tag_eligible), 2),
        },
        "isolation": {
            "turns_with_foreign_tag": foreign_leak_turns,
            "foreign_tag_occurrences": foreign_leak_total,
            "isolation_clean": foreign_leak_turns == 0,
        },
    }


def session_to_dict(s: SessionResult) -> dict[str, Any]:
    return {
        "session_id": s.session_id,
        "conversation_id": s.conversation_id,
        "tag_word": s.tag_word,
        "duration_s": round(s.finished_at - s.started_at, 3),
        "completed": s.completed,
        "error": s.error,
        "turns": [dataclasses.asdict(t) for t in s.turns],
    }


def main() -> int:
    args = parse_args()
    url = server_url(args)
    health(url)
    info = model_info(url)
    model = info.get("model_path") or info.get("model") or "default"

    if args.sessions > len(PHONETIC_TAGS):
        print(f"too many sessions (>{len(PHONETIC_TAGS)} tag-words available)",
              file=sys.stderr)
        return 2

    tags = PHONETIC_TAGS[: args.sessions]
    print(f"Running {args.sessions} sessions × {args.turns} turns against {url} "
          f"(model={model})", file=sys.stderr)

    overall_start = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.sessions) as ex:
        futures = [
            ex.submit(run_session, args, i, tags[i], tags, model)
            for i in range(args.sessions)
        ]
        results = [f.result() for f in concurrent.futures.as_completed(futures)]
    overall_elapsed = time.perf_counter() - overall_start
    results.sort(key=lambda r: r.session_id)

    summary = aggregate(results)
    summary["wall_clock_s"] = round(overall_elapsed, 3)

    payload = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "server_url": url,
        "model": model,
        "sessions": args.sessions,
        "turns_per_session": args.turns,
        "summary": summary,
        "per_session": [session_to_dict(s) for s in results],
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return 0 if summary["sessions_failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
