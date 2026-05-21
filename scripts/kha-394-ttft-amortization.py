#!/usr/bin/env python3
"""KHA-394 TTFT and compute-amortization probe.

The script expects an Engram server with snapshot persistence enabled. It keeps
the metric intentionally narrow:

1. Send a long-context chat request with streaming enabled and measure time to
   the first generated token.
2. Persist that conversation state as a snapshot.
3. Restore the snapshot and generate one token with only a short continuation.
   The restore endpoint is not streaming, so this is reported as
   restore-plus-one-token latency rather than streaming TTFT.
4. Estimate prefill work avoided over a multi-turn script by counting the long
   context tokens that do not need to be replayed on each follow-up turn.

Example:
    python scripts/kha-394-ttft-amortization.py \
      --server-url http://localhost:30000 \
      --model ibm-granite/granite-4.0-h-tiny \
      --target-words 50000 \
      --turns 50 \
      --active-params 4e9 \
      --output-json test/phases/results/kha-394-h100-20260521.json
"""

from __future__ import annotations

import argparse
import json
import math
import time
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests
from transformers import AutoTokenizer

DEFAULT_MODEL = "ibm-granite/granite-4.0-h-tiny"

CONTEXT_PARAGRAPHS = [
    (
        "The Amazon River is the largest river by discharge volume of water in "
        "the world. Its headwaters include high Andean tributaries in Peru, "
        "and its basin spans a huge portion of South America."
    ),
    (
        "The Great Barrier Reef is the world's largest coral reef system, "
        "stretching for more than 2,300 kilometers off the coast of Queensland, "
        "Australia, and supporting thousands of marine species."
    ),
    (
        "Mount Everest is Earth's highest mountain above sea level. The "
        "China-Nepal border crosses its summit, and its current official "
        "elevation is 8,848.86 meters."
    ),
    (
        "The International Space Station orbits Earth at roughly 420 kilometers "
        "altitude and travels about 28,000 kilometers per hour, completing more "
        "than fifteen orbits per day."
    ),
    (
        "Jupiter is the largest planet in the Solar System. Its Great Red Spot "
        "is a long-lived storm, and the planet's mass exceeds that of all other "
        "planets combined by more than a factor of two."
    ),
]


@dataclass
class TokenWork:
    context_tokens: int
    continuation_tokens_per_turn: int
    turns: int
    stateless_prefill_tokens: int
    engram_prefill_tokens: int
    skipped_prefill_tokens: int
    token_reduction_pct: float
    assumed_active_params: float | None
    estimated_flops_saved: float | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server-url", default="http://localhost:30000")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--target-words", type=int, default=50_000)
    parser.add_argument("--turns", type=int, default=50)
    parser.add_argument("--active-params", type=float, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--request-timeout", type=float, default=900.0)
    parser.add_argument("--conversation-id", default=None)
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def load_tokenizer(model: str):
    return AutoTokenizer.from_pretrained(model)


def build_context(target_words: int) -> str:
    words: list[str] = []
    idx = 0
    while len(words) < target_words:
        words.extend(CONTEXT_PARAGRAPHS[idx % len(CONTEXT_PARAGRAPHS)].split())
        idx += 1
    return " ".join(words[:target_words])


def chat_messages(context: str) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": (
                "You are a precise assistant. Read the reference material and "
                "answer only the final instruction."
            ),
        },
        {
            "role": "user",
            "content": (
                f"{context}\n\n"
                "Final instruction: reply with one short sentence confirming "
                "that the reference text was processed."
            ),
        },
    ]


def encode_chat(tokenizer: Any, messages: list[dict[str, str]]) -> list[int]:
    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
        rendered = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        return tokenizer.encode(rendered, add_special_tokens=False)
    rendered = "\n".join(f"{m['role']}: {m['content']}" for m in messages)
    return tokenizer.encode(rendered, add_special_tokens=False)


def encode_continuation(tokenizer: Any, question: str) -> list[int]:
    messages = [{"role": "user", "content": question}]
    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
        rendered = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        return tokenizer.encode(rendered, add_special_tokens=False)
    return tokenizer.encode(question, add_special_tokens=False)


def check_health(server_url: str) -> None:
    response = requests.get(f"{server_url}/health", timeout=10)
    response.raise_for_status()


def stream_chat_ttft(
    server_url: str,
    model: str,
    messages: list[dict[str, str]],
    rid: str,
    max_new_tokens: int,
    timeout: float,
) -> dict[str, Any]:
    payload = {
        "model": model,
        "messages": messages,
        "temperature": 0,
        "max_tokens": max_new_tokens,
        "stream": True,
        "rid": rid,
    }
    start = time.perf_counter()
    first_token_at: float | None = None
    output_chunks: list[str] = []

    with requests.post(
        f"{server_url}/v1/chat/completions",
        json=payload,
        stream=True,
        timeout=timeout,
    ) as response:
        response.raise_for_status()
        for raw_line in response.iter_lines(decode_unicode=True):
            if not raw_line or not raw_line.startswith("data: "):
                continue
            data = raw_line.removeprefix("data: ").strip()
            if data == "[DONE]":
                break
            chunk = json.loads(data)
            # Check for error or non-token events
            if "error" in chunk:
                continue
            choices = chunk.get("choices")
            if not choices or not isinstance(choices, list) or len(choices) == 0:
                continue
            delta = choices[0].get("delta", {})
            text = delta.get("content", "")
            if text:
                if first_token_at is None:
                    first_token_at = time.perf_counter()
                output_chunks.append(text)

    end = time.perf_counter()
    if first_token_at is None:
        raise RuntimeError("stream completed without a content token")

    return {
        "ttft_ms": (first_token_at - start) * 1000,
        "total_latency_ms": (end - start) * 1000,
        "output_text": "".join(output_chunks),
    }


def save_snapshot(
    server_url: str, conversation_id: str, timeout: float
) -> dict[str, Any]:
    last_body: dict[str, Any] | None = None
    for _ in range(30):
        start = time.perf_counter()
        try:
            response = requests.post(
                f"{server_url}/save_snapshot",
                json={"conversation_id": conversation_id},
                timeout=timeout,
            )
            elapsed_ms = (time.perf_counter() - start) * 1000
            response.raise_for_status()
            body = response.json()
            body["client_latency_ms"] = elapsed_ms
            if body.get("success") is True:
                return body
            last_body = body
        except (requests.exceptions.RequestException, json.JSONDecodeError) as e:
            elapsed_ms = (time.perf_counter() - start) * 1000
            last_body = {
                "success": False,
                "error": str(e),
                "client_latency_ms": elapsed_ms,
            }
        time.sleep(1)
    raise RuntimeError(f"snapshot save did not succeed: {last_body}")


def restore_one_token(
    server_url: str,
    conversation_id: str,
    continuation_ids: list[int],
    timeout: float,
) -> dict[str, Any]:
    start = time.perf_counter()
    response = requests.post(
        f"{server_url}/restore_snapshot",
        json={
            "conversation_id": conversation_id,
            "create_new_request": True,
            "continuation_ids": continuation_ids,
            "max_new_tokens": 1,
        },
        timeout=timeout,
    )
    elapsed_ms = (time.perf_counter() - start) * 1000
    response.raise_for_status()
    body = response.json()
    if body.get("success") is not True:
        raise RuntimeError(f"restore-and-generate failed: {body}")
    body["restore_plus_one_token_ms"] = elapsed_ms
    return body


def compute_token_work(
    context_tokens: int,
    continuation_tokens: int,
    turns: int,
    active_params: float | None,
) -> TokenWork:
    # Stateless clients replay the full long context for every follow-up turn.
    # Engram restores the long-context state once and only prefills the short
    # continuation. The first long-context request is common to both paths.
    followup_turns = max(turns - 1, 0)
    stateless = context_tokens + followup_turns * (context_tokens + continuation_tokens)
    engram = context_tokens + followup_turns * continuation_tokens
    skipped = stateless - engram
    reduction = (skipped / stateless * 100) if stateless else 0.0
    flops = None
    if active_params is not None:
        # Decoder-only inference is commonly approximated as 2 * active
        # parameters per prefill token. This is a buyer-facing estimate, not a
        # profiler trace.
        flops = 2 * active_params * skipped
    return TokenWork(
        context_tokens=context_tokens,
        continuation_tokens_per_turn=continuation_tokens,
        turns=turns,
        stateless_prefill_tokens=stateless,
        engram_prefill_tokens=engram,
        skipped_prefill_tokens=skipped,
        token_reduction_pct=reduction,
        assumed_active_params=active_params,
        estimated_flops_saved=flops,
    )


def main() -> int:
    args = parse_args()
    tokenizer = load_tokenizer(args.model)
    context = build_context(args.target_words)
    messages = chat_messages(context)
    prompt_ids = encode_chat(tokenizer, messages)
    continuation_question = "Give a one-word acknowledgement of the restored context."
    continuation_ids = encode_continuation(tokenizer, continuation_question)
    token_work = compute_token_work(
        len(prompt_ids), len(continuation_ids), args.turns, args.active_params
    )

    result: dict[str, Any] = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model": args.model,
        "server_url": args.server_url,
        "target_words": args.target_words,
        "conversation_id": args.conversation_id or f"kha-394-{uuid.uuid4().hex[:10]}",
        "methodology": {
            "cold_ttft": "streaming /v1/chat/completions with the full long context",
            "restore_latency": (
                "/restore_snapshot create_new_request=true with max_new_tokens=1; "
                "reported separately because restore_snapshot is not streaming"
            ),
            "flops_estimate": (
                "2 * active_params * skipped_prefill_tokens when --active-params "
                "is supplied"
            ),
        },
        "token_work": asdict(token_work),
    }

    if args.dry_run:
        result["dry_run"] = True
    else:
        check_health(args.server_url)
        cold = stream_chat_ttft(
            args.server_url,
            args.model,
            messages,
            result["conversation_id"],
            args.max_new_tokens,
            args.request_timeout,
        )
        save = save_snapshot(
            args.server_url, result["conversation_id"], args.request_timeout
        )
        restore = restore_one_token(
            args.server_url,
            result["conversation_id"],
            continuation_ids,
            args.request_timeout,
        )
        result["cold_long_context"] = cold
        result["save_snapshot"] = save
        result["restore_generate"] = restore
        result["speedup_vs_cold_ttft"] = (
            cold["ttft_ms"] / restore["restore_plus_one_token_ms"]
            if restore["restore_plus_one_token_ms"]
            else math.inf
        )

    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
