"""KHA-397 — Restore+continuation prefill validation.

Empirically determines whether continuation_ids sent to /restore_snapshot are
actually prefilled. Compares deterministic greedy outputs across three
conditions for a single conversation:

  BASELINE          /v1/chat/completions with [user:P, assistant:R, user:M]
                    (no snapshot machinery). Captures top-k logprobs of the
                    first generated token. Reference for "M was prefilled".

  CONTROL           /restore_snapshot of post-R snapshot with
                    continuation_ids=[]. Decodes from the restored Mamba state
                    with no new input. Reference for "M was skipped".

  RESTORE+CONTINUE  /restore_snapshot of post-R snapshot with continuation_ids
                    = chat-template encoding of [user:M, assistant prompt].
                    This is the test condition.

Verdict (decisive comparison):
  - RESTORE+CONTINUE matches BASELINE answer  → continuation prefilled → CORRECT
  - RESTORE+CONTINUE matches CONTROL          → continuation skipped   → DEFECT
  - Anything else                              → AMBIGUOUS

To isolate the prefill question from attention recall (Engram only preserves
SSM state across snapshots, not attention KV), the test message M is a
self-contained instruction that does NOT depend on recalling content from P
(e.g. "Now reply with only the word: BANANA"). A correct prefill yields the
instructed word; a skipped prefill cannot.
"""

from __future__ import annotations

import argparse
import json
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
P_CONTENT = "Hi. Just say 'OK'."
M_CONTENT = "Now reply with only the word: BANANA"

# How many tokens to decode in each condition. 12 is enough to see the model
# settle on a clear continuation while keeping the test cheap.
MAX_NEW_TOKENS = 12

# How many top-k logprobs to capture for the first generated token in BASELINE.
TOP_LOGPROBS = 10


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=30002)
    p.add_argument("--model-path", default="/workspace/models/granite-4.0-h-tiny")
    p.add_argument("--out", default="/tmp/kha397/granite-results.json")
    p.add_argument("--request-timeout", type=float, default=120.0)
    return p.parse_args()


def chat_with_logprobs(
    url: str,
    model: str,
    messages: list[dict[str, str]],
    rid: str,
    max_new_tokens: int,
    timeout: float,
    top_logprobs: int | None = None,
) -> dict[str, Any]:
    """OpenAI-compatible chat completion with optional top-k logprobs."""
    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": 0,
        "max_tokens": max_new_tokens,
        "stream": False,
        "rid": rid,
    }
    if top_logprobs is not None:
        payload["logprobs"] = True
        payload["top_logprobs"] = top_logprobs
    start = time.perf_counter()
    r = requests.post(f"{url}/v1/chat/completions", json=payload, timeout=timeout)
    elapsed_ms = (time.perf_counter() - start) * 1000
    r.raise_for_status()
    body = r.json()
    choice = body["choices"][0]
    text = choice["message"]["content"]
    logprobs_block = choice.get("logprobs")
    return {
        "text": text,
        "latency_ms": elapsed_ms,
        "logprobs": logprobs_block,
        "raw": body,
    }


def save_snapshot(url: str, conversation_id: str, timeout: float) -> dict[str, Any]:
    start = time.perf_counter()
    r = requests.post(
        f"{url}/save_snapshot",
        json={"conversation_id": conversation_id},
        timeout=timeout,
    )
    elapsed_ms = (time.perf_counter() - start) * 1000
    r.raise_for_status()
    body = r.json()
    body["client_latency_ms"] = elapsed_ms
    return body


def restore_and_generate(
    url: str,
    conversation_id: str,
    continuation_ids: list[int],
    max_new_tokens: int,
    timeout: float,
) -> dict[str, Any]:
    start = time.perf_counter()
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
    return body


def get_snapshot_info(url: str, conversation_id: str, timeout: float) -> dict[str, Any]:
    """List snapshots for a conversation and return the most-recent's metadata."""
    r = requests.post(
        f"{url}/list_snapshots",
        json={"conversation_id": conversation_id},
        timeout=timeout,
    )
    r.raise_for_status()
    body = r.json()
    snaps = body.get("snapshots") or []
    if not snaps:
        return {"present": False}
    last = snaps[-1]
    return {"present": True, "metadata": last, "count": len(snaps)}


def encode_user_turn_with_assistant_prompt(
    tok: AutoTokenizer, user_content: str
) -> list[int]:
    """Tokenize a single user message + assistant generation prompt.

    This is the canonical chat-template encoding for a "fresh" user turn — the
    bytes the model needs to see to advance from the end of a prior assistant
    turn to the start of a new assistant response.
    """
    formatted = tok.apply_chat_template(
        [{"role": "user", "content": user_content}],
        tokenize=False,
        add_generation_prompt=True,
    )
    return tok.encode(formatted, add_special_tokens=False)


def main() -> int:
    args = parse_args()
    url = f"http://{args.host}:{args.port}"

    print(f"[setup] loading tokenizer from {args.model_path}")
    tok = AutoTokenizer.from_pretrained(args.model_path)

    # Use the model's loaded path as the served-model-name (default behavior
    # when --served-model-name is not passed).
    model_info = requests.get(f"{url}/get_model_info", timeout=10).json()
    model_name = model_info["model_path"]
    print(f"[setup] server reports model_path={model_name}")

    # ------------------------------------------------------------------
    # STEP 1 — Establish R by running [system, user:P] once. R is the
    # response that defines the snapshot's post-state. We need it to be
    # deterministic so BASELINE can replay it.
    # ------------------------------------------------------------------
    setup_conv_id = f"kha397-setup-{uuid.uuid4().hex[:8]}"
    print(f"\n[STEP 1] establishing R via chat (rid={setup_conv_id})")
    setup_messages = [
        {"role": "system", "content": SYSTEM_MSG},
        {"role": "user", "content": P_CONTENT},
    ]
    setup_resp = chat_with_logprobs(
        url, model_name, setup_messages, setup_conv_id,
        MAX_NEW_TOKENS, args.request_timeout,
    )
    R_text = setup_resp["text"]
    print(f"  R = {R_text!r}  (latency={setup_resp['latency_ms']:.0f}ms)")

    # Save snapshot at end of R for CONTROL and RESTORE+CONTINUE to share.
    save_resp = save_snapshot(url, setup_conv_id, args.request_timeout)
    if not save_resp.get("success"):
        print(f"  [FAIL] save_snapshot: {save_resp}")
        return 1
    print(f"  snapshot saved: {save_resp.get('message')}  "
          f"(latency={save_resp['client_latency_ms']:.0f}ms)")

    snap_info = get_snapshot_info(url, setup_conv_id, args.request_timeout)
    print(f"  snapshot info: {snap_info}")

    # ------------------------------------------------------------------
    # STEP 2 — BASELINE: full 3-message chat with logprobs. Reference for
    # "M was prefilled".
    # ------------------------------------------------------------------
    baseline_conv_id = f"kha397-baseline-{uuid.uuid4().hex[:8]}"
    print(f"\n[STEP 2] BASELINE chat (rid={baseline_conv_id}) "
          f"— [user:P, assistant:R, user:M], temp=0, top_logprobs={TOP_LOGPROBS}")
    baseline_messages = [
        {"role": "system", "content": SYSTEM_MSG},
        {"role": "user", "content": P_CONTENT},
        {"role": "assistant", "content": R_text},
        {"role": "user", "content": M_CONTENT},
    ]
    baseline_resp = chat_with_logprobs(
        url, model_name, baseline_messages, baseline_conv_id,
        MAX_NEW_TOKENS, args.request_timeout,
        top_logprobs=TOP_LOGPROBS,
    )
    print(f"  BASELINE text: {baseline_resp['text']!r}  "
          f"(latency={baseline_resp['latency_ms']:.0f}ms)")
    first_tok_lp = None
    if baseline_resp["logprobs"] and baseline_resp["logprobs"].get("content"):
        first_tok_lp = baseline_resp["logprobs"]["content"][0]
        print(f"  BASELINE first token: {first_tok_lp.get('token')!r} "
              f"logprob={first_tok_lp.get('logprob')}")

    # ------------------------------------------------------------------
    # STEP 3 — CONTROL: restore from snapshot with continuation_ids=[].
    # No new input. Decodes purely from restored SSM state.
    # ------------------------------------------------------------------
    print(f"\n[STEP 3] CONTROL: /restore_snapshot conv={setup_conv_id} "
          f"continuation_ids=[] max_new_tokens={MAX_NEW_TOKENS}")
    control_resp = restore_and_generate(
        url, setup_conv_id, [], MAX_NEW_TOKENS, args.request_timeout,
    )
    if not control_resp.get("success"):
        print(f"  [FAIL] CONTROL restore: {control_resp}")
        return 1
    print(f"  CONTROL output_text: {control_resp.get('output_text')!r}  "
          f"output_ids[:8]={control_resp.get('output_ids', [])[:8]}")

    # Re-save snapshot — the prior restore consumed the live request's state,
    # but the on-disk snapshot file is still there; conv lookup should hit it.
    # (No re-save needed: load_snapshot reads from disk for each restore call.)

    # ------------------------------------------------------------------
    # STEP 4 — RESTORE+CONTINUE: restore from snapshot with continuation_ids
    # = chat-template encoding of user:M + assistant prompt.
    # ------------------------------------------------------------------
    continuation_ids = encode_user_turn_with_assistant_prompt(tok, M_CONTENT)
    print(f"\n[STEP 4] RESTORE+CONTINUE: /restore_snapshot conv={setup_conv_id} "
          f"continuation_ids (len={len(continuation_ids)}) max_new_tokens={MAX_NEW_TOKENS}")
    print(f"  continuation_ids[:16] = {continuation_ids[:16]}")
    print(f"  continuation decoded   = "
          f"{tok.decode(continuation_ids, skip_special_tokens=False)!r}")
    rc_resp = restore_and_generate(
        url, setup_conv_id, continuation_ids,
        MAX_NEW_TOKENS, args.request_timeout,
    )
    if not rc_resp.get("success"):
        print(f"  [FAIL] RESTORE+CONTINUE: {rc_resp}")
        return 1
    print(f"  RESTORE+CONTINUE output_text: {rc_resp.get('output_text')!r}  "
          f"output_ids[:8]={rc_resp.get('output_ids', [])[:8]}")

    # ------------------------------------------------------------------
    # STEP 5 — Verdict
    # ------------------------------------------------------------------
    baseline_text = baseline_resp["text"]
    control_text = control_resp.get("output_text") or ""
    rc_text = rc_resp.get("output_text") or ""
    control_ids = control_resp.get("output_ids") or []
    rc_ids = rc_resp.get("output_ids") or []

    # Decisive: did RESTORE+CONTINUE's output match CONTROL's (skipped) or
    # diverge (some processing happened)?
    rc_matches_control = (rc_ids[:MAX_NEW_TOKENS] == control_ids[:MAX_NEW_TOKENS])
    rc_says_banana = "BANANA" in rc_text.upper()
    baseline_says_banana = "BANANA" in baseline_text.upper()

    if rc_matches_control and not rc_says_banana:
        verdict = "DEFECT"
        verdict_reason = (
            "RESTORE+CONTINUE output_ids identical to CONTROL output_ids — "
            "the continuation_ids had no effect on generation. Prefill was "
            "skipped."
        )
    elif rc_says_banana and baseline_says_banana:
        verdict = "CORRECT"
        verdict_reason = (
            "RESTORE+CONTINUE produced 'BANANA' (the M-instructed answer), "
            "matching BASELINE. Continuation tokens were prefilled."
        )
    elif rc_says_banana and not baseline_says_banana:
        verdict = "AMBIGUOUS"
        verdict_reason = (
            "RESTORE+CONTINUE produced 'BANANA' but BASELINE did not — "
            "test prompt did not establish a clean reference. Re-evaluate."
        )
    else:
        verdict = "AMBIGUOUS"
        verdict_reason = (
            "RESTORE+CONTINUE differs from CONTROL but does not match the "
            "BASELINE answer. Continuation may have affected state without "
            "yielding the instructed output — possible partial prefill or "
            "different stop behavior."
        )

    print(f"\n[VERDICT] {verdict}")
    print(f"  {verdict_reason}")
    print(f"  rc_matches_control = {rc_matches_control}")
    print(f"  baseline says BANANA = {baseline_says_banana}")
    print(f"  rc says BANANA       = {rc_says_banana}")

    result = {
        "ts_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "model_path": model_name,
        "system_msg": SYSTEM_MSG,
        "P": P_CONTENT,
        "R": R_text,
        "M": M_CONTENT,
        "max_new_tokens": MAX_NEW_TOKENS,
        "top_logprobs_k": TOP_LOGPROBS,
        "setup": {
            "rid": setup_conv_id,
            "chat_latency_ms": setup_resp["latency_ms"],
            "save_latency_ms": save_resp["client_latency_ms"],
            "snapshot_info": snap_info,
        },
        "baseline": {
            "rid": baseline_conv_id,
            "messages": baseline_messages,
            "text": baseline_resp["text"],
            "latency_ms": baseline_resp["latency_ms"],
            "first_token_logprob": first_tok_lp,
            "full_logprobs_content": (
                baseline_resp["logprobs"].get("content")
                if baseline_resp["logprobs"] else None
            ),
        },
        "control": {
            "conversation_id": setup_conv_id,
            "continuation_ids": [],
            "output_text": control_text,
            "output_ids": control_ids,
            "token_count": control_resp.get("token_count"),
            "latency_ms": control_resp["client_latency_ms"],
        },
        "restore_continue": {
            "conversation_id": setup_conv_id,
            "continuation_ids_len": len(continuation_ids),
            "continuation_ids_head": continuation_ids[:32],
            "continuation_decoded": tok.decode(
                continuation_ids, skip_special_tokens=False),
            "output_text": rc_text,
            "output_ids": rc_ids,
            "token_count": rc_resp.get("token_count"),
            "latency_ms": rc_resp["client_latency_ms"],
        },
        "verdict": verdict,
        "verdict_reason": verdict_reason,
        "rc_matches_control": rc_matches_control,
        "baseline_says_banana": baseline_says_banana,
        "rc_says_banana": rc_says_banana,
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2, default=str))
    print(f"\n[OUT] {out_path}")
    return 0 if verdict != "AMBIGUOUS" else 2


if __name__ == "__main__":
    sys.exit(main())
