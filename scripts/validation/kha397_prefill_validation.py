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
    p.add_argument(
        "--mode",
        choices=["warm", "cold"],
        default="warm",
        help="warm: full setup + BASELINE + CONTROL + RC (3a) + save state. "
             "cold: read state.json, run CONTROL_COLD and RC_COLD (3b) only.",
    )
    p.add_argument(
        "--state",
        default="/tmp/kha397/state.json",
        help="State file shared between warm and cold runs.",
    )
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


def run_warm(args, tok, model_name, url) -> int:

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

    state_path = Path(args.state)
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(json.dumps({
        "ts_warm": result["ts_utc"],
        "model_name": model_name,
        "setup_conv_id": setup_conv_id,
        "R": R_text,
        "continuation_ids": continuation_ids,
        "warm": {
            "baseline_text": baseline_resp["text"],
            "baseline_first_token_logprob": first_tok_lp,
            "baseline_full_logprobs": (
                baseline_resp["logprobs"].get("content")
                if baseline_resp["logprobs"] else None
            ),
            "control_output_ids": control_ids,
            "control_output_text": control_text,
            "control_latency_ms": control_resp["client_latency_ms"],
            "rc_output_ids": rc_ids,
            "rc_output_text": rc_text,
            "rc_latency_ms": rc_resp["client_latency_ms"],
            "verdict": verdict,
            "verdict_reason": verdict_reason,
        },
    }, indent=2, default=str))
    print(f"[STATE] {state_path}  (for cold-phase run)")
    return 0 if verdict != "AMBIGUOUS" else 2


def run_cold(args, tok, model_name, url) -> int:
    """Cold-radix arm (3b): same snapshot, fresh server process.

    Assumes:
      - run_warm has already executed and wrote state.json
      - the server has been restarted between runs (this script does not
        manage the server lifecycle)
      - the on-disk snapshot at /workspace/snapshots/<conv_id>/ persists
    """
    state_path = Path(args.state)
    if not state_path.exists():
        print(f"[FAIL] state file not found: {state_path}. "
              "Run --mode warm first.")
        return 1
    state = json.loads(state_path.read_text())
    setup_conv_id = state["setup_conv_id"]
    R_text = state["R"]
    warm = state["warm"]
    continuation_ids = state["continuation_ids"]

    if state["model_name"] != model_name:
        print(f"[FAIL] model mismatch — warm used {state['model_name']}, "
              f"cold server reports {model_name}")
        return 1
    print(f"[cold] reusing snapshot conv_id={setup_conv_id} R={R_text!r}")

    # Verify snapshot still on disk via /list_snapshots on the fresh server.
    snap_info = get_snapshot_info(url, setup_conv_id, args.request_timeout)
    print(f"[cold] snapshot lookup on fresh server: {snap_info}")
    if not snap_info.get("present"):
        print("[FAIL] snapshot not visible on fresh server — check "
              "--snapshot-dir flag and on-disk persistence.")
        return 1

    # ------------------------------------------------------------------
    # STEP 3'  CONTROL_COLD — restore with continuation_ids=[] on fresh
    # server. Should produce the same output as warm CONTROL if SSM state
    # restoration is independent of radix state.
    # ------------------------------------------------------------------
    print(f"\n[COLD CONTROL] /restore_snapshot conv={setup_conv_id} "
          f"continuation_ids=[] max_new_tokens={MAX_NEW_TOKENS}")
    control_cold = restore_and_generate(
        url, setup_conv_id, [], MAX_NEW_TOKENS, args.request_timeout,
    )
    if not control_cold.get("success"):
        print(f"  [FAIL] COLD CONTROL: {control_cold}")
        return 1
    cc_ids = control_cold.get("output_ids") or []
    cc_text = control_cold.get("output_text") or ""
    print(f"  COLD CONTROL output_text: {cc_text!r}  output_ids[:8]={cc_ids[:8]}")

    # ------------------------------------------------------------------
    # STEP 4'  RC_COLD (3b) — restore with continuation_ids=M-encoded on
    # fresh server. The decisive cold-arm measurement.
    # ------------------------------------------------------------------
    print(f"\n[COLD RC (3b)] /restore_snapshot conv={setup_conv_id} "
          f"continuation_ids (len={len(continuation_ids)}) "
          f"max_new_tokens={MAX_NEW_TOKENS}")
    rc_cold = restore_and_generate(
        url, setup_conv_id, continuation_ids,
        MAX_NEW_TOKENS, args.request_timeout,
    )
    if not rc_cold.get("success"):
        print(f"  [FAIL] COLD RC: {rc_cold}")
        return 1
    rcc_ids = rc_cold.get("output_ids") or []
    rcc_text = rc_cold.get("output_text") or ""
    print(f"  COLD RC output_text: {rcc_text!r}  output_ids[:8]={rcc_ids[:8]}")

    # ------------------------------------------------------------------
    # Verdict — compare COLD RC to WARM RC, WARM CONTROL, COLD CONTROL,
    # and BASELINE.
    # ------------------------------------------------------------------
    warm_rc_ids = warm["rc_output_ids"]
    warm_control_ids = warm["control_output_ids"]
    baseline_text = warm["baseline_text"]
    cold_eq_warm = (rcc_ids[:MAX_NEW_TOKENS] == warm_rc_ids[:MAX_NEW_TOKENS])
    cold_eq_warm_control = (rcc_ids[:MAX_NEW_TOKENS]
                            == warm_control_ids[:MAX_NEW_TOKENS])
    cold_eq_cold_control = (rcc_ids[:MAX_NEW_TOKENS] == cc_ids[:MAX_NEW_TOKENS])
    cold_says_banana = "BANANA" in rcc_text.upper()
    baseline_says_banana = "BANANA" in baseline_text.upper()

    if cold_eq_warm and cold_says_banana:
        cold_verdict = "CORRECT"
        cold_reason = (
            "COLD RC produced identical output to WARM RC and contains the "
            "M-instructed answer. Cold-radix path is correct."
        )
    elif cold_eq_cold_control or cold_eq_warm_control:
        cold_verdict = "DEFECT_SKIPPED"
        cold_reason = (
            "COLD RC output identical to CONTROL — continuation_ids were "
            "skipped on the cold path."
        )
    elif not cold_eq_warm and cold_says_banana and baseline_says_banana:
        cold_verdict = "CORRECT_DIVERGENT"
        cold_reason = (
            "COLD RC differs from WARM RC at token level but still contains "
            "the M-instructed answer. Cold path is functionally correct but "
            "warm and cold are not bit-exact."
        )
    elif not cold_eq_warm and not cold_says_banana:
        cold_verdict = "DEFECT_CORRUPTED"
        cold_reason = (
            "COLD RC differs from WARM RC AND does NOT contain the M-instructed "
            "answer. Continuation appears corrupted on cold path — consistent "
            "with the history-double-prefill hypothesis from KHA-397 Part A."
        )
    else:
        cold_verdict = "AMBIGUOUS"
        cold_reason = (
            "COLD RC behavior does not cleanly match any of the reference "
            "patterns. Manual inspection required."
        )

    print(f"\n[COLD VERDICT] {cold_verdict}")
    print(f"  {cold_reason}")
    print(f"  cold_eq_warm        = {cold_eq_warm}")
    print(f"  cold_eq_warm_control= {cold_eq_warm_control}")
    print(f"  cold_eq_cold_control= {cold_eq_cold_control}")
    print(f"  cold_says_banana    = {cold_says_banana}")

    # Merge into out JSON (load warm result, add cold sections).
    out_path = Path(args.out)
    if out_path.exists():
        result = json.loads(out_path.read_text())
    else:
        result = {}
    result["ts_cold_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    result["cold_control"] = {
        "conversation_id": setup_conv_id,
        "continuation_ids": [],
        "output_text": cc_text,
        "output_ids": cc_ids,
        "token_count": control_cold.get("token_count"),
        "latency_ms": control_cold["client_latency_ms"],
        "snapshot_info_on_fresh_server": snap_info,
    }
    result["cold_restore_continue"] = {
        "conversation_id": setup_conv_id,
        "continuation_ids_len": len(continuation_ids),
        "output_text": rcc_text,
        "output_ids": rcc_ids,
        "token_count": rc_cold.get("token_count"),
        "latency_ms": rc_cold["client_latency_ms"],
    }
    result["cold_verdict"] = cold_verdict
    result["cold_verdict_reason"] = cold_reason
    result["cold_eq_warm"] = cold_eq_warm
    result["cold_eq_warm_control"] = cold_eq_warm_control
    result["cold_eq_cold_control"] = cold_eq_cold_control
    result["cold_says_banana"] = cold_says_banana
    out_path.write_text(json.dumps(result, indent=2, default=str))
    print(f"\n[OUT] {out_path}  (merged warm+cold)")
    return 0 if cold_verdict in ("CORRECT", "CORRECT_DIVERGENT") else 2


def main() -> int:
    args = parse_args()
    url = f"http://{args.host}:{args.port}"
    print(f"[setup] loading tokenizer from {args.model_path}")
    tok = AutoTokenizer.from_pretrained(args.model_path)
    model_info = requests.get(f"{url}/get_model_info", timeout=10).json()
    model_name = model_info["model_path"]
    print(f"[setup] server reports model_path={model_name}  mode={args.mode}")
    if args.mode == "warm":
        return run_warm(args, tok, model_name, url)
    return run_cold(args, tok, model_name, url)


if __name__ == "__main__":
    sys.exit(main())
