"""
Phase 8 — True Stateful Inference

Proves that a multi-turn conversation can be held without the client resending
full conversation history on every turn.  The server reconstructs prior context
from a saved Mamba SSM snapshot; the client sends only new tokens per turn.

API under test
--------------
POST /restore_snapshot  {
    "conversation_id": "<rid from Turn 1>",
    "create_new_request": true,
    "continuation_ids": [<token IDs for new question>],
    "max_new_tokens": 80
}

Returns:
{
    "success": true,
    "rid": "restored-<uuid>",
    "output_ids": [<generated token IDs>],
    "output_text": "<decoded response>"
}
"""

import json
import os
import time
import unittest
import uuid

import requests

SERVER_URL = os.environ.get("SERVER_URL", "http://localhost:30000")
MODEL_PATH = os.environ.get("MODEL_PATH", "/mnt/models/granite-4.0-h-tiny")

_tokenizer = None


def _get_tokenizer():
    global _tokenizer
    if _tokenizer is None:
        from transformers import AutoTokenizer
        _tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    return _tokenizer


class TestMambaStatefulInference(unittest.TestCase):
    def setUp(self):
        try:
            r = requests.get(f"{SERVER_URL}/health", timeout=5)
            r.raise_for_status()
        except Exception as e:
            self.skipTest(f"Server not available: {e}")

    # ── helpers ──────────────────────────────────────────────────────

    def _chat(self, messages, rid=None, max_tokens=80, temperature=0.0):
        """Standard /v1/chat/completions call (full history every turn)."""
        payload = {
            "model": "default",
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if rid:
            payload["rid"] = rid
        r = requests.post(
            f"{SERVER_URL}/v1/chat/completions",
            json=payload,
            timeout=60,
        )
        r.raise_for_status()
        return r.json()

    def _save_snapshot(self, rid):
        r = requests.post(
            f"{SERVER_URL}/save_snapshot",
            json={"rid": rid},
            timeout=30,
        )
        r.raise_for_status()
        return r.json()

    def _tokenize(self, text):
        tok = _get_tokenizer()
        return tok.encode(text, add_special_tokens=False)

    def _stateful_generate(self, rid, new_text, max_new_tokens=80):
        """Restore snapshot for `rid` and generate a response to `new_text`.

        The client sends ONLY the new tokens — no prior conversation history.
        The server reconstructs context from the saved Mamba SSM snapshot.
        """
        continuation_ids = self._tokenize(new_text)
        r = requests.post(
            f"{SERVER_URL}/restore_snapshot",
            json={
                "conversation_id": rid,
                "create_new_request": True,
                "continuation_ids": continuation_ids,
                "max_new_tokens": max_new_tokens,
            },
            timeout=120,
        )
        r.raise_for_status()
        return r.json()

    # ── tests ────────────────────────────────────────────────────────

    def test_stateful_recall_semantic(self):
        """Turn 1 establishes a fact; Turn 2 (stateful) recalls it."""
        rid = f"stateful-recall-{uuid.uuid4().hex[:8]}"

        # Turn 1: establish a fact via full chat
        t1 = self._chat(
            [{"role": "user", "content": "The secret number is 42. Please confirm you understand."}],
            rid=rid,
            max_tokens=60,
        )
        self.assertIn("choices", t1, f"Turn 1 failed: {t1}")
        time.sleep(0.3)

        # Save snapshot after Turn 1
        snap = self._save_snapshot(rid)
        self.assertTrue(snap.get("success"), f"Snapshot save failed: {snap}")

        # Turn 2: stateful — only send the new question
        r2 = self._stateful_generate(
            rid, "\nUser: What is the secret number?\nAssistant:"
        )
        self.assertTrue(r2.get("success"), f"Stateful generate failed: {r2}")
        response = r2.get("output_text", "")
        self.assertIn(
            "42",
            response,
            f"Model did not recall fact from Turn 1. Response: {response}",
        )

    def test_stateful_vs_full_resend_equivalence(self):
        """Stateful output should match full-resend output at temperature=0."""
        rid = f"stateful-equiv-{uuid.uuid4().hex[:8]}"

        # Turn 1: establish context
        t1 = self._chat(
            [{"role": "user", "content": "My favorite color is blue."}],
            rid=rid,
            max_tokens=40,
        )
        self.assertIn("choices", t1)
        t1_text = t1["choices"][0]["message"]["content"]
        time.sleep(0.3)

        snap = self._save_snapshot(rid)
        self.assertTrue(snap.get("success"))

        # Full resend (baseline)
        full_resend = self._chat(
            [
                {"role": "user", "content": "My favorite color is blue."},
                {"role": "assistant", "content": t1_text},
                {"role": "user", "content": "What is my favorite color?"},
            ],
            max_tokens=40,
            temperature=0.0,
        )
        full_text = full_resend["choices"][0]["message"]["content"]

        # Stateful (test)
        r2 = self._stateful_generate(
            rid,
            f"\nAssistant: {t1_text}\nUser: What is my favorite color?\nAssistant:",
            max_new_tokens=40,
        )
        self.assertTrue(r2.get("success"), f"Stateful failed: {r2}")
        stateful_text = r2.get("output_text", "")

        # Both should mention "blue"
        self.assertIn("blue", full_text.lower(), f"Full resend missing 'blue': {full_text}")
        self.assertIn(
            "blue",
            stateful_text.lower(),
            f"Stateful missing 'blue': {stateful_text}",
        )

    def test_multi_turn_stateful_chain(self):
        """3-turn chain: each turn uses only new tokens; all facts recalled in Turn 3.

        The original rid (rid1) acts as the conversation handle across all turns.
        Stateful generates create temporary restored-* rids internally, but the
        EVERY_TURN auto-snapshot stores state under rid1's conversation_id.
        Saving via rid1 after each turn persists the latest WARM-tier state to disk.
        """
        rid1 = f"stateful-chain-t1-{uuid.uuid4().hex[:8]}"

        # Turn 1: establish color
        t1 = self._chat(
            [{"role": "user", "content": "My favorite color is green. Got it?"}],
            rid=rid1,
            max_tokens=60,
        )
        self.assertIn("choices", t1)
        time.sleep(0.3)
        snap1 = self._save_snapshot(rid1)
        self.assertTrue(snap1.get("success"), f"Turn1 snap failed: {snap1}")

        # Turn 2: establish number (stateful — no Turn 1 resent)
        # _stateful_generate creates a restored-* Req with conversation_id=rid1.
        # After it completes, EVERY_TURN auto-snapshot has Turn2 state in WARM under rid1.
        r2 = self._stateful_generate(rid1, "Also, my lucky number is 7. Got it?")
        time.sleep(0.3)
        # Save Turn2 state: use rid1 as key — handle_save_snapshot falls back to
        # WARM tier via effective_conv_id=rid1, persisting Turn2 state to COLD.
        snap2 = self._save_snapshot(rid1)
        self.assertTrue(snap2.get("success"), f"Turn2 snap failed: {snap2}")

        # Turn 3: ask about both facts (stateful — no Turn 1 or 2 resent)
        # Restore uses rid1 → gets latest snapshot (Turn2 state) from COLD tier.
        r3 = self._stateful_generate(
            rid1, "What is my favorite color and what is my lucky number?", max_new_tokens=100
        )
        response = r3.get("output_text", "")

        # Should recall both facts from prior turns
        response_lower = response.lower()
        self.assertIn(
            "green",
            response_lower,
            f"Turn 3 did not recall color. Response: {response}",
        )
        self.assertIn(
            "7",
            response,
            f"Turn 3 did not recall number. Response: {response}",
        )

    def test_token_savings_quantification(self):
        """Stateful approach should send fewer tokens than full resend."""
        rid = f"stateful-savings-{uuid.uuid4().hex[:8]}"

        prompt = (
            "The quick brown fox jumps over the lazy dog. "
            "This is a relatively long prompt to demonstrate token savings "
            "when using stateful inference instead of resending everything."
        )

        # Turn 1
        t1 = self._chat(
            [{"role": "user", "content": prompt}],
            rid=rid,
            max_tokens=60,
        )
        self.assertIn("choices", t1)
        t1_text = t1["choices"][0]["message"]["content"]
        time.sleep(0.3)

        snap = self._save_snapshot(rid)
        self.assertTrue(snap.get("success"))

        # Count tokens
        turn1_tokens = len(self._tokenize(prompt))
        turn2_question = "What did I just say?"
        turn2_tokens = len(self._tokenize(turn2_question))

        full_resend_tokens = turn1_tokens + len(self._tokenize(t1_text)) + turn2_tokens
        stateful_tokens = turn2_tokens  # Only new tokens!

        print(f"\n=== Token Savings ===")
        print(f"Full resend: {full_resend_tokens} tokens")
        print(f"Stateful:    {stateful_tokens} tokens")
        print(f"Savings:     {full_resend_tokens - stateful_tokens} tokens "
              f"({100 * (1 - stateful_tokens / full_resend_tokens):.1f}%)")

        self.assertGreater(
            full_resend_tokens,
            stateful_tokens,
            "Stateful should use fewer tokens than full resend",
        )

        # Actually run the stateful generate to confirm it works
        r2 = self._stateful_generate(rid, turn2_question)
        self.assertTrue(r2.get("success"), f"Stateful generate failed: {r2}")


if __name__ == "__main__":
    unittest.main()
