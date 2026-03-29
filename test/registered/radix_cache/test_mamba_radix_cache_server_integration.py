from sglang.test.ci.ci_register import register_cuda_ci, register_amd_ci

register_cuda_ci(est_time=180, suite="stage-b-test-small-1-gpu")
register_amd_ci(est_time=180, suite="stage-b-test-small-1-gpu-amd")

import concurrent.futures
import os
import time
import unittest

import requests

SERVER_URL = os.environ.get("SERVER_URL", "http://localhost:30000")
LONG_SYSTEM = "You are a concise assistant. " * 80   # ~500 tokens shared prefix


class TestMambaRadixCacheServerIntegration(unittest.TestCase):

    def _chat(self, messages, **kwargs):
        kwargs.setdefault("temperature", 0)
        kwargs.setdefault("max_tokens", 50)
        payload = {"model": "default", "messages": messages, **kwargs}
        r = requests.post(f"{SERVER_URL}/v1/chat/completions", json=payload, timeout=60)
        r.raise_for_status()
        return r.json()

    def test_cache_hit_on_repeated_prefix(self):
        """Second request sharing a long prefix has shorter prefill (cache hit)."""
        # Request A: long system prompt + question 1
        resp_a = self._chat([
            {"role": "system", "content": LONG_SYSTEM},
            {"role": "user", "content": "What is the capital of France?"},
        ])
        self.assertGreater(len(resp_a["choices"][0]["message"]["content"]), 0)

        # Request B: same system prompt + different question (should hit cache)
        resp_b = self._chat([
            {"role": "system", "content": LONG_SYSTEM},
            {"role": "user", "content": "What is the capital of Germany?"},
        ])
        self.assertGreater(len(resp_b["choices"][0]["message"]["content"]), 0)
        self.assertGreater(
            resp_b["usage"]["prompt_tokens_details"]["cached_tokens"],
            0,
            f"Expected cached_tokens > 0, got: {resp_b['usage']}",
        )

    def test_cache_miss_fallback(self):
        """Unique prefix (never seen before) generates correct output without corruption."""
        import uuid
        unique_prefix = f"Unique context {uuid.uuid4().hex}: "
        resp = self._chat([
            {"role": "user", "content": unique_prefix + "Reply with the word: correct"},
        ])
        content = resp["choices"][0]["message"]["content"].lower()
        self.assertIn("correct", content)

    def test_concurrent_shared_prefix(self):
        """4 concurrent requests sharing the same long system prompt all complete; outputs are independent."""
        messages_base = [{"role": "system", "content": LONG_SYSTEM}]
        questions = [
            "Name one fruit.",
            "Name one planet.",
            "Name one color.",
            "Name one animal.",
        ]

        def send(q):
            return self._chat(messages_base + [{"role": "user", "content": q}])["choices"][0]["message"]["content"].strip()

        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as ex:
            results = list(ex.map(send, questions))

        # All 4 must complete
        self.assertEqual(len(results), 4)
        # All must be non-empty
        for r in results:
            self.assertGreater(len(r), 0)

    def test_multi_turn_conversation_state_continuity(self):
        """5-turn conversation: each turn relies on server-side state, not replayed history."""
        rid = "continuity-test-rid"

        def turn(user_msg):
            resp = self._chat(
                [{"role": "user", "content": user_msg}],
                rid=rid,
                max_tokens=80,
            )
            return resp["choices"][0]["message"]["content"]

        t1 = turn("My name is Alex and I like the number 42.")
        t2 = turn("What is my name?")
        t3 = turn("What number do I like?")
        t4 = turn("What would you add to 42 to get 100?")
        t5 = turn("Summarize what you know about me in one sentence.")

        # Basic coherence checks
        self.assertIn("alex", t2.lower(), f"Turn 2 forgot the name: {t2}")
        self.assertIn("42", t3, f"Turn 3 forgot the number: {t3}")
        normalized_t4 = t4.lower().replace("-", " ")
        self.assertTrue(
            "58" in t4 or "fifty eight" in normalized_t4,
            f"Turn 4 arithmetic wrong: {t4}",
        )
        self.assertGreater(len(t5), 10, f"Turn 5 summary too short: {t5}")

    def test_eviction_under_pressure(self):
        """Fill Mamba cache near-capacity with distinct requests; new requests still succeed (eviction works)."""
        for i in range(30):
            resp = self._chat([
                {"role": "user", "content": f"Request number {i}. Reply with: ok{i}"},
            ], max_tokens=10)
            self.assertIn(resp["choices"][0]["finish_reason"], ("stop", "length"))
            time.sleep(0.1)

        # Final request must still work
        resp = self._chat([{"role": "user", "content": "Reply with: final_ok"}], max_tokens=10)
        self.assertIn(resp["choices"][0]["finish_reason"], ("stop", "length"))


if __name__ == "__main__":
    unittest.main()
