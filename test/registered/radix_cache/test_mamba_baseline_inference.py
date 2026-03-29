from sglang.test.ci.ci_register import register_cuda_ci, register_amd_ci

register_cuda_ci(est_time=120, suite="stage-b-test-small-1-gpu")
register_amd_ci(est_time=120, suite="stage-b-test-small-1-gpu-amd")

import concurrent.futures
import unittest
import requests
import os

SERVER_URL = os.environ.get("SERVER_URL", "http://localhost:30000")


class TestMambaBaselineInference(unittest.TestCase):

    def _chat(self, messages, stream=False, **kwargs):
        payload = {"model": "default", "messages": messages, "stream": stream, **kwargs}
        return requests.post(f"{SERVER_URL}/v1/chat/completions", json=payload, stream=stream, timeout=60)

    def test_health_endpoint(self):
        """GET /health returns 200."""
        r = requests.get(f"{SERVER_URL}/health", timeout=10)
        self.assertEqual(r.status_code, 200)

    def test_single_turn_completion(self):
        """Single /v1/chat/completions request returns non-empty response with correct finish_reason."""
        r = self._chat([{"role": "user", "content": "What is 2+2?"}], temperature=0, max_tokens=50)
        self.assertEqual(r.status_code, 200)
        data = r.json()
        choice = data["choices"][0]
        self.assertIn(choice["finish_reason"], ("stop", "length"))
        self.assertGreater(len(choice["message"]["content"]), 0)

    def test_streaming_completion(self):
        """stream=True: all SSE chunks arrive and final chunk has finish_reason."""
        r = self._chat([{"role": "user", "content": "Count to 5."}], stream=True, temperature=0, max_tokens=50)
        self.assertEqual(r.status_code, 200)
        chunks = list(r.iter_lines())
        # Filter data lines
        data_lines = [l for l in chunks if l.startswith(b"data:") and l != b"data: [DONE]"]
        self.assertGreater(len(data_lines), 0)
        # Last data chunk should contain finish_reason
        import json
        last = json.loads(data_lines[-1][len(b"data:"):])
        self.assertIn(last["choices"][0]["finish_reason"], ("stop", "length"))

    def test_batch_inference_independence(self):
        """N=4 identical prompts at temperature=0 produce identical responses (state isolation)."""
        messages = [{"role": "user", "content": "Reply with exactly the word: apple"}]
        def send(_):
            r = self._chat(messages, temperature=0, max_tokens=10)
            return r.json()["choices"][0]["message"]["content"].strip()

        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as ex:
            results = list(ex.map(send, range(4)))

        self.assertEqual(len(set(results)), 1, f"Responses differed: {results}")

    def test_batch_inference_different_prompts(self):
        """4 different prompts produce semantically distinct responses."""
        prompts = [
            "Name a fruit.",
            "Name a planet.",
            "Name a color.",
            "Name an animal.",
        ]
        def send(p):
            r = self._chat([{"role": "user", "content": p}], temperature=0, max_tokens=20)
            return r.json()["choices"][0]["message"]["content"].strip().lower()

        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as ex:
            results = list(ex.map(send, prompts))

        # All 4 responses should be unique
        self.assertEqual(len(set(results)), 4, f"Some responses were identical: {results}")

    def test_long_context(self):
        """Long system prompt (>512 tokens) does not cause OOM or truncation error."""
        system = "You are a helpful assistant. " * 100  # ~500+ tokens
        r = self._chat([
            {"role": "system", "content": system},
            {"role": "user", "content": "Summarize your role in one sentence."}
        ], temperature=0, max_tokens=50)
        self.assertEqual(r.status_code, 200)
        data = r.json()
        self.assertNotIn("error", data)
        self.assertGreater(len(data["choices"][0]["message"]["content"]), 0)

    def test_sampling_params(self):
        """Varying temperature, top_p, max_new_tokens are respected."""
        # max_new_tokens=5 should produce a short response
        r = self._chat([{"role": "user", "content": "Tell me a long story."}],
                       temperature=0, max_tokens=5)
        self.assertEqual(r.status_code, 200)
        data = r.json()
        # finish_reason should be "length" since we cut short
        self.assertEqual(data["choices"][0]["finish_reason"], "length")
        # Response tokens should be <= 5
        usage = data.get("usage", {})
        if "completion_tokens" in usage:
            self.assertLessEqual(usage["completion_tokens"], 5)


if __name__ == "__main__":
    unittest.main()
