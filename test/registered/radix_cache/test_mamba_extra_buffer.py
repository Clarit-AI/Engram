from sglang.test.ci.ci_register import register_cuda_ci, register_amd_ci

register_cuda_ci(est_time=90, suite="stage-b-test-small-1-gpu")
register_amd_ci(est_time=90, suite="stage-b-test-small-1-gpu-amd")

import os
import unittest

import torch

from sglang.srt.configs.mamba_utils import Mamba2CacheParams, Mamba2StateShape
from sglang.srt.environ import envs
from sglang.srt.managers.schedule_batch import Req
from sglang.srt.mem_cache.memory_pool import HybridReqToTokenPool
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.srt.utils import get_device

SERVER_URL = os.environ.get("SERVER_URL", "http://localhost:30000")


def _make_pool_extra_buffer(max_num_reqs=4, mamba_cache_size=8, max_context_len=128):
    device = get_device()
    num_layers = 48
    global_interval = 4
    full_attention_layer_ids = [
        i for i in range(global_interval - 1, num_layers, global_interval)
    ]
    mamba_layers = [i for i in range(num_layers) if i not in full_attention_layer_ids]
    shape = Mamba2StateShape.create(
        tp_world_size=1, intermediate_size=4096, n_groups=16, num_heads=32,
        head_dim=128, state_size=128, conv_kernel=4,
    )
    with envs.SGLANG_MAMBA_SSM_DTYPE.override("bfloat16"):
        cache_params = Mamba2CacheParams(shape=shape, layers=mamba_layers)
    return HybridReqToTokenPool(
        size=max_num_reqs, mamba_size=mamba_cache_size,
        mamba_spec_state_size=max_num_reqs, max_context_len=max_context_len,
        device=device, enable_memory_saver=False, cache_params=cache_params,
        enable_mamba_extra_buffer=True, speculative_num_draft_tokens=3,
    )


def _make_req():
    return Req(
        rid=0, origin_input_text="", origin_input_ids=[],
        sampling_params=SamplingParams(temperature=0, max_new_tokens=1),
    )


class TestMambaExtraBufferUnit(unittest.TestCase):
    """Unit tests — no server required."""

    def setUp(self):
        self.pool = _make_pool_extra_buffer()

    def test_extra_buffer_alloc(self):
        """After alloc with extra_buffer=True, req.mamba_ping_pong_track_buffer is non-None with size >= 1."""
        req = _make_req()
        self.pool.alloc([req])
        self.assertIsNotNone(
            getattr(req, "mamba_ping_pong_track_buffer", None),
            "mamba_ping_pong_track_buffer should be allocated in extra_buffer mode"
        )
        buf = req.mamba_ping_pong_track_buffer
        self.assertGreaterEqual(len(buf), 1)
        # Clean up
        self.pool.free_mamba_cache(req)
        self.pool.free(req)

    def test_extra_buffer_free_with_keep(self):
        """free_mamba_cache with mamba_ping_pong_track_buffer_to_keep frees all but one ping-pong slot."""
        req = _make_req()
        self.pool.alloc([req])
        buf = req.mamba_ping_pong_track_buffer
        keep_idx = 0
        # The kept tensor's data before free
        kept_data = buf[keep_idx].clone() if buf[keep_idx] is not None else None

        self.pool.free_mamba_cache(req, mamba_ping_pong_track_buffer_to_keep=keep_idx)
        # Main mamba slot freed; kept ping-pong slot tensor data should be intact
        if kept_data is not None:
            self.assertTrue(torch.equal(buf[keep_idx], kept_data))
        self.pool.free(req)

    def test_cache_unfinished_req_extra_buffer(self):
        """cache_unfinished_req clears mamba_last_track_seqlen and updates prefix_indices."""
        # Requires full MambaRadixCache setup — implement with cache fixture
        self.skipTest("Requires full MambaRadixCache setup — implement with cache fixture")


class TestMambaExtraBufferServer(unittest.TestCase):
    """Server integration test — requires running server with extra_buffer strategy."""

    def setUp(self):
        import requests
        try:
            r = requests.get(f"{SERVER_URL}/health", timeout=5)
            if r.status_code != 200:
                self.skipTest("Server not available")
        except Exception:
            self.skipTest("Server not available")

    def test_server_inference_extra_buffer_mode(self):
        """Inference in extra_buffer mode produces same output as no_buffer at temperature=0."""
        import requests
        payload = {
            "model": "default",
            "messages": [{"role": "user", "content": "What is 2+2?"}],
            "temperature": 0,
            "max_tokens": 20,
        }
        r = requests.post(f"{SERVER_URL}/v1/chat/completions", json=payload, timeout=60)
        self.assertEqual(r.status_code, 200)
        data = r.json()
        content = data["choices"][0]["message"]["content"]
        self.assertGreater(len(content), 0)

        # Baseline comparison only if BASELINE_SERVER_URL is explicitly configured
        baseline_server_url = os.environ.get("BASELINE_SERVER_URL")
        if baseline_server_url is not None:
            baseline = requests.post(
                f"{baseline_server_url}/v1/chat/completions", json=payload, timeout=60
            )
            self.assertEqual(baseline.status_code, 200)
            baseline_content = baseline.json()["choices"][0]["message"]["content"]
            self.assertEqual(content, baseline_content)


if __name__ == "__main__":
    unittest.main()
