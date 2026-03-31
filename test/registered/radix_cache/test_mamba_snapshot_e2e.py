from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci

register_cuda_ci(est_time=150, suite="stage-b-test-small-1-gpu")
register_amd_ci(est_time=150, suite="stage-b-test-small-1-gpu-amd")

import glob
import os
import time
import unittest

import requests

SERVER_URL = os.environ.get("SERVER_URL", "http://localhost:30000")
SNAPSHOT_DIR = os.environ.get("SNAPSHOT_DIR", "/tmp/mamba_snapshots")


class TestMambaSnapshotE2E(unittest.TestCase):

    def setUp(self):
        try:
            r = requests.get(f"{SERVER_URL}/health", timeout=5)
            if r.status_code != 200:
                self.skipTest("Server not available")
        except Exception:
            self.skipTest("Server not available")

    def _chat(self, messages, rid=None, **kwargs):
        kwargs.setdefault("temperature", 0)
        kwargs.setdefault("max_tokens", 80)
        payload = {"model": "default", "messages": messages, **kwargs}
        if rid:
            payload["rid"] = rid
        r = requests.post(f"{SERVER_URL}/v1/chat/completions", json=payload, timeout=60)
        r.raise_for_status()
        return r.json()

    def _save_snapshot(self, rid):
        r = requests.post(f"{SERVER_URL}/save_snapshot", json={"rid": rid}, timeout=30)
        r.raise_for_status()
        return r.json()

    def _restore_snapshot(self, rid):
        r = requests.post(
            f"{SERVER_URL}/restore_snapshot", json={"rid": rid}, timeout=30
        )
        r.raise_for_status()
        return r.json()

    def _restore_snapshot_new_request(self, rid):
        """Restore with create_new_request=True — returns a new rid."""
        r = requests.post(
            f"{SERVER_URL}/restore_snapshot",
            json={"rid": rid, "create_new_request": True},
            timeout=30,
        )
        r.raise_for_status()
        return r.json()

    def test_save_snapshot_returns_success(self):
        """After generating tokens for a request, POST /save_snapshot returns success=True."""
        import uuid

        rid = f"test-save-{uuid.uuid4().hex[:8]}"
        self._chat([{"role": "user", "content": "Hello, what is 1+1?"}], rid=rid)
        time.sleep(0.5)  # allow request to settle

        result = self._save_snapshot(rid)
        self.assertTrue(
            result.get("success", False), f"save_snapshot returned: {result}"
        )

    def test_restore_snapshot_state_equivalence(self):
        """Save after turn N, restore, re-generate turn N+1 — output must match original at temperature=0."""
        import uuid

        rid = f"test-restore-{uuid.uuid4().hex[:8]}"
        messages = []

        # Turn 1
        messages.append({"role": "user", "content": "My secret word is: ALPHA42."})
        resp1 = self._chat(messages, rid=rid)
        messages.append(
            {"role": "assistant", "content": resp1["choices"][0]["message"]["content"]}
        )

        # Save snapshot after turn 1
        save_result = self._save_snapshot(rid)
        self.assertTrue(
            save_result.get("success", False), f"Snapshot save failed: {save_result}"
        )

        # Turn 2 — generate original output
        turn2_prompt = "What is my secret word?"
        messages_with_t2 = messages + [{"role": "user", "content": turn2_prompt}]
        resp2_original = self._chat(messages_with_t2, rid=rid)
        original_output = resp2_original["choices"][0]["message"]["content"]

        # Restore snapshot (back to post-turn-1 state)
        restore_result = self._restore_snapshot(rid)
        self.assertTrue(
            restore_result.get("success", False),
            f"Snapshot restore failed: {restore_result}",
        )

        # Re-generate turn 2 from restored state
        resp2_restored = self._chat(messages_with_t2, rid=rid)
        restored_output = resp2_restored["choices"][0]["message"]["content"]

        self.assertEqual(
            original_output.strip(),
            restored_output.strip(),
            f"Output mismatch after restore.\nOriginal: {original_output}\nRestored: {restored_output}",
        )

    def test_restore_requires_idle_request(self):
        """Restoring a snapshot for an active/unknown request returns success=False gracefully."""
        result = self._restore_snapshot("nonexistent-rid-xyz-999")
        # Should not raise — must return a response indicating failure
        self.assertFalse(
            result.get("success", True),
            f"Expected success=False for nonexistent rid, got: {result}",
        )

    def test_snapshot_disk_format(self):
        """After save, .safetensors and .json files exist on disk with expected keys."""
        import uuid

        rid = f"test-disk-{uuid.uuid4().hex[:8]}"
        self._chat([{"role": "user", "content": "Save this state."}], rid=rid)
        time.sleep(0.5)
        save_result = self._save_snapshot(rid)
        self.assertTrue(save_result.get("success", False))
        time.sleep(1.0)  # allow disk flush

        # Find safetensors files
        safetensors_files = glob.glob(
            os.path.join(SNAPSHOT_DIR, "**/*.safetensors"), recursive=True
        )
        self.assertGreater(
            len(safetensors_files), 0, f"No .safetensors files found in {SNAPSHOT_DIR}"
        )

        # Verify keys
        try:
            from safetensors import safe_open

            with safe_open(safetensors_files[0], framework="pt") as f:
                keys = list(f.keys())
            key_str = str(keys).lower()
            self.assertTrue(
                any(k in key_str for k in ("conv", "ssm", "temporal", "state")),
                f"Expected conv/ssm/temporal/state keys, got: {keys}",
            )
        except ImportError:
            self.skipTest("safetensors not installed")

    def test_snapshot_manager_tier_consistency(self):
        """MambaHostPool WARM tier: save CPU tensors, retrieve, verify data integrity."""
        try:
            import torch

            from sglang.srt.snapshot.mamba_host_pool import MambaHostPool
        except ImportError as e:
            self.skipTest(f"MambaHostPool not importable: {e}")

        pool = MambaHostPool(max_conversations=10, max_memory_gb=1.0)

        conv_id = "tier-test-conv-001"

        # Build mock Mamba conv/temporal states (CPU, no GPU needed)
        num_layers = 4
        conv_kernel = 4
        hidden = 64
        ssm_state_dim = 16

        conv_states = [
            torch.randn(1, hidden, conv_kernel, dtype=torch.float32)
            for _ in range(num_layers)
        ]
        temporal_states = torch.randn(
            num_layers, 1, ssm_state_dim, hidden, dtype=torch.float32
        )

        # Save to WARM tier (host RAM)
        ok = pool.save_state(
            conv_id, conv_states, temporal_states, metadata={"turn": 1}
        )
        self.assertTrue(ok, "MambaHostPool.save_state() returned False")

        # Retrieve from WARM tier
        result = pool.get_state(conv_id)
        self.assertIsNotNone(
            result, "MambaHostPool.get_state() returned None after save"
        )

        retrieved_conv, retrieved_temporal, retrieved_meta = result

        # Verify tensor data integrity
        self.assertEqual(
            len(retrieved_conv),
            num_layers,
            f"Expected {num_layers} conv states, got {len(retrieved_conv)}",
        )
        for i, (orig, retr) in enumerate(zip(conv_states, retrieved_conv)):
            self.assertTrue(
                torch.allclose(orig.cpu(), retr.cpu()),
                f"Conv state layer {i} mismatch after WARM tier round-trip",
            )

        self.assertTrue(
            torch.allclose(temporal_states.cpu(), retrieved_temporal.cpu()),
            "Temporal state mismatch after WARM tier round-trip",
        )

        # Verify metadata preserved
        self.assertIsNotNone(retrieved_meta, "Metadata lost in WARM tier round-trip")

        # Verify LRU eviction: add more conversations than max_conversations
        small_pool = MambaHostPool(max_conversations=3, max_memory_gb=1.0)
        for i in range(5):
            cs = [torch.zeros(1, 8, 4)]
            ts = torch.zeros(1, 1, 4, 8)
            small_pool.save_state(f"conv-{i}", cs, ts)

        # Pool should have evicted down to 3
        self.assertLessEqual(
            len(small_pool),
            3,
            f"LRU eviction failed: pool has {len(small_pool)} entries (max 3)",
        )

    def test_create_new_request_returns_new_rid(self):
        """create_new_request=True returns a new rid different from the original."""
        import uuid

        rid = f"test-new-rid-{uuid.uuid4().hex[:8]}"
        self._chat([{"role": "user", "content": "Hello, what is 1+1?"}], rid=rid)
        time.sleep(0.5)  # allow request to settle

        save_result = self._save_snapshot(rid)
        self.assertTrue(
            save_result.get("success", False), f"Snapshot save failed: {save_result}"
        )

        restore_result = self._restore_snapshot_new_request(rid)
        self.assertTrue(
            restore_result.get("success", False),
            f"create_new_request restore failed: {restore_result}",
        )

        new_rid = restore_result.get("rid")
        self.assertIsNotNone(new_rid, "Expected rid in response")
        self.assertNotEqual(new_rid, rid, "New rid must differ from original")


if __name__ == "__main__":
    unittest.main()
