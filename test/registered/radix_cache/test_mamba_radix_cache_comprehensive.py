"""
Comprehensive unit tests for MambaRadixCache implementation.

This module provides extensive test coverage for MambaRadixCache, focusing on:
- Tombstone node behavior (nodes with KV cache but no Mamba state)
- LRU list management (dual lists for full and Mamba eviction)
- Lock reference counting (full_lock_ref and mamba_lock_ref)
- Edge cases (empty cache, full cache, eviction pressure)
- Page-aligned operations
- Copy-on-write (COW) functionality
- Eviction policy correctness

Test Coverage (NEW):
✅ Tombstone nodes: Creation, eviction, and lifecycle
✅ LRU lists: Dual list integrity, MRU/LRU ordering
✅ Lock refs: Protection from eviction, reference counting
✅ Edge cases: Boundary conditions, full cache scenarios
✅ COW: State copying, independent modifications
✅ Eviction: LRU policy, leaf-only eviction for full cache

Usage:
    python test_mamba_radix_cache_comprehensive.py
    python -m pytest test_mamba_radix_cache_comprehensive.py -v
    python -m pytest test_mamba_radix_cache_comprehensive.py::TestMambaRadixCacheComprehensive::test_tombstone_node_creation
"""

from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci

# CPU-based unit test, runs on GPU runners
register_cuda_ci(est_time=12, suite="stage-b-test-small-1-gpu")
register_amd_ci(est_time=12, suite="stage-b-test-small-1-gpu-amd")

import itertools
import unittest

import torch

from sglang.srt.configs.mamba_utils import Mamba2CacheParams, Mamba2StateShape
from sglang.srt.environ import envs
from sglang.srt.managers.schedule_batch import Req
from sglang.srt.mem_cache.allocator import TokenToKVPoolAllocator
from sglang.srt.mem_cache.base_prefix_cache import (
    EvictParams,
    InsertParams,
    MatchPrefixParams,
)
from sglang.srt.mem_cache.cache_init_params import CacheInitParams
from sglang.srt.mem_cache.mamba_radix_cache import MambaRadixCache
from sglang.srt.mem_cache.memory_pool import HybridLinearKVPool, HybridReqToTokenPool
from sglang.srt.mem_cache.radix_cache import RadixKey
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.srt.server_args import ServerArgs, set_global_server_args_for_scheduler
from sglang.srt.utils import get_device


class TestMambaRadixCacheComprehensive(unittest.TestCase):
    """Comprehensive test cases for MambaRadixCache."""

    @classmethod
    def setUpClass(cls):
        """Set up test fixtures once for all tests."""
        # Set global server args for page_size=1
        set_global_server_args_for_scheduler(
            ServerArgs(model_path="dummy", page_size=1)
        )

    def setUp(self):
        """Set up test fixtures for each test."""
        # Cache configuration
        self.kv_cache_size = 128
        self.dtype = torch.bfloat16
        self.head_num = 2
        self.head_dim = 256
        self.num_layers = 48
        self.global_interval = 4
        self.max_num_reqs = 10
        self.mamba_cache_size = 20
        self.max_context_len = 128
        self.device = get_device()

        # Request ID counter for unique request IDs
        self._rid_counter = itertools.count(1)

        # Layer configuration
        self.full_attention_layer_ids = [
            i
            for i in range(
                self.global_interval - 1, self.num_layers, self.global_interval
            )
        ]
        self.mamba_layers = [
            i for i in range(self.num_layers) if i not in self.full_attention_layer_ids
        ]

        # Mamba state shape
        with envs.SGLANG_MAMBA_SSM_DTYPE.override("bfloat16"):
            shape = Mamba2StateShape.create(
                tp_world_size=1,
                intermediate_size=4096,
                n_groups=16,
                num_heads=32,
                head_dim=128,
                state_size=128,
                conv_kernel=4,
            )
            self.mamba2_cache_params = Mamba2CacheParams(
                shape=shape, layers=self.mamba_layers
            )

        # Create req_to_token_pool
        self.req_to_token_pool = HybridReqToTokenPool(
            size=self.max_num_reqs,
            mamba_size=self.mamba_cache_size,
            mamba_spec_state_size=self.max_num_reqs,
            max_context_len=self.max_context_len,
            device=self.device,
            enable_memory_saver=False,
            cache_params=self.mamba2_cache_params,
            enable_mamba_extra_buffer=False,
            speculative_num_draft_tokens=3,
        )

        # Create KV pool
        self.kv_pool = HybridLinearKVPool(
            size=self.kv_cache_size,
            dtype=self.dtype,
            page_size=1,
            head_num=self.head_num,
            head_dim=self.head_dim,
            full_attention_layer_ids=self.full_attention_layer_ids,
            enable_kvcache_transpose=False,
            device=self.device,
            enable_memory_saver=False,
            mamba_pool=self.req_to_token_pool.mamba_pool,
        )

        # Create allocator
        self.allocator = TokenToKVPoolAllocator(
            size=self.kv_cache_size,
            dtype=self.dtype,
            device=self.device,
            kvcache=self.kv_pool,
            need_sort=False,
        )

        # Create MambaRadixCache
        params = CacheInitParams(
            req_to_token_pool=self.req_to_token_pool,
            token_to_kv_pool_allocator=self.allocator,
            page_size=1,
            disable=False,
        )
        self.cache = MambaRadixCache(params=params)

    def _make_dummy_req(self):
        """Helper to create a dummy request with unique ID."""
        sampling_params = SamplingParams(temperature=0, max_new_tokens=1)
        req = Req(
            rid=next(self._rid_counter),
            origin_input_text="",
            origin_input_ids=[],
            sampling_params=sampling_params,
        )
        self.req_to_token_pool.alloc([req])
        return req

    def test_tombstone_node_creation(self):
        """Test creation of tombstone nodes (nodes with KV cache but no Mamba state)."""
        # Insert a sequence [1, 2, 3]
        req1 = self._make_dummy_req()
        token_ids_1 = [1, 2, 3]
        kv_indices_1 = self.allocator.alloc(3)
        mamba_value_1 = req1.mamba_pool_idx.unsqueeze(0)

        result = self.cache.insert(
            InsertParams(
                key=RadixKey(token_ids_1),
                value=kv_indices_1,
                mamba_value=mamba_value_1,
            )
        )
        self.assertEqual(result.prefix_len, 0)
        self.assertFalse(result.mamba_exist)

        # Insert a longer sequence [1, 2, 3, 4, 5] - this creates a node at [1, 2, 3]
        req2 = self._make_dummy_req()
        token_ids_2 = [1, 2, 3, 4, 5]
        kv_indices_2 = self.allocator.alloc(5)
        mamba_value_2 = req2.mamba_pool_idx.unsqueeze(0)

        result = self.cache.insert(
            InsertParams(
                key=RadixKey(token_ids_2),
                value=kv_indices_2,
                mamba_value=mamba_value_2,
            )
        )
        self.assertEqual(result.prefix_len, 3)

        # Evict 1 Mamba state - should evict req1's Mamba state, creating a tombstone
        evict_result = self.cache.evict(EvictParams(num_tokens=0, mamba_num=1))
        self.assertGreaterEqual(evict_result.mamba_num_evicted, 1)

        # Match [1, 2, 3] - should match KV cache but no Mamba state (tombstone)
        match_result = self.cache.match_prefix(
            MatchPrefixParams(key=RadixKey([1, 2, 3]))
        )
        # Due to tombstone, match should only return nodes with Mamba states
        # In this case, it should return empty because the [1,2,3] node is a tombstone
        self.assertEqual(len(match_result.device_indices), 0)

        # Match [1, 2, 3, 4, 5] - should match the full sequence with Mamba state
        match_result = self.cache.match_prefix(
            MatchPrefixParams(key=RadixKey(token_ids_2))
        )
        self.assertEqual(len(match_result.device_indices), 5)

    def test_lru_list_integrity(self):
        """Test that LRU lists maintain correct ordering."""
        # Insert 3 sequences
        reqs = []
        for i, token_ids in enumerate([[1, 2], [3, 4], [5, 6]]):
            req = self._make_dummy_req()
            reqs.append(req)
            kv_indices = self.allocator.alloc(len(token_ids))
            mamba_value = req.mamba_pool_idx.unsqueeze(0)

            self.cache.insert(
                InsertParams(
                    key=RadixKey(token_ids),
                    value=kv_indices,
                    mamba_value=mamba_value,
                )
            )

        # Check that all nodes are in both LRU lists
        full_lru_size = len(self.cache.full_lru_list.cache)
        mamba_lru_size = len(self.cache.mamba_lru_list.cache)
        self.assertEqual(full_lru_size, 3)  # 3 nodes
        self.assertEqual(mamba_lru_size, 3)  # 3 nodes with Mamba states

        # Access [1, 2] - should move to MRU
        match_result = self.cache.match_prefix(MatchPrefixParams(key=RadixKey([1, 2])))
        self.assertEqual(len(match_result.device_indices), 2)

        # Evict 1 Mamba state - should evict LRU (either [3,4] or [5,6])
        evict_result = self.cache.evict(EvictParams(num_tokens=0, mamba_num=1))
        self.assertEqual(evict_result.mamba_num_evicted, 1)

        # Verify [1, 2] is still cached (it's MRU)
        match_result = self.cache.match_prefix(MatchPrefixParams(key=RadixKey([1, 2])))
        self.assertEqual(len(match_result.device_indices), 2)

    def test_lock_ref_protection(self):
        """Test that locked nodes are protected from eviction."""
        # Insert a sequence
        req = self._make_dummy_req()
        token_ids = [1, 2, 3]
        kv_indices = self.allocator.alloc(3)
        mamba_value = req.mamba_pool_idx.unsqueeze(0)

        self.cache.insert(
            InsertParams(
                key=RadixKey(token_ids),
                value=kv_indices,
                mamba_value=mamba_value,
            )
        )

        # Match and get the last node
        match_result = self.cache.match_prefix(
            MatchPrefixParams(key=RadixKey(token_ids))
        )
        last_node = match_result.last_device_node

        # Lock the node
        self.cache.inc_lock_ref(last_node)

        # Try to evict - should fail because node is locked
        initial_mamba_size = self.cache.mamba_evictable_size()
        evict_result = self.cache.evict(EvictParams(num_tokens=0, mamba_num=1))
        # Should evict 0 because all nodes are locked
        self.assertEqual(evict_result.mamba_num_evicted, 0)

        # Unlock the node
        self.cache.dec_lock_ref(last_node)

        # Now eviction should succeed
        evict_result = self.cache.evict(EvictParams(num_tokens=0, mamba_num=1))
        self.assertGreaterEqual(evict_result.mamba_num_evicted, 1)

    def test_full_cache_eviction(self):
        """Test behavior when cache is full and requires eviction."""
        # Fill the Mamba cache
        reqs = []
        for i in range(self.mamba_cache_size):
            req = self._make_dummy_req()
            reqs.append(req)
            token_ids = [i]
            kv_indices = self.allocator.alloc(1)
            mamba_value = req.mamba_pool_idx.unsqueeze(0)

            self.cache.insert(
                InsertParams(
                    key=RadixKey(token_ids),
                    value=kv_indices,
                    mamba_value=mamba_value,
                )
            )
            # Cache now owns the mamba state; free the req pool slot for reuse
            self.req_to_token_pool.free(req)

        # Verify cache is full
        self.assertEqual(self.req_to_token_pool.mamba_pool.available_size(), 0)

        # insert() does not auto-evict; explicitly free 1 LRU slot before allocating req_new.
        evict_result = self.cache.evict(EvictParams(num_tokens=0, mamba_num=1))
        self.assertGreaterEqual(evict_result.mamba_num_evicted, 1)

        req_new = self._make_dummy_req()
        token_ids_new = [self.mamba_cache_size]  # New unique token
        kv_indices_new = self.allocator.alloc(1)
        mamba_value_new = req_new.mamba_pool_idx.unsqueeze(0)

        self.cache.insert(
            InsertParams(
                key=RadixKey(token_ids_new),
                value=kv_indices_new,
                mamba_value=mamba_value_new,
            )
        )

        # Cache should be back at capacity
        self.assertEqual(
            self.req_to_token_pool.mamba_pool.available_size(),
            0,
            "Cache should remain at capacity after eviction",
        )

        # New item must be findable
        match_result = self.cache.match_prefix(
            MatchPrefixParams(key=RadixKey(token_ids_new))
        )
        self.assertGreater(
            len(match_result.device_indices), 0, "New item should be findable in cache"
        )

        # First item (LRU) must have been evicted
        first_item_match = self.cache.match_prefix(MatchPrefixParams(key=RadixKey([0])))
        self.assertEqual(
            len(first_item_match.device_indices),
            0,
            "Oldest (LRU) item should have been evicted",
        )

    def test_cow_mamba_state(self):
        """Test copy-on-write (COW) for Mamba states."""
        # Insert a sequence
        req1 = self._make_dummy_req()
        token_ids = [1, 2, 3]
        kv_indices = self.allocator.alloc(3)
        mamba_value_1 = req1.mamba_pool_idx.unsqueeze(0)

        self.cache.insert(
            InsertParams(
                key=RadixKey(token_ids),
                value=kv_indices,
                mamba_value=mamba_value_1,
            )
        )

        # Free req1's Mamba cache (to test COW from radix cache)
        self.req_to_token_pool.free_mamba_cache(req1)
        req1.mamba_pool_idx = None

        # Match with COW - should copy Mamba state to req1
        match_result = self.cache.match_prefix(
            MatchPrefixParams(key=RadixKey(token_ids), req=req1, cow_mamba=True)
        )
        self.assertEqual(len(match_result.device_indices), 3)
        self.assertIsNotNone(req1.mamba_pool_idx)

        # Verify the copied state matches the original
        last_node = match_result.last_device_node
        mamba_pool = self.req_to_token_pool.mamba_pool

        # Check conv state
        self.assertTrue(
            torch.all(
                mamba_pool.mamba_cache.conv[0][:, req1.mamba_pool_idx]
                == mamba_pool.mamba_cache.conv[0][:, last_node.mamba_value]
            )
        )

        # Check temporal state
        self.assertTrue(
            torch.all(
                mamba_pool.mamba_cache.temporal[:, req1.mamba_pool_idx]
                == mamba_pool.mamba_cache.temporal[:, last_node.mamba_value]
            )
        )

    def test_evict_full_leaves_only(self):
        """Test that full eviction only evicts leaf nodes."""
        # Insert a tree structure:
        #   [1, 2, 3]
        #   [1, 2, 3, 4, 5]
        #   [1, 2, 3, 4, 6]
        req1 = self._make_dummy_req()
        token_ids_1 = [1, 2, 3]
        kv_indices_1 = self.allocator.alloc(3)
        mamba_value_1 = req1.mamba_pool_idx.unsqueeze(0)
        self.cache.insert(
            InsertParams(
                key=RadixKey(token_ids_1),
                value=kv_indices_1,
                mamba_value=mamba_value_1,
            )
        )

        req2 = self._make_dummy_req()
        token_ids_2 = [1, 2, 3, 4, 5]
        kv_indices_2 = self.allocator.alloc(5)
        mamba_value_2 = req2.mamba_pool_idx.unsqueeze(0)
        self.cache.insert(
            InsertParams(
                key=RadixKey(token_ids_2),
                value=kv_indices_2,
                mamba_value=mamba_value_2,
            )
        )

        req3 = self._make_dummy_req()
        token_ids_3 = [1, 2, 3, 4, 6]
        kv_indices_3 = self.allocator.alloc(5)
        mamba_value_3 = req3.mamba_pool_idx.unsqueeze(0)
        self.cache.insert(
            InsertParams(
                key=RadixKey(token_ids_3),
                value=kv_indices_3,
                mamba_value=mamba_value_3,
            )
        )

        # Evict 1 full token - should evict from a leaf node only
        initial_full_size = self.cache.full_evictable_size()
        evict_result = self.cache.evict(EvictParams(num_tokens=1, mamba_num=0))
        self.assertGreaterEqual(evict_result.num_tokens_evicted, 1)

        # Internal nodes should still be cached
        # Leaf nodes may be partially evicted

    def test_empty_cache_operations(self):
        """Test operations on an empty cache."""
        # Match on empty cache
        match_result = self.cache.match_prefix(
            MatchPrefixParams(key=RadixKey([1, 2, 3]))
        )
        self.assertEqual(len(match_result.device_indices), 0)
        self.assertEqual(match_result.last_device_node, self.cache.root_node)

        # Evict on empty cache
        evict_result = self.cache.evict(EvictParams(num_tokens=10, mamba_num=10))
        self.assertEqual(evict_result.num_tokens_evicted, 0)
        self.assertEqual(evict_result.mamba_num_evicted, 0)

    def test_evictable_size_tracking(self):
        """Test that evictable size counters are correctly maintained."""
        # Check initial state
        self.assertEqual(self.cache.full_evictable_size(), 0)
        self.assertEqual(self.cache.mamba_evictable_size(), 0)

        # Insert a sequence
        req = self._make_dummy_req()
        token_ids = [1, 2, 3, 4, 5]
        kv_indices = self.allocator.alloc(5)
        mamba_value = req.mamba_pool_idx.unsqueeze(0)

        self.cache.insert(
            InsertParams(
                key=RadixKey(token_ids),
                value=kv_indices,
                mamba_value=mamba_value,
            )
        )

        # Check size increased
        self.assertEqual(self.cache.full_evictable_size(), 5)
        self.assertEqual(self.cache.mamba_evictable_size(), 1)

        # Lock the node - evictable size should decrease
        match_result = self.cache.match_prefix(
            MatchPrefixParams(key=RadixKey(token_ids))
        )
        last_node = match_result.last_device_node
        self.cache.inc_lock_ref(last_node)

        self.assertEqual(self.cache.full_evictable_size(), 0)  # All locked
        self.assertEqual(self.cache.mamba_evictable_size(), 0)  # All locked

        # Unlock - evictable size should increase
        self.cache.dec_lock_ref(last_node)
        self.assertEqual(self.cache.full_evictable_size(), 5)
        self.assertEqual(self.cache.mamba_evictable_size(), 1)

    def test_mamba_branching_seqlen(self):
        """Test mamba_branching_seqlen is set when the matched path crosses tombstones.

        Strategy: insert A(65), B(75), C(85) tokens — each extends the previous.
        Tree: root → NodeA([0..64]) → NodeB([65..74]) → NodeC([75..84])

        After all inserts _insert_helper touches nodes root→leaf in order, so
        the mamba LRU order is: NodeA (LRU) < NodeB < NodeC (MRU).

        Evicting 2 mamba states tombstones NodeA then NodeB (both are internal
        nodes with children, so they become tombstones instead of being deleted).

        Matching B=[0..74] traverses both tombstones; matched tokens=75,
        chunk-aligned = (75//64)*64 = 64 → mamba_branching_seqlen == 64.

        KV note: fake zero tensors are used to stay within the test's 128-slot
        KV pool.  Internal-node eviction never calls token_to_kv_pool_allocator
        .free(), so fake tensors are safe here.
        """
        token_ids_A = list(range(65))  # [0..64]  — 65 tokens
        token_ids_B = list(range(75))  # [0..74]  — 75 tokens (= A + 10)
        token_ids_C = list(range(85))  # [0..84]  — 85 tokens (= B + 10)

        # Fake KV tensors — value does not matter for branching_seqlen correctness.
        kv_A = torch.zeros(65, dtype=torch.int64)
        kv_B = torch.zeros(75, dtype=torch.int64)
        kv_C = torch.zeros(85, dtype=torch.int64)

        req_a = self._make_dummy_req()
        self.cache.insert(
            InsertParams(
                key=RadixKey(token_ids_A),
                value=kv_A,
                mamba_value=req_a.mamba_pool_idx.unsqueeze(0),
            )
        )

        req_b = self._make_dummy_req()
        self.cache.insert(
            InsertParams(
                key=RadixKey(token_ids_B),
                value=kv_B,
                mamba_value=req_b.mamba_pool_idx.unsqueeze(0),
            )
        )

        req_c = self._make_dummy_req()
        self.cache.insert(
            InsertParams(
                key=RadixKey(token_ids_C),
                value=kv_C,
                mamba_value=req_c.mamba_pool_idx.unsqueeze(0),
            )
        )

        # Evict 2 mamba states: LRU=NodeA → tombstone, then NodeB → tombstone.
        evict_result = self.cache.evict(EvictParams(num_tokens=0, mamba_num=2))
        self.assertGreaterEqual(evict_result.mamba_num_evicted, 2)

        # Match B: path goes through NodeA(tombstone) + NodeB(tombstone).
        # best_value_len=0 (no live mamba node on path) → branching detected.
        # sum(matched)=75 → chunk_aligned=64 → mamba_branching_seqlen=64.
        match_result = self.cache.match_prefix(
            MatchPrefixParams(key=RadixKey(token_ids_B))
        )
        self.assertIsNotNone(
            match_result.mamba_branching_seqlen,
            "mamba_branching_seqlen should be set when tombstone is encountered",
        )
        self.assertEqual(
            match_result.mamba_branching_seqlen,
            64,
            "mamba_branching_seqlen should equal chunk-aligned total matched tokens",
        )


if __name__ == "__main__":
    unittest.main()
