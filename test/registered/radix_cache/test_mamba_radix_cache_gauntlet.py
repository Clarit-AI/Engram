"""
Gauntlet tests for MambaRadixCache — probing complex invariants.

Tests:
  1. Interleaved insert / evict_mamba / evict_full with sanity_check after every op
  2. Tombstoned node: KV path preserved through tombstone, mamba_value=None on tombstoned node
  3. mamba_branching_seqlen triggered correctly and is chunk-aligned
  4. COW state independence: modifying source doesn't affect copied state
  5. inc_lock_ref / dec_lock_ref symmetry on a 3-node chain
  6. full_evictable_size + full_protected_size == total tokens in cache (conservation)
"""

from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci

register_cuda_ci(est_time=45, suite="stage-b-test-small-1-gpu")
register_amd_ci(est_time=45, suite="stage-b-test-small-1-gpu-amd")

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


class TestMambaRadixCacheGauntlet(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        set_global_server_args_for_scheduler(
            ServerArgs(model_path="dummy", page_size=1)
        )

    def setUp(self):
        self.kv_cache_size = 256
        self.dtype = torch.bfloat16
        self.head_num = 2
        self.head_dim = 256
        self.num_layers = 48
        self.global_interval = 4
        self.max_num_reqs = 20
        self.mamba_cache_size = 40
        self.max_context_len = 256
        self.device = get_device()
        self._rid_counter = itertools.count(1)

        self.full_attention_layer_ids = [
            i
            for i in range(
                self.global_interval - 1, self.num_layers, self.global_interval
            )
        ]
        self.mamba_layers = [
            i for i in range(self.num_layers) if i not in self.full_attention_layer_ids
        ]

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

        self.allocator = TokenToKVPoolAllocator(
            size=self.kv_cache_size,
            dtype=self.dtype,
            device=self.device,
            kvcache=self.kv_pool,
            need_sort=False,
        )

        params = CacheInitParams(
            req_to_token_pool=self.req_to_token_pool,
            token_to_kv_pool_allocator=self.allocator,
            page_size=1,
            disable=False,
        )
        self.cache = MambaRadixCache(params=params)

    def tearDown(self):
        self.cache.sanity_check()

    def _make_dummy_req(self):
        req = Req(
            rid=next(self._rid_counter),
            origin_input_text="",
            origin_input_ids=[],
            sampling_params=SamplingParams(temperature=0, max_new_tokens=1),
        )
        self.req_to_token_pool.alloc([req])
        return req

    def _insert(self, token_ids, req=None):
        """Insert token_ids into cache; allocates req if not provided. Returns (req, result)."""
        if req is None:
            req = self._make_dummy_req()
        kv = self.allocator.alloc(len(token_ids))
        result = self.cache.insert(
            InsertParams(
                key=RadixKey(token_ids),
                value=kv,
                mamba_value=req.mamba_pool_idx.unsqueeze(0),
            )
        )
        return req, result

    # -------------------------------------------------------------------------
    # Test 1: interleaved insert / evict / match with sanity_check at every step
    # -------------------------------------------------------------------------

    def test_interleaved_insert_evict_match(self):
        """Insert 10 seqs; after each: evict_mamba(1), evict_full(1), sanity_check, match."""
        for i in range(10):
            req, _ = self._insert([i])
            # Free req_pool_idx so the pool can be recycled; cache owns the mamba slot.
            self.req_to_token_pool.free(req)
            self.cache.sanity_check()

            self.cache.evict(EvictParams(num_tokens=0, mamba_num=1))
            self.cache.sanity_check()

            self.cache.evict(EvictParams(num_tokens=1, mamba_num=0))
            self.cache.sanity_check()

            # Match is always safe even if the node was just evicted.
            self.cache.match_prefix(MatchPrefixParams(key=RadixKey([i])))
            self.cache.sanity_check()

    # -------------------------------------------------------------------------
    # Test 2: tombstoned node — KV path preserved, mamba_value is None
    # -------------------------------------------------------------------------

    def test_tombstone_does_not_match_mamba(self):
        """After evict_mamba, the tombstoned node keeps KV but loses mamba state."""
        # Insert [1,2,3] — becomes internal when [1,2,3,4] is added.
        req_a, _ = self._insert([1, 2, 3])
        req_b, _ = self._insert([1, 2, 3, 4])

        # LRU order after inserts: NodeA=[1,2,3] (traversed first each time) is LRU;
        # NodeB=[4] is MRU (just created).  evict_mamba(1) tombstones NodeA.
        evict = self.cache.evict(EvictParams(num_tokens=0, mamba_num=1))
        self.assertEqual(evict.mamba_num_evicted, 1)

        # Matching [1,2,3] alone: last matched node is the tombstone, which has no
        # live mamba state → device_indices is empty (truncated to best_value_len=0).
        r1 = self.cache.match_prefix(MatchPrefixParams(key=RadixKey([1, 2, 3])))
        self.assertEqual(
            len(r1.device_indices),
            0,
            "Tombstoned node should not contribute to device_indices",
        )

        # Matching [1,2,3,4]: passes through tombstone NodeA, ends at live NodeB=[4].
        # NodeB has mamba → best_value_len includes both nodes → KV for all 4 tokens.
        r2 = self.cache.match_prefix(MatchPrefixParams(key=RadixKey([1, 2, 3, 4])))
        self.assertEqual(
            len(r2.device_indices),
            4,
            "Full path through tombstone to live mamba node should return all 4 KV indices",
        )
        self.assertIsNotNone(
            r2.last_device_node.mamba_value,
            "last_device_node should be the live mamba node [4], not the tombstone",
        )

    # -------------------------------------------------------------------------
    # Test 3: mamba_branching_seqlen triggered and chunk-aligned
    # -------------------------------------------------------------------------

    def test_branching_seqlen_triggered(self):
        """Tombstoning internal nodes triggers mamba_branching_seqlen; value is chunk-aligned."""
        # A(65) → B(75) → C(85): NodeA and NodeB become internal nodes.
        # Evict 2 mamba states (LRU=NodeA, then NodeB) → both tombstoned.
        # Match B: 75 matched tokens, chunk_aligned = (75//64)*64 = 64.
        token_ids_A = list(range(65))
        token_ids_B = list(range(75))
        token_ids_C = list(range(85))

        # Fake KV tensors: internal-node eviction never frees KV, so zeros are safe.
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

        evict = self.cache.evict(EvictParams(num_tokens=0, mamba_num=2))
        self.assertGreaterEqual(evict.mamba_num_evicted, 2)

        match = self.cache.match_prefix(MatchPrefixParams(key=RadixKey(token_ids_B)))
        self.assertIsNotNone(
            match.mamba_branching_seqlen,
            "mamba_branching_seqlen must be set when path crosses tombstones",
        )
        chunk_size = 64  # default mamba_cache_chunk_size
        self.assertEqual(
            match.mamba_branching_seqlen % chunk_size,
            0,
            "mamba_branching_seqlen must be chunk-aligned",
        )
        self.assertEqual(match.mamba_branching_seqlen, 64)

    # -------------------------------------------------------------------------
    # Test 4: COW state independence
    # -------------------------------------------------------------------------

    def test_cow_state_independence(self):
        """Modifying the cached node's mamba data does not affect a previously COW'd copy."""
        # Insert [1,2,3] with a distinctive mamba state (fill temporal with 1.0).
        req_orig, _ = self._insert([1, 2, 3])
        node_slot = req_orig.mamba_pool_idx.item()
        pool = self.req_to_token_pool.mamba_pool

        # Write a sentinel value into the cache node's temporal state.
        pool.mamba_cache.temporal[:, node_slot] = 1.0

        # Perform COW: req_cow gets a copy of the node's mamba state.
        req_cow = self._make_dummy_req()
        # Free req_cow's auto-allocated mamba slot so COW can overwrite it.
        self.req_to_token_pool.free_mamba_cache(req_cow)
        match = self.cache.match_prefix(
            MatchPrefixParams(key=RadixKey([1, 2, 3]), req=req_cow, cow_mamba=True)
        )
        self.assertIsNotNone(
            req_cow.mamba_pool_idx, "COW should allocate a mamba slot for req_cow"
        )
        cow_slot = req_cow.mamba_pool_idx.item()

        # Verify the copy has the sentinel value.
        copied_val = pool.mamba_cache.temporal[:, cow_slot].mean().item()
        self.assertAlmostEqual(
            copied_val,
            1.0,
            places=2,
            msg="COW'd copy should reflect original sentinel value",
        )

        # Now overwrite the ORIGINAL cache node slot with zeros.
        pool.mamba_cache.temporal[:, node_slot] = 0.0

        # COW'd copy must be unaffected.
        copied_val_after = pool.mamba_cache.temporal[:, cow_slot].mean().item()
        self.assertAlmostEqual(
            copied_val_after,
            1.0,
            places=2,
            msg="Modifying original slot must not affect COW'd copy",
        )

    # -------------------------------------------------------------------------
    # Test 5: inc_lock_ref / dec_lock_ref symmetry on a 2-node chain
    # -------------------------------------------------------------------------

    def test_inc_dec_lock_ref_symmetry(self):
        """inc_lock_ref propagates full_lock_ref to ancestors; dec_lock_ref restores all to 0."""
        # Build chain: NodeA=[1,2] is internal, NodeB=[3,4] is leaf.
        req_a, _ = self._insert([1, 2])
        req_b, _ = self._insert([1, 2, 3, 4])

        # Get NodeB via match
        match = self.cache.match_prefix(MatchPrefixParams(key=RadixKey([1, 2, 3, 4])))
        node_b = match.last_device_node
        self.assertIsNotNone(node_b.mamba_value, "NodeB should have a live mamba state")

        node_a = node_b.parent
        self.assertIsNotNone(node_a, "NodeB should have a parent (NodeA)")

        # Baseline: both lock refs should be 0.
        self.assertEqual(node_b.full_lock_ref, 0)
        self.assertEqual(node_a.full_lock_ref, 0)
        self.assertEqual(node_b.mamba_lock_ref, 0)

        # Lock NodeB — full_lock_ref propagates to NodeA; mamba_lock_ref set on NodeB.
        self.cache.inc_lock_ref(node_b)
        self.assertGreater(
            node_b.full_lock_ref, 0, "NodeB.full_lock_ref should be > 0 after inc"
        )
        self.assertGreater(
            node_a.full_lock_ref, 0, "NodeA.full_lock_ref should be > 0 after inc"
        )
        self.assertGreater(
            node_b.mamba_lock_ref, 0, "NodeB.mamba_lock_ref should be > 0 after inc"
        )

        # Unlock — all lock refs must return to 0.
        self.cache.dec_lock_ref(node_b)
        self.assertEqual(
            node_b.full_lock_ref, 0, "NodeB.full_lock_ref should be 0 after dec"
        )
        self.assertEqual(
            node_a.full_lock_ref, 0, "NodeA.full_lock_ref should be 0 after dec"
        )
        self.assertEqual(
            node_b.mamba_lock_ref, 0, "NodeB.mamba_lock_ref should be 0 after dec"
        )

    # -------------------------------------------------------------------------
    # Test 6: full_evictable_size + full_protected_size == total tokens in cache
    # -------------------------------------------------------------------------

    def test_full_evictable_and_protected_size_accounting(self):
        """Conservation: full_evictable + full_protected equals total tokens in tree."""

        def total_tokens_in_tree():
            """DFS sum of all node.value lengths."""
            total = 0
            stack = list(self.cache.root_node.children.values())
            while stack:
                node = stack.pop()
                total += len(node.value)
                stack.extend(node.children.values())
            return total

        def check_conservation(label):
            evict = self.cache.full_evictable_size()
            prot = self.cache.full_protected_size()
            total = total_tokens_in_tree()
            self.assertEqual(
                evict + prot,
                total,
                f"{label}: evictable({evict}) + protected({prot}) != total({total})",
            )

        check_conservation("initial (empty)")

        # Insert [1,2,3]
        req_a, _ = self._insert([1, 2, 3])
        check_conservation("after insert [1,2,3]")

        # Insert [1,2,3,4,5] — [1,2,3] already cached, only [4,5] is new.
        req_b, _ = self._insert([1, 2, 3, 4, 5])
        check_conservation("after insert [1,2,3,4,5]")

        # Lock NodeB (the leaf [4,5]) — moves its tokens + ancestor to protected.
        match = self.cache.match_prefix(
            MatchPrefixParams(key=RadixKey([1, 2, 3, 4, 5]))
        )
        node_b = match.last_device_node
        self.cache.inc_lock_ref(node_b)
        check_conservation("after inc_lock_ref(NodeB)")

        # Unlock — back to all evictable.
        self.cache.dec_lock_ref(node_b)
        check_conservation("after dec_lock_ref(NodeB)")

        # Evict 1 full leaf token.
        self.cache.evict(EvictParams(num_tokens=1, mamba_num=0))
        check_conservation("after evict_full(1)")


if __name__ == "__main__":
    unittest.main()
