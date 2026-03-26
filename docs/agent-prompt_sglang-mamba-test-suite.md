## Agent Prompt: SGLang-Mamba Phased Testing Plan

You are a testing agent with full shell, filesystem, and Python access to a running SGLang-Mamba installation on an isolated VM. Your task is to **plan and implement a comprehensive, phased testing scenario** for this codebase. Read this entire prompt before beginning any work.

The codebase is rooted at the SGLang source tree. You will be running tests against a Mamba-family (SSM/hybrid) model. The test infrastructure is located primarily under `test/` and `phase3/`. Existing registered CI tests live in `test/registered/`. A human-in-the-loop (HITL) chat interface is already available for qualitative smoke checks.

---

### Orientation: What Was Added

This fork extends upstream SGLang with a **stateful Mamba inference architecture**. The stateful components are:

1. **`HybridReqToTokenPool` + `MambaPool`** (`python/sglang/srt/mem_cache/memory_pool.py`) — GPU tensor storage for per-request Mamba conv/temporal states, with a `req_index_to_mamba_index_mapping` for O(1) lookup. [1]0) 

2. **`HybridLinearKVPool`** (`python/sglang/srt/mem_cache/memory_pool.py`) — A hybrid KV pool that stores attention-layer KV cache and exposes Mamba pool references to the radix cache. [2]1) 

3. **`MambaRadixCache`** (`python/sglang/srt/mem_cache/mamba_radix_cache.py`) — A hybrid radix tree with **dual LRU lists** (`full_lru_list` and `mamba_lru_list`), dual lock refs (`full_lock_ref`, `mamba_lock_ref`), tombstone nodes (KV present but Mamba state evicted), and copy-on-write (`cow_mamba`). [3]2) [4]3) 

4. **`ForwardMetadata` / `Mamba2Metadata`** (`python/sglang/srt/layers/attention/mamba/mamba2_metadata.py`) — Per-forward-pass metadata carrying `mamba_cache_indices`, `track_conv_indices`, `track_ssm_h_src/dst`, `track_ssm_final_src/dst`, and chunked-prefill chunk index/offset tensors. [5]4) 

5. **`MambaMixer2` + `hybrid_linear_attn_backend`** (`python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py`) — The Mamba2 SSM kernel integration, including `causal_conv1d_fn`, `causal_conv1d_update`, and the GDN/Lightning hybrid variants. [6]5) 

6. **`mamba_branching_seqlen` / COW** — Computed during `MambaRadixCache._match_post_processor()` when a KV hit extends past the last valid Mamba state (tombstone gap); consumed during prefill to re-anchor the SSM state at the correct chunk-aligned divergence point. [7]6) 

7. **`mamba_scheduler_strategy`** (`python/sglang/srt/server_args.py`) — Two runtime modes: `no_buffer` (default, no overlap) and `extra_buffer` (ping-pong track buffer, supports chunked prefill and speculative decoding).

8. **Snapshot system** (`python/sglang/srt/snapshot/`) — `POST /save_snapshot` and `POST /restore_snapshot` HTTP endpoints; `MambaSnapshotManager` for safetensors-based disk persistence of conv and temporal states; `TierManager`, `ConversationTracker`, `SnapshotPolicy`, `MambaHostPool`. [8]7) 

---

### Existing Test Infrastructure

Familiarize yourself with these before writing anything new:

- **Registered unit tests**: `test/registered/radix_cache/test_mamba_unittest.py` — covers `test_hybrid_linear_kv_pool`, `test_mamba_pool`, `test_mamba_radix_cache_1`. [9]8) 

- **Comprehensive radix cache tests**: `test/registered/radix_cache/test_mamba_radix_cache_comprehensive.py` — 10 tests covering tombstones, LRU integrity, lock ref protection, full-cache eviction, COW, leaf-only eviction, empty cache, evictable size tracking, and `mamba_branching_seqlen`. [10]9) 

- **Phase 3 test plan** (`phase3/test/test_plan.md`) — Documents all planned but not-yet-written integration and E2E tests. Read this to understand what is **✅ done** vs **⬜ planned**. [11]10) 

- **Coverage report** (`phase3/tests/test_coverage_report.md`) — Current coverage targets for `MambaRadixCache` components. [12]11) 

- **Test runner**: `python test/run_suite.py --hw cuda --suite stage-b-test-small-1-gpu` [13]12) 

- **CI registration pattern**: Use `register_cuda_ci(est_time=..., suite="stage-b-test-small-1-gpu")` from `sglang.test.ci.ci_register` for any new test files. [14]13) 

---

### Your Mission: Implement the Following Phases

---

#### PHASE 0 — Environment Verification (Pre-flight)

Before any test, verify the install is healthy:

1. Confirm `pip install -e python/` is complete and `import sglang` succeeds.
2. Run `python -m pytest test/registered/radix_cache/test_mamba_unittest.py -v` and confirm all 3 existing tests pass (these are CPU/lightweight and require no server).
3. Run `python -m pytest test/registered/radix_cache/test_mamba_radix_cache_comprehensive.py -v` and confirm all 10 tests pass.
4. Report any failures with full tracebacks before proceeding.

---

#### PHASE 1 — Stateless Inference Baseline (No Mamba-Specific Architecture Active)

**Goal**: Verify the server boots and serves correct outputs **without touching `MambaRadixCache`, `MambaPool`, or any snapshot machinery**. This gives you a known-good inference baseline to diff against later phases.

**Server launch**: Start the server with `--disable-radix-cache` to bypass `MambaRadixCache` entirely, using the `no_buffer` scheduler strategy (the default). Confirm `mambaish_config` is set (you should see `HybridReqToTokenPool` log lines) but the radix cache code path is skipped. The server exposes `POST /v1/chat/completions` via `python/sglang/srt/entrypoints/http_server.py`.

**Test suite to implement** (`test/registered/radix_cache/test_mamba_baseline_inference.py`):

- `test_health_endpoint` — `GET /health` returns 200.
- `test_single_turn_completion` — Single `/v1/chat/completions` request returns a non-empty response with correct finish reason.
- `test_streaming_completion` — Same with `stream=True`; verify all SSE chunks arrive and the final chunk has `finish_reason`.
- `test_batch_inference_independence` — Send N=4 requests with **identical** prompts concurrently; verify all responses are identical (deterministic, `temperature=0`). This is your baseline for state isolation.
- `test_batch_inference_different_prompts` — Send 4 different prompts; verify each response is semantically distinct.
- `test_long_context` — Send a prompt with a long system prompt (>512 tokens) and verify the model responds without OOM or truncation error.
- `test_sampling_params` — Vary `temperature`, `top_p`, `max_new_tokens`; verify response length and stop conditions are respected.

**HITL check**: After the automated tests pass, use the chat interface to have a short 3-turn conversation manually and confirm coherent responses. Log the exchange.

**Pass criteria**: All automated tests green; HITL chat coherent; no CUDA errors in server logs.

---

#### PHASE 2 — `MambaPool` + `HybridReqToTokenPool` Unit Tests

**Goal**: Verify the memory pool layer in isolation, with no server running.

**What to do**:

1. Re-run the existing `test_mamba_pool` and `test_hybrid_linear_kv_pool` from `test/registered/radix_cache/test_mamba_unittest.py` with verbose output and confirm they pass. [15]14) 

2. **Write additional tests** in a new file `test/registered/radix_cache/test_mamba_pool_extended.py`:
   - `test_pool_exhaustion` — Allocate `mamba_cache_size` slots; assert next alloc fails gracefully (returns `None` or raises expected error).
   - `test_mamba_pool_reuse_on_no_free` — Allocate a req, call `free(req)` without calling `free_mamba_cache(req)`, re-alloc same req object — verify `mamba_pool.available_size()` does **not** decrease further (the leaked slot is reused). This is the exact scenario tested in `test_mamba_pool` lines 130–140. [16]15) 
   - `test_mamba_state_dtype_override` — Use `envs.SGLANG_MAMBA_SSM_DTYPE.override("bfloat16")` context and verify `Mamba2CacheParams` uses bfloat16 for temporal states, as shown in the existing test setup. [17]16) 
   - `test_get_mamba_indices_mapping` — Call `req_to_token_pool.get_mamba_indices(req_pool_indices_tensor)` after alloc and verify the returned indices match `req.mamba_pool_idx`.
   - `test_enable_mamba_extra_buffer_false` — Explicitly construct `HybridReqToTokenPool` with `enable_mamba_extra_buffer=False`; confirm no `mamba_ping_pong_track_buffer` is allocated.

**Pass criteria**: All tests green; no torch memory leaks between tests (check `mamba_pool.available_size()` returns to initial value in teardown).

---

#### PHASE 3 — `MambaRadixCache` Component Tests

**Goal**: Exercise all `MambaRadixCache` operations: insert, match, evict (full and Mamba), tombstone lifecycle, LRU integrity, and COW.

**What to do**:

1. Re-run both existing test files:
   - `python -m pytest test/registered/radix_cache/test_mamba_unittest.py::TestMamba::test_mamba_radix_cache_1 -v`
   - `python -m pytest test/registered/radix_cache/test_mamba_radix_cache_comprehensive.py -v`

2. **Gauntlet tests to write** in `test/registered/radix_cache/test_mamba_radix_cache_gauntlet.py`:

   - `test_interleaved_insert_evict_match` — Insert 10 sequences; interleave `evict_mamba(1)` and `evict_full(1)` calls between each insert; verify `sanity_check()` passes after every operation. The `sanity_check()` method validates LRU list integrity end-to-end. [18]17) 

   - `test_tombstone_does_not_match_mamba` — Insert `[1,2,3]` with mamba state; evict its mamba state via `evict_mamba(1)` (creating a tombstone); then `match_prefix([1,2,3])` and assert `last_node.mamba_value is None` but KV indices are still returned. [19]18) 

   - `test_branching_seqlen_triggered` — Build a scenario where the KV cache hit extends past the last Mamba node (tombstone gap exists); call `match_prefix` and verify `result.mamba_branching_seqlen` is not None and is chunk-aligned. [20]19) 

   - `test_cow_state_independence` — After COW, modify the original cached node's `mamba_value`; verify the copied state on the request is unaffected. This proves true copy semantics. [21]20) 

   - `test_inc_dec_lock_ref_symmetry` — For a 3-node chain (root → A → B), call `inc_lock_ref(B)` and verify `full_lock_ref` propagates up to A and root, and `mamba_lock_ref` locks only B. Then `dec_lock_ref(B)` and verify all counters return to 0. [22]21) 

   - `test_full_evictable_and_protected_size_accounting` — After every `inc_lock_ref`, `dec_lock_ref`, insert, and evict, verify that `full_evictable_size() + full_protected_size()` equals the total tokens cached (i.e., sizes are always conserved). [23]22) 

**Pass criteria**: All tests green; `sanity_check()` passes after every test case (call it in `tearDown`).

---

#### PHASE 4 — Live Server Integration: `MambaRadixCache` + `no_buffer` Strategy

**Goal**: Verify the radix cache is populated and queried correctly during actual server inference, using `mamba_scheduler_strategy=no_buffer` (the default).

**Server launch**: Start server **with** radix cache enabled (do not pass `--disable-radix-cache`), `--mamba-scheduler-strategy no_buffer`.

**Tests to implement** (`test/registered/radix_cache/test_mamba_radix_cache_server_integration.py`):

- `test_cache_hit_on_repeated_prefix` — Send request A with a long system prompt + question 1. Then send request B with the **same** system prompt + question 2. Query `GET /get_memory_pool_size` or inspect server logs; verify the second request has a shorter prefill (cache hit). At `temperature=0`, the answer to question 2 must be deterministic and correct.
- `test_cache_miss_fallback` — Send a request with a prefix that has never been seen; verify correct generation (cache miss does not corrupt output).
- `test_concurrent_shared_prefix` — Send 4 concurrent requests that all share the same long system prompt. Assert that all 4 complete successfully, outputs are independent, and no state contamination occurs (the state isolation invariant: `mamba_lock_ref` semantics).
- `test_multi_turn_conversation_state_continuity` — Simulate a 5-turn conversation using `session_id` / `rid` continuity via the native API. After each turn, the next turn should generate contextually coherent continuations. Log all 5 turns.
- `test_eviction_under_pressure` — Fill the Mamba cache to near-capacity by sending many short distinct requests; verify that new requests still succeed (eviction triggers correctly and the server does not OOM or error).

**Pass criteria**: All automated tests green; HITL: Use the chat interface to do a 5-turn conversation and confirm persistent context.

---

#### PHASE 5 — `Mamba2Metadata` / `ForwardMetadata` Integrity

**Goal**: Verify that the metadata constructed during each forward pass is correct for prefill, decode, and mixed batches.

**Approach**: Write targeted unit tests against `Mamba2Metadata.prepare_decode()` and `Mamba2Metadata.prepare_mixed()` using synthetic `ForwardBatch` objects (no server needed). [24]23) 

**Tests to implement** (`test/registered/radix_cache/test_mamba_metadata.py`):

- `test_prepare_decode_pure_decode_batch` — Verify `num_prefills=0`, `num_decodes=N`, `mixed_metadata=None`.
- `test_prepare_mixed_prefill_only` — Verify `num_prefills=N`, `num_decodes=0`, `mixed_metadata` is populated.
- `test_chunk_indices_offsets_correctness` — Use the static method `Mamba2Metadata._query_start_loc_to_chunk_indices_offsets` with the worked example from the docstring (`query_start_loc=[0,5,10]`, `chunk_size=8`) and assert `chunk_indices=[0,0,1]`, `chunk_offsets=[0,5,0]`. [25]24) 
- `test_has_initial_states_flag` — Provide `context_lens_tensor` with mixed zeros and non-zeros; verify `has_initial_states` tensor is correct and `prep_initial_states` is True only when any non-zero is present.
- `test_mamba_cache_indices_preserved` — Verify `mamba_cache_indices` from the base `ForwardMetadata` is passed through unchanged to the produced `Mamba2Metadata`.

---

#### PHASE 6 — `mamba_scheduler_strategy=extra_buffer`

**Goal**: Verify the ping-pong buffer path works end-to-end. This mode enables overlap scheduling and is required for speculative decoding.

**Server launch**: `--mamba-scheduler-strategy extra_buffer`

**Tests to implement** (`test/registered/radix_cache/test_mamba_extra_buffer.py`):

- `test_extra_buffer_alloc` — Verify that after `HybridReqToTokenPool.alloc([req])` with `enable_mamba_extra_buffer=True`, `req.mamba_ping_pong_track_buffer` is non-None and has the expected size (2 for non-speculative).
- `test_extra_buffer_free_with_keep` — Verify `free_mamba_cache(req, mamba_ping_pong_track_buffer_to_keep=idx)` frees the main slot and all but one ping-pong slot, and the kept slot's tensor data is intact.
- `test_cache_unfinished_req_extra_buffer` — Simulate the `cache_unfinished_req` code path by calling it on a mock req with `mamba_last_track_seqlen` set; verify `mamba_branching_seqlen` is cleared and `prefix_indices` is updated. [26]25) 
- `test_server_inference_extra_buffer_mode` — Do a basic inference request through the running server in `extra_buffer` mode; verify output correctness matches Phase 1 baseline for the same prompt at `temperature=0`.

---

#### PHASE 7 — Snapshot System

**Goal**: Verify `POST /save_snapshot` and `POST /restore_snapshot` end-to-end.

**Server launch**: `--enable-mamba-snapshots` (or `--enable-snapshot-persistence`) in addition to normal args.

**Tests to implement** (`test/registered/radix_cache/test_mamba_snapshot_e2e.py`):

- `test_save_snapshot_returns_success` — After generating some tokens for a request (obtain a `rid`), call `POST /save_snapshot` with that `rid`; assert `success=True` in response.
- `test_restore_snapshot_state_equivalence` — Save a snapshot after turn N. Generate turn N+1 from the live state. Then restore the snapshot and re-generate turn N+1 from the restored state (at `temperature=0`). Assert the two outputs are identical — this proves state fidelity.
- `test_restore_requires_idle_request` — Attempt to restore a snapshot while the request is in `running_batch`; verify `success=False` is returned gracefully.
- `test_snapshot_disk_format` — After save, locate the `.safetensors` and `.json` metadata files on disk and verify they contain `conv_states` and `temporal_states` keys matching the expected layer count.
- `test_snapshot_manager_tier_consistency` — Verify `TierManager` promotes/demotes snapshots between GPU, host (`MambaHostPool`), and disk tiers without data corruption. Write a small mock that exercises the tier transitions directly. [27]26) [28]27) 

**HITL check**: Use the chat interface to generate a 3-turn conversation, save a snapshot after turn 2, then continue to turn 3. Restore the snapshot and ask a variation of turn 3's question; confirm the answer reflects only turns 1–2 context, not turn 3.

---

#### PHASE 8 — Gauntlet / Stress Tests

**Goal**: Break things under load to find race conditions, memory leaks, and eviction policy failures.

**Tests to

### Citations

**File:** test/registered/radix_cache/test_mamba_unittest.py (L1-30)
```python
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
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci

register_cuda_ci(est_time=9, suite="stage-b-test-small-1-gpu")
register_amd_ci(est_time=9, suite="stage-b-test-small-1-gpu-amd")


class TestMamba(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        pass
```

**File:** test/registered/radix_cache/test_mamba_unittest.py (L36-141)
```python
    def test_hybrid_linear_kv_pool(self):
        size = 16
        head_num = 2
        head_dim = 256
        num_layers = 48
        global_interval = 4
        dtype = torch.bfloat16
        device = get_device()
        full_attention_layer_ids = [
            i for i in range(global_interval - 1, num_layers, global_interval)
        ]
        pool = HybridLinearKVPool(
            size=size,
            dtype=dtype,
            page_size=1,
            head_num=head_num,
            head_dim=head_dim,
            full_attention_layer_ids=full_attention_layer_ids,
            enable_kvcache_transpose=False,
            device=device,
            enable_memory_saver=False,
            mamba_pool=None,
        )
        assert pool._transfer_full_attention_id(global_interval - 1) == 0
        assert pool._transfer_full_attention_id(2 * global_interval - 1) == 1
        with self.assertRaises(ValueError) as context:
            pool._transfer_full_attention_id(1)
        self.assertIn(
            "layer_id=1 not in full attention layers:", str(context.exception)
        )

    def test_mamba_pool(self):
        max_num_reqs = 10
        mamba_cache_size = 20
        max_context_len = 128
        device = get_device()
        global_interval = 4
        num_layers = 48
        full_attention_layer_ids = [
            i for i in range(global_interval - 1, num_layers, global_interval)
        ]
        mamba_layers = [
            i for i in range(num_layers) if i not in full_attention_layer_ids
        ]
        shape = Mamba2StateShape.create(
            tp_world_size=1,
            intermediate_size=4096,
            n_groups=16,
            num_heads=32,
            head_dim=128,
            state_size=128,
            conv_kernel=4,
        )

        with envs.SGLANG_MAMBA_SSM_DTYPE.override("bfloat16"):
            mamba2_cache_params = Mamba2CacheParams(shape=shape, layers=mamba_layers)

        req_to_token_pool = HybridReqToTokenPool(
            size=max_num_reqs,
            mamba_size=mamba_cache_size,
            mamba_spec_state_size=max_num_reqs,
            max_context_len=max_context_len,
            device=device,
            enable_memory_saver=False,
            cache_params=mamba2_cache_params,
            enable_mamba_extra_buffer=False,
            speculative_num_draft_tokens=3,
        )

        assert req_to_token_pool.available_size() == max_num_reqs
        assert req_to_token_pool.mamba_pool.available_size() == mamba_cache_size

        sampling_params = SamplingParams(
            temperature=0,
            max_new_tokens=1,
        )
        req = Req(
            rid=0,
            origin_input_text="",
            origin_input_ids=[],
            sampling_params=sampling_params,
        )

        # alloc req
        req_to_token_pool.alloc([req])
        assert req_to_token_pool.available_size() == max_num_reqs - 1
        assert req_to_token_pool.mamba_pool.available_size() == mamba_cache_size - 1

        # free req
        req_to_token_pool.free_mamba_cache(req)
        req_to_token_pool.free(req)
        assert req_to_token_pool.available_size() == max_num_reqs
        assert req_to_token_pool.mamba_pool.available_size() == mamba_cache_size

        # alloc req without free mamba cache
        req.mamba_pool_idx = None
        req_to_token_pool.alloc([req])
        req_to_token_pool.free(req)
        assert req_to_token_pool.available_size() == max_num_reqs
        assert req_to_token_pool.mamba_pool.available_size() == mamba_cache_size - 1

        # alloc again
        req_to_token_pool.alloc([req])
        assert req_to_token_pool.available_size() == max_num_reqs - 1
        assert req_to_token_pool.mamba_pool.available_size() == mamba_cache_size - 1

```

**File:** python/sglang/srt/mem_cache/mamba_radix_cache.py (L63-115)
```python
class TreeNode:

    counter = 0
    last_access_time_counter_float = float64(1.0)

    def __init__(self, id: Optional[int] = None):
        self.children = defaultdict(TreeNode)
        self.parent: TreeNode = None
        self.key: RadixKey = None
        self.value: Optional[torch.Tensor] = None
        self.mamba_value: Optional[torch.Tensor] = None
        # invariant: for any node, if mamba_lock_ref is locked, full_lock_ref must be locked;
        # if full_lock_ref is locked, mamba_lock_ref doesn't need to be locked. So,
        # full_lock_ref is always >= mamba_lock_ref.
        # for full_lock, once it is locked, its parent must be locked as well
        # for mamba_lock, it only need lock node itself
        self.full_lock_ref = 0
        self.mamba_lock_ref = 0
        # last access time is only used for sanity check. LRU is maintained by the lru list.
        self.last_access_time = get_last_access_time()

        self.hit_count = 0
        # store the host indices of KV cache
        self.host_value = None

        # for lru list, invariant:
        # 1. prev has greater last_access_time
        # 2. next has smaller last_access_time
        self.prev = None
        self.next = None
        self.mamba_prev = None
        self.mamba_next = None

        self.id = TreeNode.counter if id is None else id
        TreeNode.counter += 1

    @property
    def evicted(self):
        return self.value is None

    @property
    def backuped(self):
        return self.host_value is not None

    def __lt__(self, other: "TreeNode"):
        return self.last_access_time < other.last_access_time


def get_last_access_time() -> float64:
    ret = TreeNode.last_access_time_counter_float
    TreeNode.last_access_time_counter_float += 1.0
    return ret

```

**File:** python/sglang/srt/mem_cache/mamba_radix_cache.py (L371-422)
```python
class MambaRadixCache(BasePrefixCache):
    def __init__(self, params: CacheInitParams):
        assert isinstance(
            params.token_to_kv_pool_allocator, TokenToKVPoolAllocator
        ) or isinstance(params.token_to_kv_pool_allocator, PagedTokenToKVPoolAllocator)
        self.req_to_token_pool: HybridReqToTokenPool = params.req_to_token_pool
        self.token_to_kv_pool_allocator = params.token_to_kv_pool_allocator

        self.page_size = params.page_size
        self.disable = params.disable
        self.enable_mamba_extra_buffer = params.enable_mamba_extra_buffer

        if not self.enable_mamba_extra_buffer:
            assert (
                self.page_size == 1
            ), f"Page size must be 1 for MambaRadixCache v1, got {self.page_size}"

        if self.token_to_kv_pool_allocator:
            self.device = self.token_to_kv_pool_allocator.device
        else:
            self.device = torch.device("cpu")

        if params.enable_metrics:
            self.init_metrics_collector()

        if self.page_size == 1:
            self.key_match_fn = _key_match_page_size1
            self.get_child_key_fn = get_child_key
        else:
            self.key_match_fn = partial(_key_match_paged, page_size=self.page_size)
            self.get_child_key_fn = partial(get_child_key, page_size=self.page_size)
        self.reset()

    ##### Public API #####

    def supports_mamba(self) -> bool:
        return True

    def reset(self) -> None:
        self.root_node = TreeNode()
        self.root_node.key = RadixKey([], None)
        self.root_node.value = []
        self.root_node.full_lock_ref = 1
        self.root_node.mamba_lock_ref = 1
        self.full_evictable_size_ = 0
        self.mamba_evictable_size_ = 0
        self.full_protected_size_ = 0
        self.mamba_protected_size_ = 0
        # LRU lists are used to maintain the order of eviction of the nodes in the tree
        self.full_lru_list = LRUList(mamba=False)
        self.mamba_lru_list = LRUList(mamba=True)

```

**File:** python/sglang/srt/mem_cache/mamba_radix_cache.py (L556-672)
```python
    def cache_unfinished_req(self, req: Req, chunked=False) -> None:
        """Cache request when it is unfinished."""

        def _skip_cache_unfinished_req(req: Req) -> None:
            kv_indices = self.req_to_token_pool.req_to_token[
                req.req_pool_idx, : len(req.fill_ids)
            ]

            # `req.prefix_indices` will be used in `PrefillAdder::add_chunked_req` later
            req.prefix_indices = kv_indices.to(dtype=torch.int64, copy=True)
            return

        token_ids = req.fill_ids
        cache_len = (
            req.mamba_last_track_seqlen
            if self.enable_mamba_extra_buffer
            else len(token_ids)
        )
        if self.disable or cache_len is None:
            return _skip_cache_unfinished_req(req)

        kv_indices_orig = self.req_to_token_pool.req_to_token[
            req.req_pool_idx, : len(token_ids)
        ]
        # kv_indices is the kv indices to be cached
        kv_indices = kv_indices_orig[:cache_len]
        if self.page_size != 1:
            page_aligned_len = len(kv_indices) // self.page_size * self.page_size
            page_aligned_kv_indices = kv_indices[:page_aligned_len].to(
                dtype=torch.int64, copy=True
            )
        else:
            page_aligned_len = len(kv_indices)
            page_aligned_kv_indices = kv_indices.to(dtype=torch.int64, copy=True)

        assert page_aligned_len == len(
            kv_indices
        ), f"page_aligned_len != len(kv_indices), {page_aligned_len=}, {len(kv_indices)=}, {cache_len=}, {self.page_size=}, {FLA_CHUNK_SIZE=}"

        page_aligned_token_ids = token_ids[:page_aligned_len]

        if self.enable_mamba_extra_buffer:
            # copy from the ping pong track buffer
            mamba_ping_pong_track_buffer_to_keep = (
                self.req_to_token_pool.get_mamba_ping_pong_other_idx(
                    req.mamba_next_track_idx
                )
            )
            mamba_value = (
                req.mamba_ping_pong_track_buffer[mamba_ping_pong_track_buffer_to_keep]
                .unsqueeze(-1)
                .clone()
            )
        else:
            mamba_value = self.req_to_token_pool.get_mamba_indices(
                req.req_pool_idx
            ).unsqueeze(-1)
        # radix tree mamba value is forked from req space
        mamba_value_forked = self.req_to_token_pool.mamba_pool.fork_from(mamba_value)

        # if alloc mamba cache failed, do evict and alloc again
        if mamba_value_forked is None:
            self.evict(EvictParams(num_tokens=0, mamba_num=1))
            mamba_value_forked = self.req_to_token_pool.mamba_pool.fork_from(
                mamba_value
            )
            assert mamba_value_forked is not None, "Can not alloc mamba cache"
        result = self.insert(
            InsertParams(
                key=RadixKey(page_aligned_token_ids, req.extra_key),
                value=page_aligned_kv_indices,
                mamba_value=mamba_value_forked,
            )
        )
        new_prefix_len, mamba_exist = result.prefix_len, result.mamba_exist
        self.token_to_kv_pool_allocator.free(
            kv_indices[req.cache_protected_len : new_prefix_len]
        )
        # there is a mamba cache in radix cache, release it
        if mamba_exist:
            self.req_to_token_pool.mamba_pool.free(mamba_value_forked)

        # The prefix indices could be updated, reuse it
        match_result = self.match_prefix(
            MatchPrefixParams(key=RadixKey(page_aligned_token_ids, req.extra_key))
        )
        (new_indices, new_last_node) = (
            match_result.device_indices,
            match_result.last_device_node,
        )

        if not mamba_exist:
            assert torch.equal(new_last_node.mamba_value, mamba_value_forked)

        assert (
            req.cache_protected_len <= len(new_indices) + self.page_size - 1
        ), f"{req.cache_protected_len=}, {len(new_indices)=}, {len(page_aligned_token_ids)=}, {mamba_exist=}"
        assert new_prefix_len <= len(
            new_indices
        ), f"{new_prefix_len=}, {len(new_indices)=}"

        self.req_to_token_pool.write(
            (req.req_pool_idx, slice(req.cache_protected_len, len(new_indices))),
            new_indices[req.cache_protected_len :],
        )

        self.dec_lock_ref(req.last_node)
        self.inc_lock_ref(new_last_node)

        # `req.prefix_indices` will be used in `PrefillAdder::add_chunked_req` later
        # NOTE: this is needed for both page_size == 1 and page_size > 1
        req.prefix_indices = torch.cat(
            [new_indices, kv_indices_orig[len(new_indices) :]]
        )
        req.cache_protected_len = len(new_indices)
        req.mamba_last_track_seqlen = None
        req.last_node = new_last_node
```

**File:** python/sglang/srt/mem_cache/mamba_radix_cache.py (L728-761)
```python
    def evict_mamba(self, mamba_num: int) -> int:
        """Evict mamba states. Returns the number of mamba states evicted."""
        if self.disable or mamba_num <= 0:
            return 0
        # get the least recently used node that is not locked, doesn't have to be a leaf
        x = self.mamba_lru_list.get_lru_no_lock()
        mamba_num_evicted = 0
        # evict lru leaf nodes until mamba_num_tokens is reached
        while mamba_num_evicted < mamba_num and (self.mamba_lru_list.in_list(x)):
            assert x.mamba_value is not None, f"node has no mamba value, {x.id=}"
            assert (
                len(x.mamba_value) == 1
            ), f"node has abnormal mamba length, {x.id=}, {len(x.mamba_value)=}"
            assert x != self.root_node, f"root node is not evictable, {x.id=}"
            assert x.mamba_lock_ref == 0, f"node is in use by mamba kv indices, {x.id=}"

            if len(x.children) > 0:
                # 1. an internal node, free mamba tokens.
                self.req_to_token_pool.mamba_pool.free(x.mamba_value)
                mamba_num_evicted += len(x.mamba_value)

                # 2. get the next node, update the lru lists
                x_next = self.mamba_lru_list.get_prev_no_lock(x)
                self.mamba_lru_list.remove_node(x)

                # 3. tombstone the node
                self._tombstone_internal_node(x)
            else:
                _, mamba_evicted_delta, _, x_next = self._evict_leaf_node(x, True)
                mamba_num_evicted += mamba_evicted_delta

            x = x_next

        return mamba_num_evicted
```

**File:** python/sglang/srt/mem_cache/mamba_radix_cache.py (L788-843)
```python
    def inc_lock_ref(self, node: TreeNode) -> Optional[int]:
        """
        Increment the lock reference count for the node.
        It locks the full_lock_ref for nodes between the [last node, root), exclusive.
        It locks the mamba_lock_ref for current node if its mamba_value exists.
        """
        if self.disable:
            return None

        # protect mamba value in current node if it exists
        if node.mamba_value is not None:
            if node.mamba_lock_ref == 0:
                self.mamba_evictable_size_ -= len(node.mamba_value)
                self.mamba_protected_size_ += len(node.mamba_value)
            node.mamba_lock_ref += 1

        while node != self.root_node:
            # lock full from node to root
            assert (
                node.full_lock_ref >= 0
            ), f"inc_lock_ref on node with {node.full_lock_ref=}, {node.id=}"
            if node.full_lock_ref == 0:
                self.full_evictable_size_ -= len(node.value)
                self.full_protected_size_ += len(node.value)
            node.full_lock_ref += 1
            node = node.parent
        return None

    def dec_lock_ref(self, node: TreeNode):
        """
        Decrement the lock reference count for the node.
        It unlocks the full_lock_ref for nodes between the [last node, root), exclusive.
        It unlocks the mamba_lock_ref for current node if its mamba_value exists.
        """
        if self.disable:
            return None

        if node.mamba_value is not None:
            assert (
                node.mamba_lock_ref > 0
            ), f"dec_lock_ref on node with {node.mamba_lock_ref=}, {node.id=}"
            if node.mamba_lock_ref == 1:
                self.mamba_evictable_size_ += len(node.mamba_value)
                self.mamba_protected_size_ -= len(node.mamba_value)
            node.mamba_lock_ref -= 1

        while node != self.root_node:
            assert (
                node.full_lock_ref > 0
            ), f"dec_lock_ref on node with {node.full_lock_ref=}, {node.id=}"
            if node.full_lock_ref == 1:
                self.full_evictable_size_ += len(node.value)
                self.full_protected_size_ -= len(node.value)
            node.full_lock_ref -= 1
            node = node.parent

```

**File:** python/sglang/srt/mem_cache/mamba_radix_cache.py (L844-848)
```python
    def sanity_check(self):
        if self.disable:
            return
        self.full_lru_list.sanity_check(self)
        self.mamba_lru_list.sanity_check(self)
```

**File:** python/sglang/srt/mem_cache/mamba_radix_cache.py (L850-878)
```python
    def evictable_size(self) -> Tuple[int, int]:
        # Note: use full_evictable_size() and mamba_evictable_size() instead.
        raise NotImplementedError

    def full_evictable_size(self) -> int:
        return self.full_evictable_size_

    def mamba_evictable_size(self) -> int:
        return self.mamba_evictable_size_

    # Note: this is expensive, only use for debug
    def full_lru_list_evictable_size(self) -> int:
        return self.full_lru_list.sanity_check_evictable_size()

    # Note: this is expensive, only use for debug
    def mamba_lru_list_evictable_size(self) -> int:
        return self.mamba_lru_list.sanity_check_evictable_size()

    def protected_size(self) -> Tuple[int, int]:
        # Note: use full_protected_size() and mamba_protected_size() instead.
        raise NotImplementedError

    def full_protected_size(self) -> int:
        # protected size refers to the size of the full cache that is locked
        return self.full_protected_size_

    def mamba_protected_size(self) -> int:
        # protected size refers to the size of the mamba cache that is locked
        return self.mamba_protected_size_
```

**File:** python/sglang/srt/mem_cache/mamba_radix_cache.py (L984-1000)
```python
        if len(value) > best_value_len:
            mamba_cache_chunk_size = get_global_server_args().mamba_cache_chunk_size
            mamba_cache_chunk_aligned_seqlen = (
                sum(len(v) for v in value) // mamba_cache_chunk_size
            ) * mamba_cache_chunk_size
            mamba_branching_seqlen = (
                mamba_cache_chunk_aligned_seqlen
                if mamba_cache_chunk_aligned_seqlen > 0
                else None
            )
        else:
            mamba_branching_seqlen = None

        # Copy mamba state to req local space if cow is true
        if cow_mamba and last_node.mamba_value is not None:
            # for reqs without mamba cache
            if req.mamba_pool_idx is None:
```

**File:** python/sglang/srt/layers/attention/mamba/mamba2_metadata.py (L26-65)
```python
@dataclass(kw_only=True)
class ForwardMetadata:
    query_start_loc: torch.Tensor
    mamba_cache_indices: torch.Tensor
    # For topk > 1 eagle
    retrieve_next_token: Optional[torch.Tensor] = None
    retrieve_next_sibling: Optional[torch.Tensor] = None
    retrieve_parent_token: Optional[torch.Tensor] = None
    # For prefill radix cache
    track_conv_indices: Optional[torch.Tensor] = None
    track_ssm_h_src: Optional[torch.Tensor] = None
    track_ssm_h_dst: Optional[torch.Tensor] = None
    track_ssm_final_src: Optional[torch.Tensor] = None
    track_ssm_final_dst: Optional[torch.Tensor] = None

    is_target_verify: bool = False
    draft_token_num: int = 1


@dataclass(kw_only=True)
class Mamba2Metadata(ForwardMetadata):
    """stable metadata across all mamba2 layers in the forward pass"""

    num_prefills: int
    num_prefill_tokens: int
    num_decodes: int

    @dataclass(kw_only=True, frozen=True)
    class MixedMetadata:
        has_initial_states: torch.Tensor
        prep_initial_states: bool

        chunk_size: int
        seq_idx: torch.Tensor
        chunk_indices: torch.Tensor
        chunk_offsets: torch.Tensor

        extend_seq_lens_cpu: list[int]

    mixed_metadata: MixedMetadata | None = None
```

**File:** python/sglang/srt/layers/attention/mamba/mamba2_metadata.py (L68-150)
```python
    @staticmethod
    def _query_start_loc_to_chunk_indices_offsets(
        query_start_loc: torch.Tensor, chunk_size: int, total_seqlens: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            query_start_loc (torch.Tensor): 1D tensor of cumulative sequence
                lengths, shape (num_seqs + 1,).
                The first element should be 0. Each entry represents the starting
                index of a sequence in the flattened token array.
            chunk_size (int): The size of each physical mamba chunk
                (number of tokens per chunk).
            total_seqlens (int): The total number of tokens in the batch.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple containing:
                - chunk_indices (torch.Tensor): 1D tensor of indices
                    indicating the physical chunk for each logical chunk.
                - chunk_offsets (torch.Tensor): 1D tensor of offsets
                    indicating the starting index of each logical chunk within
                    its physical chunk.

        This function computes the chunk indices and offsets for the given
        query_start_loc and chunk_size. Both are tensors of integers with length N,
        where N is the number of logical (pseudo) chunks.
        A logical chunk is a sequence of tokens that are all part of the same
        sequence and are all in the same physical mamba chunk.
        In other words, a logical chunk changes every time we cross a sequence
        boundary or a physical mamba chunk boundary.
        Logical chunks are needed to handle batched requests with initial states
        (see _state_passing_fwd and _chunk_scan_fwd).
        The chunk_indices tensor contains the index of the physical chunk for each
        logical chunk.
        The chunk_offsets tensor contains the offset (AKA starting index) of the
        logical chunk in the physical chunk.

        Example:
        query_start_loc = [0, 5, 10]
        chunk_size = 8
        total_seqlens = 10
        -> chunk_indices = [0, 0, 1]
        -> chunk_offsets = [0, 5, 0]

        In this example, we have 2 sequences, each with 5 tokens. The physical
        chunk size is 8 tokens.
        We have three logical chunks:
        - the first logical chunk starts at token 0 in the first physical chunk
            and contains all 5 tokens from the first sequence
        - the second logical chunk starts at token 5 in the first physical chunk
            and contains first 3 tokens from the second sequence
        - the third logical chunk starts at token 0 in the second physical chunk
            and contains the remaining 2 tokens from the second sequence
        """

        cu_seqlens = query_start_loc[1:]  # remove prepended 0

        # outputs will have length expansion of chunks that do not divide
        # chunk_size
        N = (
            math.ceil(total_seqlens / chunk_size)
            + (cu_seqlens[:-1] % chunk_size > 0).sum()
        )
        chunk_indices = torch.arange(N, dtype=torch.int, device=query_start_loc.device)
        chunk_offsets = torch.zeros(
            (N,), dtype=torch.int, device=query_start_loc.device
        )

        p = 0  # num of insertions
        for s, e in zip(cu_seqlens[:-1], cu_seqlens[1:]):

            # if does not divide chunk_size, then there is one chunk insertion
            p += s % chunk_size > 0

            # get the dimensions
            # - the + 1 for _e is to shift the boundary by one chunk
            # - this shifting is not needed if chunk_size divides e
            _s, _e = s // chunk_size + p, e // chunk_size + p + (e % chunk_size > 0)

            # adjust indices and offsets
            chunk_indices[_s:_e] -= p
            chunk_offsets[_s] = s % chunk_size

        return chunk_indices, chunk_offsets
```

**File:** python/sglang/srt/layers/attention/mamba/mamba2_metadata.py (L152-248)
```python
    @staticmethod
    def prepare_decode(
        forward_metadata: ForwardMetadata,
        seq_lens: torch.Tensor,
        *,
        is_target_verify: bool,
        draft_token_num: int,
    ) -> "Mamba2Metadata":
        """This path is run during CUDA graph capture, i.e. decode only, so `num_prefills` is 0"""
        return Mamba2Metadata(
            query_start_loc=forward_metadata.query_start_loc,
            mamba_cache_indices=forward_metadata.mamba_cache_indices,
            retrieve_next_token=forward_metadata.retrieve_next_token,
            retrieve_next_sibling=forward_metadata.retrieve_next_sibling,
            retrieve_parent_token=forward_metadata.retrieve_parent_token,
            num_decodes=len(seq_lens),
            num_prefills=0,
            num_prefill_tokens=0,
            is_target_verify=is_target_verify,
            draft_token_num=draft_token_num,
        )

    @classmethod
    def prepare_mixed(
        cls,
        forward_metadata: ForwardMetadata,
        chunk_size: int,
        forward_batch: ForwardBatch,
    ) -> "Mamba2Metadata":
        """This path cannot run with CUDA graph, as it contains extend requests."""
        if forward_batch.extend_num_tokens is None:
            draft_token_num = (
                forward_batch.spec_info.draft_token_num
                if forward_batch.spec_info is not None
                else 1
            )
            return cls.prepare_decode(
                forward_metadata,
                forward_batch.seq_lens,
                is_target_verify=forward_batch.forward_mode.is_target_verify(),
                draft_token_num=draft_token_num,
            )
        num_prefills = len(forward_batch.extend_seq_lens)
        num_prefill_tokens = forward_batch.extend_num_tokens
        num_decodes = len(forward_batch.seq_lens) - num_prefills
        context_lens_tensor = forward_batch.extend_prefix_lens
        assert context_lens_tensor is not None
        # precompute flag to avoid device syncs later
        has_initial_states = context_lens_tensor > 0
        prep_initial_states = torch.any(has_initial_states[:num_prefills]).item()

        query_start_loc = forward_metadata.query_start_loc[: num_prefills + 1]
        seq_idx = torch.repeat_interleave(
            torch.arange(
                num_prefills, dtype=torch.int32, device=query_start_loc.device
            ),
            query_start_loc.diff(),
            output_size=num_prefill_tokens,
        )
        seq_idx.unsqueeze_(0)

        # We compute metadata for chunked prefill once at the top level model
        # forward and reuse them in mamba layers. If not needed, they will be
        # ignored inside mamba kernels.
        chunk_offsets, chunk_indices = None, None
        if prep_initial_states:
            chunk_indices, chunk_offsets = (
                cls._query_start_loc_to_chunk_indices_offsets(
                    query_start_loc, chunk_size, num_prefill_tokens
                )
            )

        draft_token_num = (
            getattr(forward_batch.spec_info, "draft_token_num", 1)
            if forward_batch.spec_info is not None
            else 1
        )
        return Mamba2Metadata(
            query_start_loc=query_start_loc,
            mamba_cache_indices=forward_metadata.mamba_cache_indices,
            retrieve_next_token=forward_metadata.retrieve_next_token,
            retrieve_next_sibling=forward_metadata.retrieve_next_sibling,
            retrieve_parent_token=forward_metadata.retrieve_parent_token,
            num_prefills=num_prefills,
            num_prefill_tokens=num_prefill_tokens,
            num_decodes=num_decodes,
            is_target_verify=forward_batch.forward_mode.is_target_verify(),
            draft_token_num=draft_token_num,
            mixed_metadata=cls.MixedMetadata(
                has_initial_states=has_initial_states,
                prep_initial_states=prep_initial_states,
                chunk_size=chunk_size,
                seq_idx=seq_idx,
                chunk_indices=chunk_indices,
                chunk_offsets=chunk_offsets,
                extend_seq_lens_cpu=forward_batch.extend_seq_lens_cpu,
            ),
```

**File:** python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py (L1-44)
```python
import logging
import math
from typing import Optional, Tuple, Union

import torch
import triton
import triton.language as tl
from einops import rearrange

from sglang.srt.environ import Envs
from sglang.srt.layers.attention.base_attn_backend import AttentionBackend
from sglang.srt.layers.attention.fla.fused_gdn_gating import fused_gdn_gating
from sglang.srt.layers.attention.fla.fused_recurrent import (
    fused_recurrent_gated_delta_rule_update,
)
from sglang.srt.layers.attention.fla.fused_sigmoid_gating_recurrent import (
    fused_sigmoid_gating_delta_rule_update,
)
from sglang.srt.layers.attention.linear.lightning_attn import (
    BailingLinearKernel,
    linear_decode_forward_triton,
)
from sglang.srt.layers.attention.linear.linear_metadata import BailingLinearMetadata
from sglang.srt.layers.attention.linear.seg_la import SegLaMeta, seg_la_fwd
from sglang.srt.layers.attention.mamba.causal_conv1d_triton import (
    PAD_SLOT_ID,
    causal_conv1d_fn,
    causal_conv1d_update,
)
from sglang.srt.layers.attention.mamba.mamba import MambaMixer2
from sglang.srt.layers.attention.mamba.mamba2_metadata import (
    ForwardMetadata,
    Mamba2Metadata,
)
from sglang.srt.layers.radix_attention import RadixAttention
from sglang.srt.layers.radix_linear_attention import RadixLinearAttention
from sglang.srt.mem_cache.memory_pool import HybridReqToTokenPool, MambaPool
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
from sglang.srt.model_executor.model_runner import ModelRunner
from sglang.srt.server_args import get_global_server_args
from sglang.srt.speculative.eagle_info import EagleDraftInput, EagleVerifyInput
from sglang.srt.speculative.spec_info import SpecInput
from sglang.srt.utils import cpu_has_amx_support, is_cpu, is_cuda, is_npu
from sglang.srt.utils.common import rank0_log
```

**File:** python/sglang/srt/snapshot/mamba_snapshot.py (L1-1)
```python
"""
```

**File:** test/registered/radix_cache/test_mamba_radix_cache_comprehensive.py (L1-55)
```python
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


```

**File:** phase3/test/test_plan.md (L1-100)
```markdown
# Phase 3 Test Plan

**Date:** 2026-02-16
**Phase:** 3.1 - Test Framework Setup
**Agent:** Testing & Validation Specialist

---

## Test Coverage Goals

**Target:** 85%+ code coverage for all new Mamba integration code

---

## Test Categories

### 1. Unit Tests

**Location:** `python/sglang/test/srt/layers/mamba/`

#### 1.1 Engine Integration Tests
**File:** `test_mamba_engine.py`

Tests:
- ✅ `test_engine_init_with_mamba_model()` - Engine initialization with Mamba
- ✅ `test_engine_server_args_passed_correctly()` - Verify server_args propagation
- ✅ `test_engine_mamba_config_loading()` - Mamba config loaded from HuggingFace

#### 1.2 State Management Tests
**File:** `test_mamba_state_manager.py`

Tests:
- ✅ `test_state_initialization()` - State tensors initialized correctly
- ✅ `test_state_update()` - State updates during forward pass
- ✅ `test_state_batching()` - Multiple requests in same batch
- ✅ `test_state_isolation()` - States don't leak between requests
- ✅ `test_state_device_placement()` - States on correct GPU/device

#### 1.3 RadixCache Tests
**File:** `test_mamba_radix_cache.py`

Tests:
- ✅ `test_cache_mamba_state()` - Store state with token prefix
- ✅ `test_retrieve_mamba_state()` - Retrieve cached state
- ✅ `test_cache_hit()` - Cache hit on matching prefix
- ✅ `test_cache_miss()` - Cache miss on non-matching prefix
- ✅ `test_cache_eviction()` - LRU eviction when cache full
- ✅ `test_cache_invalidation()` - Invalidate stale states

#### 1.4 Chunked Prefill Tests
**File:** `test_mamba_chunked_prefill.py`

Tests:
- ✅ `test_chunked_vs_non_chunked()` - Output equivalence
- ✅ `test_state_continuity_across_chunks()` - State propagation
- ✅ `test_chunk_size_edge_cases()` - seq_len < chunk_size, seq_len % chunk_size != 0
- ✅ `test_chunked_prefill_cache_integration()` - Works with RadixCache
- ✅ `test_chunked_prefill_performance()` - Latency within acceptable range

#### 1.5 Snapshot Tests (Existing)
**File:** `test_mamba_snapshot.py`

Tests:
- ✅ Snapshot save/load (already implemented)
- ✅ Metadata serialization (already implemented)
- ✅ Multi-tier management (already implemented)

---

### 2. Integration Tests

**Location:** `python/sglang/test/srt/`

#### 2.1 Scheduler Integration
**File:** `test_mamba_scheduler_integration.py` *(planned — not yet written)*

Tests:
- ⬜ `test_scheduler_creates_mamba_batch()` - Scheduler creates MambaScheduleBatch
- ⬜ `test_scheduler_handles_mamba_states()` - State allocation in ReqToTokenPool
- ⬜ `test_scheduler_prefill_decode_transition()` - Prefill → Decode state transition
- ⬜ `test_scheduler_multiple_requests()` - Multiple concurrent Mamba requests
- ⬜ `test_scheduler_mixed_models()` - Mamba + Transformer in same server (future)

#### 2.2 ModelRunner Integration
**File:** `test_mamba_model_runner_integration.py` *(planned — not yet written)*

Tests:
- ⬜ `test_model_runner_forward_pass()` - ModelRunner.forward() with Mamba
- ⬜ `test_model_runner_state_management()` - ModelRunner manages states
- ⬜ `test_model_runner_batch_processing()` - Batched inference
- ⬜ `test_model_runner_tp_parallelism()` - Tensor parallel Mamba (future)

#### 2.3 Memory Management Integration
**File:** `test_mamba_memory_integration.py` *(planned — not yet written)*

Tests:
- ⬜ `test_memory_allocation()` - States allocated in ReqToTokenPool
- ⬜ `test_memory_deallocation()` - States freed on request completion
- ⬜ `test_memory_fragmentation()` - No excessive fragmentation
- ⬜ `test_memory_limits()` - Graceful handling of OOM
```

**File:** phase3/tests/test_coverage_report.md (L1-50)
```markdown
# MambaRadixCache Test Coverage Report

**Generated:** 2026-02-16
**Phase:** 3.2 - Core Implementation
**Status:** ✅ Comprehensive Test Suite Created

---

## 📊 Test Coverage Summary

### Existing Tests
- **File:** `test/registered/radix_cache/test_mamba_unittest.py`
- **Test Count:** 3 tests
- **Coverage:**
  - ✅ Basic insert/match operations
  - ✅ Eviction (full and Mamba)
  - ✅ Copy-on-write (COW) functionality
  - ✅ Hybrid pool management

### NEW: Comprehensive Tests
- **File:** `test/registered/radix_cache/test_mamba_radix_cache_comprehensive.py`
- **Test Count:** 10 tests
- **Coverage:** Advanced features and edge cases

---

## 🧪 New Test Cases Added

### 1. `test_tombstone_node_creation`
**Purpose:** Validate tombstone node behavior (nodes with KV cache but no Mamba state)

**Test Scenario:**
- Insert sequence [1, 2, 3]
- Insert longer sequence [1, 2, 3, 4, 5]
- Evict Mamba state from [1, 2, 3] → creates tombstone
- Verify match returns empty for tombstone node
- Verify match succeeds for non-tombstone node

**Coverage:**
- Tombstone creation via eviction
- Match behavior with tombstones
- KV cache retention after Mamba eviction

---

### 2. `test_lru_list_integrity`
**Purpose:** Verify LRU lists maintain correct ordering

**Test Scenario:**
- Insert 3 sequences: [1,2], [3,4], [5,6]
```

**File:** test/README.md (L116-124)
```markdown
python test/run_suite.py --hw cuda --suite stage-b-test-small-1-gpu

# Run nightly tests
python test/run_suite.py --hw cuda --suite nightly-1-gpu --nightly

# With auto-partitioning (for parallel CI jobs)
python test/run_suite.py --hw cuda --suite stage-b-test-small-1-gpu \
    --auto-partition-id 0 --auto-partition-size 4
```
```

**File:** python/sglang/srt/snapshot/tier_manager.py (L1-1)
```python
"""
```

**File:** python/sglang/srt/snapshot/mamba_host_pool.py (L1-1)
```python
"""
```
