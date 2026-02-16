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
**File:** `test_mamba_scheduler_integration.py`

Tests:
- ✅ `test_scheduler_creates_mamba_batch()` - Scheduler creates MambaScheduleBatch
- ✅ `test_scheduler_handles_mamba_states()` - State allocation in ReqToTokenPool
- ✅ `test_scheduler_prefill_decode_transition()` - Prefill → Decode state transition
- ✅ `test_scheduler_multiple_requests()` - Multiple concurrent Mamba requests
- ✅ `test_scheduler_mixed_models()` - Mamba + Transformer in same server (future)

#### 2.2 ModelRunner Integration
**File:** `test_mamba_model_runner_integration.py`

Tests:
- ✅ `test_model_runner_forward_pass()` - ModelRunner.forward() with Mamba
- ✅ `test_model_runner_state_management()` - ModelRunner manages states
- ✅ `test_model_runner_batch_processing()` - Batched inference
- ✅ `test_model_runner_tp_parallelism()` - Tensor parallel Mamba (future)

#### 2.3 Memory Management Integration
**File:** `test_mamba_memory_integration.py`

Tests:
- ✅ `test_memory_allocation()` - States allocated in ReqToTokenPool
- ✅ `test_memory_deallocation()` - States freed on request completion
- ✅ `test_memory_fragmentation()` - No excessive fragmentation
- ✅ `test_memory_limits()` - Graceful handling of OOM

---

### 3. End-to-End Tests

**Location:** `python/sglang/test/srt/`

#### 3.1 Full Pipeline Tests
**File:** `test_mamba_e2e.py`

Tests:
- ✅ `test_mamba_inference_pipeline()` - Full tokenize → forward → detokenize
- ✅ `test_mamba_streaming_generation()` - Streaming output
- ✅ `test_mamba_batch_generation()` - Multiple requests in parallel
- ✅ `test_mamba_long_context()` - Sequences > 8K tokens
- ✅ `test_mamba_with_sampling()` - Temperature, top-k, top-p sampling

#### 3.2 Prefill Cache E2E
**File:** `test_mamba_cache_e2e.py`

Tests:
- ✅ `test_cache_hit_latency_improvement()` - 2nd request faster with cache
- ✅ `test_cache_miss_fallback()` - Correct output on cache miss
- ✅ `test_cache_hit_rate_measurement()` - Cache metrics tracked
- ✅ `test_multi_turn_conversation()` - Conversation with shared prefix

#### 3.3 Performance Tests
**File:** `test_mamba_performance.py`

Tests:
- ✅ `test_throughput_tokens_per_sec()` - Throughput benchmark
- ✅ `test_latency_percentiles()` - p50, p95, p99 latency
- ✅ `test_memory_usage()` - Peak memory consumption
- ✅ `test_batch_size_scaling()` - Linear scaling with batch size

---

## Test Fixtures

**File:** `python/sglang/test/srt/layers/mamba/conftest.py`

Fixtures created:
- ✅ `server_args_fixture` - Standard ServerArgs for tests
- ✅ `mamba_model_config_fixture` - ModelConfig for Mamba-130M
- ✅ `mock_mamba_state` - Random state tensor for testing
- ✅ `mock_token_ids` - Sample token IDs
- ✅ `test_config` - Test constants (batch_size, seq_len, etc.)

---

## Test Infrastructure

### Pytest Configuration
**File:** `python/sglang/test/pytest.ini` (to be created if missing)

```ini
[pytest]
testpaths = python/sglang/test
python_files = test_*.py
python_classes = Test*
python_functions = test_*
markers =
    unit: Unit tests
    integration: Integration tests
    e2e: End-to-end tests
    slow: Slow-running tests (>5s)
    gpu: Requires GPU
addopts =
    -v
    --tb=short
    --cov=python/sglang/srt
    --cov-report=html:phase3/test/coverage.html
    --cov-report=term-missing
```

### Coverage Reporting
- HTML report: `phase3/test/coverage.html`
- Terminal output: Shows missing lines
- Target: 85%+ for new code

---

## Test Execution Strategy

### Phase 3.1 (Current)
✅ Create test structure
✅ Set up fixtures
✅ Document test plan

### Phase 3.2
- Write unit tests as features are implemented
- Run tests after each code change
- Maintain coverage ≥85%

### Phase 3.3
- Write integration tests
- Write E2E tests
- Performance benchmarking

### Phase 3.4
- Regression testing
- Final coverage verification
- Performance validation

---

## Continuous Testing

**Run tests after each change:**
```bash
# Unit tests only
pytest python/sglang/test/srt/layers/mamba/ -v

# Integration tests
pytest python/sglang/test/srt/ -k integration -v

# E2E tests
pytest python/sglang/test/srt/ -k e2e -v

# All tests with coverage
pytest python/sglang/test/srt/ -v --cov --cov-report=html
```

---

## Success Criteria

- [ ] All unit tests pass (100%)
- [ ] All integration tests pass (100%)
- [ ] All E2E tests pass (100%)
- [ ] Code coverage ≥85%
- [ ] No flaky tests
- [ ] Performance benchmarks meet targets
- [ ] Test suite runs in <5 minutes (excluding slow tests)

---

**Status:** Test framework ready for Phase 3.2 implementation
**Last Updated:** 2026-02-16
