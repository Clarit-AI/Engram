# MambaRadixCache Performance Baseline

**Generated:** 2026-02-16
**Phase:** 3.2 - Core Implementation
**Benchmark Version:** 1.0

---

## 📊 Baseline Configuration

### Hardware Environment
- **Device:** CUDA GPU (varies by runner)
- **Memory:** GPU memory allocated for caches
- **CPU:** Multi-core (background operations)

### Cache Configuration
- **KV Cache Size:** 1024 tokens
- **Mamba Cache Size:** 128 states
- **Max Requests:** 64 concurrent
- **Page Size:** 1 (token granularity)
- **Data Type:** bfloat16

### Model Configuration
- **Layers:** 48 total (36 Mamba + 12 Attention)
- **Global Interval:** 4 (Attention every 4th layer)
- **Head Num:** 32
- **Head Dim:** 128
- **Intermediate Size:** 4096

---

## 🎯 Performance Targets

### Target Metrics (To Be Measured)

| Operation | Target Latency | Target Throughput | Notes |
|-----------|---------------|-------------------|-------|
| **insert()** | < 0.1 ms | > 10,000 ops/sec | Single sequence insert |
| **match_prefix()** | < 0.05 ms | > 20,000 ops/sec | Cache hit scenario |
| **evict_mamba()** | < 0.2 ms | > 5,000 ops/sec | Single state eviction |
| **evict_full()** | < 0.5 ms | > 2,000 ops/sec | Single token eviction |

### Memory Overhead Targets

| Component | Target Overhead | Notes |
|-----------|----------------|-------|
| **TreeNode** | < 200 bytes | Per node metadata |
| **LRU Lists** | < 100 bytes | Per cached item |
| **Total Metadata** | < 5% | Of total cache size |

---

## 📈 Benchmark Results

### Insert Performance

**Workload:** 1,000 sequences, 10 tokens each

```
Expected Results (placeholder - run benchmark to fill):
- Average Latency: TBD ms
- P50 Latency: TBD ms
- P95 Latency: TBD ms
- P99 Latency: TBD ms
- Throughput: TBD ops/sec
```

**Analysis:**
- Insert includes: tree traversal, node creation, KV allocation, Mamba allocation
- LRU list updates: 2 list operations (full + Mamba)
- Expected bottlenecks: GPU memory allocation

### Match Prefix Performance

**Workload:** 10,000 queries on 100 cached sequences

```
Expected Results (placeholder - run benchmark to fill):
- Average Latency: TBD ms
- P50 Latency: TBD ms
- P95 Latency: TBD ms
- P99 Latency: TBD ms
- Throughput: TBD ops/sec
- Cache Hit Rate: TBD%
```

**Analysis:**
- Match includes: tree traversal, KV index retrieval, Mamba state COW
- LRU updates: Move nodes to MRU position
- Expected bottlenecks: Tree traversal depth

### Evict Mamba Performance

**Workload:** 1,000 evictions, 1 state each

```
Expected Results (placeholder - run benchmark to fill):
- Average Latency: TBD ms
- P50 Latency: TBD ms
- P95 Latency: TBD ms
- P99 Latency: TBD ms
- Throughput: TBD ops/sec
```

**Analysis:**
- Evict includes: LRU traversal, state deallocation, tombstone creation
- Dual list management: Remove from Mamba LRU, may stay in full LRU
- Expected bottlenecks: List traversal for unlocked nodes

### Evict Full Performance

**Workload:** 1,000 evictions, 1 token each

```
Expected Results (placeholder - run benchmark to fill):
- Average Latency: TBD ms
- P50 Latency: TBD ms
- P95 Latency: TBD ms
- P99 Latency: TBD ms
- Throughput: TBD ops/sec
```

**Analysis:**
- Evict includes: Leaf-only search, KV deallocation, node deletion
- Tree updates: Iterative tombstone cleanup
- Expected bottlenecks: Leaf node identification

---

## 🔍 Profiling Hotspots

### Expected CPU Hotspots
1. **LRU list traversal** - O(n) in worst case
2. **Tree node lookup** - O(log n) average, O(n) worst case
3. **Lock reference management** - O(depth) per operation

### Expected GPU Hotspots
1. **Memory allocation** - cudaMalloc overhead
2. **State copying** - COW operations
3. **Memory deallocation** - cudaFree overhead

---

## 🎯 Optimization Targets

### Identified Opportunities

1. **LRU List Optimization**
   - Current: Doubly-linked list with O(1) insert/remove
   - Optimization: Already optimal for individual ops
   - Potential: Batch operations for better cache locality

2. **Tree Traversal**
   - Current: Recursive traversal with dict lookups
   - Optimization: Consider trie compression for common prefixes
   - Potential: 20-30% speedup for deep trees

3. **Lock Management**
   - Current: Traverse to root on every lock
   - Optimization: Batch lock/unlock operations
   - Potential: 15-20% speedup for deep paths

4. **Memory Allocation**
   - Current: Per-operation allocation
   - Optimization: Pre-allocate pools, lazy deallocation
   - Potential: 40-50% speedup for allocation-heavy workloads

---

## 📊 Scalability Analysis

### Cache Size Scalability

| Cache Size | Expected Insert Latency | Expected Match Latency | Notes |
|------------|------------------------|----------------------|-------|
| 128 | TBD ms | TBD ms | Small cache |
| 1024 | TBD ms | TBD ms | **Baseline** |
| 4096 | TBD ms | TBD ms | Large cache |
| 16384 | TBD ms | TBD ms | Very large cache |

**Expected Trend:**
- Insert: O(log n) growth
- Match: O(log n) growth
- Evict: O(n) worst case (LRU traversal)

### Concurrency Scalability

| Concurrent Reqs | Expected Throughput | Expected Latency | Notes |
|-----------------|---------------------|------------------|-------|
| 1 | TBD ops/sec | TBD ms | Single-threaded |
| 16 | TBD ops/sec | TBD ms | Moderate concurrency |
| 64 | TBD ops/sec | TBD ms | **Baseline** |
| 256 | TBD ops/sec | TBD ms | High concurrency |

**Expected Bottleneck:**
- Lock contention on tree nodes
- GPU memory allocation serialization

---

## 🚀 Running Benchmarks

### Quick Start
```bash
cd /home/user/sglang-mamba
export PYTHONPATH=/home/user/sglang-mamba/python:$PYTHONPATH

# Basic benchmark
python phase3/benchmarks/mamba_radix_cache_benchmark.py

# With profiling
python phase3/benchmarks/mamba_radix_cache_benchmark.py --profile

# Custom cache sizes
python phase3/benchmarks/mamba_radix_cache_benchmark.py --kv-cache-size 4096 --mamba-cache-size 256
```

### Prerequisites
```bash
# Ensure dependencies are installed
pip install torch numpy

# GPU required for realistic benchmarks
nvidia-smi  # Check GPU availability
```

### Interpreting Results

**Good Performance Indicators:**
- ✅ Insert < 0.1 ms average
- ✅ Match < 0.05 ms average
- ✅ Evict < 0.2 ms average
- ✅ P99 < 2x average latency

**Performance Issues:**
- ❌ P99 > 10x average (high tail latency)
- ❌ Throughput declining with cache size
- ❌ Memory allocation failures

---

## 📝 Benchmark Methodology

### Insert Benchmark
1. Generate random token sequences
2. Allocate KV cache and Mamba states
3. Time `insert()` operation
4. Handle OOM by evicting and retrying
5. Measure throughput over 1000 iterations

### Match Benchmark
1. Pre-populate cache with 100 sequences
2. Query sequences (100% hit rate)
3. Time `match_prefix()` operation
4. Measure latency distribution
5. 10,000 iterations for statistical significance

### Evict Benchmarks
1. Pre-populate cache to trigger eviction
2. Time `evict()` operation
3. Test both Mamba and full eviction
4. Handle edge cases (empty cache, all locked)
5. 1000 iterations each

---

## 🔄 Continuous Monitoring

### CI Integration
- Run benchmarks on every PR
- Compare against baseline
- Flag regressions > 10%
- Track performance over time

### Metrics to Track
- Operation latencies (avg, P50, P95, P99)
- Throughput (ops/sec)
- Memory usage (MB)
- Cache hit rate (%)
- Eviction rate (per second)

---

## 📚 References

- **RadixCache Paper:** Efficient KV Cache Management (SGLang)
- **Mamba Paper:** Efficient Sequence Modeling with SSMs
- **LRU Implementation:** Doubly-linked list with O(1) operations

---

## ✅ Next Steps

1. **Run Baseline Benchmarks:** Execute benchmark script to fill in TBD values
2. **Analyze Results:** Identify actual vs. expected performance
3. **Profile Hotspots:** Use cProfile to find bottlenecks
4. **Optimize:** Implement identified optimizations (Phase 3.3)
5. **Validate:** Re-run benchmarks to measure improvements

**Status:** ⏳ Awaiting first benchmark run to establish baseline metrics
