# Phase 3 Performance Benchmarks

This directory contains performance benchmarking tools for MambaRadixCache and related components.

## 📁 Directory Structure

```
benchmarks/
├── README.md                           # This file
├── mamba_radix_cache_benchmark.py      # Main benchmark suite
└── PERFORMANCE_BASELINE.md             # Baseline metrics & targets
```

## 🚀 Quick Start

### Run All Benchmarks
```bash
cd /home/user/sglang-mamba
export PYTHONPATH=/home/user/sglang-mamba/python:$PYTHONPATH
python phase3/benchmarks/mamba_radix_cache_benchmark.py
```

### Run with Profiling
```bash
python phase3/benchmarks/mamba_radix_cache_benchmark.py --profile
```

### Custom Configuration
```bash
python phase3/benchmarks/mamba_radix_cache_benchmark.py \
    --kv-cache-size 4096 \
    --mamba-cache-size 256
```

## 📊 Benchmarks Included

### 1. Insert Performance
- **What:** Measures `insert()` operation latency
- **Workload:** 1,000 sequences, 10 tokens each
- **Metrics:** Avg/P50/P95/P99 latency, throughput

### 2. Match Prefix Performance
- **What:** Measures `match_prefix()` operation latency
- **Workload:** 10,000 queries on 100 cached sequences
- **Metrics:** Avg/P50/P95/P99 latency, throughput, hit rate

### 3. Evict Mamba Performance
- **What:** Measures Mamba state eviction latency
- **Workload:** 1,000 evictions, 1 state each
- **Metrics:** Avg/P50/P95/P99 latency, throughput

### 4. Evict Full Performance
- **What:** Measures full KV cache eviction latency
- **Workload:** 1,000 evictions, 1 token each
- **Metrics:** Avg/P50/P95/P99 latency, throughput

## 📈 Output Format

```
======================================================================
Operation: INSERT
======================================================================
Iterations:     1,000
Total Time:     XX.XX ms
Average Time:   X.XXXX ms
Min Time:       X.XXXX ms
Max Time:       X.XXXX ms
P50 (Median):   X.XXXX ms
P95:            X.XXXX ms
P99:            X.XXXX ms
Throughput:     XX,XXX.XX ops/sec
======================================================================
```

## 🎯 Performance Targets

| Operation | Target Latency | Target Throughput |
|-----------|---------------|-------------------|
| insert() | < 0.1 ms | > 10,000 ops/sec |
| match_prefix() | < 0.05 ms | > 20,000 ops/sec |
| evict_mamba() | < 0.2 ms | > 5,000 ops/sec |
| evict_full() | < 0.5 ms | > 2,000 ops/sec |

## 📝 Interpreting Results

### Good Performance
- ✅ Average latency meets targets
- ✅ P99 < 2x average (low tail latency)
- ✅ Throughput consistent across runs
- ✅ No OOM errors

### Performance Issues
- ❌ P99 > 10x average (high tail latency)
- ❌ Throughput declining with cache size
- ❌ Frequent memory allocation failures
- ❌ High variance between runs

## 🔧 Configuration Options

### Cache Sizes
- `--kv-cache-size`: KV cache capacity (default: 1024)
- `--mamba-cache-size`: Mamba state capacity (default: 128)

### Profiling
- `--profile`: Enable cProfile profiling (shows top 30 functions)

## 📚 Additional Documentation

- **PERFORMANCE_BASELINE.md**: Detailed baseline metrics and targets
- **../docs/performance_analysis.md**: In-depth performance analysis (to be created)
- **../docs/optimization_guide.md**: Optimization recommendations (Phase 3.3)

## ✅ Next Steps

1. Run baseline benchmarks to establish metrics
2. Compare against targets in PERFORMANCE_BASELINE.md
3. Identify optimization opportunities
4. Implement optimizations (Phase 3.3)
5. Re-run benchmarks to validate improvements
