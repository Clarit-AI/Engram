#!/usr/bin/env python3
"""
MambaRadixCache Performance Benchmark Suite

Measures baseline performance metrics for MambaRadixCache operations:
- Insert throughput (sequences/sec)
- Match latency (ms)
- Eviction throughput (items/sec)
- Memory overhead
- Cache hit rate
- LRU list operations

Usage:
    python mamba_radix_cache_benchmark.py
    python mamba_radix_cache_benchmark.py --profile
    python mamba_radix_cache_benchmark.py --iterations 10000
"""

import argparse
import cProfile
import io
import pstats
import statistics
import sys
import time
from dataclasses import dataclass
from typing import List, Tuple

import torch

# Add python directory to path for imports
sys.path.insert(0, "/home/user/sglang-mamba/python")

from sglang.srt.configs.mamba_utils import Mamba2CacheParams, Mamba2StateShape
from sglang.srt.environ import envs
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
from sglang.srt.server_args import ServerArgs, set_global_server_args_for_scheduler
from sglang.srt.utils import get_device


@dataclass
class BenchmarkResult:
    """Results from a benchmark run."""

    operation: str
    total_time_ms: float
    avg_time_ms: float
    min_time_ms: float
    max_time_ms: float
    p50_time_ms: float
    p95_time_ms: float
    p99_time_ms: float
    throughput: float  # ops/sec
    iterations: int


class MambaRadixCacheBenchmark:
    """Benchmark suite for MambaRadixCache performance testing."""

    def __init__(
        self,
        kv_cache_size: int = 1024,
        mamba_cache_size: int = 128,
        max_num_reqs: int = 64,
        page_size: int = 1,
    ):
        """Initialize benchmark environment."""
        # Set global server args
        set_global_server_args_for_scheduler(
            ServerArgs(model_path="dummy", page_size=page_size)
        )

        # Configuration
        self.kv_cache_size = kv_cache_size
        self.mamba_cache_size = mamba_cache_size
        self.max_num_reqs = max_num_reqs
        self.page_size = page_size
        self.device = get_device()

        # Model configuration
        self.dtype = torch.bfloat16
        self.head_num = 32
        self.head_dim = 128
        self.num_layers = 48
        self.global_interval = 4
        self.max_context_len = 2048

        # Setup components
        self._setup_pools()
        self._setup_cache()

    def _setup_pools(self):
        """Setup memory pools for testing."""
        # Layer configuration
        self.full_attention_layer_ids = [
            i for i in range(self.global_interval - 1, self.num_layers, self.global_interval)
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
            self.mamba2_cache_params = Mamba2CacheParams(shape=shape, layers=self.mamba_layers)

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
            page_size=self.page_size,
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

    def _setup_cache(self):
        """Setup MambaRadixCache."""
        params = CacheInitParams(
            req_to_token_pool=self.req_to_token_pool,
            token_to_kv_pool_allocator=self.allocator,
            page_size=self.page_size,
            disable=False,
        )
        self.cache = MambaRadixCache(params=params)

    def benchmark_insert(self, num_sequences: int = 1000, seq_len: int = 10) -> BenchmarkResult:
        """Benchmark insert operations."""
        timings = []

        for i in range(num_sequences):
            # Generate random token sequence
            token_ids = list(range(i, i + seq_len))
            kv_indices = self.allocator.alloc(seq_len)
            if kv_indices is None:
                # Out of memory, evict and retry
                self.cache.evict(EvictParams(num_tokens=seq_len))
                kv_indices = self.allocator.alloc(seq_len)

            # Allocate Mamba state
            mamba_value = self.req_to_token_pool.mamba_pool.alloc(1)
            if mamba_value is None:
                # Out of memory, evict and retry
                self.cache.evict(EvictParams(num_tokens=0, mamba_num=1))
                mamba_value = self.req_to_token_pool.mamba_pool.alloc(1)

            # Time the insert operation
            start = time.perf_counter()
            result = self.cache.insert(
                InsertParams(
                    key=RadixKey(token_ids),
                    value=kv_indices,
                    mamba_value=mamba_value,
                )
            )
            end = time.perf_counter()

            timings.append((end - start) * 1000)  # Convert to ms

        return self._compute_stats("insert", timings)

    def benchmark_match(self, num_queries: int = 10000) -> BenchmarkResult:
        """Benchmark match_prefix operations."""
        # Pre-populate cache with sequences
        sequences = []
        for i in range(100):
            seq_len = 10
            token_ids = list(range(i * 100, i * 100 + seq_len))
            sequences.append(token_ids)

            kv_indices = self.allocator.alloc(seq_len)
            if kv_indices is None:
                self.cache.evict(EvictParams(num_tokens=seq_len))
                kv_indices = self.allocator.alloc(seq_len)

            mamba_value = self.req_to_token_pool.mamba_pool.alloc(1)
            if mamba_value is None:
                self.cache.evict(EvictParams(num_tokens=0, mamba_num=1))
                mamba_value = self.req_to_token_pool.mamba_pool.alloc(1)

            self.cache.insert(
                InsertParams(
                    key=RadixKey(token_ids),
                    value=kv_indices,
                    mamba_value=mamba_value,
                )
            )

        # Benchmark match operations
        timings = []
        for i in range(num_queries):
            # Query a random sequence (may or may not be in cache)
            query_seq = sequences[i % len(sequences)]

            start = time.perf_counter()
            result = self.cache.match_prefix(MatchPrefixParams(key=RadixKey(query_seq)))
            end = time.perf_counter()

            timings.append((end - start) * 1000)

        return self._compute_stats("match_prefix", timings)

    def benchmark_evict_mamba(self, num_evictions: int = 1000) -> BenchmarkResult:
        """Benchmark Mamba state eviction."""
        # Pre-populate cache
        for i in range(self.mamba_cache_size):
            token_ids = [i]
            kv_indices = self.allocator.alloc(1)
            mamba_value = self.req_to_token_pool.mamba_pool.alloc(1)

            if kv_indices and mamba_value:
                self.cache.insert(
                    InsertParams(
                        key=RadixKey(token_ids),
                        value=kv_indices,
                        mamba_value=mamba_value,
                    )
                )

        # Benchmark evictions
        timings = []
        for i in range(num_evictions):
            # Re-populate if needed
            if self.cache.mamba_evictable_size() == 0:
                token_ids = [10000 + i]
                kv_indices = self.allocator.alloc(1)
                mamba_value = self.req_to_token_pool.mamba_pool.alloc(1)
                if kv_indices and mamba_value:
                    self.cache.insert(
                        InsertParams(
                            key=RadixKey(token_ids),
                            value=kv_indices,
                            mamba_value=mamba_value,
                        )
                    )

            start = time.perf_counter()
            result = self.cache.evict(EvictParams(num_tokens=0, mamba_num=1))
            end = time.perf_counter()

            timings.append((end - start) * 1000)

        return self._compute_stats("evict_mamba", timings)

    def benchmark_evict_full(self, num_evictions: int = 1000) -> BenchmarkResult:
        """Benchmark full KV cache eviction."""
        # Pre-populate cache
        for i in range(100):
            token_ids = [i]
            kv_indices = self.allocator.alloc(1)
            mamba_value = self.req_to_token_pool.mamba_pool.alloc(1)

            if kv_indices and mamba_value:
                self.cache.insert(
                    InsertParams(
                        key=RadixKey(token_ids),
                        value=kv_indices,
                        mamba_value=mamba_value,
                    )
                )

        # Benchmark evictions
        timings = []
        for i in range(num_evictions):
            # Re-populate if needed
            if self.cache.full_evictable_size() == 0:
                token_ids = [10000 + i]
                kv_indices = self.allocator.alloc(1)
                mamba_value = self.req_to_token_pool.mamba_pool.alloc(1)
                if kv_indices and mamba_value:
                    self.cache.insert(
                        InsertParams(
                            key=RadixKey(token_ids),
                            value=kv_indices,
                            mamba_value=mamba_value,
                        )
                    )

            start = time.perf_counter()
            result = self.cache.evict(EvictParams(num_tokens=1, mamba_num=0))
            end = time.perf_counter()

            timings.append((end - start) * 1000)

        return self._compute_stats("evict_full", timings)

    def _compute_stats(self, operation: str, timings: List[float]) -> BenchmarkResult:
        """Compute statistics from timing data."""
        timings_sorted = sorted(timings)
        total_time = sum(timings)
        count = len(timings)

        return BenchmarkResult(
            operation=operation,
            total_time_ms=total_time,
            avg_time_ms=statistics.mean(timings),
            min_time_ms=min(timings),
            max_time_ms=max(timings),
            p50_time_ms=timings_sorted[int(count * 0.50)],
            p95_time_ms=timings_sorted[int(count * 0.95)],
            p99_time_ms=timings_sorted[int(count * 0.99)],
            throughput=count / (total_time / 1000),  # ops/sec
            iterations=count,
        )

    def print_result(self, result: BenchmarkResult):
        """Pretty print benchmark result."""
        print(f"\n{'=' * 70}")
        print(f"Operation: {result.operation.upper()}")
        print(f"{'=' * 70}")
        print(f"Iterations:     {result.iterations:,}")
        print(f"Total Time:     {result.total_time_ms:.2f} ms")
        print(f"Average Time:   {result.avg_time_ms:.4f} ms")
        print(f"Min Time:       {result.min_time_ms:.4f} ms")
        print(f"Max Time:       {result.max_time_ms:.4f} ms")
        print(f"P50 (Median):   {result.p50_time_ms:.4f} ms")
        print(f"P95:            {result.p95_time_ms:.4f} ms")
        print(f"P99:            {result.p99_time_ms:.4f} ms")
        print(f"Throughput:     {result.throughput:,.2f} ops/sec")
        print(f"{'=' * 70}")

    def run_all_benchmarks(self):
        """Run all benchmarks and display results."""
        print("\n" + "=" * 70)
        print("MambaRadixCache Performance Benchmark Suite")
        print("=" * 70)
        print(f"Configuration:")
        print(f"  KV Cache Size:    {self.kv_cache_size}")
        print(f"  Mamba Cache Size: {self.mamba_cache_size}")
        print(f"  Max Requests:     {self.max_num_reqs}")
        print(f"  Page Size:        {self.page_size}")
        print(f"  Device:           {self.device}")
        print("=" * 70)

        results = []

        # Insert benchmark
        print("\n[1/4] Running insert benchmark...")
        result = self.benchmark_insert(num_sequences=1000, seq_len=10)
        self.print_result(result)
        results.append(result)

        # Match benchmark
        print("\n[2/4] Running match_prefix benchmark...")
        self._setup_cache()  # Reset cache
        result = self.benchmark_match(num_queries=10000)
        self.print_result(result)
        results.append(result)

        # Evict Mamba benchmark
        print("\n[3/4] Running evict_mamba benchmark...")
        self._setup_cache()  # Reset cache
        result = self.benchmark_evict_mamba(num_evictions=1000)
        self.print_result(result)
        results.append(result)

        # Evict full benchmark
        print("\n[4/4] Running evict_full benchmark...")
        self._setup_cache()  # Reset cache
        result = self.benchmark_evict_full(num_evictions=1000)
        self.print_result(result)
        results.append(result)

        # Summary
        print("\n" + "=" * 70)
        print("SUMMARY")
        print("=" * 70)
        for result in results:
            print(
                f"{result.operation:15s}: {result.avg_time_ms:8.4f} ms avg, "
                f"{result.throughput:10,.2f} ops/sec"
            )
        print("=" * 70)

        return results


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="MambaRadixCache Performance Benchmark")
    parser.add_argument(
        "--profile", action="store_true", help="Enable cProfile profiling"
    )
    parser.add_argument(
        "--kv-cache-size", type=int, default=1024, help="KV cache size"
    )
    parser.add_argument(
        "--mamba-cache-size", type=int, default=128, help="Mamba cache size"
    )
    args = parser.parse_args()

    benchmark = MambaRadixCacheBenchmark(
        kv_cache_size=args.kv_cache_size,
        mamba_cache_size=args.mamba_cache_size,
    )

    if args.profile:
        # Run with profiling
        profiler = cProfile.Profile()
        profiler.enable()
        results = benchmark.run_all_benchmarks()
        profiler.disable()

        # Print profile stats
        print("\n" + "=" * 70)
        print("PROFILING RESULTS")
        print("=" * 70)
        s = io.StringIO()
        ps = pstats.Stats(profiler, stream=s).sort_stats("cumulative")
        ps.print_stats(30)  # Top 30 functions
        print(s.getvalue())
    else:
        # Run without profiling
        results = benchmark.run_all_benchmarks()


if __name__ == "__main__":
    main()
