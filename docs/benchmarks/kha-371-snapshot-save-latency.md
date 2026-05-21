# KHA-371 — Per-Turn Snapshot Save Latency

Measurement of the per-turn `POST /save_snapshot` client-observed latency on
Engram's `warm`-tier path (the snapshot pool slot has been evicted to the host
buffer by displacement traffic before each save).

## Probe

`scripts/kha-370-codestral-benchmark.py` — `warm_evict_probe[].save_latency_ms`.
For each of N trials the probe (a) runs `--displace-n 20` short distinct
conversations against `/generate` to evict the target slot, (b) calls
`POST /save_snapshot {conversation_id}` and records the wall-clock from
`time.perf_counter()` either side of the `requests.post`. Client and server
are both on localhost, so RPC overhead is sub-millisecond — the recorded value
is effectively server-side save time.

## Test configuration

| Field | Value |
| --- | --- |
| Engram commit | `7764309705` (main, post-merge of upstream-sync-20260521) |
| Hardware | NVIDIA H100 80GB HBM3 (81559 MiB) |
| Trials per model | 30 |
| Context length | 1200 words (~1480 tokens) |
| Displacement convs per warm iteration | 20 |
| Server flags | `--disable-radix-cache --disable-piecewise-cuda-graph --enable-snapshot-persistence --mamba-scheduler-strategy no_buffer --trust-remote-code` |
| Env | `SGLANG_ENABLE_SPEC_V2=false` (explicit; `no_buffer` also auto-disables Spec-V2 via #63) |

## Results — warm-tier per-turn save latency (milliseconds)

| Model | Architecture | N | mean | stddev | min | p50 | p90 | p95 | p99 | max |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Nemotron Elastic 30B (`nvidia/NVIDIA-Nemotron-Labs-3-Elastic-30B-A3B-BF16`) | nemotron_h (hybrid) | 30 | 322.4 | 13.8 | 286.2 | 325.1 | 336.7 | 337.5 | 341.8 | 344.4 |
| Granite 4.0-h-tiny (`ibm-granite/granite-4.0-h-tiny`) | granitemoehybrid | 30 | 401.9 | 28.1 | 254.5 | 406.5 | 411.6 | 413.1 | 414.6 | 414.9 |
| Mamba Codestral 7B (`mistralai/Mamba-Codestral-7B-v0.1`) | mamba2 (pure SSM) | 30 | 2441.7 | 48.7 | 2269.7 | 2447.5 | 2461.0 | 2511.1 | 2513.2 | 2527.8 |

## Observations

- The two hybrid-attention/SSM models (Nemotron Elastic, Granite) save in
  ~0.3–0.4s with low variance.
- Pure-SSM Mamba Codestral 7B saves ~6–8× slower (~2.4s) under identical
  probe conditions, despite being a smaller model (7B vs 30B). The cost is
  consistent across all 30 trials (stddev 48.7ms; p50/max within 80ms of each
  other), suggesting a structural fast-path that the pure-Mamba2 save path
  doesn't take.
- N=30 makes p99 effectively the worst-or-second-worst sample; treat p99 as
  an upper-bound indicator rather than a tail-quantile estimate.

## Notes / data caveats

- Codestral run crashed in the unrelated `restore+gen` tier at iteration 17
  with HTTP 500 (`Conv state count (1) doesn't match metadata num_layers
  (64)`). The 30 `warm`-tier save samples completed cleanly before that
  point and are recovered from the raw log; the JSON artifact for Codestral
  was not written.
- All percentile values use the nearest-rank method on sorted samples.
- Raw log: `s3://engram/benchmarks/kha-371-snapshot-latency-h100.log`.
