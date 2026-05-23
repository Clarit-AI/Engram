# RULER Benchmark Harness

Engram's RULER harness measures whether snapshot-based statefulness reduces
compute cost for long-context inference.  Instead of re-processing a long
context on every turn, Engram restores a saved hidden state — the harness
quantifies how much that saves.

## Dataset

| Property | Value |
|---|---|
| Name | RULER (Really Urgent Long-context Evaluation for Research) |
| Source | <https://huggingface.co/datasets/hsiehjackson/RULER> |
| Paper | RULER: What's the Real Context Window Size of Your Long-Context Language Models? |
| License | MIT |
| Generation | Synthetic — generated on the fly, no fixed download corpus |
| Approx. size | <100 MB for 4K–128K configs at 10 samples per (task, context_length) |

### 13 Task Types

| Category | Tasks |
|---|---|
| Needle-in-a-Haystack (single) | `niah_single_1`, `niah_single_2`, `niah_single_3` |
| Needle-in-a-Haystack (multi-key) | `niah_multikey_1`, `niah_multikey_2`, `niah_multikey_3` |
| Needle-in-a-Haystack (multi-value) | `niah_multivalue` |
| Needle-in-a-Haystack (multi-query) | `niah_multiquery` |
| Variable Tracking | `vt` |
| Common Words Extraction | `cwe` |
| Frequent Words Extraction | `fwe` |
| Question Answering | `qa_1`, `qa_2` |

### Standard Context Lengths

`4096`, `8192`, `16384`, `32768`, `65536`, `131072` tokens

## Quick Start

### CPU / mock dry-run (no GPU required)

```bash
# From the repo root
pytest benchmark/ruler/test_dry_run.py -v
```

### GPU run (requires a running Engram or SGLang endpoint)

```bash
python -m benchmark.ruler.runner \
    --model-url http://localhost:30000 \
    --model-name elastic-30b \
    --context-lengths 4096,8192,16384,32768 \
    --tasks niah_single_1,niah_single_2,vt,cwe,qa_1 \
    --num-samples 10 \
    --snapshot-dir /tmp/ruler_snapshots \
    --output ruler_results.jsonl \
    --mode both \
    --verbose
```

### Baseline only (no snapshots)

```bash
python -m benchmark.ruler.runner \
    --model-url http://localhost:30000 \
    --mode baseline \
    --output ruler_baseline.jsonl
```

### Engram-only (use existing snapshots)

```bash
python -m benchmark.ruler.runner \
    --model-url http://localhost:30000 \
    --mode engram \
    --snapshot-dir /mnt/ruler_snapshots \
    --output ruler_engram.jsonl
```

## Expected Output Shape

Each JSONL file contains one line per `TaskResult`, plus a final summary line:

```json
{"task_id": "niah_single_1_cl4096_abc123", "task_name": "niah_single_1",
 "context_length": 4096, "restore_mode": "warm", "baseline_ttft_s": 1.23,
 "baseline_input_tokens": 4100, "baseline_score": 1.0,
 "engram_ttft_s": 0.08, "engram_input_tokens": 32, "engram_score": 1.0, ...}
...
{"_record_type": "summary", "metrics": {"token_reduction": 0.982,
 "ttft_speedup": 15.4, "compute_amortization": 4068.0,
 "warm_token_reduction": 0.992, "cold_token_reduction": 0.0}, ...}
```

Per-task score breakdown is included in the summary line under `task_score_summary`.

### Key Metrics

| Metric | Formula | Goal |
|---|---|---|
| `token_reduction` | `1 - engram_tokens / baseline_tokens` | Higher is better |
| `ttft_speedup` | `baseline_ttft / engram_ttft` (geometric mean) | Higher is better |
| `compute_amortization` | `Σ (baseline_tokens - engram_tokens)` | Higher is better |

Every result line includes `restore_mode: "warm" | "cold"`.  A result without
a warm/cold label is not publishable.

## Results Storage

Results are stored at:

```
s3://engram/benchmarks/ruler/
```

Upload with `s3cmd` targeting the `sfo3` region:

```bash
s3cmd put ruler_results.jsonl \
    s3://engram/benchmarks/ruler/$(date +%Y%m%d_%H%M%S)_ruler_results.jsonl \
    --host=sfo3.digitaloceanspaces.com \
    --host-bucket="%(bucket)s.sfo3.digitaloceanspaces.com"
```

## GPU Run Pending Checklist

- [ ] Verify Engram snapshot save/restore works end-to-end for the target model
- [ ] Confirm `/v1/completions` endpoint is reachable at `--model-url`
- [ ] Set `OPENAI_API_KEY` if using `GPT4oJudge` for qa_1 / qa_2 scoring
- [ ] Set `JUDGE_MODEL` env var if overriding the default `gpt-4o` judge
- [ ] Choose snapshot directory with sufficient disk space (estimate: ~500 MB per model per context length for Mamba-family models)
- [ ] Run with `--num-samples 25` for publication-quality results
- [ ] Upload results to `s3://engram/benchmarks/ruler/`

## Target Models

| Model | Precision | Parallelism | Notes |
|---|---|---|---|
| Elastic 30B | BF16 | Single H100 | Primary evaluation target |
| Qwen3-Coder-Next | FP8 | TP=4 (cluster) | High-throughput cluster run |

## Module Reference

| File | Purpose |
|---|---|
| `results.py` | `TaskResult`, `RunSummary` dataclasses + metrics + JSONL I/O |
| `judge.py` | `ExactMatchScorer`, `GPT4oJudge`, `MockJudge` |
| `tasks.py` | 13 task definitions, `TaskInstance`, `generate_synthetic_task()` |
| `runner.py` | `RULERRunner` (two-phase baseline/Engram), CLI entry point |
| `download.py` | `generate_dataset()`, `load_dataset()`, HF Hub metadata fetch |
| `test_dry_run.py` | CPU/mock pytest suite (no GPU, no network) |
