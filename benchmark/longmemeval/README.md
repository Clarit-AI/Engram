# LongMemEval — Engram Benchmark Harness

Measures whether Engram snapshot/statefulness reduces compute for
Mamba-family models on the LongMemEval benchmark.

## Dataset

| Property | Value |
|---|---|
| HuggingFace repo | `xiaowu0162/LongMemEval` |
| License | MIT |
| Size | ~2 GB |
| Questions | 500 |
| Memory types | episodic, semantic, temporal, spatial, factual |
| Context scale | 115K–1.5M tokens |

## How it works

**Baseline (stateless):** Every question sends the full `memory_text` + question
to the model. No snapshot is consulted.

**Engram (stateful):**
- **Warm** — a snapshot for this question's memory already exists. Only the
  question is sent; the model state encodes the context. Tokens saved ≈
  context length.
- **Cold** — no snapshot exists. Full context + question is sent; a stub
  snapshot is saved for future warm reuse.

The `restore_mode` field (`"warm"` / `"cold"`) on every `QuestionResult`
reflects the CS-161 warm/cold gate.

## Metrics

| Metric | Definition |
|---|---|
| `token_reduction` | `(baseline_input − engram_input) / baseline_input` |
| `ttft_speedup` | `baseline_ttft_s / engram_ttft_s` |
| `compute_amortization` | `baseline_input − engram_input` (linear FLOPs proxy) |

## Quick Start

### 1. CPU dry-run (no GPU, no server, no API key)

```bash
# Generate synthetic data
python -m benchmark.longmemeval.download --dry-run

# Run tests
pytest benchmark/longmemeval/test_dry_run.py -v

# Run the harness in dry-run mode
python -m benchmark.longmemeval.runner \
  --dry-run \
  --mode both \
  --num-questions 5 \
  --output /tmp/lme_results.jsonl
```

### 2. Real GPU run (requires Engram server + real dataset)

```bash
# Download dataset (~2 GB)
python -m benchmark.longmemeval.download

# Start Engram server (separate terminal)
python -m sglang.launch_server \
  --model-path <model> \
  --port 30000

# Run benchmark
python -m benchmark.longmemeval.runner \
  --model-url http://localhost:30000 \
  --snapshot-dir /tmp/lme_snapshots \
  --output /tmp/lme_results.jsonl \
  --num-questions 500 \
  --mode both
```

## Expected Output Shape

```
Loaded 500 questions from ...
Results written to: /tmp/lme_results.jsonl
  Questions:             500
  Warm:                  450
  Cold:                  50
  Mean token reduction:  0.874
  Mean TTFT speedup:     3.2x
  Mean compute amort.:   48200.0 tokens saved
```

JSONL format: first line is the run summary (metadata), subsequent lines are
one `QuestionResult` per line.

## Results Storage

Results are stored to DigitalOcean Spaces (sfo3):

```bash
s3cmd put /tmp/lme_results.jsonl s3://engram/benchmarks/longmemeval/
```

Endpoint: `https://sfo3.digitaloceanspaces.com`

## GPU Run Checklist

- [ ] Engram server running with snapshot support enabled
- [ ] `snapshot_dir` persists between baseline and Engram runs
- [ ] Full dataset downloaded (`longmemeval_s.jsonl`, `longmemeval_m.jsonl`)
- [ ] `OPENAI_API_KEY` set for GPT-4o judge (or use `MockJudge` for token metrics only)
- [ ] Results uploaded to `s3://engram/benchmarks/longmemeval/`
- [ ] Results reviewed against prior runs in Linear
