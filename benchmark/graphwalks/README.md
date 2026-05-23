# GraphWalks Benchmark — Engram

Multi-hop graph-traversal QA benchmark that measures whether Engram's
snapshot/restore capability reduces compute versus a stateless baseline.

---

## Dataset

| Field    | Value |
|----------|-------|
| HF repo  | [`openai/graphwalks`](https://huggingface.co/datasets/openai/graphwalks) |
| Task     | Multi-hop graph traversal QA |
| License  | MIT (OpenAI GraphWalks) |
| Size     | ~50 MB (confirmed on first download) |
| Format   | JSONL; fields: `question_id`, `graph_text`, `question`, `answer`, `num_hops`, `num_nodes` |

Each example presents a textual description of a directed graph and asks the
model to traverse it for N hops to identify a target node.  Ground-truth answers
are exact node labels, so scoring is deterministic via `ExactMatchScorer`.

---

## Metrics

| Metric | Formula | Unit |
|---|---|---|
| `token_reduction` | `1 - engram_input_tokens / baseline_input_tokens` | fraction [0, 1] |
| `ttft_speedup` | `baseline_ttft_s / engram_ttft_s` | ratio (>1 = Engram faster) |
| `compute_amortization` | `token_reduction` for warm results; 0 for cold | fraction [0, 1] |

All three metrics are computed per-question and aggregated across the run, with
separate warm/cold breakdowns.

---

## Run Commands

### CPU dry-run (no GPU, no server — CI-safe)

```bash
# Run all dry-run tests
pytest benchmark/graphwalks/test_dry_run.py -v

# Generate synthetic dataset
python -m benchmark.graphwalks.download --dry-run

# Dry-run the full runner pipeline
python -m benchmark.graphwalks.runner --dry-run --output /tmp/gw_dry_run.jsonl
```

### GPU run (H100, Elastic 30B BF16 single-GPU)

```bash
# 1. Download dataset
python -m benchmark.graphwalks.download

# 2. Start Engram inference server
python -m sglang.launch_server \
    --model-path elastic-30b-bf16 \
    --snapshot-dir /mnt/snapshots/graphwalks \
    --port 30000

# 3. Run benchmark (both baseline and Engram phases)
python -m benchmark.graphwalks.runner \
    --model-url http://localhost:30000/v1 \
    --model elastic-30b-bf16 \
    --snapshot-dir /mnt/snapshots/graphwalks \
    --output results/graphwalks/elastic30b_h100.jsonl \
    --num-questions 1000 \
    --mode both
```

### Cluster run (Qwen3-Coder-Next FP8, TP=4)

```bash
python -m benchmark.graphwalks.runner \
    --model-url http://localhost:30000/v1 \
    --model qwen3-coder-next-fp8 \
    --snapshot-dir /mnt/snapshots/graphwalks \
    --output results/graphwalks/qwen3_tp4.jsonl \
    --num-questions 1000 \
    --mode both
```

---

## Expected Output Shape

Results are written as JSONL.  The first line is the run-level summary header;
subsequent lines are per-question records.

**Summary header** (line 1):
```json
{
  "benchmark": "graphwalks",
  "model": "elastic-30b-bf16",
  "timestamp": "2026-05-22T...",
  "n_total": 1000,
  "n_warm": 850,
  "n_cold": 150,
  "token_reduction": 0.82,
  "ttft_speedup": 3.1,
  "compute_amortization": 0.82,
  "warm_token_reduction": 0.82,
  "warm_ttft_speedup": 3.1,
  "cold_token_reduction": 0.0,
  "cold_ttft_speedup": 0.95,
  "metrics_by_hop": {
    "1": {"n": 200, "token_reduction": 0.83, "ttft_speedup": 3.2, ...},
    "2": {"n": 300, "token_reduction": 0.81, "ttft_speedup": 3.0, ...},
    "3": {"n": 500, "token_reduction": 0.82, "ttft_speedup": 3.1, ...}
  }
}
```

**Per-question record** (lines 2+):
```json
{
  "question_id": "gw_001",
  "graph_size": 20,
  "num_hops": 3,
  "restore_mode": "warm",
  "baseline_ttft_s": 0.95,
  "baseline_input_tokens": 412,
  "baseline_output_tokens": 8,
  "baseline_answer": "Node F",
  "baseline_score": 1.0,
  "engram_ttft_s": 0.31,
  "engram_input_tokens": 74,
  "engram_output_tokens": 8,
  "engram_answer": "Node F",
  "engram_score": 1.0,
  "token_reduction": 0.82,
  "ttft_speedup": 3.06,
  "compute_amortization": 0.82
}
```

---

## Results Storage

Upload completed result files to S3 (DigitalOcean Spaces, sfo3):

```bash
s3cmd put results/graphwalks/elastic30b_h100.jsonl \
    s3://engram/benchmarks/graphwalks/elastic30b_h100.jsonl \
    --acl-private
```

Base path: `s3://engram/benchmarks/graphwalks/`

---

## GPU Run Checklist

- [ ] Engram server starts cleanly with `--snapshot-dir`
- [ ] Baseline phase completes without errors
- [ ] Engram cold phase creates snapshot files
- [ ] Engram warm phase reads from snapshots (check logs for "warm" labels)
- [ ] `token_reduction` > 0.7 for warm questions
- [ ] `ttft_speedup` > 2.0 for warm questions
- [ ] Results uploaded to `s3://engram/benchmarks/graphwalks/`

---

## Models

| Model | Precision | HW | Notes |
|---|---|---|---|
| Elastic 30B | BF16 | Single H100 | Reference model |
| Qwen3-Coder-Next | FP8 | 4× H100 (TP=4) | Cluster run |

---

## File Layout

```
benchmark/graphwalks/
├── __init__.py
├── download.py        # Dataset download + synthetic generation
├── judge.py           # ExactMatchScorer, GPT4oJudge, MockJudge
├── results.py         # QuestionResult, RunSummary dataclasses + JSONL I/O
├── runner.py          # GraphWalksRunner + CLI entry point
├── test_dry_run.py    # CPU/mock pytest suite
├── data/
│   └── graphwalks/
│       ├── test_dry_run.jsonl   (synthetic, generated locally)
│       └── graphwalks_test.jsonl (downloaded from HF)
└── README.md
```
