# Phase 0 — Environment Verification (Pre-flight)

## Purpose

Before any test is written or server is launched, verify that the Engram installation is healthy and the existing baseline test suites pass clean. This phase catches broken installs, missing dependencies, or upstream regressions before they contaminate later phases. It produces no new test files — only a pass/fail signal for the environment.

## Prerequisites

- None. This is the entry point.
- Model checkpoint present. Default: `granite-4.0-h-tiny` at `/home/jeanclawdai/models/granite-4.0-h-tiny` (`GraniteMoeHybridForCausalLM`, 40-layer Mamba/attention hybrid, already downloaded).
- See `config.sh` for alternative models (Nemotron-4B fallback, Granite-Q4 quantized comparison).
- GPU (CUDA) available and `nvidia-smi` shows at least one device.

## Key Files

- `python/` — the installable Python package (`pip install -e python/`)
- `test/registered/radix_cache/test_mamba_unittest.py` — 3 existing CPU/GPU unit tests
- `test/registered/radix_cache/test_mamba_radix_cache_comprehensive.py` — 10 existing comprehensive tests
- `test/run_suite.py` — suite runner

## Environment Setup

```bash
# From project root
REPO_ROOT=$(git rev-parse --show-toplevel)
cd "$REPO_ROOT"
source test/phases/config.sh   # sets MODEL_PATH, RESULTS_DIR, etc.

# 1. Verify the package is installed
pip install -e python/ --quiet

# 2. Confirm import works
python -c "import sglang; print('sglang version:', sglang.__version__)"

# 3. Confirm CUDA is visible
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('Device:', torch.cuda.get_device_name(0))"

# 4. Confirm the model checkpoint is present
# (uses MODEL_PATH from config.sh; default is granite-4.0-h-tiny)
ls $MODEL_PATH   # expect: config.json, model-*.safetensors, tokenizer.json
```

## Tasks

### Task 1: Run existing unit tests

```bash
cd "$REPO_ROOT"

python -m pytest test/registered/radix_cache/test_mamba_unittest.py -v 2>&1 | tee /tmp/phase0_unittest.log
```

Expected: **3 tests pass** (`test_hybrid_linear_kv_pool`, `test_mamba_pool`, `test_mamba_radix_cache_1`).

### Task 2: Run comprehensive radix cache tests

```bash
python -m pytest test/registered/radix_cache/test_mamba_radix_cache_comprehensive.py -v 2>&1 | tee /tmp/phase0_comprehensive.log
```

Expected: **10 tests pass**.

### Task 3: Check for CUDA errors in output

```bash
grep -i "cuda\|error\|traceback\|failed" /tmp/phase0_unittest.log /tmp/phase0_comprehensive.log || echo "No errors found"
```

### Task 4: Report failures

If any test fails, capture the full traceback:

```bash
python -m pytest test/registered/radix_cache/test_mamba_unittest.py \
    test/registered/radix_cache/test_mamba_radix_cache_comprehensive.py \
    -v --tb=long 2>&1 | tee /tmp/phase0_full.log
```

**Stop here and report if any failures occur. Do not proceed to Phase 1 until all 13 tests pass.**

## New Test File(s) to Write

None. Phase 0 only runs existing tests.

## Pass Criteria

- `import sglang` succeeds without error
- `torch.cuda.is_available()` returns `True`
- All 3 tests in `test_mamba_unittest.py` pass
- All 10 tests in `test_mamba_radix_cache_comprehensive.py` pass
- No `CUDA error` or `RuntimeError` in any log output
- `ls /home/jeanclawdai/models/granite-4.0-h-tiny` shows `config.json` + 3 safetensors shards

## Write Report

At the end of the session, write a detailed markdown report to disk:

```bash
REPORT="$RESULTS_DIR/phase-00-${MODEL_NAME}-$(date +%Y%m%d-%H%M).md"
cat > "$REPORT" << 'EOF'
# Phase 0 — Environment Verification
**Model**: granite-4.0-h-tiny
**Date**: <date>
**Result**: PASS | FAIL

## Environment
- sglang version: <version>
- CUDA device: <device name>
- Python: <version>
- Model path confirmed: YES | NO

## Test Results
| Test File | Tests | Passed | Failed |
|-----------|-------|--------|--------|
| test_mamba_unittest.py | 3 | X | Y |
| test_mamba_radix_cache_comprehensive.py | 10 | X | Y |

## Failures
<paste full tracebacks here, or "None">

## Notes
<anything unexpected>
EOF
echo "Report written to $REPORT"
```

## Reporting

```
PHASE 0 RESULT: PASS | FAIL
Tests run: 13  Passed: X  Failed: Y
sglang version: <version>
CUDA device: <device name>
Model path confirmed: YES | NO
Report: $RESULTS_DIR/phase-00-<model>-<date>.md
```
