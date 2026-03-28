# SGLang-Mamba Test Suite — Phase Documents

Each file in this directory is a self-contained agent prompt for one phase of the test plan. A fresh Claude Code session can pick up any phase document and execute it autonomously.

## Phase Dependency Graph

```
Phase 0 (Environment Verification)
  └─► Phase 1 (Stateless Inference Baseline)
        └─► Phase 2 (MambaPool Unit Tests)      ◄─ no server needed
              └─► Phase 3 (RadixCache Component) ◄─ no server needed
                    ├─► Phase 4 (Live Server, no_buffer)
                    │     └─► Phase 5 (Mamba2Metadata Integrity) ◄─ no server
                    │           └─► Phase 6 (extra_buffer Strategy)
                    │                 └─► Phase 7 (Snapshot System)
                    │                       └─► Phase 8 (Gauntlet / Stress)
                    └─► (can also run Phase 2, 3, 5 in parallel after Phase 1)
```

## Index

| File | Phase | Requires Server | New Test File |
|------|-------|-----------------|---------------|
| `phase-00-environment-verification.md` | 0 — Pre-flight | No | None |
| `phase-01-stateless-inference-baseline.md` | 1 — Baseline Inference | Yes (`--disable-radix-cache`) | `test_mamba_baseline_inference.py` |
| `phase-02-mamba-pool-unit-tests.md` | 2 — MambaPool Unit Tests | No | `test_mamba_pool_extended.py` |
| `phase-03-mamba-radix-cache-component-tests.md` | 3 — RadixCache Gauntlet | No | `test_mamba_radix_cache_gauntlet.py` |
| `phase-04-live-server-integration-no-buffer.md` | 4 — Server + RadixCache | Yes (`no_buffer`) | `test_mamba_radix_cache_server_integration.py` |
| `phase-05-mamba2metadata-integrity.md` | 5 — Metadata Integrity | No | `test_mamba_metadata.py` |
| `phase-06-extra-buffer-strategy.md` | 6 — extra_buffer Mode | Yes (`extra_buffer`) | `test_mamba_extra_buffer.py` |
| `phase-07-snapshot-system.md` | 7 — Snapshot E2E | Yes (`--enable-mamba-snapshots`) | `test_mamba_snapshot_e2e.py` |
| `phase-08-gauntlet-stress-tests.md` | 8 — Stress / Gauntlet | Yes | `test_mamba_gauntlet_stress.py` |

## Configuration

`config.sh` — single source of truth for `MODEL_PATH`, `MODEL_NAME`, `SERVER_PORT`, `SNAPSHOT_DIR`, and `RESULTS_DIR`. All phase documents source this file at the top of their Environment Setup section:

```bash
source test/phases/config.sh
```

**To test a different model**: edit `MODEL_PATH` and `MODEL_NAME` in `config.sh` only. All phases pick it up automatically, and the `MODEL_NAME` variable is embedded in every report filename so results from different models don't collide.

## Results

Each phase writes a detailed markdown report to `test/phases/results/phase-NN-<model>-<date>.md`. The `results/` directory is created automatically by `config.sh`. Reports include: per-test pass/fail table, HITL transcripts, tracebacks, server log excerpts, and phase-specific observations.

## Codemap

`codemap.md` — precise file paths, line numbers, method signatures, and field invariants for every component touched across all phases. Sourced from the original agent prompt's citation block. Read this first when implementing any phase.

## Common Conventions

- **CI registration**: All new test files use `register_cuda_ci(est_time=..., suite="stage-b-test-small-1-gpu")` at the top.
- **Test runner**: `python test/run_suite.py --hw cuda --suite stage-b-test-small-1-gpu`
- **Project root**: `/home/bbrenner/sglang-mamba/`
- **Registered test dir**: `test/registered/radix_cache/`
- **HITL interface**: Available for qualitative smoke checks where noted.

## Reporting Convention

At the end of each phase session, the executing agent should output:

```
PHASE N RESULT: PASS | FAIL
Tests run: X  Passed: Y  Failed: Z
HITL: PASS | SKIP | N/A
Notes: <any failures, tracebacks, or observations>
```
