# Project Memory

## Session History
- 2026-03-25: Initial CLAUDE.md created, gcloud instance validated, tests passing

## Project Context

### Engram Fork
**Purpose**: Add snapshot persistence for Mamba SSM hidden states to enable fast multi-turn conversations (25x+ speedup)

**Key Files**:
- `python/sglang/srt/snapshot/mamba_snapshot.py` - Core snapshot save/restore
- `python/sglang/srt/mem_cache/mamba_radix_cache.py` - 1233 lines, dual tree cache
- `python/sglang/snapshot.py` - High-level SnapshotManager API
- `python/sglang/lang/interpreter.py` - ProgramState with save/restore methods

### Phase Status
- 3.1 ✅ Foundation
- 3.2 ✅ Core Implementation (MambaRadixCache already done)
- 3.3 ✅ Static Analysis (optimizations identified)
- 3.4 ⬜ Final Audit (pending)

### Test Status
- `test/sglang/snapshot/`: 46 passed
- `test/sglang/agents/`: 37 passed

### Known Issues
1. Mamba model config bug (`architectures: None`) - crashes server start
2. Granite GGUF `granitehybrid` not supported by transformers
3. V100 16GB too small for full Granite MoE

### GCloud Instance
- **Name**: sglang-test-v100-20260325-230245
- **Zone**: asia-east1-c
- **Project**: gen-lang-client-0471830999
- **Clone**: `/home/bbrenner/sglang-mamba`
- **Tunnel**: mamba.clarit.ai → localhost:30000

## Phase Plans Location
All phase plans, validation reports, and analysis in:
- `MAMBA_SNAPSHOT_RESTORATION_PLAN.md`
- `PHASE_3_PLAN.md`
- `phase3/` directory

## Next Steps
1. Fix Mamba model_config.py bug to enable server startup
2. Or proceed with Phase 3.4 (Final Audit) based on static analysis
3. Test with small Mamba model once server is running
