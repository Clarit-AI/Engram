# Engine Parameter Naming Audit Report

**Date:** 2026-02-16
**Phase:** 3.1 - Foundation
**Task:** 3.1.2 - Parameter Naming Audit
**Agent:** Engine Refactoring Specialist

---

## Executive Summary

✅ **GOOD NEWS:** There is **NO existing parameter naming inconsistency** in the current codebase.

The codebase already uses `server_args: ServerArgs` consistently across all engine components. The anticipated `engine_config` vs `config` issue does not exist.

**However**, since Mamba model integration into the Engine/Scheduler/ModelRunner is **not yet implemented**, we have the opportunity to **establish the correct naming convention NOW** before implementing.

---

## Current State Analysis

### 1. Existing Parameter Convention ✅

All existing components use `server_args: ServerArgs`:

**Engine (`python/sglang/srt/entrypoints/engine.py`)**
```python
class Engine(EngineBase):
    def __init__(self, **kwargs):
        if "server_args" in kwargs:
            server_args = kwargs["server_args"]
        else:
            server_args = self.server_args_class(**kwargs)
        self.server_args = server_args
        # ...
```

**Scheduler (`python/sglang/srt/managers/scheduler.py`)**
```python
class Scheduler:
    def __init__(
        self,
        server_args: ServerArgs,  # ✅ Uses server_args
        port_args: PortArgs,
        gpu_id: int,
        # ...
    ):
        self.server_args = server_args
```

**ModelRunner (`python/sglang/srt/model_executor/model_runner.py`)**
```python
class ModelRunner:
    def __init__(
        self,
        model_config: ModelConfig,
        # ...
        server_args: ServerArgs,  # ✅ Uses server_args
        # ...
    ):
        self.device = server_args.device
        self.dp_size = server_args.dp_size
```

### 2. Mamba Integration Status 🚧

**Implemented:**
- ✅ Snapshot infrastructure (`python/sglang/srt/snapshot/mamba_snapshot.py`)
- ✅ Mamba host pool (`python/sglang/srt/snapshot/mamba_host_pool.py`)
- ✅ Snapshot hooks and policies
- ✅ Tier management for state offloading

**Not Yet Implemented:**
- ❌ Mamba model class in `python/sglang/srt/models/` (no `mamba.py`)
- ❌ Mamba-specific forward pass integration
- ❌ MambaScheduleBatch integration with Scheduler
- ❌ Mamba state management in ModelRunner
- ❌ Prefill caching for Mamba states
- ❌ Chunked prefill support
- ❌ RadixCache integration for Mamba

---

## Search Results

### Query 1: `engine_config`
```bash
grep -rn "engine_config" python/sglang/srt/
```
**Result:** No matches found ✅

### Query 2: `server_args` in key files
```bash
grep -n "server_args" python/sglang/srt/managers/scheduler.py
grep -n "server_args" python/sglang/srt/model_executor/model_runner.py
grep -n "server_args" python/sglang/srt/entrypoints/engine.py
```
**Result:** All use `server_args` consistently ✅

---

## Identified Risks (None)

❌ No naming inconsistencies found
❌ No conflicting parameter names
❌ No refactoring needed for existing code

---

## Recommendations

### 1. **Adopt Existing Convention** ✅ RECOMMENDED
**Decision:** Use `server_args: ServerArgs` for all future Mamba integration code.

**Rationale:**
- Already established across codebase
- Type-safe with ServerArgs dataclass
- Clear ownership (server-level configuration)
- No migration needed

**Implementation:**
When creating Mamba model integration:
```python
# python/sglang/srt/models/mamba.py (to be created)
class MambaForCausalLM:
    def __init__(
        self,
        config: PretrainedConfig,  # Model config (from HF)
        server_args: ServerArgs,   # ✅ Server runtime config
        # ...
    ):
        self.config = config           # Model architecture config
        self.server_args = server_args # Server/engine config
```

### 2. **Parameter Terminology Clarity**

To avoid future confusion, establish clear terminology:

| Parameter | Type | Scope | Purpose |
|-----------|------|-------|---------|
| `config` | `PretrainedConfig` | Model | HuggingFace model architecture config |
| `model_config` | `ModelConfig` | Model | SGLang model configuration wrapper |
| `server_args` | `ServerArgs` | Engine/Server | Runtime server configuration |
| `port_args` | `PortArgs` | IPC | Inter-process communication ports |

### 3. **Documentation Update**

Add to SDP.md Section 9 (Engine Integration):
```markdown
## Parameter Naming Convention

All SGLang components use the following parameter naming:
- `server_args: ServerArgs` - Server runtime configuration (device, tp_size, etc.)
- `config` or `model_config: ModelConfig` - Model architecture configuration
- Never use `engine_config` or ambiguous `config` for server args
```

---

## Refactoring Map

**Files to create (future implementation):**
- `python/sglang/srt/models/mamba.py` → Use `server_args: ServerArgs`
- `python/sglang/srt/layers/mamba/` (various files) → Use `server_args: ServerArgs`
- `python/sglang/srt/managers/mamba_batch.py` → Use `server_args: ServerArgs`

**Files to modify: NONE** ✅
(Existing codebase already follows correct convention)

---

## Conclusion

✅ **Audit Result:** PASSED - No issues found
✅ **Recommendation:** Adopt existing `server_args: ServerArgs` convention
✅ **Action Required:** Document convention in ADR before Phase 3.2 implementation

**Next Steps:**
1. ✅ Complete this audit report
2. 🔄 Create ADR 001 documenting the naming standard
3. 🔄 Get user approval on convention
4. 🔄 Proceed to Phase 3.2 implementation using approved convention

---

**Report Generated:** 2026-02-16
**Status:** Complete
**Agent:** engine-refactoring-specialist
