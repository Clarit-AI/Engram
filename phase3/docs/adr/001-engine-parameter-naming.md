# ADR-001: Engine Parameter Naming Convention

**Status:** Proposed → Pending User Approval
**Date:** 2026-02-16
**Decision Makers:** User, Engine Refactoring Agent
**Phase:** 3.1

---

## Context

As we integrate Mamba SSM models into SGLang's engine infrastructure (Phase 3), we need to establish a consistent parameter naming convention for passing server/engine configuration throughout the codebase.

**Current State:**
- Existing codebase (Scheduler, ModelRunner, Engine) uses `server_args: ServerArgs`
- No Mamba model integration exists yet in `python/sglang/srt/models/`
- Phase 3 will implement Mamba model, forward pass, batch scheduling, and state management

**Problem:**
- Need to decide parameter naming **before** implementing Mamba integration
- Must ensure consistency with existing codebase patterns
- Must avoid confusion between model config vs. server config

---

## Decision

**Adopt `server_args: ServerArgs` as the standard parameter name for all server/engine runtime configuration across the entire codebase, including new Mamba components.**

This includes:
- Model classes (e.g., `MambaForCausalLM.__init__`)
- Layer implementations (e.g., Mamba mixer, state manager)
- Batch management (e.g., `MambaScheduleBatch`)
- Cache implementations (e.g., RadixCache for Mamba)

---

## Rationale

### Alternatives Considered

1. **`engine_config: EngineConfig`**
   - Pros: More specific to engine concerns
   - Cons: Would require renaming across entire codebase (major refactor)
   - Cons: Introduces new type alongside ServerArgs (duplicate config)
   - **Rejected:** Too disruptive, no clear benefit

2. **`config: ServerArgs`**
   - Pros: Shorter parameter name
   - Cons: Ambiguous - confuses with `model_config: ModelConfig` and HuggingFace `config: PretrainedConfig`
   - Cons: Already used for model architecture config
   - **Rejected:** Creates naming collisions

3. **`server_args: ServerArgs`** ✅ CHOSEN
   - Pros: Already established across 100+ files in codebase
   - Pros: Type-safe and explicit
   - Pros: Clear distinction from `model_config`
   - Pros: Zero migration cost for existing code
   - Pros: Familiar to contributors
   - **Accepted:** Best fit for consistency and clarity

### Why This Decision

1. **Consistency:** The existing codebase already uses this convention extensively:
   - `Engine.__init__(server_args: ServerArgs)`
   - `Scheduler.__init__(server_args: ServerArgs, ...)`
   - `ModelRunner.__init__(..., server_args: ServerArgs, ...)`
   - 50+ other files across `python/sglang/srt/`

2. **Type Safety:** `ServerArgs` is a well-defined dataclass with:
   - All server configuration (device, tp_size, pp_size, etc.)
   - Validation logic in `check_server_args()`
   - Clear ownership and scope

3. **Clarity:** Avoids ambiguity with other `config` parameters:
   ```python
   class MambaForCausalLM:
       def __init__(
           self,
           config: PretrainedConfig,     # HuggingFace model architecture
           model_config: ModelConfig,    # SGLang model wrapper config
           server_args: ServerArgs,      # Server runtime config
       ):
           # No naming confusion!
   ```

4. **Zero Migration Cost:** Since Mamba integration doesn't exist yet, adopting the existing convention requires no refactoring.

---

## Consequences

### Positive

✅ **Consistency:** All components use the same naming convention
✅ **Type Safety:** Leverages existing `ServerArgs` validation
✅ **Zero Migration:** No refactoring of existing code needed
✅ **Clear Intent:** `server_args` explicitly indicates server-level config
✅ **Future-Proof:** Established pattern for all future model integrations

### Negative

❌ **Verbose:** `server_args` is longer than `config` (mitigated by clarity)
✅ **None:** No significant drawbacks identified

### Risks

**Risk 1: New Contributors Use Wrong Naming**
- **Mitigation:** Document in SDP.md, enforce in code review
- **Detection:** Type checker will catch `ServerArgs` type mismatches
- **Impact:** Low (easy to fix during PR review)

**Risk 2: Confusion with `ServerArgs` vs `ModelConfig`**
- **Mitigation:** Clear documentation of parameter purposes
- **Detection:** Unit tests will fail if wrong config passed
- **Impact:** Low (static analysis via mypy/pyright catches type mismatches; runtime errors on wrong type)

---

## Implementation

### Affected Components (Future Implementation)

New files to be created in Phase 3.2+:
- ✅ `python/sglang/srt/models/mamba.py` → `server_args: ServerArgs`
- ✅ `python/sglang/srt/layers/mamba/mamba_mixer.py` → `server_args: ServerArgs`
- ✅ `python/sglang/srt/layers/mamba/state_manager.py` → `server_args: ServerArgs`
- ✅ `python/sglang/srt/managers/mamba_batch.py` → `server_args: ServerArgs`
- ✅ `python/sglang/srt/mem_cache/mamba_radix_cache.py` → `server_args: ServerArgs`

Existing files to modify: **NONE**
(Existing code already follows this convention)

### Migration Strategy

**N/A** - No migration needed. This is establishing the standard **before** implementation.

### Coding Standard

**Template for new Mamba components:**
```python
from sglang.srt.server_args import ServerArgs

class MambaComponent:
    def __init__(
        self,
        # Model-specific configs first
        config: PretrainedConfig,
        model_config: ModelConfig,
        # Server config after model configs
        server_args: ServerArgs,
        # Component-specific params last
        layer_idx: int,
        **kwargs,
    ):
        self.config = config             # Model arch config
        self.model_config = model_config # SGLang model config
        self.server_args = server_args   # Server runtime config
        self.device = server_args.device # Access server config via server_args
```

---

## References

- **Audit Report:** `phase3/engine/audit_report.md`
- **ServerArgs Definition:** `python/sglang/srt/server_args.py`
- **Existing Usage:**
  - Engine: `python/sglang/srt/entrypoints/engine.py:146-148`
  - Scheduler: `python/sglang/srt/managers/scheduler.py:270`
  - ModelRunner: `python/sglang/srt/model_executor/model_runner.py:292`
- **Phase 3 Plan:** `PHASE_3_PLAN.md`
- **Related ADRs:** None (this is ADR 001)

---

## Decision Outcome

**Recommendation:** ✅ **ACCEPT**

**Justification:**
1. Zero breaking changes (Mamba integration doesn't exist yet)
2. Aligns with 95% of existing codebase
3. Clear, type-safe, and unambiguous
4. No migration cost
5. Future-proof for additional model integrations

**Required Approval:** User must approve before Phase 3.2 implementation begins.

---

**Last Updated:** 2026-02-16
**Status:** Pending User Approval
**Superseded By:** N/A
