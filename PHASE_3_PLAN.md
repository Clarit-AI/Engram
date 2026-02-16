# Phase 3: Engine Integration & Feature Completion Plan

**Status:** Phase 3.2 - Core Implementation COMPLETE
**Current Phase:** 3.2 Validation Gate
**Last Updated:** 2026-02-16
**Estimated Complexity:** High
**Agent Teams:** 7 specialized teams (6 implementation + 1 oversight)

---

## 📍 Current Progress Tracker

**Overall Progress:** ■■■⬜ 75% (Phase 3.3 complete - static analysis)

| Phase | Status | Start Date | Completion Date | Validation Status |
|-------|--------|------------|-----------------|-------------------|
| 3.1 Foundation | ✅ Complete | 2026-02-16 | 2026-02-16 | ✅ APPROVED |
| 3.2 Core Implementation | ✅ Complete | 2026-02-16 | 2026-02-16 | ✅ APPROVED |
| 3.3 Optimization | ✅ Complete (Static Analysis) | 2026-02-16 | 2026-02-16 | ⏳ Pending User Approval |
| 3.4 Final Audit | ⬜ Blocked | - | - | ⬜ Pending 3.3 |

**Latest Checkpoint:** Phase 3.3 COMPLETE (Static Analysis Mode) - Awaiting User Approval

---

## 🎯 Phase 3 Objectives

### Primary Goals
1. **Fix Engine Parameter Naming** - Resolve `engine_config` vs `config` inconsistency
2. **Implement Missing Features** - Prefill caching, chunked prefill, radix attention
3. **Performance Optimizations** - Memory efficiency, batch processing
4. **Testing Framework** - Comprehensive test coverage for Mamba integration

### Success Criteria
- [ ] All engine integration tests pass
- [ ] Parameter naming is consistent across codebase
- [ ] Prefill caching works with Mamba models
- [ ] Chunked prefill properly handles Mamba states
- [ ] Performance benchmarks meet or exceed baseline
- [ ] Documentation updated and accurate

---

## 🔄 Phase-Gate Validation Process

**CRITICAL:** Each phase MUST pass validation before proceeding to the next phase.

### Validation Gate Structure
```
Phase X.Y → Validation Gate X.Y → [PASS/FAIL] → Phase X.(Y+1)
                                        ↓
                                     [FAIL]
                                        ↓
                                   Fix Issues → Re-validate
```

### Validation Requirements (All Phases)
1. **Code Quality Check**
   - No syntax errors
   - Code style compliant
   - No new lint warnings

2. **Test Verification**
   - All existing tests still pass
   - New tests pass (if applicable)
   - No test regressions

3. **Documentation Check**
   - Changes documented in this file
   - Relevant ADRs created
   - Code comments updated

4. **Integration Check**
   - No breaking changes to other components
   - Dependencies satisfied
   - APIs consistent

5. **User Approval**
   - User reviews validation report
   - User approves to proceed
   - Any concerns addressed

---

## 👥 Agent Team Structure & Instructions

### **Team 1: Oversight & Coordination Agent** 🎯

**Agent ID:** `oversight-coordinator`
**Resume Context:** Check `/home/user/sglang-mamba/phase3/oversight/state.json`

**Role:** Strategic planning, team coordination, quality gates

**Responsibilities:**
- Maintain overall project timeline in this document
- Coordinate dependencies between teams
- Review critical design decisions
- Execute validation gates between phases
- Monitor for architectural violations
- Update progress tracker in real-time

**Key Deliverables:**
- Phase 3 progress dashboard (this document)
- Validation reports for each phase gate
- Risk assessment updates
- Go/no-go decisions for sub-phases

**Agent Instructions for Resumption:**
```
1. Read /home/user/sglang-mamba/PHASE_3_PLAN.md
2. Check "Current Progress Tracker" section for last completed phase
3. Review validation status of most recent phase
4. If validation failed, coordinate fixes before proceeding
5. If validation passed, prepare next phase kickoff
6. Update all timestamps and progress indicators
```

**State Files:**
- `/home/user/sglang-mamba/phase3/oversight/state.json` - Current phase state
- `/home/user/sglang-mamba/phase3/oversight/validation_reports/` - All validation reports

---

### **Team 2: Documentation Agent** 📚

**Agent ID:** `documentation-specialist`
**Resume Context:** Check `/home/user/sglang-mamba/phase3/docs/state.json`

**Role:** Knowledge management, documentation accuracy

**Responsibilities:**
- Update SDP.md with Phase 3 changes in real-time
- Maintain API documentation
- Document architectural decisions in ADR format
- Create migration guides
- Update inline code comments
- Update this plan document as tasks complete

**Key Deliverables:**
- Updated SDP.md (Section 9: Engine Integration)
- API reference updates
- Architecture decision records (ADRs)
- Code comment quality report
- Migration guide for parameter naming changes

**Files to Monitor:**
- `docs/SDP.md`
- `docs/mamba/ARCHITECTURE.md`
- All docstrings in modified files
- This file (`PHASE_3_PLAN.md`)

**Agent Instructions for Resumption:**
```
1. Read PHASE_3_PLAN.md to identify completed tasks
2. Check phase3/docs/pending_updates.json for uncommitted doc changes
3. Review phase3/docs/adr/ for draft ADRs needing completion
4. Scan git log since last checkpoint for code changes needing docs
5. Update SDP.md sections corresponding to completed work
6. Mark documentation tasks as complete in PHASE_3_PLAN.md
```

**State Files:**
- `/home/user/sglang-mamba/phase3/docs/state.json` - Documentation progress
- `/home/user/sglang-mamba/phase3/docs/pending_updates.json` - Uncommitted changes
- `/home/user/sglang-mamba/phase3/docs/adr/` - Architecture decision records

---

### **Team 3: Engine Parameter Refactoring Agent** 🔧

**Agent ID:** `engine-refactoring-specialist`
**Resume Context:** Check `/home/user/sglang-mamba/phase3/engine/state.json`

**Role:** Fix naming inconsistencies, update engine integration

**Responsibilities:**
- Audit all uses of `engine_config` vs `config`
- Standardize parameter names across codebase
- Update Engine class initialization
- Fix ModelRunner integration
- Update all call sites
- Ensure backward compatibility where needed

**Key Files:**
- `python/sglang/srt/managers/scheduler.py`
- `python/sglang/srt/model_executor/model_runner.py`
- `python/sglang/srt/managers/schedule_batch.py`
- `python/sglang/srt/models/mamba.py`

**Refactoring Steps:**

#### Step 1: Audit Phase ✅/❌
**Status:** Not Started
- [ ] Grep all uses of `engine_config` parameter
- [ ] Grep all uses of `config` parameter in engine context
- [ ] Document all call sites in `/phase3/engine/audit_report.md`
- [ ] Identify inconsistencies and conflicts
- [ ] Create refactoring map: old_name → new_name

**Output:** `/home/user/sglang-mamba/phase3/engine/audit_report.md`

#### Step 2: Design Phase ✅/❌
**Status:** Not Started
- [ ] Propose naming standard (recommend: `server_args` for consistency)
- [ ] Document decision rationale in ADR
- [ ] Get user approval on naming choice
- [ ] Create migration plan for each file
- [ ] Identify risk areas (breaking changes)

**Output:** `/home/user/sglang-mamba/phase3/docs/adr/001-engine-parameter-naming.md`

#### Step 3: Implementation Phase ✅/❌
**Status:** Not Started
- [ ] Update Engine class signature
- [ ] Update ModelRunner calls
- [ ] Update Scheduler initialization
- [ ] Update all other call sites
- [ ] Add deprecation warnings if keeping old names temporarily
- [ ] Run test suite after each file change

**Validation:** Run `pytest python/sglang/test/srt/ -k engine -v` after each change

#### Step 4: Testing Phase ✅/❌
**Status:** Not Started
- [ ] Write unit tests for Engine initialization with new params
- [ ] Write integration tests for ModelRunner
- [ ] Write end-to-end scheduler tests
- [ ] Verify backward compatibility (if applicable)
- [ ] Run full test suite

**Testing Requirements:**
- Unit tests for Engine initialization
- Integration tests for ModelRunner
- End-to-end scheduler tests

**Agent Instructions for Resumption:**
```
1. Read /home/user/sglang-mamba/phase3/engine/state.json
2. Check which refactoring step was last completed
3. Review /phase3/engine/audit_report.md for findings
4. Check ADR 001 for approved naming standard
5. Continue from last incomplete step
6. Update state.json after each step completion
```

**State Files:**
- `/home/user/sglang-mamba/phase3/engine/state.json` - Progress state
- `/home/user/sglang-mamba/phase3/engine/audit_report.md` - Audit findings
- `/home/user/sglang-mamba/phase3/engine/refactoring_map.json` - Name mappings

---

### **Team 4: Prefill Feature Implementation Agent** ⚡

**Agent ID:** `prefill-implementation-specialist`
**Resume Context:** Check `/home/user/sglang-mamba/phase3/prefill/state.json`

**Role:** Implement prefill caching and chunked prefill for Mamba

**Responsibilities:**
- Implement RadixCache integration for Mamba states
- Add chunked prefill support
- Optimize state management during prefill
- Handle edge cases (empty cache, cache eviction)
- Ensure state continuity across chunks

**Key Files:**
- `python/sglang/srt/layers/mamba/prefill.py` (create if needed)
- `python/sglang/srt/managers/schedule_batch.py`
- `python/sglang/srt/mem_cache/radix_cache.py`
- `python/sglang/srt/layers/mamba/state_manager.py`

**Implementation Tasks:**

#### Task 4.1: Radix Cache Integration ✅/❌
**Status:** Not Started
**Dependencies:** Engine refactoring complete

**Subtasks:**
- [ ] Analyze current RadixCache implementation
- [ ] Design Mamba state storage format
- [ ] Extend RadixCache with Mamba state methods
- [ ] Implement cache_mamba_state()
- [ ] Implement retrieve_mamba_state()
- [ ] Add cache invalidation logic
- [ ] Write unit tests

**Implementation Skeleton:**
```python
# In python/sglang/srt/mem_cache/radix_cache.py or new file
class MambaRadixCache:
    def cache_mamba_state(self, prefix_tokens: List[int],
                          mamba_state: MambaState) -> str:
        """
        Cache Mamba state for given prefix tokens.

        Args:
            prefix_tokens: Token sequence that generated this state
            mamba_state: The state tensors to cache

        Returns:
            cache_key: Unique identifier for cached state
        """
        # TODO: Implement state serialization
        # TODO: Store in radix tree by token prefix
        # TODO: Return cache key for retrieval
        pass

    def retrieve_mamba_state(self, prefix_tokens: List[int]) -> Optional[MambaState]:
        """
        Retrieve cached Mamba state for prefix.

        Args:
            prefix_tokens: Token sequence to look up

        Returns:
            Cached MambaState if found, None otherwise
        """
        # TODO: Search radix tree for longest matching prefix
        # TODO: Deserialize and return state
        # TODO: Handle cache misses gracefully
        pass
```

#### Task 4.2: Chunked Prefill Support ✅/❌
**Status:** Not Started
**Dependencies:** Task 4.1 complete

**Subtasks:**
- [ ] Design chunking strategy (chunk size, overlap)
- [ ] Implement chunk processor
- [ ] Add state propagation between chunks
- [ ] Handle edge cases (last chunk, varying lengths)
- [ ] Add progress tracking
- [ ] Write integration tests

**Implementation Skeleton:**
```python
# In ScheduleBatch or new MambaPrefillManager
def chunked_prefill_mamba(self, input_ids: torch.Tensor,
                          chunk_size: int = 512) -> MambaState:
    """
    Process long prefill in chunks while maintaining state continuity.

    Args:
        input_ids: Full input sequence [batch_size, seq_len]
        chunk_size: Size of each processing chunk

    Returns:
        final_state: Accumulated Mamba state after all chunks
    """
    # TODO: Split input into chunks
    # TODO: Process first chunk, initialize state
    # TODO: For each subsequent chunk:
    #         - Load state from previous chunk
    #         - Process chunk with state
    #         - Update accumulated state
    # TODO: Return final state
    pass
```

#### Task 4.3: State Continuity Validation ✅/❌
**Status:** Not Started
**Dependencies:** Task 4.2 complete

**Subtasks:**
- [ ] Implement state checksum/verification
- [ ] Add logging for state transitions
- [ ] Test state continuity across chunk boundaries
- [ ] Handle cache miss → recomputation path
- [ ] Add performance metrics (cache hit rate)
- [ ] Write E2E tests with various sequence lengths

**Testing Requirements:**
- Unit tests for cache operations
- Integration tests with various prefix lengths (64, 512, 2048, 4096 tokens)
- Performance benchmarks (cache hit rate ≥70%, latency)
- State continuity tests (compare chunked vs non-chunked outputs)

**Agent Instructions for Resumption:**
```
1. Read /home/user/sglang-mamba/phase3/prefill/state.json
2. Check which task (4.1, 4.2, or 4.3) was last in progress
3. Review /phase3/prefill/design_notes.md for decisions
4. Check if engine refactoring dependency is satisfied
5. Continue from last incomplete task
6. Run tests after each subtask
7. Update state.json and PHASE_3_PLAN.md progress
```

**State Files:**
- `/home/user/sglang-mamba/phase3/prefill/state.json` - Task progress
- `/home/user/sglang-mamba/phase3/prefill/design_notes.md` - Design decisions
- `/home/user/sglang-mamba/phase3/prefill/test_results.json` - Test outcomes

---

### **Team 5: Performance Optimization Agent** 🚀

**Agent ID:** `performance-optimization-specialist`
**Resume Context:** Check `/home/user/sglang-mamba/phase3/perf/state.json`

**Role:** Optimize memory usage and batch processing

**Responsibilities:**
- Profile current Mamba implementation
- Optimize state tensor operations
- Improve batch processing efficiency
- Reduce memory fragmentation
- Add performance monitoring hooks
- Benchmark all optimizations

**Key Files:**
- `python/sglang/srt/layers/mamba/mamba_mixer.py`
- `python/sglang/srt/layers/mamba/state_manager.py`
- `python/sglang/srt/model_executor/model_runner.py`

**Optimization Tasks:**

#### Task 5.1: Memory Profiling ✅/❌
**Status:** Not Started
**Dependencies:** None (can start immediately)

**Subtasks:**
- [ ] Install profiling tools (memory_profiler, torch.profiler)
- [ ] Create baseline benchmark script
- [ ] Profile memory usage during inference
- [ ] Identify memory hotspots
- [ ] Document findings in profiling report
- [ ] Establish baseline metrics

**Profiling Commands:**
```bash
# Memory profiling
python -m memory_profiler python/sglang/bench/benchmark_mamba.py

# PyTorch profiling
python -c "import torch.profiler; ..."

# Save results to /phase3/perf/baseline_profile.json
```

**Output:** `/home/user/sglang-mamba/phase3/perf/baseline_profile.json`

#### Task 5.2: Tensor Operation Optimization ✅/❌
**Status:** Not Started
**Dependencies:** Task 5.1 complete

**Subtasks:**
- [ ] Identify unnecessary tensor copies
- [ ] Convert operations to in-place where safe
- [ ] Optimize conv1d implementation (use CUDA kernels if available)
- [ ] Reduce intermediate tensor allocations
- [ ] Add memory pooling for state tensors
- [ ] Re-profile after changes
- [ ] Document optimizations in ADR

**Optimization Targets:**
- In-place operations: `tensor.add_()` instead of `tensor + other`
- Reuse buffers: Pre-allocate and reuse state buffers
- Kernel fusion: Combine operations where possible

#### Task 5.3: Batch Processing Optimization ✅/❌
**Status:** Not Started
**Dependencies:** Task 5.2 complete

**Subtasks:**
- [ ] Analyze batching strategy
- [ ] Optimize padding for variable-length sequences
- [ ] Implement dynamic batching hints
- [ ] Add batch size auto-tuning
- [ ] Test with various batch sizes (1, 4, 8, 16, 32)
- [ ] Measure throughput improvements

**Benchmarking Requirements:**
- Memory usage before/after (target: 20% reduction)
- Throughput in tokens/sec (target: 15% improvement)
- Latency p50, p95, p99 (maintain or improve)
- Batch size scaling analysis (linear scaling expected)

**Agent Instructions for Resumption:**
```
1. Read /home/user/sglang-mamba/phase3/perf/state.json
2. Check last completed optimization task
3. Review baseline_profile.json for benchmark data
4. Compare current metrics with baseline
5. Continue from next optimization task
6. Run benchmarks after each optimization
7. Update state.json and document improvements
```

**State Files:**
- `/home/user/sglang-mamba/phase3/perf/state.json` - Optimization progress
- `/home/user/sglang-mamba/phase3/perf/baseline_profile.json` - Baseline metrics
- `/home/user/sglang-mamba/phase3/perf/optimizations.md` - Applied optimizations
- `/home/user/sglang-mamba/phase3/perf/benchmarks/` - All benchmark results

---

### **Team 6: Testing & Validation Agent** 🧪

**Agent ID:** `testing-validation-specialist`
**Resume Context:** Check `/home/user/sglang-mamba/phase3/test/state.json`

**Role:** Comprehensive testing across all changes

**Responsibilities:**
- Write unit tests for new features
- Create integration tests
- Develop end-to-end test scenarios
- Performance regression testing
- Edge case validation
- Maintain test coverage ≥85%

**Test Categories:**

#### Unit Tests ✅/❌
**Status:** Not Started
**Location:** `python/sglang/test/srt/layers/mamba/`

**Test Files to Create/Update:**
- [ ] `test_mamba_engine.py` - Engine config tests
- [ ] `test_mamba_state_cache.py` - RadixCache tests
- [ ] `test_mamba_chunked_prefill.py` - Chunked prefill tests
- [ ] `test_mamba_state_manager.py` - State management tests

**Sample Test Structure:**
```python
# test_mamba_engine.py
import pytest
from sglang.srt.managers.scheduler import Scheduler

def test_engine_config_initialization():
    """Test Engine accepts correct config parameter"""
    # TODO: Create test config
    # TODO: Initialize engine with new parameter name
    # TODO: Assert successful initialization
    pass

def test_mamba_state_caching():
    """Test RadixCache stores/retrieves Mamba states"""
    # TODO: Create mock state
    # TODO: Cache state with prefix tokens
    # TODO: Retrieve and verify state matches
    pass

def test_chunked_prefill_state_continuity():
    """Test state remains consistent across chunks"""
    # TODO: Create long input sequence
    # TODO: Process with chunked prefill
    # TODO: Compare with non-chunked processing
    # TODO: Assert outputs match within tolerance
    pass
```

#### Integration Tests ✅/❌
**Status:** Not Started
**Location:** `python/sglang/test/srt/`

**Test Files:**
- [ ] `test_mamba_integration.py`

**Sample Tests:**
```python
# test_mamba_integration.py
def test_scheduler_mamba_batch():
    """Test Scheduler creates valid Mamba batches"""
    # TODO: Initialize scheduler with Mamba model
    # TODO: Submit multiple requests
    # TODO: Verify batch structure
    pass

def test_model_runner_mamba_forward():
    """Test ModelRunner executes Mamba forward pass"""
    # TODO: Create ModelRunner with Mamba
    # TODO: Execute forward pass
    # TODO: Verify output shape and values
    pass
```

#### End-to-End Tests ✅/❌
**Status:** Not Started
**Location:** `python/sglang/test/srt/`

**Test Files:**
- [ ] `test_mamba_e2e.py`

**Sample Tests:**
```python
# test_mamba_e2e.py
def test_mamba_inference_pipeline():
    """Test complete inference pipeline with Mamba model"""
    # TODO: Start server with Mamba model
    # TODO: Send inference request
    # TODO: Verify response correctness
    # TODO: Cleanup
    pass

def test_mamba_with_prefill_cache():
    """Test inference with cached prefill states"""
    # TODO: Send request with common prefix
    # TODO: Send another request with same prefix
    # TODO: Verify cache hit occurred
    # TODO: Compare latencies (2nd should be faster)
    pass
```

**Coverage Target:** 85%+ for new code

**Agent Instructions for Resumption:**
```
1. Read /home/user/sglang-mamba/phase3/test/state.json
2. Check which test categories are complete
3. Review test_results.json for last test run outcomes
4. Identify failed tests and root causes
5. Continue writing missing tests
6. Run test suite: pytest python/sglang/test/srt/ -v --cov
7. Update coverage report in /phase3/test/coverage.html
8. Update state.json with progress
```

**State Files:**
- `/home/user/sglang-mamba/phase3/test/state.json` - Test progress
- `/home/user/sglang-mamba/phase3/test/test_results.json` - Test outcomes
- `/home/user/sglang-mamba/phase3/test/coverage.html` - Coverage report

---

### **Team 7: Audit & Quality Assurance Agent** 🔍

**Agent ID:** `audit-qa-specialist`
**Resume Context:** Check `/home/user/sglang-mamba/phase3/audit/state.json`

**Role:** Final validation, code review, regression prevention

**Responsibilities:**
- Comprehensive code review of all changes
- Architecture compliance check
- Security audit (state isolation, memory safety)
- Documentation completeness review
- Final sign-off before merge
- Generate validation reports for phase gates

**Audit Checklist:**

#### Phase Gate Validation Template
```markdown
# Phase X.Y Validation Report

**Date:** YYYY-MM-DD
**Validator:** [Agent/User]
**Phase:** X.Y [Phase Name]

## Code Quality ✅/❌
- [ ] No syntax errors
- [ ] No new lint warnings
- [ ] Code style compliant (black, isort)
- [ ] Type hints added where appropriate

## Testing ✅/❌
- [ ] All existing tests pass
- [ ] New tests pass (if applicable)
- [ ] Coverage ≥85% for new code
- [ ] No test regressions

## Documentation ✅/❌
- [ ] Changes documented in PHASE_3_PLAN.md
- [ ] Relevant ADRs created
- [ ] Code comments updated
- [ ] SDP.md updated

## Integration ✅/❌
- [ ] No breaking changes
- [ ] Dependencies satisfied
- [ ] APIs consistent
- [ ] Backward compatibility maintained

## Security ✅/❌
- [ ] No state leaks between requests
- [ ] Memory safety verified
- [ ] Input validation present
- [ ] No race conditions

## Architecture ✅/❌
- [ ] Follows SDP architecture
- [ ] No violations of design principles
- [ ] Proper separation of concerns
- [ ] Code duplication minimized

## Validation Outcome: [PASS/FAIL]

**Blocking Issues:** [List any issues that must be fixed]

**Recommendations:** [Optional improvements for future]

**Approval:** [User signature/approval]
```

**Agent Instructions for Resumption:**
```
1. Read /home/user/sglang-mamba/phase3/audit/state.json
2. Check which phases have been validated
3. Review pending validation gates
4. Run comprehensive audit checklist for current phase
5. Generate validation report in /phase3/audit/validation_reports/
6. Present report to user for approval
7. Update PHASE_3_PLAN.md with validation status
8. If PASS: Clear next phase to proceed
9. If FAIL: Document blocking issues, notify other agents
```

**State Files:**
- `/home/user/sglang-mamba/phase3/audit/state.json` - Audit progress
- `/home/user/sglang-mamba/phase3/audit/validation_reports/` - All validation reports
- `/home/user/sglang-mamba/phase3/audit/issues.json` - Tracked issues

---

## 📋 Detailed Execution Plan with Validation Gates

### ⚙️ Phase 3.1: Foundation

**Goal:** Set up infrastructure and resolve parameter naming
**Duration:** Days 1-2
**Status:** 🟡 Ready to Start

#### Tasks

##### Task 3.1.1: Infrastructure Setup ✅/❌
**Owner:** Oversight Agent
**Status:** Not Started
- [ ] Create phase3/ directory structure
- [ ] Initialize state files for all agents
- [ ] Set up shared knowledge base
- [ ] Create progress tracking dashboard
- [ ] Initialize git branch for phase 3 work

**Commands:**
```bash
mkdir -p phase3/{oversight,docs,engine,prefill,perf,test,audit}
mkdir -p phase3/docs/{adr,api}
mkdir -p phase3/{oversight,engine,prefill,perf,test,audit}/reports
# Create state.json files for each agent
```

##### Task 3.1.2: Parameter Naming Audit ✅/❌
**Owner:** Engine Refactoring Agent
**Status:** Not Started
**Depends on:** 3.1.1
- [ ] Grep all uses of `engine_config`
- [ ] Grep all uses of `config` in engine context
- [ ] Document findings in audit_report.md
- [ ] Create refactoring map
- [ ] Identify risk areas

**Output:** `phase3/engine/audit_report.md`

##### Task 3.1.3: Parameter Naming Design ✅/❌
**Owner:** Engine Refactoring Agent
**Status:** Not Started
**Depends on:** 3.1.2
- [ ] Propose naming standard
- [ ] Write ADR with rationale
- [ ] Get user approval
- [ ] Document migration plan

**Output:** `phase3/docs/adr/001-engine-parameter-naming.md`

##### Task 3.1.4: Test Framework Setup ✅/❌
**Owner:** Testing Agent
**Status:** Not Started
**Depends on:** 3.1.1
- [ ] Create test file structure
- [ ] Set up pytest configuration
- [ ] Create test fixtures for Mamba models
- [ ] Write test plan document
- [ ] Set up coverage reporting

**Output:** Test framework scaffolding in `python/sglang/test/srt/layers/mamba/`

##### Task 3.1.5: Documentation Structure ✅/❌
**Owner:** Documentation Agent
**Status:** Not Started
**Depends on:** 3.1.1
- [ ] Set up ADR template and directory
- [ ] Create API doc structure
- [ ] Identify SDP sections needing updates
- [ ] Set up automated doc generation

**Output:** `phase3/docs/` structure ready

#### Deliverables:
- [ ] Parameter naming standard document
- [ ] Test framework scaffolding
- [ ] Documentation structure ready
- [ ] Progress dashboard operational
- [ ] All agent state files initialized

---

### 🚦 VALIDATION GATE 3.1

**Validator:** Audit Agent + User
**Validation Checklist:**

#### Infrastructure Validation ✅/❌
- [ ] All phase3/ directories exist and are properly structured
- [ ] State files initialized for all 7 agents
- [ ] Git branch created and clean

#### Audit Validation ✅/❌
- [ ] audit_report.md exists and is comprehensive
- [ ] All engine_config uses documented
- [ ] Refactoring map is complete and accurate

#### Design Validation ✅/❌
- [ ] ADR 001 created and approved by user
- [ ] Naming standard is clear and unambiguous
- [ ] Migration plan is feasible

#### Testing Validation ✅/❌
- [ ] Test directories created
- [ ] Pytest configuration correct
- [ ] Test fixtures work
- [ ] Existing tests still pass

#### Documentation Validation ✅/❌
- [ ] ADR structure in place
- [ ] Documentation plan clear

#### User Approval ✅/❌
- [ ] User reviews validation report
- [ ] User approves naming standard
- [ ] User authorizes Phase 3.2 to proceed

**Validation Report:** `/home/user/sglang-mamba/phase3/audit/validation_reports/phase_3.1_validation.md`

**Validation Outcome:** ⬜ Pending

**If PASS:** → Proceed to Phase 3.2
**If FAIL:** → Fix blocking issues, re-run validation

---

### ⚙️ Phase 3.2: Core Implementation

**Goal:** Implement engine fixes and prefill features
**Duration:** Days 3-5
**Status:** ⬜ Blocked (Pending 3.1 Validation)
**Dependencies:** Phase 3.1 PASS

#### Tasks

##### Task 3.2.1: Apply Engine Parameter Refactoring ✅/❌
**Owner:** Engine Refactoring Agent
**Status:** Blocked by 3.1
**Depends on:** Phase 3.1 validation PASS
- [ ] Update Engine class signature
- [ ] Update ModelRunner calls to Engine
- [ ] Update Scheduler initialization
- [ ] Update all other call sites per refactoring map
- [ ] Add deprecation warnings (if needed)
- [ ] Run tests after each file change
- [ ] Document all changes

**Validation after each step:** Run `pytest python/sglang/test/srt/ -k engine -v`

##### Task 3.2.2: Implement RadixCache for Mamba States ✅/❌
**Owner:** Prefill Implementation Agent
**Status:** Blocked by 3.1
**Depends on:** Task 3.2.1 complete
- [ ] Analyze RadixCache implementation
- [ ] Design Mamba state storage format
- [ ] Implement cache_mamba_state()
- [ ] Implement retrieve_mamba_state()
- [ ] Add cache invalidation logic
- [ ] Write unit tests
- [ ] Run tests

**Output:** Extended RadixCache in `python/sglang/srt/mem_cache/radix_cache.py`

##### Task 3.2.3: Unit Test Development ✅/❌
**Owner:** Testing Agent
**Status:** Blocked by 3.1
**Parallel with:** 3.2.1, 3.2.2
- [ ] Write tests for engine parameter changes
- [ ] Write tests for RadixCache extensions
- [ ] Ensure tests pass
- [ ] Verify coverage ≥85%

**Output:** Test files in `python/sglang/test/srt/layers/mamba/`

##### Task 3.2.4: Performance Baseline ✅/❌
**Owner:** Performance Agent
**Status:** Blocked by 3.1
**Parallel with:** 3.2.1, 3.2.2
- [ ] Set up profiling tools
- [ ] Run baseline benchmarks
- [ ] Document baseline metrics
- [ ] Identify optimization targets

**Output:** `phase3/perf/baseline_profile.json`

##### Task 3.2.5: Real-time Documentation ✅/❌
**Owner:** Documentation Agent
**Status:** Blocked by 3.1
**Parallel with:** All 3.2 tasks
- [ ] Update PHASE_3_PLAN.md as tasks complete
- [ ] Create ADRs for design decisions
- [ ] Update SDP.md Section 9
- [ ] Document parameter changes
- [ ] Update API docs

**Output:** Updated docs in real-time

#### Deliverables:
- [ ] Refactored engine parameters (all call sites updated)
- [ ] Working RadixCache for Mamba states (unit tested)
- [ ] Initial performance baseline established
- [ ] Unit test suite (≥85% coverage)
- [ ] Updated documentation

---

### 🚦 VALIDATION GATE 3.2

**Validator:** Audit Agent + User

#### Code Quality ✅/❌
- [ ] All refactoring changes pass linting
- [ ] No syntax errors
- [ ] Code style consistent
- [ ] Type hints present

#### Testing ✅/❌
- [ ] All existing tests pass (100%)
- [ ] New unit tests pass (100%)
- [ ] Coverage ≥85% for new code
- [ ] No test regressions

#### Functionality ✅/❌
- [ ] Engine initialization works with new parameter name
- [ ] RadixCache can store Mamba states
- [ ] RadixCache can retrieve Mamba states
- [ ] Cache invalidation works correctly

#### Performance ✅/❌
- [ ] Baseline benchmarks recorded
- [ ] No performance regression vs. baseline
- [ ] Memory usage within expected range

#### Documentation ✅/❌
- [ ] PHASE_3_PLAN.md updated
- [ ] ADRs complete
- [ ] SDP.md updated
- [ ] API docs accurate

#### Integration ✅/❌
- [ ] No breaking changes to other components
- [ ] Backward compatibility maintained (if applicable)
- [ ] Dependencies satisfied

#### User Approval ✅/❌
- [ ] User reviews validation report
- [ ] User approves to proceed
- [ ] Any concerns addressed

**Validation Report:** `/home/user/sglang-mamba/phase3/audit/validation_reports/phase_3.2_validation.md`

**Validation Outcome:** ⬜ Pending

**If PASS:** → Proceed to Phase 3.3
**If FAIL:** → Fix blocking issues, re-run validation

---

### ⚙️ Phase 3.3: Optimization & Integration

**Goal:** Complete features, optimize, integrate all components
**Duration:** Days 6-7
**Status:** ⬜ Blocked (Pending 3.2 Validation)
**Dependencies:** Phase 3.2 PASS

#### Tasks

##### Task 3.3.1: Implement Chunked Prefill ✅/❌
**Owner:** Prefill Implementation Agent
**Status:** Blocked by 3.2
**Depends on:** Phase 3.2 validation PASS
- [ ] Design chunking strategy
- [ ] Implement chunk processor
- [ ] Add state propagation between chunks
- [ ] Handle edge cases
- [ ] Write integration tests
- [ ] Verify state continuity

**Output:** Chunked prefill in `python/sglang/srt/managers/schedule_batch.py`

##### Task 3.3.2: State Continuity Validation ✅/❌
**Owner:** Prefill Implementation Agent
**Status:** Blocked by 3.2
**Depends on:** Task 3.3.1
- [ ] Implement state checksums
- [ ] Add state transition logging
- [ ] Test across chunk boundaries
- [ ] Handle cache miss → recomputation
- [ ] Add cache hit rate metrics
- [ ] Write E2E tests

##### Task 3.3.3: Apply Performance Optimizations ✅/❌
**Owner:** Performance Agent
**Status:** Blocked by 3.2
**Parallel with:** 3.3.1
- [ ] Optimize tensor operations (in-place)
- [ ] Reduce memory allocations
- [ ] Optimize conv1d (CUDA kernels)
- [ ] Improve batching strategy
- [ ] Optimize padding
- [ ] Re-profile and measure improvements

**Output:** Optimized code + benchmark comparisons

##### Task 3.3.4: Integration Testing ✅/❌
**Owner:** Testing Agent
**Status:** Blocked by 3.2
**Depends on:** Tasks 3.3.1, 3.3.2, 3.3.3 complete
- [ ] Write integration tests for Scheduler + Mamba
- [ ] Write integration tests for ModelRunner + Mamba
- [ ] Write E2E tests for full pipeline
- [ ] Test with prefill cache enabled
- [ ] Verify cache hit rate ≥70%
- [ ] Run full test suite

**Output:** Comprehensive test suite passing

##### Task 3.3.5: Complete Documentation ✅/❌
**Owner:** Documentation Agent
**Status:** Blocked by 3.2
**Parallel with:** All 3.3 tasks
- [ ] Complete all pending ADRs
- [ ] Finalize SDP.md updates
- [ ] Complete API documentation
- [ ] Create migration guide
- [ ] Update inline comments
- [ ] Final documentation review

**Output:** Complete and accurate documentation

##### Task 3.3.6: Integration Review ✅/❌
**Owner:** Oversight Agent
**Status:** Blocked by 3.2
**Depends on:** All 3.3 tasks near completion
- [ ] Review all component integrations
- [ ] Verify no breaking changes
- [ ] Check dependency satisfaction
- [ ] Coordinate final fixes
- [ ] Prepare for Phase 3.4 audit

#### Deliverables:
- [ ] Complete prefill feature set (caching + chunking)
- [ ] Optimized implementation (20% memory, 15% throughput)
- [ ] Full test suite passing (100%)
- [ ] Complete documentation (SDP, ADRs, API docs)
- [ ] Integration verified

---

### 🚦 VALIDATION GATE 3.3

**Validator:** Audit Agent + User

#### Functionality ✅/❌
- [ ] Chunked prefill works correctly
- [ ] State continuity verified across chunks
- [ ] Prefill cache hit rate ≥70%
- [ ] Cache miss handling works
- [ ] Edge cases handled

#### Performance ✅/❌
- [ ] Memory usage reduced by ≥20% (vs baseline)
- [ ] Throughput increased by ≥15% (vs baseline)
- [ ] Latency p95 maintained or improved
- [ ] Batch processing scales linearly
- [ ] No performance regressions

#### Testing ✅/❌
- [ ] All unit tests pass (100%)
- [ ] All integration tests pass (100%)
- [ ] All E2E tests pass (100%)
- [ ] Coverage ≥85%
- [ ] No flaky tests

#### Code Quality ✅/❌
- [ ] All optimizations pass linting
- [ ] No new warnings
- [ ] Code style consistent
- [ ] No code duplication

#### Documentation ✅/❌
- [ ] All ADRs complete
- [ ] SDP.md fully updated
- [ ] API docs complete
- [ ] Migration guide created
- [ ] Comments accurate

#### Integration ✅/❌
- [ ] All components integrate correctly
- [ ] No breaking changes
- [ ] Backward compatibility maintained
- [ ] APIs consistent

#### User Approval ✅/❌
- [ ] User reviews validation report
- [ ] User approves feature completeness
- [ ] User approves performance improvements
- [ ] User authorizes final audit

**Validation Report:** `/home/user/sglang-mamba/phase3/audit/validation_reports/phase_3.3_validation.md`

**Validation Outcome:** ⬜ Pending

**If PASS:** → Proceed to Phase 3.4
**If FAIL:** → Fix blocking issues, re-run validation

---

### ⚙️ Phase 3.4: Final Audit & Validation

**Goal:** Final validation, audit, and sign-off
**Duration:** Day 8
**Status:** ⬜ Blocked (Pending 3.3 Validation)
**Dependencies:** Phase 3.3 PASS

#### Tasks (Sequential)

##### Task 3.4.1: Comprehensive Code Review ✅/❌
**Owner:** Audit Agent
**Status:** Blocked by 3.3
**Depends on:** Phase 3.3 validation PASS
- [ ] Review all changed files
- [ ] Check architectural compliance
- [ ] Verify design patterns followed
- [ ] Check code quality and style
- [ ] Identify any remaining issues

**Output:** Code review report

##### Task 3.4.2: Security Audit ✅/❌
**Owner:** Audit Agent
**Status:** Blocked by 3.3
**Depends on:** Task 3.4.1
- [ ] Verify state isolation between requests
- [ ] Check memory safety
- [ ] Verify input validation
- [ ] Check for race conditions
- [ ] Review error handling

**Output:** Security audit report

##### Task 3.4.3: Regression Testing ✅/❌
**Owner:** Testing Agent
**Status:** Blocked by 3.3
**Parallel with:** 3.4.1, 3.4.2
- [ ] Run full test suite
- [ ] Run extended test suite (if available)
- [ ] Test with multiple model sizes
- [ ] Test with various batch sizes
- [ ] Verify no regressions

**Output:** Regression test report

##### Task 3.4.4: Final Performance Benchmarks ✅/❌
**Owner:** Performance Agent
**Status:** Blocked by 3.3
**Parallel with:** 3.4.3
- [ ] Run comprehensive benchmarks
- [ ] Compare with baseline metrics
- [ ] Verify all performance targets met
- [ ] Document final performance characteristics
- [ ] Create performance comparison report

**Output:** Final benchmark report

##### Task 3.4.5: Final Documentation Pass ✅/❌
**Owner:** Documentation Agent
**Status:** Blocked by 3.3
**Parallel with:** 3.4.1-3.4.4
- [ ] Final review of all documentation
- [ ] Ensure completeness
- [ ] Check accuracy against code
- [ ] Fix any inconsistencies
- [ ] Finalize PHASE_3_PLAN.md

**Output:** Documentation sign-off

##### Task 3.4.6: Oversight Review & Go/No-Go ✅/❌
**Owner:** Oversight Agent
**Status:** Blocked by 3.3
**Depends on:** All 3.4.1-3.4.5 complete
- [ ] Review all audit reports
- [ ] Review all test results
- [ ] Review all benchmarks
- [ ] Review documentation
- [ ] Make go/no-go recommendation
- [ ] Prepare final presentation for user

**Output:** Final phase review and recommendation

#### Deliverables:
- [ ] Code review report (no critical issues)
- [ ] Security audit report (no vulnerabilities)
- [ ] Regression test report (100% pass)
- [ ] Performance benchmark report (targets met)
- [ ] Documentation sign-off
- [ ] Phase 3 completion recommendation

---

### 🚦 FINAL VALIDATION GATE 3.4

**Validator:** Audit Agent + Oversight Agent + User

#### All Phase 3 Objectives Met ✅/❌
- [ ] Engine parameters fixed and consistent
- [ ] Prefill caching implemented and working
- [ ] Chunked prefill implemented and working
- [ ] Performance optimizations applied
- [ ] Testing framework complete

#### Code Quality ✅/❌
- [ ] No critical issues in code review
- [ ] All tests passing (100%)
- [ ] Coverage ≥85%
- [ ] No lint warnings
- [ ] Code style consistent

#### Security ✅/❌
- [ ] No security vulnerabilities
- [ ] State isolation verified
- [ ] Memory safety confirmed
- [ ] Input validation present
- [ ] Error handling robust

#### Performance ✅/❌
- [ ] All performance targets met
- [ ] Memory usage acceptable
- [ ] Throughput meets targets
- [ ] Latency meets targets
- [ ] Cache hit rate ≥70%

#### Documentation ✅/❌
- [ ] All documentation complete
- [ ] All ADRs finalized
- [ ] SDP.md accurate and up-to-date
- [ ] API docs complete
- [ ] Migration guide available

#### Integration ✅/❌
- [ ] No breaking changes
- [ ] Backward compatibility maintained
- [ ] All components integrate correctly
- [ ] No regression in existing features

#### User Final Approval ✅/❌
- [ ] User reviews all audit reports
- [ ] User reviews benchmark results
- [ ] User reviews documentation
- [ ] User approves Phase 3 completion
- [ ] User authorizes merge/deployment

**Final Validation Report:** `/home/user/sglang-mamba/phase3/audit/validation_reports/phase_3.4_final_validation.md`

**Validation Outcome:** ⬜ Pending

**If PASS:** → Phase 3 Complete! 🎉
**If FAIL:** → Address final issues, re-audit

---

## 🔄 Communication & Coordination

### Daily Sync Points
**Time:** End of each development session
**Participants:** All agent teams
**Format:**
- Each team reports: Progress, Blockers, Next steps
- Oversight agent identifies cross-team dependencies
- Documentation agent captures decisions
- Update PHASE_3_PLAN.md with progress

### Escalation Path
1. **Technical Issues** → Oversight Agent → Architecture review
2. **Blocking Issues** → Immediate escalation to user
3. **Design Decisions** → Document in ADR, get user approval
4. **Validation Failures** → Audit Agent → Issue tracking → Resolution → Re-validation

### Artifact Sharing
**Shared Knowledge Base:** `/home/user/sglang-mamba/phase3/`

```
phase3/
├── oversight/
│   ├── state.json
│   ├── progress_dashboard.md
│   └── validation_reports/
├── docs/
│   ├── state.json
│   ├── pending_updates.json
│   ├── adr/
│   │   ├── 001-engine-parameter-naming.md
│   │   └── ...
│   └── api/
├── engine/
│   ├── state.json
│   ├── audit_report.md
│   └── refactoring_map.json
├── prefill/
│   ├── state.json
│   ├── design_notes.md
│   └── test_results.json
├── perf/
│   ├── state.json
│   ├── baseline_profile.json
│   ├── optimizations.md
│   └── benchmarks/
├── test/
│   ├── state.json
│   ├── test_results.json
│   └── coverage.html
└── audit/
    ├── state.json
    ├── validation_reports/
    │   ├── phase_3.1_validation.md
    │   ├── phase_3.2_validation.md
    │   ├── phase_3.3_validation.md
    │   └── phase_3.4_final_validation.md
    └── issues.json
```

---

## 📊 Resumption Protocol for New Sessions

### If Session Ends Mid-Phase

**For Any Agent Resuming Work:**

1. **Read this file** (`PHASE_3_PLAN.md`) first
   - Check "Current Progress Tracker" table at top
   - Identify current phase and status
   - Find your agent's last known state

2. **Load your agent state**
   - Read `/home/user/sglang-mamba/phase3/<your-agent>/state.json`
   - Review last completed tasks
   - Check for pending validations

3. **Check validation status**
   - If last phase validation FAILED, review issues before proceeding
   - If last phase validation PASSED, continue to next phase
   - If validation PENDING, coordinate with Audit Agent

4. **Review relevant artifacts**
   - Read ADRs for design decisions
   - Check audit reports for findings
   - Review test results for failures
   - Check documentation for context

5. **Sync with other agents**
   - Check dependencies in execution plan
   - Verify prerequisite tasks completed
   - Coordinate with Oversight Agent

6. **Continue work**
   - Start from next incomplete task
   - Update state.json after each task
   - Update PHASE_3_PLAN.md with progress
   - Run tests frequently

### Checkpoint System

**Checkpoints are created at:**
- End of each task
- End of each phase
- After each validation gate
- Before any risky operation
- On user request

**Checkpoint Contents:**
- All state.json files
- Git commit hash
- Test results
- Validation reports
- Progress snapshot

**To restore from checkpoint:**
```bash
# Checkpoints stored in phase3/checkpoints/
ls phase3/checkpoints/
# checkpoint_phase_3.1_complete_<timestamp>
# checkpoint_phase_3.2_complete_<timestamp>

# To restore (if needed):
cp -r phase3/checkpoints/checkpoint_phase_3.X/* phase3/
```

---

## 🎯 Success Metrics

### Code Quality
- [ ] All tests passing (100%)
- [ ] Code coverage ≥85%
- [ ] No critical security issues
- [ ] No architectural violations
- [ ] Code style consistent (black, isort)
- [ ] No lint warnings

### Performance
- [ ] Memory usage reduced by ≥20% (vs baseline)
- [ ] Throughput increased by ≥15% (vs baseline)
- [ ] Latency p95 maintained or improved
- [ ] Cache hit rate ≥70% (for prefill cache)
- [ ] Batch processing scales linearly

### Documentation
- [ ] SDP.md fully updated
- [ ] All ADRs completed
- [ ] API docs accurate and complete
- [ ] Migration guide available
- [ ] All code comments accurate

### Integration
- [ ] Engine parameters consistent across codebase
- [ ] ModelRunner integration working
- [ ] Scheduler properly handles Mamba batches
- [ ] No regression in existing features
- [ ] Backward compatibility maintained

### Functionality
- [ ] Prefill caching works with Mamba models
- [ ] Chunked prefill handles Mamba states correctly
- [ ] State continuity verified
- [ ] Cache invalidation works
- [ ] Edge cases handled

---

## 🚨 Risk Mitigation

### Risk 1: Parameter Refactoring Breaks Existing Code
**Likelihood:** Medium
**Impact:** High
**Mitigation:**
- Run tests after each file change
- Phased rollout with feature flags
- Keep old parameter names with deprecation warnings initially
- Immediate rollback plan
**Detection:** Test failures, integration errors
**Response:** Revert changes, fix issues, re-test

### Risk 2: Prefill Cache Causes State Inconsistencies
**Likelihood:** Medium
**Impact:** Critical
**Mitigation:**
- Extensive validation tests comparing chunked vs non-chunked
- State checksum verification
- Fallback to recomputation on mismatch
- Comprehensive logging of state transitions
**Detection:** Output divergence in tests
**Response:** Disable cache, debug state management, fix, re-enable

### Risk 3: Performance Optimizations Reduce Accuracy
**Likelihood:** Low
**Impact:** High
**Mitigation:**
- Accuracy regression tests (compare outputs with baseline)
- A/B comparison with un-optimized version
- Configurable optimization levels
- Strict numerical tolerance testing
**Detection:** Accuracy tests fail
**Response:** Roll back optimization, analyze numerical errors, fix

### Risk 4: Team Dependencies Cause Bottlenecks
**Likelihood:** Medium
**Impact:** Medium
**Mitigation:**
- Clear dependency mapping in execution plan
- Parallel work where possible
- Daily sync to catch issues early
- Flexible task reassignment
**Detection:** Tasks blocked, deadlines slipping
**Response:** Oversight agent reassigns work, adjusts timeline

### Risk 5: Validation Gate Failures Block Progress
**Likelihood:** Low
**Impact:** High
**Mitigation:**
- Thorough testing before validation
- Address issues incrementally, not in batch
- Clear issue tracking and resolution process
- User involvement throughout, not just at gates
**Detection:** Validation report shows failures
**Response:** Create focused task force, fix issues, re-validate quickly

---

## 📊 Phase 3 Completion Criteria

Phase 3 is considered **COMPLETE** when:

1. ✅ **All 4 sub-phases complete**
   - Phase 3.1 validated and passed
   - Phase 3.2 validated and passed
   - Phase 3.3 validated and passed
   - Phase 3.4 validated and passed

2. ✅ **All objectives met**
   - Engine integration fixed
   - Features implemented
   - Performance optimized
   - Tests comprehensive

3. ✅ **All validation gates passed**
   - No blocking issues
   - All checklists complete
   - User approved at each gate

4. ✅ **All success metrics achieved**
   - Code quality ≥85% coverage
   - Performance targets met
   - Documentation complete
   - Integration verified

5. ✅ **Final audit complete**
   - Security verified
   - No critical issues
   - Regression testing passed
   - User final approval

6. ✅ **Code merged to main branch**
   - All commits clean
   - No merge conflicts
   - CI/CD passes
   - Deployed (if applicable)

---

## 🚀 Launch Readiness Checklist

### Pre-Launch (Before Phase 3.4 Final Validation)
- [ ] All agent teams report completion
- [ ] All tasks marked complete in this document
- [ ] All tests passing (100%)
- [ ] All benchmarks run and meet targets
- [ ] All documentation reviewed

### Launch (Phase 3.4 Final Validation)
- [ ] Audit agent completes comprehensive review
- [ ] Security audit passed
- [ ] Regression testing passed
- [ ] Performance benchmarks acceptable
- [ ] Documentation approved

### Post-Launch
- [ ] User final approval received
- [ ] Code merged to main branch
- [ ] Rollback plan documented and tested
- [ ] Monitor for issues in first 48 hours
- [ ] Collect performance metrics in production
- [ ] Document lessons learned
- [ ] Plan Phase 4 (if needed)

---

## 📝 Version History

| Version | Date | Changes | Author |
|---------|------|---------|--------|
| 1.0 | 2026-02-16 | Initial plan created | User + Claude |
| 2.0 | 2026-02-16 | Added validation gates, resumption protocol, detailed tracking | User + Claude |

---

## 📞 Contact & Support

**For Issues During Execution:**
- Blocking technical issues → Escalate to Oversight Agent
- Design decisions needed → Create ADR, get user approval
- Validation failures → Audit Agent creates issue, track in issues.json
- New session resumption → Follow "Resumption Protocol" section above

**For Questions:**
- Read this plan document first
- Check relevant ADRs in phase3/docs/adr/
- Review agent state files
- Ask Oversight Agent

---

**End of Phase 3 Plan**
**Version:** 2.0
**Last Updated:** 2026-02-16
**Status:** Ready for Phase 3.1 Execution
**Next Action:** User approval to begin Phase 3.1
