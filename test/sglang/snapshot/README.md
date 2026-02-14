# Snapshot System Tests

This directory contains tests for the Mamba state snapshot and persistence system.

## Running Tests

### Run all snapshot tests:
```bash
pytest test/sglang/snapshot/ -v
```

### Run specific test file:
```bash
pytest test/sglang/snapshot/test_mamba_snapshot.py -v
pytest test/sglang/snapshot/test_snapshot_policy.py -v
```

### Run specific test:
```bash
pytest test/sglang/snapshot/test_mamba_snapshot.py::TestMambaSnapshotManager::test_save_and_load_snapshot -v
```

## Test Coverage

### test_mamba_snapshot.py
- Metadata serialization and deserialization
- Snapshot save/load operations
- Conversation and snapshot listing
- Branch creation and management
- Disk size calculations

### test_snapshot_policy.py
- Trigger policy decision logic (EVERY_TURN, EVERY_N_TURNS, ON_TOOL_CALL, MANUAL_ONLY)
- Minimum interval enforcement
- Automatic snapshot pruning
- Branch creation and deletion
- Total size calculations

## Integration Tests

Integration tests with actual Mamba models will be added in Phase 5.

These will test:
- Full snapshot system with live SGLang server
- State continuity across server restarts
- Mamba state extraction and injection with real MambaPool
- Multi-turn conversation snapshot/restore cycles
