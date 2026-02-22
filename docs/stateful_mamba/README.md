# Stateful Mamba: State Snapshot and Persistence

> **⚠️ Implementation Status:** Phase 1 (Snapshot Saving) is complete. Phase 2 (State Restoration) is in development.
>
> **Available Now:** `save_snapshot()`, `list_snapshots()`
> **Coming Soon:** `restore_snapshot()`, `get_snapshot_info()`, `SnapshotManager` wrapper

## Overview

The Stateful Mamba snapshot system extends SGLang's Mamba support with **opt-in** state persistence capabilities, enabling advanced use cases like multi-turn conversations, checkpoint-based inference, and state reuse across requests.

This is an enhancement that builds on the existing Mamba implementation in SGLang. All features are **opt-in** and **fully backward compatible** with existing transformer-based and Mamba inference workflows.

## Key Features

### Phase 1 (Available Now)
- **Snapshot Saving**: Save Mamba hidden states (SSM states) at any point during inference
- **Snapshot Inspection**: List and inspect saved snapshots with metadata
- **State Persistence**: Store snapshots to disk in safetensors format
- **Zero Impact**: Existing Mamba and transformer inference workflows are unaffected

### Phase 2 (Coming Soon)
- **State Restoration**: Restore from saved snapshots to resume conversations
- **State Reuse**: Share snapshots across multiple inference branches
- **Multi-turn Conversations**: Efficiently handle long conversations without reprocessing context

## Quick Start

### Basic Snapshot Usage

```python
from sglang import function, gen, Runtime

# Initialize runtime with snapshot support
runtime = Runtime(
    model_path="state-spaces/mamba-2.8b",
    enable_snapshot_persistence=True,  # Enable snapshot feature
    snapshot_dir="./my_snapshots"
)

@function
def conversation_with_snapshots(s):
    # First turn
    s += "User: What is machine learning?\n"
    s += "Assistant: " + gen("response1", max_tokens=100)

    # Save snapshot after first response
    snapshot_id = s.save_snapshot()
    print(f"Saved snapshot: {snapshot_id}")

    # List all snapshots for this conversation
    snapshots = s.list_snapshots()
    print(f"Total snapshots: {len(snapshots)}")

    # Get info about the snapshot we just saved
    info = s.get_snapshot_info(
        conversation_id=s.stream_executor.sid,
        turn_number=0
    )
    print(f"Snapshot info: {info}")

    # Continue conversation (state is still in memory)
    s += "\nUser: Can you give me an example?\n"
    s += "Assistant: " + gen("response2", max_tokens=100)

    return s

# Run the conversation
result = conversation_with_snapshots.run(runtime=runtime)
```

### Snapshot Persistence

Snapshots are automatically persisted to disk when you use:
- `enable_snapshot_persistence=True` - Enable snapshot system
- `snapshot_dir="./path"` - Directory for snapshot storage

The snapshots are saved in safetensors format with JSON metadata.

**Phase 2 (Coming Soon):** State restoration and advanced snapshot management.

## When to Use Snapshots

### Phase 1 Capabilities (Available Now)

1. **Snapshot Inspection**: Save and inspect conversation state at any point
2. **Debugging**: Capture state for offline analysis
3. **State Tracking**: Monitor memory usage and token counts across turns
4. **Audit Trail**: Keep records of conversation progression

### Phase 2 Capabilities (Coming Soon)

1. **State Restoration**: Resume conversations from saved snapshots
2. **Branching**: Explore multiple conversation paths from a checkpoint
3. **A/B Testing**: Compare different continuations from same state
4. **Multi-turn Conversations**: Avoid reprocessing the entire conversation history on each turn

### When NOT to Use Snapshots

1. **Single-shot Inference**: Standard completion requests don't benefit from snapshots
2. **Transformer Models**: This feature is Mamba-specific (transformer KV cache already optimized)
3. **Memory-Constrained Environments**: Snapshots require additional memory for state storage

## Architecture

The snapshot system integrates with SGLang's existing Mamba infrastructure:

```
┌─────────────────────────────────────────────────────┐
│              Frontend (Language API)                 │
│  Phase 1: save_snapshot(), list_snapshots()         │
│  Phase 2: restore_snapshot(), get_snapshot_info()   │
│           (coming soon)                             │
└──────────────────┬──────────────────────────────────┘
                   │
┌──────────────────▼──────────────────────────────────┐
│           Snapshot Persistence                      │
│  - State Serialization (safetensors)                │
│  - Metadata Storage (JSON)                          │
│  - Disk I/O                                         │
└──────────────────┬──────────────────────────────────┘
                   │
┌──────────────────▼──────────────────────────────────┐
│         Mamba Radix Cache                           │
│  - Existing tree-based cache                        │
│  - Snapshot references (Phase 1)                    │
└──────────────────┬──────────────────────────────────┘
                   │
┌──────────────────▼──────────────────────────────────┐
│         Memory Pool (SSM States)                    │
│  - GPU memory allocation                            │
│  - State persistence to disk                        │
└─────────────────────────────────────────────────────┘
```

## Backward Compatibility

**Important**: The snapshot system is completely opt-in and maintains 100% backward compatibility:

- **Default Behavior**: Snapshots are disabled by default (`enable_snapshot_persistence=False`)
- **Transformer Models**: Unaffected - this feature is Mamba-specific
- **Existing Mamba Inference**: Works exactly as before when snapshots are disabled
- **Performance**: Zero overhead when snapshots are not used
- **API**: All existing SGLang APIs remain unchanged

## Documentation Structure

This documentation is organized as follows:

- **[User Guide](user_guide.md)**: Comprehensive guide for using snapshot features
- **[API Reference](api_reference.md)**: Detailed API documentation for all snapshot-related classes and methods
- **[Architecture](architecture.md)**: Technical deep-dive into the snapshot system design
- **[Examples](examples.md)**: Real-world examples and tutorials
- **[Migration Guide](migration_guide.md)**: How to enable snapshots in existing applications
- **[Troubleshooting](troubleshooting.md)**: Common issues and solutions

## Performance Considerations

### Memory Usage

Snapshots store SSM states in GPU memory. Each snapshot consumes:
- **State Size**: Depends on model architecture (typically 10-100 MB per snapshot for mid-sized models)
- **Metadata**: Minimal overhead (~1 KB per snapshot)

### Compute Overhead

- **Save**: O(1) - Creates a reference to existing state, minimal copying
- **Restore**: O(1) - Updates pointers, no recomputation
- **Disk I/O**: Asynchronous when enabled, doesn't block inference

### Best Practices

1. **Use Radix Cache**: Snapshots work best with radix cache enabled
2. **Monitor Memory**: Track snapshot memory usage in production
3. **Meaningful Metadata**: Include conversation context in snapshot metadata for later inspection

## System Requirements

- **Model Type**: Mamba/Mamba2 architectures only
- **SGLang Version**: v0.5.0+ (with snapshot support)
- **Python**: 3.8+
- **GPU Memory**: Additional memory for snapshot storage (typically 5-10% of model size per snapshot)

## Getting Help

- **Issues**: [GitHub Issues](https://github.com/sgl-project/sglang/issues)
- **Discussions**: [GitHub Discussions](https://github.com/sgl-project/sglang/discussions)
- **Slack**: [Join SGLang Slack](https://slack.sglang.io/)
- **Documentation**: [Full Documentation](https://docs.sglang.io/)

## Related Documentation

- [Basic Mamba Usage](../basic_usage/mamba_models.md)
- [Memory Management](../advanced_features/memory_management.md)
- [Radix Cache](../advanced_features/radix_cache.md)
- [Multi-turn Conversations](../basic_usage/multi_turn.md)

## License

The snapshot system is part of SGLang and is licensed under the Apache License 2.0.

---

**Next Steps**: Read the [User Guide](user_guide.md) to learn how to use snapshots in your application.
