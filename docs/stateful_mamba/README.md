# Stateful Mamba: State Snapshot and Persistence

## Overview

The Stateful Mamba snapshot system extends SGLang's Mamba support with **opt-in** state persistence capabilities, enabling advanced use cases like multi-turn conversations, checkpoint-based inference, and state reuse across requests.

This is a **Phase 2** enhancement that builds on the existing Mamba implementation in SGLang. All features are **opt-in** and **fully backward compatible** with existing transformer-based and Mamba inference workflows.

## Key Features

- **Snapshot API**: Save and restore Mamba hidden states (SSM states) at any point during inference
- **Multi-turn Conversations**: Efficiently handle long conversations without reprocessing context
- **State Persistence**: Store and load states from disk for cross-session resumption
- **State Reuse**: Share snapshots across multiple inference branches
- **Zero Impact**: Existing Mamba and transformer inference workflows are unaffected

## Quick Start

### Basic Snapshot Usage

```python
from sglang import Engine
from sglang.lang import function, gen

# Initialize engine with snapshot support (opt-in)
engine = Engine(
    model_path="state-spaces/mamba-2.8b",
    enable_mamba_snapshots=True  # Enable snapshot feature
)

@function
def conversation_with_snapshots(s):
    # First turn
    s += "User: What is machine learning?\n"
    s += "Assistant: " + gen("response1", max_tokens=100)

    # Save state after first response
    snapshot_id = s.save_snapshot()

    # Second turn - builds on saved state
    s += "\nUser: Can you give me an example?\n"
    s += "Assistant: " + gen("response2", max_tokens=100)

    return s

# Run the conversation
result = conversation_with_snapshots.run(engine=engine)
```

### Loading and Restoring States

```python
from sglang.snapshot import SnapshotManager

# Initialize snapshot manager
manager = SnapshotManager(engine)

# Save snapshot to disk
snapshot_id = manager.save("conversation_state_1", metadata={"turn": 1})

# Later, restore from disk
manager.load(snapshot_id)
state = manager.restore(snapshot_id)
```

## When to Use Snapshots

### Ideal Use Cases

1. **Multi-turn Conversations**: Avoid reprocessing the entire conversation history on each turn
2. **Branching Scenarios**: Explore multiple conversation paths from a single checkpoint
3. **Interrupted Inference**: Resume generation from a saved state across sessions
4. **A/B Testing**: Compare different continuations from the same initial state
5. **Long Context Efficiency**: Save intermediate states in long documents

### When NOT to Use Snapshots

1. **Single-shot Inference**: Standard completion requests don't benefit from snapshots
2. **Transformer Models**: This feature is Mamba-specific (transformer KV cache already optimized)
3. **Memory-Constrained Environments**: Snapshots require additional memory for state storage

## Architecture

The snapshot system integrates with SGLang's existing Mamba infrastructure:

```
┌─────────────────────────────────────────────────────┐
│              Frontend (Language API)                 │
│  - save_snapshot() / restore_snapshot()             │
└──────────────────┬──────────────────────────────────┘
                   │
┌──────────────────▼──────────────────────────────────┐
│           Snapshot Manager                          │
│  - Snapshot Registry                                │
│  - State Serialization                              │
│  - Disk I/O (Optional)                              │
└──────────────────┬──────────────────────────────────┘
                   │
┌──────────────────▼──────────────────────────────────┐
│         Mamba Radix Cache                           │
│  - Existing tree-based cache                        │
│  - Modified to support snapshot references          │
└──────────────────┬──────────────────────────────────┘
                   │
┌──────────────────▼──────────────────────────────────┐
│         Memory Pool (SSM States)                    │
│  - GPU memory allocation                            │
│  - Reference counting                               │
└─────────────────────────────────────────────────────┘
```

## Backward Compatibility

**Important**: The snapshot system is completely opt-in and maintains 100% backward compatibility:

- **Default Behavior**: Snapshots are disabled by default (`enable_mamba_snapshots=False`)
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

1. **Clean Up Unused Snapshots**: Call `delete_snapshot()` to free memory
2. **Use Radix Cache**: Snapshots work best with radix cache enabled
3. **Monitor Memory**: Track snapshot memory usage in production
4. **Batch Operations**: Save multiple snapshots in a single transaction when possible

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
