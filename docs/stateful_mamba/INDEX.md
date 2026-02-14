# Stateful Mamba Documentation Index

Complete index of all documentation for the Stateful Mamba snapshot system.

## Quick Reference

| I want to... | Go to... |
|--------------|----------|
| **Get an overview** | [README.md](README.md) |
| **Learn how to use snapshots** | [User Guide](user_guide.md) |
| **Look up API details** | [API Reference](api_reference.md) |
| **Understand the architecture** | [Architecture](architecture.md) |
| **See complete examples** | [Examples](examples.md) |
| **Enable snapshots in my app** | [Migration Guide](migration_guide.md) |
| **Fix an issue** | [Troubleshooting](troubleshooting.md) |
| **Add docstrings to code** | [Docstring Templates](DOCSTRING_TEMPLATES.md) |
| **Get a summary** | [SUMMARY.md](SUMMARY.md) |

## Documentation Files

### 1. README.md
**Purpose**: Overview and quick start guide
**Audience**: All users
**Topics**:
- Introduction to snapshots
- Key features and benefits
- Quick start example
- When to use snapshots
- Backward compatibility guarantee
- System requirements
- Documentation roadmap

**Key Sections**:
- [Overview](README.md#overview)
- [Key Features](README.md#key-features)
- [Quick Start](README.md#quick-start)
- [When to Use Snapshots](README.md#when-to-use-snapshots)
- [Architecture Diagram](README.md#architecture)
- [Backward Compatibility](README.md#backward-compatibility)

---

### 2. user_guide.md
**Purpose**: Comprehensive usage guide
**Audience**: Application developers
**Topics**:
- Getting started with snapshots
- Basic snapshot operations
- Advanced features (COW, persistence, etc.)
- Best practices
- Common patterns
- Performance tuning

**Key Sections**:
- [Introduction](user_guide.md#introduction)
- [Getting Started](user_guide.md#getting-started)
- [Basic Snapshot Operations](user_guide.md#basic-snapshot-operations)
  - [Creating Snapshots](user_guide.md#creating-snapshots)
  - [Restoring Snapshots](user_guide.md#restoring-snapshots)
  - [Querying Snapshots](user_guide.md#querying-snapshots)
  - [Deleting Snapshots](user_guide.md#deleting-snapshots)
- [Advanced Features](user_guide.md#advanced-features)
  - [Disk Persistence](user_guide.md#disk-persistence)
  - [Copy-on-Write](user_guide.md#copy-on-write-cow)
  - [Snapshot Chaining](user_guide.md#snapshot-chaining)
- [Best Practices](user_guide.md#best-practices)
  - [Memory Management](user_guide.md#memory-management)
  - [Performance Optimization](user_guide.md#performance-optimization)
  - [Error Handling](user_guide.md#error-handling)
- [Common Patterns](user_guide.md#common-patterns)
  - [Multi-Turn Conversation](user_guide.md#pattern-1-multi-turn-conversation)
  - [Branching Narratives](user_guide.md#pattern-2-branching-narratives)
  - [Checkpoint-Based Generation](user_guide.md#pattern-3-checkpoint-based-generation)
  - [A/B Testing](user_guide.md#pattern-4-ab-testing)
  - [Session Persistence](user_guide.md#pattern-5-session-persistence)
- [Performance Tuning](user_guide.md#performance-tuning)

---

### 3. api_reference.md
**Purpose**: Complete API documentation
**Audience**: Developers implementing snapshot features
**Topics**:
- Frontend API methods
- SnapshotManager class
- Snapshot data structures
- Configuration options
- Exception types
- Utility functions

**Key Sections**:
- [Frontend API](api_reference.md#frontend-api)
  - [save_snapshot()](api_reference.md#save_snapshot)
  - [restore_snapshot()](api_reference.md#restore_snapshot)
  - [delete_snapshot()](api_reference.md#delete_snapshot)
  - [persist_snapshot()](api_reference.md#persist_snapshot)
  - [load_snapshot()](api_reference.md#load_snapshot)
- [Snapshot Manager](api_reference.md#snapshot-manager)
  - [get_snapshot()](api_reference.md#get_snapshot)
  - [list_snapshots()](api_reference.md#list_snapshots)
  - [delete_snapshots_before()](api_reference.md#delete_snapshots_before)
  - [get_total_snapshot_memory()](api_reference.md#get_total_snapshot_memory)
- [Snapshot Class](api_reference.md#snapshot-class)
  - [Attributes](api_reference.md#attributes)
  - [Properties](api_reference.md#properties)
- [Configuration](api_reference.md#configuration)
  - [SnapshotConfig](api_reference.md#snapshotconfig)
- [Exceptions](api_reference.md#exceptions)
  - [SnapshotError](api_reference.md#snapshoterrtor)
  - [SnapshotNotFoundError](api_reference.md#snapshotnotfounderror)
  - [OutOfMemoryError](api_reference.md#outofmemoryerror)
- [Utility Functions](api_reference.md#utility-functions)

---

### 4. architecture.md
**Purpose**: Technical deep-dive into system design
**Audience**: Core developers and contributors
**Topics**:
- System architecture
- Core components
- Data structures
- Memory management
- Concurrency model
- Design decisions

**Key Sections**:
- [Overview](architecture.md#overview)
- [Core Components](architecture.md#core-components)
  - [SnapshotManager](architecture.md#1-snapshotmanager)
  - [SnapshotRegistry](architecture.md#2-snapshotregistry)
  - [Snapshot Data Class](architecture.md#3-snapshot-data-class)
  - [SnapshotSerializer](architecture.md#4-snapshotserializer)
- [System Architecture](architecture.md#system-architecture)
  - [High-Level Data Flow](architecture.md#high-level-data-flow)
  - [Snapshot Save Flow](architecture.md#snapshot-save-flow)
  - [Snapshot Restore Flow](architecture.md#snapshot-restore-flow)
- [Data Structures](architecture.md#data-structures)
  - [Snapshot Memory Layout](architecture.md#snapshot-memory-layout)
  - [Radix Cache Integration](architecture.md#radix-cache-integration)
- [Memory Management](architecture.md#memory-management)
  - [Reference Counting](architecture.md#reference-counting-strategy)
  - [Copy-on-Write](architecture.md#copy-on-write-cow-optimization)
  - [Memory Pressure Handling](architecture.md#memory-pressure-handling)
- [Concurrency Model](architecture.md#concurrency-model)
  - [Thread Safety](architecture.md#thread-safety)
  - [Multi-Request Concurrency](architecture.md#multi-request-concurrency)
  - [Async Disk I/O](architecture.md#async-disk-io)
- [Integration Points](architecture.md#integration-points)
- [Design Decisions](architecture.md#design-decisions)
- [Performance Characteristics](architecture.md#performance-characteristics)

---

### 5. examples.md
**Purpose**: Real-world, runnable examples
**Audience**: Application developers
**Topics**:
- Complete application examples
- Integration patterns
- Advanced use cases

**Key Sections**:
- [Complete Examples](examples.md#complete-examples)
  - [Interactive Chatbot](examples.md#interactive-chatbot)
  - [Story Generator with Branching](examples.md#story-generator-with-branching)
  - [Document Summarization Pipeline](examples.md#document-summarization-pipeline)
  - [Code Generation with Checkpoints](examples.md#code-generation-with-checkpoints)
  - [Multi-Agent Conversation](examples.md#multi-agent-conversation)
- [Integration Examples](examples.md#integration-examples)
  - [FastAPI Server](examples.md#fastapi-server)
  - [Gradio Interface](examples.md#gradio-interface) *(to be completed)*
  - [Batch Processing](examples.md#batch-processing) *(to be completed)*
- [Advanced Patterns](examples.md#advanced-patterns)
  - [Snapshot Versioning](examples.md#snapshot-versioning) *(to be completed)*
  - [Performance Monitoring](examples.md#performance-monitoring) *(to be completed)*

---

### 6. migration_guide.md
**Purpose**: Guide for enabling snapshots in existing apps
**Audience**: Developers upgrading to snapshot-enabled SGLang
**Topics**:
- Prerequisites
- Migration steps
- Compatibility matrix
- Common scenarios
- Performance impact
- Rollback procedure

**Key Sections**:
- [Overview](migration_guide.md#overview)
- [Prerequisites](migration_guide.md#prerequisites)
  - [Version Requirements](migration_guide.md#version-requirements)
  - [Memory Requirements](migration_guide.md#memory-requirements)
- [Migration Steps](migration_guide.md#migration-steps)
  - [Step 1: Update SGLang](migration_guide.md#step-1-update-sglang)
  - [Step 2: Enable Snapshots](migration_guide.md#step-2-enable-snapshots-globally)
  - [Step 3: Update Code](migration_guide.md#step-3-update-your-code-to-use-snapshots)
  - [Step 4: Test](migration_guide.md#step-4-test-the-migration)
- [Compatibility Matrix](migration_guide.md#compatibility-matrix)
- [Common Migration Scenarios](migration_guide.md#common-migration-scenarios)
  - [Chatbot Application](migration_guide.md#scenario-1-chatbot-application)
  - [Document Processing](migration_guide.md#scenario-2-document-processing)
  - [A/B Testing](migration_guide.md#scenario-3-ab-testing)
- [Breaking Changes](migration_guide.md#breaking-changes)
- [Performance Impact](migration_guide.md#performance-impact)
- [Rollback Procedure](migration_guide.md#rollback-procedure)
- [Migration Checklist](migration_guide.md#migration-checklist)

---

### 7. troubleshooting.md
**Purpose**: Solutions to common problems
**Audience**: All users
**Topics**:
- Common issues and fixes
- Error messages
- Memory problems
- Performance issues
- Debugging tools
- FAQ

**Key Sections**:
- [Common Issues](troubleshooting.md#common-issues)
  - [SnapshotDisabledError](troubleshooting.md#issue-snapshotdisablederror)
  - [SnapshotNotFoundError](troubleshooting.md#issue-snapshotnotfounderror)
  - [OutOfMemoryError](troubleshooting.md#issue-outofmemoryerror)
  - [Snapshot Restore Changes Behavior](troubleshooting.md#issue-snapshot-restore-changes-behavior)
  - [Slow Snapshot Operations](troubleshooting.md#issue-slow-snapshot-operations)
- [Error Messages](troubleshooting.md#error-messages)
- [Memory Problems](troubleshooting.md#memory-problems)
  - [Diagnosing Memory Issues](troubleshooting.md#diagnosing-memory-issues)
  - [Memory Optimization Strategies](troubleshooting.md#memory-optimization-strategies)
- [Performance Issues](troubleshooting.md#performance-issues)
  - [Profiling Snapshot Operations](troubleshooting.md#profiling-snapshot-operations)
  - [Performance Tuning](troubleshooting.md#performance-tuning)
- [Disk Persistence Issues](troubleshooting.md#disk-persistence-issues)
- [Debugging Tools](troubleshooting.md#debugging-tools)
  - [Snapshot Inspector](troubleshooting.md#snapshot-inspector)
  - [Health Check](troubleshooting.md#health-check)
- [FAQ](troubleshooting.md#faq)

---

### 8. DOCSTRING_TEMPLATES.md
**Purpose**: Templates for code documentation
**Audience**: Core developers implementing snapshots
**Topics**:
- Module-level docstrings
- Class docstrings
- Method docstrings
- Property docstrings
- Exception docstrings
- Best practices

**Key Sections**:
- [Module-Level Docstrings](DOCSTRING_TEMPLATES.md#module-level-docstrings)
- [Class Docstrings](DOCSTRING_TEMPLATES.md#class-docstrings)
- [Method Docstrings](DOCSTRING_TEMPLATES.md#method-docstrings)
  - [save_snapshot()](DOCSTRING_TEMPLATES.md#snapshotmanagersave_snapshot)
  - [restore_snapshot()](DOCSTRING_TEMPLATES.md#snapshotmanagerrestore_snapshot)
  - [delete_snapshot()](DOCSTRING_TEMPLATES.md#snapshotmanagerdelete_snapshot)
  - [persist_snapshot()](DOCSTRING_TEMPLATES.md#snapshotmanagerpersist_snapshot)
- [Property Docstrings](DOCSTRING_TEMPLATES.md#property-docstrings)
- [Exception Docstrings](DOCSTRING_TEMPLATES.md#exception-docstrings)
- [Best Practices](DOCSTRING_TEMPLATES.md#best-practices-for-docstrings)
- [Documentation Review Checklist](DOCSTRING_TEMPLATES.md#documentation-review-checklist)

---

### 9. SUMMARY.md
**Purpose**: High-level summary and navigation
**Audience**: All users
**Topics**:
- Documentation overview
- Quick navigation
- Key concepts
- Implementation checklist

**Key Sections**:
- [Documentation Overview](SUMMARY.md#documentation-overview)
- [Documentation Structure](SUMMARY.md#documentation-structure)
- [Quick Navigation](SUMMARY.md#quick-navigation)
- [Key Concepts](SUMMARY.md#key-concepts)
- [Documentation Status](SUMMARY.md#documentation-status)
- [Implementation Status](SUMMARY.md#implementation-status)
- [Code Implementation Checklist](SUMMARY.md#code-implementation-checklist)

---

## Documentation by Audience

### For First-Time Users
1. [README.md](README.md) - Start here
2. [User Guide - Getting Started](user_guide.md#getting-started)
3. [Examples - Interactive Chatbot](examples.md#interactive-chatbot)

### For Application Developers
1. [User Guide](user_guide.md) - Complete usage guide
2. [Examples](examples.md) - Real-world patterns
3. [API Reference](api_reference.md) - API details
4. [Troubleshooting](troubleshooting.md) - When things go wrong

### For Migrating Existing Apps
1. [Migration Guide](migration_guide.md) - Step-by-step migration
2. [Compatibility Matrix](migration_guide.md#compatibility-matrix)
3. [Common Scenarios](migration_guide.md#common-migration-scenarios)
4. [Rollback Procedure](migration_guide.md#rollback-procedure)

### For Core Contributors
1. [Architecture](architecture.md) - System design
2. [DOCSTRING_TEMPLATES](DOCSTRING_TEMPLATES.md) - Code documentation
3. [Implementation Checklist](SUMMARY.md#code-implementation-checklist)
4. [API Reference](api_reference.md) - API specifications

### For Troubleshooting
1. [Troubleshooting](troubleshooting.md) - Common issues
2. [FAQ](troubleshooting.md#faq) - Frequently asked questions
3. [Debugging Tools](troubleshooting.md#debugging-tools)
4. [Performance Tuning](user_guide.md#performance-tuning)

## Documentation by Topic

### Memory Management
- [Architecture - Memory Management](architecture.md#memory-management)
- [User Guide - Memory Management](user_guide.md#memory-management)
- [Troubleshooting - Memory Problems](troubleshooting.md#memory-problems)
- [Configuration - Memory Options](api_reference.md#snapshotconfig)

### Performance
- [Architecture - Performance Characteristics](architecture.md#performance-characteristics)
- [User Guide - Performance Tuning](user_guide.md#performance-tuning)
- [Troubleshooting - Performance Issues](troubleshooting.md#performance-issues)
- [Migration Guide - Performance Impact](migration_guide.md#performance-impact)

### Disk Persistence
- [User Guide - Disk Persistence](user_guide.md#disk-persistence)
- [API Reference - persist_snapshot()](api_reference.md#persist_snapshot)
- [Architecture - Snapshot Serializer](architecture.md#4-snapshotserializer)
- [Troubleshooting - Disk Persistence Issues](troubleshooting.md#disk-persistence-issues)

### Thread Safety
- [Architecture - Concurrency Model](architecture.md#concurrency-model)
- [Architecture - Thread Safety](architecture.md#thread-safety)
- [API Reference - Thread Safety Notes](api_reference.md#snapshotmanager)

### Copy-on-Write (COW)
- [User Guide - Copy-on-Write](user_guide.md#copy-on-write-cow)
- [Architecture - COW Optimization](architecture.md#copy-on-write-cow-optimization)
- [Configuration - COW Settings](api_reference.md#snapshotconfig)

## Quick Answers

### How do I enable snapshots?
→ [Migration Guide - Step 2](migration_guide.md#step-2-enable-snapshots-globally)

### How do I save a snapshot?
→ [API Reference - save_snapshot()](api_reference.md#save_snapshot)

### How do I restore a snapshot?
→ [API Reference - restore_snapshot()](api_reference.md#restore_snapshot)

### What if I run out of memory?
→ [Troubleshooting - OutOfMemoryError](troubleshooting.md#issue-outofmemoryerror)

### How do I persist snapshots to disk?
→ [User Guide - Disk Persistence](user_guide.md#disk-persistence)

### Are snapshots compatible with transformer models?
→ [FAQ - Transformer Models](troubleshooting.md#q-can-i-use-snapshots-with-transformer-models)

### How much memory does a snapshot use?
→ [FAQ - Memory Usage](troubleshooting.md#q-how-much-memory-does-each-snapshot-use)

### Will snapshots break my existing code?
→ [README - Backward Compatibility](README.md#backward-compatibility)

## Code Examples Index

### Basic Operations
- [Save snapshot](user_guide.md#auto-generated-id)
- [Restore snapshot](user_guide.md#basic-restore)
- [Delete snapshot](user_guide.md#delete-single-snapshot)
- [Query snapshot](user_guide.md#get-snapshot-info)

### Advanced Features
- [Disk persistence](user_guide.md#saving-to-disk)
- [Copy-on-write](user_guide.md#enabling-cow)
- [Snapshot chaining](user_guide.md#snapshot-chaining)
- [Metadata filtering](user_guide.md#filter-by-metadata)

### Complete Applications
- [Interactive chatbot](examples.md#interactive-chatbot)
- [Story branching](examples.md#story-generator-with-branching)
- [Document pipeline](examples.md#document-summarization-pipeline)
- [Code generation](examples.md#code-generation-with-checkpoints)
- [Multi-agent](examples.md#multi-agent-conversation)
- [FastAPI server](examples.md#fastapi-server)

### Common Patterns
- [Multi-turn conversation](user_guide.md#pattern-1-multi-turn-conversation)
- [Branching narratives](user_guide.md#pattern-2-branching-narratives)
- [Checkpoint recovery](user_guide.md#pattern-3-checkpoint-based-generation)
- [A/B testing](user_guide.md#pattern-4-ab-testing)
- [Session persistence](user_guide.md#pattern-5-session-persistence)

## Glossary

- **Snapshot**: A saved state of a Mamba model at a specific point in the token sequence
- **SSM States**: State Space Model hidden states (conv states + SSM states)
- **Radix Cache**: Tree-based prefix cache for efficient KV/state reuse
- **COW**: Copy-on-Write, memory optimization technique
- **Reference Counting**: Memory management technique tracking object usage
- **Mamba**: State Space Model architecture for sequence modeling
- **Serialization**: Converting snapshots to disk-storable format
- **Registry**: Thread-safe storage for active snapshots
- **Memory Pool**: GPU memory allocator for SSM states

## Contributing

To contribute to this documentation:

1. Read [SUMMARY.md](SUMMARY.md) for overview
2. Check [DOCSTRING_TEMPLATES.md](DOCSTRING_TEMPLATES.md) for style
3. Follow existing structure and formatting
4. Test all code examples
5. Update this index if adding new sections
6. Submit PR with documentation changes

## Version Information

- **Documentation Version**: 1.0
- **Created**: 2026-02-14
- **Last Updated**: 2026-02-14
- **Target SGLang Version**: v0.5.0
- **Status**: Complete (awaiting implementation)

## Getting Help

- **GitHub Issues**: [Report bugs or request features](https://github.com/sgl-project/sglang/issues)
- **Discussions**: [Ask questions](https://github.com/sgl-project/sglang/discussions)
- **Slack**: [Join community](https://slack.sglang.io/)
- **Email**: sglang@lmsys.org

---

**This index is maintained as part of the Stateful Mamba documentation suite.**
