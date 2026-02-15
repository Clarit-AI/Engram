# Agent Framework Architecture

Comprehensive architecture documentation for the SGLang Agent Framework.

## Table of Contents

- [Overview](#overview)
- [System Architecture](#system-architecture)
- [Component Design](#component-design)
- [Data Flow](#data-flow)
- [Memory Tiers](#memory-tiers)
- [Tool System](#tool-system)
- [API Layer](#api-layer)
- [Performance Considerations](#performance-considerations)
- [Scalability](#scalability)

---

## Overview

The SGLang Agent Framework provides stateful tool-calling capabilities for Mamba-based language models. It consists of four major subsystems:

1. **Tool System** - Registry, execution, and parsing
2. **Memory System** - 3-tier conversation state management
3. **Snapshot System** - Mamba state persistence
4. **API Layer** - REST endpoints for management

---

## System Architecture

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     HTTP/REST API Layer                      │
│                  (FastAPI - http_server.py)                  │
└────────────────────────┬────────────────────────────────────┘
                         │
┌────────────────────────┴────────────────────────────────────┐
│                    Agent Framework Core                      │
│  ┌──────────────┬──────────────┬──────────────────────────┐ │
│  │ Tool System  │ Memory Tiers │  Snapshot Persistence    │ │
│  │              │              │                          │ │
│  │ - Registry   │ - Active     │  - Mamba State           │ │
│  │ - Executor   │ - Warm       │  - Serialization         │ │
│  │ - Parser     │ - Cold       │  - Disk I/O              │ │
│  └──────────────┴──────────────┴──────────────────────────┘ │
└────────────────────────┬────────────────────────────────────┘
                         │
┌────────────────────────┴────────────────────────────────────┐
│                      Scheduler Core                          │
│  (Request routing, conversation tracking, state management)  │
└────────────────────────┬────────────────────────────────────┘
                         │
┌────────────────────────┴────────────────────────────────────┐
│                    Model Execution Layer                     │
│            (Mamba model, CUDA kernels, inference)            │
└──────────────────────────────────────────────────────────────┘
```

### Component Interaction

```
Client Request
      │
      ▼
┌──────────────┐
│  HTTP API    │
└──────┬───────┘
       │
       ▼
┌──────────────┐      ┌─────────────────┐
│  Scheduler   │◄────►│ Conversation    │
└──────┬───────┘      │ Tracker         │
       │              └─────────────────┘
       ▼
┌──────────────┐      ┌─────────────────┐
│ Tool Parser  │─────►│ Tool Registry   │
└──────┬───────┘      └─────────────────┘
       │
       ▼
┌──────────────┐      ┌─────────────────┐
│ Tool Executor│─────►│ Tool            │
└──────┬───────┘      │ Implementation  │
       │              └─────────────────┘
       ▼
┌──────────────┐
│   Response   │
└──────────────┘
```

---

## Component Design

### 1. Tool System

#### Tool Registry
**Location:** `python/sglang/srt/agents/tools/registry.py`

**Responsibilities:**
- Tool registration and discovery
- Parameter validation schemas
- Tool metadata management
- Tag-based organization

**Key Classes:**
```python
class ToolRegistry:
    - register(tool: Tool)
    - get(name: str) -> Tool
    - list_tools(tag: str) -> List[Tool]
    - validate_parameters(tool_name, params)
```

#### Tool Executor
**Location:** `python/sglang/srt/agents/tools/executor.py`

**Responsibilities:**
- Async/sync tool execution
- Timeout handling
- Error recovery
- Result caching
- Execution statistics

**Execution Flow:**
```
Tool Call Request
    ↓
Parameter Validation
    ↓
Context Preparation (if needed)
    ↓
Execute Tool Function
    ↓
Error Handling / Timeout
    ↓
Result Serialization
    ↓
Return ToolExecutionResult
```

#### Tool Parser
**Location:** `python/sglang/srt/agents/tools/parser.py`

**Responsibilities:**
- Extract tool calls from model output
- Parse JSON/XML/custom formats
- Validate tool call syntax
- Handle malformed responses

**Parsing Strategy:**
```
Model Output
    ↓
Format Detection (JSON/XML/plain)
    ↓
Pattern Matching / Regex
    ↓
JSON Schema Validation
    ↓
ToolCall Objects
```

---

### 2. Memory Tier System

#### Three-Tier Architecture

```
┌─────────────────────────────────────────────┐
│            Active Tier (VRAM)               │
│  - Currently processing conversations        │
│  - Ultra-low latency (<1ms)                 │
│  - Limited capacity (GPU memory)            │
└────────────┬────────────────────────────────┘
             │ Evict (LRU)
             ▼
┌─────────────────────────────────────────────┐
│            Warm Tier (RAM)                  │
│  - Recently used conversations              │
│  - Fast access (~10ms)                      │
│  - Moderate capacity (System RAM)           │
└────────────┬────────────────────────────────┘
             │ Archive
             ▼
┌─────────────────────────────────────────────┐
│            Cold Tier (Disk)                 │
│  - Long-term storage                        │
│  - Slower access (~100ms)                   │
│  - Large capacity (SSD/HDD)                 │
└─────────────────────────────────────────────┘
```

#### Tier Manager
**Location:** `python/sglang/srt/snapshot/tier_manager.py`

**Key Features:**
- Automatic tier transitions based on access patterns
- LRU eviction from active tier
- Time-based archival to cold tier
- Async snapshot serialization
- Background cleanup cycles

**Transition Policy:**
```python
# Active → Warm
if not_accessed_for(5_minutes) or memory_pressure_high():
    transition_to_warm()

# Warm → Cold
if not_accessed_for(30_minutes):
    transition_to_cold()

# Restore to Active
if access_requested():
    restore_from_tier(conversation_id)
```

#### Conversation Tracker
**Location:** `python/sglang/srt/snapshot/conversation_tracker.py`

**Responsibilities:**
- Track conversation states
- Maintain access timestamps
- Trigger tier transitions
- Provide conversation metadata

**State Tracking:**
```python
class ConversationState:
    conversation_id: str
    tier: ConversationTier  # ACTIVE/WARM/COLD
    last_access_time: float
    access_count: int
    metadata: Dict[str, Any]
    snapshot_location: Optional[str]
```

---

### 3. Snapshot Persistence

#### Snapshot System
**Location:** `python/sglang/srt/snapshot/snapshot_*.py`

**Components:**
- **SnapshotStorage** - Disk I/O operations
- **SnapshotSerializer** - Mamba state serialization
- **SnapshotMetadata** - Conversation metadata tracking

**Mamba State Persistence:**
```
Mamba Hidden States (VRAM)
    ↓
Serialize to bytes (NumPy → bytes)
    ↓
Compress (optional - zstd)
    ↓
Write to disk (async I/O)
    ↓
Update metadata (conversation ID, turn, timestamp)
```

**File Structure:**
```
snapshots/
├── conv_123/
│   ├── turn_001.snapshot
│   ├── turn_002.snapshot
│   └── metadata.json
├── conv_456/
│   ├── turn_001.snapshot
│   └── metadata.json
└── index.json
```

---

### 4. API Layer

#### REST API Architecture
**Location:** `python/sglang/srt/agents/api/`

**Structure:**
```
api/
├── __init__.py          # Module exports
├── models.py            # Pydantic request/response models
└── handlers.py          # FastAPI route handlers
```

#### Request Flow

```
HTTP Request
    ↓
FastAPI App (http_server.py)
    ↓
Request Validation (Pydantic)
    ↓
AgentAPIHandler
    ↓
┌──────────┬──────────┬──────────┐
│ Tool     │ Memory   │ Tier     │
│ Registry │ System   │ Manager  │
└──────────┴──────────┴──────────┘
    ↓
Response Serialization (Pydantic)
    ↓
HTTP Response (JSON)
```

#### Handler Pattern

```python
class AgentAPIHandler:
    def __init__(self, scheduler):
        self.scheduler = scheduler
        self.tool_registry = scheduler.tool_registry
        self.tier_manager = scheduler.tier_manager
        # ...

    async def execute_tool(self, request: ToolCallRequest):
        # 1. Validate request
        # 2. Execute via tool executor
        # 3. Return response
        pass
```

---

## Data Flow

### Tool Execution Flow

```
1. Client sends HTTP POST /v1/agent/tools/execute
   {
     "tool_name": "calculator",
     "parameters": {"expression": "2+2"}
   }

2. FastAPI validates request (Pydantic)

3. AgentAPIHandler.execute_tool() called

4. ToolExecutor.execute() invoked
   - Validates parameters
   - Calls tool function
   - Handles errors/timeout

5. Tool function executes
   def calculator(expression: str) -> float:
       return eval(expression)  # (simplified)

6. Result wrapped in ToolExecutionResult
   {
     "tool_name": "calculator",
     "status": "success",
     "result": 4.0,
     "execution_time_ms": 1.2
   }

7. Response serialized and returned to client
```

### Memory Storage Flow

```
1. Client stores memory via API
   POST /v1/agent/memory/store
   {
     "conversation_id": "conv_123",
     "key": "user_name",
     "value": "Alice"
   }

2. ToolExecutor executes memory_store tool

3. Memory stored in conversation metadata
   conversations["conv_123"].metadata["user_name"] = "Alice"

4. ConversationTracker updates access time

5. If conversation in WARM/COLD tier:
   - Metadata persisted to disk
   - Snapshot updated

6. Success response returned
```

### Tier Transition Flow

```
1. Background cleanup cycle runs (every 60s)

2. TierManager.run_cleanup_cycle()

3. For each conversation:
   - Check last access time
   - Calculate time since access
   - Determine target tier

4. Trigger transitions:
   ACTIVE → WARM (>5 min idle)
     - Evict from GPU
     - Store in RAM
     - Update tracker

   WARM → COLD (>30 min idle)
     - Serialize snapshot
     - Write to disk
     - Free RAM
     - Update tracker

5. Update statistics and metrics

6. Log transition summary
```

---

## Memory Tiers

### Tier Characteristics

| Tier   | Storage | Latency | Capacity | Use Case                    |
|--------|---------|---------|----------|-----------------------------|
| Active | VRAM    | <1ms    | Small    | Current conversations       |
| Warm   | RAM     | ~10ms   | Medium   | Recent conversations        |
| Cold   | Disk    | ~100ms  | Large    | Long-term storage           |

### Access Patterns

**Hot Path (Active Tier):**
```python
# Direct GPU memory access
snapshot = host_pool.get(conversation_id)
# → Returns immediately from VRAM
```

**Warm Path (Warm Tier):**
```python
# Load from RAM
snapshot = warm_cache.get(conversation_id)
# → ~10ms to copy to GPU
```

**Cold Path (Cold Tier):**
```python
# Load from disk
snapshot = storage.load_snapshot(conversation_id, turn_number)
# → ~100ms to read + deserialize + copy to GPU
```

### Capacity Planning

**Active Tier Sizing:**
```
Max Active = GPU_VRAM / Snapshot_Size
Example: 24GB / 500MB = ~48 conversations
```

**Warm Tier Sizing:**
```
Max Warm = System_RAM * 0.5 / Snapshot_Size
Example: 64GB * 0.5 / 500MB = ~64 conversations
```

**Cold Tier Sizing:**
```
Max Cold = Disk_Space / Snapshot_Size
Example: 1TB / 500MB = ~2000 conversations
```

---

## Tool System

### Tool Lifecycle

```
1. Registration (Startup)
   ┌─────────────────────────────┐
   │ def my_tool(param: str):    │
   │     return result           │
   │                             │
   │ registry.register(my_tool)  │
   └─────────────────────────────┘

2. Discovery (Runtime)
   GET /v1/agent/tools
   → Returns tool metadata

3. Execution (Request)
   POST /v1/agent/tools/execute
   → Validates params
   → Calls tool function
   → Returns result

4. Statistics (Monitoring)
   GET /v1/agent/stats
   → Execution counts, timing, errors
```

### Tool Categories

**Built-in Tools:**
- `calculator` - Mathematical evaluation
- `memory_store` - Store to conversation memory
- `memory_recall` - Retrieve from memory
- `memory_search` - Search memory by keyword

**Custom Tools (Future):**
- User-defined via API
- Plugin system
- Hot-reload support

---

## API Layer

### Endpoint Organization

```
/v1/agent/
├── health                    # Health check
├── stats                     # System statistics
├── tools/
│   ├── GET                  # List tools
│   ├── {name}               # Get tool details
│   └── execute              # Execute tool
├── memory/
│   ├── store                # Store memory
│   ├── recall               # Recall memory
│   └── search               # Search memory
├── conversations/
│   ├── GET                  # List conversations
│   ├── search               # Search conversations
│   └── restore              # Restore from tier
└── tiers/
    ├── stats                # Tier statistics
    ├── transition           # Manual transition
    └── cleanup              # Trigger cleanup
```

### Authentication & Security (Future)

**Planned Features:**
- API key authentication
- Role-based access control (RBAC)
- Rate limiting per client
- Request signing
- TLS/HTTPS enforcement

---

## Performance Considerations

### Latency Targets

| Operation              | Target Latency | Notes                          |
|------------------------|----------------|--------------------------------|
| Tool execution         | <10ms          | Excluding tool function time   |
| Memory store           | <5ms           | Active tier only               |
| Memory recall          | <5ms           | Active tier, <20ms warm tier   |
| Tier transition        | <100ms         | Active↔Warm                    |
| Snapshot save          | <200ms         | Async, doesn't block requests  |
| Snapshot load          | <200ms         | From cold tier                 |
| API request            | <50ms          | Including validation           |

### Optimization Strategies

**1. Caching:**
```python
# LRU cache for frequently accessed tools
@lru_cache(maxsize=1000)
def get_tool_metadata(tool_name: str):
    return tool_registry.get(tool_name)
```

**2. Async I/O:**
```python
# Non-blocking snapshot writes
async def save_snapshot_async(snapshot, path):
    async with aiofiles.open(path, 'wb') as f:
        await f.write(snapshot.serialize())
```

**3. Batching:**
```python
# Batch tier transitions
def run_cleanup_cycle():
    transitions = collect_pending_transitions()
    batch_execute(transitions)  # Single disk I/O
```

**4. Connection Pooling:**
```python
# Reuse HTTP connections
session = aiohttp.ClientSession()
# Reused across requests
```

---

## Scalability

### Horizontal Scaling

**Multi-Node Architecture (Future):**
```
                    Load Balancer
                          │
        ┌─────────────────┼─────────────────┐
        ▼                 ▼                 ▼
    Node 1            Node 2            Node 3
    ┌──────┐          ┌──────┐          ┌──────┐
    │ API  │          │ API  │          │ API  │
    └──┬───┘          └──┬───┘          └──┬───┘
       │                 │                 │
       └─────────────────┼─────────────────┘
                         ▼
               Shared Storage Layer
               (Redis / S3 / NFS)
```

### Vertical Scaling

**GPU Scaling:**
- Multi-GPU support for parallel conversations
- GPU memory pooling
- Dynamic GPU allocation

**Memory Scaling:**
- Larger warm tier with more RAM
- SSD-based cold tier for faster I/O
- NVMe for snapshot storage

### Current Limits

| Resource               | Current Limit | Bottleneck        |
|------------------------|---------------|-------------------|
| Active conversations   | ~48           | GPU VRAM          |
| Warm conversations     | ~64           | System RAM        |
| Cold conversations     | ~2000         | Disk space        |
| API requests/sec       | ~1000         | CPU/Network       |
| Tool executions/sec    | ~500          | Tool logic        |

---

## Design Principles

### 1. Modularity
Each component is independent and can be tested/deployed separately.

### 2. Extensibility
New tools, tiers, and APIs can be added without core changes.

### 3. Performance
Optimized for low latency with caching, async I/O, and batching.

### 4. Reliability
Graceful degradation, error handling, and automatic recovery.

### 5. Observability
Comprehensive metrics, logging, and health checks.

---

## Next Steps

- [API Reference](API_REFERENCE.md)
- [Integration Guide](INTEGRATION_GUIDE.md)
- [Code Examples](EXAMPLES.md)
- [WebSocket API](WEBSOCKET_API.md) (Phase 4.5)
