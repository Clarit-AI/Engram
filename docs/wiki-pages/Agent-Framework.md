# SGLang Agent Framework

> This page consolidates documentation originally in `docs/agent_framework/` in the repo.

🤖 **Stateful Tool-Calling Framework for Mamba Models**

The SGLang Agent Framework enables stateful, tool-calling capabilities for Mamba language models with multi-tier memory management and comprehensive REST/WebSocket APIs.

## Quick Start

```bash
# Start server with agent tools enabled
python -m sglang.launch_server \
  --model-path ibm-granite/granite-4.0-h-small \
  --enable-snapshot-persistence \
  --enable-memory-tiers \
  --enable-agent-tools \
  --port 30000

# Execute a tool
curl -X POST http://localhost:30000/v1/agent/tools/execute \
  -H "Content-Type: application/json" \
  -d '{
    "tool_name": "calculator",
    "parameters": {"expression": "sqrt(16) + 2 * 3"}
  }'

# Response:
# {"tool_name": "calculator", "status": "success", "result": 10.0}
```

## Features

### ✨ Tool-Calling System
- **4 Built-in Tools**: Calculator, memory store/recall/search
- **Async Execution**: Non-blocking tool execution
- **Parameter Validation**: Schema-based validation
- **Timeout Handling**: Configurable execution timeouts
- **Execution Stats**: Track performance metrics

### 🧠 3-Tier Memory Management
- **Active Tier (VRAM)**: Ultra-fast access (<1ms), limited capacity
- **Warm Tier (RAM)**: Fast access (~10ms), moderate capacity
- **Cold Tier (Disk)**: Large capacity (~100ms), long-term storage
- **Automatic Transitions**: LRU-based eviction and archival
- **Snapshot Persistence**: Mamba state serialization to disk

### 🌐 REST + WebSocket APIs
- **13 REST Endpoints**: Tool execution, memory, conversations, tiers
- **WebSocket Streaming**: Real-time event updates
- **Pydantic Validation**: Type-safe request/response models
- **OpenAPI Compatible**: Auto-generated schemas

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    REST + WebSocket API                      │
│                  (FastAPI - Port 30000)                      │
└────────────────────────┬────────────────────────────────────┘
                         │
┌────────────────────────┴────────────────────────────────────┐
│                    Agent Framework                           │
│  ┌──────────────┬──────────────┬──────────────────────────┐ │
│  │ Tool System  │ Memory Tiers │  Snapshot Persistence    │ │
│  │ - Registry   │ - Active     │  - Mamba State           │ │
│  │ - Executor   │ - Warm       │  - Serialization         │ │
│  │ - Parser     │ - Cold       │  - Disk I/O              │ │
│  └──────────────┴──────────────┴──────────────────────────┘ │
└────────────────────────┬────────────────────────────────────┘
                         │
┌────────────────────────┴────────────────────────────────────┐
│                    Mamba Model Execution                     │
└──────────────────────────────────────────────────────────────┘
```

## API Endpoints

### Tool Management
```bash
GET  /v1/agent/tools                 # List all tools
GET  /v1/agent/tools/{name}          # Get tool details
POST /v1/agent/tools/execute         # Execute a tool
```

### Memory Management
```bash
POST /v1/agent/memory/store          # Store to memory
POST /v1/agent/memory/recall         # Recall from memory
POST /v1/agent/memory/search         # Search memory
```

### Conversation Management
```bash
GET  /v1/agent/conversations         # List conversations
POST /v1/agent/conversations/search  # Search conversations
POST /v1/agent/conversations/restore # Restore from tier
```

### Tier Management
```bash
GET  /v1/agent/tiers/stats           # Tier statistics
POST /v1/agent/tiers/transition      # Manual transition
POST /v1/agent/tiers/cleanup         # Trigger cleanup
```

### WebSocket
```bash
WS   /v1/agent/ws                    # Real-time streaming
```

## Usage Examples

### Python Client

```python
import requests

base_url = "http://localhost:30000/v1/agent"

# Execute calculator
response = requests.post(
    f"{base_url}/tools/execute",
    json={
        "tool_name": "calculator",
        "parameters": {"expression": "2 + 2"}
    }
)
print(f"Result: {response.json()['result']}")  # 4.0

# Store memory
requests.post(
    f"{base_url}/memory/store",
    json={
        "conversation_id": "conv_123",
        "key": "user_name",
        "value": "Alice",
        "category": "profile"
    }
)

# Recall memory
response = requests.post(
    f"{base_url}/memory/recall",
    json={
        "conversation_id": "conv_123",
        "category": "profile"
    }
)
print(f"Memory: {response.json()['data']}")
```

### JavaScript Client

```javascript
const axios = require('axios');

const baseURL = 'http://localhost:30000/v1/agent';

// Execute calculator
const result = await axios.post(`${baseURL}/tools/execute`, {
  tool_name: 'calculator',
  parameters: { expression: '2 + 2' }
});
console.log('Result:', result.data.result);  // 4
```

### WebSocket Streaming

```python
import asyncio
import websockets
import json

async def stream_tool():
    uri = "ws://localhost:30000/v1/agent/ws"

    async with websockets.connect(uri) as ws:
        await ws.send(json.dumps({
            "command": "execute_tool",
            "tool_name": "calculator",
            "parameters": {"expression": "2 + 2"}
        }))

        while True:
            event = json.loads(await ws.recv())
            if event['type'] == 'tool_result':
                print(f"Result: {event['data']['result']}")
                break

asyncio.run(stream_tool())
```

## Configuration

### Command-Line Options

```bash
# Snapshot Persistence
--enable-snapshot-persistence       # Enable Mamba state persistence
--snapshot-dir /path/to/snapshots # Snapshot directory (default: ./snapshots)

# Memory Tiers
--enable-memory-tiers              # Enable 3-tier memory management
--tier-active-timeout 300          # Active→Warm timeout (seconds)
--tier-warm-timeout 1800          # Warm→Cold timeout (seconds)
--tier-cleanup-interval 60         # Cleanup cycle interval (seconds)

# Agent Tools
--enable-agent-tools               # Enable tool-calling framework
```

### Example Configurations

**Minimal (agent tools only):**
```bash
python -m sglang.launch_server \
  --model-path ibm-granite/granite-4.0-h-small \
  --enable-agent-tools \
  --port 30000
```

**Full (all features):**
```bash
python -m sglang.launch_server \
  --model-path ibm-granite/granite-4.0-h-small \
  --enable-snapshot-persistence \
  --enable-memory-tiers \
  --enable-agent-tools \
  --snapshot-dir ./snapshots \
  --tier-active-timeout 300 \
  --tier-warm-timeout 1800 \
  --tier-cleanup-interval 60 \
  --port 30000
```

**Production:**
```bash
python -m sglang.launch_server \
  --model-path ibm-granite/granite-4.0-h-small \
  --enable-snapshot-persistence \
  --enable-memory-tiers \
  --enable-agent-tools \
  --snapshot-dir /mnt/fast-ssd/snapshots \
  --mem-fraction-static 0.8 \
  --max-running-requests 1024 \
  --host 0.0.0.0 \
  --port 30000
```

## Built-in Tools

### 1. Calculator
Evaluate mathematical expressions.

```python
{
  "tool_name": "calculator",
  "parameters": {"expression": "sqrt(16) + 2 * 3"}
}
# Returns: 10.0
```

### 2. Memory Store
Store information to conversation memory.

```python
{
  "tool_name": "memory_store",
  "parameters": {
    "key": "user_name",
    "value": "Alice",
    "category": "profile"
  },
  "conversation_id": "conv_123"
}
```

### 3. Memory Recall
Retrieve information from memory.

```python
{
  "tool_name": "memory_recall",
  "parameters": {"category": "profile"},
  "conversation_id": "conv_123"
}
# Returns: {"user_name": "Alice"}
```

### 4. Memory Search
Search memory by keyword.

```python
{
  "tool_name": "memory_search",
  "parameters": {"query": "user"},
  "conversation_id": "conv_123"
}
# Returns: [{"key": "user_name", "value": "Alice", "relevance_score": 0.95}]
```

## Performance

| Operation              | Latency  | Notes                          |
|------------------------|----------|--------------------------------|
| Tool execution         | <10ms    | Excluding tool function time   |
| Memory store/recall    | <5ms     | Active tier                    |
| Tier transition (A→W)  | <100ms   | Active to Warm                 |
| Snapshot save          | <200ms   | Async, non-blocking            |
| Snapshot load          | <200ms   | From cold tier                 |
| API request            | <50ms    | Including validation           |
| WebSocket event        | <5ms     | Per event                      |

## Memory Tiers

| Tier   | Storage | Latency | Capacity | Use Case                    |
|--------|---------|---------|----------|-----------------------------|
| Active | VRAM    | <1ms    | Small    | Current conversations       |
| Warm   | RAM     | ~10ms   | Medium   | Recent conversations        |
| Cold   | Disk    | ~100ms  | Large    | Long-term storage           |

## Backward Compatibility

✅ **100% backward compatible** - All features are opt-in via command-line flags.

Without flags, server behavior is unchanged:
```bash
python -m sglang.launch_server --model-path <model>
# → No agent features, existing behavior
```

## Dependencies

**No new external dependencies!** All features use existing dependencies:
- FastAPI (already in requirements)
- Pydantic (already in requirements)
- asyncio (Python stdlib)

## Related Documentation

- [Stateful Mamba Guide](./Stateful-Mamba-Guide.md) - Snapshot persistence system
- [Upstream Sync](./Upstream-Sync-Q1-2026.md) - Q1 2026 upstream synchronization history
- [GitHub Repo](https://github.com/clarit-ai/engram)
