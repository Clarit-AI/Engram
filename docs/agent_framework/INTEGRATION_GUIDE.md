# Agent Framework Integration Guide

Step-by-step guide for integrating the SGLang Agent Framework into your applications.

## Table of Contents

- [Quick Start](#quick-start)
- [Server Setup](#server-setup)
- [Python Client](#python-client)
- [JavaScript Client](#javascript-client)
- [Building Custom Tools](#building-custom-tools)
- [Memory Management](#memory-management)
- [Production Deployment](#production-deployment)
- [Troubleshooting](#troubleshooting)

---

## Quick Start

### 1. Start the Server

```bash
python -m sglang.launch_server \
  --model-path ibm-granite/granite-4.0-h-small \
  --enable-snapshot-persistence \
  --enable-memory-tiers \
  --enable-agent-tools \
  --port 30000 \
  --host 0.0.0.0
```

### 2. Verify Server is Running

```bash
# Health check
curl http://localhost:30000/v1/agent/health

# Expected response:
# {"status": "healthy", "agent_tools_enabled": true, ...}
```

### 3. Execute Your First Tool

```bash
curl -X POST http://localhost:30000/v1/agent/tools/execute \
  -H "Content-Type: application/json" \
  -d '{
    "tool_name": "calculator",
    "parameters": {"expression": "2 + 2"}
  }'

# Expected response:
# {"tool_name": "calculator", "status": "success", "result": 4.0, ...}
```

---

## Server Setup

### Configuration Options

#### Basic Configuration

```bash
# Minimal setup (no agent tools)
python -m sglang.launch_server \
  --model-path ibm-granite/granite-4.0-h-small \
  --port 30000

# With agent tools only
python -m sglang.launch_server \
  --model-path ibm-granite/granite-4.0-h-small \
  --enable-agent-tools \
  --port 30000

# Full agent framework (tools + memory + persistence)
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

#### Advanced Configuration

```bash
# Multi-GPU deployment
python -m sglang.launch_server \
  --model-path ibm-granite/granite-4.0-h-small \
  --enable-agent-tools \
  --enable-memory-tiers \
  --tp-size 2 \
  --port 30000

# Production deployment
python -m sglang.launch_server \
  --model-path ibm-granite/granite-4.0-h-small \
  --enable-agent-tools \
  --enable-memory-tiers \
  --enable-snapshot-persistence \
  --snapshot-dir /mnt/fast-ssd/snapshots \
  --mem-fraction-static 0.8 \
  --max-running-requests 1024 \
  --host 0.0.0.0 \
  --port 30000
```

### Environment Variables

```bash
# Set log level
export SGLANG_LOG_LEVEL=DEBUG

# Disable OpenAPI docs
export DISABLE_OPENAPI_DOC=1

# Custom snapshot directory
export SGLANG_SNAPSHOT_DIR=/data/snapshots

# Launch server
python -m sglang.launch_server --enable-agent-tools ...
```

---

## Python Client

### Installation

```bash
pip install requests aiohttp
```

### Synchronous Client

```python
import requests
from typing import Any, Dict, Optional


class SGLangAgentClient:
    """Python client for SGLang Agent Framework."""

    def __init__(self, base_url: str = "http://localhost:30000"):
        self.base_url = base_url.rstrip("/")
        self.agent_url = f"{self.base_url}/v1/agent"

    def health(self) -> Dict[str, Any]:
        """Check server health."""
        response = requests.get(f"{self.agent_url}/health")
        response.raise_for_status()
        return response.json()

    def list_tools(self, tag: Optional[str] = None) -> Dict[str, Any]:
        """List available tools."""
        params = {"tag": tag} if tag else {}
        response = requests.get(f"{self.agent_url}/tools", params=params)
        response.raise_for_status()
        return response.json()

    def execute_tool(
        self,
        tool_name: str,
        parameters: Dict[str, Any],
        conversation_id: Optional[str] = None,
        timeout: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Execute a tool."""
        data = {
            "tool_name": tool_name,
            "parameters": parameters,
        }
        if conversation_id:
            data["conversation_id"] = conversation_id
        if timeout:
            data["timeout"] = timeout

        response = requests.post(
            f"{self.agent_url}/tools/execute",
            json=data,
        )
        response.raise_for_status()
        return response.json()

    def store_memory(
        self,
        conversation_id: str,
        key: str,
        value: str,
        category: str = "general",
    ) -> Dict[str, Any]:
        """Store to conversation memory."""
        data = {
            "conversation_id": conversation_id,
            "key": key,
            "value": value,
            "category": category,
        }
        response = requests.post(
            f"{self.agent_url}/memory/store",
            json=data,
        )
        response.raise_for_status()
        return response.json()

    def recall_memory(
        self,
        conversation_id: str,
        key: Optional[str] = None,
        category: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Recall from conversation memory."""
        data = {"conversation_id": conversation_id}
        if key:
            data["key"] = key
        if category:
            data["category"] = category

        response = requests.post(
            f"{self.agent_url}/memory/recall",
            json=data,
        )
        response.raise_for_status()
        return response.json()

    def list_conversations(self, tier: Optional[str] = None) -> Dict[str, Any]:
        """List conversations."""
        params = {"tier": tier} if tier else {}
        response = requests.get(
            f"{self.agent_url}/conversations",
            params=params,
        )
        response.raise_for_status()
        return response.json()

    def get_tier_stats(self) -> Dict[str, Any]:
        """Get tier statistics."""
        response = requests.get(f"{self.agent_url}/tiers/stats")
        response.raise_for_status()
        return response.json()


# Example usage
if __name__ == "__main__":
    client = SGLangAgentClient("http://localhost:30000")

    # Check health
    health = client.health()
    print(f"Server health: {health['status']}")

    # List tools
    tools = client.list_tools()
    print(f"Available tools: {[t['name'] for t in tools['tools']]}")

    # Execute calculator
    result = client.execute_tool(
        "calculator",
        {"expression": "sqrt(16) + 2 * 3"}
    )
    print(f"Calculator result: {result['result']}")

    # Store and recall memory
    client.store_memory(
        "conv_123",
        "user_name",
        "Alice",
        "user_info"
    )

    memory = client.recall_memory("conv_123", category="user_info")
    print(f"Recalled memory: {memory['data']}")
```

### Async Client

```python
import aiohttp
import asyncio
from typing import Any, Dict, Optional


class AsyncSGLangAgentClient:
    """Async Python client for SGLang Agent Framework."""

    def __init__(self, base_url: str = "http://localhost:30000"):
        self.base_url = base_url.rstrip("/")
        self.agent_url = f"{self.base_url}/v1/agent"
        self.session: Optional[aiohttp.ClientSession] = None

    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()

    async def execute_tool(
        self,
        tool_name: str,
        parameters: Dict[str, Any],
        conversation_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Execute a tool asynchronously."""
        data = {
            "tool_name": tool_name,
            "parameters": parameters,
        }
        if conversation_id:
            data["conversation_id"] = conversation_id

        async with self.session.post(
            f"{self.agent_url}/tools/execute",
            json=data,
        ) as response:
            response.raise_for_status()
            return await response.json()

    async def batch_execute_tools(
        self,
        tool_calls: list[tuple[str, Dict[str, Any]]],
    ) -> list[Dict[str, Any]]:
        """Execute multiple tools in parallel."""
        tasks = [
            self.execute_tool(name, params)
            for name, params in tool_calls
        ]
        return await asyncio.gather(*tasks)


# Example usage
async def main():
    async with AsyncSGLangAgentClient() as client:
        # Execute multiple tools in parallel
        results = await client.batch_execute_tools([
            ("calculator", {"expression": "2 + 2"}),
            ("calculator", {"expression": "sqrt(16)"}),
            ("calculator", {"expression": "10 * 5"}),
        ])

        for result in results:
            print(f"{result['tool_name']}: {result['result']}")


if __name__ == "__main__":
    asyncio.run(main())
```

---

## JavaScript Client

### Installation

```bash
npm install axios
```

### Node.js Client

```javascript
const axios = require('axios');

class SGLangAgentClient {
  constructor(baseURL = 'http://localhost:30000') {
    this.client = axios.create({
      baseURL: `${baseURL}/v1/agent`,
      headers: { 'Content-Type': 'application/json' },
    });
  }

  async health() {
    const response = await this.client.get('/health');
    return response.data;
  }

  async listTools(tag = null) {
    const params = tag ? { tag } : {};
    const response = await this.client.get('/tools', { params });
    return response.data;
  }

  async executeTool(toolName, parameters, conversationId = null, timeout = null) {
    const data = {
      tool_name: toolName,
      parameters,
    };
    if (conversationId) data.conversation_id = conversationId;
    if (timeout) data.timeout = timeout;

    const response = await this.client.post('/tools/execute', data);
    return response.data;
  }

  async storeMemory(conversationId, key, value, category = 'general') {
    const response = await this.client.post('/memory/store', {
      conversation_id: conversationId,
      key,
      value,
      category,
    });
    return response.data;
  }

  async recallMemory(conversationId, key = null, category = null) {
    const data = { conversation_id: conversationId };
    if (key) data.key = key;
    if (category) data.category = category;

    const response = await this.client.post('/memory/recall', data);
    return response.data;
  }

  async getTierStats() {
    const response = await this.client.get('/tiers/stats');
    return response.data;
  }
}

// Example usage
(async () => {
  const client = new SGLangAgentClient('http://localhost:30000');

  // Check health
  const health = await client.health();
  console.log('Server health:', health.status);

  // Execute calculator
  const result = await client.executeTool('calculator', {
    expression: '2 + 2',
  });
  console.log('Calculator result:', result.result);

  // Store memory
  await client.storeMemory('conv_123', 'user_name', 'Alice', 'user_info');

  // Recall memory
  const memory = await client.recallMemory('conv_123', null, 'user_info');
  console.log('Recalled memory:', memory.data);
})();
```

### Browser Client

```html
<!DOCTYPE html>
<html>
<head>
  <title>SGLang Agent Framework</title>
</head>
<body>
  <h1>Agent Tool Executor</h1>

  <input type="text" id="expression" placeholder="Enter expression">
  <button onclick="calculate()">Calculate</button>

  <div id="result"></div>

  <script>
    const API_BASE = 'http://localhost:30000/v1/agent';

    async function calculate() {
      const expression = document.getElementById('expression').value;

      try {
        const response = await fetch(`${API_BASE}/tools/execute`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            tool_name: 'calculator',
            parameters: { expression }
          })
        });

        const data = await response.json();

        if (data.status === 'success') {
          document.getElementById('result').innerText =
            `Result: ${data.result}`;
        } else {
          document.getElementById('result').innerText =
            `Error: ${data.error}`;
        }
      } catch (error) {
        document.getElementById('result').innerText =
          `Error: ${error.message}`;
      }
    }
  </script>
</body>
</html>
```

---

## Building Custom Tools

### Tool Registration (Future)

```python
from sglang.srt.agents.tools import Tool, ToolParameter, ToolRegistry

# Define custom tool
def weather_lookup(city: str, units: str = "celsius") -> dict:
    """
    Look up weather for a city.

    Args:
        city: City name
        units: Temperature units (celsius/fahrenheit)

    Returns:
        Weather data dictionary
    """
    # Implementation here
    return {
        "city": city,
        "temperature": 22,
        "units": units,
        "conditions": "sunny"
    }

# Create tool metadata
weather_tool = Tool(
    name="weather_lookup",
    description="Look up current weather for a city",
    function=weather_lookup,
    parameters=[
        ToolParameter(
            name="city",
            type="string",
            required=True,
            description="City name"
        ),
        ToolParameter(
            name="units",
            type="string",
            required=False,
            description="celsius or fahrenheit",
            default="celsius"
        ),
    ],
    metadata={
        "tags": ["weather", "utility"],
        "version": "1.0.0"
    }
)

# Register tool
registry = ToolRegistry.get_instance()
registry.register(weather_tool)
```

---

## Memory Management

### Storing Conversation State

```python
# Store user preferences
client.store_memory(
    conversation_id="conv_123",
    key="language",
    value="English",
    category="preferences"
)

client.store_memory(
    conversation_id="conv_123",
    key="theme",
    value="dark",
    category="preferences"
)

# Store user profile
client.store_memory(
    conversation_id="conv_123",
    key="name",
    value="Alice",
    category="profile"
)

client.store_memory(
    conversation_id="conv_123",
    key="age",
    value="25",
    category="profile"
)
```

### Retrieving Memory

```python
# Get all memory for conversation
all_memory = client.recall_memory("conv_123")
print(all_memory["data"])
# {"language": "English", "theme": "dark", "name": "Alice", "age": "25"}

# Get specific category
preferences = client.recall_memory("conv_123", category="preferences")
print(preferences["data"])
# {"language": "English", "theme": "dark"}

# Get specific key
name = client.recall_memory("conv_123", key="name")
print(name["data"])
# {"name": "Alice"}
```

### Searching Memory

```python
# Search by keyword
results = client.search_memory(
    conversation_id="conv_123",
    query="language"
)

print(results["results"])
# [{"key": "language", "value": "English", "relevance_score": 0.95}]
```

---

## Production Deployment

### Docker Deployment

```dockerfile
# Dockerfile
FROM nvcr.io/nvidia/pytorch:24.01-py3

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt

# Install SGLang
RUN pip install sglang

# Copy configuration
COPY server_config.sh .

# Expose port
EXPOSE 30000

# Run server
CMD ["bash", "server_config.sh"]
```

```bash
# server_config.sh
#!/bin/bash

python -m sglang.launch_server \
  --model-path ${MODEL_PATH:-ibm-granite/granite-4.0-h-small} \
  --enable-snapshot-persistence \
  --enable-memory-tiers \
  --enable-agent-tools \
  --snapshot-dir /data/snapshots \
  --host 0.0.0.0 \
  --port 30000
```

```bash
# Build and run
docker build -t sglang-agent .

docker run --gpus all \
  -p 30000:30000 \
  -v /data/snapshots:/data/snapshots \
  -e MODEL_PATH=ibm-granite/granite-4.0-h-small \
  sglang-agent
```

### Kubernetes Deployment

```yaml
# deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: sglang-agent
spec:
  replicas: 3
  selector:
    matchLabels:
      app: sglang-agent
  template:
    metadata:
      labels:
        app: sglang-agent
    spec:
      containers:
      - name: sglang
        image: sglang-agent:latest
        ports:
        - containerPort: 30000
        resources:
          limits:
            nvidia.com/gpu: 1
        env:
        - name: MODEL_PATH
          value: "ibm-granite/granite-4.0-h-small"
        volumeMounts:
        - name: snapshots
          mountPath: /data/snapshots
      volumes:
      - name: snapshots
        persistentVolumeClaim:
          claimName: sglang-snapshots
---
apiVersion: v1
kind: Service
metadata:
  name: sglang-agent-service
spec:
  selector:
    app: sglang-agent
  ports:
  - protocol: TCP
    port: 80
    targetPort: 30000
  type: LoadBalancer
```

### Monitoring

```python
import time
from prometheus_client import Counter, Histogram, Gauge

# Metrics
tool_executions = Counter(
    'sglang_tool_executions_total',
    'Total tool executions',
    ['tool_name', 'status']
)

tool_duration = Histogram(
    'sglang_tool_duration_seconds',
    'Tool execution duration',
    ['tool_name']
)

active_conversations = Gauge(
    'sglang_active_conversations',
    'Number of active conversations'
)

# Instrumentation
def execute_tool_with_metrics(client, tool_name, parameters):
    start_time = time.time()

    try:
        result = client.execute_tool(tool_name, parameters)
        status = result['status']
        tool_executions.labels(tool_name=tool_name, status=status).inc()
        return result
    finally:
        duration = time.time() - start_time
        tool_duration.labels(tool_name=tool_name).observe(duration)
```

---

## Troubleshooting

### Common Issues

#### 1. Agent tools not available

**Error:**
```json
{
  "error": "Agent tools not enabled",
  "status_code": 503
}
```

**Solution:**
```bash
# Start server with --enable-agent-tools flag
python -m sglang.launch_server \
  --enable-agent-tools \
  ...
```

#### 2. Tool not found

**Error:**
```json
{
  "error": "Tool 'unknown_tool' not found",
  "status_code": 404
}
```

**Solution:**
```python
# List available tools first
tools = client.list_tools()
print([t['name'] for t in tools['tools']])
```

#### 3. Memory tier not available

**Error:**
```json
{
  "error": "Tier manager not available",
  "status_code": 503
}
```

**Solution:**
```bash
# Enable memory tiers
python -m sglang.launch_server \
  --enable-memory-tiers \
  ...
```

#### 4. Conversation not found

**Error:**
```json
{
  "error": "Conversation 'conv_123' not found",
  "status_code": 404
}
```

**Solution:**
```python
# Verify conversation exists
conversations = client.list_conversations()
conv_ids = [c['conversation_id'] for c in conversations['conversations']]
print(f"Available conversations: {conv_ids}")
```

### Debug Mode

```bash
# Enable debug logging
export SGLANG_LOG_LEVEL=DEBUG

python -m sglang.launch_server \
  --enable-agent-tools \
  ...
```

### Health Monitoring

```python
import time

def monitor_health(client, interval=60):
    """Monitor server health continuously."""
    while True:
        try:
            health = client.health()
            stats = client.get_tier_stats()

            print(f"Status: {health['status']}")
            print(f"Active: {stats['conversation_tracker_stats']['tier_counts']['active']}")
            print(f"Memory: {stats['host_pool_stats']['memory_utilization']:.2%}")

        except Exception as e:
            print(f"Health check failed: {e}")

        time.sleep(interval)
```

---

## Next Steps

- [API Reference](API_REFERENCE.md) - Complete endpoint documentation
- [Architecture](ARCHITECTURE.md) - System design and internals
- [Examples](EXAMPLES.md) - Code examples and tutorials
- [WebSocket API](WEBSOCKET_API.md) - Streaming endpoints (Phase 4.5)
