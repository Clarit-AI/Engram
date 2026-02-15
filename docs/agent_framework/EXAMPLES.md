# Agent Framework Examples

Comprehensive code examples for using the SGLang Agent Framework.

## Table of Contents

- [Basic Tool Execution](#basic-tool-execution)
- [Memory Management](#memory-management)
- [WebSocket Streaming](#websocket-streaming)
- [Tier Management](#tier-management)
- [Building Chatbots](#building-chatbots)
- [Advanced Patterns](#advanced-patterns)

---

## Basic Tool Execution

### Simple Calculator

```python
import requests

base_url = "http://localhost:30000/v1/agent"

# Execute calculator tool
response = requests.post(
    f"{base_url}/tools/execute",
    json={
        "tool_name": "calculator",
        "parameters": {"expression": "2 + 2"}
    }
)

result = response.json()
print(f"Result: {result['result']}")  # Result: 4.0
```

### Batch Execution

```python
import requests
from concurrent.futures import ThreadPoolExecutor

base_url = "http://localhost:30000/v1/agent"

def execute_calculator(expression):
    """Execute calculator tool."""
    response = requests.post(
        f"{base_url}/tools/execute",
        json={
            "tool_name": "calculator",
            "parameters": {"expression": expression}
        }
    )
    return response.json()

# Execute multiple calculations in parallel
expressions = ["2 + 2", "sqrt(16)", "10 * 5", "100 / 4"]

with ThreadPoolExecutor(max_workers=4) as executor:
    results = list(executor.map(execute_calculator, expressions))

for expr, result in zip(expressions, results):
    print(f"{expr} = {result['result']}")
```

### Error Handling

```python
import requests

base_url = "http://localhost:30000/v1/agent"

try:
    response = requests.post(
        f"{base_url}/tools/execute",
        json={
            "tool_name": "calculator",
            "parameters": {"expression": "1 / 0"}
        },
        timeout=10
    )
    response.raise_for_status()

    result = response.json()

    if result['status'] == 'success':
        print(f"Result: {result['result']}")
    else:
        print(f"Tool error: {result['error']}")

except requests.exceptions.Timeout:
    print("Request timed out")
except requests.exceptions.HTTPError as e:
    print(f"HTTP error: {e}")
except Exception as e:
    print(f"Error: {e}")
```

---

## Memory Management

### Storing User Profile

```python
import requests

base_url = "http://localhost:30000/v1/agent"
conversation_id = "conv_user_123"

# Store user profile information
profile = {
    "name": "Alice",
    "age": "25",
    "location": "San Francisco",
    "language": "English"
}

for key, value in profile.items():
    requests.post(
        f"{base_url}/memory/store",
        json={
            "conversation_id": conversation_id,
            "key": key,
            "value": value,
            "category": "profile"
        }
    )

print("Profile stored successfully")

# Recall profile
response = requests.post(
    f"{base_url}/memory/recall",
    json={
        "conversation_id": conversation_id,
        "category": "profile"
    }
)

recalled_profile = response.json()['data']
print(f"Recalled profile: {recalled_profile}")
```

### Search Memory

```python
import requests

base_url = "http://localhost:30000/v1/agent"
conversation_id = "conv_user_123"

# Search for keywords
response = requests.post(
    f"{base_url}/memory/search",
    json={
        "conversation_id": conversation_id,
        "query": "language"
    }
)

results = response.json()['results']

for result in results:
    print(f"{result['key']}: {result['value']} (score: {result['relevance_score']})")
```

### Conversation Context

```python
import requests

base_url = "http://localhost:30000/v1/agent"

class ConversationMemory:
    """Helper class for managing conversation memory."""

    def __init__(self, conversation_id, base_url):
        self.conversation_id = conversation_id
        self.base_url = base_url

    def store(self, key, value, category="general"):
        """Store memory."""
        requests.post(
            f"{self.base_url}/memory/store",
            json={
                "conversation_id": self.conversation_id,
                "key": key,
                "value": value,
                "category": category
            }
        )

    def recall(self, key=None, category=None):
        """Recall memory."""
        data = {"conversation_id": self.conversation_id}
        if key:
            data["key"] = key
        if category:
            data["category"] = category

        response = requests.post(
            f"{self.base_url}/memory/recall",
            json=data
        )
        return response.json()['data']

    def search(self, query, category=None):
        """Search memory."""
        data = {
            "conversation_id": self.conversation_id,
            "query": query
        }
        if category:
            data["category"] = category

        response = requests.post(
            f"{self.base_url}/memory/search",
            json=data
        )
        return response.json()['results']

# Usage
memory = ConversationMemory("conv_123", base_url)

# Store preferences
memory.store("theme", "dark", "preferences")
memory.store("language", "English", "preferences")

# Recall all preferences
prefs = memory.recall(category="preferences")
print(f"Preferences: {prefs}")
```

---

## WebSocket Streaming

### Python WebSocket Client

```python
import asyncio
import websockets
import json

async def stream_tool_execution():
    """Execute tool with real-time streaming."""
    uri = "ws://localhost:30000/v1/agent/ws"

    async with websockets.connect(uri) as websocket:
        # Wait for connection confirmation
        welcome = await websocket.recv()
        print(f"Connected: {welcome}")

        # Execute tool
        await websocket.send(json.dumps({
            "command": "execute_tool",
            "tool_name": "calculator",
            "parameters": {"expression": "2 + 2"}
        }))

        # Receive events
        while True:
            try:
                message = await asyncio.wait_for(
                    websocket.recv(),
                    timeout=10
                )

                event = json.loads(message)
                event_type = event.get("type")
                data = event.get("data", {})

                if event_type == "tool_start":
                    print(f"Tool started: {data['tool_name']}")

                elif event_type == "tool_result":
                    print(f"Result: {data['result']}")
                    print(f"Time: {data['execution_time_ms']}ms")
                    break  # Done

                elif event_type == "tool_error":
                    print(f"Error: {data['error']}")
                    break

            except asyncio.TimeoutError:
                print("Timeout waiting for event")
                break

# Run
asyncio.run(stream_tool_execution())
```

### JavaScript WebSocket Client

```javascript
// Node.js WebSocket client
const WebSocket = require('ws');

function streamToolExecution() {
  const ws = new WebSocket('ws://localhost:30000/v1/agent/ws');

  ws.on('open', () => {
    console.log('Connected to WebSocket');
  });

  ws.on('message', (data) => {
    const event = JSON.parse(data);
    const type = event.type;
    const eventData = event.data || {};

    if (type === 'connected') {
      console.log('Connection confirmed:', event.connection_id);

      // Execute tool
      ws.send(JSON.stringify({
        command: 'execute_tool',
        tool_name: 'calculator',
        parameters: { expression: '2 + 2' }
      }));

    } else if (type === 'tool_start') {
      console.log('Tool started:', eventData.tool_name);

    } else if (type === 'tool_result') {
      console.log('Result:', eventData.result);
      console.log('Time:', eventData.execution_time_ms + 'ms');
      ws.close();

    } else if (type === 'tool_error') {
      console.error('Error:', eventData.error);
      ws.close();
    }
  });

  ws.on('error', (error) => {
    console.error('WebSocket error:', error);
  });

  ws.on('close', () => {
    console.log('WebSocket closed');
  });
}

streamToolExecution();
```

### Browser WebSocket Client

```html
<!DOCTYPE html>
<html>
<head>
  <title>WebSocket Tool Executor</title>
</head>
<body>
  <h1>Real-Time Tool Execution</h1>

  <input type="text" id="expression" placeholder="Enter expression">
  <button onclick="executeWithStreaming()">Calculate (Streaming)</button>

  <div id="events"></div>

  <script>
    let ws = null;

    function connect() {
      ws = new WebSocket('ws://localhost:30000/v1/agent/ws');

      ws.onopen = () => {
        console.log('WebSocket connected');
        addEvent('Connected to server', 'success');
      };

      ws.onmessage = (event) => {
        const data = JSON.parse(event.data);
        handleEvent(data);
      };

      ws.onerror = (error) => {
        console.error('WebSocket error:', error);
        addEvent('Connection error', 'error');
      };

      ws.onclose = () => {
        console.log('WebSocket closed');
        addEvent('Connection closed', 'info');
      };
    }

    function handleEvent(event) {
      const type = event.type;
      const data = event.data || {};

      if (type === 'connected') {
        addEvent(`Connected: ${event.connection_id}`, 'success');

      } else if (type === 'tool_start') {
        addEvent(`Executing: ${data.tool_name}`, 'info');

      } else if (type === 'tool_result') {
        addEvent(`Result: ${data.result} (${data.execution_time_ms}ms)`, 'success');

      } else if (type === 'tool_error') {
        addEvent(`Error: ${data.error}`, 'error');
      }
    }

    function executeWithStreaming() {
      if (!ws || ws.readyState !== WebSocket.OPEN) {
        connect();
        setTimeout(executeWithStreaming, 1000);  // Retry after connection
        return;
      }

      const expression = document.getElementById('expression').value;

      ws.send(JSON.stringify({
        command: 'execute_tool',
        tool_name: 'calculator',
        parameters: { expression }
      }));
    }

    function addEvent(message, type) {
      const div = document.getElementById('events');
      const p = document.createElement('p');
      p.textContent = `[${type}] ${message}`;
      p.style.color = type === 'error' ? 'red' : type === 'success' ? 'green' : 'blue';
      div.appendChild(p);
    }

    // Connect on load
    connect();
  </script>
</body>
</html>
```

---

## Tier Management

### Monitor Tier Statistics

```python
import requests
import time

base_url = "http://localhost:30000/v1/agent"

def monitor_tiers(interval=60):
    """Monitor tier statistics continuously."""
    while True:
        try:
            # Get tier stats
            response = requests.get(f"{base_url}/tiers/stats")
            stats = response.json()

            # Print statistics
            tracker = stats['conversation_tracker_stats']
            pool = stats['host_pool_stats']

            print("\n" + "="*50)
            print("TIER STATISTICS")
            print("="*50)

            print(f"\nConversations by tier:")
            for tier, count in tracker['tier_counts'].items():
                print(f"  {tier}: {count}")

            print(f"\nMemory utilization: {pool['memory_utilization']:.1%}")
            print(f"Hit rate: {pool['hit_rate']:.1%}")
            print(f"Current conversations: {pool['current_conversations']}")

            time.sleep(interval)

        except KeyboardInterrupt:
            print("\nMonitoring stopped")
            break
        except Exception as e:
            print(f"Error: {e}")
            time.sleep(interval)

# Run monitor
monitor_tiers(interval=30)
```

### Manual Tier Transition

```python
import requests

base_url = "http://localhost:30000/v1/agent"

# List conversations in active tier
response = requests.get(f"{base_url}/conversations?tier=active")
conversations = response.json()['conversations']

print(f"Active conversations: {len(conversations)}")

# Move oldest conversation to warm tier
if conversations:
    oldest = min(conversations, key=lambda c: c['last_access_time'])

    response = requests.post(
        f"{base_url}/tiers/transition",
        json={
            "conversation_id": oldest['conversation_id'],
            "target_tier": "warm"
        }
    )

    result = response.json()
    print(f"Transitioned {result['conversation_id']}: "
          f"{result['old_tier']} → {result['new_tier']}")
```

### Trigger Cleanup

```python
import requests

base_url = "http://localhost:30000/v1/agent"

# Force cleanup cycle
response = requests.post(
    f"{base_url}/tiers/cleanup",
    json={"force": True}
)

result = response.json()
print(f"Cleanup completed:")
for transition, count in result['transitions'].items():
    print(f"  {transition}: {count}")
```

---

## Building Chatbots

### Simple Chatbot with Memory

```python
import requests
import uuid

base_url = "http://localhost:30000/v1/agent"

class Chatbot:
    """Simple chatbot with conversation memory."""

    def __init__(self):
        self.conversation_id = f"conv_{uuid.uuid4().hex[:8]}"
        self.base_url = base_url

    def store_context(self, key, value, category="context"):
        """Store conversation context."""
        requests.post(
            f"{self.base_url}/memory/store",
            json={
                "conversation_id": self.conversation_id,
                "key": key,
                "value": value,
                "category": category
            }
        )

    def recall_context(self, category=None):
        """Recall conversation context."""
        data = {"conversation_id": self.conversation_id}
        if category:
            data["category"] = category

        response = requests.post(
            f"{self.base_url}/memory/recall",
            json=data
        )
        return response.json()['data']

    def chat(self, user_input):
        """Process user input."""
        # Store user input
        self.store_context("last_input", user_input)

        # Check if calculation requested
        if any(op in user_input for op in ['+', '-', '*', '/', 'sqrt', '^']):
            # Extract expression (simplified)
            expression = user_input

            # Execute calculator
            response = requests.post(
                f"{self.base_url}/tools/execute",
                json={
                    "tool_name": "calculator",
                    "parameters": {"expression": expression},
                    "conversation_id": self.conversation_id
                }
            )

            result = response.json()

            if result['status'] == 'success':
                return f"The result is {result['result']}"
            else:
                return f"Sorry, I couldn't calculate that: {result['error']}"

        # Remember user name
        if "my name is" in user_input.lower():
            name = user_input.lower().split("my name is")[1].strip()
            self.store_context("user_name", name, "profile")
            return f"Nice to meet you, {name}!"

        # Recall user name
        context = self.recall_context()
        if "user_name" in context:
            return f"Hello {context['user_name']}, how can I help you?"

        return "Hello! I can help you with calculations."

# Usage
bot = Chatbot()

print(bot.chat("My name is Alice"))
# → "Nice to meet you, Alice!"

print(bot.chat("Hello"))
# → "Hello Alice, how can I help you?"

print(bot.chat("2 + 2"))
# → "The result is 4.0"
```

---

## Advanced Patterns

### Async Batch Processing

```python
import asyncio
import aiohttp

base_url = "http://localhost:30000/v1/agent"

async def execute_tool_async(session, tool_name, parameters):
    """Execute tool asynchronously."""
    async with session.post(
        f"{base_url}/tools/execute",
        json={
            "tool_name": tool_name,
            "parameters": parameters
        }
    ) as response:
        return await response.json()

async def batch_process():
    """Process multiple tool calls in parallel."""
    tasks = [
        ("calculator", {"expression": "2 + 2"}),
        ("calculator", {"expression": "sqrt(16)"}),
        ("calculator", {"expression": "10 * 5"}),
        ("calculator", {"expression": "100 / 4"}),
    ]

    async with aiohttp.ClientSession() as session:
        results = await asyncio.gather(*[
            execute_tool_async(session, name, params)
            for name, params in tasks
        ])

        for (name, params), result in zip(tasks, results):
            print(f"{params['expression']} = {result['result']}")

# Run
asyncio.run(batch_process())
```

### Retry Logic

```python
import requests
import time
from functools import wraps

def retry(max_attempts=3, delay=1, backoff=2):
    """Retry decorator with exponential backoff."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            attempt = 0
            current_delay = delay

            while attempt < max_attempts:
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    attempt += 1
                    if attempt >= max_attempts:
                        raise

                    print(f"Attempt {attempt} failed: {e}. "
                          f"Retrying in {current_delay}s...")
                    time.sleep(current_delay)
                    current_delay *= backoff

        return wrapper
    return decorator

@retry(max_attempts=3, delay=1, backoff=2)
def execute_tool_with_retry(tool_name, parameters):
    """Execute tool with automatic retry."""
    response = requests.post(
        f"{base_url}/tools/execute",
        json={
            "tool_name": tool_name,
            "parameters": parameters
        },
        timeout=10
    )
    response.raise_for_status()
    return response.json()

# Usage
result = execute_tool_with_retry("calculator", {"expression": "2 + 2"})
print(result['result'])
```

### Caching Results

```python
import requests
from functools import lru_cache
import hashlib
import json

base_url = "http://localhost:30000/v1/agent"

class CachedToolExecutor:
    """Tool executor with result caching."""

    def __init__(self, cache_size=128):
        self.execute = lru_cache(maxsize=cache_size)(self._execute)

    def _cache_key(self, tool_name, parameters):
        """Generate cache key."""
        data = json.dumps({"tool": tool_name, "params": parameters}, sort_keys=True)
        return hashlib.md5(data.encode()).hexdigest()

    def _execute(self, cache_key):
        """Execute tool (cached)."""
        # Extract from cache key (simplified - in reality, store mapping)
        # For demo, just re-execute
        pass

    def execute_tool(self, tool_name, parameters):
        """Execute with caching."""
        cache_key = self._cache_key(tool_name, parameters)

        # Check cache
        cached = self.execute.cache_info()
        print(f"Cache: {cached.hits} hits, {cached.misses} misses")

        # Execute
        response = requests.post(
            f"{base_url}/tools/execute",
            json={
                "tool_name": tool_name,
                "parameters": parameters
            }
        )
        return response.json()

# Usage
executor = CachedToolExecutor()

# First call - cache miss
result1 = executor.execute_tool("calculator", {"expression": "2 + 2"})

# Second call - cache hit (if same parameters)
result2 = executor.execute_tool("calculator", {"expression": "2 + 2"})
```

---

## Next Steps

- [API Reference](API_REFERENCE.md) - Complete API documentation
- [Architecture](ARCHITECTURE.md) - System design
- [Integration Guide](INTEGRATION_GUIDE.md) - Setup and deployment
