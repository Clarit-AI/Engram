# Agent Framework API Reference

Complete API reference for the SGLang Agent Framework REST endpoints.

## Table of Contents

- [Overview](#overview)
- [Base URL](#base-url)
- [Authentication](#authentication)
- [Health & Monitoring](#health--monitoring)
- [Tool Management](#tool-management)
- [Memory Management](#memory-management)
- [Conversation Management](#conversation-management)
- [Tier Management](#tier-management)
- [Error Handling](#error-handling)
- [Rate Limits](#rate-limits)

---

## Overview

The Agent Framework API provides REST endpoints for managing tools, conversations, memory, and tier operations in SGLang. All endpoints return JSON responses and follow OpenAPI specifications.

**API Version:** v1
**Content-Type:** application/json
**Protocol:** HTTP/HTTPS

---

## Base URL

```
http://localhost:30000/v1/agent
```

Replace `localhost:30000` with your server address and port.

---

## Authentication

Currently, the API does not require authentication. Future versions may add:
- API key authentication
- OAuth 2.0 support
- JWT tokens

---

## Health & Monitoring

### GET /health

Health check endpoint for monitoring system status.

**Request:**
```bash
curl http://localhost:30000/v1/agent/health
```

**Response:**
```json
{
  "status": "healthy",
  "agent_tools_enabled": true,
  "memory_tiers_enabled": true,
  "snapshot_persistence_enabled": true,
  "components": {
    "tool_registry": true,
    "tool_executor": true,
    "tool_parser": true,
    "tier_manager": true,
    "conversation_tracker": true,
    "host_pool": true
  }
}
```

**Status Values:**
- `healthy` - All systems operational
- `degraded` - Some components unavailable
- `unhealthy` - Critical failures

---

### GET /stats

Get system statistics and metrics.

**Request:**
```bash
curl http://localhost:30000/v1/agent/stats
```

**Response:**
```json
{
  "tool_execution_stats": {
    "total_executions": 1523,
    "successful_executions": 1487,
    "failed_executions": 36,
    "avg_execution_time_ms": 12.3,
    "executions_by_tool": {
      "calculator": 523,
      "memory_store": 412,
      "memory_recall": 588
    }
  },
  "tier_stats": {
    "active_conversations": 3,
    "warm_conversations": 10,
    "cold_conversations": 2,
    "memory_utilization": 0.35,
    "hit_rate": 0.87
  },
  "registered_tools": 4
}
```

---

## Tool Management

### GET /tools

List all registered tools with optional filtering.

**Query Parameters:**
- `tag` (optional) - Filter tools by tag

**Request:**
```bash
# List all tools
curl http://localhost:30000/v1/agent/tools

# Filter by tag
curl http://localhost:30000/v1/agent/tools?tag=memory
```

**Response:**
```json
{
  "tools": [
    {
      "name": "calculator",
      "description": "Perform mathematical calculations",
      "parameters": [
        {
          "name": "expression",
          "type": "string",
          "required": true,
          "description": "Mathematical expression to evaluate"
        }
      ],
      "requires_conversation_context": false,
      "is_async": false,
      "metadata": {
        "tags": ["utility", "math"],
        "version": "1.0.0"
      }
    },
    {
      "name": "memory_store",
      "description": "Store information to conversation memory",
      "parameters": [
        {
          "name": "key",
          "type": "string",
          "required": true,
          "description": "Memory key"
        },
        {
          "name": "value",
          "type": "string",
          "required": true,
          "description": "Value to store"
        },
        {
          "name": "category",
          "type": "string",
          "required": false,
          "description": "Memory category (default: general)"
        }
      ],
      "requires_conversation_context": true,
      "is_async": false,
      "metadata": {
        "tags": ["memory", "storage"]
      }
    }
  ],
  "count": 2
}
```

---

### GET /tools/{tool_name}

Get detailed information about a specific tool.

**Path Parameters:**
- `tool_name` - Name of the tool

**Request:**
```bash
curl http://localhost:30000/v1/agent/tools/calculator
```

**Response:**
```json
{
  "name": "calculator",
  "description": "Perform mathematical calculations",
  "parameters": [
    {
      "name": "expression",
      "type": "string",
      "required": true,
      "description": "Mathematical expression to evaluate"
    }
  ],
  "requires_conversation_context": false,
  "is_async": false,
  "metadata": {
    "tags": ["utility", "math"],
    "version": "1.0.0",
    "examples": [
      {"input": "2 + 2", "output": 4},
      {"input": "sqrt(16)", "output": 4.0}
    ]
  }
}
```

**Error Response (404):**
```json
{
  "error": "Tool 'unknown_tool' not found",
  "status_code": 404
}
```

---

### POST /tools/execute

Execute a tool with specified parameters.

**Request Body:**
```json
{
  "tool_name": "calculator",
  "parameters": {
    "expression": "sqrt(16) + 2 * 3"
  },
  "conversation_id": "conv_123",
  "timeout": 30.0
}
```

**Fields:**
- `tool_name` (required) - Name of tool to execute
- `parameters` (optional) - Tool-specific parameters (default: {})
- `conversation_id` (optional) - Conversation context for memory tools
- `timeout` (optional) - Execution timeout in seconds

**Request:**
```bash
curl -X POST http://localhost:30000/v1/agent/tools/execute \
  -H "Content-Type: application/json" \
  -d '{
    "tool_name": "calculator",
    "parameters": {"expression": "sqrt(16) + 2 * 3"}
  }'
```

**Success Response:**
```json
{
  "tool_name": "calculator",
  "status": "success",
  "result": 10.0,
  "error": null,
  "execution_time_ms": 2.5,
  "metadata": {
    "cached": false
  }
}
```

**Error Response:**
```json
{
  "tool_name": "calculator",
  "status": "error",
  "result": null,
  "error": "Invalid expression: division by zero",
  "execution_time_ms": 1.2,
  "metadata": {}
}
```

**Status Values:**
- `success` - Tool executed successfully
- `error` - Tool execution failed
- `timeout` - Execution exceeded timeout

---

## Memory Management

### POST /memory/store

Store information to conversation memory.

**Request Body:**
```json
{
  "conversation_id": "conv_123",
  "key": "user_name",
  "value": "Alice",
  "category": "user_info"
}
```

**Fields:**
- `conversation_id` (required) - Conversation identifier
- `key` (required) - Memory key
- `value` (required) - Value to store
- `category` (optional) - Memory category (default: "general")

**Request:**
```bash
curl -X POST http://localhost:30000/v1/agent/memory/store \
  -H "Content-Type: application/json" \
  -d '{
    "conversation_id": "conv_123",
    "key": "user_name",
    "value": "Alice",
    "category": "user_info"
  }'
```

**Response:**
```json
{
  "status": "success",
  "conversation_id": "conv_123",
  "key": "user_name",
  "category": "user_info"
}
```

---

### POST /memory/recall

Recall information from conversation memory.

**Request Body:**
```json
{
  "conversation_id": "conv_123",
  "key": "user_name",
  "category": "user_info"
}
```

**Fields:**
- `conversation_id` (required) - Conversation identifier
- `key` (optional) - Specific key to recall (omit to get all)
- `category` (optional) - Filter by category

**Request:**
```bash
curl -X POST http://localhost:30000/v1/agent/memory/recall \
  -H "Content-Type: application/json" \
  -d '{
    "conversation_id": "conv_123",
    "category": "user_info"
  }'
```

**Response:**
```json
{
  "status": "success",
  "conversation_id": "conv_123",
  "data": {
    "user_name": "Alice",
    "user_age": "25",
    "user_location": "San Francisco"
  }
}
```

---

### POST /memory/search

Search memory by keyword or pattern.

**Request Body:**
```json
{
  "conversation_id": "conv_123",
  "query": "user",
  "category": "user_info"
}
```

**Fields:**
- `conversation_id` (required) - Conversation identifier
- `query` (required) - Search query
- `category` (optional) - Filter by category

**Request:**
```bash
curl -X POST http://localhost:30000/v1/agent/memory/search \
  -H "Content-Type: application/json" \
  -d '{
    "conversation_id": "conv_123",
    "query": "user"
  }'
```

**Response:**
```json
{
  "status": "success",
  "conversation_id": "conv_123",
  "query": "user",
  "results": [
    {
      "key": "user_name",
      "value": "Alice",
      "category": "user_info",
      "relevance_score": 0.95
    },
    {
      "key": "user_age",
      "value": "25",
      "category": "user_info",
      "relevance_score": 0.87
    }
  ],
  "count": 2
}
```

---

## Conversation Management

### GET /conversations

List all conversations with optional tier filtering.

**Query Parameters:**
- `tier` (optional) - Filter by tier (active/warm/cold/archived)

**Request:**
```bash
# List all conversations
curl http://localhost:30000/v1/agent/conversations

# Filter by tier
curl http://localhost:30000/v1/agent/conversations?tier=warm
```

**Response:**
```json
{
  "conversations": [
    {
      "conversation_id": "conv_123",
      "tier": "active",
      "last_access_time": 1707849600.0,
      "access_count": 15,
      "metadata": {
        "user_id": "user_456",
        "session_start": 1707846000.0
      }
    },
    {
      "conversation_id": "conv_456",
      "tier": "warm",
      "last_access_time": 1707843000.0,
      "access_count": 8,
      "metadata": {}
    }
  ],
  "count": 2,
  "tier_counts": {
    "active": 3,
    "warm": 10,
    "cold": 2,
    "archived": 0
  }
}
```

---

### POST /conversations/search

Search conversations by criteria.

**Request Body:**
```json
{
  "query": "user_456",
  "tier": "warm",
  "limit": 100
}
```

**Fields:**
- `query` (optional) - Search query
- `tier` (optional) - Filter by tier
- `limit` (optional) - Max results (default: 100)

**Request:**
```bash
curl -X POST http://localhost:30000/v1/agent/conversations/search \
  -H "Content-Type: application/json" \
  -d '{
    "tier": "warm",
    "limit": 10
  }'
```

**Response:**
```json
{
  "conversations": [...],
  "count": 10,
  "tier_counts": {...}
}
```

---

### POST /conversations/restore

Restore a conversation from tier storage.

**Request Body:**
```json
{
  "conversation_id": "conv_123",
  "turn_number": 5
}
```

**Fields:**
- `conversation_id` (required) - Conversation to restore
- `turn_number` (optional) - Specific turn to restore

**Request:**
```bash
curl -X POST http://localhost:30000/v1/agent/conversations/restore \
  -H "Content-Type: application/json" \
  -d '{
    "conversation_id": "conv_123"
  }'
```

**Response:**
```json
{
  "status": "success",
  "conversation_id": "conv_123",
  "tier": "warm",
  "turn_number": 5
}
```

---

## Tier Management

### GET /tiers/stats

Get detailed tier system statistics.

**Request:**
```bash
curl http://localhost:30000/v1/agent/tiers/stats
```

**Response:**
```json
{
  "host_pool_stats": {
    "current_conversations": 15,
    "max_conversations": 100,
    "memory_utilization": 0.35,
    "hit_rate": 0.87,
    "evictions": 23,
    "total_memory_gb": 24.0,
    "used_memory_gb": 8.4
  },
  "conversation_tracker_stats": {
    "total_conversations": 15,
    "tier_counts": {
      "active": 3,
      "warm": 10,
      "cold": 2
    },
    "transitions_last_hour": 42,
    "avg_access_time_ms": 12.5
  },
  "tier_manager_stats": {
    "cleanup_cycles": 156,
    "transitions_performed": 423,
    "errors": 0,
    "last_cleanup_time": 1707849600.0
  }
}
```

---

### POST /tiers/transition

Manually transition a conversation between tiers.

**Request Body:**
```json
{
  "conversation_id": "conv_123",
  "target_tier": "warm"
}
```

**Fields:**
- `conversation_id` (required) - Conversation to transition
- `target_tier` (required) - Target tier (active/warm/cold/archived)

**Request:**
```bash
curl -X POST http://localhost:30000/v1/agent/tiers/transition \
  -H "Content-Type: application/json" \
  -d '{
    "conversation_id": "conv_123",
    "target_tier": "warm"
  }'
```

**Response:**
```json
{
  "status": "success",
  "conversation_id": "conv_123",
  "old_tier": "active",
  "new_tier": "warm"
}
```

---

### POST /tiers/cleanup

Trigger manual tier cleanup cycle.

**Request Body:**
```json
{
  "force": false
}
```

**Fields:**
- `force` (optional) - Force cleanup regardless of timeouts (default: false)

**Request:**
```bash
curl -X POST http://localhost:30000/v1/agent/tiers/cleanup \
  -H "Content-Type: application/json" \
  -d '{
    "force": true
  }'
```

**Response:**
```json
{
  "status": "success",
  "transitions": {
    "active_to_warm": 2,
    "warm_to_cold": 5,
    "cold_to_archived": 1
  }
}
```

---

## Error Handling

All endpoints return consistent error responses:

### Error Response Format

```json
{
  "error": "Tool 'unknown_tool' not found",
  "detail": "The requested tool does not exist in the registry",
  "status_code": 404
}
```

### HTTP Status Codes

- `200 OK` - Request successful
- `400 Bad Request` - Invalid request parameters
- `404 Not Found` - Resource not found
- `500 Internal Server Error` - Server-side error
- `503 Service Unavailable` - Service not enabled

### Common Error Scenarios

**Agent tools not enabled:**
```json
{
  "error": "Agent tools not enabled",
  "detail": "Start server with --enable-agent-tools",
  "status_code": 503
}
```

**Invalid tier:**
```json
{
  "error": "Invalid tier: invalid_tier",
  "detail": "Valid tiers: active, warm, cold, archived",
  "status_code": 400
}
```

**Tool execution timeout:**
```json
{
  "tool_name": "slow_tool",
  "status": "timeout",
  "error": "Execution exceeded timeout of 30.0 seconds",
  "execution_time_ms": 30000.0
}
```

---

## Rate Limits

Currently, there are no enforced rate limits. Future versions may implement:

- Request rate limiting (requests per minute)
- Tool execution quotas
- Memory storage limits
- Concurrent connection limits

---

## API Versioning

The API uses URL-based versioning:

- Current version: `/v1/agent/...`
- Future versions: `/v2/agent/...`

Breaking changes will be introduced in new versions while maintaining backward compatibility with older versions.

---

## Next Steps

- [Architecture Documentation](ARCHITECTURE.md)
- [Integration Guide](INTEGRATION_GUIDE.md)
- [Code Examples](EXAMPLES.md)
- [WebSocket API Reference](WEBSOCKET_API.md) (Phase 4.5)
