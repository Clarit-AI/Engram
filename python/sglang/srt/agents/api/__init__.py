"""
Agent API module for REST and WebSocket endpoints.

Provides HTTP/WebSocket APIs for managing tools, conversations,
memory, and tier operations.

**Phase 4**: API endpoints and management layer

Key Components:
    - Tool management API (register, list, execute)
    - Conversation management API (list, search, restore)
    - Memory API (CRUD operations)
    - Tier management API (stats, transitions)
    - Health & monitoring endpoints
    - WebSocket streaming support

**Backward Compatibility**: API endpoints only activate when
agent framework is enabled via --enable-agent-tools.
"""

from sglang.srt.agents.api.models import (
    ConversationListResponse,
    MemoryRecallRequest,
    MemoryStoreRequest,
    TierStatsResponse,
    ToolCallRequest,
    ToolCallResponse,
)

try:
    from sglang.srt.agents.api.handlers import (
        AgentAPIHandler,
        register_agent_api_routes,
    )
    from sglang.srt.agents.api.websocket import (
        StreamingToolExecutor,
        WebSocketManager,
        register_websocket_routes,
    )
except ImportError:
    # Allow partial imports during development
    pass

__all__ = [
    # Models
    "ToolCallRequest",
    "ToolCallResponse",
    "MemoryStoreRequest",
    "MemoryRecallRequest",
    "ConversationListResponse",
    "TierStatsResponse",
    # Handlers
    "register_agent_api_routes",
    "AgentAPIHandler",
    # WebSocket
    "register_websocket_routes",
    "WebSocketManager",
    "StreamingToolExecutor",
]
