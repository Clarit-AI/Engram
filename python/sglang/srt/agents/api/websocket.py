"""
WebSocket handlers for streaming agent responses.

Provides real-time streaming of tool execution events,
agent responses, and system updates via WebSocket.
"""

import asyncio
import json
import logging
from typing import Any, Dict, Optional

from fastapi import WebSocket, WebSocketDisconnect

logger = logging.getLogger(__name__)


class WebSocketEventType:
    """WebSocket event types."""

    # Tool execution events
    TOOL_START = "tool_start"
    TOOL_PROGRESS = "tool_progress"
    TOOL_RESULT = "tool_result"
    TOOL_ERROR = "tool_error"

    # Agent events
    AGENT_START = "agent_start"
    AGENT_THINKING = "agent_thinking"
    AGENT_RESPONSE = "agent_response"
    AGENT_COMPLETE = "agent_complete"

    # System events
    TIER_TRANSITION = "tier_transition"
    MEMORY_UPDATE = "memory_update"
    HEARTBEAT = "heartbeat"
    ERROR = "error"


class WebSocketManager:
    """
    Manages WebSocket connections and message broadcasting.

    **Features:**
    - Connection lifecycle management
    - Message broadcasting to clients
    - Event filtering by subscription
    - Automatic reconnection handling
    """

    def __init__(self):
        """Initialize WebSocket manager."""
        # Active connections: {connection_id: WebSocket}
        self.active_connections: Dict[str, WebSocket] = {}

        # Subscriptions: {connection_id: set of event types}
        self.subscriptions: Dict[str, set] = {}

        logger.info("WebSocketManager initialized")

    async def connect(
        self, websocket: WebSocket, connection_id: str, subscribe_to: Optional[list] = None
    ):
        """
        Accept new WebSocket connection.

        Args:
            websocket: WebSocket instance
            connection_id: Unique connection identifier
            subscribe_to: List of event types to subscribe to (None = all)
        """
        await websocket.accept()
        self.active_connections[connection_id] = websocket

        if subscribe_to:
            self.subscriptions[connection_id] = set(subscribe_to)
        else:
            self.subscriptions[connection_id] = set()  # Empty = all events

        logger.info(
            f"WebSocket connected: {connection_id}, "
            f"subscribed to: {subscribe_to or 'all'}"
        )

        # Send welcome message
        await self.send_message(
            websocket,
            {
                "type": "connected",
                "connection_id": connection_id,
                "subscriptions": list(self.subscriptions[connection_id]),
            },
        )

    def disconnect(self, connection_id: str):
        """
        Disconnect WebSocket.

        Args:
            connection_id: Connection identifier
        """
        if connection_id in self.active_connections:
            del self.active_connections[connection_id]
        if connection_id in self.subscriptions:
            del self.subscriptions[connection_id]

        logger.info(f"WebSocket disconnected: {connection_id}")

    async def send_message(self, websocket: WebSocket, message: Dict[str, Any]):
        """
        Send message to specific WebSocket.

        Args:
            websocket: WebSocket instance
            message: Message dictionary
        """
        try:
            await websocket.send_json(message)
        except Exception as e:
            logger.error(f"Failed to send message: {e}", exc_info=True)

    async def broadcast(self, event_type: str, data: Dict[str, Any]):
        """
        Broadcast message to all subscribed connections.

        Args:
            event_type: Event type
            data: Event data
        """
        message = {"type": event_type, "data": data}

        disconnected = []

        for connection_id, websocket in self.active_connections.items():
            # Check subscription
            subscriptions = self.subscriptions.get(connection_id, set())
            if subscriptions and event_type not in subscriptions:
                continue  # Skip if not subscribed

            try:
                await self.send_message(websocket, message)
            except WebSocketDisconnect:
                disconnected.append(connection_id)
            except Exception as e:
                logger.error(
                    f"Error broadcasting to {connection_id}: {e}", exc_info=True
                )
                disconnected.append(connection_id)

        # Clean up disconnected clients
        for connection_id in disconnected:
            self.disconnect(connection_id)

    async def send_to_connection(
        self, connection_id: str, event_type: str, data: Dict[str, Any]
    ):
        """
        Send message to specific connection.

        Args:
            connection_id: Connection identifier
            event_type: Event type
            data: Event data
        """
        websocket = self.active_connections.get(connection_id)
        if not websocket:
            logger.warning(f"Connection {connection_id} not found")
            return

        message = {"type": event_type, "data": data}
        await self.send_message(websocket, message)

    def get_stats(self) -> Dict[str, Any]:
        """
        Get WebSocket statistics.

        Returns:
            Statistics dictionary
        """
        return {
            "active_connections": len(self.active_connections),
            "connections": list(self.active_connections.keys()),
            "subscriptions": {
                conn_id: list(subs) for conn_id, subs in self.subscriptions.items()
            },
        }


class StreamingToolExecutor:
    """
    Tool executor with WebSocket streaming support.

    Wraps ToolExecutor to emit real-time events during execution.
    """

    def __init__(self, tool_executor, ws_manager: WebSocketManager):
        """
        Initialize streaming executor.

        Args:
            tool_executor: ToolExecutor instance
            ws_manager: WebSocketManager instance
        """
        self.tool_executor = tool_executor
        self.ws_manager = ws_manager

    async def execute_with_streaming(
        self,
        tool_name: str,
        parameters: Dict[str, Any],
        connection_id: str,
        conversation_context: Optional[Dict[str, Any]] = None,
    ):
        """
        Execute tool with real-time event streaming.

        Args:
            tool_name: Tool name
            parameters: Tool parameters
            connection_id: WebSocket connection ID
            conversation_context: Conversation context

        Yields:
            Event dictionaries
        """
        # Emit start event
        await self.ws_manager.send_to_connection(
            connection_id,
            WebSocketEventType.TOOL_START,
            {"tool_name": tool_name, "parameters": parameters},
        )

        try:
            # Execute tool
            result = self.tool_executor.execute(
                tool_name=tool_name,
                parameters=parameters,
                conversation_context=conversation_context,
            )

            # Emit result event
            if result.is_success():
                await self.ws_manager.send_to_connection(
                    connection_id,
                    WebSocketEventType.TOOL_RESULT,
                    {
                        "tool_name": tool_name,
                        "status": result.status.value,
                        "result": result.result,
                        "execution_time_ms": result.execution_time_ms,
                    },
                )
            else:
                await self.ws_manager.send_to_connection(
                    connection_id,
                    WebSocketEventType.TOOL_ERROR,
                    {
                        "tool_name": tool_name,
                        "status": result.status.value,
                        "error": result.error,
                        "execution_time_ms": result.execution_time_ms,
                    },
                )

            return result

        except Exception as e:
            logger.error(f"Tool execution failed: {e}", exc_info=True)

            # Emit error event
            await self.ws_manager.send_to_connection(
                connection_id,
                WebSocketEventType.TOOL_ERROR,
                {"tool_name": tool_name, "error": str(e)},
            )

            raise


async def handle_websocket_connection(
    websocket: WebSocket,
    ws_manager: WebSocketManager,
    tool_executor,
):
    """
    Handle WebSocket connection lifecycle.

    Args:
        websocket: WebSocket instance
        ws_manager: WebSocketManager instance
        tool_executor: ToolExecutor instance
    """
    connection_id = f"ws_{id(websocket)}"

    try:
        # Accept connection
        await ws_manager.connect(websocket, connection_id)

        # Create streaming executor
        streaming_executor = StreamingToolExecutor(tool_executor, ws_manager)

        # Message loop
        while True:
            # Receive message
            data = await websocket.receive_json()

            command = data.get("command")

            if command == "execute_tool":
                # Execute tool with streaming
                tool_name = data.get("tool_name")
                parameters = data.get("parameters", {})
                conversation_id = data.get("conversation_id")

                conversation_context = None
                if conversation_id:
                    conversation_context = {"conversation_id": conversation_id}

                await streaming_executor.execute_with_streaming(
                    tool_name=tool_name,
                    parameters=parameters,
                    connection_id=connection_id,
                    conversation_context=conversation_context,
                )

            elif command == "subscribe":
                # Update subscriptions
                event_types = data.get("event_types", [])
                ws_manager.subscriptions[connection_id] = set(event_types)

                await ws_manager.send_to_connection(
                    connection_id,
                    "subscription_updated",
                    {"event_types": event_types},
                )

            elif command == "ping":
                # Heartbeat
                await ws_manager.send_to_connection(
                    connection_id, WebSocketEventType.HEARTBEAT, {"status": "ok"}
                )

            else:
                await ws_manager.send_to_connection(
                    connection_id,
                    WebSocketEventType.ERROR,
                    {"error": f"Unknown command: {command}"},
                )

    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected: {connection_id}")
    except Exception as e:
        logger.error(f"WebSocket error: {e}", exc_info=True)
    finally:
        ws_manager.disconnect(connection_id)


def register_websocket_routes(app, scheduler):
    """
    Register WebSocket routes to FastAPI app.

    Args:
        app: FastAPI application
        scheduler: Scheduler instance

    Returns:
        True if routes registered successfully
    """
    server_args = scheduler.server_args

    if not getattr(server_args, "enable_agent_tools", False):
        logger.info("Agent tools not enabled, skipping WebSocket routes")
        return False

    # Initialize WebSocket manager
    ws_manager = WebSocketManager()

    # Get tool executor
    tool_executor = getattr(scheduler, "tool_executor", None)

    if not tool_executor:
        logger.warning("Tool executor not available, WebSocket streaming disabled")
        return False

    @app.websocket("/v1/agent/ws")
    async def websocket_endpoint(websocket: WebSocket):
        """WebSocket endpoint for agent streaming."""
        await handle_websocket_connection(websocket, ws_manager, tool_executor)

    # Store manager for access from other components
    scheduler.ws_manager = ws_manager

    logger.info("WebSocket routes registered at /v1/agent/ws")
    return True
