"""
API handlers for agent endpoints.

Implements FastAPI route handlers for tool, conversation,
memory, and tier management.
"""

import logging
import time
from typing import Optional

from fastapi import APIRouter, HTTPException, status

from sglang.srt.agents.api.models import (
    AgentHistoryResponse,
    AgentRunRequest,
    AgentRunResponse,
    ConversationInfo,
    ConversationListResponse,
    ConversationRestoreRequest,
    ConversationRestoreResponse,
    ConversationSearchRequest,
    ErrorResponse,
    HealthResponse,
    MemoryRecallRequest,
    MemoryRecallResponse,
    MemorySearchRequest,
    MemorySearchResponse,
    MemoryStoreRequest,
    MemoryStoreResponse,
    StatsResponse,
    TierCleanupRequest,
    TierCleanupResponse,
    TierStatsResponse,
    TierTransitionRequest,
    TierTransitionResponse,
    ToolCallRequest,
    ToolCallResponse,
    ToolInfo,
    ToolListResponse,
)

logger = logging.getLogger(__name__)


class AgentAPIHandler:
    """
    API handler for agent framework endpoints.

    This class provides all the route handlers for the agent API,
    integrating with the scheduler's agent components (tool registry,
    tier manager, etc.).

    **Usage:**
        handler = AgentAPIHandler(scheduler)
        router = handler.create_router()
        app.include_router(router, prefix="/v1/agent")
    """

    def __init__(self, scheduler):
        """
        Initialize API handler.

        Args:
            scheduler: Scheduler instance with agent components
        """
        self.scheduler = scheduler

        # Get agent components from scheduler
        self.tool_registry = getattr(scheduler, "tool_registry", None)
        self.tool_executor = getattr(scheduler, "tool_executor", None)
        self.tool_parser = getattr(scheduler, "tool_parser", None)
        self.tier_manager = getattr(scheduler, "tier_manager", None)
        self.conversation_tracker = getattr(scheduler, "conversation_tracker", None)
        self.host_pool = getattr(scheduler, "host_pool", None)

        logger.info("AgentAPIHandler initialized")

    def create_router(self) -> APIRouter:
        """
        Create FastAPI router with all agent endpoints.

        Returns:
            APIRouter with agent routes
        """
        router = APIRouter(tags=["agent"])

        # Health & monitoring
        router.add_api_route("/health", self.health, methods=["GET"])
        router.add_api_route("/stats", self.stats, methods=["GET"])

        # Tool management
        router.add_api_route("/tools", self.list_tools, methods=["GET"])
        router.add_api_route("/tools/{tool_name}", self.get_tool, methods=["GET"])
        router.add_api_route("/tools/execute", self.execute_tool, methods=["POST"])

        # Memory management
        router.add_api_route("/memory/store", self.store_memory, methods=["POST"])
        router.add_api_route("/memory/recall", self.recall_memory, methods=["POST"])
        router.add_api_route("/memory/search", self.search_memory, methods=["POST"])

        # Conversation management
        router.add_api_route(
            "/conversations", self.list_conversations, methods=["GET"]
        )
        router.add_api_route(
            "/conversations/search", self.search_conversations, methods=["POST"]
        )
        router.add_api_route(
            "/conversations/restore", self.restore_conversation, methods=["POST"]
        )

        # Tier management
        router.add_api_route("/tiers/stats", self.tier_stats, methods=["GET"])
        router.add_api_route(
            "/tiers/transition", self.tier_transition, methods=["POST"]
        )
        router.add_api_route("/tiers/cleanup", self.tier_cleanup, methods=["POST"])

        # Agent execution (future - requires model integration)
        # router.add_api_route("/agent/run", self.run_agent, methods=["POST"])
        # router.add_api_route("/agent/history/{conversation_id}", self.agent_history, methods=["GET"])

        logger.info("Agent API router created with all endpoints")
        return router

    # ========================================================================
    # Health & Monitoring
    # ========================================================================

    async def health(self) -> HealthResponse:
        """Health check endpoint."""
        server_args = self.scheduler.server_args

        components = {
            "tool_registry": self.tool_registry is not None,
            "tool_executor": self.tool_executor is not None,
            "tool_parser": self.tool_parser is not None,
            "tier_manager": self.tier_manager is not None,
            "conversation_tracker": self.conversation_tracker is not None,
            "host_pool": self.host_pool is not None,
        }

        # Determine overall health
        agent_enabled = getattr(server_args, "enable_agent_tools", False)
        core_healthy = all(
            [self.tool_registry, self.tool_executor, self.tool_parser]
        )

        if agent_enabled and core_healthy:
            health_status = "healthy"
        elif agent_enabled and not core_healthy:
            health_status = "degraded"
        else:
            health_status = "healthy"  # Not enabled = healthy (expected state)

        return HealthResponse(
            status=health_status,
            agent_tools_enabled=agent_enabled,
            memory_tiers_enabled=getattr(server_args, "enable_memory_tiers", False),
            snapshot_persistence_enabled=getattr(
                server_args, "enable_snapshot_persistence", False
            ),
            components=components,
        )

    async def stats(self) -> StatsResponse:
        """General statistics endpoint."""
        if not self.tool_executor:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Agent tools not enabled",
            )

        tool_stats = self.tool_executor.get_stats()

        tier_stats = None
        if self.tier_manager:
            tier_stats = self.tier_manager.get_stats()

        registered_tools = len(self.tool_registry) if self.tool_registry else 0

        return StatsResponse(
            tool_execution_stats=tool_stats,
            tier_stats=tier_stats,
            registered_tools=registered_tools,
        )

    # ========================================================================
    # Tool Management
    # ========================================================================

    async def list_tools(self, tag: Optional[str] = None) -> ToolListResponse:
        """List all registered tools."""
        if not self.tool_registry:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Tool registry not available",
            )

        tools = self.tool_registry.list_tools(tag=tag)

        tool_infos = [
            ToolInfo(
                name=tool.name,
                description=tool.description,
                parameters=[p.to_dict() for p in tool.parameters],
                requires_conversation_context=tool.requires_conversation_context,
                is_async=tool.is_async,
                metadata=tool.metadata,
            )
            for tool in tools
        ]

        return ToolListResponse(tools=tool_infos, count=len(tool_infos))

    async def get_tool(self, tool_name: str) -> ToolInfo:
        """Get specific tool information."""
        if not self.tool_registry:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Tool registry not available",
            )

        tool = self.tool_registry.get(tool_name)

        if not tool:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Tool '{tool_name}' not found",
            )

        return ToolInfo(
            name=tool.name,
            description=tool.description,
            parameters=[p.to_dict() for p in tool.parameters],
            requires_conversation_context=tool.requires_conversation_context,
            is_async=tool.is_async,
            metadata=tool.metadata,
        )

    async def execute_tool(self, request: ToolCallRequest) -> ToolCallResponse:
        """Execute a tool."""
        if not self.tool_executor:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Tool executor not available",
            )

        # Build conversation context if provided
        conversation_context = None
        if request.conversation_id:
            conversation_context = {"conversation_id": request.conversation_id}

        # Execute tool
        result = self.tool_executor.execute(
            tool_name=request.tool_name,
            parameters=request.parameters,
            conversation_context=conversation_context,
            timeout=request.timeout,
        )

        return ToolCallResponse(
            tool_name=result.tool_name,
            status=result.status.value,
            result=result.result,
            error=result.error,
            execution_time_ms=result.execution_time_ms,
            metadata=result.metadata,
        )

    # ========================================================================
    # Memory Management
    # ========================================================================

    async def store_memory(self, request: MemoryStoreRequest) -> MemoryStoreResponse:
        """Store information to memory."""
        if not self.tool_executor:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Tool executor not available",
            )

        # Execute memory_store tool
        result = self.tool_executor.execute(
            tool_name="memory_store",
            parameters={
                "key": request.key,
                "value": request.value,
                "category": request.category,
            },
            conversation_context={"conversation_id": request.conversation_id},
        )

        if not result.is_success():
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Memory store failed: {result.error}",
            )

        return MemoryStoreResponse(
            status="success",
            conversation_id=request.conversation_id,
            key=request.key,
            category=request.category,
        )

    async def recall_memory(
        self, request: MemoryRecallRequest
    ) -> MemoryRecallResponse:
        """Recall information from memory."""
        if not self.tool_executor:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Tool executor not available",
            )

        # Execute memory_recall tool
        params = {}
        if request.key:
            params["key"] = request.key
        if request.category:
            params["category"] = request.category

        result = self.tool_executor.execute(
            tool_name="memory_recall",
            parameters=params,
            conversation_context={"conversation_id": request.conversation_id},
        )

        if not result.is_success():
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Memory recall failed: {result.error}",
            )

        return MemoryRecallResponse(
            status="success",
            conversation_id=request.conversation_id,
            data=result.result,
        )

    async def search_memory(
        self, request: MemorySearchRequest
    ) -> MemorySearchResponse:
        """Search memory by keyword."""
        if not self.tool_executor:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Tool executor not available",
            )

        # Execute memory_search tool
        params = {"query": request.query}
        if request.category:
            params["category"] = request.category

        result = self.tool_executor.execute(
            tool_name="memory_search",
            parameters=params,
            conversation_context={"conversation_id": request.conversation_id},
        )

        if not result.is_success():
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Memory search failed: {result.error}",
            )

        return MemorySearchResponse(
            status="success",
            conversation_id=request.conversation_id,
            query=request.query,
            results=result.result.get("results", []),
            count=result.result.get("count", 0),
        )

    # ========================================================================
    # Conversation Management
    # ========================================================================

    async def list_conversations(
        self, tier: Optional[str] = None
    ) -> ConversationListResponse:
        """List all conversations."""
        if not self.conversation_tracker:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Conversation tracker not available",
            )

        stats = self.conversation_tracker.get_stats()

        # Get conversations by tier if specified
        from sglang.srt.snapshot.conversation_tracker import ConversationTier

        if tier:
            try:
                tier_enum = ConversationTier(tier)
                conversations_list = self.conversation_tracker.list_conversations_by_tier(
                    tier_enum
                )
            except ValueError:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Invalid tier: {tier}",
                )
        else:
            # Get all conversations
            conversations_list = [
                self.conversation_tracker.get_state(conv_id)
                for conv_id in self.conversation_tracker._conversations.keys()
            ]

        conversation_infos = [
            ConversationInfo(
                conversation_id=conv.conversation_id,
                tier=conv.tier.value,
                last_access_time=conv.last_access_time,
                access_count=conv.access_count,
                metadata=conv.metadata or {},
            )
            for conv in conversations_list
            if conv is not None
        ]

        return ConversationListResponse(
            conversations=conversation_infos,
            count=len(conversation_infos),
            tier_counts=stats["tier_counts"],
        )

    async def search_conversations(
        self, request: ConversationSearchRequest
    ) -> ConversationListResponse:
        """Search conversations (placeholder - future implementation)."""
        # For now, just filter by tier if specified
        return await self.list_conversations(tier=request.tier)

    async def restore_conversation(
        self, request: ConversationRestoreRequest
    ) -> ConversationRestoreResponse:
        """Restore a conversation from tier storage."""
        if not self.tier_manager:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Tier manager not available",
            )

        try:
            result = self.tier_manager.restore_conversation(
                request.conversation_id, request.turn_number
            )

            if not result:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Conversation '{request.conversation_id}' not found",
                )

            # Get current tier
            tier = self.conversation_tracker.get_tier(request.conversation_id)

            return ConversationRestoreResponse(
                status="success",
                conversation_id=request.conversation_id,
                tier=tier.value if tier else "unknown",
                turn_number=request.turn_number or 0,
            )

        except Exception as e:
            logger.error(f"Failed to restore conversation: {e}", exc_info=True)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Restore failed: {str(e)}",
            )

    # ========================================================================
    # Tier Management
    # ========================================================================

    async def tier_stats(self) -> TierStatsResponse:
        """Get tier system statistics."""
        if not self.tier_manager:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Tier manager not available",
            )

        stats = self.tier_manager.get_stats()

        return TierStatsResponse(
            host_pool_stats=stats.get("host_pool", {}),
            conversation_tracker_stats=stats.get("conversation_tracker", {}),
            tier_manager_stats=stats,
        )

    async def tier_transition(
        self, request: TierTransitionRequest
    ) -> TierTransitionResponse:
        """Manually transition a conversation between tiers."""
        if not self.conversation_tracker:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Conversation tracker not available",
            )

        from sglang.srt.snapshot.conversation_tracker import ConversationTier

        # Get current tier
        old_tier = self.conversation_tracker.get_tier(request.conversation_id)

        if not old_tier:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Conversation '{request.conversation_id}' not found",
            )

        # Parse target tier
        try:
            target_tier_enum = ConversationTier(request.target_tier)
        except ValueError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid tier: {request.target_tier}",
            )

        # Perform transition
        success = self.conversation_tracker.transition_tier(
            request.conversation_id, target_tier_enum
        )

        if not success:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Tier transition failed",
            )

        return TierTransitionResponse(
            status="success",
            conversation_id=request.conversation_id,
            old_tier=old_tier.value,
            new_tier=target_tier_enum.value,
        )

    async def tier_cleanup(self, request: TierCleanupRequest) -> TierCleanupResponse:
        """Trigger tier cleanup cycle."""
        if not self.tier_manager:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Tier manager not available",
            )

        try:
            transitions = self.tier_manager.run_cleanup_cycle()

            return TierCleanupResponse(status="success", transitions=transitions)

        except Exception as e:
            logger.error(f"Cleanup failed: {e}", exc_info=True)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Cleanup failed: {str(e)}",
            )


def register_agent_api_routes(app, scheduler):
    """
    Register agent API routes to FastAPI app.

    Args:
        app: FastAPI application
        scheduler: Scheduler instance with agent components

    Returns:
        True if routes registered, False if agent not enabled
    """
    server_args = scheduler.server_args

    if not getattr(server_args, "enable_agent_tools", False):
        logger.info("Agent tools not enabled, skipping API routes")
        return False

    handler = AgentAPIHandler(scheduler)
    router = handler.create_router()

    app.include_router(router, prefix="/v1/agent")

    logger.info("Agent API routes registered at /v1/agent")
    return True
