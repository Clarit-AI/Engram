"""
Request/response models for Agent API.

Pydantic models for API validation and serialization.
"""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


# ============================================================================
# Tool Management Models
# ============================================================================


class ToolInfo(BaseModel):
    """Tool information model."""

    name: str = Field(..., description="Tool name")
    description: str = Field(..., description="Tool description")
    parameters: List[Dict[str, Any]] = Field(
        default_factory=list, description="Tool parameters"
    )
    requires_conversation_context: bool = Field(
        default=False, description="Whether tool needs conversation context"
    )
    is_async: bool = Field(default=False, description="Whether tool is async")
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Additional metadata"
    )


class ToolListResponse(BaseModel):
    """Response for listing tools."""

    tools: List[ToolInfo] = Field(..., description="List of available tools")
    count: int = Field(..., description="Total tool count")


class ToolCallRequest(BaseModel):
    """Request to execute a tool."""

    tool_name: str = Field(..., description="Name of tool to execute")
    parameters: Dict[str, Any] = Field(
        default_factory=dict, description="Tool parameters"
    )
    conversation_id: Optional[str] = Field(
        None, description="Optional conversation ID for context"
    )
    timeout: Optional[float] = Field(
        None, description="Execution timeout in seconds"
    )


class ToolCallResponse(BaseModel):
    """Response from tool execution."""

    tool_name: str = Field(..., description="Tool name")
    status: str = Field(..., description="Execution status")
    result: Optional[Any] = Field(None, description="Tool result")
    error: Optional[str] = Field(None, description="Error message if failed")
    execution_time_ms: float = Field(..., description="Execution time in ms")
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Additional metadata"
    )


class ToolRegisterRequest(BaseModel):
    """Request to register a custom tool (future feature)."""

    name: str = Field(..., description="Tool name")
    description: str = Field(..., description="Tool description")
    parameters: List[Dict[str, Any]] = Field(..., description="Parameter definitions")
    code: Optional[str] = Field(None, description="Tool implementation code")


# ============================================================================
# Memory Management Models
# ============================================================================


class MemoryStoreRequest(BaseModel):
    """Request to store memory."""

    conversation_id: str = Field(..., description="Conversation ID")
    key: str = Field(..., description="Memory key")
    value: str = Field(..., description="Value to store")
    category: str = Field(default="general", description="Memory category")


class MemoryStoreResponse(BaseModel):
    """Response from memory store operation."""

    status: str = Field(..., description="Operation status")
    conversation_id: str = Field(..., description="Conversation ID")
    key: str = Field(..., description="Stored key")
    category: str = Field(..., description="Memory category")


class MemoryRecallRequest(BaseModel):
    """Request to recall memory."""

    conversation_id: str = Field(..., description="Conversation ID")
    key: Optional[str] = Field(None, description="Specific key to recall")
    category: Optional[str] = Field(None, description="Category filter")


class MemoryRecallResponse(BaseModel):
    """Response from memory recall operation."""

    status: str = Field(..., description="Operation status")
    conversation_id: str = Field(..., description="Conversation ID")
    data: Dict[str, Any] = Field(..., description="Memory data")


class MemorySearchRequest(BaseModel):
    """Request to search memory."""

    conversation_id: str = Field(..., description="Conversation ID")
    query: str = Field(..., description="Search query")
    category: Optional[str] = Field(None, description="Category filter")


class MemorySearchResponse(BaseModel):
    """Response from memory search."""

    status: str = Field(..., description="Operation status")
    conversation_id: str = Field(..., description="Conversation ID")
    query: str = Field(..., description="Search query")
    results: List[Dict[str, Any]] = Field(..., description="Search results")
    count: int = Field(..., description="Result count")


# ============================================================================
# Conversation Management Models
# ============================================================================


class ConversationInfo(BaseModel):
    """Conversation information."""

    conversation_id: str = Field(..., description="Conversation ID")
    tier: str = Field(..., description="Current tier (active/warm/cold)")
    last_access_time: float = Field(..., description="Last access timestamp")
    access_count: int = Field(..., description="Access count")
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Conversation metadata"
    )


class ConversationListResponse(BaseModel):
    """Response for listing conversations."""

    conversations: List[ConversationInfo] = Field(
        ..., description="List of conversations"
    )
    count: int = Field(..., description="Total count")
    tier_counts: Dict[str, int] = Field(..., description="Count per tier")


class ConversationSearchRequest(BaseModel):
    """Request to search conversations."""

    query: Optional[str] = Field(None, description="Search query")
    tier: Optional[str] = Field(None, description="Filter by tier")
    limit: int = Field(default=100, description="Max results")


class ConversationRestoreRequest(BaseModel):
    """Request to restore a conversation."""

    conversation_id: str = Field(..., description="Conversation ID to restore")
    turn_number: Optional[int] = Field(None, description="Specific turn to restore")


class ConversationRestoreResponse(BaseModel):
    """Response from conversation restore."""

    status: str = Field(..., description="Operation status")
    conversation_id: str = Field(..., description="Conversation ID")
    tier: str = Field(..., description="Tier restored from")
    turn_number: int = Field(..., description="Turn number restored")


# ============================================================================
# Tier Management Models
# ============================================================================


class TierStatsResponse(BaseModel):
    """Response with tier statistics."""

    host_pool_stats: Dict[str, Any] = Field(..., description="Host pool statistics")
    conversation_tracker_stats: Dict[str, Any] = Field(
        ..., description="Conversation tracker stats"
    )
    tier_manager_stats: Dict[str, Any] = Field(
        default_factory=dict, description="Tier manager stats"
    )


class TierTransitionRequest(BaseModel):
    """Request to manually transition conversation tier."""

    conversation_id: str = Field(..., description="Conversation ID")
    target_tier: str = Field(
        ..., description="Target tier (active/warm/cold/archived)"
    )


class TierTransitionResponse(BaseModel):
    """Response from tier transition."""

    status: str = Field(..., description="Operation status")
    conversation_id: str = Field(..., description="Conversation ID")
    old_tier: str = Field(..., description="Previous tier")
    new_tier: str = Field(..., description="New tier")


class TierCleanupRequest(BaseModel):
    """Request to trigger tier cleanup."""

    force: bool = Field(
        default=False, description="Force cleanup regardless of timeouts"
    )


class TierCleanupResponse(BaseModel):
    """Response from tier cleanup."""

    status: str = Field(..., description="Operation status")
    transitions: Dict[str, int] = Field(..., description="Transitions performed")


# ============================================================================
# Agent Execution Models
# ============================================================================


class AgentRunRequest(BaseModel):
    """Request to run agent loop."""

    user_input: str = Field(..., description="User input message")
    conversation_id: Optional[str] = Field(None, description="Conversation ID")
    system_prompt: Optional[str] = Field(None, description="Optional system prompt")
    max_iterations: Optional[int] = Field(None, description="Max iterations override")


class AgentRunResponse(BaseModel):
    """Response from agent execution."""

    status: str = Field(..., description="Execution status")
    response: str = Field(..., description="Agent response")
    conversation_id: str = Field(..., description="Conversation ID")
    iterations: int = Field(..., description="Number of iterations")
    tool_calls: int = Field(..., description="Total tool calls made")
    execution_time_ms: float = Field(..., description="Total execution time")


class AgentHistoryResponse(BaseModel):
    """Response with agent conversation history."""

    conversation_id: str = Field(..., description="Conversation ID")
    messages: List[Dict[str, Any]] = Field(..., description="Conversation messages")
    message_count: int = Field(..., description="Total message count")


# ============================================================================
# Health & Monitoring Models
# ============================================================================


class HealthResponse(BaseModel):
    """Health check response."""

    status: str = Field(..., description="Health status (healthy/degraded/unhealthy)")
    agent_tools_enabled: bool = Field(..., description="Agent tools enabled")
    memory_tiers_enabled: bool = Field(..., description="Memory tiers enabled")
    snapshot_persistence_enabled: bool = Field(
        ..., description="Snapshot persistence enabled"
    )
    components: Dict[str, bool] = Field(
        ..., description="Component health status"
    )


class StatsResponse(BaseModel):
    """General statistics response."""

    tool_execution_stats: Dict[str, Any] = Field(
        ..., description="Tool execution statistics"
    )
    tier_stats: Optional[Dict[str, Any]] = Field(
        None, description="Tier statistics if available"
    )
    registered_tools: int = Field(..., description="Number of registered tools")


# ============================================================================
# Error Response
# ============================================================================


class ErrorResponse(BaseModel):
    """Standard error response."""

    error: str = Field(..., description="Error message")
    detail: Optional[str] = Field(None, description="Detailed error information")
    status_code: int = Field(..., description="HTTP status code")
