"""
Unit tests for API models.

Tests Pydantic models for request/response validation.
"""

import pytest
from pydantic import ValidationError

from sglang.srt.agents.api.models import (
    AgentRunRequest,
    ConversationInfo,
    HealthResponse,
    MemoryRecallRequest,
    MemoryStoreRequest,
    TierTransitionRequest,
    ToolCallRequest,
    ToolInfo,
)


class TestAPIModels:
    """Test API Pydantic models."""

    def test_tool_call_request_valid(self):
        """Test valid tool call request."""
        request = ToolCallRequest(
            tool_name="calculator",
            parameters={"expression": "2+2"},
            conversation_id="conv_123",
            timeout=30.0,
        )

        assert request.tool_name == "calculator"
        assert request.parameters == {"expression": "2+2"}
        assert request.conversation_id == "conv_123"
        assert request.timeout == 30.0

    def test_tool_call_request_minimal(self):
        """Test minimal tool call request."""
        request = ToolCallRequest(tool_name="test")

        assert request.tool_name == "test"
        assert request.parameters == {}
        assert request.conversation_id is None
        assert request.timeout is None

    def test_tool_call_request_invalid(self):
        """Test invalid tool call request (missing required field)."""
        with pytest.raises(ValidationError):
            ToolCallRequest(parameters={})  # Missing tool_name

    def test_memory_store_request(self):
        """Test memory store request."""
        request = MemoryStoreRequest(
            conversation_id="conv_123",
            key="user_name",
            value="Alice",
            category="user_info",
        )

        assert request.conversation_id == "conv_123"
        assert request.key == "user_name"
        assert request.value == "Alice"
        assert request.category == "user_info"

    def test_memory_store_request_default_category(self):
        """Test memory store with default category."""
        request = MemoryStoreRequest(
            conversation_id="conv_123",
            key="test",
            value="value",
        )

        assert request.category == "general"  # Default

    def test_memory_recall_request(self):
        """Test memory recall request."""
        request = MemoryRecallRequest(
            conversation_id="conv_123",
            key="user_name",
            category="user_info",
        )

        assert request.conversation_id == "conv_123"
        assert request.key == "user_name"
        assert request.category == "user_info"

    def test_memory_recall_request_optional_fields(self):
        """Test memory recall with optional fields."""
        request = MemoryRecallRequest(conversation_id="conv_123")

        assert request.key is None
        assert request.category is None

    def test_conversation_info(self):
        """Test conversation info model."""
        info = ConversationInfo(
            conversation_id="conv_123",
            tier="active",
            last_access_time=1234567890.0,
            access_count=5,
            metadata={"user_id": "user_456"},
        )

        assert info.conversation_id == "conv_123"
        assert info.tier == "active"
        assert info.last_access_time == 1234567890.0
        assert info.access_count == 5
        assert info.metadata["user_id"] == "user_456"

    def test_tier_transition_request(self):
        """Test tier transition request."""
        request = TierTransitionRequest(
            conversation_id="conv_123",
            target_tier="warm",
        )

        assert request.conversation_id == "conv_123"
        assert request.target_tier == "warm"

    def test_health_response(self):
        """Test health response model."""
        response = HealthResponse(
            status="healthy",
            agent_tools_enabled=True,
            memory_tiers_enabled=True,
            snapshot_persistence_enabled=True,
            components={
                "tool_registry": True,
                "tool_executor": True,
                "tier_manager": True,
            },
        )

        assert response.status == "healthy"
        assert response.agent_tools_enabled is True
        assert response.components["tool_registry"] is True

    def test_agent_run_request(self):
        """Test agent run request."""
        request = AgentRunRequest(
            user_input="What is 2+2?",
            conversation_id="conv_123",
            system_prompt="You are a helpful assistant.",
            max_iterations=5,
        )

        assert request.user_input == "What is 2+2?"
        assert request.conversation_id == "conv_123"
        assert request.system_prompt == "You are a helpful assistant."
        assert request.max_iterations == 5

    def test_agent_run_request_minimal(self):
        """Test minimal agent run request."""
        request = AgentRunRequest(user_input="Hello")

        assert request.user_input == "Hello"
        assert request.conversation_id is None
        assert request.system_prompt is None
        assert request.max_iterations is None

    def test_tool_info_serialization(self):
        """Test tool info can be serialized to dict."""
        info = ToolInfo(
            name="test_tool",
            description="Test tool",
            parameters=[{"name": "param1", "type": "string"}],
            requires_conversation_context=True,
            is_async=False,
            metadata={"tags": ["test"]},
        )

        data = info.model_dump()

        assert data["name"] == "test_tool"
        assert data["description"] == "Test tool"
        assert len(data["parameters"]) == 1
        assert data["requires_conversation_context"] is True
        assert data["metadata"]["tags"] == ["test"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
