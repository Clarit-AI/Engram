"""
Unit tests for ToolExecutionEngine.

Tests tool execution, timeout handling, and error recovery.
"""

import time

import pytest

from sglang.srt.agents.tool_execution import ToolExecutionEngine, ToolExecutionStatus
from sglang.srt.agents.tool_registry import (
    Tool,
    ToolParameter,
    ToolParameterType,
    ToolRegistry,
)


class TestToolExecutionEngine:
    """Test ToolExecutionEngine functionality."""

    def test_engine_initialization(self):
        """Test engine initialization."""
        registry = ToolRegistry()
        engine = ToolExecutionEngine(registry, default_timeout=10.0)

        assert engine.default_timeout == 10.0
        assert engine._total_executions == 0

    def test_execute_simple_tool(self):
        """Test executing a simple tool."""
        registry = ToolRegistry()
        engine = ToolExecutionEngine(registry)

        def add(a: int, b: int) -> int:
            return a + b

        tool = Tool(
            name="add",
            description="Add two numbers",
            parameters=[
                ToolParameter("a", ToolParameterType.INTEGER, "First number"),
                ToolParameter("b", ToolParameterType.INTEGER, "Second number"),
            ],
            function=add,
        )

        registry.register(tool)

        result = engine.execute("add", {"a": 2, "b": 3})

        assert result.is_success()
        assert result.result == 5
        assert result.status == ToolExecutionStatus.SUCCESS

    def test_execute_with_conversation_context(self):
        """Test tool execution with conversation context."""
        registry = ToolRegistry()
        engine = ToolExecutionEngine(registry)

        def get_user(
            _conversation_context: dict,
        ) -> str:
            return _conversation_context.get("user_id", "unknown")

        tool = Tool(
            name="get_user",
            description="Get current user",
            parameters=[],
            function=get_user,
            requires_conversation_context=True,
        )

        registry.register(tool)

        result = engine.execute(
            "get_user",
            {},
            conversation_context={"user_id": "alice"},
        )

        assert result.is_success()
        assert result.result == "alice"

    def test_execute_nonexistent_tool(self):
        """Test executing non-existent tool."""
        registry = ToolRegistry()
        engine = ToolExecutionEngine(registry)

        result = engine.execute("nonexistent", {})

        assert result.status == ToolExecutionStatus.ERROR
        assert "not found" in result.error

    def test_parameter_validation_error(self):
        """Test parameter validation failure."""
        registry = ToolRegistry()
        engine = ToolExecutionEngine(registry)

        tool = Tool(
            name="test",
            description="Test",
            parameters=[
                ToolParameter(
                    "required_param",
                    ToolParameterType.STRING,
                    "Required",
                    required=True,
                )
            ],
            function=lambda required_param: required_param,
        )

        registry.register(tool)

        # Missing required parameter
        result = engine.execute("test", {})

        assert result.status == ToolExecutionStatus.VALIDATION_ERROR

    def test_tool_execution_error(self):
        """Test tool execution error handling."""
        registry = ToolRegistry()
        engine = ToolExecutionEngine(registry)

        def failing_tool():
            raise ValueError("Intentional error")

        tool = Tool(
            name="fail",
            description="Failing tool",
            parameters=[],
            function=failing_tool,
        )

        registry.register(tool)

        result = engine.execute("fail", {})

        assert result.status == ToolExecutionStatus.ERROR
        assert "ValueError" in result.error
        assert not result.is_success()

    def test_execution_stats(self):
        """Test execution statistics tracking."""
        registry = ToolRegistry()
        engine = ToolExecutionEngine(registry)

        tool = Tool("test", "Test", [], lambda: "success")
        registry.register(tool)

        # Execute successfully
        engine.execute("test", {})
        engine.execute("test", {})

        # Execute with error
        engine.execute("nonexistent", {})

        stats = engine.get_stats()

        assert stats["total_executions"] == 3
        assert stats["successful_executions"] == 2
        assert stats["failed_executions"] == 1
        assert stats["success_rate"] == pytest.approx(2.0 / 3.0)

    def test_reset_stats(self):
        """Test resetting statistics."""
        registry = ToolRegistry()
        engine = ToolExecutionEngine(registry)

        tool = Tool("test", "Test", [], lambda: "success")
        registry.register(tool)

        engine.execute("test", {})

        engine.reset_stats()

        stats = engine.get_stats()

        assert stats["total_executions"] == 0
        assert stats["successful_executions"] == 0

    def test_execute_batch(self):
        """Test batch tool execution."""
        registry = ToolRegistry()
        engine = ToolExecutionEngine(registry)

        tool1 = Tool(
            "double",
            "Double",
            [ToolParameter("x", ToolParameterType.INTEGER, "X")],
            lambda x: x * 2,
        )
        tool2 = Tool(
            "triple",
            "Triple",
            [ToolParameter("x", ToolParameterType.INTEGER, "X")],
            lambda x: x * 3,
        )

        registry.register(tool1)
        registry.register(tool2)

        tool_calls = [
            {"name": "double", "parameters": {"x": 5}},
            {"name": "triple", "parameters": {"x": 5}},
        ]

        results = engine.execute_batch(tool_calls)

        assert len(results) == 2
        assert results[0].result == 10
        assert results[1].result == 15

    def test_execution_time_tracking(self):
        """Test execution time tracking."""
        registry = ToolRegistry()
        engine = ToolExecutionEngine(registry)

        def slow_tool():
            time.sleep(0.1)
            return "done"

        tool = Tool("slow", "Slow tool", [], slow_tool)
        registry.register(tool)

        result = engine.execute("slow", {})

        assert result.is_success()
        assert result.execution_time_ms >= 100  # At least 100ms


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
