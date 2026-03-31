"""
Unit tests for ToolRegistry.

Tests tool registration, discovery, and schema generation.
"""

import pytest

from sglang.srt.agents.tool_registry import (
    Tool,
    ToolParameter,
    ToolParameterType,
    ToolRegistry,
)


class TestToolRegistry:
    """Test ToolRegistry functionality."""

    def test_registry_initialization(self):
        """Test registry initialization."""
        registry = ToolRegistry()

        assert len(registry) == 0
        assert registry.get_tool_names() == []

    def test_register_tool(self):
        """Test registering a tool."""
        registry = ToolRegistry()

        def test_func(x: int) -> int:
            return x * 2

        tool = Tool(
            name="double",
            description="Double a number",
            parameters=[ToolParameter("x", ToolParameterType.INTEGER, "Input number")],
            function=test_func,
        )

        registry.register(tool)

        assert len(registry) == 1
        assert "double" in registry
        assert registry.has_tool("double")

    def test_register_duplicate_fails(self):
        """Test that registering duplicate tool name fails."""
        registry = ToolRegistry()

        def test_func():
            pass

        tool1 = Tool("test", "Test tool 1", [], test_func)
        tool2 = Tool("test", "Test tool 2", [], test_func)

        registry.register(tool1)

        with pytest.raises(ValueError, match="already registered"):
            registry.register(tool2)

    def test_get_tool(self):
        """Test getting a tool by name."""
        registry = ToolRegistry()

        def test_func():
            return "test"

        tool = Tool("test", "Test tool", [], test_func)
        registry.register(tool)

        retrieved = registry.get("test")

        assert retrieved is not None
        assert retrieved.name == "test"
        assert retrieved.function() == "test"

    def test_get_nonexistent_tool(self):
        """Test getting non-existent tool returns None."""
        registry = ToolRegistry()

        assert registry.get("nonexistent") is None

    def test_unregister_tool(self):
        """Test unregistering a tool."""
        registry = ToolRegistry()

        tool = Tool("test", "Test tool", [], lambda: None)
        registry.register(tool)

        assert registry.has_tool("test")

        success = registry.unregister("test")

        assert success is True
        assert not registry.has_tool("test")
        assert len(registry) == 0

    def test_unregister_nonexistent(self):
        """Test unregistering non-existent tool."""
        registry = ToolRegistry()

        success = registry.unregister("nonexistent")

        assert success is False

    def test_list_tools(self):
        """Test listing all tools."""
        registry = ToolRegistry()

        for i in range(3):
            tool = Tool(f"tool_{i}", f"Tool {i}", [], lambda: None)
            registry.register(tool)

        tools = registry.list_tools()

        assert len(tools) == 3
        assert all(isinstance(t, Tool) for t in tools)

    def test_get_tool_names(self):
        """Test getting tool names."""
        registry = ToolRegistry()

        names = ["alpha", "beta", "gamma"]
        for name in names:
            tool = Tool(name, f"{name} tool", [], lambda: None)
            registry.register(tool)

        retrieved_names = sorted(registry.get_tool_names())

        assert retrieved_names == sorted(names)

    def test_tags(self):
        """Test filtering tools by tags."""
        registry = ToolRegistry()

        tool1 = Tool(
            "tool1",
            "Tool 1",
            [],
            lambda: None,
            metadata={"tags": ["math", "utility"]},
        )
        tool2 = Tool(
            "tool2",
            "Tool 2",
            [],
            lambda: None,
            metadata={"tags": ["memory"]},
        )
        tool3 = Tool(
            "tool3",
            "Tool 3",
            [],
            lambda: None,
            metadata={"tags": ["math"]},
        )

        registry.register(tool1)
        registry.register(tool2)
        registry.register(tool3)

        # Get tools by tag
        math_tools = registry.list_tools(tag="math")
        memory_tools = registry.list_tools(tag="memory")

        assert len(math_tools) == 2
        assert len(memory_tools) == 1

        math_tool_names = sorted([t.name for t in math_tools])
        assert math_tool_names == ["tool1", "tool3"]

    def test_parameter_validation(self):
        """Test parameter validation."""

        def test_func(x: int, y: str):
            return f"{x}: {y}"

        tool = Tool(
            name="test",
            description="Test",
            parameters=[
                ToolParameter("x", ToolParameterType.INTEGER, "X param", required=True),
                ToolParameter("y", ToolParameterType.STRING, "Y param", required=True),
            ],
            function=test_func,
        )

        # Valid parameters
        assert tool.validate_parameters({"x": 1, "y": "test"}) is True

        # Missing required parameter
        assert tool.validate_parameters({"x": 1}) is False

        # Unknown parameter
        assert tool.validate_parameters({"x": 1, "y": "test", "z": "extra"}) is False

    def test_parameter_enum_validation(self):
        """Test enum parameter validation."""
        tool = Tool(
            name="test",
            description="Test",
            parameters=[
                ToolParameter(
                    "color",
                    ToolParameterType.STRING,
                    "Color",
                    enum=["red", "green", "blue"],
                )
            ],
            function=lambda color: color,
        )

        # Valid enum value
        assert tool.validate_parameters({"color": "red"}) is True

        # Invalid enum value
        assert tool.validate_parameters({"color": "yellow"}) is False

    def test_get_tool_schemas(self):
        """Test OpenAI-compatible schema generation."""
        registry = ToolRegistry()

        tool = Tool(
            name="calculator",
            description="Perform calculation",
            parameters=[
                ToolParameter(
                    "expression",
                    ToolParameterType.STRING,
                    "Math expression",
                    required=True,
                )
            ],
            function=lambda expression: eval(expression),
        )

        registry.register(tool)

        schemas = registry.get_tool_schemas()

        assert len(schemas) == 1
        schema = schemas[0]

        assert schema["type"] == "function"
        assert schema["function"]["name"] == "calculator"
        assert schema["function"]["description"] == "Perform calculation"
        assert "expression" in schema["function"]["parameters"]["properties"]
        assert "expression" in schema["function"]["parameters"]["required"]

    def test_clear_registry(self):
        """Test clearing registry."""
        registry = ToolRegistry()

        for i in range(3):
            tool = Tool(f"tool_{i}", "Tool", [], lambda: None)
            registry.register(tool)

        assert len(registry) == 3

        registry.clear()

        assert len(registry) == 0
        assert registry.get_tool_names() == []


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
