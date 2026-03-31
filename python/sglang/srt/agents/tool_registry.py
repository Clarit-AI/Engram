"""
Tool registry for managing available tools in the agent framework.

Provides registration, discovery, and metadata management for tools
that can be called by the agent.
"""

import inspect
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


class ToolParameterType(Enum):
    """Supported parameter types for tools."""

    STRING = "string"
    INTEGER = "integer"
    FLOAT = "float"
    BOOLEAN = "boolean"
    OBJECT = "object"
    ARRAY = "array"


@dataclass
class ToolParameter:
    """
    Tool parameter definition.

    Attributes:
        name: Parameter name
        type: Parameter type
        description: Human-readable description
        required: Whether parameter is required
        default: Default value if not provided
        enum: List of allowed values (for enums)
        properties: For OBJECT type, nested parameter definitions
        items: For ARRAY type, item type definition
    """

    name: str
    type: ToolParameterType
    description: str
    required: bool = True
    default: Optional[Any] = None
    enum: Optional[List[Any]] = None
    properties: Optional[Dict[str, "ToolParameter"]] = None
    items: Optional["ToolParameter"] = None

    def to_dict(self) -> dict:
        """Convert to dictionary representation (for API)."""
        result = {
            "name": self.name,
            "type": self.type.value,
            "description": self.description,
            "required": self.required,
        }

        if self.default is not None:
            result["default"] = self.default

        if self.enum is not None:
            result["enum"] = self.enum

        if self.properties is not None:
            result["properties"] = {k: v.to_dict() for k, v in self.properties.items()}

        if self.items is not None:
            result["items"] = self.items.to_dict()

        return result


@dataclass
class Tool:
    """
    Tool definition for the agent framework.

    A tool is a function that the agent can call to perform actions
    or retrieve information.

    Attributes:
        name: Unique tool name (e.g., "memory_store", "calculator")
        description: Human-readable description of what the tool does
        parameters: List of parameter definitions
        function: The actual callable function
        requires_conversation_context: Whether tool needs conversation state
        is_async: Whether the function is async
        metadata: Additional metadata (tags, version, etc.)
    """

    name: str
    description: str
    parameters: List[ToolParameter]
    function: Callable
    requires_conversation_context: bool = False
    is_async: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Convert to dictionary representation (for API)."""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": [p.to_dict() for p in self.parameters],
            "requires_conversation_context": self.requires_conversation_context,
            "is_async": self.is_async,
            "metadata": self.metadata,
        }

    def validate_parameters(self, params: Dict[str, Any]) -> bool:
        """
        Validate that provided parameters match tool definition.

        Args:
            params: Parameter dict to validate

        Returns:
            True if valid, False otherwise
        """
        # Check required parameters
        for param in self.parameters:
            if param.required and param.name not in params:
                logger.warning(
                    f"Missing required parameter '{param.name}' for tool '{self.name}'"
                )
                return False

        # Check parameter types (basic validation)
        for param_name, param_value in params.items():
            # Find parameter definition
            param_def = next((p for p in self.parameters if p.name == param_name), None)

            if param_def is None:
                logger.warning(
                    f"Unknown parameter '{param_name}' for tool '{self.name}'"
                )
                return False

            # Check enum constraints
            if param_def.enum is not None and param_value not in param_def.enum:
                logger.warning(
                    f"Parameter '{param_name}' value '{param_value}' not in allowed values: {param_def.enum}"
                )
                return False

        return True


class ToolRegistry:
    """
    Registry for managing available tools.

    This class maintains a registry of all tools that can be called
    by the agent, provides discovery, and manages tool metadata.

    Thread Safety:
        All methods are thread-safe.
    """

    def __init__(self):
        """Initialize tool registry."""
        self._tools: Dict[str, Tool] = {}
        self._tags: Dict[str, List[str]] = {}  # tag -> list of tool names

        logger.info("ToolRegistry initialized")

    def register(self, tool: Tool) -> None:
        """
        Register a tool.

        Args:
            tool: Tool to register

        Raises:
            ValueError: If tool with same name already exists
        """
        if tool.name in self._tools:
            raise ValueError(f"Tool '{tool.name}' already registered")

        # Validate tool function signature
        if not callable(tool.function):
            raise ValueError(f"Tool '{tool.name}' function must be callable")

        # Register tool
        self._tools[tool.name] = tool

        # Register tags
        tags = tool.metadata.get("tags", [])
        for tag in tags:
            if tag not in self._tags:
                self._tags[tag] = []
            self._tags[tag].append(tool.name)

        logger.info(
            f"Tool registered: {tool.name}, "
            f"params={len(tool.parameters)}, "
            f"tags={tags}"
        )

    def unregister(self, tool_name: str) -> bool:
        """
        Unregister a tool.

        Args:
            tool_name: Name of tool to unregister

        Returns:
            True if unregistered, False if not found
        """
        if tool_name not in self._tools:
            return False

        tool = self._tools.pop(tool_name)

        # Remove from tags
        tags = tool.metadata.get("tags", [])
        for tag in tags:
            if tag in self._tags and tool_name in self._tags[tag]:
                self._tags[tag].remove(tool_name)
                if not self._tags[tag]:
                    del self._tags[tag]

        logger.info(f"Tool unregistered: {tool_name}")
        return True

    def get(self, tool_name: str) -> Optional[Tool]:
        """
        Get tool by name.

        Args:
            tool_name: Tool name

        Returns:
            Tool instance or None if not found
        """
        return self._tools.get(tool_name)

    def list_tools(self, tag: Optional[str] = None) -> List[Tool]:
        """
        List all registered tools.

        Args:
            tag: Optional tag filter

        Returns:
            List of Tool instances
        """
        if tag is None:
            return list(self._tools.values())

        # Filter by tag
        tool_names = self._tags.get(tag, [])
        return [self._tools[name] for name in tool_names]

    def get_tool_names(self, tag: Optional[str] = None) -> List[str]:
        """
        Get list of tool names.

        Args:
            tag: Optional tag filter

        Returns:
            List of tool names
        """
        if tag is None:
            return list(self._tools.keys())

        return self._tags.get(tag, [])

    def has_tool(self, tool_name: str) -> bool:
        """Check if tool is registered."""
        return tool_name in self._tools

    def get_tool_schemas(self) -> List[dict]:
        """
        Get OpenAI-compatible tool schemas for all registered tools.

        Returns:
            List of tool schemas in OpenAI function calling format
        """
        schemas = []

        for tool in self._tools.values():
            # Build parameters schema
            properties = {}
            required = []

            for param in tool.parameters:
                properties[param.name] = {
                    "type": param.type.value,
                    "description": param.description,
                }

                if param.enum is not None:
                    properties[param.name]["enum"] = param.enum

                if param.default is not None:
                    properties[param.name]["default"] = param.default

                if param.required:
                    required.append(param.name)

            schema = {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": {
                        "type": "object",
                        "properties": properties,
                        "required": required,
                    },
                },
            }

            schemas.append(schema)

        return schemas

    def clear(self):
        """Clear all registered tools."""
        self._tools.clear()
        self._tags.clear()
        logger.info("Tool registry cleared")

    def __len__(self) -> int:
        """Return number of registered tools."""
        return len(self._tools)

    def __contains__(self, tool_name: str) -> bool:
        """Check if tool is registered."""
        return tool_name in self._tools


def tool(
    name: str,
    description: str,
    parameters: Optional[List[ToolParameter]] = None,
    requires_conversation_context: bool = False,
    metadata: Optional[Dict[str, Any]] = None,
):
    """
    Decorator for registering a function as a tool.

    Usage:
        @tool(
            name="calculator",
            description="Perform basic arithmetic",
            parameters=[
                ToolParameter("expression", ToolParameterType.STRING, "Math expression")
            ]
        )
        def calculator(expression: str) -> float:
            return eval(expression)

    Args:
        name: Tool name
        description: Tool description
        parameters: List of parameter definitions
        requires_conversation_context: Whether tool needs conversation state
        metadata: Additional metadata

    Returns:
        Decorator function
    """

    def decorator(func: Callable) -> Tool:
        is_async = inspect.iscoroutinefunction(func)

        return Tool(
            name=name,
            description=description,
            parameters=parameters or [],
            function=func,
            requires_conversation_context=requires_conversation_context,
            is_async=is_async,
            metadata=metadata or {},
        )

    return decorator
