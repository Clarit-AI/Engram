"""
Agent framework for stateful Mamba models.

This module provides tool-calling capabilities and agent loops for
Mamba/hybrid models with persistent state management.

**Phase 3**: Tool-calling framework with memory integration

Key Components:
    - ToolRegistry: Register and manage available tools
    - ToolExecutionEngine: Execute tool calls safely with sandboxing
    - ToolCallParser: Parse tool calls from model outputs
    - AgentLoop: Main execution loop for tool-calling agents
    - Built-in tools: Memory, search, calculation, etc.

**Backward Compatibility**: This module is opt-in. Standard inference
is unaffected. Agent features only activate when explicitly enabled.
"""

from sglang.srt.agents.tool_execution import ToolExecutionEngine, ToolExecutionResult
from sglang.srt.agents.tool_registry import Tool, ToolParameter, ToolRegistry

try:
    from sglang.srt.agents.agent_loop import AgentLoop
    from sglang.srt.agents.builtin_tools import (
        CalculatorTool,
        MemoryRecallTool,
        MemoryStoreTool,
    )
    from sglang.srt.agents.tool_parser import ToolCallParser
except ImportError:
    # Allow partial imports during development
    pass

__all__ = [
    # Core framework
    "Tool",
    "ToolParameter",
    "ToolRegistry",
    "ToolExecutionEngine",
    "ToolExecutionResult",
    "ToolCallParser",
    "AgentLoop",
    # Built-in tools
    "MemoryStoreTool",
    "MemoryRecallTool",
    "CalculatorTool",
]
