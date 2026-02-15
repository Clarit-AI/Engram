"""
Built-in tools for the agent framework.

Provides essential tools for memory management, calculation,
and other common agent tasks.
"""

import ast
import json
import logging
import math
import operator
import re
from typing import Any, Dict, Optional

from sglang.srt.agents.tool_registry import Tool, ToolParameter, ToolParameterType

logger = logging.getLogger(__name__)


# ============================================================================
# Safe Expression Evaluator
# ============================================================================


class SafeExpressionEvaluator(ast.NodeVisitor):
    """
    Safe mathematical expression evaluator using AST.

    Avoids the security risks of eval() by only allowing
    whitelisted mathematical operations and functions.
    """

    # Allowed binary operations
    BINARY_OPS = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.FloorDiv: operator.floordiv,
        ast.Mod: operator.mod,
        ast.Pow: operator.pow,
    }

    # Allowed unary operations
    UNARY_OPS = {
        ast.UAdd: operator.pos,
        ast.USub: operator.neg,
    }

    # Allowed comparison operations
    COMPARE_OPS = {
        ast.Lt: operator.lt,
        ast.LtE: operator.le,
        ast.Gt: operator.gt,
        ast.GtE: operator.ge,
        ast.Eq: operator.eq,
        ast.NotEq: operator.ne,
    }

    # Allowed functions
    FUNCTIONS = {
        "abs": abs,
        "round": round,
        "min": min,
        "max": max,
        "pow": pow,
        "sqrt": math.sqrt,
        "sin": math.sin,
        "cos": math.cos,
        "tan": math.tan,
        "asin": math.asin,
        "acos": math.acos,
        "atan": math.atan,
        "log": math.log,
        "log10": math.log10,
        "log2": math.log2,
        "exp": math.exp,
        "floor": math.floor,
        "ceil": math.ceil,
        "degrees": math.degrees,
        "radians": math.radians,
    }

    # Allowed constants
    CONSTANTS = {
        "pi": math.pi,
        "e": math.e,
        "tau": math.tau,
        "inf": math.inf,
    }

    def __init__(self):
        """Initialize evaluator."""
        self.variables = {}

    def evaluate(self, expression: str) -> float:
        """
        Safely evaluate a mathematical expression.

        Args:
            expression: Mathematical expression string

        Returns:
            Evaluation result

        Raises:
            ValueError: If expression contains unsafe operations
            SyntaxError: If expression is invalid
        """
        # Parse expression to AST
        try:
            tree = ast.parse(expression, mode="eval")
        except SyntaxError as e:
            raise SyntaxError(f"Invalid expression syntax: {e}")

        # Evaluate AST
        return self.visit(tree.body)

    def visit_BinOp(self, node):
        """Handle binary operations (+, -, *, /, etc.)."""
        op_class = type(node.op)
        if op_class not in self.BINARY_OPS:
            raise ValueError(f"Unsupported operation: {op_class.__name__}")

        left = self.visit(node.left)
        right = self.visit(node.right)
        return self.BINARY_OPS[op_class](left, right)

    def visit_UnaryOp(self, node):
        """Handle unary operations (+, -)."""
        op_class = type(node.op)
        if op_class not in self.UNARY_OPS:
            raise ValueError(f"Unsupported operation: {op_class.__name__}")

        operand = self.visit(node.operand)
        return self.UNARY_OPS[op_class](operand)

    def visit_Compare(self, node):
        """Handle comparison operations (<, >, ==, etc.)."""
        left = self.visit(node.left)
        result = True

        for op, comparator in zip(node.ops, node.comparators):
            op_class = type(op)
            if op_class not in self.COMPARE_OPS:
                raise ValueError(f"Unsupported comparison: {op_class.__name__}")

            right = self.visit(comparator)
            result = result and self.COMPARE_OPS[op_class](left, right)
            left = right

        return result

    def visit_Call(self, node):
        """Handle function calls."""
        if not isinstance(node.func, ast.Name):
            raise ValueError("Only simple function calls are allowed")

        func_name = node.func.id
        if func_name not in self.FUNCTIONS:
            raise ValueError(f"Function '{func_name}' is not allowed")

        # Evaluate arguments
        args = [self.visit(arg) for arg in node.args]

        # Call function
        return self.FUNCTIONS[func_name](*args)

    def visit_Constant(self, node):
        """Handle numeric constants."""
        if isinstance(node.value, (int, float)):
            return node.value
        raise ValueError(f"Unsupported constant type: {type(node.value)}")

    def visit_Num(self, node):
        """Handle numeric literals (Python <3.8 compatibility)."""
        return node.n

    def visit_Name(self, node):
        """Handle variable/constant names."""
        name = node.id
        if name in self.CONSTANTS:
            return self.CONSTANTS[name]
        if name in self.variables:
            return self.variables[name]
        raise ValueError(f"Undefined name: '{name}'")

    def visit_Expr(self, node):
        """Handle expression nodes."""
        return self.visit(node.value)

    def generic_visit(self, node):
        """Catch-all for unsupported node types."""
        raise ValueError(
            f"Unsupported expression type: {type(node).__name__}. "
            "Only mathematical expressions are allowed."
        )


# ============================================================================
# Memory Management Tools
# ============================================================================


class MemoryStoreTool:
    """
    Tool for storing information to conversation memory.

    This tool integrates with the snapshot/tier system to persist
    important facts, user preferences, and conversation context.
    """

    @staticmethod
    def create_tool(tier_manager=None) -> Tool:
        """
        Create memory store tool.

        Args:
            tier_manager: Optional TierManager instance for integration

        Returns:
            Tool instance
        """

        def memory_store(
            key: str,
            value: str,
            category: str = "general",
            _conversation_context: Optional[Dict[str, Any]] = None,
        ) -> Dict[str, Any]:
            """
            Store information to memory.

            Args:
                key: Memory key (e.g., "user_name", "favorite_color")
                value: Value to store
                category: Category for organization (general, user_info, preferences, etc.)
                _conversation_context: Injected conversation context

            Returns:
                Status dict
            """
            conversation_id = (
                _conversation_context.get("conversation_id")
                if _conversation_context
                else "unknown"
            )

            # Store in conversation context metadata
            if _conversation_context is not None:
                if "memory" not in _conversation_context:
                    _conversation_context["memory"] = {}

                if category not in _conversation_context["memory"]:
                    _conversation_context["memory"][category] = {}

                _conversation_context["memory"][category][key] = value

            logger.info(
                f"Memory stored: conversation={conversation_id}, "
                f"category={category}, key={key}"
            )

            return {
                "status": "success",
                "message": f"Stored '{key}' in category '{category}'",
                "conversation_id": conversation_id,
            }

        return Tool(
            name="memory_store",
            description="Store information to conversation memory for later recall",
            parameters=[
                ToolParameter(
                    name="key",
                    type=ToolParameterType.STRING,
                    description="Memory key (e.g., 'user_name', 'project_goal')",
                    required=True,
                ),
                ToolParameter(
                    name="value",
                    type=ToolParameterType.STRING,
                    description="Value to store",
                    required=True,
                ),
                ToolParameter(
                    name="category",
                    type=ToolParameterType.STRING,
                    description="Category for organization (default: general)",
                    required=False,
                    default="general",
                    enum=["general", "user_info", "preferences", "facts", "tasks"],
                ),
            ],
            function=memory_store,
            requires_conversation_context=True,
            metadata={"tags": ["memory", "storage"]},
        )


class MemoryRecallTool:
    """
    Tool for recalling information from conversation memory.
    """

    @staticmethod
    def create_tool(tier_manager=None) -> Tool:
        """
        Create memory recall tool.

        Args:
            tier_manager: Optional TierManager instance for integration

        Returns:
            Tool instance
        """

        def memory_recall(
            key: Optional[str] = None,
            category: Optional[str] = None,
            _conversation_context: Optional[Dict[str, Any]] = None,
        ) -> Dict[str, Any]:
            """
            Recall information from memory.

            Args:
                key: Specific key to recall (None = list all in category)
                category: Category to search (None = search all)
                _conversation_context: Injected conversation context

            Returns:
                Memory data
            """
            conversation_id = (
                _conversation_context.get("conversation_id")
                if _conversation_context
                else "unknown"
            )

            if _conversation_context is None or "memory" not in _conversation_context:
                return {
                    "status": "not_found",
                    "message": "No memory data available",
                    "conversation_id": conversation_id,
                }

            memory = _conversation_context["memory"]

            # Recall specific key in category
            if key is not None and category is not None:
                if category in memory and key in memory[category]:
                    return {
                        "status": "success",
                        "key": key,
                        "value": memory[category][key],
                        "category": category,
                        "conversation_id": conversation_id,
                    }
                else:
                    return {
                        "status": "not_found",
                        "message": f"Key '{key}' not found in category '{category}'",
                        "conversation_id": conversation_id,
                    }

            # List all in category
            elif category is not None:
                if category in memory:
                    return {
                        "status": "success",
                        "category": category,
                        "data": memory[category],
                        "conversation_id": conversation_id,
                    }
                else:
                    return {
                        "status": "not_found",
                        "message": f"Category '{category}' not found",
                        "conversation_id": conversation_id,
                    }

            # List all memory
            else:
                return {
                    "status": "success",
                    "data": memory,
                    "conversation_id": conversation_id,
                }

        return Tool(
            name="memory_recall",
            description="Recall information from conversation memory",
            parameters=[
                ToolParameter(
                    name="key",
                    type=ToolParameterType.STRING,
                    description="Specific key to recall (None = list all)",
                    required=False,
                ),
                ToolParameter(
                    name="category",
                    type=ToolParameterType.STRING,
                    description="Category to search (None = search all)",
                    required=False,
                ),
            ],
            function=memory_recall,
            requires_conversation_context=True,
            metadata={"tags": ["memory", "retrieval"]},
        )


class MemorySearchTool:
    """
    Tool for searching memory by keyword or pattern.
    """

    @staticmethod
    def create_tool() -> Tool:
        """Create memory search tool."""

        def memory_search(
            query: str,
            category: Optional[str] = None,
            _conversation_context: Optional[Dict[str, Any]] = None,
        ) -> Dict[str, Any]:
            """
            Search memory by keyword or pattern.

            Args:
                query: Search query (substring match)
                category: Optional category filter
                _conversation_context: Injected conversation context

            Returns:
                Search results
            """
            conversation_id = (
                _conversation_context.get("conversation_id")
                if _conversation_context
                else "unknown"
            )

            if _conversation_context is None or "memory" not in _conversation_context:
                return {
                    "status": "not_found",
                    "message": "No memory data available",
                    "results": [],
                }

            memory = _conversation_context["memory"]
            results = []

            # Search in specified category or all categories
            categories_to_search = (
                [category] if category else list(memory.keys())
            )

            for cat in categories_to_search:
                if cat not in memory:
                    continue

                for key, value in memory[cat].items():
                    # Search in key or value
                    if query.lower() in key.lower() or query.lower() in str(
                        value
                    ).lower():
                        results.append(
                            {
                                "category": cat,
                                "key": key,
                                "value": value,
                            }
                        )

            return {
                "status": "success",
                "query": query,
                "results": results,
                "count": len(results),
                "conversation_id": conversation_id,
            }

        return Tool(
            name="memory_search",
            description="Search conversation memory by keyword or pattern",
            parameters=[
                ToolParameter(
                    name="query",
                    type=ToolParameterType.STRING,
                    description="Search query",
                    required=True,
                ),
                ToolParameter(
                    name="category",
                    type=ToolParameterType.STRING,
                    description="Optional category filter",
                    required=False,
                ),
            ],
            function=memory_search,
            requires_conversation_context=True,
            metadata={"tags": ["memory", "search"]},
        )


# ============================================================================
# Utility Tools
# ============================================================================


class CalculatorTool:
    """
    Tool for performing mathematical calculations.
    """

    @staticmethod
    def create_tool() -> Tool:
        """Create calculator tool."""

        def calculator(expression: str) -> Dict[str, Any]:
            """
            Evaluate a mathematical expression safely.

            Uses AST-based evaluation to prevent code injection attacks.
            Only whitelisted mathematical operations are allowed.

            Args:
                expression: Math expression (e.g., "2 + 2", "sqrt(16)")

            Returns:
                Calculation result

            Example:
                >>> calculator("2 + 2")
                {'status': 'success', 'expression': '2 + 2', 'result': 4.0}

                >>> calculator("sqrt(16) + 2 * 3")
                {'status': 'success', 'expression': 'sqrt(16) + 2 * 3', 'result': 10.0}
            """
            try:
                # Use safe AST-based evaluator instead of eval()
                evaluator = SafeExpressionEvaluator()
                result = evaluator.evaluate(expression)

                return {
                    "status": "success",
                    "expression": expression,
                    "result": float(result),
                }

            except (ValueError, SyntaxError, ZeroDivisionError, TypeError) as e:
                logger.warning(f"Calculator error for '{expression}': {e}")
                return {
                    "status": "error",
                    "expression": expression,
                    "error": str(e),
                }
            except Exception as e:
                logger.error(f"Unexpected calculator error for '{expression}': {e}", exc_info=True)
                return {
                    "status": "error",
                    "expression": expression,
                    "error": f"Calculation failed: {str(e)}",
                }

        return Tool(
            name="calculator",
            description="Perform mathematical calculations (basic arithmetic, sqrt, sin, cos, etc.)",
            parameters=[
                ToolParameter(
                    name="expression",
                    type=ToolParameterType.STRING,
                    description="Mathematical expression to evaluate (e.g., '2 + 2', 'sqrt(16)')",
                    required=True,
                ),
            ],
            function=calculator,
            metadata={"tags": ["utility", "math"]},
        )


# ============================================================================
# Helper Functions
# ============================================================================


def register_builtin_tools(tool_registry, tier_manager=None):
    """
    Register all built-in tools to a registry.

    Args:
        tool_registry: ToolRegistry instance
        tier_manager: Optional TierManager for memory tools

    Returns:
        Number of tools registered
    """
    tools = [
        MemoryStoreTool.create_tool(tier_manager),
        MemoryRecallTool.create_tool(tier_manager),
        MemorySearchTool.create_tool(),
        CalculatorTool.create_tool(),
    ]

    for tool in tools:
        tool_registry.register(tool)

    logger.info(f"Registered {len(tools)} built-in tools")

    return len(tools)
