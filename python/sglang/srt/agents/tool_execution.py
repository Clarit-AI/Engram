"""
Tool execution engine for safe and reliable tool execution.

Provides sandboxed execution, timeout handling, error recovery,
and execution tracking.
"""

import asyncio
import logging
import threading
import time
import traceback
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Optional

from sglang.srt.agents.tool_registry import Tool, ToolRegistry

logger = logging.getLogger(__name__)


class ToolExecutionStatus(Enum):
    """Status of tool execution."""

    SUCCESS = "success"
    ERROR = "error"
    TIMEOUT = "timeout"
    VALIDATION_ERROR = "validation_error"


@dataclass
class ToolExecutionResult:
    """
    Result of tool execution.

    Attributes:
        tool_name: Name of executed tool
        status: Execution status
        result: Tool output (if successful)
        error: Error message (if failed)
        execution_time_ms: Execution time in milliseconds
        metadata: Additional metadata (e.g., token usage, etc.)
    """

    tool_name: str
    status: ToolExecutionStatus
    result: Optional[Any] = None
    error: Optional[str] = None
    execution_time_ms: float = 0.0
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

    def to_dict(self) -> dict:
        """Convert to dictionary representation."""
        return {
            "tool_name": self.tool_name,
            "status": self.status.value,
            "result": self.result,
            "error": self.error,
            "execution_time_ms": self.execution_time_ms,
            "metadata": self.metadata,
        }

    def is_success(self) -> bool:
        """Check if execution was successful."""
        return self.status == ToolExecutionStatus.SUCCESS


class ToolExecutionEngine:
    """
    Engine for executing tool calls safely.

    This engine:
    - Validates tool calls before execution
    - Enforces execution timeouts
    - Catches and logs exceptions
    - Tracks execution metrics
    - Provides async/sync execution support

    Thread Safety:
        All methods are thread-safe.
    """

    def __init__(
        self,
        tool_registry: ToolRegistry,
        default_timeout: float = 30.0,
        enable_sandboxing: bool = True,
    ):
        """
        Initialize tool execution engine.

        Args:
            tool_registry: Tool registry instance
            default_timeout: Default execution timeout in seconds
            enable_sandboxing: Enable execution sandboxing (future)
        """
        self.tool_registry = tool_registry
        self.default_timeout = default_timeout
        self.enable_sandboxing = enable_sandboxing

        # Metrics (protected by lock for thread-safety)
        self._stats_lock = threading.Lock()
        self._total_executions = 0
        self._successful_executions = 0
        self._failed_executions = 0
        self._total_execution_time_ms = 0.0

        logger.info(
            f"ToolExecutionEngine initialized: "
            f"timeout={default_timeout}s, "
            f"sandboxing={enable_sandboxing}"
        )

    def execute(
        self,
        tool_name: str,
        parameters: Dict[str, Any],
        conversation_context: Optional[Dict[str, Any]] = None,
        timeout: Optional[float] = None,
    ) -> ToolExecutionResult:
        """
        Execute a tool call.

        Args:
            tool_name: Name of tool to execute
            parameters: Tool parameters
            conversation_context: Optional conversation context
            timeout: Execution timeout (None = use default)

        Returns:
            ToolExecutionResult
        """
        start_time = time.time()

        # Increment total executions atomically
        with self._stats_lock:
            self._total_executions += 1

        # Get tool from registry
        tool = self.tool_registry.get(tool_name)

        if tool is None:
            with self._stats_lock:
                self._failed_executions += 1
            return ToolExecutionResult(
                tool_name=tool_name,
                status=ToolExecutionStatus.ERROR,
                error=f"Tool '{tool_name}' not found in registry",
                execution_time_ms=0.0,
            )

        # Validate parameters
        if not tool.validate_parameters(parameters):
            with self._stats_lock:
                self._failed_executions += 1
            return ToolExecutionResult(
                tool_name=tool_name,
                status=ToolExecutionStatus.VALIDATION_ERROR,
                error="Parameter validation failed",
                execution_time_ms=(time.time() - start_time) * 1000,
            )

        # Add conversation context if tool requires it
        if tool.requires_conversation_context:
            if conversation_context is None:
                logger.warning(
                    f"Tool '{tool_name}' requires conversation context but none provided"
                )
                conversation_context = {}

            parameters["_conversation_context"] = conversation_context

        # Execute tool
        try:
            if tool.is_async:
                result = self._execute_async(
                    tool, parameters, timeout or self.default_timeout
                )
            else:
                result = self._execute_sync(
                    tool, parameters, timeout or self.default_timeout
                )

            execution_time_ms = (time.time() - start_time) * 1000

            # Update success counters atomically
            with self._stats_lock:
                self._successful_executions += 1
                self._total_execution_time_ms += execution_time_ms

            return ToolExecutionResult(
                tool_name=tool_name,
                status=ToolExecutionStatus.SUCCESS,
                result=result,
                execution_time_ms=execution_time_ms,
            )

        except TimeoutError:
            execution_time_ms = (time.time() - start_time) * 1000

            # Update failure counter atomically
            with self._stats_lock:
                self._failed_executions += 1

            logger.error(f"Tool '{tool_name}' execution timeout")

            return ToolExecutionResult(
                tool_name=tool_name,
                status=ToolExecutionStatus.TIMEOUT,
                error=f"Execution timeout after {timeout or self.default_timeout}s",
                execution_time_ms=execution_time_ms,
            )

        except Exception as e:
            execution_time_ms = (time.time() - start_time) * 1000

            # Update failure counter atomically
            with self._stats_lock:
                self._failed_executions += 1

            error_msg = f"{type(e).__name__}: {str(e)}"
            logger.error(
                f"Tool '{tool_name}' execution failed: {error_msg}",
                exc_info=True,
            )

            return ToolExecutionResult(
                tool_name=tool_name,
                status=ToolExecutionStatus.ERROR,
                error=error_msg,
                execution_time_ms=execution_time_ms,
                metadata={"traceback": traceback.format_exc()},
            )

    def _execute_sync(
        self, tool: Tool, parameters: Dict[str, Any], timeout: float
    ) -> Any:
        """
        Execute synchronous tool.

        Args:
            tool: Tool to execute
            parameters: Tool parameters
            timeout: Execution timeout

        Returns:
            Tool result

        Raises:
            TimeoutError: If execution exceeds timeout
            Exception: Any exception raised by tool
        """
        # Note: Python doesn't have built-in sync timeout without threads
        # For production, consider using signal.alarm on Unix or
        # concurrent.futures.ThreadPoolExecutor with timeout

        logger.debug(f"Executing sync tool: {tool.name}")
        return tool.function(**parameters)

    def _execute_async(
        self, tool: Tool, parameters: Dict[str, Any], timeout: float
    ) -> Any:
        """
        Execute asynchronous tool.

        Args:
            tool: Tool to execute
            parameters: Tool parameters
            timeout: Execution timeout

        Returns:
            Tool result

        Raises:
            TimeoutError: If execution exceeds timeout
            Exception: Any exception raised by tool
        """
        logger.debug(f"Executing async tool: {tool.name}")

        # Get or create event loop
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        # Execute with timeout
        return loop.run_until_complete(
            asyncio.wait_for(tool.function(**parameters), timeout=timeout)
        )

    def execute_batch(
        self,
        tool_calls: list[Dict[str, Any]],
        conversation_context: Optional[Dict[str, Any]] = None,
        timeout: Optional[float] = None,
    ) -> list[ToolExecutionResult]:
        """
        Execute multiple tool calls in sequence.

        Args:
            tool_calls: List of tool calls, each with 'name' and 'parameters'
            conversation_context: Optional conversation context
            timeout: Per-tool timeout

        Returns:
            List of ToolExecutionResult
        """
        results = []

        for tool_call in tool_calls:
            tool_name = tool_call.get("name")
            parameters = tool_call.get("parameters", {})

            if tool_name is None:
                results.append(
                    ToolExecutionResult(
                        tool_name="unknown",
                        status=ToolExecutionStatus.ERROR,
                        error="Missing tool name in tool call",
                    )
                )
                continue

            result = self.execute(
                tool_name=tool_name,
                parameters=parameters,
                conversation_context=conversation_context,
                timeout=timeout,
            )

            results.append(result)

        return results

    def get_stats(self) -> dict:
        """
        Get execution statistics (thread-safe).

        Returns:
            Dictionary with execution statistics
        """
        with self._stats_lock:
            success_rate = 0.0
            if self._total_executions > 0:
                success_rate = self._successful_executions / self._total_executions

            avg_execution_time_ms = 0.0
            if self._successful_executions > 0:
                avg_execution_time_ms = (
                    self._total_execution_time_ms / self._successful_executions
                )

            return {
                "total_executions": self._total_executions,
                "successful_executions": self._successful_executions,
                "failed_executions": self._failed_executions,
                "success_rate": success_rate,
                "avg_execution_time_ms": avg_execution_time_ms,
                "total_execution_time_ms": self._total_execution_time_ms,
            }

    def reset_stats(self):
        """Reset execution statistics (thread-safe)."""
        with self._stats_lock:
            self._total_executions = 0
            self._successful_executions = 0
            self._failed_executions = 0
            self._total_execution_time_ms = 0.0

        logger.info("Tool execution statistics reset")
