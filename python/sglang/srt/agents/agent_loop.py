"""
Agent loop for tool-calling workflow.

Orchestrates the conversation flow with tool execution:
1. User input → Model generation
2. Parse tool calls from output
3. Execute tools
4. Feed results back to model
5. Continue until no more tool calls or max iterations
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from sglang.srt.agents.tool_execution import (
    ToolExecutionEngine,
    ToolExecutionResult,
)
from sglang.srt.agents.tool_parser import ToolCallParser
from sglang.srt.agents.tool_registry import ToolRegistry

logger = logging.getLogger(__name__)


@dataclass
class AgentMessage:
    """
    Message in agent conversation.

    Attributes:
        role: Message role (user, assistant, tool)
        content: Message content
        tool_calls: Optional tool calls in message
        tool_results: Optional tool execution results
        metadata: Additional metadata
    """

    role: str  # user, assistant, tool
    content: str
    tool_calls: Optional[List[Dict[str, Any]]] = None
    tool_results: Optional[List[ToolExecutionResult]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Convert to dictionary representation."""
        result = {
            "role": self.role,
            "content": self.content,
            "metadata": self.metadata,
        }

        if self.tool_calls:
            result["tool_calls"] = self.tool_calls

        if self.tool_results:
            result["tool_results"] = [r.to_dict() for r in self.tool_results]

        return result


@dataclass
class AgentLoopConfig:
    """
    Configuration for agent loop.

    Attributes:
        max_iterations: Maximum tool-calling iterations
        max_tool_calls_per_iteration: Max tool calls per iteration
        enable_parallel_tools: Execute tools in parallel (future)
        tool_timeout: Tool execution timeout in seconds
        conversation_context: Persistent conversation context
    """

    max_iterations: int = 10
    max_tool_calls_per_iteration: int = 5
    enable_parallel_tools: bool = False
    tool_timeout: float = 30.0
    conversation_context: Dict[str, Any] = field(default_factory=dict)


class AgentLoop:
    """
    Agent execution loop with tool-calling support.

    This class orchestrates the conversation flow between the model
    and tools, handling:
    - Tool call parsing from model outputs
    - Tool execution
    - Result formatting and feedback to model
    - Multi-turn tool-calling workflows
    - Conversation context management

    **Usage:**
        agent = AgentLoop(
            tool_registry=registry,
            tool_parser=parser,
            tool_executor=executor,
            model_generate_fn=lambda msgs: model.generate(msgs),
        )

        result = agent.run("What is 2+2? Store the result in memory.")

    Thread Safety:
        Methods are not thread-safe. Use separate instances per conversation.
    """

    def __init__(
        self,
        tool_registry: ToolRegistry,
        tool_parser: ToolCallParser,
        tool_executor: ToolExecutionEngine,
        model_generate_fn,
        config: Optional[AgentLoopConfig] = None,
    ):
        """
        Initialize agent loop.

        Args:
            tool_registry: Tool registry instance
            tool_parser: Tool call parser
            tool_executor: Tool execution engine
            model_generate_fn: Function to generate model responses
                               Signature: fn(messages: List[dict]) -> str
            config: Agent loop configuration
        """
        self.tool_registry = tool_registry
        self.tool_parser = tool_parser
        self.tool_executor = tool_executor
        self.model_generate_fn = model_generate_fn
        self.config = config or AgentLoopConfig()

        # Conversation history
        self.messages: List[AgentMessage] = []

        # Metrics
        self._total_iterations = 0
        self._total_tool_calls = 0
        self._total_model_calls = 0

        logger.info(
            f"AgentLoop initialized: "
            f"max_iterations={self.config.max_iterations}, "
            f"max_tool_calls={self.config.max_tool_calls_per_iteration}"
        )

    def run(
        self,
        user_input: str,
        system_prompt: Optional[str] = None,
        conversation_id: Optional[str] = None,
    ) -> str:
        """
        Run agent loop for a user input.

        This method:
        1. Adds user message to history
        2. Generates model response
        3. Parses tool calls
        4. Executes tools
        5. Feeds results back to model
        6. Repeats until no more tool calls or max iterations

        Args:
            user_input: User's input message
            system_prompt: Optional system prompt
            conversation_id: Optional conversation ID for context

        Returns:
            Final assistant response (without tool calls)
        """
        start_time = time.time()

        # Update conversation context
        if conversation_id:
            self.config.conversation_context["conversation_id"] = conversation_id

        # Add system prompt if provided (only once)
        if system_prompt and not self.messages:
            self.messages.append(
                AgentMessage(
                    role="system",
                    content=system_prompt,
                )
            )

        # Add user message
        self.messages.append(
            AgentMessage(
                role="user",
                content=user_input,
            )
        )

        # Main loop
        iteration = 0
        final_response = ""

        while iteration < self.config.max_iterations:
            iteration += 1
            self._total_iterations += 1

            logger.debug(f"Agent iteration {iteration}/{self.config.max_iterations}")

            # Generate model response
            model_response = self._generate_model_response()
            self._total_model_calls += 1

            # Parse tool calls
            tool_calls = self.tool_parser.extract_tool_calls_from_response(
                model_response
            )

            if not tool_calls:
                # No tool calls - we're done
                final_response = model_response
                logger.debug(f"No tool calls found, finishing. Response: {model_response[:100]}")
                break

            # Limit number of tool calls
            if len(tool_calls) > self.config.max_tool_calls_per_iteration:
                logger.warning(
                    f"Too many tool calls ({len(tool_calls)}), "
                    f"limiting to {self.config.max_tool_calls_per_iteration}"
                )
                tool_calls = tool_calls[: self.config.max_tool_calls_per_iteration]

            # Add assistant message with tool calls
            tool_calls_dict = [tc.to_dict() for tc in tool_calls]
            self.messages.append(
                AgentMessage(
                    role="assistant",
                    content=model_response,
                    tool_calls=tool_calls_dict,
                )
            )

            # Execute tools
            tool_results = self._execute_tools(tool_calls)
            self._total_tool_calls += len(tool_results)

            # Add tool results message
            self.messages.append(
                AgentMessage(
                    role="tool",
                    content=self._format_tool_results(tool_results),
                    tool_results=tool_results,
                )
            )

        # If we hit max iterations without final response
        if not final_response:
            final_response = (
                f"Maximum iterations ({self.config.max_iterations}) reached. "
                "Unable to complete request."
            )
            logger.error(
                "Agent loop exhausted max iterations (%d) without producing a final response",
                self.config.max_iterations,
            )

        # Add final assistant message
        self.messages.append(
            AgentMessage(
                role="assistant",
                content=final_response,
            )
        )

        execution_time = time.time() - start_time
        logger.info(
            f"Agent loop completed: iterations={iteration}, "
            f"tool_calls={self._total_tool_calls}, "
            f"time={execution_time:.2f}s"
        )

        return final_response

    def _generate_model_response(self) -> str:
        """
        Generate model response using the model_generate_fn.

        Returns:
            Model response text
        """
        # Convert messages to format expected by model
        formatted_messages = [msg.to_dict() for msg in self.messages]

        # Call model
        response = self.model_generate_fn(formatted_messages)

        return response

    def _execute_tools(self, tool_calls: List) -> List[ToolExecutionResult]:
        """
        Execute tool calls.

        Args:
            tool_calls: List of ParsedToolCall objects

        Returns:
            List of ToolExecutionResult
        """
        results = []

        for tool_call in tool_calls:
            logger.debug(f"Executing tool: {tool_call.name}")

            result = self.tool_executor.execute(
                tool_name=tool_call.name,
                parameters=tool_call.parameters,
                conversation_context=self.config.conversation_context,
                timeout=self.config.tool_timeout,
            )

            results.append(result)

        return results

    def _format_tool_results(self, tool_results: List[ToolExecutionResult]) -> str:
        """
        Format tool results for model consumption.

        Args:
            tool_results: List of tool execution results

        Returns:
            Formatted string
        """
        if not tool_results:
            return "No tool results."

        lines = ["Tool execution results:"]

        for i, result in enumerate(tool_results, 1):
            lines.append(f"\n{i}. Tool: {result.tool_name}")
            lines.append(f"   Status: {result.status.value}")

            if result.is_success():
                lines.append(f"   Result: {result.result}")
            else:
                lines.append(f"   Error: {result.error}")

            lines.append(
                f"   Execution time: {result.execution_time_ms:.2f}ms"
            )

        return "\n".join(lines)

    def get_conversation_history(self) -> List[AgentMessage]:
        """Get full conversation history."""
        return self.messages.copy()

    def clear_history(self):
        """Clear conversation history."""
        self.messages.clear()
        logger.info("Conversation history cleared")

    def get_stats(self) -> dict:
        """Get agent loop statistics."""
        return {
            "total_iterations": self._total_iterations,
            "total_tool_calls": self._total_tool_calls,
            "total_model_calls": self._total_model_calls,
            "message_count": len(self.messages),
        }

    def reset_stats(self):
        """Reset statistics."""
        self._total_iterations = 0
        self._total_tool_calls = 0
        self._total_model_calls = 0
        logger.info("Agent loop statistics reset")
