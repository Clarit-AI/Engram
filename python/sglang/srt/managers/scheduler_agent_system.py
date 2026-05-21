# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
# ENGRAM_MODIFIED — Agent tool framework lifecycle, extracted from scheduler.py
"""Agent tool framework lifecycle helpers.

Module-level functions extracted from ``scheduler.py`` as part of the
scheduler-decomposition extraction port (2026-05-20).

These helpers operate on a Scheduler-shaped object passed as the first
argument and mutate its attributes in place. The Scheduler continues to
own the agent attributes (``tool_registry``, ``tool_parser``,
``tool_executor``).

This is logic relocation, not attribute-ownership migration. See
``docs/upstream-sync/scheduler-decomposition-port.md`` §7 R5 for the
re-scoping decision.

The module-level logger is bound to ``sglang.srt.managers.scheduler``
so log records continue to appear under the Scheduler logger name
post-extraction (semantics-preserving).
"""

import logging

logger = logging.getLogger("sglang.srt.managers.scheduler")


def init_agent_system(scheduler):
    """
    Initialize the agent framework for tool-calling support.

    This system enables:
    - Tool registration and management
    - Tool call parsing from model outputs
    - Safe tool execution
    - Agent loop for multi-turn tool workflows

    **Backward Compatibility**: This method only activates when
    --enable-agent-tools is set. Otherwise, it's a no-op.
    """
    server_args = scheduler.server_args

    # Initialize agent components as None (default)
    scheduler.tool_registry = None
    scheduler.tool_parser = None
    scheduler.tool_executor = None

    # Only initialize if agent tools enabled
    if not getattr(server_args, "enable_agent_tools", False):
        logger.info("Agent tools disabled (standard mode)")
        return

    logger.info("Initializing agent tool framework...")

    # Import agent modules (lazy import)
    try:
        from sglang.srt.agents import (
            ToolCallParser,
            ToolExecutionEngine,
            ToolRegistry,
        )
        from sglang.srt.agents.builtin_tools import register_builtin_tools
    except ImportError as e:
        logger.error(f"Failed to import agent modules: {e}")
        logger.warning("Agent system will be disabled")
        return

    try:
        # Create tool registry
        scheduler.tool_registry = ToolRegistry()

        # Create tool parser
        scheduler.tool_parser = ToolCallParser()

        # Create tool execution engine
        scheduler.tool_executor = ToolExecutionEngine(
            tool_registry=scheduler.tool_registry,
            default_timeout=getattr(server_args, "agent_tool_timeout", 30.0),
            enable_sandboxing=True,
        )

        # Register built-in tools
        tier_manager = getattr(scheduler, "tier_manager", None)
        num_tools = register_builtin_tools(scheduler.tool_registry, tier_manager)

        logger.info(
            f"Agent system initialized successfully: {num_tools} built-in tools registered"
        )

    except Exception as e:
        logger.error(f"Failed to initialize agent system: {e}", exc_info=True)
        logger.warning("Agent system will be disabled")
        scheduler.tool_registry = None
        scheduler.tool_parser = None
        scheduler.tool_executor = None
