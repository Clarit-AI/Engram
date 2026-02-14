"""
Snapshot hook integration for scheduler and cache systems.

This module provides the hook manager that integrates state snapshotting
into SGLang's request lifecycle. Hooks are triggered at specific points
during inference to capture and persist Mamba states.

**Backward Compatibility**: Hooks only activate when snapshot persistence
is enabled via --enable-snapshot-persistence flag. Otherwise, this module
has zero runtime overhead.
"""

import logging
import threading
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class SnapshotTrigger:
    """
    Context information passed to snapshot hooks.

    Attributes:
        req: Request object (from schedule_batch.py)
        mamba_pool: MambaPool instance
        req_pool: ReqToTokenPool instance
        turn_number: Current turn number in conversation
        trigger_reason: Why snapshot was triggered (e.g., "post_forward", "pre_evict")
    """

    req: Any  # Type is Req from schedule_batch.py
    mamba_pool: Any  # Type is MambaPool from memory_pool.py
    req_pool: Any  # Type is ReqToTokenPool from memory_pool.py
    turn_number: int
    trigger_reason: str
    additional_context: Optional[Dict[str, Any]] = None


class SnapshotHookManager:
    """
    Manages snapshot hooks throughout the request lifecycle.

    This class provides a centralized registry for snapshot callbacks
    that get triggered at various points during inference:

    - Post-forward: After token generation, before sending to user
    - Pre-eviction: Before Mamba state is evicted from cache
    - On-demand: Manual snapshot trigger via API

    Thread Safety:
        This class is thread-safe. Hooks can be registered/triggered
        from different threads.

    Usage:
        # In scheduler initialization
        hook_manager = SnapshotHookManager()
        hook_manager.register_post_forward_hook(save_snapshot_callback)

        # In scheduler.process_batch_result()
        if self.args.enable_snapshot_persistence:
            hook_manager.trigger_post_forward(req, mamba_pool, ...)
    """

    def __init__(self, enabled: bool = True):
        """
        Initialize hook manager.

        Args:
            enabled: Whether hooks are enabled. If False, all trigger
                     methods become no-ops for zero overhead.
        """
        self.enabled = enabled
        self._post_forward_hooks: List[Callable[[SnapshotTrigger], None]] = []
        self._pre_eviction_hooks: List[Callable[[SnapshotTrigger], None]] = []
        self._on_demand_hooks: List[Callable[[SnapshotTrigger], None]] = []
        self._lock = threading.Lock()

        logger.info(f"SnapshotHookManager initialized (enabled={enabled})")

    def register_post_forward_hook(
        self, callback: Callable[[SnapshotTrigger], None]
    ) -> None:
        """
        Register a callback to run after forward pass.

        The callback receives a SnapshotTrigger with current request state
        and should save the snapshot if appropriate.

        Args:
            callback: Function that takes SnapshotTrigger and returns None

        Example:
            def save_snapshot(trigger: SnapshotTrigger):
                conv_states, temporal = snapshot_mgr.extract_state_from_pool(
                    trigger.mamba_pool, trigger.req.mamba_pool_idx
                )
                metadata = MambaSnapshotMetadata(...)
                snapshot_mgr.save_snapshot(conv_states, temporal, metadata)

            hook_manager.register_post_forward_hook(save_snapshot)
        """
        with self._lock:
            self._post_forward_hooks.append(callback)
            logger.debug(
                f"Registered post-forward hook: {callback.__name__} "
                f"(total: {len(self._post_forward_hooks)})"
            )

    def register_pre_eviction_hook(
        self, callback: Callable[[SnapshotTrigger], None]
    ) -> None:
        """
        Register a callback to run before Mamba state eviction.

        This allows saving state before it's freed from memory.

        Args:
            callback: Function that takes SnapshotTrigger and returns None
        """
        with self._lock:
            self._pre_eviction_hooks.append(callback)
            logger.debug(
                f"Registered pre-eviction hook: {callback.__name__} "
                f"(total: {len(self._pre_eviction_hooks)})"
            )

    def register_on_demand_hook(
        self, callback: Callable[[SnapshotTrigger], None]
    ) -> None:
        """
        Register a callback for manual/on-demand snapshots.

        These hooks are triggered via trigger_on_demand() API call.

        Args:
            callback: Function that takes SnapshotTrigger and returns None
        """
        with self._lock:
            self._on_demand_hooks.append(callback)
            logger.debug(
                f"Registered on-demand hook: {callback.__name__} "
                f"(total: {len(self._on_demand_hooks)})"
            )

    def trigger_post_forward(
        self,
        req: Any,
        mamba_pool: Any,
        req_pool: Any,
        turn_number: int,
        additional_context: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Trigger all post-forward hooks.

        This should be called from scheduler after updating req.output_ids
        with the newly generated tokens.

        Args:
            req: Request object
            mamba_pool: MambaPool instance
            req_pool: ReqToTokenPool instance
            turn_number: Current turn number
            additional_context: Optional additional context dict
        """
        if not self.enabled:
            return

        # Skip if no Mamba pool (standard transformer model)
        if mamba_pool is None or not hasattr(req, "mamba_pool_idx"):
            return

        # Skip if mamba_pool_idx not allocated
        if req.mamba_pool_idx is None:
            return

        trigger = SnapshotTrigger(
            req=req,
            mamba_pool=mamba_pool,
            req_pool=req_pool,
            turn_number=turn_number,
            trigger_reason="post_forward",
            additional_context=additional_context,
        )

        with self._lock:
            hooks = list(self._post_forward_hooks)  # Copy to release lock quickly

        for hook in hooks:
            try:
                hook(trigger)
            except Exception as e:
                logger.error(
                    f"Post-forward hook {hook.__name__} failed: {e}",
                    exc_info=True,
                )
                # Continue with other hooks even if one fails

    def trigger_pre_eviction(
        self,
        req: Any,
        mamba_pool: Any,
        req_pool: Any,
        turn_number: int,
        additional_context: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Trigger all pre-eviction hooks.

        This should be called before evicting Mamba state from the pool.

        Args:
            req: Request object
            mamba_pool: MambaPool instance
            req_pool: ReqToTokenPool instance
            turn_number: Current turn number
            additional_context: Optional additional context dict
        """
        if not self.enabled:
            return

        if mamba_pool is None or not hasattr(req, "mamba_pool_idx"):
            return

        if req.mamba_pool_idx is None:
            return

        trigger = SnapshotTrigger(
            req=req,
            mamba_pool=mamba_pool,
            req_pool=req_pool,
            turn_number=turn_number,
            trigger_reason="pre_eviction",
            additional_context=additional_context,
        )

        with self._lock:
            hooks = list(self._pre_eviction_hooks)

        for hook in hooks:
            try:
                hook(trigger)
            except Exception as e:
                logger.error(
                    f"Pre-eviction hook {hook.__name__} failed: {e}",
                    exc_info=True,
                )

    def trigger_on_demand(
        self,
        req: Any,
        mamba_pool: Any,
        req_pool: Any,
        turn_number: int,
        additional_context: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Trigger all on-demand hooks.

        This can be called manually via API to force a snapshot.

        Args:
            req: Request object
            mamba_pool: MambaPool instance
            req_pool: ReqToTokenPool instance
            turn_number: Current turn number
            additional_context: Optional additional context dict
        """
        if not self.enabled:
            return

        if mamba_pool is None or not hasattr(req, "mamba_pool_idx"):
            return

        if req.mamba_pool_idx is None:
            return

        trigger = SnapshotTrigger(
            req=req,
            mamba_pool=mamba_pool,
            req_pool=req_pool,
            turn_number=turn_number,
            trigger_reason="on_demand",
            additional_context=additional_context,
        )

        with self._lock:
            hooks = list(self._on_demand_hooks)

        for hook in hooks:
            try:
                hook(trigger)
            except Exception as e:
                logger.error(
                    f"On-demand hook {hook.__name__} failed: {e}",
                    exc_info=True,
                )

    def clear_hooks(self, hook_type: Optional[str] = None) -> None:
        """
        Clear registered hooks.

        Args:
            hook_type: Type of hooks to clear ("post_forward", "pre_eviction",
                       "on_demand", or None for all)
        """
        with self._lock:
            if hook_type is None or hook_type == "post_forward":
                self._post_forward_hooks.clear()
            if hook_type is None or hook_type == "pre_eviction":
                self._pre_eviction_hooks.clear()
            if hook_type is None or hook_type == "on_demand":
                self._on_demand_hooks.clear()

        logger.info(f"Cleared hooks: {hook_type or 'all'}")

    def get_hook_count(self) -> Dict[str, int]:
        """
        Get count of registered hooks by type.

        Returns:
            Dict with counts: {"post_forward": n, "pre_eviction": n, "on_demand": n}
        """
        with self._lock:
            return {
                "post_forward": len(self._post_forward_hooks),
                "pre_eviction": len(self._pre_eviction_hooks),
                "on_demand": len(self._on_demand_hooks),
            }
