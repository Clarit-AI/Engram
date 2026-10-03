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
# ENGRAM_MODIFIED — Snapshot system lifecycle (init + startup restore), extracted from scheduler.py
"""Snapshot system lifecycle helpers.

Module-level functions extracted from ``scheduler.py`` as part of the
scheduler-decomposition extraction port (2026-05-20).

These helpers operate on a Scheduler-shaped object passed as the first
argument and mutate its attributes in place. The Scheduler continues to
own the ~10 snapshot/tier attributes (``snapshot_manager``,
``snapshot_hook_manager``, ``snapshot_policy``, ``host_pool``,
``conversation_tracker``, ``tier_manager``, ``state_health_monitor``,
``_health_check_counter``, ``pending_restore_registry``,
``pending_restore_aliases``).

This is logic relocation, not attribute-ownership migration. The
attribute-ownership migration (to upstream's ``scheduler_components``
classes) happens in the post-sync re-homing PR. See
``docs/upstream-sync/scheduler-decomposition-port.md`` §7 R5 for the
re-scoping decision.

The module-level logger is bound to ``sglang.srt.managers.scheduler``
so that log records continue to appear under the Scheduler logger name
post-extraction (semantics-preserving).
"""

import logging
import time
from collections import OrderedDict
from pathlib import Path

logger = logging.getLogger("sglang.srt.managers.scheduler")


def init_snapshot_system(scheduler):
    """
    Initialize the Mamba state snapshot system for stateful inference.

    This system enables:
    - Saving Mamba SSM states to disk
    - Restoring states on server restart
    - Managing snapshot retention and branching

    **Backward Compatibility**: This method only activates when
    --enable-snapshot-persistence is set. Otherwise, it's a no-op.
    """
    server_args = scheduler.server_args

    # Initialize snapshot components as None (default)
    scheduler.snapshot_manager = None
    scheduler.snapshot_hook_manager = None
    scheduler.snapshot_policy = None

    # Initialize tier management components as None (default)
    scheduler.host_pool = None
    scheduler.conversation_tracker = None
    scheduler.tier_manager = None

    # Tier 2: State health monitoring
    scheduler.state_health_monitor = None
    scheduler._health_check_counter = {}  # conversation_id → snapshot count

    # Pending-restore registry: canonical rid → loaded snapshot state staged
    # for the next incoming request. Populated by handle_restore_snapshot when
    # the originating Req has already finished; consumed by
    # _maybe_hydrate_from_pending_restore during request creation. Bounded by
    # PENDING_RESTORE_REGISTRY_MAX on logical (canonical) entries with LRU
    # eviction. conversation_id aliases are tracked separately in
    # pending_restore_aliases so replacement and eviction remain atomic.
    scheduler.pending_restore_registry = OrderedDict()
    scheduler.pending_restore_aliases = {}  # alias (conv_id) → canonical rid

    # Only initialize if snapshot persistence is enabled
    if not server_args.enable_snapshot_persistence:
        logger.info("Snapshot persistence disabled (standard mode)")
        return

    # Check if model supports Mamba/hybrid architecture
    # HybridReqToTokenPool has mamba_pool attribute
    mamba_pool = getattr(scheduler.req_to_token_pool, "mamba_pool", None)

    if mamba_pool is None:
        logger.warning(
            "Snapshot persistence enabled but model doesn't have Mamba pool. "
            "This is a standard transformer model - snapshot system will not activate. "
            "Snapshot features only work with Mamba/hybrid models."
        )
        return

    # Determine snapshot directory
    snapshot_dir = server_args.snapshot_dir
    if snapshot_dir is None:
        snapshot_dir = "./sglang_snapshots"

    logger.info(
        f"Initializing Mamba snapshot system: dir={snapshot_dir}, "
        f"retention={server_args.snapshot_retention_count}, "
        f"policy={server_args.snapshot_trigger_policy}"
    )

    # Import snapshot modules (lazy import)
    try:
        from sglang.srt.snapshot import (
            MambaSnapshotManager,
            SnapshotHookManager,
            SnapshotRetentionPolicy,
        )
        from sglang.srt.snapshot.snapshot_policy import (
            SnapshotRetentionConfig,
            SnapshotTriggerPolicy,
        )
    except ImportError as e:
        logger.error(f"Failed to import snapshot modules: {e}")
        logger.warning("Snapshot system will be disabled")
        return

    # Create snapshot manager
    try:
        scheduler.snapshot_manager = MambaSnapshotManager(Path(snapshot_dir))
    except Exception as e:
        logger.error(f"Failed to create snapshot manager: {e}")
        logger.warning("Snapshot system will be disabled")
        return

    # Create hook manager
    scheduler.snapshot_hook_manager = SnapshotHookManager(enabled=True)

    # Create retention policy
    try:
        trigger_policy = SnapshotTriggerPolicy(server_args.snapshot_trigger_policy)
    except ValueError:
        logger.warning(
            f"Invalid snapshot trigger policy: {server_args.snapshot_trigger_policy}, "
            "defaulting to EVERY_TURN"
        )
        trigger_policy = SnapshotTriggerPolicy.EVERY_TURN

    retention_config = SnapshotRetentionConfig(
        max_snapshots_per_conversation=server_args.snapshot_retention_count,
        snapshot_trigger_policy=trigger_policy,
        snapshot_every_n_turns=server_args.snapshot_every_n_turns,
        min_snapshot_interval_seconds=server_args.snapshot_min_interval_seconds,
        keep_named_branches=server_args.snapshot_keep_named_branches,
    )

    scheduler.snapshot_policy = SnapshotRetentionPolicy(
        scheduler.snapshot_manager, retention_config
    )

    # Initialize tier management system (Phase 2.5)
    if server_args.enable_memory_tiers:
        try:
            from sglang.srt.snapshot import (
                ConversationTracker,
                MambaHostPool,
                TierManager,
            )

            # Create host memory pool
            scheduler.host_pool = MambaHostPool(
                max_conversations=server_args.max_warm_conversations,
                max_memory_gb=server_args.max_warm_memory_gb,
                enable_cross_session_refs=server_args.enable_cross_session_refs,
            )

            # Create conversation tracker
            scheduler.conversation_tracker = ConversationTracker(
                active_timeout=server_args.conversation_active_timeout,
                warm_timeout=server_args.conversation_warm_timeout,
                cold_retention=server_args.conversation_cold_retention,
            )

            # Create tier manager
            scheduler.tier_manager = TierManager(
                conversation_tracker=scheduler.conversation_tracker,
                host_pool=scheduler.host_pool,
                snapshot_manager=scheduler.snapshot_manager,
                enable_background_cleanup=server_args.enable_tier_background_cleanup,
                cleanup_interval=server_args.tier_cleanup_interval,
                model_path=server_args.model_path,
            )

            logger.info(
                "Memory tier system initialized: "
                f"max_warm={server_args.max_warm_conversations}, "
                f"max_warm_memory={server_args.max_warm_memory_gb}GB, "
                f"cross_session_refs={server_args.enable_cross_session_refs}"
            )

        except Exception as e:
            logger.error(f"Failed to initialize tier management: {e}", exc_info=True)
            logger.warning(
                "Tier management disabled, falling back to disk-only snapshots"
            )
            scheduler.host_pool = None
            scheduler.conversation_tracker = None
            scheduler.tier_manager = None

    # Tier 2: Initialize state health monitor if interval > 0
    if getattr(server_args, "snapshot_health_check_interval", 0) > 0:
        from sglang.srt.snapshot.state_health import StateHealthMonitor

        scheduler.state_health_monitor = StateHealthMonitor()
        logger.info(
            "State health monitoring enabled (interval=%d, policy=%s)",
            server_args.snapshot_health_check_interval,
            server_args.snapshot_health_failure_policy,
        )

    # Register post-forward hook
    def post_forward_snapshot_callback(trigger):
        """Callback triggered after forward pass to save snapshot."""
        req = trigger.req
        mamba_pool = trigger.mamba_pool
        turn_number = trigger.turn_number

        # Get conversation ID (use rid as fallback)
        conversation_id = getattr(req, "conversation_id", None)
        if conversation_id is None:
            conversation_id = req.rid

        # Check if we should snapshot based on policy
        additional_context = trigger.additional_context or {}
        if not scheduler.snapshot_policy.should_snapshot(
            req, turn_number, conversation_id, additional_context
        ):
            return

        # Extract Mamba state from pool
        try:
            conv_states, temporal_states = (
                scheduler.snapshot_manager.extract_state_from_pool(
                    mamba_pool, req.mamba_pool_idx
                )
            )
        except Exception as e:
            logger.error(f"Failed to extract Mamba state for snapshot: {e}")
            return

        # Tier 2: Health monitoring (if enabled)
        if scheduler.state_health_monitor is not None:
            try:
                conv_count = scheduler._health_check_counter.get(conversation_id, 0) + 1
                scheduler._health_check_counter[conversation_id] = conv_count
                interval = scheduler.server_args.snapshot_health_check_interval
                if conv_count % interval == 0:
                    health = scheduler.state_health_monitor.check_state_health(
                        conversation_id, conv_states, temporal_states, turn_number
                    )
                    if not health.healthy:
                        policy = scheduler.server_args.snapshot_health_failure_policy
                        logger.warning(
                            "State health anomaly for %s at turn %d: "
                            "%d anomalous layers. Policy: %s",
                            conversation_id,
                            turn_number,
                            len(health.anomalous_layers),
                            policy,
                        )
                        if policy == "skip_snapshot":
                            return
                        # else: log_and_continue — fall through to save
            except Exception:
                logger.error(
                    "State health check failed for %s, continuing with save",
                    conversation_id,
                    exc_info=True,
                )

        # Build metadata
        from sglang.srt.snapshot import MambaSnapshotMetadata

        # Get layer config from model
        layer_config = {
            "num_layers": mamba_pool.num_mamba_layers,
            "model_type": "hybrid" if hasattr(req, "mamba_pool_idx") else "mamba",
        }

        metadata = MambaSnapshotMetadata(
            conversation_id=conversation_id,
            turn_number=turn_number,
            timestamp=time.time(),
            token_count=(
                len(req.fill_ids)
                if hasattr(req, "fill_ids") and req.fill_ids is not None
                else 0
            ),
            model_name=scheduler.server_args.model_path,
            mamba_pool_idx=int(req.mamba_pool_idx),
            req_pool_idx=int(req.req_pool_idx),
            layer_config=layer_config,
            fill_ids=(
                (
                    req.fill_ids.tolist()
                    if hasattr(req.fill_ids, "tolist")
                    else list(req.fill_ids)
                )
                if hasattr(req, "fill_ids") and req.fill_ids is not None
                else None
            ),
        )

        # Save snapshot (use tier manager if available, else direct to disk)
        try:
            if scheduler.tier_manager is not None:
                # Save to warm tier (host RAM)
                success = scheduler.tier_manager.save_to_warm_tier(
                    conversation_id,
                    conv_states,
                    temporal_states,
                    metadata.to_dict(),
                )

                if success:
                    request_id = getattr(req, "rid", None)
                    if request_id and request_id != conversation_id:
                        scheduler.tier_manager.host_pool.alias_state(
                            request_id, conversation_id
                        )
                    scheduler.snapshot_policy.mark_snapshot_taken(conversation_id)
                    logger.debug(
                        f"Snapshot saved to WARM tier: conversation={conversation_id}, "
                        f"turn={turn_number}"
                    )
                else:
                    logger.warning(
                        f"Failed to save to WARM tier, falling back to COLD: {conversation_id}"
                    )
                    # Fall back to saving directly to disk
                    scheduler.snapshot_manager.save_snapshot(
                        conv_states, temporal_states, metadata
                    )
                    scheduler.snapshot_policy.mark_snapshot_taken(conversation_id)

            else:
                # No tier manager, save directly to disk
                scheduler.snapshot_manager.save_snapshot(
                    conv_states, temporal_states, metadata
                )
                scheduler.snapshot_policy.mark_snapshot_taken(conversation_id)

                logger.debug(
                    f"Snapshot saved to COLD tier (disk): conversation={conversation_id}, "
                    f"turn={turn_number}"
                )

            # Prune old snapshots if needed
            scheduler.snapshot_policy.prune_old_snapshots(conversation_id)

        except Exception as e:
            logger.error(f"Failed to save snapshot: {e}", exc_info=True)

    scheduler.snapshot_hook_manager.register_post_forward_hook(
        post_forward_snapshot_callback
    )

    logger.info("Snapshot system initialized successfully")

    # Restore snapshots if auto-restore is enabled
    if server_args.snapshot_auto_restore:
        scheduler.restore_snapshots_on_startup()


def restore_snapshots_on_startup(scheduler):
    """
    Restore the latest snapshots for all conversations on server startup.

    This pre-loads the latest snapshot for each conversation into the WARM
    tier so restart-time continuity remains a server-level capability.
    """
    if scheduler.snapshot_manager is None:
        return

    if scheduler.tier_manager is None or scheduler.conversation_tracker is None:
        logger.warning(
            "Snapshot auto-restore requires memory tiers; skipping startup restore"
        )
        return

    logger.info("Attempting to restore snapshots from previous sessions...")
    from sglang.srt.snapshot.tier_manager import (
        restore_latest_snapshots_to_warm_tier,
    )

    restore_latest_snapshots_to_warm_tier(
        snapshot_manager=scheduler.snapshot_manager,
        tier_manager=scheduler.tier_manager,
        restore_logger=logger,
    )
