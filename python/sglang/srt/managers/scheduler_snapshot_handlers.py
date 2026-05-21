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
# ENGRAM_MODIFIED — Snapshot RPC handlers + pending-restore registry helpers, extracted from scheduler.py
"""Snapshot RPC handlers and pending-restore registry helpers.

Module-level functions extracted from ``scheduler.py`` as part of the
scheduler-decomposition extraction port (2026-05-20).

These helpers operate on a Scheduler-shaped object passed as the first
argument and mutate its attributes in place. The Scheduler continues to
own all snapshot-related attributes (snapshot_manager, tier_manager,
pending_restore_registry, pending_restore_aliases, conversation_tracker,
running_batch, waiting_queue, etc.).

This is logic relocation, not attribute-ownership migration. See
``docs/upstream-sync/scheduler-decomposition-port.md`` §7 R5 for the
re-scoping decision.

The module-level logger is bound to ``sglang.srt.managers.scheduler`` so
log records continue to appear under the Scheduler logger name
post-extraction (semantics-preserving).
"""

import logging
import time
import uuid
from typing import List, Optional

import torch

from sglang.srt.managers.scheduler_pending_restore import (
    PENDING_RESTORE_REGISTRY_MAX,
    PendingRestoreEntry,
)

logger = logging.getLogger("sglang.srt.managers.scheduler")


def _find_request_by_rid(scheduler, rid: str):
    """Find a request by its rid across scheduler-owned request queues.

    Args:
        rid: Request ID to find

    Returns:
        The request object if found, None otherwise
    """
    seen_batches = set()

    def search_batch(batch):
        if batch is None or id(batch) in seen_batches:
            return None
        seen_batches.add(id(batch))
        if batch.is_empty():
            return None
        for req in batch.reqs:
            if req.rid == rid:
                return req
        return None

    # Snapshot RPCs are processed before get_next_batch_to_run() merges the
    # previous iteration's extend batch into running_batch. During that window,
    # a live request can be present only in cur_batch/last_batch.
    for batch in (
        scheduler.running_batch,
        getattr(scheduler, "cur_batch", None),
        getattr(scheduler, "last_batch", None),
    ):
        req = search_batch(batch)
        if req is not None:
            return req

    # Search waiting_queue
    for req in scheduler.waiting_queue:
        if req.rid == rid:
            return req

    return None


def _load_snapshot_for_pending_restore(
    scheduler,
    conversation_id: str,
    turn_number: Optional[int],
    branch_name: Optional[str],
):
    """Load a snapshot for staging into the pending-restore registry.

    Prefers the WARM tier when no specific turn or branch is requested
    (latest state, host RAM, ~10-50ms). Falls back to disk for explicit
    turn/branch selection or when the WARM tier has no entry. Returns
    (conv_states, temporal_states, metadata) where metadata is a dict from
    the WARM tier or a MambaSnapshotMetadata from disk, or None when no
    snapshot is available.
    """
    if (
        scheduler.tier_manager is not None
        and turn_number is None
        and branch_name is None
        and scheduler.tier_manager.host_pool.has_state(conversation_id)
    ):
        warm = scheduler.tier_manager.restore_from_warm_tier(conversation_id)
        if warm is not None:
            return warm

    try:
        return scheduler.snapshot_manager.load_snapshot(
            conversation_id, turn_number, branch_name
        )
    except (FileNotFoundError, ValueError):
        return None


def _stage_pending_restore(
    scheduler,
    rid: str,
    conversation_id: str,
    conv_states: List[torch.Tensor],
    temporal_states: torch.Tensor,
    fill_ids: Optional[List[int]],
) -> None:
    """Stash a loaded snapshot in the pending-restore registry.

    Canonical OrderedDict keyed by rid, plus a separate alias map for
    conversation_id → canonical rid. Bounded by PENDING_RESTORE_REGISTRY_MAX
    on logical (canonical) entries with LRU eviction.

    Aliases whose value collides with an existing canonical entry evict that
    canonical first, ensuring one logical entry is never stored under
    multiple keys.
    """
    if getattr(scheduler, "pending_restore_registry", None) is None:
        return

    canonical = rid

    # Drop any old entry for this canonical key
    if canonical in scheduler.pending_restore_registry:
        scheduler.pending_restore_registry.pop(canonical)
    # Drop any aliases that previously pointed at this canonical
    for alias_key, target in list(scheduler.pending_restore_aliases.items()):
        if target == canonical:
            del scheduler.pending_restore_aliases[alias_key]

    # Case B: canonical (rid) was previously an alias for some other
    # canonical. Evict that other canonical + all aliases targeting it
    # so the new canonical insertion doesn't leave a stale entry
    # reachable via direct rid lookup on that other canonical's key.
    if canonical in scheduler.pending_restore_aliases:
        other_canonical = scheduler.pending_restore_aliases[canonical]
        if (
            other_canonical != canonical
            and other_canonical in scheduler.pending_restore_registry
        ):
            scheduler.pending_restore_registry.pop(other_canonical)
        for alias_key, target in list(scheduler.pending_restore_aliases.items()):
            if target == other_canonical:
                del scheduler.pending_restore_aliases[alias_key]

    # Insert canonical
    entry = PendingRestoreEntry(
        conv_states=conv_states,
        temporal_states=temporal_states,
        fill_ids=list(fill_ids) if fill_ids is not None else None,
        conversation_id=conversation_id,
        timestamp=time.time(),
    )
    scheduler.pending_restore_registry[canonical] = entry

    # Map conversation_id alias if distinct from canonical
    if conversation_id and conversation_id != canonical:
        # If conv_id was previously a canonical key for a different entry,
        # evict that entry first
        if conversation_id in scheduler.pending_restore_registry:
            scheduler.pending_restore_registry.pop(conversation_id)
            for alias_key, target in list(scheduler.pending_restore_aliases.items()):
                if target == conversation_id:
                    del scheduler.pending_restore_aliases[alias_key]

        # Finding 3: conv_id was previously an alias for some other
        # canonical. Evict that other canonical + all aliases targeting
        # it so the stale entry isn't left reachable via direct rid
        # lookup on that old canonical's key.
        old_canonical = scheduler.pending_restore_aliases.get(conversation_id)
        if old_canonical is not None and old_canonical != canonical:
            if old_canonical in scheduler.pending_restore_registry:
                scheduler.pending_restore_registry.pop(old_canonical)
            for alias_key, target in list(scheduler.pending_restore_aliases.items()):
                if target == old_canonical:
                    del scheduler.pending_restore_aliases[alias_key]

        scheduler.pending_restore_aliases[conversation_id] = canonical

    # Eviction: bound applies to canonical (logical) entries
    while len(scheduler.pending_restore_registry) > PENDING_RESTORE_REGISTRY_MAX:
        evicted_canonical, _ = scheduler.pending_restore_registry.popitem(last=False)
        for alias_key, target in list(scheduler.pending_restore_aliases.items()):
            if target == evicted_canonical:
                del scheduler.pending_restore_aliases[alias_key]
        logger.debug(
            "pending-restore registry: evicted oldest canonical=%s " "(capacity=%d)",
            evicted_canonical,
            PENDING_RESTORE_REGISTRY_MAX,
        )


def _clear_pending_restore(scheduler, rid: str, conversation_id: Optional[str]) -> None:
    """Remove any pending-restore entry for (rid, conversation_id).

    Called after a live /restore_snapshot succeeds (path-B-with-live-req) to
    prevent _maybe_hydrate_from_pending_restore from later resurrecting stale
    state for a follow-up request with the same rid or conv_id.
    """
    registry = getattr(scheduler, "pending_restore_registry", None)
    aliases = getattr(scheduler, "pending_restore_aliases", None)
    if not registry:
        return

    # Resolve canonical via rid first, then via conversation_id alias
    canonical = rid if rid in registry else None
    if canonical is None and conversation_id and aliases:
        canonical = aliases.get(conversation_id)
    if canonical is None or canonical not in registry:
        return

    del registry[canonical]
    if aliases:
        for alias_key, target in list(aliases.items()):
            if target == canonical:
                del aliases[alias_key]


def _maybe_hydrate_from_pending_restore(scheduler, req) -> Optional[bool]:
    """Attach restored Mamba state to ``req`` if its rid (or conversation_id)
    has a staged entry in the pending-restore registry.

    Allocates a fresh mamba_pool slot, injects the staged state into it,
    and assigns ``req.mamba_pool_idx``. The new turn's tokens stay in
    ``req.origin_input_ids`` unmodified — the SSM state in the pool slot
    carries the prior conversation history, so prefill processes only the
    new tokens on top of injected state.

    Returns:
        None  — no staged entry for this rid/conv_id (true miss, normal flow)
        True  — staged entry hydrated successfully
        False — staged entry exists but could not be applied (pool unavailable,
                pool full, or inject_state_to_pool raised). Caller MUST surface
                this as an error and not proceed with the request, otherwise the
                client gets silent fresh-start behavior despite a successful
                /restore_snapshot.
    """
    registry = getattr(scheduler, "pending_restore_registry", None)
    if not registry:
        return None
    if getattr(req, "mamba_pool_idx", None) is not None:
        return None

    rid = req.rid
    conv_id = getattr(req, "conversation_id", None)

    # Lookup: canonical first, then resolve via alias
    canonical = rid if rid in registry else None
    matched_via_rid = canonical is not None
    if canonical is None and conv_id and conv_id != rid:
        canonical = scheduler.pending_restore_aliases.get(conv_id)
    if canonical is None or canonical not in registry:
        return None

    entry = registry[canonical]

    mamba_pool = getattr(scheduler.req_to_token_pool, "mamba_pool", None)
    if mamba_pool is None:
        logger.error(
            "pending-restore registry hit for rid=%s but mamba_pool "
            "unavailable; hydration will fail",
            rid,
        )
        return False

    new_pool_idx = mamba_pool.alloc(1)
    if new_pool_idx is None:
        logger.error(
            "pending-restore registry hit for rid=%s but no free Mamba "
            "pool slots; hydration will fail",
            rid,
        )
        return False

    new_pool_idx_0d = new_pool_idx[0]
    new_pool_idx_scalar = new_pool_idx_0d.item()

    try:
        scheduler.snapshot_manager.inject_state_to_pool(
            entry.conv_states,
            entry.temporal_states,
            mamba_pool,
            new_pool_idx_scalar,
        )
    except Exception as e:
        logger.error(
            "Failed to inject pending-restore state for rid=%s: %s",
            rid,
            e,
            exc_info=True,
        )
        try:
            mamba_pool.free(new_pool_idx)
        except Exception as cleanup_error:
            logger.error(
                "Failed to free mamba pool slot during pending-restore "
                "hydration cleanup: %s",
                cleanup_error,
            )
        return False

    req.mamba_pool_idx = new_pool_idx_0d
    # Assign or reconcile conversation_id from the hydrated entry.
    # When the entry was matched via rid (not alias) and the request
    # already carries a different conversation_id, the entry's is
    # authoritative — state belongs to the conversation stored with it,
    # not to whatever the request happened to carry.
    if (
        matched_via_rid
        and getattr(req, "conversation_id", None)
        and req.conversation_id != entry.conversation_id
    ):
        logger.warning(
            "pending-restore hydration: req.conversation_id=%s differs from "
            "entry.conversation_id=%s for rid=%s; overwriting "
            "req.conversation_id with entry's (state is authoritative for "
            "the conversation it belongs to)",
            req.conversation_id,
            entry.conversation_id,
            rid,
        )
        req.conversation_id = entry.conversation_id
    elif not getattr(req, "conversation_id", None):
        req.conversation_id = entry.conversation_id

    # Touch the canonical entry to mark it most-recently-used.
    # Entries stay after consumption so back-to-back requests with the
    # same rid reuse the staged state without another /restore_snapshot.
    registry.move_to_end(canonical)

    # Reset health baselines so anomaly detection treats restored state
    # as a fresh starting point rather than a divergence from prior live
    # state.
    if scheduler.state_health_monitor is not None:
        scheduler.state_health_monitor.reset_baseline(entry.conversation_id)
        scheduler._health_check_counter[entry.conversation_id] = 0

    logger.info(
        "pending-restore registry hit: hydrated rid=%s "
        "(canonical=%s, conv_id=%s, mamba_pool_idx=%d)",
        rid,
        canonical,
        entry.conversation_id,
        new_pool_idx_scalar,
    )
    return True


def handle_save_snapshot(scheduler, recv_req):
    """Handle a manual snapshot save request from the Lang API.

    Args:
        recv_req: SaveSnapshotReqInput with snapshot details

    Returns:
        SaveSnapshotReqOutput with success status and snapshot_id
    """
    from sglang.srt.managers.io_struct import SaveSnapshotReqOutput
    from sglang.srt.snapshot import MambaSnapshotMetadata

    if scheduler.snapshot_manager is None:
        return SaveSnapshotReqOutput(
            success=False,
            snapshot_id=None,
            message="Snapshot system not enabled. Start server with --enable-mamba-snapshots",
        )

    try:
        effective_conv_id = recv_req.conversation_id or recv_req.rid
        effective_snapshot_id = (
            recv_req.snapshot_id or f"{effective_conv_id}-t{recv_req.turn_number or 0}"
        )

        # Find the request by rid
        req = scheduler._find_request_by_rid(recv_req.rid)
        if req is None:
            # Request already completed — check WARM tier for auto-saved state
            if (
                scheduler.tier_manager is not None
                and scheduler.tier_manager.host_pool.has_state(effective_conv_id)
            ):
                # State was auto-saved to WARM tier; persist to COLD tier now
                warm_result = scheduler.tier_manager.restore_from_warm_tier(
                    effective_conv_id
                )
                if warm_result is not None:
                    conv_states_w, temporal_states_w, meta_dict = warm_result
                    snap_meta = MambaSnapshotMetadata(
                        conversation_id=effective_conv_id,
                        turn_number=recv_req.turn_number
                        or meta_dict.get("turn_number", 0),
                        branch_name=recv_req.branch_name,
                        timestamp=time.time(),
                        token_count=int(meta_dict.get("token_count", 0)),
                        model_name=scheduler.server_args.model_path,
                        mamba_pool_idx=int(meta_dict.get("mamba_pool_idx", 0)),
                        req_pool_idx=int(meta_dict.get("req_pool_idx", 0)),
                        layer_config=meta_dict.get("layer_config", {}),
                        fill_ids=(
                            [int(x) for x in meta_dict["fill_ids"]]
                            if meta_dict.get("fill_ids") is not None
                            else None
                        ),
                    )
                    scheduler.snapshot_manager.save_snapshot(
                        conv_states_w, temporal_states_w, snap_meta
                    )
                    logger.info(
                        f"Manual snapshot saved from WARM tier: conversation={effective_conv_id}, "
                        f"snapshot_id={effective_snapshot_id}"
                    )
                    return SaveSnapshotReqOutput(
                        success=True,
                        snapshot_id=effective_snapshot_id,
                        message="Snapshot saved successfully (from WARM tier)",
                    )
            return SaveSnapshotReqOutput(
                success=False,
                snapshot_id=None,
                message=f"Request not found and no WARM tier state for: {recv_req.rid}",
            )

        # Check if request has mamba state
        if not hasattr(req, "mamba_pool_idx") or req.mamba_pool_idx is None:
            return SaveSnapshotReqOutput(
                success=False,
                snapshot_id=None,
                message="Request does not have Mamba state (not a Mamba model or state not allocated)",
            )

        # Get mamba_pool from req_to_token_pool
        mamba_pool = getattr(scheduler.req_to_token_pool, "mamba_pool", None)
        if mamba_pool is None:
            return SaveSnapshotReqOutput(
                success=False, snapshot_id=None, message="Mamba pool not available"
            )

        # Extract state from live pool
        conv_states, temporal_states = (
            scheduler.snapshot_manager.extract_state_from_pool(
                mamba_pool, req.mamba_pool_idx
            )
        )

        # Build metadata
        layer_config = {
            "num_layers": mamba_pool.num_mamba_layers,
            "model_type": "hybrid" if hasattr(req, "mamba_pool_idx") else "mamba",
        }

        metadata = MambaSnapshotMetadata(
            conversation_id=effective_conv_id,
            turn_number=recv_req.turn_number,
            branch_name=recv_req.branch_name,
            timestamp=time.time(),
            token_count=len(req.fill_ids) if hasattr(req, "fill_ids") else 0,
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

        # Save snapshot
        scheduler.snapshot_manager.save_snapshot(conv_states, temporal_states, metadata)

        logger.info(
            f"Manual snapshot saved: conversation={effective_conv_id}, "
            f"snapshot_id={effective_snapshot_id}, turn={recv_req.turn_number}"
        )

        return SaveSnapshotReqOutput(
            success=True,
            snapshot_id=effective_snapshot_id,
            message="Snapshot saved successfully",
        )

    except Exception as e:
        logger.error(f"Failed to save snapshot: {e}", exc_info=True)
        return SaveSnapshotReqOutput(
            success=False,
            snapshot_id=None,
            message=f"Error saving snapshot: {str(e)}",
        )


def handle_list_snapshots(scheduler, recv_req):
    """Handle a request to list snapshots for a conversation.

    Args:
        recv_req: ListSnapshotsReqInput with conversation_id

    Returns:
        ListSnapshotsReqOutput with list of snapshot metadata
    """
    from sglang.srt.managers.io_struct import ListSnapshotsReqOutput

    if scheduler.snapshot_manager is None:
        return ListSnapshotsReqOutput(
            success=False, snapshots=None, message="Snapshot system not enabled"
        )

    try:
        # Get all turn-based snapshots for this conversation
        turn_numbers = scheduler.snapshot_manager.list_snapshots(
            recv_req.conversation_id
        )

        # Load metadata for each turn
        snapshot_list = []
        for turn_number in turn_numbers:
            metadata = scheduler.snapshot_manager.load_metadata(
                recv_req.conversation_id, turn_number=turn_number
            )
            if metadata:
                snapshot_list.append(metadata.to_dict())

        # Also get all named branches
        branches = scheduler.snapshot_manager.list_branches(recv_req.conversation_id)
        for branch_name in branches:
            metadata = scheduler.snapshot_manager.load_metadata(
                recv_req.conversation_id, branch_name=branch_name
            )
            if metadata:
                snapshot_list.append(metadata.to_dict())

        return ListSnapshotsReqOutput(
            success=True, snapshots=snapshot_list, message=None
        )

    except Exception as e:
        logger.error(f"Failed to list snapshots: {e}", exc_info=True)
        return ListSnapshotsReqOutput(
            success=False,
            snapshots=None,
            message=f"Error listing snapshots: {str(e)}",
        )


def handle_get_snapshot_info(scheduler, recv_req):
    """Handle a request to get metadata for a specific snapshot.

    Args:
        recv_req: GetSnapshotInfoReqInput with conversation_id, turn_number, branch_name

    Returns:
        GetSnapshotInfoReqOutput with snapshot metadata
    """
    from sglang.srt.managers.io_struct import GetSnapshotInfoReqOutput

    if scheduler.snapshot_manager is None:
        return GetSnapshotInfoReqOutput(
            success=False, metadata=None, message="Snapshot system not enabled"
        )

    try:
        # Load metadata
        metadata = scheduler.snapshot_manager.load_metadata(
            recv_req.conversation_id, recv_req.turn_number, recv_req.branch_name
        )

        if metadata is None:
            return GetSnapshotInfoReqOutput(
                success=False,
                metadata=None,
                message=f"Snapshot not found: conversation={recv_req.conversation_id}, "
                f"turn={recv_req.turn_number}, branch={recv_req.branch_name}",
            )

        return GetSnapshotInfoReqOutput(
            success=True, metadata=metadata.to_dict(), message=None
        )

    except Exception as e:
        logger.error(f"Failed to get snapshot info: {e}", exc_info=True)
        return GetSnapshotInfoReqOutput(
            success=False,
            metadata=None,
            message=f"Error getting snapshot info: {str(e)}",
        )


def handle_restore_snapshot(scheduler, recv_req):
    """Handle a request to restore Mamba state from a snapshot.

    Args:
        recv_req: RestoreSnapshotReqInput with rid, conversation_id, turn_number, branch_name

    Returns:
        RestoreSnapshotReqOutput with success status
    """
    from sglang.srt.managers.io_struct import RestoreSnapshotReqOutput

    if scheduler.snapshot_manager is None:
        return RestoreSnapshotReqOutput(
            success=False,
            message="Snapshot system not enabled. Start server with --enable-mamba-snapshots",
        )

    # Derive effective_conv_id once for both paths
    effective_conv_id = recv_req.conversation_id or recv_req.rid

    # Note: create_new_request is deferred to Phase 3
    if recv_req.create_new_request:
        try:
            # Get mamba_pool from req_to_token_pool
            mamba_pool = getattr(scheduler.req_to_token_pool, "mamba_pool", None)
            if mamba_pool is None:
                return RestoreSnapshotReqOutput(
                    success=False, message="Mamba pool not available"
                )

            # Resolve turn_number: if not provided, use latest
            turn_number = recv_req.turn_number
            if turn_number is None and recv_req.branch_name is None:
                latest = scheduler.snapshot_manager.get_latest_snapshot(
                    effective_conv_id
                )
                if latest is None:
                    return RestoreSnapshotReqOutput(
                        success=False,
                        message=f"No snapshots found for conversation={effective_conv_id}",
                    )
                turn_number, _ = latest

            # Load snapshot from disk (use rid as conversation_id fallback)
            try:
                snapshot = scheduler.snapshot_manager.load_snapshot(
                    effective_conv_id, turn_number, recv_req.branch_name
                )
            except (FileNotFoundError, ValueError):
                snapshot = None

            if snapshot is None:
                return RestoreSnapshotReqOutput(
                    success=False,
                    message=(
                        f"Snapshot not found: conversation={effective_conv_id}, "
                        f"turn={turn_number}, branch={recv_req.branch_name}"
                    ),
                )

            conv_states, temporal_states, metadata = snapshot

            # Model compatibility check — prevent injecting state from a
            # different model which would produce garbage output
            running_model = getattr(scheduler.server_args, "model_path", None)
            if (
                running_model
                and metadata.model_name
                and metadata.model_name != running_model
            ):
                logger.warning(
                    f"Snapshot model mismatch: snapshot saved with "
                    f"'{metadata.model_name}' but server running "
                    f"'{running_model}'"
                )
                return RestoreSnapshotReqOutput(
                    success=False,
                    message=(
                        f"Model mismatch: snapshot from "
                        f"'{metadata.model_name}', server running "
                        f"'{running_model}'"
                    ),
                )

            # Allocate a new Mamba pool slot
            new_pool_idx = mamba_pool.alloc(1)
            if new_pool_idx is None:
                return RestoreSnapshotReqOutput(
                    success=False, message="No free Mamba pool slots available"
                )

            # new_pool_idx is a (1,)-tensor; get a 0-dim tensor and a scalar
            new_pool_idx_0d = new_pool_idx[0]  # 0-dim tensor — assign to req
            new_pool_idx_scalar = new_pool_idx_0d.item()  # Python int — for inject
            pool_freed = False  # guard for outer except cleanup

            # Inject state into the new pool slot
            scheduler.snapshot_manager.inject_state_to_pool(
                conv_states, temporal_states, mamba_pool, new_pool_idx_scalar
            )

            # Create new request ID
            new_rid = f"restored-{uuid.uuid4().hex[:8]}"

            # Create minimal Req object
            # We need to import Req here to avoid circular imports
            from sglang.srt.managers.schedule_batch import Req
            from sglang.srt.sampling.sampling_params import SamplingParams

            # Build origin_input_ids from metadata.fill_ids
            if metadata.fill_ids is None:
                # Snapshot lacks fill_ids — incompatible; fabricating tokens would
                # silently desync the token stream from the injected Mamba state.
                mamba_pool.free(new_pool_idx)
                return RestoreSnapshotReqOutput(
                    success=False,
                    message=(
                        "Snapshot is incompatible: fill_ids missing. "
                        "Re-run the original conversation to create a compatible snapshot."
                    ),
                )
            origin_input_ids = metadata.fill_ids

            # Stateful generation: append new tokens after the restored context.
            # When continuation_ids are provided, generation happens async and the
            # result is routed back via RestoreSnapshotReqOutput (not BatchTokenIDOut).
            # Empty list [] is a valid continuation signal — use `is not None`
            # instead of bool() so that an empty continuation_ids list still
            # routes through the stateful-generate path.
            stateful_generate = recv_req.continuation_ids is not None
            if stateful_generate:
                origin_input_ids = list(origin_input_ids) + list(
                    recv_req.continuation_ids
                )
                # Guard against context overflow before allocating
                if len(origin_input_ids) >= scheduler.max_req_input_len:
                    mamba_pool.free(new_pool_idx)
                    return RestoreSnapshotReqOutput(
                        success=False,
                        message=(
                            f"Input length ({len(origin_input_ids)}) meets or exceeds "
                            f"max_req_input_len ({scheduler.max_req_input_len}). "
                            "Reduce continuation_ids length or increase max_req_input_len."
                        ),
                    )

            _sp = SamplingParams()
            _sp.normalize(None)  # initialize stop_strs etc. (no tokenizer needed)
            if recv_req.max_new_tokens is not None:
                _sp.max_new_tokens = recv_req.max_new_tokens
            new_req = Req(
                rid=new_rid,
                origin_input_text="",  # We don't have original text
                origin_input_ids=origin_input_ids,
                sampling_params=_sp,
                vocab_size=scheduler.model_config.vocab_size,
                http_worker_ipc=recv_req.http_worker_ipc,
            )
            new_req.conversation_id = effective_conv_id
            new_req.mamba_pool_idx = (
                new_pool_idx_0d  # 0-dim tensor, required by free_mamba_cache
            )
            new_req.fill_ids = (
                torch.tensor(
                    origin_input_ids, dtype=torch.int64, device=scheduler.device
                )
                if origin_input_ids
                else torch.tensor([], dtype=torch.int64, device=scheduler.device)
            )
            # Signal the output processor to route finished output back via the
            # snapshot result channel instead of the normal BatchTokenIDOut path.
            new_req._stateful_generate = stateful_generate

            # Register the new request in the waiting queue so it can be found by _find_request_by_rid
            try:
                scheduler.init_req_max_new_tokens(new_req)
                scheduler._add_request_to_queue(new_req)
                logger.info(
                    f"Created new request from snapshot: conversation={recv_req.conversation_id}, "
                    f"turn={recv_req.turn_number}, rid={new_rid}, mamba_pool_idx={new_pool_idx_scalar}, "
                    f"stateful_generate={stateful_generate}"
                )

                # Reset health baselines after restore to avoid false anomalies
                if scheduler.state_health_monitor is not None:
                    scheduler.state_health_monitor.reset_baseline(effective_conv_id)
                    scheduler._health_check_counter[effective_conv_id] = 0

                if stateful_generate:
                    # Output will be sent after generation completes via the
                    # _stateful_generate hook in scheduler_output_processor_mixin.
                    return None

                return RestoreSnapshotReqOutput(
                    success=True,
                    message=f"Created new request from snapshot. rid={new_rid}, mamba_pool_idx={new_pool_idx_scalar}",
                    token_count=metadata.token_count if metadata else None,
                    rid=new_rid,
                    mamba_pool_idx=new_pool_idx_scalar,
                )
            except Exception as reg_error:
                # If registration fails, clean up the allocated mamba pool slot
                logger.error(
                    f"Failed to register restored request: {reg_error}",
                    exc_info=True,
                )
                try:
                    mamba_pool.free(new_pool_idx)
                    pool_freed = True
                except Exception as cleanup_error:
                    logger.error(
                        f"Failed to free mamba pool slot during cleanup: {cleanup_error}"
                    )
                raise

        except Exception as e:
            logger.error(
                f"Failed to create new request from snapshot: {e}", exc_info=True
            )
            # If we allocated a mamba pool slot but failed before registering, try to free it
            if "new_pool_idx" in locals() and not pool_freed:
                try:
                    mamba_pool.free(new_pool_idx)
                except Exception as cleanup_error:
                    logger.error(
                        f"Failed to free mamba pool slot during error cleanup: {cleanup_error}"
                    )
            return RestoreSnapshotReqOutput(
                success=False,
                message=f"Error creating new request from snapshot: {str(e)}",
            )

    try:
        # Find the request by rid
        req = scheduler._find_request_by_rid(recv_req.rid)
        if req is None:
            # Request has already completed. Load the snapshot and stage it
            # in the pending-restore registry so the next incoming request
            # with this rid (or conversation_id) hydrates from it.
            loaded = scheduler._load_snapshot_for_pending_restore(
                effective_conv_id, recv_req.turn_number, recv_req.branch_name
            )
            if loaded is None:
                return RestoreSnapshotReqOutput(
                    success=False,
                    message=(
                        f"Snapshot not found: conversation={effective_conv_id}, "
                        f"turn={recv_req.turn_number}, "
                        f"branch={recv_req.branch_name}"
                    ),
                )
            conv_states_p, temporal_states_p, metadata_p = loaded

            # Model compatibility check — same guard as the live-req path.
            running_model = getattr(scheduler.server_args, "model_path", None)
            snap_model = (
                metadata_p.get("model_name")
                if isinstance(metadata_p, dict)
                else getattr(metadata_p, "model_name", None)
            )
            if running_model and snap_model and snap_model != running_model:
                return RestoreSnapshotReqOutput(
                    success=False,
                    message=(
                        f"Model mismatch: snapshot from "
                        f"'{snap_model}', server running '{running_model}'"
                    ),
                )

            fill_ids_p = (
                metadata_p.get("fill_ids")
                if isinstance(metadata_p, dict)
                else getattr(metadata_p, "fill_ids", None)
            )
            token_count_p = (
                metadata_p.get("token_count")
                if isinstance(metadata_p, dict)
                else getattr(metadata_p, "token_count", None)
            )

            scheduler._stage_pending_restore(
                rid=recv_req.rid,
                conversation_id=effective_conv_id,
                conv_states=conv_states_p,
                temporal_states=temporal_states_p,
                fill_ids=fill_ids_p,
            )

            logger.info(
                "Snapshot staged for pending restore: "
                "conversation=%s, rid=%s, token_count=%s",
                effective_conv_id,
                recv_req.rid,
                token_count_p,
            )

            return RestoreSnapshotReqOutput(
                success=True,
                message=(
                    f"Snapshot staged for next request "
                    f"(rid={recv_req.rid}, conversation_id={effective_conv_id})"
                ),
                token_count=token_count_p,
                rid=recv_req.rid,
            )

        # Check that request is idle (not currently running)
        if scheduler.running_batch and req in scheduler.running_batch.reqs:
            return RestoreSnapshotReqOutput(
                success=False,
                message="Cannot restore snapshot while request is running. Wait until request is idle.",
            )

        # Check if request has mamba pool index
        if not hasattr(req, "mamba_pool_idx") or req.mamba_pool_idx is None:
            return RestoreSnapshotReqOutput(
                success=False, message="Request does not have Mamba state allocated"
            )

        # Load snapshot (use rid as conversation_id fallback)
        snapshot = scheduler.snapshot_manager.load_snapshot(
            effective_conv_id, recv_req.turn_number, recv_req.branch_name
        )

        if snapshot is None:
            return RestoreSnapshotReqOutput(
                success=False,
                message=f"Snapshot not found: conversation={effective_conv_id}, "
                f"turn={recv_req.turn_number}, branch={recv_req.branch_name}",
            )

        conv_states, temporal_states, metadata = snapshot

        # Model compatibility check — same guard as Path A
        running_model = getattr(scheduler.server_args, "model_path", None)
        if (
            running_model
            and metadata.model_name
            and metadata.model_name != running_model
        ):
            return RestoreSnapshotReqOutput(
                success=False,
                message=(
                    f"Model mismatch: snapshot from "
                    f"'{metadata.model_name}', server running "
                    f"'{running_model}'"
                ),
            )

        # Validate fill_ids BEFORE mutating the live request — if missing the
        # request would be left with new Mamba state but stale token history.
        if metadata.fill_ids is None:
            return RestoreSnapshotReqOutput(
                success=False,
                message=(
                    "Snapshot is incompatible: fill_ids missing. "
                    "Re-run the original conversation to create a compatible snapshot."
                ),
            )

        # Get mamba_pool from req_to_token_pool
        mamba_pool = getattr(scheduler.req_to_token_pool, "mamba_pool", None)
        if mamba_pool is None:
            return RestoreSnapshotReqOutput(
                success=False, message="Mamba pool not available"
            )

        # Inject state back into pool
        scheduler.snapshot_manager.inject_state_to_pool(
            conv_states, temporal_states, mamba_pool, req.mamba_pool_idx
        )

        # Clear any pending-restore entry so a follow-up request with the
        # same rid doesn't hydrate from the now-stale staged entry.
        scheduler._clear_pending_restore(recv_req.rid, effective_conv_id)

        # Sync fill_ids from metadata to request
        req.fill_ids = torch.tensor(
            metadata.fill_ids, dtype=torch.int64, device=scheduler.device
        )
        req.origin_input_ids = metadata.fill_ids  # keep as list for downstream code
        logger.info(f"Synced fill_ids after restore, len={len(metadata.fill_ids)}")

        # Conv_id conflict resolution: state is authoritative for the conversation
        # it belongs to. If req.conversation_id differs from effective_conv_id,
        # log a warning and overwrite (mirrors Finding 5 fix in
        # _maybe_hydrate_from_pending_restore for the symmetric pending path).
        if (
            getattr(req, "conversation_id", None)
            and req.conversation_id != effective_conv_id
        ):
            logger.warning(
                "Live restore: req.conversation_id=%s differs from effective_conv_id=%s "
                "for rid=%s; overwriting req.conversation_id with effective_conv_id "
                "(state is authoritative for the conversation it belongs to)",
                req.conversation_id,
                effective_conv_id,
                recv_req.rid,
            )
            req.conversation_id = effective_conv_id
        elif not getattr(req, "conversation_id", None):
            req.conversation_id = effective_conv_id

        # Reset health baselines after restore to avoid false anomalies
        if scheduler.state_health_monitor is not None:
            scheduler.state_health_monitor.reset_baseline(effective_conv_id)
            scheduler._health_check_counter[effective_conv_id] = 0

        logger.info(
            f"Snapshot restored: conversation={recv_req.conversation_id}, "
            f"turn={recv_req.turn_number}, branch={recv_req.branch_name}, "
            f"rid={recv_req.rid}"
        )

        return RestoreSnapshotReqOutput(
            success=True,
            message="Snapshot restored successfully",
            token_count=metadata.token_count if metadata else None,
            rid=recv_req.rid,
            mamba_pool_idx=req.mamba_pool_idx,
        )

    except Exception as e:
        logger.error(f"Failed to restore snapshot: {e}", exc_info=True)
        return RestoreSnapshotReqOutput(
            success=False, message=f"Error restoring snapshot: {str(e)}"
        )


def handle_delete_snapshot(scheduler, recv_req):
    """Handle a request to delete a snapshot.

    Args:
        recv_req: DeleteSnapshotReqInput with conversation_id, turn_number, branch_name

    Returns:
        DeleteSnapshotReqOutput with success status
    """
    from sglang.srt.managers.io_struct import DeleteSnapshotReqOutput

    if scheduler.snapshot_manager is None:
        return DeleteSnapshotReqOutput(
            success=False,
            message="Snapshot system not enabled. Start server with --enable-mamba-snapshots",
        )

    try:
        # Delete the snapshot
        success = scheduler.snapshot_manager.delete_snapshot(
            recv_req.conversation_id, recv_req.turn_number, recv_req.branch_name
        )

        if success:
            logger.info(
                f"Snapshot deleted: conversation={recv_req.conversation_id}, "
                f"turn={recv_req.turn_number}, branch={recv_req.branch_name}"
            )
            return DeleteSnapshotReqOutput(
                success=True, message="Snapshot deleted successfully"
            )
        else:
            return DeleteSnapshotReqOutput(
                success=False,
                message=f"Snapshot not found: conversation={recv_req.conversation_id}, "
                f"turn={recv_req.turn_number}, branch={recv_req.branch_name}",
            )

    except Exception as e:
        logger.error(f"Failed to delete snapshot: {e}", exc_info=True)
        return DeleteSnapshotReqOutput(
            success=False, message=f"Error deleting snapshot: {str(e)}"
        )


def handle_mamba_state_eviction(
    scheduler,
    conversation_id: str,
    mamba_pool_idx: int,
    req_pool_idx: int,
    metadata: Optional[dict] = None,
):
    """
    Handle Mamba state eviction from VRAM.

    This method is called when Mamba state is evicted from the GPU pool.
    It saves the state to the warm tier (host RAM) if tier management
    is enabled, otherwise it saves directly to cold tier (disk).

    Args:
        conversation_id: Conversation identifier
        mamba_pool_idx: Index in MambaPool being evicted
        req_pool_idx: Index in ReqToTokenPool
        metadata: Optional metadata dict

    Note:
        This is a helper method that should be called from eviction code
        paths (e.g., when MambaRadixCache evicts a node).
    """
    if scheduler.tier_manager is None:
        # No tier management, just log
        logger.debug(
            f"Mamba state evicted from VRAM (no tier management): {conversation_id}"
        )
        return

    # Get mamba pool
    mamba_pool = getattr(scheduler.req_to_token_pool, "mamba_pool", None)
    if mamba_pool is None:
        return

    try:
        # Extract state from pool before it's evicted
        conv_states, temporal_states = (
            scheduler.snapshot_manager.extract_state_from_pool(
                mamba_pool, mamba_pool_idx
            )
        )

        # Save to warm tier
        success = scheduler.tier_manager.save_to_warm_tier(
            conversation_id,
            conv_states,
            temporal_states,
            metadata or {},
        )

        if success:
            logger.info(
                f"State saved to WARM tier on eviction: {conversation_id}, "
                f"host_pool_size={len(scheduler.host_pool)}"
            )
        else:
            logger.warning(
                f"Failed to save to WARM tier on eviction: {conversation_id}"
            )

    except Exception as e:
        logger.error(f"Error handling Mamba state eviction: {e}", exc_info=True)
