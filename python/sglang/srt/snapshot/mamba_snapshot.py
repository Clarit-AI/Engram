"""
Core snapshot serialization and management for Mamba SSM states.

This module handles the low-level serialization of Mamba state tensors to disk
using safetensors format, along with associated metadata tracking.
"""

import json
import logging
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from safetensors.torch import load_file, save_file

logger = logging.getLogger(__name__)


@dataclass
class MambaSnapshotMetadata:
    """
    Metadata associated with a Mamba state snapshot.

    Attributes:
        conversation_id: Unique identifier for the conversation
        turn_number: Turn number within the conversation (0-indexed)
        timestamp: Unix timestamp when snapshot was created
        token_count: Total tokens processed up to this point
        model_name: Full model identifier (e.g., "ibm-granite/granite-4.0-h-small")
        mamba_pool_idx: Index in MambaPool where state is stored
        req_pool_idx: Index in ReqToTokenPool for request mapping
        active_tool_context: Optional JSON string of active tool state
        layer_config: Configuration dict with num_layers, state_size, etc.
        branch_name: Optional named branch for this snapshot
        parent_snapshot: Optional path to parent snapshot (for branching)
    """

    conversation_id: str
    turn_number: int
    timestamp: float
    token_count: int
    model_name: str
    mamba_pool_idx: int
    req_pool_idx: int
    layer_config: Dict[str, Any]
    active_tool_context: Optional[str] = None
    branch_name: Optional[str] = None
    parent_snapshot: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert metadata to dictionary for JSON serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MambaSnapshotMetadata":
        """Create metadata from dictionary loaded from JSON."""
        return cls(**data)

    def to_json(self, path: Path) -> None:
        """Save metadata to JSON file."""
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def from_json(cls, path: Path) -> "MambaSnapshotMetadata":
        """Load metadata from JSON file."""
        with open(path, "r") as f:
            data = json.load(f)
        return cls.from_dict(data)


class MambaSnapshotManager:
    """
    Manages serialization and restoration of Mamba SSM states.

    This class handles:
    - Saving Mamba state tensors to disk (using safetensors)
    - Loading and validating saved states
    - Managing snapshot directory structure
    - Injecting restored states back into active MambaPool

    Directory Structure:
        {base_dir}/
            conversation_{id}/
                turn_{n}_metadata.json
                turn_{n}_state.safetensors
                branches/
                    {branch_name}_metadata.json
                    {branch_name}_state.safetensors

    Thread Safety:
        This class is NOT thread-safe. Caller must ensure exclusive access
        when saving/loading snapshots during active inference.
    """

    def __init__(self, base_dir: Path):
        """
        Initialize snapshot manager.

        Args:
            base_dir: Root directory for all snapshot storage
        """
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"MambaSnapshotManager initialized at {self.base_dir}")

    def _get_conversation_dir(self, conversation_id: str) -> Path:
        """Get directory for a specific conversation."""
        conv_dir = self.base_dir / f"conversation_{conversation_id}"
        conv_dir.mkdir(parents=True, exist_ok=True)
        return conv_dir

    def _get_snapshot_paths(
        self,
        conversation_id: str,
        turn_number: Optional[int] = None,
        branch_name: Optional[str] = None,
    ) -> Tuple[Path, Path]:
        """
        Get paths for metadata and state files.

        Args:
            conversation_id: Conversation identifier
            turn_number: Turn number (for main conversation)
            branch_name: Branch name (for named branches)

        Returns:
            Tuple of (metadata_path, state_path)
        """
        conv_dir = self._get_conversation_dir(conversation_id)

        if branch_name:
            branch_dir = conv_dir / "branches"
            branch_dir.mkdir(parents=True, exist_ok=True)
            metadata_path = branch_dir / f"{branch_name}_metadata.json"
            state_path = branch_dir / f"{branch_name}_state.safetensors"
        elif turn_number is not None:
            metadata_path = conv_dir / f"turn_{turn_number}_metadata.json"
            state_path = conv_dir / f"turn_{turn_number}_state.safetensors"
        else:
            raise ValueError("Must specify either turn_number or branch_name")

        return metadata_path, state_path

    def save_snapshot(
        self,
        conv_states: List[torch.Tensor],
        temporal_states: torch.Tensor,
        metadata: MambaSnapshotMetadata,
    ) -> Tuple[Path, Path]:
        """
        Save Mamba state snapshot to disk.

        Args:
            conv_states: List of convolution state tensors (one per layer)
            temporal_states: Temporal SSM state tensor
            metadata: Snapshot metadata

        Returns:
            Tuple of (metadata_path, state_path) where files were saved

        Raises:
            ValueError: If state tensors don't match layer_config
            IOError: If disk write fails
        """
        # Validate state shapes match metadata
        num_layers = metadata.layer_config.get("num_layers")
        if num_layers is not None and len(conv_states) != num_layers:
            raise ValueError(
                f"conv_states length ({len(conv_states)}) doesn't match "
                f"num_layers ({num_layers})"
            )

        # Get save paths
        metadata_path, state_path = self._get_snapshot_paths(
            metadata.conversation_id,
            metadata.turn_number,
            metadata.branch_name,
        )

        # Prepare state tensors for safetensors
        state_dict = {}

        # Add temporal state
        state_dict["temporal"] = temporal_states

        # Add conv states (one per layer)
        for layer_idx, conv_state in enumerate(conv_states):
            state_dict[f"conv_layer_{layer_idx}"] = conv_state

        # Save tensors atomically
        try:
            # Write to temporary file first
            temp_state_path = state_path.with_suffix(".safetensors.tmp")
            save_file(state_dict, temp_state_path)

            # Atomic rename
            temp_state_path.rename(state_path)

            logger.debug(
                f"Saved state tensors to {state_path} "
                f"(size: {state_path.stat().st_size / 1024 / 1024:.2f} MB)"
            )
        except Exception as e:
            logger.error(f"Failed to save state tensors: {e}")
            # Clean up temp file if it exists
            if temp_state_path.exists():
                temp_state_path.unlink()
            raise IOError(f"Failed to save state: {e}") from e

        # Save metadata
        try:
            temp_metadata_path = metadata_path.with_suffix(".json.tmp")
            metadata.to_json(temp_metadata_path)
            temp_metadata_path.rename(metadata_path)

            logger.info(
                f"Snapshot saved: conversation={metadata.conversation_id}, "
                f"turn={metadata.turn_number}, branch={metadata.branch_name}"
            )
        except Exception as e:
            logger.error(f"Failed to save metadata: {e}")
            # Clean up state file if metadata save fails
            if state_path.exists():
                state_path.unlink()
            raise IOError(f"Failed to save metadata: {e}") from e

        return metadata_path, state_path

    def load_snapshot(
        self,
        conversation_id: str,
        turn_number: Optional[int] = None,
        branch_name: Optional[str] = None,
    ) -> Tuple[List[torch.Tensor], torch.Tensor, MambaSnapshotMetadata]:
        """
        Load Mamba state snapshot from disk.

        Args:
            conversation_id: Conversation identifier
            turn_number: Turn number to load (for main conversation)
            branch_name: Branch name to load (for named branches)

        Returns:
            Tuple of (conv_states, temporal_states, metadata)

        Raises:
            FileNotFoundError: If snapshot doesn't exist
            ValueError: If snapshot data is corrupted
        """
        # Get load paths
        metadata_path, state_path = self._get_snapshot_paths(
            conversation_id, turn_number, branch_name
        )

        # Check files exist
        if not metadata_path.exists():
            raise FileNotFoundError(f"Metadata not found: {metadata_path}")
        if not state_path.exists():
            raise FileNotFoundError(f"State not found: {state_path}")

        # Load metadata
        try:
            metadata = MambaSnapshotMetadata.from_json(metadata_path)
        except Exception as e:
            raise ValueError(f"Failed to load metadata: {e}") from e

        # Load state tensors
        try:
            state_dict = load_file(state_path)
        except Exception as e:
            raise ValueError(f"Failed to load state tensors: {e}") from e

        # Extract temporal state
        if "temporal" not in state_dict:
            raise ValueError("Snapshot missing 'temporal' state")
        temporal_states = state_dict["temporal"]

        # Extract conv states
        conv_states = []
        layer_idx = 0
        while f"conv_layer_{layer_idx}" in state_dict:
            conv_states.append(state_dict[f"conv_layer_{layer_idx}"])
            layer_idx += 1

        if not conv_states:
            raise ValueError("Snapshot has no conv states")

        # Validate against metadata
        num_layers = metadata.layer_config.get("num_layers")
        if num_layers is not None and len(conv_states) != num_layers:
            logger.warning(
                f"Conv state count ({len(conv_states)}) doesn't match "
                f"metadata num_layers ({num_layers})"
            )

        logger.info(
            f"Snapshot loaded: conversation={conversation_id}, "
            f"turn={turn_number}, branch={branch_name}, "
            f"layers={len(conv_states)}"
        )

        return conv_states, temporal_states, metadata

    def list_conversations(self) -> List[str]:
        """
        List all conversation IDs with saved snapshots.

        Returns:
            List of conversation IDs (sorted)
        """
        conversations = []
        for path in self.base_dir.iterdir():
            if path.is_dir() and path.name.startswith("conversation_"):
                conv_id = path.name.replace("conversation_", "")
                conversations.append(conv_id)
        return sorted(conversations)

    def list_snapshots(self, conversation_id: str) -> List[int]:
        """
        List all turn numbers with snapshots for a conversation.

        Args:
            conversation_id: Conversation identifier

        Returns:
            List of turn numbers (sorted)
        """
        conv_dir = self._get_conversation_dir(conversation_id)
        turns = []

        for path in conv_dir.iterdir():
            if path.is_file() and path.name.startswith("turn_") and path.suffix == ".json":
                turn_str = path.stem.replace("turn_", "").replace("_metadata", "")
                try:
                    turn_num = int(turn_str)
                    turns.append(turn_num)
                except ValueError:
                    continue

        return sorted(turns)

    def list_branches(self, conversation_id: str) -> List[str]:
        """
        List all named branches for a conversation.

        Args:
            conversation_id: Conversation identifier

        Returns:
            List of branch names (sorted)
        """
        conv_dir = self._get_conversation_dir(conversation_id)
        branch_dir = conv_dir / "branches"

        if not branch_dir.exists():
            return []

        branches = []
        for path in branch_dir.iterdir():
            if path.is_file() and path.suffix == ".json":
                branch_name = path.stem.replace("_metadata", "")
                branches.append(branch_name)

        return sorted(branches)

    def get_latest_snapshot(
        self, conversation_id: str
    ) -> Optional[Tuple[int, MambaSnapshotMetadata]]:
        """
        Get the most recent snapshot for a conversation.

        Args:
            conversation_id: Conversation identifier

        Returns:
            Tuple of (turn_number, metadata) or None if no snapshots
        """
        turns = self.list_snapshots(conversation_id)

        if not turns:
            return None

        latest_turn = turns[-1]
        metadata_path, _ = self._get_snapshot_paths(conversation_id, latest_turn)
        metadata = MambaSnapshotMetadata.from_json(metadata_path)

        return latest_turn, metadata

    def delete_snapshot(
        self,
        conversation_id: str,
        turn_number: Optional[int] = None,
        branch_name: Optional[str] = None,
    ) -> None:
        """
        Delete a snapshot.

        Args:
            conversation_id: Conversation identifier
            turn_number: Turn number to delete
            branch_name: Branch name to delete
        """
        metadata_path, state_path = self._get_snapshot_paths(
            conversation_id, turn_number, branch_name
        )

        if metadata_path.exists():
            metadata_path.unlink()
        if state_path.exists():
            state_path.unlink()

        logger.info(
            f"Snapshot deleted: conversation={conversation_id}, "
            f"turn={turn_number}, branch={branch_name}"
        )

    def get_snapshot_size(
        self,
        conversation_id: str,
        turn_number: Optional[int] = None,
        branch_name: Optional[str] = None,
    ) -> int:
        """
        Get total disk size of a snapshot in bytes.

        Args:
            conversation_id: Conversation identifier
            turn_number: Turn number
            branch_name: Branch name

        Returns:
            Total size in bytes
        """
        metadata_path, state_path = self._get_snapshot_paths(
            conversation_id, turn_number, branch_name
        )

        total_size = 0
        if metadata_path.exists():
            total_size += metadata_path.stat().st_size
        if state_path.exists():
            total_size += state_path.stat().st_size

        return total_size

    def extract_state_from_pool(
        self, mamba_pool, mamba_pool_idx: int
    ) -> Tuple[List[torch.Tensor], torch.Tensor]:
        """
        Extract Mamba state tensors from MambaPool at specific index.

        This is a helper method to extract the state from a live MambaPool.
        The extracted tensors are cloned to CPU to avoid holding GPU memory.

        Args:
            mamba_pool: MambaPool instance (from memory_pool.py)
            mamba_pool_idx: Index in the pool (typically from req.mamba_pool_idx)

        Returns:
            Tuple of (conv_states, temporal_states) with tensors on CPU

        Note:
            This method assumes mamba_pool is NOT None and mamba_pool_idx is valid.
            Caller should check these conditions before calling.
        """
        # Extract conv states (list of tensors, one per layer)
        # Shape: [num_layers, size+1, ...] -> extract [:, mamba_pool_idx, ...]
        conv_states = []
        for conv_tensor in mamba_pool.mamba_cache.conv:
            # Extract state for this index across all layers
            conv_state = conv_tensor[:, mamba_pool_idx].clone().cpu()
            conv_states.append(conv_state)

        # Extract temporal state
        # Shape: [num_layers, size+1, ...] -> extract [:, mamba_pool_idx, ...]
        temporal_states = mamba_pool.mamba_cache.temporal[:, mamba_pool_idx].clone().cpu()

        logger.debug(
            f"Extracted state from pool idx {mamba_pool_idx}: "
            f"{len(conv_states)} conv tensors, "
            f"temporal shape {temporal_states.shape}"
        )

        return conv_states, temporal_states

    def inject_state_to_pool(
        self,
        conv_states: List[torch.Tensor],
        temporal_states: torch.Tensor,
        mamba_pool,
        mamba_pool_idx: int,
    ) -> None:
        """
        Inject restored Mamba state tensors back into MambaPool.

        This method takes state tensors (typically loaded from disk on CPU)
        and injects them into the active MambaPool at the specified index.

        Args:
            conv_states: List of conv state tensors (one per layer), on CPU
            temporal_states: Temporal state tensor, on CPU
            mamba_pool: MambaPool instance (from memory_pool.py)
            mamba_pool_idx: Index in the pool where state should be written

        Raises:
            ValueError: If state shapes don't match pool configuration
            RuntimeError: If pool index is out of bounds

        Note:
            This operation is NOT thread-safe. Caller must ensure no
            concurrent access to mamba_pool during injection.

            Tensors are automatically moved to the pool's device (GPU/CPU).
        """
        # Validate pool index
        if mamba_pool_idx < 1 or mamba_pool_idx > mamba_pool.size:
            raise RuntimeError(
                f"Invalid mamba_pool_idx {mamba_pool_idx}, "
                f"must be in range [1, {mamba_pool.size}]"
            )

        # Validate conv state count
        if len(conv_states) != len(mamba_pool.mamba_cache.conv):
            raise ValueError(
                f"Conv state count mismatch: got {len(conv_states)}, "
                f"expected {len(mamba_pool.mamba_cache.conv)}"
            )

        # Inject conv states
        for i, conv_state in enumerate(conv_states):
            # Move to device and inject
            conv_state_device = conv_state.to(mamba_pool.device)

            # Validate shape
            expected_shape = mamba_pool.mamba_cache.conv[i][:, mamba_pool_idx].shape
            if conv_state_device.shape != expected_shape:
                raise ValueError(
                    f"Conv state {i} shape mismatch: got {conv_state_device.shape}, "
                    f"expected {expected_shape}"
                )

            mamba_pool.mamba_cache.conv[i][:, mamba_pool_idx] = conv_state_device

        # Inject temporal state
        temporal_states_device = temporal_states.to(mamba_pool.device)

        # Validate shape
        expected_shape = mamba_pool.mamba_cache.temporal[:, mamba_pool_idx].shape
        if temporal_states_device.shape != expected_shape:
            raise ValueError(
                f"Temporal state shape mismatch: got {temporal_states_device.shape}, "
                f"expected {expected_shape}"
            )

        mamba_pool.mamba_cache.temporal[:, mamba_pool_idx] = temporal_states_device

        logger.info(
            f"Injected state to pool idx {mamba_pool_idx}: "
            f"{len(conv_states)} conv tensors, "
            f"temporal shape {temporal_states.shape} "
            f"(moved to {mamba_pool.device})"
        )
