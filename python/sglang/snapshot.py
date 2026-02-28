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
"""
High-level snapshot management API for SGLang.

This module provides a user-friendly interface for managing Mamba state snapshots
without requiring direct access to the RuntimeEndpoint internals.
"""

from typing import Dict, List, Optional


__all__ = [
    "SnapshotManager",
    "SnapshotError",
    "SnapshotDisabledError",
    "SnapshotNotFoundError",
    "SnapshotInvalidError",
    "SnapshotInUseError",
    "SnapshotDeserializationError",
]


# Snapshot-specific exceptions
class SnapshotError(Exception):
    """Base exception for snapshot operations."""
    pass


class SnapshotDisabledError(SnapshotError):
    """Raised when snapshot operations are attempted but snapshots are not enabled."""
    pass


class SnapshotNotFoundError(SnapshotError):
    """Raised when a requested snapshot ID doesn't exist."""
    pass


class SnapshotInvalidError(SnapshotError):
    """Raised when a snapshot's state is corrupted or invalid."""
    pass


class SnapshotInUseError(SnapshotError):
    """Raised when attempting to delete/modify a snapshot currently in use."""
    pass


class SnapshotDeserializationError(SnapshotError):
    """Raised when snapshot file is corrupted or incompatible."""
    pass


class SnapshotManager:
    """
    High-level API for managing Mamba state snapshots.

    This class wraps the RuntimeEndpoint snapshot APIs to provide a cleaner
    interface for working with snapshots, similar to how git commands abstract
    over the .git directory.

    Example:
        ```python
        import sglang as sgl

        # Initialize runtime
        runtime = sgl.Runtime(
            model_path="state-spaces/mamba-2.8b",
            enable_snapshot_persistence=True,
        )

        # Create snapshot manager
        sm = sgl.SnapshotManager(runtime.endpoint)

        # List snapshots for a conversation
        snapshots = sm.list_conversation("conv_123")

        # Get specific snapshot info
        info = sm.get_info("conv_123", turn_number=5)

        # Delete a snapshot
        sm.delete("conv_123", turn_number=3)
        ```
    """

    def __init__(self, endpoint):
        """
        Initialize the SnapshotManager with a RuntimeEndpoint.

        Args:
            endpoint: A RuntimeEndpoint instance with snapshot support enabled

        Raises:
            TypeError: If endpoint is not a RuntimeEndpoint
            RuntimeError: If endpoint doesn't support snapshots
        """
        from sglang.lang.backend.runtime_endpoint import RuntimeEndpoint

        if not isinstance(endpoint, RuntimeEndpoint):
            raise TypeError(
                f"SnapshotManager requires a RuntimeEndpoint, got {type(endpoint).__name__}. "
                f"Usage: sm = SnapshotManager(runtime.endpoint)"
            )

        # Verify snapshot support by checking for the method
        if not hasattr(endpoint, 'list_snapshots'):
            raise RuntimeError(
                "RuntimeEndpoint does not support snapshots. "
                "Start the server with --enable-snapshot-persistence flag."
            )

        self.endpoint = endpoint

    def list_conversation(self, conversation_id: str) -> List[Dict]:
        """
        List all snapshots for a specific conversation.

        Args:
            conversation_id: The conversation identifier

        Returns:
            List of snapshot metadata dictionaries, each containing:
                - conversation_id: str
                - turn_number: int (or None for named branches)
                - branch_name: str (or None for main conversation)
                - timestamp: float (Unix timestamp)
                - token_count: int
                - model_name: str

        Example:
            ```python
            snapshots = sm.list_conversation("conv_123")
            for snap in snapshots:
                print(f"Turn {snap['turn_number']}: {snap['token_count']} tokens")
            ```
        """
        try:
            return self.endpoint.list_snapshots(conversation_id)
        except Exception as e:
            raise RuntimeError(f"Failed to list snapshots: {e}") from e

def get_info(
        self,
        conversation_id: str,
        turn_number: Optional[int] = None,
        branch_name: Optional[str] = None,
    ) -> Dict:
        """
        Get detailed metadata for a specific snapshot.

        Args:
            conversation_id: The conversation identifier
            turn_number: The turn number (for main conversation snapshots)
            branch_name: The branch name (for named branch snapshots)

        Returns:
            Snapshot metadata dictionary with all available information

        Raises:
            ValueError: If neither turn_number nor branch_name is specified
            RuntimeError: If the snapshot doesn't exist or another error occurs

        Example:

    def restore(
        self,
        rid: str,
        conversation_id: str,
        turn_number: Optional[int] = None,
        branch_name: Optional[str] = None,
    ) -> Dict:
        """
        Restore Mamba state from a snapshot.

        This operation restores the internal Mamba state to a previous point,
        allowing you to continue generation from that state.

        Args:
            rid: Request ID to restore the snapshot into
            conversation_id: The conversation identifier
            turn_number: The turn number (for main conversation snapshots)
            branch_name: The branch name (for named branch snapshots)

        Returns:
            Result dictionary with:
                - success: bool
                - message: str
                - token_count: int (number of tokens in restored state)

        Raises:
            ValueError: If neither turn_number nor branch_name is specified
            RuntimeError: If restore fails (snapshot not found, request not found, etc.)

        Example:
            ```python
            # Restore from turn 5
            result = sm.restore(
                rid="req_123",
                conversation_id="conv_123",
                turn_number=5
            )
            print(f"Restored {result['token_count']} tokens")

            # Restore from a named branch
            sm.restore(
                rid="req_456",
                conversation_id="conv_123",
                branch_name="alternative"
            )
            ```

        Note:
            The request must exist and be idle (not currently generating).
            The restore operation only affects internal Mamba state, not
            the request's input text or metadata.
        """
        if turn_number is None and branch_name is None:
            raise ValueError(
                "Must specify either turn_number or branch_name"
            )

        try:
            result = self.endpoint.restore_snapshot(
                rid=rid,
                conversation_id=conversation_id,
                turn_number=turn_number,
                branch_name=branch_name,
            )
            if not result.get('success'):
                raise RuntimeError(f"Restore failed: {result.get('message')}")
            return result
        except RuntimeError:
            raise
        except Exception as e:
            raise RuntimeError(f"Failed to restore snapshot: {e}") from e

    def delete(
        self,
        conversation_id: str,
        turn_number: Optional[int] = None,
        branch_name: Optional[str] = None,
    ) -> bool:
        """
        Delete a specific snapshot.

        Args:
            conversation_id: The conversation identifier
            turn_number: The turn number (for main conversation snapshots)
            branch_name: The branch name (for named branch snapshots)

        Returns:
            True if snapshot was deleted, False if it didn't exist

        Raises:
            ValueError: If neither turn_number nor branch_name is specified

        Example:
            ```python
            # Delete turn 3
            sm.delete("conv_123", turn_number=3)

            # Delete a branch
            sm.delete("conv_123", branch_name="alternative")
            ```
        """
        if turn_number is None and branch_name is None:
            raise ValueError(
                "Must specify either turn_number or branch_name"
            )

        try:
            result = self.endpoint.delete_snapshot(
                conversation_id=conversation_id,
                turn_number=turn_number,
                branch_name=branch_name,
            )
            return result.get('success', False)
        except Exception as e:
            raise RuntimeError(f"Failed to delete snapshot: {e}") from e

    # Note: list_all_conversations() and get_size() are deferred to future phases
    # as they require additional server-side endpoints
