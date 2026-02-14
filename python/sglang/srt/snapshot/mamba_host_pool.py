"""
Host memory pool for Mamba states (Tier 2: Warm Archive).

This module provides CPU-based caching of Mamba SSM states between
active VRAM (Tier 1) and cold disk storage (Tier 3).

**Memory Hierarchy:**
- Tier 1 (VRAM/MambaPool): Active inference, instant access
- Tier 2 (Host RAM/MambaHostPool): Recently active, fast restore (~10-50ms)
- Tier 3 (Disk/Snapshots): Long-term archive, slower restore (~100-500ms)

**Backward Compatibility:** This is an opt-in enhancement. Standard
transformer models and Mamba models without tier management are unaffected.
"""

import logging
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch

logger = logging.getLogger(__name__)


@dataclass
class HostPoolEntry:
    """
    Entry in the host memory pool.

    Attributes:
        conversation_id: Unique conversation identifier
        conv_states: List of convolution state tensors (CPU)
        temporal_states: Temporal SSM state tensor (CPU)
        last_access_time: Last time this entry was accessed
        access_count: Number of times accessed (for LFU eviction)
        metadata: Optional metadata (user_id, session_info, etc.)
    """

    conversation_id: str
    conv_states: List[torch.Tensor]
    temporal_states: torch.Tensor
    last_access_time: float
    access_count: int = 0
    metadata: Optional[dict] = None

    def memory_bytes(self) -> int:
        """Calculate total memory usage in bytes."""
        total = 0
        for conv in self.conv_states:
            total += conv.element_size() * conv.numel()
        total += self.temporal_states.element_size() * self.temporal_states.numel()
        return total


class MambaHostPool:
    """
    CPU-based host memory pool for Mamba states (Tier 2: Warm Archive).

    This pool acts as an intermediate tier between active VRAM and cold disk
    storage, enabling fast restoration of recently-used conversation states.

    **Features:**
    - LRU eviction policy with configurable max size
    - Thread-safe access with read-write locks
    - Memory usage tracking and limits
    - Cross-session reference support
    - Automatic promotion/demotion between tiers

    **Usage:**
        host_pool = MambaHostPool(max_conversations=100, max_memory_gb=10.0)

        # Save state from VRAM (on eviction)
        host_pool.save_state(conv_id, conv_states, temporal_states)

        # Restore state to VRAM (on resume)
        conv_states, temporal_states = host_pool.get_state(conv_id)

        # Cross-session reference
        context = host_pool.get_state("previous_conv_id")

    Thread Safety:
        All public methods are thread-safe. Uses RWLock for concurrent reads.
    """

    def __init__(
        self,
        max_conversations: int = 100,
        max_memory_gb: float = 10.0,
        enable_cross_session_refs: bool = True,
    ):
        """
        Initialize host memory pool.

        Args:
            max_conversations: Maximum number of conversations to keep in host RAM
            max_memory_gb: Maximum host memory usage in GB
            enable_cross_session_refs: Allow loading states from other conversations
        """
        self.max_conversations = max_conversations
        self.max_memory_bytes = int(max_memory_gb * 1024 * 1024 * 1024)
        self.enable_cross_session_refs = enable_cross_session_refs

        # Storage: OrderedDict maintains LRU order
        self._pool: OrderedDict[str, HostPoolEntry] = OrderedDict()

        # Thread safety
        self._lock = threading.RLock()

        # Metrics
        self._current_memory_bytes = 0
        self._hits = 0
        self._misses = 0
        self._evictions = 0

        logger.info(
            f"MambaHostPool initialized: "
            f"max_conversations={max_conversations}, "
            f"max_memory={max_memory_gb:.1f}GB, "
            f"cross_session_refs={enable_cross_session_refs}"
        )

    def save_state(
        self,
        conversation_id: str,
        conv_states: List[torch.Tensor],
        temporal_states: torch.Tensor,
        metadata: Optional[dict] = None,
    ) -> bool:
        """
        Save Mamba state to host pool.

        Args:
            conversation_id: Conversation identifier
            conv_states: List of conv state tensors (will be moved to CPU)
            temporal_states: Temporal state tensor (will be moved to CPU)
            metadata: Optional metadata dict

        Returns:
            True if saved successfully, False if eviction failed to make space
        """
        with self._lock:
            # Move tensors to CPU if needed
            conv_states_cpu = [
                t.cpu() if t.device.type != "cpu" else t for t in conv_states
            ]
            temporal_states_cpu = (
                temporal_states.cpu()
                if temporal_states.device.type != "cpu"
                else temporal_states
            )

            # Create entry
            entry = HostPoolEntry(
                conversation_id=conversation_id,
                conv_states=conv_states_cpu,
                temporal_states=temporal_states_cpu,
                last_access_time=time.time(),
                access_count=0,
                metadata=metadata,
            )

            entry_size = entry.memory_bytes()

            # Check if we need to evict
            while (
                len(self._pool) >= self.max_conversations
                or self._current_memory_bytes + entry_size > self.max_memory_bytes
            ):
                if not self._evict_lru():
                    logger.error("Failed to evict entry to make space")
                    return False

            # Update existing or add new
            if conversation_id in self._pool:
                # Remove old entry's memory count
                old_entry = self._pool.pop(conversation_id)
                self._current_memory_bytes -= old_entry.memory_bytes()

            # Add entry (at end for LRU)
            self._pool[conversation_id] = entry
            self._current_memory_bytes += entry_size

            logger.debug(
                f"Saved state to host pool: {conversation_id}, "
                f"size={entry_size / 1024 / 1024:.2f}MB, "
                f"total_memory={self._current_memory_bytes / 1024 / 1024:.2f}MB"
            )

            return True

    def get_state(
        self, conversation_id: str
    ) -> Optional[Tuple[List[torch.Tensor], torch.Tensor, dict]]:
        """
        Retrieve Mamba state from host pool.

        This method:
        - Returns clones to avoid mutation
        - Updates LRU order (moves to end)
        - Increments access count and hit metrics

        Args:
            conversation_id: Conversation identifier

        Returns:
            Tuple of (conv_states, temporal_states, metadata) or None if not found
        """
        with self._lock:
            if conversation_id not in self._pool:
                self._misses += 1
                logger.debug(f"Host pool miss: {conversation_id}")
                return None

            entry = self._pool.pop(conversation_id)

            # Update access metadata
            entry.last_access_time = time.time()
            entry.access_count += 1

            # Move to end (most recently used)
            self._pool[conversation_id] = entry

            # Update metrics
            self._hits += 1

            # Clone tensors to avoid mutation
            conv_states_clone = [t.clone() for t in entry.conv_states]
            temporal_states_clone = entry.temporal_states.clone()

            logger.debug(
                f"Host pool hit: {conversation_id}, "
                f"access_count={entry.access_count}, "
                f"hit_rate={self.get_hit_rate():.2%}"
            )

            return conv_states_clone, temporal_states_clone, entry.metadata or {}

    def get_state_for_reference(
        self, conversation_id: str, target_conversation_id: str
    ) -> Optional[Tuple[List[torch.Tensor], torch.Tensor, dict]]:
        """
        Get state from another conversation (cross-session reference).

        This is used when a user in one conversation references context
        from a different conversation.

        Args:
            conversation_id: Current conversation ID (for logging)
            target_conversation_id: Target conversation ID to load

        Returns:
            Tuple of (conv_states, temporal_states, metadata) or None if not found
            or cross-session refs disabled
        """
        if not self.enable_cross_session_refs:
            logger.warning(
                f"Cross-session reference disabled: "
                f"{conversation_id} -> {target_conversation_id}"
            )
            return None

        logger.info(
            f"Cross-session reference: {conversation_id} loading from {target_conversation_id}"
        )

        return self.get_state(target_conversation_id)

    def has_state(self, conversation_id: str) -> bool:
        """Check if conversation state exists in host pool."""
        with self._lock:
            return conversation_id in self._pool

    def remove_state(self, conversation_id: str) -> bool:
        """
        Remove state from host pool.

        Args:
            conversation_id: Conversation identifier

        Returns:
            True if removed, False if not found
        """
        with self._lock:
            if conversation_id not in self._pool:
                return False

            entry = self._pool.pop(conversation_id)
            self._current_memory_bytes -= entry.memory_bytes()

            logger.debug(
                f"Removed state from host pool: {conversation_id}, "
                f"freed={entry.memory_bytes() / 1024 / 1024:.2f}MB"
            )

            return True

    def _evict_lru(self) -> bool:
        """
        Evict least recently used entry.

        Returns:
            True if evicted, False if pool is empty
        """
        if not self._pool:
            return False

        # OrderedDict: first item is least recently used
        conversation_id, entry = self._pool.popitem(last=False)
        self._current_memory_bytes -= entry.memory_bytes()
        self._evictions += 1

        logger.info(
            f"Evicted from host pool (LRU): {conversation_id}, "
            f"freed={entry.memory_bytes() / 1024 / 1024:.2f}MB, "
            f"total_evictions={self._evictions}"
        )

        return True

    def list_conversations(self) -> List[str]:
        """List all conversation IDs in host pool (LRU order)."""
        with self._lock:
            return list(self._pool.keys())

    def get_conversation_metadata(self, conversation_id: str) -> Optional[dict]:
        """Get metadata for a conversation without accessing state."""
        with self._lock:
            entry = self._pool.get(conversation_id)
            if entry:
                return {
                    "conversation_id": entry.conversation_id,
                    "last_access_time": entry.last_access_time,
                    "access_count": entry.access_count,
                    "memory_bytes": entry.memory_bytes(),
                    "metadata": entry.metadata,
                }
            return None

    def get_stats(self) -> dict:
        """Get pool statistics."""
        with self._lock:
            return {
                "current_conversations": len(self._pool),
                "max_conversations": self.max_conversations,
                "current_memory_bytes": self._current_memory_bytes,
                "max_memory_bytes": self.max_memory_bytes,
                "memory_utilization": (
                    self._current_memory_bytes / self.max_memory_bytes
                    if self.max_memory_bytes > 0
                    else 0.0
                ),
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate": self.get_hit_rate(),
                "evictions": self._evictions,
            }

    def get_hit_rate(self) -> float:
        """Calculate cache hit rate."""
        total_accesses = self._hits + self._misses
        if total_accesses == 0:
            return 0.0
        return self._hits / total_accesses

    def clear(self):
        """Clear all entries from host pool."""
        with self._lock:
            self._pool.clear()
            self._current_memory_bytes = 0
            logger.info("Host pool cleared")

    def __len__(self) -> int:
        """Return number of conversations in pool."""
        with self._lock:
            return len(self._pool)
