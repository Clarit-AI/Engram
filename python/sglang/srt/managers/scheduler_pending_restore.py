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
# ENGRAM_MODIFIED — Pending-restore registry data structure (Engram-added)
"""Pending-restore registry types.

Module-level data shapes for the pending-restore registry that lets a snapshot
restore complete asynchronously: a restore RPC stages loaded Mamba state into
this registry keyed by the originating rid (or conversation_id alias), and the
next matching incoming Req on the Scheduler consumes it via
``_maybe_hydrate_from_pending_restore``.

Extracted from ``scheduler.py`` as part of the scheduler-decomposition port
(2026-05-20). The Scheduler still owns the live registry instance — only the
type + capacity constant live here.
"""

from dataclasses import dataclass
from typing import List, Optional

import torch

PENDING_RESTORE_REGISTRY_MAX = 256


@dataclass
class PendingRestoreEntry:
    """Loaded snapshot state staged for a future incoming request.

    Populated by handle_restore_snapshot when the originating Req has already
    completed, and consumed by _maybe_hydrate_from_pending_restore on the next
    incoming Req with a matching rid (or conversation_id alias). The Mamba
    state is held in CPU host tensors here and moved into a freshly allocated
    pool slot at consumption time via inject_state_to_pool.
    """

    conv_states: List[torch.Tensor]
    temporal_states: torch.Tensor
    fill_ids: Optional[List[int]]
    conversation_id: str
    timestamp: float
