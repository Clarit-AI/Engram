"""Pytest configuration for GraphWalks benchmark tests.

Adds the repo root to sys.path so that ``import benchmark.graphwalks.*``
works when pytest is invoked from within the benchmark/graphwalks/ directory
or from the repo root.
"""

import sys
from pathlib import Path

# Repo root is three levels up: benchmark/graphwalks/conftest.py → repo root
_REPO_ROOT = Path(__file__).parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
