"""Root conftest.py — ensures the repo root is on sys.path for all pytest runs."""

import sys
from pathlib import Path

# Add the repository root so that `import benchmark.longmemeval.*` works
# regardless of where pytest is invoked from.
_ROOT = Path(__file__).parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
