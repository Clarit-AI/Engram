#!/usr/bin/env bash
# preflight_base.sh — verify repo is within 5 commits of origin/main
# and has no unresolved conflict markers in tracked files.
#
# Exit 0 — clean; safe to proceed.
# Exit 1 — diagnostic printed; do not run benchmark.
#
# Usage: bash scripts/benchmarks/preflight_base.sh

set -euo pipefail

PASS=0
FAIL=1
MAX_COMMITS_BEHIND=5

echo "=== preflight_base: repo sync check ==="

# ---- 1. Fetch latest origin/main ----
echo "[preflight] Fetching origin/main ..."
git fetch origin main 2>&1

# ---- 2. Compare HEAD to origin/main ----
BEHIND=$(git rev-list --count HEAD..origin/main 2>/dev/null || echo "error")
AHEAD=$(git rev-list --count origin/main..HEAD 2>/dev/null || echo "error")

if [[ "$BEHIND" == "error" || "$AHEAD" == "error" ]]; then
  echo "[preflight] ERROR: could not compute ahead/behind count." >&2
  exit $FAIL
fi

echo "[preflight] HEAD is ${AHEAD} commit(s) ahead and ${BEHIND} commit(s) behind origin/main."

if [[ "$BEHIND" -gt "$MAX_COMMITS_BEHIND" ]]; then
  echo "[preflight] FAIL: HEAD is ${BEHIND} commits behind origin/main (max allowed: ${MAX_COMMITS_BEHIND})." >&2
  echo "[preflight] Run: git merge origin/main" >&2
  exit $FAIL
fi

# ---- 3. Check for unresolved conflict markers ----
echo "[preflight] Checking for unresolved conflict markers ..."

CONFLICT_FILES=$(git diff --name-only --diff-filter=U 2>/dev/null) || true
if [[ -n "$CONFLICT_FILES" ]]; then
  echo "[preflight] FAIL: unresolved merge conflicts in:" >&2
  echo "$CONFLICT_FILES" >&2
  exit $FAIL
fi

# Also grep tracked text files for inline markers (catches manually-added markers)
MARKER_FILES=$(git grep -l "^<<<<<<< \|^>>>>>>> \|^=======$" -- '*.py' '*.sh' '*.json' '*.md' 2>/dev/null || true)
if [[ -n "$MARKER_FILES" ]]; then
  echo "[preflight] FAIL: conflict markers found in:" >&2
  echo "$MARKER_FILES" >&2
  exit $FAIL
fi

echo ""
echo "[preflight] PASS — repo is clean and within ${MAX_COMMITS_BEHIND} commits of origin/main."
exit $PASS
