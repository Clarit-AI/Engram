---
name: engram-gate
description: Run the same lint + CPU-test gate as the pre-push hook (ruff F401/F821 + pytest test/srt/cpu/), on demand.
allowed-tools:
  - Bash(python -m ruff:*)
  - Bash(python -m pytest:*)
  - Bash(git rev-parse:*)
---
<objective>
Run the exact gate the `.engram/hooks/pre-push` hook enforces, on demand:
ruff lint (`F401`/`F821`) followed by the CPU-only test suite
(`test/srt/cpu/`). Each step's exit code is captured and reported pass/fail.
The command exits non-zero if either step fails, so it can be chained in
scripts.
</objective>

<process>
Run exactly the following block from the repository root. It mirrors the
pre-push hook's ruff scope and test target verbatim so results match what a
push would gate on.

```bash
cd "$(git rev-parse --show-toplevel)"

ruff_rc=0
pytest_rc=0

# Same interpreter choice as the hook: ENGRAM_PYTHON, else `python`.
PY="${ENGRAM_PYTHON:-python}"
for mod in ruff pytest; do
    if ! "$PY" -c "import ${mod}" >/dev/null 2>&1; then
        echo "engram-gate: CANNOT RUN: '${PY}' has no ${mod}. Nothing was checked." >&2
        echo "             Point ENGRAM_PYTHON at an environment with ruff, pytest and sglang." >&2
        exit 2
    fi
done

echo "engram-gate: ruff check (F401,F821) on python/ test/ benchmark/ examples/ ..."
"$PY" -m ruff check --select=F401,F821 python/ test/ benchmark/ examples/ || ruff_rc=$?

echo ""
echo "engram-gate: pytest -q test/srt/cpu/ ..."
"$PY" -m pytest -q test/srt/cpu/ || pytest_rc=$?

echo ""
echo "===== engram-gate summary ====="
[[ ${ruff_rc}   -eq 0 ]] && echo "ruff   : PASS" || echo "ruff   : FAIL (exit ${ruff_rc})"
[[ ${pytest_rc} -eq 0 ]] && echo "pytest : PASS" || echo "pytest : FAIL (exit ${pytest_rc})"

if [[ ${ruff_rc} -ne 0 || ${pytest_rc} -ne 0 ]]; then
    exit 1
fi
echo "engram-gate: all gates passed."
```

Report each step's pass/fail and the overall result back to the user. Exit code
2 means the gate could not run in this environment: report that as "not run",
never as a pass or a fail.
</process>
