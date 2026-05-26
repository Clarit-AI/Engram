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

echo "engram-gate: ruff check (F401,F821) on python/ test/ benchmark/ examples/ ..."
python -m ruff check --select=F401,F821 python/ test/ benchmark/ examples/ || ruff_rc=$?

echo ""
echo "engram-gate: pytest -q test/srt/cpu/ ..."
python -m pytest -q test/srt/cpu/ || pytest_rc=$?

echo ""
echo "===== engram-gate summary ====="
[[ ${ruff_rc}   -eq 0 ]] && echo "ruff   : PASS" || echo "ruff   : FAIL (exit ${ruff_rc})"
[[ ${pytest_rc} -eq 0 ]] && echo "pytest : PASS" || echo "pytest : FAIL (exit ${pytest_rc})"

if [[ ${ruff_rc} -ne 0 || ${pytest_rc} -ne 0 ]]; then
    exit 1
fi
echo "engram-gate: all gates passed."
```

Report each step's pass/fail and the overall result back to the user.
</process>
