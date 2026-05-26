---
name: engram-hooks-off
description: Deactivate the Engram git hooks — unset core.hooksPath, or restore Husky with --restore-husky.
argument-hint: "[--restore-husky]"
allowed-tools:
  - Bash(git config:*)
---
<objective>
Deactivate the Engram git hooks. By default this unsets `core.hooksPath`
(returning to git's default `.git/hooks`). With `--restore-husky` it points
`core.hooksPath` at `.husky` instead. Always verify and print the resulting
state.
</objective>

<process>
Arguments: `$ARGUMENTS`

Run exactly the following block. The `--restore-husky` flag is detected from the
arguments above.

```bash
ARGS="$ARGUMENTS"
prior="$(git config --get core.hooksPath || true)"
echo "prior core.hooksPath : ${prior:-<unset>}"

if [[ "${ARGS}" == *"--restore-husky"* ]]; then
    git config core.hooksPath .husky
    now="$(git config --get core.hooksPath)"
    if [[ "${now}" == ".husky" ]]; then
        echo "now   core.hooksPath : ${now}"
        echo "OK: restored Husky hooks path (.husky)."
    else
        echo "ERROR: core.hooksPath is '${now}', expected '.husky'." >&2
        exit 1
    fi
else
    git config --unset core.hooksPath || true
    now="$(git config --get core.hooksPath || true)"
    if [[ -z "${now}" ]]; then
        echo "now   core.hooksPath : <unset>"
        echo "OK: Engram hooks deactivated; git falls back to .git/hooks."
    else
        echo "ERROR: core.hooksPath still set to '${now}' after unset." >&2
        exit 1
    fi
fi
```

Report the prior value and the resulting state back to the user.
</process>
