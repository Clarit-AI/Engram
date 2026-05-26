---
name: engram-hooks-on
description: Activate the Engram git hooks (pre-push gate + start-feature helper) by pointing core.hooksPath at .engram/hooks.
allowed-tools:
  - Bash(git config:*)
---
<objective>
Activate the Engram git hooks for this checkout by setting `core.hooksPath` to
`.engram/hooks`. Surface any prior value first so the user knows if a Husky (or
other) hooks path was overridden, then verify the activation.
</objective>

<process>
Run exactly the following block. Do not improvise additional steps.

```bash
prior="$(git config --get core.hooksPath || true)"
echo "prior core.hooksPath : ${prior:-<unset>}"

git config core.hooksPath .engram/hooks

now="$(git config --get core.hooksPath)"
echo "now   core.hooksPath : ${now}"

if [[ "${now}" == ".engram/hooks" ]]; then
    echo "OK: Engram hooks activated (pre-push gate + start-feature helper live)."
else
    echo "ERROR: core.hooksPath is '${now}', expected '.engram/hooks'." >&2
    exit 1
fi

if [[ -n "${prior}" && "${prior}" != ".engram/hooks" ]]; then
    echo "NOTE: previous hooks path '${prior}' was overridden." \
         "Run /engram-hooks-off --restore-husky to restore it if it was .husky."
fi
```

Report the prior value, the new value, and the one-line activation
confirmation back to the user.
</process>
