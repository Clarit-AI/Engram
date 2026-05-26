---
name: engram-pr
description: Open a PR for the current Engram branch with the canonical body template; type prefix inferred from the branch name.
argument-hint: "<title>"
allowed-tools:
  - Bash(git rev-parse:*)
  - Bash(git push:*)
  - Bash(gh pr create:*)
  - Bash(gh pr view:*)
---
<objective>
Open a pull request for the current branch against `Clarit-AI/Engram`. The
conventional-commit type prefix (`feat` / `fix` / `chore`) is inferred from the
branch name and prepended to the supplied title. If the branch has no upstream
yet, push it first. The PR body uses the canonical template: Summary,
Acceptance Criteria (three empty checkboxes), and Reference (both Linear
runbooks).
</objective>

<process>
Arguments (the title portion only — do NOT include a type prefix): `$ARGUMENTS`

Run exactly the following block.

```bash
set -euo pipefail
TITLE="$ARGUMENTS"
if [[ -z "${TITLE}" ]]; then
    echo "ERROR: missing required <title> argument." >&2
    exit 1
fi

BRANCH="$(git rev-parse --abbrev-ref HEAD)"
case "${BRANCH}" in
    feat/*)  TYPE="feat"  ;;
    fix/*)   TYPE="fix"   ;;
    chore/*) TYPE="chore" ;;
    *)
        echo "ERROR: branch '${BRANCH}' has no feat/|fix/|chore/ prefix; cannot infer type." >&2
        exit 1
        ;;
esac

# Push with upstream tracking if the branch isn't published yet.
if ! git rev-parse --abbrev-ref --symbolic-full-name '@{upstream}' >/dev/null 2>&1; then
    git push -u origin "${BRANCH}"
fi

BODY="$(cat <<'EOF'
## Summary

<!-- Describe the change in 1-3 sentences. -->

## Acceptance Criteria

- [ ]
- [ ]
- [ ]

## Reference

- Feature & Issue Runbook: https://linear.app/khaentertainment/document/engram-feature-and-issue-runbook-d8916813a279
- Upstream Merge Runbook: https://linear.app/khaentertainment/document/engram-upstream-merge-runbook-a2451e78eecc
EOF
)"

gh pr create \
    --repo Clarit-AI/Engram \
    --title "${TYPE}: ${TITLE}" \
    --body "${BODY}"

echo ""
echo "PR URL:"
gh pr view --json url --jq '.url'
```

Report the resulting PR URL back to the user.
</process>
