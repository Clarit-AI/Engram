---
name: engram-feature
description: Start an Engram feature branch off the latest origin/main (or a worktree with --worktree).
argument-hint: "<type>/<slug> [--worktree]"
allowed-tools:
  - Bash(bash:*)
  - Bash(git fetch:*)
  - Bash(git worktree:*)
  - Bash(git rev-parse:*)
  - Bash(git show-ref:*)
---
<objective>
Create a properly based Engram feature branch. By default this delegates to
`.engram/hooks/start-feature.sh`, which branches off the latest `origin/main`
and refuses to clobber existing local/remote branches. With `--worktree` it
creates an isolated git worktree at `../engram-<type>-<slug>` instead. Either
way, print the resulting branch name and base SHA.
</objective>

<process>
Arguments: `$ARGUMENTS`

The first argument is the branch name and must match
`(feat|fix|chore)/[a-z0-9-]+`. The optional `--worktree` flag selects worktree
mode. Run exactly the following block.

```bash
set -euo pipefail
ARGS="$ARGUMENTS"

# Parse: first token is the branch, presence of --worktree selects worktree mode.
WORKTREE=0
BRANCH=""
for tok in ${ARGS}; do
    if [[ "${tok}" == "--worktree" ]]; then
        WORKTREE=1
    elif [[ -z "${BRANCH}" ]]; then
        BRANCH="${tok}"
    fi
done

if [[ -z "${BRANCH}" ]]; then
    echo "ERROR: missing required <type>/<slug> argument (e.g. feat/my-feature)." >&2
    exit 1
fi
if ! [[ "${BRANCH}" =~ ^(feat|fix|chore)/[a-z0-9-]+$ ]]; then
    echo "ERROR: branch '${BRANCH}' must match (feat|fix|chore)/[a-z0-9-]+." >&2
    exit 1
fi

if [[ "${WORKTREE}" -eq 1 ]]; then
    # Worktree mode: ../engram-<type>-<slug> off a fresh origin/main.
    WT_DIR="../engram-${BRANCH//\//-}"
    git fetch origin
    git worktree add -b "${BRANCH}" "${WT_DIR}" origin/main
    BASE_SHA="$(git rev-parse --short origin/main)"
    echo ""
    echo "Branch   : ${BRANCH}"
    echo "Worktree : ${WT_DIR}"
    echo "Base     : ${BASE_SHA}  (origin/main)"
else
    # Default mode: local branch via the canonical helper.
    # Invoked via `bash` so it works regardless of the script's execute bit.
    bash .engram/hooks/start-feature.sh "${BRANCH}"
fi
```

Report the resulting branch name and base SHA (and worktree path, in worktree
mode) back to the user.
</process>
