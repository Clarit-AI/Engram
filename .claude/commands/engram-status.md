---
name: engram-status
description: Show Engram branch status — branch, distance from origin/main, hook state, base SHA, and any open PR.
allowed-tools:
  - Bash(git rev-parse:*)
  - Bash(git fetch:*)
  - Bash(git rev-list:*)
  - Bash(git merge-base:*)
  - Bash(git config:*)
  - Bash(gh pr view:*)
---
<objective>
Print a clean key/value status block for the current Engram checkout: the
current branch, ahead/behind distance from `origin/main`, git-hook activation
state, the merge-base SHA with `origin/main`, and any open PR on the branch.
</objective>

<process>
Run exactly the following block and present its output as the status report.

```bash
set -uo pipefail
git fetch origin --quiet 2>/dev/null || true

BRANCH="$(git rev-parse --abbrev-ref HEAD)"

# Ahead/behind: left = ahead of origin/main, right = behind origin/main.
if counts="$(git rev-list --left-right --count HEAD...origin/main 2>/dev/null)"; then
    AHEAD="$(echo "${counts}" | awk '{print $1}')"
    BEHIND="$(echo "${counts}" | awk '{print $2}')"
else
    AHEAD="?"; BEHIND="?"
fi

HOOKS="$(git config --get core.hooksPath || echo '<unset>')"
BASE="$(git merge-base --short HEAD origin/main 2>/dev/null || git merge-base HEAD origin/main 2>/dev/null || echo '<none>')"
PR="$(gh pr view --json url,state,title --jq '"\(.state)  \(.url)  — \(.title)"' 2>/dev/null || echo 'none')"

printf '%-22s %s\n' "Branch:"            "${BRANCH}"
printf '%-22s %s\n' "Ahead of main:"     "${AHEAD}"
printf '%-22s %s\n' "Behind main:"       "${BEHIND}"
printf '%-22s %s\n' "Hooks (hooksPath):" "${HOOKS}"
printf '%-22s %s\n' "Base (merge-base):" "${BASE}"
printf '%-22s %s\n' "Open PR:"           "${PR}"
```
</process>
