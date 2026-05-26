#!/usr/bin/env bash
# start-feature.sh: create a feature branch off the latest origin/main.
# Usage: .engram/hooks/start-feature.sh <type>/<slug>
#   e.g. .engram/hooks/start-feature.sh feat/snapshot-keying
set -euo pipefail

BRANCH="${1-}"

if [[ -z "$BRANCH" ]]; then
    echo >&2 "Usage: $0 <type>/<slug>   (e.g. feat/snapshot-keying)"
    exit 1
fi

if ! [[ "$BRANCH" =~ ^(feat|fix|chore)/[a-z0-9-]+$ ]]; then
    echo >&2 "ERROR: Branch name '${BRANCH}' is invalid."
    echo >&2 "       Must match (feat|fix|chore)/[a-z0-9-]+  — e.g. feat/my-feature"
    exit 1
fi

# Refuse to clobber an existing local branch
if git show-ref --verify --quiet "refs/heads/${BRANCH}"; then
    echo >&2 "ERROR: Branch '${BRANCH}' already exists locally. Choose a different name."
    exit 1
fi

# Refuse to clobber an existing remote branch
git fetch origin --quiet
if git show-ref --verify --quiet "refs/remotes/origin/${BRANCH}"; then
    echo >&2 "ERROR: Branch '${BRANCH}' already exists on origin. Choose a different name."
    exit 1
fi

BASE_SHA="$(git rev-parse --short origin/main)"
git checkout -b "${BRANCH}" origin/main

echo ""
echo "Branch : ${BRANCH}"
echo "Base   : ${BASE_SHA}  (origin/main)"
echo ""
echo "Next steps:"
echo "  git push -u origin ${BRANCH}"
echo "  Open a PR when ready — do not push directly to main."
