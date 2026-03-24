#!/bin/bash
# Sync upstream main to local main and cherry-pick new commits to dev
# Usage: ./sync-upstream.sh [--dry-run]

set -e

DRY_RUN=${1:-""}

if [ "$DRY_RUN" = "--dry-run" ]; then
    echo "=== DRY RUN MODE (no changes will be made) ==="
fi

echo "=== Fetching upstream ==="
if [ "$DRY_RUN" != "--dry-run" ]; then
    git fetch upstream
else
    echo "[dry-run] git fetch upstream"
fi

echo "=== Syncing main to upstream/main ==="
if [ "$DRY_RUN" != "--dry-run" ]; then
    git checkout main
    git reset --hard upstream/main
else
    echo "[dry-run] git checkout main"
    echo "[dry-run] git reset --hard upstream/main"
fi

echo "=== Finding commits to cherry-pick ==="
MERGE_BASE=$(git merge-base dev main)
NEW_COMMITS=$(git log --reverse --format="%H" "${MERGE_BASE}..main")

if [ -z "$NEW_COMMITS" ]; then
    echo "✓ No new commits to cherry-pick. Dev is up to date with main."
    exit 0
fi

echo "Commits to cherry-pick:"
git log --reverse --format="  %h %s" "${MERGE_BASE}..main"

if [ "$DRY_RUN" = "--dry-run" ]; then
    echo ""
    echo "=== DRY RUN complete (no changes made) ==="
    exit 0
fi

echo ""
echo "=== Switching to dev and cherry-picking ==="
git checkout dev

for commit in $NEW_COMMITS; do
    msg=$(git log -1 --format='%h %s' "$commit")
    echo ""
    echo "Cherry-picking: $msg"

    if git cherry-pick "$commit" --no-commit 2>/dev/null; then
        git commit -m "$(git log -1 --format='%B' "$commit")" --no-edit
        echo "  ✓ Picked cleanly"
    else
        echo ""
        echo "╔════════════════════════════════════════════════════════════╗"
        echo "║  ✗ CONFLICT DETECTED                                       ║"
        echo "╠════════════════════════════════════════════════════════════╣"
        echo "║ Commit: $msg"
        echo "╠════════════════════════════════════════════════════════════╣"
        echo "║ Conflicted files:"
        git diff --name-only --diff-filter=U | sed 's/^/║   /'
        echo "╠════════════════════════════════════════════════════════════╣"
        echo "║ To resolve:                                                ║"
        echo "║   1. Edit conflicted files                                 ║"
        echo "║   2. git add .                                             ║"
        echo "║   3. git cherry-pick --continue                            ║"
        echo "║   4. Re-run this script to continue with remaining commits ║"
        echo "╚════════════════════════════════════════════════════════════╝"
        exit 1
    fi
done

echo ""
echo "╔════════════════════════════════════════════════════════════╗"
echo "║  ✓ SYNC COMPLETE                                           ║"
echo "╚════════════════════════════════════════════════════════════╝"
git log --oneline -3
