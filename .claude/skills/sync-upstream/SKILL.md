---
name: sync-upstream
description: |
  Sync upstream huggingface/lerobot main branch to local main and cherry-pick new commits to dev branch.
  Use when: (1) user says "sync upstream", "update from upstream", "pull upstream", (2) user wants to keep dev branch in sync with upstream, (3) user needs to merge upstream changes into their dev branch, (4) user asks to "pick commits from main to dev".
  This skill handles: fetching upstream, resetting main to upstream/main, identifying new commits, and cherry-picking them to dev with conflict detection.
---

# Sync Upstream

Update upstream main branch and auto cherry-pick new commits to dev branch.

**Logic:**
1. Hard-reset local `main` to `upstream/main` (force sync)
2. Find commits in `main` but not in `dev`: `git log $(git merge-base dev main)..main`
3. Auto cherry-pick those commits to `dev`
4. On conflict: stop and ask user for resolution

## Quick Run

```bash
bash .claude/skills/sync-upstream/sync-upstream.sh
```

Or with dry-run (preview only):

```bash
bash .claude/skills/sync-upstream/sync-upstream.sh --dry-run
```

## Output Handling

### Case 1: Success (No Conflicts)
Output ends with:
```
╔════════════════════════════════════════════════════════════╗
║  ✓ SYNC COMPLETE                                           ║
╚════════════════════════════════════════════════════════════╝
```
→ Done. No further action needed.

### Case 2: Conflict Detected
Output shows:
```
╔════════════════════════════════════════════════════════════╗
║  ✗ CONFLICT DETECTED                                       ║
╠════════════════════════════════════════════════════════════╣
║ Commit: <hash> <message>
╠════════════════════════════════════════════════════════════╣
║ Conflicted files:
║   <file1>
║   <file2>
╚════════════════════════════════════════════════════════════╝
```

**Action:** Ask user how to resolve each conflicted file:

```bash
# 1. Show current status
git status
git diff --name-only --diff-filter=U

# 2. For each file, ask user:
#    "Keep main version (theirs), keep dev version (ours), or manual edit?"
```

Then apply resolution:

```bash
# Keep main version (--theirs = incoming from main)
git checkout --theirs <file>
git add <file>

# Keep dev version (--ours = current dev)
git checkout --ours <file>
git add <file>

# Manual edit: user edits file, then
git add <file>
```

After all files resolved:

```bash
git cherry-pick --continue
```

Then **re-run the script** to continue with remaining commits.

## Manual Commands (Fallback)

If script fails, run manually:

```bash
# Step 1: Sync main
git fetch upstream
git checkout main
git reset --hard upstream/main

# Step 2: Find commits to pick
MERGE_BASE=$(git merge-base dev main)
git log --reverse --oneline "${MERGE_BASE}..main"

# Step 3: Cherry-pick to dev
git checkout dev
git cherry-pick "${MERGE_BASE}..main"
```
