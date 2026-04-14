#!/bin/bash
set -e

# Defaults
DEFAULT_CKPT_BASE="/root/autodl-tmp/outputs/smolvla_so101/20260413_152921/checkpoints"
REPO_ID="Atticuxz/smolvla_so101"

usage() {
    echo "Usage: $0 [OPTIONS] [STEP...]"
    echo ""
    echo "Upload checkpoints to HuggingFace Hub."
    echo ""
    echo "Arguments:"
    echo "  STEP...              Step numbers to upload (e.g. 010000 020000). Default: last 5."
    echo ""
    echo "Options:"
    echo "  -d, --dir DIR        Checkpoint base directory (default: $DEFAULT_CKPT_BASE)"
    echo "  -m, --main STEP      Step to promote to main branch (default: last uploaded step)"
    echo "  -r, --repo REPO_ID   HuggingFace repo ID (default: $REPO_ID)"
    echo "  -h, --help           Show this help"
    exit 0
}

CKPT_BASE="$DEFAULT_CKPT_BASE"
MAIN_STEP=""
STEPS=()

while [[ $# -gt 0 ]]; do
    case $1 in
        -d|--dir)   CKPT_BASE="$2"; shift 2 ;;
        -m|--main)  MAIN_STEP="$2"; shift 2 ;;
        -r|--repo)  REPO_ID="$2"; shift 2 ;;
        -h|--help)  usage ;;
        *)          STEPS+=("$1"); shift ;;
    esac
done

# If no steps specified, use last 5
if [[ ${#STEPS[@]} -eq 0 ]]; then
    while IFS= read -r dir; do
        STEPS+=("$(basename "$dir")")
    done < <(ls -d "$CKPT_BASE"/*/ | sort | tail -5)
fi

# Upload each step as a separate branch
for step in "${STEPS[@]}"; do
    dir="$CKPT_BASE/$step/"
    if [[ ! -d "$dir" ]]; then
        echo "WARNING: $dir does not exist, skipping."
        continue
    fi
    echo "==> Uploading step-$step ..."
    hf upload "$REPO_ID" "${dir}pretrained_model" . \
        --revision "step-$step" \
        --commit-message "Upload checkpoint step-$step"
done

# Determine which step to promote to main
if [[ -z "$MAIN_STEP" ]]; then
    MAIN_STEP="${STEPS[-1]}"
fi

MAIN_DIR="$CKPT_BASE/$MAIN_STEP/"
if [[ ! -d "$MAIN_DIR" ]]; then
    echo "ERROR: main step dir $MAIN_DIR does not exist."
    exit 1
fi

echo "==> Promoting step-$MAIN_STEP to main ..."
hf upload "$REPO_ID" "${MAIN_DIR}pretrained_model" . \
    --revision main \
    --commit-message "Promote step-$MAIN_STEP to main"

echo ""
echo "Done. step-$MAIN_STEP is now on main."
