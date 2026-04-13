#!/bin/bash
set -e

CKPT_BASE="/root/autodl-tmp/outputs/smolvla_so101/20260413_152921/checkpoints"
REPO_ID="Atticuxz/smolvla_so101"

# Upload last 5 checkpoints, each as a separate branch
ls -d "$CKPT_BASE"/*/  | sort | tail -5 | while read dir; do
    step=$(basename "$dir")
    echo "==> Uploading step-$step ..."
    hf upload "$REPO_ID" "${dir}pretrained_model" . \
        --revision "step-$step" \
        --commit-message "Upload checkpoint step-$step"
done

LAST_DIR=$(ls -d "$CKPT_BASE"/*/ | sort | tail -1)
LAST_STEP=$(basename "$LAST_DIR")
echo "==> Promoting step-$LAST_STEP to main ..."
hf upload "$REPO_ID" "${LAST_DIR}pretrained_model" . \
    --revision main \
    --commit-message "Promote step-$LAST_STEP to main"

echo ""
echo "Done. Last checkpoint (step-$LAST_STEP) is now on main."
