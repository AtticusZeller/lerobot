#!/bin/bash
set -e

# Load environment variables from .env if present
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -f "$SCRIPT_DIR/.env" ]; then
    set -a
    source "$SCRIPT_DIR/.env"
    set +a
fi

# Default values
REMOTE_HOST="${HOST:-}"
REMOTE_PORT="${PORT:-22}"
REMOTE_PASSWORD="${PASSWORD:-}"
REMOTE_PATH="/root/autodl-tmp/outputs/"
LOCAL_PATH="$(pwd)/output"

usage() {
    cat <<EOF
Usage: dev.sh <command> [options]

Commands:
    sync [target] [options]   Sync training outputs from remote AutoDL
                               target: pi05 | smolvla | (empty for all)

Options:
    --step <step>       Sync a specific checkpoint step (e.g. --step 018000)
    --latest <n>        Sync only the latest <n> checkpoints (default: all)
    --include-training  Include training_state/ directories (excluded by default)

Examples:
    dev.sh sync                         Sync all outputs
    dev.sh sync pi05                    Sync pi05_expert_so101
    dev.sh sync --latest 3              Sync only latest 3 checkpoints
    dev.sh sync --step 018000           Sync a specific checkpoint step
    dev.sh sync pi05 --step 002000      Sync specific step from pi05
    dev.sh sync-pi05                    Shorthand for 'sync pi05'
    dev.sh sync-smolvla                 Shorthand for 'sync smolvla'
EOF
    exit 1
}

# Parse options after the target argument
parse_opts() {
    CKPT_STEP=""
    CKPT_LATEST=""
    INCLUDE_TRAINING=false

    while [[ $# -gt 0 ]]; do
        case "$1" in
            --step)
                CKPT_STEP="$2"
                shift 2
                ;;
            --latest)
                CKPT_LATEST="$2"
                shift 2
                ;;
            --include-training)
                INCLUDE_TRAINING=true
                shift
                ;;
            --)
                shift
                break
                ;;
            *)
                break
                ;;
        esac
    done
}

cmd_sync() {
    local TARGET="${1:-}"
    shift 2>/dev/null || true
    parse_opts "$@"

    if [ -z "$REMOTE_HOST" ] || [ -z "$REMOTE_PASSWORD" ]; then
        echo "Error: HOST and PASSWORD must be set in .env"
        echo "Required variables:"
        echo "  HOST=<remote_ip>"
        echo "  PORT=<ssh_port>"
        echo "  PASSWORD=<ssh_password>"
        exit 1
    fi

    if ! command -v sshpass &> /dev/null; then
        echo "Error: sshpass is not installed."
        exit 1
    fi

    mkdir -p "$LOCAL_PATH"

    echo "Starting sync from AutoDL ($REMOTE_HOST:$REMOTE_PATH)..."
    echo "Local destination: $LOCAL_PATH"

    local SSH_OPTS="ssh -p $REMOTE_PORT -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null"

    # Build exclude args: always exclude tfevents and mp4
    local EXCLUDE_ARGS="--exclude '*.tfevents*' --exclude '*.mp4'"

    # Exclude training_state by default (optimizer states are large and not needed for inference)
    if [ "$INCLUDE_TRAINING" = false ]; then
        EXCLUDE_ARGS="$EXCLUDE_ARGS --exclude '*/training_state/'"
    fi

    # Build the remote source path (rsync format) and remote-only path (for SSH commands)
    local REMOTE_SRC REMOTE_DIR
    case "$TARGET" in
        pi05)
            echo "Target: pi05_expert_so101"
            REMOTE_DIR="${REMOTE_PATH%/}/pi05_expert_so101"
            ;;
        smolvla)
            echo "Target: smolvla_so101"
            REMOTE_DIR="${REMOTE_PATH%/}/smolvla_so101"
            ;;
        "")
            echo "Target: all training outputs"
            REMOTE_DIR="${REMOTE_PATH%/}"
            ;;
        *)
            echo "Unknown target: $TARGET"
            usage
            ;;
    esac
    REMOTE_SRC="root@$REMOTE_HOST:${REMOTE_DIR%/}/"

    # --step: sync a specific checkpoint only
    if [ -n "$CKPT_STEP" ]; then
        echo "Checkpoint step: $CKPT_STEP"
        # Structure: REMOTE_DIR/<run_id>/checkpoints/<step>/
        local RUN_DIRS
        RUN_DIRS=$(sshpass -p "$REMOTE_PASSWORD" $SSH_OPTS \
            root@$REMOTE_HOST \
            "ls -1d $REMOTE_DIR/*/checkpoints/$CKPT_STEP" || true)

        if [ -z "$RUN_DIRS" ]; then
            echo "Error: checkpoint step $CKPT_STEP not found on remote"
            echo "Remote dir: $REMOTE_DIR"
            exit 1
        fi

        for ckpt_path in $RUN_DIRS; do
            local run_name
            run_name=$(basename "$(dirname "$(dirname "$ckpt_path")")")
            echo "Syncing: $run_name/checkpoints/$CKPT_STEP"
            sshpass -p "$REMOTE_PASSWORD" rsync -avzP --compress-level=6 \
                -e "$SSH_OPTS" \
                $EXCLUDE_ARGS \
                "root@$REMOTE_HOST:$ckpt_path/" "$LOCAL_PATH/$run_name/checkpoints/$CKPT_STEP/"
        done

        echo "Sync finished."
        return
    fi

    # --latest N: sync only the latest N checkpoints from each run
    if [ -n "$CKPT_LATEST" ]; then
        echo "Syncing latest $CKPT_LATEST checkpoint(s) per run"

        # Step 1: list run directories (e.g. 20260413_152921)
        local RUN_IDS
        RUN_IDS=$(sshpass -p "$REMOTE_PASSWORD" $SSH_OPTS \
            root@$REMOTE_HOST \
            "ls -1d $REMOTE_DIR/*/checkpoints" || true)

        if [ -z "$RUN_IDS" ]; then
            echo "No checkpoints found on remote"
            echo "Tried: ls -1d $REMOTE_DIR/*/checkpoints"
            exit 1
        fi

        for ckpts_dir in $RUN_IDS; do
            local run_name
            run_name=$(basename "$(dirname "$ckpts_dir")")

            # Get latest N checkpoint steps, sorted numerically descending
            local STEPS
            STEPS=$(sshpass -p "$REMOTE_PASSWORD" $SSH_OPTS \
                root@$REMOTE_HOST \
                "ls -1 $ckpts_dir/ | grep -E '^[0-9]+$' | sort -rn | head -n $CKPT_LATEST" || true)

            for step_name in $STEPS; do
                echo "Syncing: $run_name/checkpoints/$step_name"
                sshpass -p "$REMOTE_PASSWORD" rsync -avzP --compress-level=6 \
                    -e "$SSH_OPTS" \
                    $EXCLUDE_ARGS \
                    "root@$REMOTE_HOST:$ckpts_dir/$step_name/" "$LOCAL_PATH/$run_name/checkpoints/$step_name/"
            done

            # Also sync the 'last' symlink
            sshpass -p "$REMOTE_PASSWORD" rsync -avzP --compress-level=6 \
                -e "$SSH_OPTS" \
                $EXCLUDE_ARGS \
                -l \
                "root@$REMOTE_HOST:$ckpts_dir/last" "$LOCAL_PATH/$run_name/checkpoints/" 2>/dev/null || true
        done

        echo "Sync finished."
        return
    fi

    # Default: full sync
    sshpass -p "$REMOTE_PASSWORD" rsync -avzP --compress-level=6 \
        -e "$SSH_OPTS" \
        $EXCLUDE_ARGS \
        "$REMOTE_SRC" "$LOCAL_PATH/"

    echo "Sync finished."
}

# Main dispatcher
COMMAND="${1:-}"
shift 2>/dev/null || true

case "$COMMAND" in
    sync)
        cmd_sync "$@"
        ;;
    sync-pi05)
        cmd_sync pi05 "$@"
        ;;
    sync-smolvla)
        cmd_sync smolvla "$@"
        ;;
    *)
        usage
        ;;
esac
