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
Usage: dev.sh <command>

Commands:
    sync        Sync training outputs from remote AutoDL machine to local
    sync-pi05   Sync only pi05_expert_so101
    sync-smolvla Sync only smolvla_so101
EOF
    exit 1
}

cmd_sync() {
    local TARGET="${1:-}"

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

    local EXCLUDE_ARGS="--exclude '*.tfevents*' --exclude '*.mp4'"

    case "$TARGET" in
        pi05)
            echo "Syncing: pi05_expert_so101"
            sshpass -p "$REMOTE_PASSWORD" rsync -avzP \
                -e "ssh -p $REMOTE_PORT -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null" \
                $EXCLUDE_ARGS \
                "root@$REMOTE_HOST:$REMOTE_PATH/pi05_expert_so101/" "$LOCAL_PATH/"
            ;;
        smolvla)
            echo "Syncing: smolvla_so101"
            sshpass -p "$REMOTE_PASSWORD" rsync -avzP \
                -e "ssh -p $REMOTE_PORT -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null" \
                $EXCLUDE_ARGS \
                "root@$REMOTE_HOST:$REMOTE_PATH/smolvla_so101/" "$LOCAL_PATH/"
            ;;
        "")
            echo "Syncing: all training outputs"
            sshpass -p "$REMOTE_PASSWORD" rsync -avzP \
                -e "ssh -p $REMOTE_PORT -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null" \
                $EXCLUDE_ARGS \
                "root@$REMOTE_HOST:$REMOTE_PATH/" "$LOCAL_PATH/"
            ;;
        *)
            echo "Unknown target: $TARGET"
            usage
            ;;
    esac

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
        cmd_sync pi05
        ;;
    sync-smolvla)
        cmd_sync smolvla
        ;;
    *)
        usage
        ;;
esac
