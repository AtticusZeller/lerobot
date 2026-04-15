#!/bin/bash
set -e

# Dev convenience wrappers for LeRobot SO-101 table-cleanup workflow

# ── Model presets ─────────────────────────────────────────────────────────────
declare -A POLICY_TYPE=([smolvla]=smolvla [pi05]=pi05)
declare -A HF_REPO=(
    [smolvla]="Atticuxz/smolvla_so101"
    [pi05]="Atticuxz/pi05_expert_so101"
)
declare -A CONFIG_FILE=(
    [smolvla]="experiments/smolvla_so101_table_cleanup.yaml"
    [pi05]="experiments/pi05_expert_so101_table_cleanup.yaml"
)

# ── Defaults ──────────────────────────────────────────────────────────────────
TASK="Grab pens and place into box"
HOST="0.0.0.0"
PORT="8080"
ROBOT_TYPE="so101_follower"
ROBOT_PORT="/dev/ttyACM1"
ROBOT_ID="so101_follower"
CAMERAS='{ front: {type: opencv, index_or_path: /dev/video10, width: 640, height: 480, fps: 30}, wrist: {type: intelrealsense, serial_number_or_name: 233522074606, width: 640, height: 480, fps: 30}}'
ACTIONS_PER_CHUNK=50
CHUNK_THRESHOLD=0.5

# ── Usage ─────────────────────────────────────────────────────────────────────
usage() {
    cat <<EOF
Usage: dev.sh <command> [model] [options]

Commands:
  check                        Check hardware connectivity (cameras + robot)
  serve [options]              Start async policy server (gRPC)
  infer [model] [options]      Start async robot client
  train [model] [options]      Train with experiment config

Models: smolvla (default), pi05

Serve options:
  --host HOST         (default: $HOST)
  --port PORT         (default: $PORT)

Infer options:
  --server ADDR       Policy server address (default: localhost:$PORT)
  --ckpt PATH         Override checkpoint (default: HF repo preset)
  --robot-port PORT   (default: $ROBOT_PORT)
  --cameras JSON      Camera config JSON
  --actions N         Actions per chunk (default: $ACTIONS_PER_CHUNK)
  --threshold F       Chunk size threshold (default: $CHUNK_THRESHOLD)
  --debug             Enable action queue visualization

Train options:
  --steps N           Override training steps
  --batch-size N      Override batch size
  --output-dir PATH   Override output dir (default: ./outputs/<model>_so101_<timestamp>)
  --repo-id REPO      Override HF Hub repo_id (default: <preset>_<timestamp>)
  --no-push           Skip pushing to HF Hub
  Any other args forwarded to lerobot-train

Examples:
  dev.sh check
  dev.sh serve
  dev.sh serve --host 0.0.0.0 --port 8080
  dev.sh infer                              # smolvla, local server
  dev.sh infer pi05                         # pi05, local server
  dev.sh infer --server 192.168.1.100:8080  # remote server
  dev.sh infer --ckpt ./output/last/pretrained_model
  dev.sh train
  dev.sh train pi05 --steps 10000
EOF
    exit 1
}

# ── check ─────────────────────────────────────────────────────────────────────
cmd_check() {
    local ok=true

    echo "=== 硬件连通性检查 ==="
    echo ""

    # ── Robot port ──
    echo -n "机械臂 ($ROBOT_PORT): "
    if [[ -e "$ROBOT_PORT" ]]; then
        echo "OK"
    else
        echo "未检测到"
        ok=false
    fi

    # ── Cameras (via lerobot-find-cameras) ──
    echo ""
    echo "Cameras (lerobot-find-cameras):"
    if lerobot-find-cameras 2>&1; then
        echo "  相机检测通过"
    else
        echo "  相机检测失败"
        ok=false
    fi

    # ── Summary ──
    echo ""
    echo "数据集相机 key:"
    uv run python -c "
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
meta = LeRobotDatasetMetadata('Atticuxz/so101-table-cleanup')
for k, v in meta.features.items():
    dtype = v.get('dtype', '')
    if 'image' in dtype or 'video' in dtype:
        key = k.removeprefix('observation.images.')
        shape = v['shape']
        print(f'  {key}: {shape[1]}x{shape[0]} @ {v[\"info\"][\"video.fps\"]}fps')
" || true

    # ── Summary ──
    echo ""
    echo "任务: $TASK"
    echo ""
    if $ok; then
        echo "✓ 硬件检查全部通过"
    else
        echo "✗ 部分检查失败，请查看上方详情"
        return 1
    fi
}

# ── serve ─────────────────────────────────────────────────────────────────────
cmd_serve() {
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --host) HOST="$2"; shift 2 ;;
            --port) PORT="$2"; shift 2 ;;
            *) echo "Unknown option: $1"; usage ;;
        esac
    done

    echo "Starting policy server on ${HOST}:${PORT} ..."
    python -m lerobot.async_inference.policy_server \
        --host="$HOST" --port="$PORT"
}

# ── infer ─────────────────────────────────────────────────────────────────────
cmd_infer() {
    local model="smolvla"
    if [[ $# -gt 0 && ! "$1" =~ ^-- ]]; then
        model="$1"; shift
    fi

    local server="localhost:${PORT}"
    local ckpt="${HF_REPO[$model]:?Unknown model: $model}"
    local ptype="${POLICY_TYPE[$model]}"
    local rport="$ROBOT_PORT"
    local cams="$CAMERAS"
    local actions="$ACTIONS_PER_CHUNK"
    local thresh="$CHUNK_THRESHOLD"
    local debug=""

    while [[ $# -gt 0 ]]; do
        case "$1" in
            --server)     server="$2"; shift 2 ;;
            --ckpt)       ckpt="$2"; shift 2 ;;
            --robot-port) rport="$2"; shift 2 ;;
            --cameras)    cams="$2"; shift 2 ;;
            --actions)    actions="$2"; shift 2 ;;
            --threshold)  thresh="$2"; shift 2 ;;
            --debug)      debug="--debug_visualize_queue_size=True"; shift ;;
            *) echo "Unknown option: $1"; usage ;;
        esac
    done

    echo "Inference: $ptype ($ckpt) → $server"
    python -m lerobot.async_inference.robot_client \
        --server_address="$server" \
        --robot.type="$ROBOT_TYPE" \
        --robot.port="$rport" \
        --robot.id="$ROBOT_ID" \
        --robot.cameras="$cams" \
        --task="$TASK" \
        --policy_type="$ptype" \
        --pretrained_name_or_path="$ckpt" \
        --policy_device="cuda" \
        --actions_per_chunk="$actions" \
        --chunk_size_threshold="$thresh" \
        --aggregate_fn_name="weighted_average" \
        $debug
}

# ── train ─────────────────────────────────────────────────────────────────────
cmd_train() {
    local model="smolvla"
    if [[ $# -gt 0 && ! "$1" =~ ^-- ]]; then
        model="$1"; shift
    fi

    local config="${CONFIG_FILE[$model]:?Unknown model: $model}"
    local extra=()
    local output_dir=""
    local repo_id=""
    local push=""

    while [[ $# -gt 0 ]]; do
        case "$1" in
            --steps)       extra+=("--steps" "$2"); shift 2 ;;
            --batch-size)  extra+=("--batch_size" "$2"); shift 2 ;;
            --output-dir)  output_dir="$2"; shift 2 ;;
            --repo-id)     repo_id="$2"; shift 2 ;;
            --no-push)     push="false"; shift ;;
            *)             extra+=("$1"); shift ;;
        esac
    done

    # Auto-generate timestamped output dir if not explicitly set
    local ts; ts=$(date +%Y%m%d_%H%M)
    if [[ -z "$output_dir" ]]; then
        output_dir="./outputs/${model}_so101_${ts}"
    fi

    # Auto-generate timestamped repo_id for Hub, unless overridden
    if [[ -z "$repo_id" ]]; then
        repo_id="${HF_REPO[$model]}_${ts}"
    fi

    # Build push args
    local hub_args=()
    if [[ "$push" == "false" ]]; then
        hub_args+=("--policy.push_to_hub=false")
    else
        hub_args+=("--policy.push_to_hub=true" "--policy.repo_id=$repo_id")
    fi

    echo "Training: $config → $output_dir"
    [[ "$push" != "false" ]] && echo "Hub: $repo_id"
    lerobot-train --yaml_config="$config" --output_dir="$output_dir" "${hub_args[@]}" "${extra[@]}"
}

# ── Main ──────────────────────────────────────────────────────────────────────
cmd="${1:-}"; shift 2>/dev/null || true
case "$cmd" in
    check) cmd_check ;;
    serve) cmd_serve "$@" ;;
    infer) cmd_infer "$@" ;;
    train) cmd_train "$@" ;;
    *)     usage ;;
esac
