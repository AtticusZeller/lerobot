#!/bin/bash
set -e

# Dev wrappers for RL Token 仿真复现工作流。
# 真机 SO-101 wrapper 已归档；如需历史，见 docs/archive/so101/。

PI05_CKPT="${PI05_CKPT:-lerobot/pi05_libero_finetuned}"
LIBERO_SUITE="${LIBERO_SUITE:-libero_spatial}"   # libero_spatial | libero_object | libero_goal | libero_10
CONFIG="${CONFIG:-experiments/rltoken_pi05_libero.yaml}"
WANDB_PROJECT="${WANDB_PROJECT:-rltoken-pi05-libero}"
export WANDB_PROJECT

# If LEROBOT_OUTPUT_ROOT is set, redirect each command's default output dir to
# that root (data disk on AutoDL). eval_baseline already reads it via
# EvalPipelineConfig; the other commands take a flag, so we inject one when the
# user hasn't passed --<flag> explicitly.
inject_output_flag() {
    local flag=$1
    local subpath=$2
    shift 2
    if [[ -z "${LEROBOT_OUTPUT_ROOT:-}" ]]; then
        return
    fi
    for arg in "$@"; do
        case "$arg" in
            --${flag}|--${flag}=*) return ;;
        esac
    done
    printf -- '--%s=%s/%s\n' "$flag" "$LEROBOT_OUTPUT_ROOT" "$subpath"
}

usage() {
    cat <<EOF
Usage: dev.sh <command> [options]

Commands:
  eval_baseline    [options]              跑 pi05_libero_finetuned 在 LIBERO 上的 SFT 基线（成功率 + 步数分布）
  train_token      [options]              阶段一：RL Token 编码器-解码器离线训练
  train_online     --rl_token_checkpoint PATH [options]
                                           阶段二：冻结编码器，TD3 actor-critic 在线训练
  eval_throughput  --ckpt PATH [options]  评估吞吐率（平均步数 / 吞吐率 / 成功率）

环境变量：
  PI05_CKPT       冻结主干检查点，默认 ${PI05_CKPT}
  LIBERO_SUITE    LIBERO 子任务集，默认 ${LIBERO_SUITE}
  CONFIG          训练配置 yaml，默认 ${CONFIG}
  WANDB_PROJECT   W&B 项目名，默认 ${WANDB_PROJECT}
  LEROBOT_OUTPUT_ROOT
                  输出根目录前缀；未显式传 --output_dir / --save_dir 时，
                  各子命令默认输出会改写到 <前缀>/<subpath>/...：
                    eval_baseline   -> <前缀>/eval/<日期>/<时间>_<job_name>/   (由 EvalPipelineConfig 读取)
                    train_token     -> <前缀>/rltoken/encoder_decoder/<suite>/task_NN/
                    train_online    -> <前缀>/rltoken/online/<run_name>/
                    eval_throughput -> <前缀>/baseline/
                  显式传 --output_dir=PATH 或 --save_dir=PATH 时优先使用 PATH

示例：
  LIBERO_SUITE=libero_spatial bash dev.sh eval_baseline --env.task_ids='[0]'
  LEROBOT_OUTPUT_ROOT=/root/autodl-tmp/outputs \\
    LIBERO_SUITE=libero_spatial bash dev.sh eval_baseline --env.task_ids='[0]'
  LIBERO_SUITE=libero_spatial bash dev.sh eval_baseline \\
    --env.task_ids='[0]' --output_dir=/root/autodl-tmp/outputs/manual_eval

参考文档：docs/rltoken_plan.md
EOF
    exit 1
}

cmd_eval_baseline() {
    uv run python -m lerobot.scripts.lerobot_eval \
        --policy.path="$PI05_CKPT" \
        --env.type=libero \
        --env.task="$LIBERO_SUITE" \
        "$@"
}

cmd_train_token() {
    local extra
    extra=$(inject_output_flag output_dir rltoken/encoder_decoder "$@")
    uv run python -m lerobot.rltoken.train_token \
        --pretrained="$PI05_CKPT" \
        --suite="$LIBERO_SUITE" \
        --yaml_config="$CONFIG" \
        ${extra:+$extra} "$@"
}

cmd_train_online() {
    local extra
    extra=$(inject_output_flag save_dir rltoken/online "$@")
    uv run python -m lerobot.rltoken.train_online \
        --pretrained="$PI05_CKPT" \
        --suite="$LIBERO_SUITE" \
        --yaml_config="$CONFIG" \
        ${extra:+$extra} "$@"
}

cmd_eval_throughput() {
    local extra
    extra=$(inject_output_flag output_dir baseline "$@")
    uv run python -m lerobot.rltoken.eval_throughput \
        --suite="$LIBERO_SUITE" \
        ${extra:+$extra} "$@"
}

cmd="${1:-}"
shift 2>/dev/null || true
case "$cmd" in
    eval_baseline)   cmd_eval_baseline   "$@" ;;
    train_token)     cmd_train_token     "$@" ;;
    train_online)    cmd_train_online    "$@" ;;
    eval_throughput) cmd_eval_throughput "$@" ;;
    *)               usage ;;
esac
