# RL Token 单任务验证运行计划

本文档记录当前代码完成后的用户侧运行顺序。目标是先在 **LIBERO-Spatial task 0** 跑通最小闭环，再扩大到更多 task。

## 0. 前置检查

确认当前分支和依赖：

```bash
git branch --show-current
uv sync --extra "pi" --extra "test"
```

如果要跑 LIBERO 仿真，确认本机已有 LIBERO 依赖、MuJoCo/EGL 环境、`lerobot/pi05_libero` 权重访问权限、`lerobot/libero` 数据集缓存或 HuggingFace 网络访问。

## 1. 单任务 SFT baseline

先拿同一个 task 的 SFT-only 成功率和步数分布做对照：

```bash
LIBERO_SUITE=libero_spatial bash dev.sh eval_baseline \
  --env.task_ids='[0]' \
  --eval.n_episodes=5
```

期望：能启动 LIBERO-Spatial task 0，输出成功率、平均步数和吞吐率指标。若这里失败，先修 LIBERO/π0.5 环境，不进入 RL Token 训练。

## 2. 阶段一：训练 task 0 RL Token

先跑 20 步烟雾测试：

```bash
LIBERO_SUITE=libero_spatial bash dev.sh train_token \
  --dataset.task_index=0 \
  --steps=20
```

烟雾测试通过后跑正式 5000 步：

```bash
LIBERO_SUITE=libero_spatial bash dev.sh train_token \
  --dataset.task_index=0 \
  --steps=5000
```

期望 checkpoint：

```text
outputs/rltoken/encoder_decoder/libero_spatial/task_00/step_005000.safetensors
```

记录到 `docs/plan.md`：

- `L_ro` 初始值、末段平台值
- checkpoint 实际路径
- wandb run id（如果启用）

## 3. 阶段二：task 0 在线 TD3 smoke

用阶段一 checkpoint 跑最小在线训练烟雾测试：

```bash
LIBERO_SUITE=libero_spatial bash dev.sh train_online \
  --task_index=0 \
  --rl_token_checkpoint=outputs/rltoken/encoder_decoder/libero_spatial/task_00/step_005000.safetensors \
  --warmup_steps=2 \
  --max_env_steps=20
```

期望：VLA-only warmup 能写入 replay buffer，actor/critic 至少完成一次 update，并保存 online checkpoint。

如果 smoke 失败，优先检查：

- action 后处理尺度是否与 LIBERO action space 一致
- `LiberoChunkEnv.step()` 中稀疏 reward / success 终止是否正常
- `LeRobotPI05Adapter` 输出的 `z_rl + proprio` shape 是否为 `[1, 2056]`
- Stage 1 checkpoint metadata 是否能正确加载出 2048D `RLTokenModel`

## 4. 阶段二：task 0 正式试跑

smoke 通过后再扩大训练预算：

```bash
LIBERO_SUITE=libero_spatial bash dev.sh train_online \
  --task_index=0 \
  --rl_token_checkpoint=outputs/rltoken/encoder_decoder/libero_spatial/task_00/step_005000.safetensors \
  --warmup_steps=1000 \
  --max_env_steps=20000
```

观察指标：

- `episode/success` 是否不低于 SFT baseline 太多
- `episode/steps` 是否开始下降
- `loss/critic` 是否有限且无持续爆炸
- `q/q1_mean`、`q/q2_mean` 是否没有明显发散

## 5. 扩展到 LIBERO-Spatial 10 个 task

task 0 跑通后再批量训练 Stage 1：

```bash
for t in 0 1 2 3 4 5 6 7 8 9; do
  LIBERO_SUITE=libero_spatial bash dev.sh train_token \
    --dataset.task_index=$t \
    --steps=5000
done
```

Stage 2 仍建议逐个 task 启动，不要一开始并发，以便定位环境和策略问题。

## 当前代码状态

- Stage 1 已迁移为 rlt-openpi 风格 `RLTokenModel`，`z_rl` 为 2048D，训练时剥离语言 token。
- Stage 2 已迁移 rlt-openpi 的 Actor / TwinQCritic / ReplayBuffer / TD3 更新 / RolloutWorker / OnlineTrainer。
- LeRobot 适配层为 `src/lerobot/rltoken/adapter.py` 和 `src/lerobot/rltoken/libero_chunk_env.py`。
- 当前优先级是跑通 task 0 的真实 LIBERO 环境闭环，而不是继续扩展新功能。
