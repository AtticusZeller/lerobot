# RL Token 单任务验证运行计划

本文档记录当前代码完成后的用户侧运行顺序。目标是先在 **LIBERO-Spatial task 0** 跑通最小闭环，再扩大到更多 task。

## 0. 前置检查

确认当前分支和依赖：

```bash
git branch --show-current
# LIBERO 仿真需要 libero extra；egl-probe 在 CMake 4.x 下需要 policy override
CMAKE_POLICY_VERSION_MINIMUM=3.5 uv sync --extra "pi" --extra "test" --extra "libero"
```

确认本机已有 MuJoCo/EGL 环境、`lerobot/pi05_libero_finetuned` 权重访问权限、`HuggingFaceVLA/libero` 数据集缓存或 HuggingFace 网络访问。如 `egl-probe` 构建失败，详见 `docs/bug.md`。

## 1. 单任务 SFT baseline

先拿同一个 task 的 SFT-only 成功率和步数分布做对照：

```bash
LIBERO_SUITE=libero_spatial bash dev.sh eval_baseline \
  --env.task_ids='[0]' \
  --eval.n_episodes=5
```

期望：能启动 LIBERO-Spatial task 0，输出成功率、平均步数和吞吐率指标。若这里失败，先修 LIBERO/π0.5 环境，不进入 RL Token 训练。

如果在 AutoDL 上运行，且希望默认输出保留自动日期/时间目录但写到数据盘，使用：

```bash
LEROBOT_OUTPUT_ROOT=/root/autodl-tmp/outputs \
LIBERO_SUITE=libero_spatial bash dev.sh eval_baseline \
  --env.task_ids='[0]' \
  --eval.n_episodes=5
```

期望输出目录形如：

```text
/root/autodl-tmp/outputs/eval/<日期>/<时间>_<job_name>/
```

已验证结果（2026-05-16，`LIBERO_SUITE=libero_spatial`，`task_id=0`，5 episodes）：

- 成功率：`100.0%`（`5/5`）
- 平均步数：`76.6`
- 成功回合平均步数：`76.6`
- 步数分位数：`p25=74`，`p50=77`，`p75=77`
- 吞吐率：`13.05`（successes / 1000 env steps）
- 总耗时：`186.5s`
- 结果目录：`/root/autodl-tmp/outputs/eval/2026-05-16/19-07-03_libero_pi05/`

逐回合明细：

- `successes=[True, True, True, True, True]`
- `episode_lengths=[77, 74, 74, 77, 81]`
- `sum_rewards=[1.0, 1.0, 1.0, 1.0, 1.0]`

结论：阶段一 baseline 已通过，可进入 task 0 的 RL Token 训练。

## 2. 阶段一：训练 task 0 RL Token

先跑 20 步烟雾测试：

```bash
LEROBOT_OUTPUT_ROOT=/root/autodl-tmp/outputs \
LIBERO_SUITE=libero_spatial bash dev.sh train_token \
  --dataset.task_index=0 \
  --steps=20
```

> `LEROBOT_OUTPUT_ROOT` 现在对所有 `dev.sh` 子命令生效（详见 CLAUDE.md / AGENTS.md "输出路径约定"）。AutoDL 上别忘了带上，否则 checkpoint 落在系统盘。

烟雾测试通过后跑正式 5000 步：

```bash
LEROBOT_OUTPUT_ROOT=/root/autodl-tmp/outputs \
LIBERO_SUITE=libero_spatial bash dev.sh train_token \
  --dataset.task_index=0 \
  --steps=5000
```

显存约束：32GB GPU 上 `batch_size=32` 会 OOM（峰值 ~31GB），当前 `experiments/rltoken_pi05_libero.yaml` 默认 `dataset.batch_size=16`（~24GB），loss 仍稳定下降。改回 32 需 ≥40GB 显存或开 `PYTORCH_ALLOC_CONF=expandable_segments:True` 并接受首步附近 OOM 风险。

期望 checkpoint：

```text
<LEROBOT_OUTPUT_ROOT>/rltoken/encoder_decoder/libero_spatial/task_00/step_005000.safetensors
```

已验证结果（`2026-05-17`，`libero_spatial task 0`，5000 step，batch=16，单 32GB GPU）：

| 项 | 值 |
|---|---|
| L_ro 起点（step 1） | 2.21 |
| L_ro 末段 EMA（step 5000） | 0.291 |
| 1k / 2k / 3k / 4k / 5k EMA | 0.513 / 0.425 / 0.371 / 0.324 / 0.291 |
| wandb run | `rltoken-pi05-libero/quiet-valley-7` (id `uo9fkt4p`) |
| 训练用时 | ~1h15min（含 ~3min 冷启动） |

结论：Stage 1 task 0 已验证收敛，可进入阶段二 smoke。L_ro 0.29 已接近 512:1 压缩下的重构 floor，单 L_ro 不再是后续 Stage 2 是否能学的判据 —— actor/critic 收敛是硬证据。

## 3. 阶段二：task 0 在线 TD3 smoke

用阶段一 checkpoint 跑最小在线训练烟雾测试：

```bash
LIBERO_SUITE=libero_spatial bash dev.sh train_online \
  --task_index=0 \
  --rl_token_checkpoint=/root/autodl-tmp/outputs/rltoken/encoder_decoder/libero_spatial/task_00/step_005000.safetensors \
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
  --rl_token_checkpoint=/root/autodl-tmp/outputs/rltoken/encoder_decoder/libero_spatial/task_00/step_005000.safetensors \
  --warmup_steps=1000 \
  --max_env_steps=20000
```

观察指标：

- `episode/success` 是否不低于 SFT baseline 太多
- `episode/steps` 是否开始下降
- `loss/critic` 是否有限且无持续爆炸
- `q/q1_mean`、`q/q2_mean` 是否没有明显发散

## 5. 扩展到 LIBERO-Spatial 10 个 task

task 0 + Stage 2 跑通后再批量训练 Stage 1。注意以下 for 循环每 task 重启进程 → 每次 ~3min 冷启（数据扫描 + π0.5 加载），10 task 仅启动就要浪费 ~30min，可接受但不优雅：

```bash
for t in 0 1 2 3 4 5 6 7 8 9; do
  LEROBOT_OUTPUT_ROOT=/root/autodl-tmp/outputs \
  LIBERO_SUITE=libero_spatial bash dev.sh train_token \
    --dataset.task_index=$t \
    --steps=5000
done
```

若 task 0 路径都通顺，后续优化方向是**预编码 z_vis 离线缓存**（rlt-openpi 论文思路）：跑一次 `encode_libero_task.py` 把 π0.5 的视觉 hidden state 落 safetensors，`train_token` 走快路径直接读张量，避免每个 task 重新加载 π0.5。这是后置优化，先把 task 0 的 Stage 2 算法链路跑稳再说。

Stage 2 仍建议逐个 task 启动，不要一开始并发，以便定位环境和策略问题。

## 当前代码状态

- Stage 1 已迁移为 rlt-openpi 风格 `RLTokenModel`，`z_rl` 为 2048D，训练时剥离语言 token。
- Stage 2 已迁移 rlt-openpi 的 Actor / TwinQCritic / ReplayBuffer / TD3 更新 / RolloutWorker / OnlineTrainer。
- LeRobot 适配层为 `src/lerobot/rltoken/adapter.py` 和 `src/lerobot/rltoken/libero_chunk_env.py`。
- 当前优先级是跑通 task 0 的 Stage 2 smoke / 正式训练，而不是继续扩展新功能。
