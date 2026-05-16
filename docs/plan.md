# RL Token 仿真复现 — 执行日志

主线设计见 [`docs/rltoken_plan.md`](rltoken_plan.md)（V3）。本文档记录每个里程碑的实际产出、命令、关键路径、待办；区别于设计文档，本文档随实施增量更新。

## 当前进度

- [x] 阶段零：repo 脚手架（`rltoken` 分支 commits `b4e05f8e` + `8d7151f9`）
- [x] 阶段一：SFT 基线 + 吞吐率基准线（`2026-05-16` 已验证 `libero_spatial task 0`，5/5 成功）
- [ ] 阶段二a：RL Token 编码器-解码器离线训练（代码已就绪，待 task 0 训练）
- [x] 阶段二b：LIBERO env 适配（块级奖励累加 + VLA-only warmup replay）
- [x] 阶段二c：块级 TD3 在线训练代码迁移（待 task 0 smoke / 正式训练验证）

---

## 阶段一：SFT 基线 + 吞吐率基准线

### 设计

冻结 `lerobot/pi05_libero_finetuned`，在 LIBERO-Spatial / Object / Goal / Libero-10 上跑评估，记录 per-episode 步数 + 成功率，计算吞吐率：

$$\text{throughput} = \frac{\text{成功回合数}}{\text{总 env steps}} \times 1000$$

为 阶段二a/c 改进幅度提供对照基线。

### 关键改动

- `src/lerobot/scripts/lerobot_eval.py` — `rollout()` 新增 `episode_lengths` 记录（每个 env 首次 `done` 的 step+1），向上传播到 `eval_policy` / `eval_one` / `eval_policy_all`。聚合指标新增 `avg_episode_length` / `p25/p50/p75_episode_length` / `throughput`。
- `src/lerobot/configs/eval.py` — `eval_baseline` 默认输出根目录支持环境变量 `LEROBOT_OUTPUT_ROOT`；未显式传 `--output_dir` 时输出到 `<输出根>/eval/<日期>/<时间>_<job_name>/`，适配 AutoDL 数据盘。
- `src/lerobot/rltoken/eval_throughput.py` — 4-suite batch wrapper：调用 `eval_main`，把结果汇总成 `outputs/baseline/libero_throughput.json`，per-suite lengths 写到 `outputs/baseline/<suite>/episode_steps.csv`。
- `dev.sh` — `eval_baseline` 与 `eval_throughput` 两个入口已就位。

### 命令

```bash
# 单 suite 烟雾测试
LIBERO_SUITE=libero_spatial bash dev.sh eval_baseline --eval.n_episodes=5

# AutoDL: 默认输出写入数据盘，同时保留自动日期/时间目录
LEROBOT_OUTPUT_ROOT=/root/autodl-tmp/outputs \
LIBERO_SUITE=libero_spatial bash dev.sh eval_baseline --eval.n_episodes=5

# 4-suite 全量基线
bash dev.sh eval_throughput \
    --policy_path=lerobot/pi05_libero_finetuned \
    --n_episodes=50 \
    --output_dir=outputs/baseline
```

### 产出

已验证运行（2026-05-16）：

```bash
LEROBOT_OUTPUT_ROOT=/root/autodl-tmp/outputs \
LIBERO_SUITE=libero_spatial bash dev.sh eval_baseline \
  --env.task_ids='[0]' \
  --eval.n_episodes=5
```

| Suite           | 成功率 | 平均步数 | P25 / P50 / P75 | 吞吐率（successes / 1000 env steps） |
| --------------- | ------ | -------- | --------------- | ------------------------------------ |
| libero_spatial  | 100.0% | 76.6     | 74 / 77 / 77    | 13.05                                |
| libero_object   | TBD    | TBD      | TBD / TBD / TBD | TBD            |
| libero_goal     | TBD    | TBD      | TBD / TBD / TBD | TBD            |
| libero_10       | TBD    | TBD      | TBD / TBD / TBD | TBD            |

Task 0 逐回合结果：

- `successes=[True, True, True, True, True]`
- `episode_lengths=[77, 74, 74, 77, 81]`
- `sum_rewards=[1.0, 1.0, 1.0, 1.0, 1.0]`
- 输出目录：`/root/autodl-tmp/outputs/eval/2026-05-16/19-07-03_libero_pi05/`

raw log:

- baseline eval videos: `/root/autodl-tmp/outputs/eval/2026-05-16/19-07-03_libero_pi05/videos/libero_spatial_0/`
- 4-suite wrapper csv: `outputs/baseline/<suite>/episode_steps.csv`

---

## 阶段二a：RL Token 编码器-解码器

### 设计

冻结 `pi05_libero_finetuned`，提取 VLM 最后一层视觉 hidden state $z^{vis}_{1:M} \in \mathbb{R}^{B \times M \times 2048}$（去除固定语言嵌入），按 LIBERO task 训练独立编码器-解码器：

- 编码器 $E([z^{vis}_{1:M}, e_{rl}]) \to z_{rl} \in \mathbb{R}^{2048}$（rlt-openpi TransformerEncoder，2 层）
- 解码器 teacher-forced AR：$D(z_{rl}, z^{vis}_{1:M}) \to \hat z^{vis}_{1:M}$（rlt-openpi TransformerDecoder，2 层）
- 损失 $L_{ro} = \mathbb{E}\|D(E(z^{vis}_{1:M})) - \text{sg}(z^{vis}_{1:M})\|_2^2$
- 演示数据：HF Hub `lerobot/libero`（LeRobotDataset 加载）
- 训练粒度：per-task，4 suites × 10 tasks = 40 个 ckpt
- 训练步数：5000 / task

### 关键改动

- `src/lerobot/rltoken/rl_token.py` — rlt-openpi `RLTokenModel` API + LeRobot π0.5 visual-only embedding helper
- `src/lerobot/rltoken/train_token.py` — per-task 5K 步 Stage 1 训练循环
- `src/lerobot/rltoken/{actor,critic,replay_buffer,td3_utils,rollout_worker,online_trainer}.py` — rlt-openpi Stage 2 TD3 核心迁移
- `src/lerobot/rltoken/{adapter,libero_chunk_env}.py` — LeRobot π0.5 / LIBERO 适配层
- `experiments/rltoken_pi05_libero.yaml` — V3 超参（`embedding_dim=2048`、batch 32、warmup 500）

### 命令

```bash
# 烟雾测试（20 步）
bash dev.sh train_token --dataset.task_index=0 --steps=20

# 单 task 正式 run
bash dev.sh train_token --dataset.task_index=0 --steps=5000

# LIBERO-Spatial 10 个 task
for t in 0 1 2 3 4 5 6 7 8 9; do
  bash dev.sh train_token --dataset.task_index=$t --steps=5000
done
```

### 产出（运行后填）

- 检查点：`outputs/rltoken/encoder_decoder/<suite>/task_<NN>/step_005000.safetensors`
- 训练曲线：wandb run `rltoken-pi05-libero/<id>`
- `L_ro` 末段平台值：TBD（参考 rlt-openpi：初始 ~1.0 → 末段 < 0.1）
- `z_rl` 验证集协方差矩阵秩：TBD（目标不塌缩；维度 2048）

---

## 后续阶段 2b / 2c 待办（脚手架缺口）

按 `docs/rltoken_plan.md` §2.4 + §5.2，2b/2c 的核心代码已迁移进 `src/lerobot/rltoken/`。当前剩余待办是运行验证和性能精修：

- 跑通单任务 Stage 1：`bash dev.sh train_token --dataset.task_index=0 --steps=5000`
- 跑通单任务 Stage 2 smoke：`bash dev.sh train_online --task_index=0 --rl_token_checkpoint=<stage1_ckpt> --warmup_steps=2 --max_env_steps=20`
- 如 Critic 不收敛，优先检查 reward 稀疏度、action 后处理尺度、`z_rl + proprio` 状态归一化
- 演示预填 replay buffer 当前用 VLA-only 在线 warmup；后续可补离线 demo 预填以减少仿真 warmup 成本

---

## 参考

- 主线设计：[`docs/rltoken_plan.md`](rltoken_plan.md)
- 用户运行清单：[`plan_user.md`](../plan_user.md)
- 原论文：[`docs/paper/RL Token: Bootstrapping Online RL with Vision-Language-Action Models.md`](paper/)
- π0.5 原论文：[`docs/paper/pi_0.5 a Vision-Language-Action Model with Open-World Generalization.md`](paper/)
- 代码参考：[rlt-openpi](https://github.com/yknxh/rlt-openpi)（主参考）、`~/DevSpace/RLinf/`（LIBERO env + OpenPI 集成参考）
