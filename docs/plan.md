# RL Token 仿真复现 — 执行日志

主线设计见 [`docs/rltoken_plan.md`](rltoken_plan.md)（V2）。本文档记录每个里程碑的实际产出、命令、关键路径、待办；区别于设计文档，本文档随实施增量更新。

## 当前进度

- [x] 阶段零：repo 脚手架（`rltoken` 分支 commits `b4e05f8e` + `8d7151f9`）
- [ ] 阶段一：SFT 基线 + 吞吐率基准线（代码已就绪，待跑数据）
- [ ] 阶段二a：RL Token 编码器-解码器离线训练（代码已就绪，待训练）
- [~] 阶段二b：LIBERO env 块级 wrapper（代码骨架就绪 on `rltoken_p2`）
- [~] 阶段二c：块级 TD3 在线训练 + 评估（代码骨架就绪 on `rltoken_p2`，端到端 run 待回灌）

---

## 阶段一：SFT 基线 + 吞吐率基准线

### 设计

冻结 `lerobot/pi05_libero`，在 LIBERO-Spatial / Object / Goal / Libero-10 上跑评估，记录 per-episode 步数 + 成功率，计算吞吐率：

$$\text{throughput} = \frac{\text{成功回合数}}{\text{总步数预算}} \times 1000$$

为 阶段二a/c 改进幅度提供对照基线。

### 关键改动

- `src/lerobot/scripts/lerobot_eval.py` — `rollout()` 新增 `episode_lengths` 记录（每个 env 首次 `done` 的 step+1），向上传播到 `eval_policy` / `eval_one` / `eval_policy_all`。聚合指标新增 `avg_episode_length` / `p25/p50/p75_episode_length` / `throughput`。
- `src/lerobot/rltoken/eval_throughput.py` — 4-suite batch wrapper：调用 `eval_main`，把结果汇总成 `outputs/baseline/libero_throughput.json`，per-suite lengths 写到 `outputs/baseline/<suite>/episode_steps.csv`。
- `dev.sh` — `eval_baseline` 与 `eval_throughput` 两个入口已就位。

### 命令

```bash
# 单 suite 烟雾测试
LIBERO_SUITE=libero_spatial bash dev.sh eval_baseline --eval.n_episodes=5

# 4-suite 全量基线
bash dev.sh eval_throughput \
    --policy_path=lerobot/pi05_libero \
    --n_episodes=50 \
    --output_dir=outputs/baseline
```

### 产出（运行后填）

| Suite           | 成功率 | 平均步数 | P25 / P50 / P75 | 吞吐率 (×1000) |
| --------------- | ------ | -------- | --------------- | -------------- |
| libero_spatial  | TBD    | TBD      | TBD / TBD / TBD | TBD            |
| libero_object   | TBD    | TBD      | TBD / TBD / TBD | TBD            |
| libero_goal     | TBD    | TBD      | TBD / TBD / TBD | TBD            |
| libero_10       | TBD    | TBD      | TBD / TBD / TBD | TBD            |

raw log: `outputs/baseline/<suite>/episode_steps.csv`

---

## 阶段二a：RL Token 编码器-解码器

### 设计

冻结 `pi05_libero`，提取 VLM 最后一层 hidden state $z_{1:M} \in \mathbb{R}^{B \times M \times 2048}$（M ≈ 456），训练轻量编码器-解码器构建信息瓶颈：

- 编码器 $E(z_{1:M}) \to z_{rl} \in \mathbb{R}^{256}$（CLS-pooling Transformer，2 层）
- 解码器 $D(z_{rl}) \to \hat z_{1:M}$（learnable queries + cross-attn，2 层）
- 损失 $L_{ro} = \mathbb{E}\|D(E(z_{1:M})) - \text{sg}(z_{1:M})\|_2^2$
- 演示数据：HF Hub `lerobot/libero`（LeRobotDataset 加载）
- 训练步数：5000

### 关键改动

- `src/lerobot/rltoken/rl_token.py` — `RLTokenEncoder` / `RLTokenDecoder` / `extract_vlm_embeddings()` helper
- `src/lerobot/rltoken/datasets.py` — `make_libero_demo_dataloader()` 薄包装
- `src/lerobot/rltoken/train_token.py` — 5K 步训练循环 entry
- `experiments/rltoken_pi05_libero.yaml` — 补 `dataset` + `stage1` 节

### 命令

```bash
# 烟雾测试（20 步）
bash dev.sh train_token --steps=20

# 正式 run
bash dev.sh train_token --steps=5000
```

### 产出（运行后填）

- 检查点：`outputs/rltoken/encoder_decoder/step_5000.safetensors`（< 50 MB）
- 训练曲线：wandb run `rltoken-pi05-libero/<id>`
- `L_ro` 末段平台值：TBD（参考 rlt-openpi：初始 ~1.0 → 末段 < 0.1）
- `z_rl` 验证集协方差矩阵秩：TBD（目标 ≈ 256，未塌缩）

---

## 阶段二b / 二c — 并行实施中（`rltoken_p2` 分支）

通过 `git worktree add ../lerobot-rltoken_p2 rltoken_p2`（基于 `rltoken@97f985c8`）开副工作树，与 `rltoken` 上阶段一/二a 的验证并行推进。设计依据 `docs/rltoken_plan.md` §2.4 + §5.2。

### 已落代码（rltoken_p2 本地 commit）

- [x] `src/lerobot/rltoken/block_env.py` — `ChunkedLiberoEnv`：把 `LiberoEnv` 单步接口折叠为块级接口，一次 step 内执行 C 个 inner step、累加二值成功奖励、提前 break on terminate
- [x] `src/lerobot/rltoken/td3.py` — `TD3Actor`（残差头零初始化 + ref dropout 0.5）、`TD3Critic`（双 Q）、`soft_update_target`（Polyak τ=0.005）、`td3_critic_loss`（块级 Bellman y = r + (1-done)·γ^C·min_j Q'_j(s_{t+C}, a')）、`td3_actor_loss`（-Q1 + β·BC）
- [x] `src/lerobot/rltoken/train_online.py` — 替换 NotImplementedError；主循环复用 `ReplayBuffer`（dict state + chunk action）；`--stage2.smoke=True` 走纯 TD3 单元 loop（无 LIBERO / π0.5 依赖）
- [x] `src/lerobot/rltoken/tests/{test_block_env,test_td3}.py` — 13 个单元 smoke test 全绿（chunk reward 累加、shape、零初始化语义、梯度流、Polyak 中点）

### 复用结论

- `src/lerobot/rl/buffer.py:ReplayBuffer` ✅ 原样复用：dict-typed state、任意 action shape；chunk action 存为 flat tensor `(B, C*A)`，sample 后 view 回 `(B, C, A)`
- `src/lerobot/rl/{learner,actor}.py` ❌ 不复用：强耦合 gRPC + HILSerl gamepad；标准单进程 main loop 直接写在 `train_online.py` 即可
- `src/lerobot/policies/sac/modeling_sac.py:CriticEnsemble` 模式借鉴（双 Q），但因耦合 SACObservationEncoder 改为自写

### 回灌后待办（标记为 `# TODO(reconcile):`）

- 演示预填 replay buffer：用 `lerobot/libero` HF 数据集 + 阶段二a 风格 dataloader 填 warmup transitions（避免热启依赖 π0.5 rollout）
- 把阶段二 ckpt wire 进 `lerobot.rltoken.eval_throughput` 以做 RL Token vs SFT 吞吐率对比
- 多 task 支持：当前固定 `--stage2.task_id=0`；之后扩展为 task_id 循环 / 并行
- 完整 10K+ 步端到端 run on LIBERO-Spatial 单 task

---

## 参考

- 主线设计：[`docs/rltoken_plan.md`](rltoken_plan.md)
- 原论文：[`docs/paper/RL Token: Bootstrapping Online RL with Vision-Language-Action Models.md`](paper/)
- π0.5 原论文：[`docs/paper/pi_0.5 a Vision-Language-Action Model with Open-World Generalization.md`](paper/)
- 代码参考：[rlt-openpi](https://github.com/yknxh/rlt-openpi)（主参考）、`~/DevSpace/RLinf/`（LIBERO env + OpenPI 集成参考）
