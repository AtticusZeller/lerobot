# RL Token 仿真复现 — 执行日志

主线设计见 [`docs/rltoken_plan.md`](rltoken_plan.md)（V2）。本文档记录每个里程碑的实际产出、命令、关键路径、待办；区别于设计文档，本文档随实施增量更新。

## 当前进度

- [x] 阶段零：repo 脚手架（`rltoken` 分支 commits `b4e05f8e` + `8d7151f9`）
- [ ] 阶段一：SFT 基线 + 吞吐率基准线（代码已就绪，待跑数据）
- [ ] 阶段二a：RL Token 编码器-解码器离线训练（代码已就绪，待训练）
- [ ] 阶段二b：LIBERO env 适配（块级奖励累加 + 演示预填 replay）
- [ ] 阶段二c：块级 TD3 在线训练 + 评估

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

## 后续阶段 2b / 2c 待办（脚手架缺口）

按 `docs/rltoken_plan.md` §2.4 + §5.2，2b/2c 需新增/扩展以下模块，本次阶段未触及：

- `src/lerobot/rl/buffer.py` —— 扩展 `action_chunk_size` 字段或新写 `ChunkedReplayBuffer`，存储宏动作 transition
- `src/lerobot/rl/learner.py` —— 现仅 SAC，需新增 TD3：双 Q + Polyak 软更新 + 块级 Bellman 累加 $y = \sum_{i=0}^{C-1} \gamma^i r_{t+i} + \gamma^C \min_j Q'_{\theta_j}(s_{t+C}, a')$
- `src/lerobot/rl/actor.py` —— 现仅 HILSerl gamepad，需新增 `RLTokenActor`：调用阶段二a checkpoint 抽 `z_rl`，参考动作 50% dropout，残差末层零初始化
- `src/lerobot/envs/libero.py` —— 加块级 reward 累加 wrapper（C 步聚合一次 transition）；可选去除人类介入接口
- `src/lerobot/rltoken/train_online.py` —— 当前为 NotImplementedError 占位，需实现 actor-critic 主循环
- 演示预填脚本 —— 用 HF Hub `lerobot/libero` 演示填充 replay buffer 作为 warmup

---

## 参考

- 主线设计：[`docs/rltoken_plan.md`](rltoken_plan.md)
- 原论文：[`docs/paper/RL Token: Bootstrapping Online RL with Vision-Language-Action Models.md`](paper/)
- π0.5 原论文：[`docs/paper/pi_0.5 a Vision-Language-Action Model with Open-World Generalization.md`](paper/)
- 代码参考：[rlt-openpi](https://github.com/yknxh/rlt-openpi)（主参考）、`~/DevSpace/RLinf/`（LIBERO env + OpenPI 集成参考）
