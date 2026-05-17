# RL Token 仿真复现 — 执行日志

主线设计见 [`docs/rltoken_plan.md`](rltoken_plan.md)（V3）。本文档记录每个里程碑的实际产出、命令、关键路径、待办；区别于设计文档，本文档随实施增量更新。

## 当前进度

- [x] 阶段零：repo 脚手架（`rltoken` 分支 commits `b4e05f8e` + `8d7151f9`）
- [x] 阶段一：SFT 基线 + 吞吐率基准线（`2026-05-16` 已验证 `libero_spatial task 0`，5/5 成功）
- [x] 阶段二a：RL Token 编码器-解码器离线训练（`2026-05-17` 已验证 `libero_spatial task 0`，5000 step；L_ro 2.21 → 0.29）
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
- 演示数据：HF Hub `HuggingFaceVLA/libero`（LeRobotDataset 加载；与 `docs/source/libero.mdx` 一致）
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

### 产出

已验证运行（`2026-05-17`，`libero_spatial task 0`，5000 step）：

| 项 | 值 |
|---|---|
| checkpoint | `outputs/rltoken/encoder_decoder/libero_spatial/task_00/step_005000.safetensors` |
| wandb run | `rltoken-pi05-libero/quiet-valley-7` (id `uo9fkt4p`) |
| L_ro 起点 / 终点 (EMA) | 2.21 / 0.291 |
| L_ro 1k / 2k / 3k / 4k / 5k (EMA) | 0.513 / 0.425 / 0.371 / 0.324 / 0.291 |
| 末段 grad norm | 0.55 |
| `token/z_rl_norm` 末段 | 9.21 |
| 吞吐 | 1.19 step/s |
| 训练机器 / 用时 | 单 32GB GPU，约 1h15min |

**配置偏差**：因为 batch=32 在单 GPU 会 OOM（峰值 ~31GB），实际跑用 `experiments/rltoken_pi05_libero.yaml` 里的 `dataset.batch_size=16`（GPU 占用 ~24GB）。loss 仍稳定下降，未观察到收敛差异。

**关键 bug 修复（在阶段二a 落地过程中暴露）**：

1. **HuggingFaceVLA/libero 上游数据 bug**：`meta/episodes/*.parquet` 的 `data/file_index` 列与实际数据 parquet 布局不一致（[community issue #5](https://huggingface.co/datasets/HuggingFaceVLA/libero/discussions/5)）。按 `episodes=[task_eps]` 过滤会下到错文件、`Dataset.from_parquet` 返回 0 行 → `ValueError: Instruction "train" corresponds to no data!`。
   - 修复：committed JSON 映射 `experiments/dataset_overrides/HuggingFaceVLA_libero_v3.0_episodes_fix.json`（1693 项，由 `scripts/generate_libero_episode_fix.py` 扫 377 个 parquet footer 生成）
   - 运行时 patch：`src/lerobot/rltoken/dataset_repair.py` 在 `LeRobotDatasetMetadata` 加载后改写本地 parquet，原文件备份为 `*.broken.bak`，幂等
   - hook 调用点：`train_token._resolve_episode_filter`
2. **π0.5 prefix-only embedding dtype / attention backend 不匹配**：`prefix_embs` 初始 fp32，但 π0.5 language model 权重是 bf16；进入 `PiGemmaModel.forward` 后 hidden states 会转 bf16，SDPA 路径要求 attention bias 与 query dtype 一致，抛 `RuntimeError: invalid dtype for bias - should match query's dtype`。
   - 参考 `~/DevSpace/rlt-openpi/src/rlt_openpi/vla/embedding_extractor.py`
   - 修复：只在 `src/lerobot/rltoken/rl_token.py:extract_vlm_embeddings` 的 RL Token prefix-only 路径设置 `language_model.config._attn_implementation = "eager"`，并把 `prefix_embs` / `attn_4d` 对齐到 language model `q_proj.weight.dtype`
   - 不改 `src/lerobot/policies/pi05/` 或 `src/lerobot/policies/pi_gemma.py` 官方移植代码
3. **LeRobot dataset 单 task 加载慢**：`Dataset.from_parquet` 默认 glob 全部 377 文件。改 `src/lerobot/datasets/io_utils.py:load_nested_dataset` 接受 `paths` 参数，由 `DatasetReader._load_hf_dataset` 用 `meta.get_data_file_path()` 算出实际涉及的 ~10 个文件传入，跳过全 glob。
4. **`LEROBOT_OUTPUT_ROOT` 之前只对 eval_baseline 生效**：现在 `dev.sh` 在用户没传 `--output_dir` / `--save_dir` 时给 train_token / train_online / eval_throughput 都注入。详见 CLAUDE.md / AGENTS.md "输出路径约定"。

---

## 后续阶段 2b / 2c 待办（脚手架缺口）

按 `docs/rltoken_plan.md` §2.4 + §5.2，2b/2c 的核心代码已迁移进 `src/lerobot/rltoken/`。当前剩余待办是运行验证和性能精修：

- 已跑通单任务 Stage 1：`bash dev.sh train_token --dataset.task_index=0 --steps=5000`
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
