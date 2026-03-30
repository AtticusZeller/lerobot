# SmolVLA 模型微调指南（SO-ARM101）

> **本文档是 [so101_pipeline.md](./so101_pipeline.md) 的模型子文档，仅包含 SmolVLA 特定内容。**
> 通用流程（数据集准备、硬件映射、gRPC 部署架构）请参考主文档。

---

## 目录

1. [架构与微调策略](#1-架构与微调策略)
2. [训练命令与参数](#2-训练命令与参数)
3. [显存估算](#3-显存估算)
4. [训练监控与 Checkpoint 选择](#4-训练监控与-checkpoint-选择)

---

## 1. 架构与微调策略

### SmolVLA 架构

```
SmolVLA (~450M 参数)
├── SigLIP Vision Encoder            ← freeze_vision_encoder=true 时冻结
├── SmolVLM2-500M VLM (16 层)        ← train_expert_only=true 时冻结
│   └── Text Model (语言理解)
└── Action Expert (~12 层, 75% width) ← 始终参与训练
    ├── action_in_proj / action_out_proj
    ├── state_proj
    ├── action_time_mlp_in / action_time_mlp_out
    └── cross_attn (交叉注意力到 VLM KV cache)
```

**与 Pi0.5 的关键区别**：

| 特性 | SmolVLA | Pi0.5 |
|------|---------|-------|
| 总参数量 | ~450M | ~3B |
| VLM 骨干 | SmolVLM2-500M | PaliGemma (~2.7B) |
| Action Expert | ~12层, 75% width | ~300M, Gemma |
| 注意力机制 | Cross-attention to VLM KV | Self-attention |
| 归一化模式 | **MEAN_STD** | QUANTILES |
| 预训练基座 | `lerobot/smolvla_base` | `lerobot/pi05_base` |
| 依赖 extra | `smolvla` | `pi` |

### 策略选择：全量微调 Action Expert（默认）

SmolVLA 默认配置即为 `train_expert_only=true` + `freeze_vision_encoder=true`：

- **冻结**：SigLIP 视觉编码器 + SmolVLM2 语言模型
- **训练**：Action Expert 全部参数 + state_proj + action projections

由于模型总参数量仅 ~450M，相比 Pi0.5 显著更轻量，单卡 16GB+ 即可微调。

### 源码定位

* **SmolVLA 策略配置**: `src/lerobot/policies/smolvla/configuration_smolvla.py`
* **模型实现**: `src/lerobot/policies/smolvla/modeling_smolvla.py`
* **VLM+Expert 架构**: `src/lerobot/policies/smolvla/smolvlm_with_expert.py`
* **冻结逻辑**: `modeling_smolvla.py` — `_set_requires_grad()` 方法

### 归一化：MEAN_STD（无需额外处理）

与 Pi0.5 不同，SmolVLA 使用标准的 MEAN_STD 归一化：

```python
normalization_mapping = {
    "VISUAL": NormalizationMode.IDENTITY,
    "STATE": NormalizationMode.MEAN_STD,    # ← 标准归一化
    "ACTION": NormalizationMode.MEAN_STD,   # ← 标准归一化
}
```

**这意味着不需要运行 `augment_dataset_quantile_stats.py`**，LeRobot 数据集默认就包含 mean/std 统计量。

---

## 2. 训练命令与参数

### 安装依赖

```bash
uv sync --extra "smolvla" --extra "dev" --extra "feetech"
```

> SmolVLA 依赖：`transformers>=5.3.0`、`num2words`、`accelerate`、`safetensors`，由 `smolvla` extra 自动安装。

### 启动训练

使用预训练基座 [`lerobot/smolvla_base`](https://hf.co/lerobot/smolvla_base) 微调：

```bash
RUN_DIR="/root/autodl-tmp/outputs/smolvla_so101/$(date +%Y%m%d_%H%M%S)"

lerobot-train \
    --policy.path=lerobot/smolvla_base \
    --dataset.repo_id=Atticuxz/so101-table-cleanup \
    --batch_size=64 \
    --steps=20000 \
    --output_dir="$RUN_DIR" \
    --job_name=smolvla_so101 \
    --policy.device=cuda \
    --wandb.enable=true
```

也可以使用 YAML 配置文件统一管理（推荐）：

```bash
lerobot-train --yaml_config=experiments/smolvla_so101_pick_orange.yaml \
    --output_dir="$RUN_DIR"
```

YAML 配置文件位于 `experiments/smolvla_so101_pick_orange.yaml`，详见文件内注释。

### 关键参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `policy.path` | `lerobot/smolvla_base` | 预训练基座模型 |
| `batch_size` | 64 | 批量大小（可从小到大递增） |
| `steps` | 20000 | 训练步数（~4h on A100） |
| `policy.train_expert_only` | `true` | 仅训练 Action Expert |
| `policy.freeze_vision_encoder` | `true` | 冻结视觉编码器 |
| `policy.train_state_proj` | `true` | 训练 state 投影层 |

### 优化器/调度器预设

| 参数 | 默认值 | 说明 |
|------|--------|------|
| **Optimizer** | AdamW | - |
| `lr` | 1e-4 | 学习率（比 Pi0.5 的 2.5e-5 更高） |
| `betas` | (0.9, 0.95) | Adam beta |
| `weight_decay` | 1e-10 | 极小的 L2 正则 |
| `grad_clip_norm` | 10 | 梯度裁剪（比 Pi0.5 的 1.0 更宽松） |
| **Scheduler** | CosineDecayWithWarmup | - |
| `warmup_steps` | 1000 | 预热步数 |
| `decay_steps` | 30000 | 衰减步数 |
| `decay_lr` | 2.5e-6 | 最小学习率 |

### 根据数据量确定 Steps

SmolVLA 官方建议以 20k steps 为起点。根据数据量调整：

| 数据规模 | episodes | 建议 steps | 说明 |
|---------|---------|---------|------|
| 少 | ~25 | 10000~15000 | 官方论文指出 25 episodes 效果不佳 |
| 标准 | ~50 | 15000~20000 | 官方推荐，5 个变体 × 10 episodes |
| 中 | 50~200 | 20000~30000 | 增加数据后适当延长 |
| 大 | > 200 | 30000+ | - |

### 推理部署

```bash
# 方式一：lerobot-record 直接推理（推荐评估用）
lerobot-record \
    --robot.type=so101_follower \
    --robot.port=/dev/ttyACM0 \
    --robot.id=my_so101 \
    --robot.cameras="{ front: {type: opencv, index_or_path: 8, width: 640, height: 480, fps: 30}}" \
    --dataset.single_task="把桌上的文具收到笔盒里" \
    --dataset.repo_id=Atticuxz/eval_smolvla_so101 \
    --dataset.episode_time_s=50 \
    --dataset.num_episodes=10 \
    --policy.path=Atticuxz/smolvla_so101

# 方式二：gRPC 异步推理（远程 GPU 部署）
# Policy Server (GPU)
python -m lerobot.async_inference.policy_server --host=0.0.0.0 --port=8080

# Robot Client (SO-ARM101)
python -m lerobot.async_inference.robot_client \
    --server_address=<GPU_IP>:8080 \
    --robot.type=so101_follower \
    --robot.port=/dev/ttyACM1 \
    --robot.id=my_so101 \
    --robot.cameras="{ top: {type: opencv, index_or_path: /dev/video10, width: 640, height: 480, fps: 30 }}" \
    --task="把桌上的文具收到笔盒里" \
    --policy_type=smolvla \
    --pretrained_name_or_path=Atticuxz/smolvla_so101 \
    --actions_per_chunk=50 \
    --chunk_size_threshold=0.5
```

详细部署文档请参考 [inference.md](./inference.md)。

---

## 3. 显存估算

SmolVLA 总参数量仅 ~450M，显著低于 Pi0.5 (~3B)，单卡友好。

### 显存使用情况

| 配置 | 显存需求 | 适合 GPU |
|------|----------|---------|
| train_expert_only + bf16 + bs=64 | ~16-20 GB | A100-40G, RTX 4090 |
| train_expert_only + bf16 + bs=32 | ~12-16 GB | RTX 4090, RTX 3090 |
| train_expert_only + bf16 + bs=16 | ~8-12 GB | RTX 3060, RTX 4080 |

> SmolVLA 相比 Pi0.5 可以使用更大的 batch_size，官方示例即以 bs=64 为起点。

### 优化策略

1. **Mixed Precision bfloat16**：`--policy.dtype=bfloat16`（SmolVLA 默认即 bf16）
2. **降低 Batch Size**：从 64 逐步减小直到显存允许
3. **Gradient Checkpointing**：显存紧张时启用

---

## 4. 训练监控与 Checkpoint 选择

### SmolVLA Loss 计算方式

SmolVLA 同样使用 **Flow Matching**，loss 为 MSE：

```python
# modeling_smolvla.py
u_t = noise - actions                          # 目标流场
v_t = action_out_proj(suffix_out)             # 模型预测
loss = F.mse_loss(u_t, v_t, reduction="none") # 逐元素 MSE
# 对有效帧取均值（排除 action_is_pad 的帧）
```

### 推理去噪步数

SmolVLA 推理时使用 **10 步** Flow Matching 去噪（`num_steps=10`），生成 50 步的动作 chunk。

### WandB 关键曲线

| WandB 字段 | 含义 | 期望形态 |
|------------|------|----------|
| `loss` | Flow Matching MSE | 全程稳定下降 |
| `grad_norm` | 梯度范数 | 无爆炸 |
| `lr` | 学习率 | warmup + cosine decay |

### Checkpoint 选择

与 Pi0.5 相同：训练完成后查看 loss 曲线 → 选候选 checkpoint → 真机评估 → 按评分 rubric 选最优。

详见 [eval.md](./eval.md)。

---

## 常见注意事项

### Task 描述需要换行符

SmolVLA 的 tokenizer 处理器（`SmolVLANewLineProcessor`）会自动在 task 描述末尾添加 `\n`。正常使用 `lerobot-train` / `lerobot-record` 时无需手动处理。

### 图像预处理

SmolVLA 使用 512×512 的图像输入（带 padding 保持宽高比），不同于 Pi0.5 的 224×224。训练和推理时由 `SmolVLAProcessor` 自动处理。

---

**相关文档**：
* [SO-ARM101 Pipeline 主文档](./so101_pipeline.md)
* [Pi0.5 模型指南](./so101_pi05.md)
* [推理部署](./inference.md)
* [评估 SOP](./eval.md)
* [SmolVLA 官方论文](https://arxiv.org/abs/2506.01844)
