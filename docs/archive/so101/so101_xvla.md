# X-VLA 模型微调指南（SO-ARM101）

> **本文档是 [so101_pipeline.md](./so101_pipeline.md) 的模型子文档，仅包含 X-VLA 特定内容。**
> 通用流程（数据集准备、硬件映射、gRPC 部署架构）请参考主文档。

---

## 目录

1. [架构与微调策略](#1-架构与微调策略)
2. [训练命令与参数](#2-训练命令与参数)
3. [显存估算](#3-显存估算)
4. [训练监控与 Checkpoint 选择](#4-训练监控与-checkpoint-选择)

---

## 1. 架构与微调策略

### X-VLA 架构

```
X-VLA (~0.9B 参数)
├── Florence2 VLM Backbone
│   ├── Vision Encoder (ViT)        ← freeze_vision_encoder=true 时冻结
│   └── Language Encoder (BART)     ← freeze_language_encoder=true 时冻结
├── Policy Transformer (24 层)       ← train_policy_transformer=true 时训练
│   ├── hidden_size=1024, num_heads=16, mlp_ratio=4.0
│   └── Soft Prompts (len=32)       ← train_soft_prompts=true 时训练
└── Action Head
    ├── action_mode=auto (自适应)
    └── max_action_dim=20 (pad 到统一维度)
```

**与 SmolVLA / Pi0.5 的关键区别**：

| 特性 | X-VLA | SmolVLA | Pi0.5 |
|------|-------|---------|-------|
| 总参数量 | ~0.9B | ~450M | ~3B |
| VLM 骨干 | Florence2 | SmolVLM2-500M | PaliGemma |
| 归一化模式 | VISUAL/STATE: IDENTITY, **ACTION: MEAN_STD** | MEAN_STD | QUANTILES |
| 默认冻结策略 | **不冻结** | 冻结 VLM | 冻结 VLM |
| 预训练基座 | `lerobot/xvla-base` | `lerobot/smolvla_base` | `lerobot/pi05_base` |
| 依赖 extra | `xvla` | `smolvla` | `pi` |

### 策略选择：冻结 VLM 骨干，训练 Policy Transformer

冻结 Vision Encoder 和 Language Encoder，仅训练 Policy Transformer + Soft Prompts：

* **冻结**：Vision Encoder、Language Encoder（`freeze_vision_encoder=true`, `freeze_language_encoder=true`）
* **训练**：Policy Transformer + Soft Prompts（`train_policy_transformer=true`, `train_soft_prompts=true`）

**实测结果**：SO-ARM101 table-cleanup 任务，~50 episodes 数据集，约 1000 steps 基本收敛。

与 SmolVLA 的 `train_expert_only` 模式不同，X-VLA 没有该选项。X-VLA 的冻结逻辑由 `freeze_vision_encoder` 和 `freeze_language_encoder` 两个独立开关控制。

> **SO-ARM101 注意**：`action_mode=auto` 会自动将数据集的 6-dim action（6 DOF + gripper = 7 维）pad 到 `max_action_dim=20`。无需手动配置 action 维度映射。

### 源码定位

* **X-VLA 策略配置**: `src/lerobot/policies/xvla/configuration_xvla.py` — `XVLAConfig` 类（L43）
* **模型实现**: `src/lerobot/policies/xvla/modeling_xvla.py`
* **处理器**: `src/lerobot/policies/xvla/processor_xvla.py`
* **Action Hub**: `src/lerobot/policies/xvla/action_hub.py`

### 归一化配置（必须在 YAML 显式设置）

`lerobot/xvla-base` 预训练模型使用的归一化来自 [config.json](https://huggingface.co/lerobot/xvla-base/blob/main/config.json)：

```yaml
normalization_mapping:
  VISUAL: IDENTITY
  STATE: IDENTITY
  ACTION: MEAN_STD   # ← 预训练用 MEAN_STD，不是 IDENTITY
```

> **陷阱**：`configuration_xvla.py` 的 Python 默认值是 `ACTION: IDENTITY`，但预训练模型用的是 `MEAN_STD`。`from_pretrained` 传入 YAML 构建的 config 时会绕过 hub 的 config.json，所以**必须在 YAML 里显式写出**。不写则 loss 初始值 ~2000 且不收敛。

VISUAL 和 STATE 使用 IDENTITY，无需运行 `augment_dataset_quantile_stats.py`。ACTION 的 MEAN_STD 统计量由 LeRobot 标准数据集自动提供，无需额外计算。

### 图像预处理

X-VLA 的图像预处理由 `processor_xvla.py` 中的处理器步骤完成：

1. `XVLAImageToFloatProcessorStep` — 将 uint8 图像转为 float [0, 1]
2. `XVLAImageNetNormalizeProcessorStep` — 应用 ImageNet 归一化（mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]）
3. `XVLAAddDomainIdProcessorStep` — 添加 domain_id（SO-ARM101 新微调用 domain_id=0）

X-VLA 默认**不固定 resize 图像**（`resize_imgs_with_padding` 默认为 `None`）。**注意：** 因为 Florence-2 模型内部计算位置编码时强行校验特征图长宽一致性（`assert h * w == num_tokens`），它**严格要求输入图像必须是正方形**。如果数据集原图非正方形，**必须**在 YAML 中设置 `policy.resize_imgs_with_padding: [224, 224]` 以强制加黑边填充，否则会导致 `AssertionError`。

---

## 2. 训练命令与参数

### 安装依赖

```bash
uv sync --extra "xvla" --extra "dev" --extra "feetech"
```

> X-VLA 依赖由 `xvla` extra 自动安装，包括 Florence2 相关依赖。

### 启动训练

使用预训练基座 [`lerobot/xvla-base`](https://hf.co/lerobot/xvla-base) 微调：

```bash
RUN_DIR="./outputs/xvla_so101/$(date +%Y%m%d_%H%M%S)"
lerobot-train \
    --policy.path=lerobot/xvla-base \
    --dataset.repo_id=Atticuxz/so101-table-cleanup \
    --batch_size=32 \
    --steps=20000 \
    --output_dir="$RUN_DIR" \
    --job_name=xvla_so101 \
    --policy.device=cuda \
    --policy.dtype=bfloat16 \
    --wandb.enable=true
```

也可以使用 YAML 配置文件统一管理（推荐）：

```bash
lerobot-train --yaml_config=experiments/xvla_so101_table_cleanup.yaml \
    --output_dir="$RUN_DIR"
```

YAML 配置文件位于 `experiments/xvla_so101_table_cleanup.yaml`，详见文件内注释。

### 关键参数

以下字段必须在 YAML 中显式设置以匹配预训练模型（`from_pretrained` 传入 config 时绕过 hub 的 config.json）：

| 参数 | **必须设置的值** | Python 默认值 | 说明 |
|------|--------|------|------|
| `normalization_mapping.ACTION` | **`MEAN_STD`** | `IDENTITY` | 不设置 loss ~2000 |
| `chunk_size` | **`30`** | `32` | 预训练序列长度 |
| `n_action_steps` | **`30`** | `32` | 同上 |
| `tokenizer_max_length` | **`1024`** | `64` | 预训练 token 长度 |
| `action_mode` | **`auto`** | `ee6d` | SO-ARM101 单臂 7-dim |
| `max_action_dim` | **`20`** | `20` | 与预训练一致 |
| `max_state_dim` | **`20`** | `32` | 与预训练一致 |
| `resize_imgs_with_padding` | **`[224, 224]`** | `null` | Florence2 要求正方形 |

### 优化器/调度器预设

| 参数 | X-VLA 默认值 | 说明 |
|------|-------------|------|
| **Optimizer** | AdamW | - |
| `optimizer_lr` | 1e-4 | 学习率 |
| `optimizer_betas` | (0.9, 0.95) | 来源：lerobot/xvla-base config.json |
| `optimizer_weight_decay` | 0.0001 | 来源：lerobot/xvla-base config.json |
| `optimizer_grad_clip_norm` | 10.0 | 梯度裁剪 |
| `optimizer_soft_prompt_lr_scale` | 1.0 | Soft-prompt LR 缩放因子 |
| **Scheduler** | CosineDecayWithWarmup | - |
| `scheduler_warmup_steps` | 1000 | 预热步数 |
| `scheduler_decay_steps` | 30000 | 衰减步数 |
| `scheduler_decay_lr` | 2.5e-6 | 最小学习率 |

### 根据数据量确定 Steps

| 数据规模 | episodes | 建议 steps | 说明 |
|---------|---------|---------|------|
| 少 | ~25 | 10000~15000 | 数据量不足时适当缩短 |
| 标准 | ~50 | 15000~20000 | 推荐起点 |
| 中 | 50~200 | 20000~30000 | 增加数据后适当延长 |
| 大 | > 200 | 20000 | 249 episodes 实测，收敛快约 1000 steps |

### Domain ID 说明

X-VLA 使用 `domain_id` 区分不同的机器人 embodiment。SO-ARM101 新微调时使用默认 `domain_id=0` 即可。该值由 `XVLAAddDomainIdProcessorStep` 自动添加到推理输入中，无需手动设置。

### 推理部署

```bash
# 方式一：lerobot-record 直接推理（推荐评估用）
lerobot-record \
    --robot.type=so101_follower \
    --robot.port=/dev/ttyACM0 \
    --robot.id=my_so101 \
    --robot.cameras="{ front: {type: opencv, index_or_path: 8, width: 640, height: 480, fps: 30}}" \
    --dataset.single_task="把桌上的文具收到笔盒里" \
    --dataset.repo_id=Atticuxz/eval_xvla_so101 \
    --dataset.episode_time_s=50 \
    --dataset.num_episodes=10 \
    --policy.path=Atticuxz/xvla_so101

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
    --policy_type=xvla \
    --pretrained_name_or_path=Atticuxz/xvla_so101 \
    --actions_per_chunk=32 \
    --chunk_size_threshold=0.5
```

详细部署文档请参考 [inference.md](./inference.md)。

---

## 3. 显存估算

X-VLA 总参数量 ~0.9B，介于 SmolVLA (~450M) 和 Pi0.5 (~3B) 之间。

### 显存使用情况

| 配置 | 显存需求 | 适合 GPU |
|------|----------|---------|
| 全参微调 + bf16 + bs=32 | ~20-24 GB | A100-40G, RTX 4090 |
| 全参微调 + bf16 + bs=24 | ~18-22 GB | RTX 4090, A5000 |
| 全参微调 + bf16 + bs=16 | ~16-20 GB | RTX 4090, RTX 3090 |
| Phase II (PEFT) + bf16 + bs=16 | ~8-12 GB | RTX 4060, RTX 3060 |

> X-VLA 即使在 Phase II 领域自适应模式下显存需求也较高，建议针对 8GB 显存设备将 batch size 降至 8 或 16 尝试。对于 24GB 显存设备，如果 bs=32 导致 OOM，而 bs=16 偏小导致 loss 下降不稳，推荐使用 **24**（8 的倍数，最大化 Tensor Core 效率）作为折中方案。

### 优化策略

1. **Mixed Precision bfloat16**：`--policy.dtype=bfloat16`（推荐，避免 OOM）
2. **冻结部分组件**：采用 Phase II (PEFT) 冻结视觉、语言编码器及策略 Transformer
3. **降低 Batch Size**：从 32 逐步减小直到显存允许

---

## 4. 训练监控与 Checkpoint 选择

### X-VLA Loss 计算方式

X-VLA 使用 **Flow Matching**，loss 为 MSE（与 SmolVLA / Pi0.5 相同范式）。

### 推理去噪步数

X-VLA 推理时使用 **10 步** Flow Matching 去噪（`num_denoising_steps=10`），生成 **30 步**的动作 chunk（`chunk_size=30`，匹配预训练模型）。

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

### YAML 配置必须显式设置 action_mode

SO-ARM101 是单臂 7-dim action，必须在 YAML 中设置：

```yaml
policy:
  action_mode: auto
  max_action_dim: 20
```

否则默认 `action_mode=ee6d` 会导致 action 维度不匹配。

### 不要使用 SmolVLA 的配置字段

X-VLA **不支持**以下 SmolVLA 特有字段，配置中不要出现：

* `train_expert_only` — X-VLA 没有此选项
* `train_state_proj` — X-VLA 没有此选项
* `vlm_model_name` / `load_vlm_weights` / `add_image_special_tokens` — X-VLA 使用 Florence2 内置
* `attention_mode` / `prefix_length` / `num_expert_layers` 等 — SmolVLA 专家架构特有

### 图像预处理

X-VLA 默认不固定 resize 图像（`resize_imgs_with_padding=None`）。输入图像由 `XVLAImageToFloatProcessorStep` + `XVLAImageNetNormalizeProcessorStep` 自动处理。若需强制 resize 到固定尺寸，可在 YAML 中设置 `policy.resize_imgs_with_padding: [H, W]`。

---

**相关文档**：
* [SO-ARM101 Pipeline 主文档](./so101_pipeline.md)
* [Pi0.5 模型指南](./so101_pi05.md)
* [推理部署](./inference.md)
* [评估 SOP](./eval.md)
