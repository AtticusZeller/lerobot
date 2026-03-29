# Pi0.5 + SO-ARM101 单卡微调与部署完整 Pipeline

> **本文档基于 LeRobot 官方代码库源码分析生成，所有内容均来自实际代码读取，非推测或概念性说明。**
>
> **当前策略：全量微调 Action Expert**（`train_expert_only=true`），冻结 PaliGemma VLM (~2.7B)，仅训练 Action Expert (~300M, ~5-10% 参数)。适用于 ~80 episodes 数据集、单卡 24GB+ 显存场景。

---

## 目录

1. [前置依赖安装](#0-前置依赖安装)
2. [Pi0.5 微调策略与配置](#1-pi05-微调策略与配置)
3. [数据集格式转换与上传（v2.1 → v3.0）](#1b-数据集格式转换与上传v21--v30)
4. [SO-ARM101 数据集对齐与输入映射](#2-so-arm101-数据集对齐与输入映射)
5. [微调训练与验证 Pipeline](#3-微调训练与验证-pipeline)
6. [Async gRPC 分离式部署方案](#4-async-grpc-分离式部署方案)
7. [端到端流程总结](#5-完整端到端流程总结)
8. [显存估算](#6-单卡显存估算与建议)
9. [附录A：训练结果验证](#附录-a训练结果验证)

---

## 0. 前置依赖安装

```bash
# 基础安装 + Pi0.5 依赖 + 异步推理依赖
uv sync --extra "pi" --extra "async" --extra "dev" --extra "feetech"
```

> `transformers>=5.3.0` 由 `pi` extra 自动安装。全量微调 Action Expert 不需要 `peft` 库。

---

## 1. Pi0.5 微调策略与配置

### 策略选择：全量微调 Action Expert

Pi0.5 由两部分组成：

```
Pi0.5 (~3B 参数)
├── PaliGemma VLM (~2.7B)        ← train_expert_only=true 时冻结
│   ├── Vision Tower (视觉编码器)
│   └── Gemma Language Model
└── Gemma Action Expert (~300M)  ← 始终参与训练（全量更新所有参数）
    ├── action_in_proj / action_out_proj
    ├── time_mlp_in / time_mlp_out
    └── self_attn (Q/K/V proj)
```

**当前方案**：设置 `train_expert_only: true`，冻结整个 PaliGemma VLM，仅全量训练 Action Expert（~300M, ~5-10% 参数）。对于 ~80 episodes 的小规模数据集，这是表达能力与泛化之间的最佳平衡点：

| 方案 | 训练参数量 | 显存需求 | 适用场景 |
|------|-----------|---------|---------|
| **全量 Action Expert**（当前） | ~5-10% (~300M) | ~22-26 GB | 80 episodes 级别，单卡 24GB+ |
| LoRA r=16 | ~0.1-1% | ~14-22 GB | 显存极紧张 (<20 GB) |
| 全参微调 | 100% (~3B) | ~35-45 GB | 数据量大 (>200 ep)，A100-80G+ |

### 源码定位

* **冻结逻辑**: `src/lerobot/policies/pi05/modeling_pi05.py` — `_set_requires_grad()` 方法
* **训练配置**: `src/lerobot/configs/train.py:37` (`TrainPipelineConfig`)
* **Pi0.5 策略配置**: `src/lerobot/policies/pi05/configuration_pi05.py:29` (`PI05Config`)
* **优化器预设**: `src/lerobot/policies/pi05/configuration_pi05.py:142-157`

### `train_expert_only` 的工作原理

源码 `modeling_pi05.py` 的 `_set_requires_grad()` 方法：

```python
def _set_requires_grad(self):
    if self.train_expert_only:       # 冻结整个 PaliGemma，只训练 expert
        self.paligemma.eval()
        for param in self.paligemma.parameters():
            param.requires_grad = False
```

**效果**：PaliGemma VLM 的所有权重被冻结（不计算梯度、不更新参数），仅 Action Expert 的全部参数参与训练。

### YAML 配置文件

所有训练参数统一在 YAML 文件中管理，位于 `experiments/` 目录：

| 配置文件 | 策略 | 说明 |
|---------|------|------|
| `experiments/pi05_expert_so101_table_cleanup.yaml` | **全量 Action Expert**（当前） | 主配置，详见文件内注释 |
| `experiments/pi05_lora_so101_table_cleanup.yaml` | LoRA r=16 | 备选，显存极紧张时使用 |

> **`train_expert_only` 不要与 `peft.*` 混用**：LoRA 的 `wrap_with_peft()` 会先冻结所有参数再注入适配器，与 `train_expert_only` 的手动冻结逻辑冲突。选一种方案即可。

---

## 1b. 数据集格式转换与上传（v2.1 → v3.0）

LeRobot v3.0 引入了与 v2.1 不兼容的新数据集格式。若你使用的是社区 v2.1 数据集（如 `youliangtan/so101-table-cleanup` ），需先转换再上传到自己账号。

### 判断数据集版本

```bash
python -c "
from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata
m = LeRobotDatasetMetadata('youliangtan/so101-table-cleanup')
"
# 若报 BackwardCompatibilityError: dataset is in 2.1 format → 需要转换
```

### 步骤 1：下载并转换（不自动上传）

```bash
python -m lerobot.scripts.convert_dataset_v21_to_v30 \
    --repo-id youliangtan/so101-table-cleanup \
    --root /root/autodl-tmp/lerobot/youliangtan/so101-table-cleanup\
    --push-to-hub False
```

转换后数据存放在 `/root/autodl-tmp/lerobot/youliangtan/so101-table-cleanup/` 。

> **注意**： `--push-to-hub True` 会将转换结果推送回原始 `--repo-id` （即别人账号），通常无权限。应先关闭，手动上传到自己账号。

### 步骤 2：上传到自己的 HF 账号

```bash
# 确保已登录
hf auth login
# check user name
hf auth whoami
# 上传整个数据集目录 assume `Atticuxz` is your_hf_username
hf upload Atticuxz/so101-table-cleanup \
    /root/autodl-tmp/lerobot/youliangtan/so101-table-cleanup \
    --repo-type dataset
```

之后训练命令中使用 `--dataset.repo_id=Atticuxz/so101-table-cleanup` 即可。

```bash
  hf download Atticuxz/so101-table-cleanup \                                                                     
      --repo-type dataset \                                                                                      
      --local-dir /root/autodl-tmp/lerobot/Atticuxz/so101-table-cleanup
```

---

## 2. SO-ARM101 数据集对齐与输入映射

### 源码定位

* **SO follower 配置**: `src/lerobot/robots/so_follower/config_so_follower.py`
* **机器人实现**: `src/lerobot/robots/so_follower/so_follower.py`
* **电机定义**: 6 个 Feetech STS3215 伺服电机 + 1 个夹爪

### SO-ARM101 的观测/动作结构

#### 硬件配置

| 关节名 | 电机 ID | 类型 |
|--------|--------|------|
| `shoulder_pan` | 1 | STS3215 |
| `shoulder_lift` | 2 | STS3215 |
| `elbow_flex` | 3 | STS3215 |
| `wrist_flex` | 4 | STS3215 |
| `wrist_roll` | 5 | STS3215 |
| `gripper` | 6 | STS3215 (torque control) |

#### LeRobot 数据集中的特征映射

当你用 `lerobot-record --robot.type=so101_follower` 采集数据后，原始电机读数被自动映射为：

| LeRobot 数据集键 | 维度 | 说明 |
|---|---|---|
| `observation.state` | (6, ) | 所有关节位置拼接： `[shoulder_pan.pos, shoulder_lift.pos, ..., gripper.pos]` |
| `observation.images.<camera_key>` | (H, W, 3) | 各摄像头视图，键取决于你的 `cameras` 配置 |
| `action` | (6, ) | 目标关节位置 |

#### 典型摄像头配置

你在录制时会配置摄像头，例如：

```python
# lerobot-record 时的 robot config
cameras = {
    "front": OpenCVCameraConfig(index_or_path=0, width=640, height=480, fps=30),
    "wrist": OpenCVCameraConfig(index_or_path=1, width=640, height=480, fps=30),
}
```

对应数据集的观测键：
* `observation.images.front` → 主视角
* `observation.images.wrist` → 腕部视角

### ⚠️ 电机和摄像头命名必须与代码一致

`so_follower.py:53-59` 硬编码了电机字典的顺序：

```python
{
    "shoulder_pan": Motor(1, ...),
    "shoulder_lift": Motor(2, ...),
    "elbow_flex": Motor(3, ...),
    "wrist_flex": Motor(4, ...),
    "wrist_roll": Motor(5, ...),
    "gripper": Motor(6, ...),
}
```

这个顺序决定了 `observation.state` 和 `action` 向量 (6, ) 的维度语义。

**常见踩坑**：标定时电机 ID 和代码不匹配，导致采集到的 state 向量里每个维度的物理含义都是错的。

同样，摄像头 key（ `observation.images.front` 、 `observation.images.wrist` ）必须在录制和推理时保持一致。如果录制时摄像头叫 `cam_0` / `cam_1` ，推理时也必须用同样的名字。

只要你用 LeRobot 框架完成采集 → 训练 → 推理全流程，数据集特征的顺序自然和推理代码一致。**唯一会出问题的点就是电机标定 ID 与代码不一致。**

---

### Pi0.5 如何处理这些特征

源码 `configuration_pi05.py:118-140` 的 `validate_features()` 方法：

```python
def validate_features(self) -> None:
    """Validate and set up input/output features."""
    # 如果有空摄像头参数，补充占位特征
    for i in range(self.empty_cameras):
        key = OBS_IMAGES + f".empty_camera_{i}"
        self.input_features[key] = PolicyFeature(
            type=FeatureType.VISUAL,
            shape=(3, 224, 224),
        )

    # 如果 state 不存在，创建默认
    if OBS_STATE not in self.input_features:
        self.input_features[OBS_STATE] = PolicyFeature(
            type=FeatureType.STATE,
            shape=(self.max_state_dim,),  # Padding to 32
        )

    # 如果 action 不存在，创建默认
    if ACTION not in self.output_features:
        self.output_features[ACTION] = PolicyFeature(
            type=FeatureType.ACTION,
            shape=(self.max_action_dim,),  # Padding to 32
        )
```

**关键点**：你的 6-DoF 动作完全兼容——多余维度自动用 0 填充到 `max_action_dim=32` 。

### 摄像头特征是自动读取的

训练时 `make_policy()` 调用 `dataset_to_policy_features(ds_meta.features)` （ `factory.py:458` ），直接从数据集的 `meta/info.json` 读取所有特征定义，按键名前缀自动分类：

* `observation.images.*` → `FeatureType.VISUAL`
* `observation.state` → `FeatureType.STATE`
* `action` → `FeatureType.ACTION`

**不需要手动映射摄像头**。无论你的数据集摄像头叫 `front` / `wrist` 还是 `cam_0` / `cam_1` ，模型只看特征类型和 shape，不看名字。基座模型 `lerobot/pi05_base` 在异构多机器人数据上预训练，不绑定特定摄像头名称或数量。

#### `empty_cameras` （极少用到）

数据集只有一个摄像头时，可补充空白占位：

```bash
--policy.empty_cameras=1
```

#### `rename_map` （极少用到）

仅在合并多个数据集键名冲突、或键名不符合 `observation.images.*` 标准格式时才需要。**正常单数据集微调不需要**。

### Quantile 统计量处理（Pi0.5 数据预处理必需步骤）

Pi0.5 默认使用 **QUANTILES** 归一化（而非 MEAN_STD）。这是 Pi0.5 的架构偏好——状态值被离散化为 256 个 bin，需要 q01/q99 分位数来确定归一化范围。源码 `configuration_pi05.py:66-72` ：

```python
normalization_mapping: dict[str, NormalizationMode] = field(
    default_factory=lambda: {
        "VISUAL": NormalizationMode.IDENTITY,
        "STATE": NormalizationMode.QUANTILES,   # ← Pi0.5 特色
        "ACTION": NormalizationMode.QUANTILES,  # ← Pi0.5 特色
    }
)
```

大部分 LeRobot 数据集在创建时只包含 mean/std 统计量，**不包含** quantile 统计量（q01, q10, q50, q90, q99）。使用 Pi0.5 训练前，必须先补充这些统计量，否则会报错：

```
ValueError: QUANTILES normalization mode requires q01 and q99 stats,
please update the dataset with the correct stats using the `augment_dataset_quantile_stats.py` script
```

#### 方案 A：生成 quantile 统计量（推荐）

> **前提条件**：数据集必须已经下载到本地。脚本不会自动下载，本地不存在会触发 `get_safe_version` 的 `huggingface_hub` 版本兼容 bug。

**Step 1：确保数据集在本地**

```bash
# 如果本地还没有数据集，先下载
hf download Atticuxz/so101-table-cleanup \
    --repo-type dataset \
    --local-dir /root/autodl-tmp/lerobot/Atticuxz/so101-table-cleanup
```

**Step 2：计算 quantile 统计量**

```bash
python src/lerobot/scripts/augment_dataset_quantile_stats.py \
    --repo-id=Atticuxz/so101-table-cleanup \
    --root=/root/autodl-tmp/lerobot/Atticuxz/so101-table-cleanup
```

脚本执行流程（ `augment_dataset_quantile_stats.py` ）：
1. 加载本地数据集 → 检查是否已有 quantile stats
2. 逐 episode 计算 q01/q10/q50/q90/q99 统计量（视频帧串行处理，`line 149-153`）
3. 写入本地 `meta/stats.safetensors`
4. 自动 `push_to_hub()` 推回 HF Hub（需写权限）

80 episodes + 视频帧，预计耗时 **10~30 分钟**。

> 📚 **详细说明**: [augment_dataset_quantile_stats.py 脚本详解与 Quantile 统计量格式](./augment_quantile_stats_explained.md) — 包含 stats.json 格式、特征含义、源码定位等。

#### 方案 B：训练时覆盖归一化方式（备选）

如果不想修改数据集，可在训练命令中直接覆盖为 MEAN_STD（所有数据集都有此统计量）：

```bash
--policy.normalization_mapping='{"ACTION": "MEAN_STD", "STATE": "MEAN_STD", "VISUAL": "IDENTITY"}'
```

精度上与 QUANTILES 差异不大，适合快速验证。后续正式训练建议用方案 A。

---

## 3. 微调训练与验证 Pipeline

### 源码定位

* **训练入口**: `src/lerobot/scripts/lerobot_train.py:153` (`train()` 函数)
* **训练配置**: `src/lerobot/configs/train.py:37` (`TrainPipelineConfig`)
* **优化器预设**: `src/lerobot/policies/pi05/configuration_pi05.py:142-157`

### 验证（Validation）机制

#### SO-ARM101 的特殊性

* **无仿真环境**：SO-ARM101 没有内置仿真器。训练脚本中的 `eval_env` 仅在 `cfg.env is not None` 时创建（`lerobot_train.py:234`）。
* **仅离线验证**：训练过程中只记录 **Training Loss**，没有 sim rollout 评估。
* **实机验证**：训练完成后，需使用 `lerobot-eval` 或 async inference 在真实机械臂上验证。

### Pi0.5 的优化器/调度器预设

源码 `configuration_pi05.py:142-157` 的 `get_optimizer_preset()` 和 `get_scheduler_preset()` ：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| **Optimizer** | AdamW | - |
| `lr` | 2.5e-5 | 学习率 |
| `betas` | (0.9, 0.95) | Adam beta 参数 |
| `eps` | 1e-8 | 数值稳定性 |
| `weight_decay` | 0.01 | L2 正则化 |
| `grad_clip_norm` | 1.0 | 梯度裁剪 |
| **Scheduler** | CosineDecayWithWarmup | - |
| `warmup_steps` | 1000 | 预热步数（自动缩放） |
| `decay_steps` | 30000 | 衰减步数（自动缩放） |
| `decay_lr` | 2.5e-6 | 最小学习率 |

**自动缩放**：当 `--steps < scheduler_decay_steps` 时，会等比缩放 warmup 和 decay 步数。例如 `--steps=3000` 时，warmup 缩放为 100，decay 为 3000。

### 启动训练

使用 YAML 配置文件启动训练（所有参数见 YAML 文件内注释）：

```bash
# 用时间戳区分每次训练，无需手动清理目录
RUN_DIR="/root/autodl-tmp/outputs/pi05_expert_so101/$(date +%Y%m%d_%H%M%S)"

lerobot-train \
    --yaml_config=experiments/pi05_expert_so101_table_cleanup.yaml \
    --output_dir="$RUN_DIR"
```

CLI 参数可覆盖 YAML 中的任意值（用于临时调参）：

```bash
# 调整 steps 和 batch_size
lerobot-train --yaml_config=experiments/pi05_expert_so101_table_cleanup.yaml \
    --steps=10000 --batch_size=8

# 切换数据集
lerobot-train --yaml_config=experiments/pi05_expert_so101_table_cleanup.yaml \
    --dataset.repo_id=Atticuxz/so101-new-task
```

> 备选 LoRA 方案（显存 <20GB 时）：
> `lerobot-train --yaml_config=experiments/pi05_lora_so101_table_cleanup.yaml`

#### 根据数据量确定 Steps

| 数据规模 | episodes | 约 frames | 建议 steps | 说明 |
|---------|---------|---------|---------|------|
| 极少 | < 30 | < 3000 | 2000~3000 | 数据少，防过拟合 |
| 少 | 30~100 | 3000~10000 | 3000~8000 | 典型小规模采集 |
| 中 | 100~300 | 10000~30000 | 10000~20000 | 社区参考量级 |
| 大 | > 300 | > 30000 | 30000~50000 | 参考命令量级 |

**80 episodes 推荐**：约 8000 frames，batch_size=4 下约 2000 steps/epoch，`steps: 8000` ≈ 4 epoch。

### 数据集摄像头键不匹配时

如果你的数据集用了 `laptop` 和 `phone` 作为摄像头键，可在 YAML 中添加 `rename_map`：

```yaml
rename_map:
  "observation.images.laptop": "observation.images.front"
  "observation.images.phone": "observation.images.wrist"
```

---

## 4. 推理部署（gRPC 异步推理）

LeRobot 提供 gRPC 异步推理架构，将模型推理与机器人控制解耦，支持远程部署。

### 架构概览

```
GPU 服务器 (Policy Server) ← gRPC → 笔记本 (Robot Client) ← USB → SO-101
   Pi0.5 checkpoint                              控制循环                电机控制
```

**关键特性**:
- Pi0.5 (`pi05`) 和 SO-ARM101 (`so101_follower`) 均在支持列表中
- 异步推理消除"等待推理"的空闲帧，实现更平滑的控制
- 支持本地部署和远程部署

### 快速启动

```bash
# Terminal 1 (GPU 服务器): 启动 Policy Server
python -m lerobot.async_inference.policy_server \
    --host=0.0.0.0 \
    --port=8080

# Terminal 2 (SO-ARM101 控制器): 启动 Robot Client
python -m lerobot.async_inference.robot_client \
    --server_address=<GPU_IP>:8080 \
    --robot.type=so101_follower \
    --robot.port=/dev/ttyACM1 \
    --robot.id=my_so101 \
    --robot.cameras="{ top: {type: opencv, index_or_path: /dev/video10, width: 640, height: 480, fps: 30 }}" \
    --task="把桌上的文具收到笔盒里" \
    --policy_type=pi05 \
    --pretrained_name_or_path=Atticuxz/pi05_expert_so101 \
    --actions_per_chunk=50 \
    --chunk_size_threshold=0.5
```

### 详细文档

完整的推理部署流程、参数调优、故障排查请参考:

**[inference.md](./inference.md)** - 包含:
- 环境准备与依赖安装
- Policy Server 与 Robot Client 详细配置
- 关键参数调优 (`actions_per_chunk`, `chunk_size_threshold`)
- Checkpoint 快速切换（评估用）
- 录制评估数据
- 常见故障排查

---

## 5. 完整端到端流程总结

### 完整工作流

```
┌─────────────────────────────────────────────────────────┐
│ Step 1: 数据采集 (Real Robot)                            │
│ 命令: lerobot-record --robot.type=so101_follower ...     │
│ 输出: HF Dataset (your_username/so101_pickplace)         │
└─────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────┐
│ Step 1b: 数据集格式转换（v2.1 → v3.0，如需）            │
│ 命令: python -m lerobot.scripts.convert_dataset_v21_to_v30 \│
│         --repo-id=<源数据集> \                           │
│         --root=/path/to/converted \                      │
│         --push-to-hub False                              │
│ 作用: 将 HF Hub 上的 v2.1 数据集转换为 v3.0 格式         │
└─────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────┐
│ Step 1c: 上传转换后数据集到 Hub                          │
│ 命令: hf upload Atticuxz/so101-table-cleanup \           │
│         /path/to/converted/<源数据集> \                  │
│         --repo-type dataset                              │
│ 作用: 将本地 v3.0 数据集推送到自己账号                    │
└─────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────┐
│ Step 2: 数据准备 (Optional)                              │
│ 命令: python src/lerobot/datasets/v30/                   │
│       augment_dataset_quantile_stats.py                  │
│       --repo-id=your_username/so101_pickplace            │
│ 作用: 生成 Pi0.5 需要的 quantile 统计量                   │
└─────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────┐
│ Step 3: 单卡微调 (train_expert_only)                      │
│ 命令: lerobot-train                                       │
│         --yaml_config=experiments/                        │
│           pi05_expert_so101_table_cleanup.yaml            │
│         --output_dir=/path/to/output                      │
│ 输出: Checkpoint (outputs/pi05_expert_so101/...)          │
│ 时长: ~1-3 小时 (取决于 GPU 和数据量)                      │
└─────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────┐
│ Step 4a: 离线验证 (可选)                                 │
│ 命令: lerobot-eval --policy.type=pi05 \                 │
│         --policy.pretrained_path=./outputs/... \         │
│         --dataset.repo_id=your_username/... \            │
│         --splits=["test"]                                │
│ 输出: Loss 指标                                          │
└─────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────┐
│ Step 4b: Async gRPC 部署                                │
│                                                          │
│ Terminal 1 (GPU 机器):                                  │
│ python -m lerobot.async_inference.policy_server \       │
│   --host=0.0.0.0 --port=8080                           │
│                                                          │
│ Terminal 2 (SO-ARM101 控制器):                           │
│ python -m lerobot.async_inference.robot_client \        │
│   --server_address=<GPU_IP>:8080 \                      │
│   --robot.type=so101_follower \                         │
│   --pretrained_name_or_path=./outputs/... \             │
│   --policy_type=pi05 \                                  │
│   ...                                                    │
│                                                          │
│ 实时运行，观察机械臂执行微调后的行为                     │
└─────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────┐
│ Step 5: 推送到 HF Hub (可选)                              │
│ 命令: hf upload Atticuxz/pi05_expert_so101 \              │
│         outputs/pi05_expert_so101/ \                      │
│         --repo-type model                                 │
│ 或在 YAML 中设置:                                         │
│   push_to_hub: true                                       │
│   repo_id: your_hf_username/pi05_expert_so101             │
└─────────────────────────────────────────────────────────┘
```

### 目录结构示意

```
outputs/
├── pi05_expert_so101/
│   ├── train_config.json              # 训练配置快照
│   ├── training_state/
│   │   ├── optimizer_state.safetensors
│   │   ├── scheduler_state.json
│   │   └── ...
│   └── checkpoints/
│       ├── 1000/
│       │   └── pretrained_model/
│       │       ├── config.json        # Pi0.5 策略配置
│       │       ├── model.safetensors  # 完整权重 (~15GB)
│       │       ├── processor.json
│       │       └── ...
│       ├── 2000/
│       └── last/                      # 最新检查点，部署时用这个
│           └── pretrained_model/
│               └── ...
```

**关键**：部署时使用 `checkpoints/last/pretrained_model/` ，它包含训练后的完整 Action Expert 权重。

---

## 6. 单卡显存估算与建议

### 显存使用情况

| 配置 | 显存需求 | 适合 GPU | 备注 |
|------|----------|---------|------|
| **train_expert_only**（当前主方案）+ bf16 + gradient_checkpointing + bs=4 | ~22-26 GB | A100-40G, A5000 | 全参但仅 expert，~80 episodes 推荐 |
| train_expert_only + bf16 + gradient_checkpointing + bs=2 | ~18-22 GB | RTX 4090, A5000 | 单卡 24GB 保守设置 |
| LoRA r=16 + bf16 + gradient_checkpointing + bs=4 | ~18-22 GB | A5000, A100-40G | 显存紧张时备选 |
| LoRA r=16 + bf16 + gradient_checkpointing + bs=2 | ~14-16 GB | RTX 4090 | 最紧凑 |
| 全参微调 + bf16 + gradient_checkpointing + bs=4 | ~35-45 GB | A100-80G, H100 | 数据量大 (>200 ep) |

### 优化策略

#### 优先级 1: Gradient Checkpointing（显存 -50%）

```bash
--policy.gradient_checkpointing=true
```

原理：不保存所有中间激活值，只在反向传播时重新计算。显存减半，速度减 ~20%。

#### 优先级 2: Mixed Precision (bfloat16)（显存 -50%，速度 +10-20%）

```bash
--policy.dtype=bfloat16
```

权重和激活从 float32 降至 bfloat16。显存减半，计算快。

#### 优先级 3: 降低 Batch Size

```bash
--batch_size=2  # 从 4 降至 2，显存减 50%
--batch_size=1  # 极端情况
```

#### 优先级 4: LoRA Rank 降低

```bash
--peft.r=8   # 从 16 降至 8，参数量减 50%
--peft.r=4   # 极端情况，适应能力下降
```

### 推荐单卡配置

**24GB 显存（RTX 4090, A5000）**:

```bash
--policy.dtype=bfloat16 \
--policy.gradient_checkpointing=true \
--peft.r=16 \
--batch_size=4 \
--steps=5000
```

**16GB 显存（RTX 3090, RTX 4080）**:

```bash
--policy.dtype=bfloat16 \
--policy.gradient_checkpointing=true \
--peft.r=8 \
--batch_size=2 \
--steps=3000
```

**12GB 显存（RTX 3060）**:

```bash
--policy.dtype=bfloat16 \
--policy.gradient_checkpointing=true \
--peft.r=4 \
--batch_size=1 \
--steps=2000
```

### 监测显存使用

训练中可用 `nvidia-smi` 动态监测：

```bash
watch -n 1 nvidia-smi
```

或创建单独终端持续监控：

```bash
nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits -l 1
```

---

## 总结

### 关键要点

1. **Pi0.5 + train_expert_only** 的全量微调流程：
   - 冻结 PaliGemma VLM (~2.7B)，仅全量训练 Action Expert (~300M, ~5-10% 参数)
   - 通过 YAML 配置文件管理所有训练参数 (`experiments/pi05_expert_so101_table_cleanup.yaml`)
   - CLI 参数可覆盖 YAML 值，用于临时调参

2. **SO-ARM101 特性**：
   - 6-DoF + 1-Gripper（共 6 维度）
   - Pi0.5 自动 pad 到 32 维
   - 支持多摄像头（典型双目配置）
   - 无仿真环境，仅离线 loss 验证

3. **训练优化**：
   - 使用 bfloat16 + gradient_checkpointing 最高效
   - `train_expert_only=true` 适合 ~80 episodes 数据规模
   - 单卡 22-26GB 显存可行（A100-40G, A5000, RTX 4090）

4. **部署架构**：
   - Async gRPC 解耦推理和执行
   - PolicyServer 运行在 GPU，RobotClient 连接 SO-ARM101
   - 30 Hz 控制频率、50 步 chunk、0.5 阈值为推荐配置

### 文档引用

* [LeRobot Pi0.5 文档](./pi05.mdx)
* [LeRobot Async 推理文档](./async.mdx)
* [LeRobot 安装指南](./installation)

---

## 附录 A：训练结果验证

### 为什么 SO-ARM101 没有内置验证指标？

LeRobot 的 `lerobot-eval` 依赖仿真环境（ `gym.vector.VectorEnv` ）来执行 rollouts 并计算 reward 和 success rate。但 `envs/` 下没有 SO-ARM101 对应的仿真器，因此：

* `eval_freq=0`（关闭训练中评估）
* 训练全程**只记录 Training Loss**，没有 validation loss
* 推理验证只能通过 **离线脚本** 或 **实机测试**

### Pi0.5 Loss 计算方式

Pi0.5 使用 **Flow Matching** 作为动作生成范式，loss 是标准的 MSE：

```python
# modeling_pi05.py:730-783
noise = sample_noise(actions.shape)           # 随机噪声
time = sample_time(batch_size)                  # t ∈ [0, 1]

x_t = t * noise + (1 - t) * actions            # 在噪声和动作之间插值
v_t = noise - actions                            # 目标速度场

v_pred = model(x_t, time)                       # 模型预测的速度
loss = MSE(v_t, v_pred)                         # 对所有 action 维度和时间步求均值
```

训练时： `t` 越接近 1（噪声端），模型学习去噪； `t` 越接近 0（动作端），模型学习重建动作。

### WandB 中应关注的关键曲线

| WandB 字段 | 含义 | 期望形态 |
|------------|------|----------|
| `loss` | Flow Matching MSE scalar | 全程稳定下降 |
| `loss_per_dim` | 各 action 维度的 loss 均值（数组） | 各维度大致均衡 |
| `grad_norm` | 梯度范数 | 100 以下波动，无爆炸 |
| `lr` | 学习率 | 按 warmup + decay 曲线变化 |
| `num_learnable_params / num_total_params` | 可训练参数比例 | LoRA 应为 0.03%~1% |

**收敛阶段参考**（以 5000 步、batch_size=4 为例）：

| 阶段 | Steps | 期望 |
|------|-------|------|
| 快速下降期 | 0 ~ 1000 | Loss 从高位快速回落，下降斜率最大 |
| 缓慢收敛期 | 1000 ~ 3000 | 下降速度减缓，曲线趋于平滑 |
| 平台期 | 3000 ~ 5000 | Loss 接近平台，若仍有微降可延长训练 |

### 过拟合信号识别

**在有仿真器的任务中**：可以通过 `eval_freq` 的 eval loss 曲线对比训练 loss — 若 eval loss 上升而训练 loss 持续下降，则过拟合。

**在 SO-ARM101 中**（无仿真器）：无法从训练曲线直接判断过拟合。过拟合的**间接信号**：
* 数据集规模很小（< 500 episodes）但训练 steps 很多（> 10000）
* `loss_per_dim` 中某一维度持续远高于其他维度
* 实机测试时动作剧烈抖动或超出关节限位

**缓解方法**：
* 降低 `steps`（如从 10000 降到 5000）
* 增加 `lora_dropout`（PEFT config 中的 dropout）
* 使用更大的 `r` 值（提升 adapter 容量同时防止欠拟合）

### Checkpoint 选择流程

```
训练完成
  ↓
查看 wandb loss 曲线
  ↓
loss 正常收敛?
  ├── 否 → 检查数据、配置、重训
  └── 是 → 进入下一步
  ↓
选出 3-5 个候选 checkpoint（平台期前/中/后段各 1-2 个）
  ↓
真机评估（详见 [eval.md](./eval.md)）
  ↓
按 π0.5 评分 rubric 打分（接近/抓取/运输/放置）
  ↓
选择平均得分最高的 checkpoint
```

---

---

* [LeRobot Pi0.5 文档](./pi05.mdx)
* [LeRobot Async 推理文档](./async.mdx)
* [LeRobot 安装指南](./installation)

---

---

## 常见报错与修复

### `ValueError: 'policy.repo_id' argument missing`

**原因**： `PreTrainedConfig.push_to_hub` 默认为 `True` （ `configs/policies.py:70` ）。

**修复**：所有训练命令中加入：

```bash
--policy.push_to_hub=false
```

或者指定上传目标（二选一）：

```bash
--policy.push_to_hub=true \
--policy.repo_id=your_hf_username/pi05_so101_lora
```

---

**本文档最后更新于 2026-03-26，基于 LeRobot dev 分支**

**更新记录**：
* v1.0：初始版本，覆盖 LoRA 配置、数据集对齐、训练、gRPC 部署
* v1.1：补充 `--policy.push_to_hub=false` 必须项、`train_expert_only` 与 LoRA 冲突警告、按数据量计算 steps 指南、社区全参微调命令对比
* v1.2：新增 Section 1b「数据集格式转换与上传（v2.1 → v3.0）」，流程图补充 Step 1b/1c
* v1.3：新增 YAML 配置文件启动方式（`--yaml_config`）、方案 A-2 文档；`experiments/` 目录放置实验配置
* v1.4：策略切换至 `train_expert_only=true`（方案 B），暂不使用 LoRA；新增 `experiments/pi05_expert_so101_table_cleanup.yaml`
* v1.5：**文档重构**：主方案改为全量微调 Action Expert，移除 CLI 参数拼接示例，统一使用 YAML 配置启动；精简冗余参数表格，配置文件详情移入 YAML 文件内注释
