# SO-ARM101 单卡微调与部署完整 Pipeline

> **本文档基于 LeRobot 官方代码库源码分析生成，所有内容均来自实际代码读取，非推测或概念性说明。**
>
> 本文档为通用 SO-ARM101 Pipeline，适用于所有支持的 VLA 模型。模型特定的配置、训练参数和显存估算请参考对应子文档。

### 模型子文档

| 模型 | 子文档 | 基座 | 参数量 | 最低显存 |
|------|--------|------|--------|---------|
| **SmolVLA** | [so101_smolvla.md](./so101_smolvla.md) | `lerobot/smolvla_base` (~450M) | ~450M | ~8-12 GB |
| **Pi0.5** | [so101_pi05.md](./so101_pi05.md) | `lerobot/pi05_base` (~3B) | ~3B | ~14-16 GB |

---

## 目录

1. [前置依赖安装](#0-前置依赖安装)
2. [数据集格式转换与上传（v2.1 → v3.0）](#1-数据集格式转换与上传v21--v30)
3. [SO-ARM101 数据集对齐与输入映射](#2-so-arm101-数据集对齐与输入映射)
4. [微调训练（通用流程）](#3-微调训练通用流程)
5. [推理部署（gRPC 异步推理）](#4-推理部署grpc-异步推理)
6. [端到端流程总结](#5-端到端流程总结)
7. [常见报错与修复](#6-常见报错与修复)

---

## 0. 前置依赖安装

按你选择的模型安装对应依赖：

```bash
# SmolVLA
uv sync --extra "smolvla" --extra "dev" --extra "feetech"

# Pi0.5
uv sync --extra "pi" --extra "dev" --extra "feetech"

# 如需 gRPC 异步推理
uv sync --extra "async"
```

---

## 1. 数据集格式转换与上传（v2.1 → v3.0）

LeRobot v3.0 引入了与 v2.1 不兼容的新数据集格式。若你使用的是社区 v2.1 数据集（如 `youliangtan/so101-table-cleanup`），需先转换再上传到自己账号。

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
    --root /root/autodl-tmp/lerobot/youliangtan/so101-table-cleanup \
    --push-to-hub False
```

> **注意**：`--push-to-hub True` 会推送回原始 `--repo-id`（即别人账号），通常无权限。应先关闭，手动上传到自己账号。

### 步骤 2：上传到自己的 HF 账号

```bash
# 确保已登录
hf auth login
hf auth whoami

# 上传（Atticuxz 替换为你的 HF 用户名）
hf upload Atticuxz/so101-table-cleanup \
    /root/autodl-tmp/lerobot/youliangtan/so101-table-cleanup \
    --type dataset
```

之后训练命令中使用 `--dataset.repo_id=Atticuxz/so101-table-cleanup` 即可。

```bash
# 下载已上传的数据集到本地

hf download Atticuxz/so101-table-cleanup \
    --type dataset \
    --local-dir /root/autodl-tmp/lerobot/Atticuxz/so101-table-cleanup
```

---

## 2. SO-ARM101 数据集对齐与输入映射

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
| `observation.state` | (6, ) | 所有关节位置拼接：`[shoulder_pan.pos, ..., gripper.pos]` |
| `observation.images.<camera_key>` | (H, W, 3) | 各摄像头视图 |
| `action` | (6, ) | 目标关节位置 |

#### 典型摄像头配置

```python
cameras = {
    "front": OpenCVCameraConfig(index_or_path=0, width=640, height=480, fps=30),
    "wrist": OpenCVCameraConfig(index_or_path=1, width=640, height=480, fps=30),
}
```

对应数据集观测键：`observation.images.front`、`observation.images.wrist`。

### 电机和摄像头命名必须与代码一致

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

**常见踩坑**：标定时电机 ID 和代码不匹配，导致 state 向量维度语义错误。

摄像头 key 必须在录制和推理时保持一致。只要用 LeRobot 框架完成采集 → 训练 → 推理全流程，数据集特征的顺序自然一致。

### 特征自动映射

训练时 `make_policy()` 调用 `dataset_to_policy_features(ds_meta.features)`，直接从数据集的 `meta/info.json` 读取所有特征定义，按键名前缀自动分类：

* `observation.images.*` → `FeatureType.VISUAL`
* `observation.state` → `FeatureType.STATE`
* `action` → `FeatureType.ACTION`

**不需要手动映射摄像头**。模型只看特征类型和 shape，不看名字。

#### 数据集摄像头键不匹配时

如果录制时摄像头叫 `laptop` / `phone`，可通过 `rename_map` 重命名：

```yaml
rename_map:
  "observation.images.laptop": "observation.images.front"
  "observation.images.phone": "observation.images.wrist"
```

### 源码定位

* **SO follower 配置**: `src/lerobot/robots/so_follower/config_so_follower.py`
* **机器人实现**: `src/lerobot/robots/so_follower/so_follower.py`

---

## 3. 微调训练（通用流程）

### 模型特定准备

不同模型在训练前有不同的数据准备需求：

| 模型 | 归一化模式 | 是否需要额外数据处理 |
|------|-----------|-------------------|
| **SmolVLA** | MEAN_STD | **不需要**，数据集默认包含 mean/std |
| **Pi0.5** | MEAN_STD（推荐覆盖） | **不需要**，YAML 已配置覆盖为 MEAN_STD，详见 [so101_pi05.md](./so101_pi05.md#3-归一化模式选择) |

### 训练前：诊断控制延迟（State–Action Temporal Alignment）

在 visualize_dataset 的 **Action Insights → State–Action Temporal Alignment** 面板（详见 [`data_analysis_guide.md §2.4`](./data_analysis_guide.md)），查看数据集的 **Mean control delay**：

| Mean control delay | 处理 |
|---|---|
| 0–2 步 | 无需处理，直接训练 ✅ |
| 3–5 步 | 考虑在训练配置中将 action label 前移 L 步（`delta_timestamps` 偏移） |
| > 5 步 | **必须**调整，否则 closed-loop 推理会震荡 |

延迟较大时的对齐方法见 `data_analysis_guide.md §2.4`（录制机制与内置延迟的来源）。

### 训练命令

> 具体训练参数、YAML 配置、steps 建议请参考对应模型子文档。

**SmolVLA**（详见 [so101_smolvla.md](./so101_smolvla.md#2-训练命令与参数)）：

```bash
lerobot-train \
    --policy.path=lerobot/smolvla_base \
    --dataset.repo_id=Atticuxz/so101-table-cleanup \
    --batch_size=64 --steps=20000 \
    --output_dir=outputs/smolvla_so101
```

**Pi0.5**（详见 [so101_pi05.md](./so101_pi05.md#4-训练命令与参数)）：

```bash
lerobot-train \
    --yaml_config=experiments/pi05_expert_so101_table_cleanup.yaml \
    --output_dir=outputs/pi05_expert_so101
```

### 验证机制

* **无仿真环境**：SO-ARM101 没有内置仿真器，训练过程中只记录 Training Loss
* **实机验证**：训练完成后，通过 `lerobot-record` 或 async inference 在真实机械臂上验证
* 详见 [eval.md](./eval.md)

---

## 4. 推理部署（gRPC 异步推理）

LeRobot 提供 gRPC 异步推理架构，将模型推理与机器人控制解耦，支持远程部署。

### 架构概览

```
GPU 服务器 (Policy Server) ← gRPC → 笔记本 (Robot Client) ← USB → SO-101
   VLA checkpoint                          控制循环                电机控制
```

**关键特性**:
- SmolVLA (`smolvla`) 和 Pi0.5 (`pi05`) 均在支持列表中
- 异步推理消除"等待推理"的空闲帧，实现更平滑的控制
- 支持本地部署和远程部署

### 快速启动

```bash
# Terminal 1 (GPU 服务器): 启动 Policy Server
python -m lerobot.async_inference.policy_server \
    --host=0.0.0.0 --port=8080

# Terminal 2 (SO-ARM101 控制器): 启动 Robot Client
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

> 将 `--policy_type` 和 `--pretrained_name_or_path` 替换为你使用的模型。

### 详细文档

完整的推理部署流程、参数调优、故障排查请参考 **[inference.md](./inference.md)**。

---

## 5. 端到端流程总结

```
┌─────────────────────────────────────────────────────────┐
│ Step 1: 数据采集 (Real Robot)                            │
│ 命令: lerobot-record --robot.type=so101_follower ...     │
│ 输出: HF Dataset (your_username/so101_pickplace)         │
└─────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────┐
│ Step 1b: 数据集格式转换（v2.1 → v3.0，如需）             │
│ 命令: python -m lerobot.scripts.convert_dataset_v21_to_v30│
│ 作用: 将 v2.1 数据集转换为 v3.0 格式                     │
└─────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────┐
│ Step 1c: 上传转换后数据集到 Hub                           │
│ 命令: hf upload your_username/dataset_name ...            │
└─────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────┐
│ Step 2: 模型特定数据准备                                  │
│ - SmolVLA: 无需额外处理                                   │
│ - Pi0.5:  推荐 MEAN_STD，无需额外处理                     │
│ 详见对应模型子文档                                        │
└─────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────┐
│ Step 3: 单卡微调                                          │
│ 命令: lerobot-train --policy.path=<基座模型> ...          │
│ 详见对应模型子文档的训练参数                               │
└─────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────┐
│ Step 4: 实机验证                                          │
│ 方式 A: lerobot-record --policy.path=<checkpoint>         │
│ 方式 B: gRPC 异步推理                                     │
│ 评估标准: eval.md                                         │
└─────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────┐
│ Step 5: 推送到 HF Hub (可选)                              │
│ 命令: hf upload your_username/model_name outputs/... \    │
│       --repo-type model                                   │
└─────────────────────────────────────────────────────────┘
```

### 目录结构示意

```
outputs/
├── smolvla_so101/                    # 或 pi05_expert_so101/
│   ├── train_config.json
│   ├── training_state/
│   └── checkpoints/
│       ├── 1000/pretrained_model/
│       ├── 2000/pretrained_model/
│       └── last/pretrained_model/    # ← 部署时用这个
│           ├── config.json
│           ├── model.safetensors
│           └── ...
```

---

## 6. 常见报错与修复

### `ValueError: 'policy.repo_id' argument missing`

**原因**：`PreTrainedConfig.push_to_hub` 默认为 `True`（`configs/policies.py:70`）。

**修复**：

```bash
--policy.push_to_hub=false
# 或指定上传目标
--policy.push_to_hub=true --policy.repo_id=your_hf_username/model_name
```

---

**本文档最后更新于 2026-03-30，基于 LeRobot dev 分支**

**更新记录**：
* v1.0~v1.5：初始版本至 Pi0.5 专属文档（历史记录见 git log）
* v2.0：**文档重构**：拆分为通用 Pipeline + 模型子文档（Pi0.5 / SmolVLA），支持多模型切换
