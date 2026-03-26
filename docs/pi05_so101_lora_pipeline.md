# Pi0.5 + SO-ARM101 单卡 LoRA 微调与部署完整 Pipeline

> **本文档基于 LeRobot 官方代码库源码分析生成，所有内容均来自实际代码读取，非推测或概念性说明。**

---

## 目录

1. [前置依赖安装](#0-前置依赖安装)
2. [Pi0.5 的 LoRA 微调配置](#1-pi05-的-lora-微调配置)
3. [数据集格式转换与上传（v2.1 → v3.0）](#1b-数据集格式转换与上传v21--v30)
4. [SO-ARM101 数据集对齐与输入映射](#2-so-arm101-数据集对齐与输入映射)
4. [微调训练与验证 Pipeline](#3-微调训练与验证-pipeline)
5. [Async gRPC 分离式部署方案](#4-async-grpc-分离式部署方案)
6. [端到端流程总结](#5-完整端到端流程总结)
7. [显存估算](#6-单卡显存估算与建议)
8. [附录A：训练结果验证](#附录-a训练结果验证)

---

## 0. 前置依赖安装

```bash
# 基础安装 + Pi0.5 依赖 + 异步推理依赖
uv sync --extra "pi" --extra "async" --extra "dev" --extra "feetech"

# 确保 PEFT 库已安装 (wrap_with_peft 需要)
pip install peft
```

---

## 1. Pi0.5 的 LoRA 微调配置

### 源码定位

* **PEFT 配置定义**: `src/lerobot/configs/default.py:84-108` (`PeftConfig`)
* **Pi0.5 默认 LoRA 目标模块**: `src/lerobot/policies/pi05/modeling_pi05.py:1285-1294`
* **PEFT 注入入口**: `src/lerobot/scripts/lerobot_train.py:246-250`
* **PEFT 封装逻辑**: `src/lerobot/policies/pretrained.py:269-316` (`wrap_with_peft`)

### LeRobot 如何注入 LoRA 到 Pi0.5

LeRobot **不在** Pi0.5 内部实现 LoRA，而是通过 HuggingFace `peft` 库在训练脚本中外部包装：

```python
# lerobot_train.py:246-250
if cfg.peft is not None:
    peft_cli_overrides = dataclasses.asdict(cfg.peft)
    policy = policy.wrap_with_peft(peft_cli_overrides=peft_cli_overrides)
```

`wrap_with_peft()` 在 `pretrained.py:269-316` 中的执行流程：

1. **冻结所有参数**: `for p in self.parameters(): p.requires_grad_(False)`
2. **构建 PEFT 配置**: 调用 `_build_peft_config()` 合并 Pi0.5 的 `_get_default_peft_targets()` + CLI 覆盖
3. **应用 PEFT**: 用 `peft.get_peft_model(self, final_config)` 包装模型
4. **标记状态**: 设置 `peft_model.config.use_peft = True`

### Pi0.5 默认 LoRA 目标模块

源码 `modeling_pi05.py:1285-1294` :

```python
def _get_default_peft_targets(self) -> dict[str, any]:
    """Return default PEFT target modules for PI0.5 fine-tuning."""
    common_projections = (
        "state_proj|action_in_proj|action_out_proj|action_time_mlp_in|action_time_mlp_out"
    )
    target_modules = rf"(.*\.gemma_expert\..*\.self_attn\.(q|v)_proj|model\.({common_projections}))"
    return {
        "target_modules": target_modules,
        "modules_to_save": [],
    }
```

**含义**：仅对以下模块注入 LoRA：
* **Gemma Action Expert 的 Q/V 投影**: `.*\.gemma_expert\..*\.self_attn\.(q|v)_proj`（注意：**不包括 K 投影**）
* **动作和状态投影层**: `state_proj`,   `action_in_proj`,   `action_out_proj`,   `action_time_mlp_in`,   `action_time_mlp_out`

**VLM 主干（PaliGemma）不动**——这是轻量级微调的关键。

### LeRobot PeftConfig 字段说明

源码 `configs/default.py:84-108` :

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `target_modules` | `list[str] \| str \| None` | None | LoRA 目标模块名称或正则，若为 None 则使用 policy 默认 |
| `full_training_modules` | `list[str] \| None` | None | 完整训练的模块（不用 LoRA），内部映射为 PEFT 的 `modules_to_save` |
| `method_type` | `str` | `"LORA"` | PEFT 方法类型，支持 "LORA", "MISS" 等 |
| `init_type` | `str \| None` | None | 初始化方式（映射为 PEFT 的 `init_lora_weights` ） |
| `r` | `int` | 16 | LoRA 秩（rank），控制参数量与适应能力的平衡 |

**重要**： `lora_alpha` 不在 LeRobot 的 `PeftConfig` 中暴露，默认使用 PEFT 库的值（通常 `lora_alpha = 8` ）。如需自定义，需直接创建 `peft.LoraConfig` 并传给 `wrap_with_peft()` 。

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

这个顺序决定了 `observation.state` 和 `action` 向量 (6,) 的维度语义。

**常见踩坑**：标定时电机 ID 和代码不匹配，导致采集到的 state 向量里每个维度的物理含义都是错的。

同样，摄像头 key（`observation.images.front`、`observation.images.wrist`）必须在录制和推理时保持一致。如果录制时摄像头叫 `cam_0`/`cam_1`，推理时也必须用同样的名字。

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

训练时 `make_policy()` 调用 `dataset_to_policy_features(ds_meta.features)`（`factory.py:458`），直接从数据集的 `meta/info.json` 读取所有特征定义，按键名前缀自动分类：

- `observation.images.*` → `FeatureType.VISUAL`
- `observation.state` → `FeatureType.STATE`
- `action` → `FeatureType.ACTION`

**不需要手动映射摄像头**。无论你的数据集摄像头叫 `front`/`wrist` 还是 `cam_0`/`cam_1`，模型只看特征类型和 shape，不看名字。基座模型 `lerobot/pi05_base` 在异构多机器人数据上预训练，不绑定特定摄像头名称或数量。

#### `empty_cameras`（极少用到）

数据集只有一个摄像头时，可补充空白占位：

```bash
--policy.empty_cameras=1
```

#### `rename_map`（极少用到）

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

脚本执行流程（`augment_dataset_quantile_stats.py`）：
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
* **PEFT 注入**: `src/lerobot/scripts/lerobot_train.py:246-250`

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

### 方案 A：LoRA 微调（推荐，显存最低）

```bash
# Acknowledge license and Access repository in https://huggingface.co/google/paligemma-3b-pt-224
# 用时间戳区分每次训练，无需手动清理目录
RUN_DIR="/root/autodl-tmp/outputs/pi05_lora_so101/$(date +%Y%m%d_%H%M%S)"

lerobot-train \
    --dataset.repo_id=Atticuxz/so101-table-cleanup \
    --policy.type=pi05 \
    --policy.pretrained_path=lerobot/pi05_base \
    --policy.dtype=bfloat16 \
    --policy.gradient_checkpointing=true \
    --policy.compile_model=true \
    --policy.device=cuda \
    --batch_size=4 \
    --steps=5000 \
    --save_freq=1000 \
    --log_freq=50 \
    --eval_freq=0 \
    --output_dir="$RUN_DIR" \
    --job_name=pi05_lora_so101 \
    --peft.method_type=LORA \
    --peft.r=16 \
    --policy.push_to_hub=false \
    --wandb.enable=true \
    --wandb.project=pi05_so101
```

> 每次训练结果落在独立子目录，如 `/root/autodl-tmp/outputs/pi05_lora_so101/20260323_143022/`，便于多次实验对比。

#### 方案 A-2：YAML 配置文件启动（推荐）

上述 CLI 参数已整理为 YAML 配置文件，修改参数更方便：

```bash
# 直接用 YAML 启动（所有参数在文件中）
lerobot-train --yaml_config=experiments/pi05_lora_so101_table_cleanup.yaml

# CLI 参数覆盖 YAML（用于临时调参）
lerobot-train --yaml_config=experiments/pi05_lora_so101_table_cleanup.yaml --steps=8000 --batch_size=8

# 切换数据集也只需覆盖一个参数
lerobot-train --yaml_config=experiments/pi05_lora_so101_table_cleanup.yaml --dataset.repo_id=Atticuxz/so101-new-task
```

> YAML 配置文件位于 `experiments/` 目录下，每个实验一个文件，便于版本管理和对比。

> ⚠️ **必须加 `--policy.push_to_hub=false` **
>
> `PreTrainedConfig.push_to_hub` 默认为 `True` （ `configs/policies.py:70` ），如果不设置此参数且没有指定 `--policy.repo_id` ，训练会立即报错：
> ```bash
> ValueError: 'policy.repo_id' argument missing. Please specify it to push the model to the hub.
> ```

#### 参数详解

| 参数 | 值 | 说明 |
|------|-----|------|
| `--dataset.repo_id` | `your_hf_username/so101_pickplace` | HF Hub 数据集 ID |
| `--policy.type` | `pi05` | 使用 Pi0.5 策略 |
| `--policy.pretrained_path` | `lerobot/pi05_base` | 预训练权重，可选： `lerobot/pi05_libero` |
| `--policy.dtype` | `bfloat16` | 混合精度，降低显存和计算量 |
| `--policy.gradient_checkpointing` | `true` | 启用梯度检查点，显存减半，速度略降 |
| `--policy.device` | `cuda` | 推理设备 |
| `--policy.push_to_hub` | `false` | **必须显式关闭**，否则报错（默认为 true） |
| `--policy.compile_model` | `false` （默认） | 开启后首次编译慢 5-10 min，之后每 step 快 ~20%，steps 少时不划算 |
| `--batch_size` | 4 | 单卡保守值，根据 GPU 显存调整 |
| `--steps` | 见下方数据量指导 | 总训练步数 |
| `--save_freq` | 1000 | 每 1000 步保存一次检查点 |
| `--log_freq` | 50 | 每 50 步记录一次日志 |
| `--eval_freq` | 0 | 无仿真环境，关闭评估 |
| `--peft.method_type` | `LORA` | PEFT 方法，选项： `LORA` 、 `MISS` 等 |
| `--peft.r` | 16 | LoRA 秩（rank），可调范围 8-64 |
| `--peft.target_modules` | *不设* | 使用 Pi0.5 默认值（Action Expert Q/V + 投影层） |

#### 根据数据量确定 Steps

```bash
# 查看数据集信息
python -c "
from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata
m = LeRobotDatasetMetadata('youliangtan/so101-table-cleanup')
print('episodes:', m.total_episodes)
print('frames:', m.total_frames)
print('建议 steps (3 epoch):', m.total_frames // 4 * 3)
"
```

| 数据规模 | episodes | 约 frames | 建议 steps | 说明 |
|---------|---------|---------|---------|------|
| 极少 | < 30 | < 3000 | 2000~3000 | 数据少，防过拟合 |
| 少 | 30~100 | 3000~10000 | 3000~8000 | 典型小规模采集 |
| 中 | 100~300 | 10000~30000 | 10000~20000 | 社区参考量级 |
| 大 | > 300 | > 30000 | 30000~50000 | 参考命令量级 |

**80 episodes 推荐**：约 8000 frames，batch_size=4 下约 2000 steps/epoch， `--steps=8000` ≈ 4 epoch。

#### LoRA 目标模块的逐步处理

1. **PEFT 配置阶段** (`lerobot_train.py:249`)：
   

```python
   peft_cli_overrides = dataclasses.asdict(cfg.peft)  # {'method_type': 'LORA', 'r': 16, ...}
   ```

2. **构建最终 PEFT 配置** (`pretrained.py:381-406`)：
   

```python
   def _build_peft_config(self, cli_overrides: dict):
       # 1. 获取 policy 默认目标模块正则
       config_dict = dict(self._get_default_peft_targets() or {})
       # 2. 合并 CLI 覆盖
       for key, value in cli_overrides.items():
           if value is not None:
               config_dict[key] = value
       # 3. 创建 PEFT LoraConfig
       return LoraConfig(**config_dict)
   ```

3. **应用 PEFT** (`pretrained.py:301-310`)：
   

```python
   # 冻结所有参数
   for p in self.parameters():
       p.requires_grad_(False)
   # 用 PEFT 包装，自动 unfroze LoRA 参数
   peft_model = get_peft_model(self, final_config)
   ```

### `train_expert_only` 与 `freeze_vision_encoder` 详解

源码在 `modeling_pi05.py:420-428` 的 `_set_requires_grad()` 方法：

```python
def _set_requires_grad(self):
    if self.freeze_vision_encoder:   # 仅冻结视觉塔
        self.paligemma.model.vision_tower.eval()
        for param in self.paligemma.model.vision_tower.parameters():
            param.requires_grad = False
    if self.train_expert_only:       # 冻结整个 PaliGemma，只训练 expert
        self.paligemma.eval()
        for param in self.paligemma.parameters():
            param.requires_grad = False
```

Pi0.5 由两部分组成：

```
Pi0.5
├── PaliGemma VLM (~2.7B 参数)        ← train_expert_only=true 时冻结
│   ├── Vision Tower (视觉编码器)       ← freeze_vision_encoder=true 时冻结
│   └── Gemma Language Model
└── Gemma Action Expert (~300M 参数)  ← 始终参与训练
    ├── action_in_proj / action_out_proj
    ├── time_mlp_in / time_mlp_out
    └── self_attn (Q/V/K proj)
```

| 参数 | 冻结范围 | 训练范围 | 显存影响 | 适用场景 |
|------|---------|---------|---------|---------|
| 均为 `false` （默认） | 无 | 全部参数 | 最高 | 全参微调，数据量大 |
| `freeze_vision_encoder=true` | 视觉塔 | VLM 文本 + Expert | 中 | 图像特征稳定无需调整 |
| `train_expert_only=true` | 整个 PaliGemma | 仅 Action Expert | 低 ~30% | 显存紧张，快速适配 |

> ⚠️ **不要将 `train_expert_only=true` 与 `--peft.*` 混用**
>
> LoRA（ `--peft.r=16` ）的 `wrap_with_peft()` 会先冻结**所有**参数再注入适配器。
> `train_expert_only` 的手动冻结发生在 PEFT 包装**之前**，两者并行只会让逻辑更复杂且不可预测。
> **选一种方案即可**：
> - 显存够 → `train_expert_only=true` （全参但仅 expert，约 5-10% 参数量）
> - 显存紧 → `--peft.r=16` （LoRA，约 0.1-1% 参数量）

### 方案 B：冻结 VLM，全参训练 Action Expert

```bash
lerobot-train \
    --dataset.repo_id=your_hf_username/so101_pickplace \
    --policy.type=pi05 \
    --policy.pretrained_path=lerobot/pi05_base \
    --policy.dtype=bfloat16 \
    --policy.gradient_checkpointing=true \
    --policy.train_expert_only=true \
    --policy.push_to_hub=false \
    --policy.device=cuda \
    --batch_size=4 \
    --steps=8000 \
    --save_freq=2000 \
    --log_freq=50 \
    --eval_freq=0 \
    --output_dir=./outputs/pi05_expert_so101 \
    --job_name=pi05_expert_so101 \
    --wandb.enable=true \
    --wandb.project=pi05_so101
```

### 方案对比总览

| 方案 | 冻结范围 | 训练参数量 | 推荐场景 |
|------|--------|-----------|---------|
| **LoRA r=16**（方案 A） | 全部 → 仅 LoRA 适配器可训 | ~0.1-1% | 显存最紧张，快速实验 |
| **train_expert_only**（方案 B） | PaliGemma VLM | ~5-10%（Expert 全参） | 显存够，适应能力强 |
| **全参微调**（参考命令风格） | 无 | 100% | 数据多（>200 ep），显存大 |

### 社区参考命令对比（全参微调 vs LoRA）

社区中常见的全参微调写法（来自 TommyZihao/shake_hands 案例）：

```bash
# 全参微调（无 --peft.*）
lerobot-train \
    --dataset.repo_id=TommyZihao/lerobot_zihao_dataset_shake_hands \
    --policy.type=pi05 \
    --policy.pretrained_path=lerobot/pi05_base \
    --policy.compile_model=true \     # 编译加速，首次慢
    --policy.gradient_checkpointing=true \
    --policy.dtype=bfloat16 \
    --policy.freeze_vision_encoder=false \   # 显式写出默认值
    --policy.train_expert_only=false \       # 显式写出默认值
    --policy.push_to_hub=false \
    --steps=50000 \                          # 数据量大
    --batch_size=8 \                         # 显存充足（≥40GB）
    --wandb.enable=true
```

**与 LoRA 方案的本质差异**：

| | 全参微调（参考命令） | LoRA 微调（本文方案 A） |
|--|--|--|
| 参数更新量 | 100% | 0.1-1% |
| 显存需求 | ~35-45 GB | ~14-22 GB |
| 适合 steps | 20000-50000 | 3000-10000 |
| 适合数据量 | > 200 episodes | < 100 episodes |
| 检查点大小 | 完整权重（~15GB） | 仅适配器（~100MB） |

两种写法都正确，根据你的显存和数据量选择。

### 数据集摄像头键不匹配时

如果你的数据集用了 `laptop` 和 `phone` 作为摄像头键，但训练时想映射到 `front` 和 `wrist` ：

```bash
--rename_map='{
  "observation.images.laptop": "observation.images.front",
  "observation.images.phone": "observation.images.wrist"
}'
```

---

## 4. Async gRPC 分离式部署方案

### 源码定位

* **策略服务器**: `src/lerobot/async_inference/policy_server.py`
* **机器人客户端**: `src/lerobot/async_inference/robot_client.py`
* **配置**: `src/lerobot/async_inference/configs.py`
* **常量**: `src/lerobot/async_inference/constants.py:25-30`

### 支持列表

```python
SUPPORTED_POLICIES = ["act", "smolvla", "diffusion", "tdmpc", "vqbet", "pi0", "pi05", "groot"]
SUPPORTED_ROBOTS = ["so100_follower", "so101_follower", "bi_so_follower", "omx_follower"]
```

✓ Pi0.5 ( `pi05` ) 在支持列表中
✓ SO-ARM101 ( `so101_follower` ) 在支持列表中

### 架构原理

Async inference 将**策略推理**和**动作执行**解耦：

```
同步推理 (Sync)：
  Robot: Wait → Inference → Execute → Wait → ...
  CPU/GPU:      Computing    Idle    Computing

异步推理 (Async)：
  Robot:      Execute → Execute → Execute → Execute
  CPU/GPU: Computing → Computing → Computing → Computing
           (always busy, next chunk precomputed)
```

Async 消除了"等待推理"的空闲帧，使机械臂能更平稳、更快地响应。

### 步骤 1：启动 Policy Server（GPU 机器）

```bash
python -m lerobot.async_inference.policy_server \
    --host=0.0.0.0 \
    --port=8080 \
    --fps=30 \
    --inference_latency=0.033 \
    --obs_queue_timeout=2
```

#### 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--host` | `localhost` | 服务器监听地址， `0.0.0.0` 接受所有网络连接 |
| `--port` | 8080 | gRPC 监听端口 |
| `--fps` | 30 | 服务器运行频率（Hz），关系到推理期望延迟 |
| `--inference_latency` | `1/fps` | 推理允许延迟（秒），通常 = 1/fps |
| `--obs_queue_timeout` | 2 | 等待观测超时（秒） |

**启动后**，服务器处于空闲状态。所有关于策略、设备、超参的信息由客户端首次握手时传送。

### 步骤 2：启动 Robot Client（连接 SO-ARM101）

#### 使用 LoRA 微调后的本地检查点

```bash
python -m lerobot.async_inference.robot_client \
    --server_address=<GPU_MACHINE_IP>:8080 \
    --robot.type=so101_follower \
    --robot.port=/dev/ttyUSB0 \
    --robot.id=my_so101 \
    --robot.cameras='{"front": {"type": "opencv", "index_or_path": 0, "width": 640, "height": 480, "fps": 30}, "wrist": {"type": "opencv", "index_or_path": 1, "width": 640, "height": 480, "fps": 30}}' \
    --task="pick up the cube" \
    --policy_type=pi05 \
    --pretrained_name_or_path=./outputs/pi05_lora_so101/checkpoints/last/pretrained_model \
    --policy_device=cuda \
    --client_device=cpu \
    --actions_per_chunk=50 \
    --chunk_size_threshold=0.5 \
    --aggregate_fn_name=weighted_average \
    --fps=30
```

#### 使用 HF Hub 预训练模型

```bash
python -m lerobot.async_inference.robot_client \
    --server_address=<GPU_MACHINE_IP>:8080 \
    --robot.type=so101_follower \
    --robot.port=/dev/ttyUSB0 \
    --robot.id=my_so101 \
    --robot.cameras='{"front": {"type": "opencv", "index_or_path": 0, "width": 640, "height": 480, "fps": 30}}' \
    --task="pick up the object" \
    --policy_type=pi05 \
    --pretrained_name_or_path=lerobot/pi05_base \
    --policy_device=cuda \
    --client_device=cpu \
    --actions_per_chunk=50 \
    --chunk_size_threshold=0.5
```

#### 关键参数详解

| 参数 | 值 | 说明 |
|------|-----|------|
| `--server_address` | `<IP>:8080` | gRPC 服务器地址，本机用 `127.0.0.1:8080` ，远程用实际 IP |
| `--robot.type` | `so101_follower` | 机器人类型 |
| `--robot.port` | `/dev/ttyUSB0` | 串口设备路径（Linux/Mac）或 COM 端口（Windows） |
| `--robot.id` | `my_so101` | 机器人唯一标识，用于加载校准文件 |
| `--policy_type` | `pi05` | 使用 Pi0.5 策略 |
| `--pretrained_name_or_path` | 本地路径或 HF repo | 检查点位置 |
| `--policy_device` | `cuda` | 服务器推理设备（通常 GPU） |
| `--client_device` | `cpu` | 客户端接收动作后移动到的设备 |
| `--actions_per_chunk` | 50 | **每次推理输出多少动作步**，Pi0.5 chunk_size=50，建议相等或更小 |
| `--chunk_size_threshold` | 0.5 | **队列低于 50% 时发送新观测**，范围 [0, 1]，越小越接近同步 |
| `--aggregate_fn_name` | `weighted_average` | 重叠动作块的聚合策略，可选： `weighted_average` （0.3×old + 0.7×new）、 `latest_only` 、 `average` 、 `conservative` |
| `--fps` | 30 | 客户端控制循环频率 |

### 执行流程示意

```
┌─────────────────────────────────────────────────────┐
│ RobotClient (SO-ARM101 机器)                         │
│ ┌──────────────────────┐                            │
│ │ main: control_loop() │                            │
│ │ (30 Hz)              │                            │
│ └──────────────────────┘                            │
│   │                                                  │
│   ├─ 握手: Ready() ───────────────────────────────→ │
│   │                    ┌─ PolicyServer (GPU 机器) │
│   ├─ SendPolicyInstructions(pi05_checkpoint)───→ │ 加载模型到 CUDA
│   │                                                 │
│   ├─ [background] receive_actions() ↓              │ 持续监听推理结果
│   │                                                 │
│   ├─ 采集观测: robot.get_observation()             │
│   │   state: (6,) 关节位置                         │
│   │   images: {front: (480,640,3), wrist: ...}    │
│   │                                                 │
│   ├─ SendObservations(obs) ──────────────────────→ │ 队列观测
│   │                                                 │
│   │ ↓ queue size < 50%?                            │
│   ├─ GetActions(empty) ────────────────────────→  │ 请求推理
│   │              ↓                                  │
│   │         preprocess obs                         │
│   │         run model.sample_actions() (10 steps)  │
│   │         postprocess actions                    │
│   │          ↓                                      │
│   │ ←─────── [TimedAction] ←──────────────────── │
│   │                                                 │
│   ├─ receive_actions() 后台接收:                    │
│   │   • 解序列化 action chunk (shape: 50×6)       │
│   │   • 聚合重叠部分 (weighted_average)           │
│   │   • 入队 action_queue                         │
│   │                                                 │
│   ├─ pop action from queue                         │
│   ├─ robot.send_action(action[6])                  │
│   │   → SO-ARM101 执行关节目标位置               │
│   │                                                 │
│   └─ sleep(1/30) → 循环                           │
│                                                     │
└─────────────────────────────────────────────────────┘
```

### 同机部署（GPU 和 SO-ARM101 控制器在同一台机器）

如果所有组件都在同一台机器上：

```bash
# Terminal 1: 启动 Policy Server
python -m lerobot.async_inference.policy_server \
    --host=127.0.0.1 \
    --port=8080

# Terminal 2: 启动 Robot Client
python -m lerobot.async_inference.robot_client \
    --server_address=127.0.0.1:8080 \
    --policy_type=pi05 \
    --pretrained_name_or_path=./outputs/pi05_lora_so101/checkpoints/last/pretrained_model \
    ...
```

### 参数调优建议

源码 `async_inference/constants.py` 与文档中的建议：

#### 1. 根据推理延迟调整 FPS

如果 Pi0.5 的推理延迟是 1 秒（在你的 GPU 上），而你设置 `--fps=30` （每帧 33ms），客户端会很快耗尽动作队列。

**解决**：降低 `--fps` 或增加 `--actions_per_chunk` 。

```bash
# 推理延迟 ~1s，行动需时 ~2.5s (50 actions @ 20 Hz)
--fps=20  # 每帧 50ms，更宽松
```

#### 2. 调整 `chunk_size_threshold`

* **0.0**: 完全同步（每动作步推理一次，延迟大）
* **0.5**: 平衡（推荐）
* **0.9**: 激进（频繁通信，依赖良好的网络和推理速度）

**可视化调试**：

```bash
--debug_visualize_queue_size=true
```

会在程序退出时显示动作队列大小演变图，帮助你找到最优的 `chunk_size_threshold` 。

#### 3. `aggregate_fn_name` 选择

| 函数 | 公式 | 特点 |
|------|------|------|
| `weighted_average` | 0.3×old + 0.7×new | **推荐**，偏向最新预测 |
| `latest_only` | new | 激进，可能快速变化 |
| `average` | 0.5×old + 0.5×new | 平衡 |
| `conservative` | 0.7×old + 0.3×new | 保守，平滑但滞后 |

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
│ Step 3: 单卡 LoRA 微调                                   │
│ 命令: lerobot-train \                                    │
│         --dataset.repo_id=your_username/so101_pickplace\ │
│         --policy.type=pi05 \                             │
│         --policy.pretrained_path=lerobot/pi05_base \     │
│         --peft.r=16 \                                    │
│         --batch_size=4 \                                 │
│         --steps=5000 \                                   │
│         --policy.dtype=bfloat16 \                        │
│         ...                                              │
│ 输出: Checkpoint (outputs/pi05_lora_so101/...)           │
│ 时长: ~1-3 小时 (取决于 GPU 和数据量)                     │
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
│ 命令: hf upload Atticuxz/pi05_so101_lora \               │
│         outputs/pi05_lora_so101/ \                       │
│         --repo-type model                                │
│ 或直接在 lerobot-train 命令中设置:                        │
│ --policy.push_to_hub=true \                             │
│ --policy.repo_id=your_repo_id                           │
└─────────────────────────────────────────────────────────┘
```

### 目录结构示意

```
outputs/
├── pi05_lora_so101/
│   ├── train_config.json              # 训练配置快照
│   ├── training_state/
│   │   ├── optimizer_state.safetensors
│   │   ├── scheduler_state.json
│   │   └── ...
│   └── checkpoints/
│       ├── 1000/
│       │   └── pretrained_model/
│       │       ├── config.json        # Pi0.5 策略配置
│       │       ├── model.safetensors  # LoRA 权重 (adapter_config.json + adapter_model.safetensors)
│       │       ├── processor.json
│       │       └── ...
│       ├── 2000/
│       └── last/                      # 最新检查点，部署时用这个
│           └── pretrained_model/
│               └── ...
```

**关键**：部署时使用 `checkpoints/last/pretrained_model/` ，它包含最终训练的 LoRA 适配器权重。

---

## 6. 单卡显存估算与建议

### 显存使用情况

| 配置 | 显存需求 | 适合 GPU | 备注 |
|------|----------|---------|------|
| LoRA r=16 + bf16 + gradient_checkpointing + bs=2 | ~14-16 GB | RTX 4090, A5000 | 最紧凑 |
| LoRA r=16 + bf16 + gradient_checkpointing + bs=4 | ~18-22 GB | A5000, A100-40G | 推荐 |
| LoRA r=32 + bf16 + gradient_checkpointing + bs=4 | ~24-28 GB | A100-40G | 更强适应 |
| train_expert_only + bf16 + gradient_checkpointing + bs=4 | ~22-26 GB | A100-40G | 全参但仅 expert |
| 全参微调 + bf16 + gradient_checkpointing + bs=4 | ~35-45 GB | A100-80G, H100 | 资源密集 |

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

1. **Pi0.5 + LoRA** 的集成流程：
   - LeRobot 通过 `wrap_with_peft()` 在训练脚本中外部包装 PEFT
   - Pi0.5 默认目标模块：Gemma Action Expert 的 Q/V 投影 + 投影层
   - VLM 主干（PaliGemma）保持冻结，仅适配器参数更新

2. **SO-ARM101 特性**：
   - 6-DoF + 1-Gripper（共 6 维度）
   - Pi0.5 自动 pad 到 32 维
   - 支持多摄像头（典型双目配置）
   - 无仿真环境，仅离线 loss 验证

3. **训练优化**：
   - 使用 bfloat16 + gradient_checkpointing 最高效
   - LoRA r=16 在显存和适应能力间达到平衡
   - 单卡 16-24GB 显存可行

4. **部署架构**：
   - Async gRPC 解耦推理和执行
   - PolicyServer 运行在 GPU，RobotClient 连接 SO-ARM101
   - 30 Hz 控制频率、50 步 chunk、0.5 阈值为推荐配置

### 文档引用

* [LeRobot Pi0.5 文档](./pi05.mdx)
* [LeRobot Async 推理文档](./async.mdx)
* [LeRobot 安装指南](./installation)
* [LoRA 基础概念](./what_is_lora.md)

---

## 附录 A：训练结果验证

### 为什么 SO-ARM101 没有内置验证指标？

LeRobot 的 `lerobot-eval` 依赖仿真环境（`gym.vector.VectorEnv`）来执行 rollouts 并计算 reward 和 success rate。但 `envs/` 下没有 SO-ARM101 对应的仿真器，因此：

- `eval_freq=0`（关闭训练中评估）
- 训练全程**只记录 Training Loss**，没有 validation loss
- 推理验证只能通过 **离线脚本** 或 **实机测试**

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

训练时：`t` 越接近 1（噪声端），模型学习去噪；`t` 越接近 0（动作端），模型学习重建动作。

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
- 数据集规模很小（< 500 episodes）但训练 steps 很多（> 10000）
- `loss_per_dim` 中某一维度持续远高于其他维度
- 实机测试时动作剧烈抖动或超出关节限位

**缓解方法**：
- 降低 `steps`（如从 10000 降到 5000）
- 增加 `lora_dropout`（PEFT config 中的 dropout）
- 使用更大的 `r` 值（提升 adapter 容量同时防止欠拟合）

### 离线 Action Plotting（Checkpoint 横向对比）

由于没有仿真环境，可通过**离线推理对比**来验证不同 checkpoint 的质量。

**核心思路**：加载数据集的 batch，用 checkpoint 推理预测动作，和 ground-truth 动作画在同一张图上。

**待实现脚本**：`scripts/eval_offline_action_plot.py`

预期功能：
```python
# 伪代码
for checkpoint_dir in [checkpoint_1000, checkpoint_3000, checkpoint_5000]:
    policy = Pi05Policy.from_pretrained(checkpoint_dir)
    for batch in dataset:
        with torch.no_grad():
            pred_actions = policy.select_action(batch)  # (B, chunk_size, action_dim)
        gt_actions = batch["action"]                    # (B, chunk_size, action_dim)

        # 按时间维度画线图，每个 action 维度一条线
        plot_trajectories(pred_actions[0], gt_actions[0], step=checkpoint_step)

    # 保存对比图到 outputs/checkpoint_eval/
```

**判断标准**：
- 预测动作和 GT 动作轨迹**形状接近** → checkpoint 质量好
- 预测动作在某些维度上**持续偏移**（固定偏差） → 可能是 action projection 层未充分适配
- 预测动作**抖动剧烈** → 过拟合或 LoRA r 值太小

### Checkpoint 选择流程总结

```
训练完成
  ↓
查看 wandb loss 曲线
  ↓
loss 正常收敛?
  ├── 否 → 检查数据、配置、重训
  └── 是 → 进入下一步
  ↓
选出 2-3 个关键 checkpoint（如 2000步 / 5000步 / 末步）
  ↓
运行离线 Action Plotting 对比
  ↓
哪个 checkpoint 的预测轨迹最接近 GT?
  ↓
选出最佳 checkpoint 用于实机部署
  ↓
实机测试（gRPC inference）
  ↓
验证动作是否平滑、合理
```

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

**本文档最后更新于 2026-03-23，基于 LeRobot commit f90db58c**

**更新记录**：
* v1.0：初始版本，覆盖 LoRA 配置、数据集对齐、训练、gRPC 部署
* v1.1：补充 `--policy.push_to_hub=false` 必须项、`train_expert_only` 与 LoRA 冲突警告、按数据量计算 steps 指南、社区全参微调命令对比
* v1.2：新增 Section 1b「数据集格式转换与上传（v2.1 → v3.0）」，流程图补充 Step 1b/1c
* v1.3：新增 YAML 配置文件启动方式（`--yaml_config`）、方案 A-2 文档；`experiments/` 目录放置实验配置
* v1.4：策略切换至 `train_expert_only=true`（方案 B），暂不使用 LoRA；新增 `experiments/pi05_expert_so101_table_cleanup.yaml`
