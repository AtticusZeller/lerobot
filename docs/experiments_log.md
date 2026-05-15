# 实验记录 (Experiments Log)

## 2026-05-15

### 实验 #4 — SmolVLA SO101 冻结VLM 数据增强

| 字段 | 内容 |
|------|------|
| **WandB** | [5tma3wrj](https://wandb.ai/atticux/smolvla_so101/runs/5tma3wrj) |
| **模型权重** | [Atticuxz/smolvla_so101_20260515_0913](https://huggingface.co/Atticuxz/smolvla_so101_20260515_0913) |
| **模型** | SmolVLA lerobot/smolvla_base 0.5B |
| **微调方式** | 冻结 VLM，仅微调 action head |
| **配置** | 20k steps，启用数据增强，n_obs_steps=4 |
| **数据集描述** | 节奏比之前更慢，视角更清晰，边界内随机摆放，有回程，手动结束录制，idle 帧时间比之前短 |
| **训练时长** | ~3 小时 |
| **训练结果** | loss 稳定在 0.05-0.1 |
| **验证效果** | — |
| **失败原因分析** | #1-#3 主因：预训练数据集中无 SO-ARM101；次因：对模型结构不够了解无法定位错误；训练步长可能偏短（损失稳定后需再跑 1-20k step 再取 checkpoint） |

### 实验配置变更 — SmolVLA 数据增强与时间平滑性测试

| 字段 | 内容 |
|------|------|
| **变更模块** | SmolVLA SO101 训练配置 |
| **变更文件** | `experiments/smolvla_so101_table_cleanup.yaml` |
| **详细配置** | `dataset.image_transforms.enable=true`, `dataset.image_transforms.random_order=true`, `policy.n_obs_steps=4` |
| **预期效果** | 引入数据增强提高鲁棒性，设置 `n_obs_steps=4` 增强观测时间维度的平滑性。 |

## 2026-05-13

### 实验 #3 — XVLa SO101 全量微调 bs24

| 字段 | 内容 |
|------|------|
| **WandB** | [ue4j5esj](https://wandb.ai/atticux/xvla_so101/runs/ue4j5esj) |
| **模型权重** | [Atticuxz/xvla_so101_20260513_0911](https://huggingface.co/Atticuxz/xvla_so101_20260513_0911) |
| **模型** | XVLa 0.9B |
| **微调方式** | 全量微调 |
| **配置** | Batch Size 24, 2000 steps |
| **显存占用** | ~32GB |
| **数据集** | 同门自采红色方块 60 回合 |
| **数据集描述** | 无回程；起点姿态单一；夹完后空闲帧较长 |
| **训练时长** | ~12 分钟 |
| **训练结果** | loss 稳定在 0.03-0.05 |
| **验证效果** | — |

### 实验 #2 — XVLa SO101 冻结VLM 微调

| 字段 | 内容 |
|------|------|
| **WandB** | [2i9akaxt](https://wandb.ai/atticux/xvla_so101/runs/2i9akaxt) |
| **模型权重** | [Atticuxz/xvla_so101_20260513_0905](https://huggingface.co/Atticuxz/xvla_so101_20260513_0905) |
| **模型** | XVLa 0.9B |
| **微调方式** | 冻结 VLM，仅微调 action head |
| **配置** | Batch Size 32, 2000 steps |
| **数据集** | [Atticuxz/so101-table-cleanup](https://huggingface.co/datasets/Atticuxz/so101-table-cleanup) |
| **数据集描述** | 无回程；起点姿态单一；夹完后空闲帧较长 |
| **训练时长** | ~10 分钟 |
| **训练结果** | loss 在 2000 step 收敛至 ~0.03 |
| **验证效果** | — |

### 实验 #1 — XVLa SO101 红色方块抓取

| 字段 | 内容 |
|------|------|
| **WandB** | [4j0w9t5b](https://wandb.ai/atticux/xvla_so101/runs/4j0w9t5b) |
| **模型权重** | [Atticuxz/xvla_so101_20260513_0838](https://huggingface.co/Atticuxz/xvla_so101_20260513_0838) |
| **模型** | XVLa 0.9B |
| **微调方式** | 全量微调 |
| **配置** | Batch Size 16, 2000 steps |
| **显存占用** | ~22GB |
| **数据集** | 同门自采红色方块 60 回合 |
| **数据集描述** | 无回程；起点姿态单一；夹完后空闲帧较长 |
| **训练时长** | ~11 分钟 |
| **训练结果** | loss 在 1000 step 后降至 ~0.04 |
| **验证效果** | — |
