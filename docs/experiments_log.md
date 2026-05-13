# 实验记录 (Experiments Log)

## 2026-05-13

### 实验 #3 — XVLa SO101 全量微调 bs24

| 字段 | 内容 |
|------|------|
| **WandB** | [链接](https://wandb.ai/atticux/xvla_so101/runs/ue4j5esj) |
| **模型权重** | [链接](https://huggingface.co/Atticuxz/xvla_so101_20260513_0911) |
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
| **WandB** | [链接](https://wandb.ai/atticux/xvla_so101/runs/2i9akaxt) |
| **模型权重** | [链接](https://huggingface.co/Atticuxz/xvla_so101_20260513_0905) |
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
| **WandB** | [链接](https://wandb.ai/atticux/xvla_so101/runs/4j0w9t5b) |
| **模型权重** | [链接](https://huggingface.co/Atticuxz/xvla_so101_20260513_0838) |
| **模型** | XVLa 0.9B |
| **微调方式** | 全量微调 |
| **配置** | Batch Size 16, 2000 steps |
| **显存占用** | ~22GB |
| **数据集** | 同门自采红色方块 60 回合 |
| **数据集描述** | 无回程；起点姿态单一；夹完后空闲帧较长 |
| **训练时长** | ~11 分钟 |
| **训练结果** | loss 在 1000 step 后降至 ~0.04 |
| **验证效果** | — |
