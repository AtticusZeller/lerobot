# LoRA 微调基础概念

## 什么是 LoRA？

**LoRA** (Low-Rank Adaptation，低秩适配) 是一种**参数高效微调** (PEFT, Parameter-Efficient Fine-Tuning) 方法。它的核心思想是：大规模预训练模型的权重变化量是低秩的，因此可以用两个小矩阵来近似。

### 全量微调 vs LoRA

假设一个模型的权重矩阵 **W** 形状为 `[4096 × 4096]`（约 1600 万参数）：

| 方式 | 更新内容 | 参数量 | 显存占用 |
|------|----------|--------|----------|
| 全量微调 | 整个 W 都要更新 | 16,777,216 | 巨大（梯度+优化器+动量） |
| LoRA | 只更新 A × B | 4096×r + r×4096 | 极小 |

```
全量微调：  y = (W + ΔW) · x     ← 整个 W + ΔW 都要存和更新
LoRA：      y = W · x + (A · B) · x  ← W 冻住，只训练 A 和 B
```

其中 `A ∈ ℝ^{d×r}`，`B ∈ ℝ^{r×d}`，`r` 称为 **rank**（秩），是 LoRA 最核心的超参数。r 越小，训练参数越少；r 越大，表达力越强。常用范围是 8~64。

---

## LoRA 的三大优势

### 1. 防止灾难性遗忘 (Catastrophic Forgetting)

因为原始权重 **W 完全冻结**，一个参数都没改。LoRA 只在模型旁边"附加"了一个小修正项 `(A·B)·x`。预训练学到的知识（视觉理解、语言理解、通用运动模式）完整保留，只让 adapter 来适应新任务。

### 2. 训练负担极小

只有 A 和 B 矩阵需要：
- 计算梯度
- 存储优化器状态（如 Adam 的动量）
- 存储梯度

典型 Pi0.5 全量微调需要 ~12GB 显存，LoRA 微调可能只需要 ~4GB。这让**单卡甚至消费级 GPU** 训练成为可能。

### 3. 可组合、可插拔、可共享

- **只存 adapter**：训练结果只保存 A 和 B 的权重，几十 MB。原始几 GB 模型不需要重复存储。
- **随意切换**：同一份预训练模型可以叠加不同的 adapter 来适配不同机器人。
- **便于分享**：只需要分享 adapter 文件（而不是整个模型），方便社区共享。

---

## 关键超参数

| 参数 | 含义 | 常用值 |
|------|------|--------|
| `r` (rank) | A/B 矩阵的中间维度，决定 adapter 的表达能力 | 8, 16, 32, 64 |
| `lora_alpha` | 缩放系数，实际缩放为 `alpha/r`，控制修正力度 | 通常设为 `2r` |
| `lora_dropout` | LoRA 路径上的 Dropout 概率，防止过拟合 | 0.0 ~ 0.1 |
| `target_modules` | 把 LoRA 注入到哪些层（哪些权重矩阵旁边加 A 和 B） | 需根据模型结构指定 |

---

## LeRobot 如何注入 LoRA 到 Pi0.5

### 注入的模块

Pi0.5 的 LoRA 只注入到两类关键模块：

**1. Gemma Expert 的注意力层（Q 和 V 投影）**

```
target_modules:  .*\.gemma_expert\..*\.self_attn\.(q|v)_proj
```

这是 LoRA 论文的经典做法——attention 层中负责"查询 (Query)"和"值 (Value)"的线性层对任务适应最敏感、最有效。

**2. 动作相关的投射层**

```
target_modules:  model\.(state_proj|action_in_proj|action_out_proj|
                         action_time_mlp_in|action_time_mlp_out)
```

这些是 Pi0.5 特有的层，负责将机器人的状态/动作数据映射到 Transformer 的隐藏空间。微调这些层可以让模型适配新机器人的动作空间和传感器配置。

### 注入流程（三步）

**Step 1：冻结原始权重**

```python
# src/lerobot/policies/pretrained.py:302-303
for p in self.parameters():
    p.requires_grad_(False)  # W 全部冻结
```

**Step 2：用 PEFT 库注入 LoRA 适配器**

```python
# src/lerobot/policies/pretrained.py:310
peft_model = get_peft_model(self, final_config)
```

`get_peft_model()` 遍历模型，找到所有匹配 `target_modules` 正则的 `nn.Linear` 层，在每个旁边插入 A 和 B 矩阵。

**Step 3：只训练 adapter 参数**

PEFT 会自动过滤出只有 A 和 B 参数是可训练的。运行 `print_trainable_parameters()` 会看到类似输出：

```
trainable params: 8,452,000 || all params: 3,200,000,000 || trainable%: 0.26%
```

即：只训练约 0.26% 的参数。

### 训练时

```python
loss.backward()           # 梯度只回传到 A 和 B，W 的梯度全是 0
optimizer.step()          # 只更新 A 和 B 的权重
```

### 推理时

```python
# 加载完整预训练模型
policy = Pi05Policy.from_pretrained("lerobot/pi05_so101")

# 叠加 LoRA adapter
policy = PeftModel.from_pretrained(policy, "path/to/adapter")
```

前向传播自动变成 `W·x + (A·B)·x`，无需额外修改推理代码。

---

## 与其他 PEFT 方法的对比

| 方法 | 原理 | 可训参数比例 | 表达能力 |
|------|------|-------------|---------|
| LoRA | 在注意力 Q/K/V 和 FFN 层旁注入低秩矩阵 | ~0.1%~1% | 强 |
| QLoRA | LoRA + 4-bit 量化 | 更少 | 强 |
| IA³ | 元素级缩放向量 (Infused Adapter) | 极低 (~0.01%) | 中等 |
| Adapters | 在 Transformer 层间插入 bottleneck 层 | ~1%~5% | 强 |
| Full FT | 更新全部权重 | 100% | 最强，但代价大 |

LoRA 在表达能力和效率之间取得了最佳平衡，这也是它成为机器人和 LLM 领域微调首选的原因。

---

## 延伸阅读

- [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685) — 原始论文
- [PEFT 库文档](https://huggingface.co/docs/peft) — LeRobot 使用的底层库
- `docs/so101_pipeline.md` — SO-ARM101 微调完整 Pipeline（通用）
- `docs/so101_pi05.md` — Pi0.5 模型微调指南
