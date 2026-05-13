---
name: experiment-log
description: Format and update experiment logs in docs/experiments_log.md. Use when the user wants to record training results, log WandB runs, save model checkpoint locations, document dataset info, or add validation results. Trigger on: "记录实验", "log experiment", "实验结果", "training results", "wandb", "模型权重", "model weights", "保存实验", "save experiment", "实验日志", "experiment log", or when user shares WandB links, HuggingFace model links, or training metrics.
---

# Experiment Log Skill

Format and maintain `docs/experiments_log.md` as a structured experiment record.

## Core Principle

The log file is a **pure factual record** — no subjective analysis, no opinions, no evaluations. Just formatted data. When discussing with the user during the process, you can add your analysis and understanding, but the file itself stays objective.

## File Structure

```markdown
# 实验记录 (Experiments Log)

## YYYY-MM-DD

### 实验 #N — [简短描述]
...

### 实验 #M — [简短描述]
...

## YYYY-MM-DD

### 实验 #X — [简短描述]
...
```

**Ordering rules:**
- Date sections: newest date on top (reverse chronological)
- Within each date: newest experiment number on top
- When a new experiment has a different date than the previous top entry, create a new date section

## Entry Format

Each experiment entry uses this structure:

```markdown
### 实验 #N — [简短描述]

| 字段 | 内容 |
|------|------|
| **WandB** | [链接](URL) |
| **模型权重** | [链接](URL) |
| **模型** | 模型名 + 参数量 |
| **微调方式** | 全量微调 / LoRA / 冻结 VLM 等 |
| **配置** | Batch Size X, N steps, LR=..., ... |
| **显存占用** | ~XGB（可选） |
| **数据集** | [repo_id](HF datasets URL) 或文字描述 |
| **数据集描述** | 数据集内容简述 |
| **训练时长** | X 分钟/小时 |
| **训练结果** | 最终 loss / 关键指标 |
| **验证效果** | 实机测试结果（可选） |
```

## Workflow

### 1. Collect Information

Gather from the user or context:
- WandB run URL
- HuggingFace model weight URL
- Model name and size
- Fine-tuning approach (full / LoRA / frozen VLM, etc.)
- Training config (batch size, steps, learning rate, etc.)
- GPU memory usage (if mentioned)
- Dataset info: HuggingFace datasets repo URL + description of content
- Training duration
- Final metrics (loss, etc.)
- Validation results (if available)

### 2. Format Entry

- Determine experiment number: count all existing `### 实验 #` entries across all date sections + 1
- Use the table format above
- Leave optional fields empty if not provided
- Keep descriptions concise and factual

### 3. Update File

- Read current `docs/experiments_log.md`
- Insert new entry at the top of the matching date section (or create a new date section if needed)
- Preserve all existing content

### 4. Confirm to User

Show the formatted entry and confirm it was added. During this conversation, you may add your analysis — but the file content remains purely factual.

## Example Entry

```markdown
### 实验 #2 — XVLa SO101 冻结VLM 微调

| 字段 | 内容 |
|------|------|
| **WandB** | [链接](https://wandb.ai/atticux/xvla_so101/runs/2i9akaxt) |
| **模型权重** | [链接](https://huggingface.co/Atticuxz/xvla_so101_20260513_0905) |
| **模型** | XVLa 0.9B |
| **微调方式** | 冻结 VLM，仅微调 action head |
| **配置** | Batch Size 32, 2000 steps |
| **数据集** | [Atticuxz/so101-table-cleanup](https://huggingface.co/datasets/Atticuxz/so101-table-cleanup) |
| **数据集描述** | 桌面清理任务 |
| **训练时长** | ~10 分钟 |
| **训练结果** | loss 在 2000 step 收敛至 ~0.03 |
| **验证效果** | — |
```

## Handling Validation Results

When user provides validation/evaluation results:
- Add to the `验证效果` field
- Include specific metrics if available (success rate, task completion, etc.)
- Keep it factual: "5/10 次成功抓取" not "效果还不错"

## Updating Existing Entries

If user wants to update an existing entry (e.g., add validation results later):
- Find the entry by experiment number or WandB link
- Update the specific field
- Don't duplicate entries
