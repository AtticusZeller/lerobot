# Bug Journal
> 开发过程中遇到的值得记录的问题与解决方案。

---

## 数据集 total_frames 与实际帧数不一致导致训练 IndexError
**日期**: 2026-03-23
**分类**: 数据集元数据
**状态**: ✅ 已解决

### 1. 问题情况 (Context & Issue)
- **触发条件**: 使用社区数据集 `youliangtan/so101-table-cleanup`（v2.1 格式）经 `convert_dataset_v21_to_v30.py` 转换为 v3.0 后，上传至 `Atticuxz/so101-table-cleanup`，启动 `lerobot-train` 训练 Pi0.5 LoRA。
- **具体表现**: 训练第 1 步后 DataLoader worker 抛出 `IndexError`，尝试访问不存在的帧索引（47081 > 实际大小 46963）。

### 2. 关键日志 (Key Logs)
```text
IndexError: Caught IndexError in DataLoader worker process 2.
  File "lerobot/datasets/lerobot_dataset.py", line 611, in __getitem__
    item = self.hf_dataset[idx]
  File "datasets/formatting/formatting.py", line 552, in _check_valid_index_key
    raise IndexError(f"Invalid key: {key} is out of bounds for size {size}")
IndexError: Invalid key: 47081 is out of bounds for size 46963
```

### 3. 排查过程 (Reasoning)
初步怀疑转换脚本 `convert_info()` 复制旧 `info.json` 的 `total_frames` 而不重新计算是 bug。
经实际验证三层数据后排除该假设：

| 来源 | 帧数 |
|------|------|
| `youliangtan` v2.1 `info.json` 记录的 `total_frames` | 47,513 |
| `youliangtan` v2.1 实际 80 个 parquet 文件行数之和 | 46,963 |
| `youliangtan` v2.1 `episodes` parquet 的 `length` 总和 | 46,963 |

结论：**源数据集在上传 HuggingFace Hub 时 `info.json` 的 `total_frames` 就已经比实际多了 550 帧**，转换脚本行为正确，只是继承了这个错误值。

### 4. 解决方案 (Resolution)
转换完成后，用实际 parquet 行数覆盖 `info.json` 的 `total_frames`：

```python
import json, pyarrow.parquet as pq
from pathlib import Path

dataset_dir = Path("/root/autodl-tmp/lerobot/Atticuxz/so101-table-cleanup")
info_path = dataset_dir / "meta/info.json"

actual_frames = sum(
    pq.read_metadata(p).num_rows
    for p in sorted((dataset_dir / "data").glob("**/*.parquet"))
)

with open(info_path) as f:
    info = json.load(f)
info["total_frames"] = actual_frames
with open(info_path, "w") as f:
    json.dump(info, f, indent=2)

print(f"Fixed: old -> {actual_frames}")
```

**为何有效**: `LeRobotDataset.__getitem__` 用 `info.json` 的 `total_frames` 决定采样范围上界，将其与实际 parquet 行数对齐后，DataLoader 不再生成越界索引。

> **注意**: 使用社区 v2.1 数据集转换后建议统一执行此校验，`youliangtan/so101-table-cleanup` 已确认存在此问题。

---

## `lerobot-train` 启动即崩溃：policy.repo_id missing
**日期**: 2026-03-23
**分类**: 训练配置
**状态**: ✅ 已解决

### 1. 问题情况
- **触发条件**: 运行 `lerobot-train` 未加 `--policy.push_to_hub=false`
- **具体表现**: 进入 `cfg.validate()` 就报错退出，训练未启动

### 2. 关键日志
```text
ValueError: 'policy.repo_id' argument missing.
Please specify it to push the model to the hub.
```

### 3. 根本原因
`PreTrainedConfig.push_to_hub` 默认为 `True`（`configs/policies.py:70`）。LeRobot 默认假设你要把训练结果上传到 HF Hub，所以要求同时指定 `policy.repo_id`。报错信息没有提示"你可以关掉这个功能"，容易误以为是别的配置错误。

### 4. 解决方案
所有训练命令加上：
```bash
--policy.push_to_hub=false
```
或者指定上传目标（二选一）：
```bash
--policy.push_to_hub=true --policy.repo_id=your_hf_username/model_name
```

**为何有效**: `cfg.validate()` 中对 `push_to_hub=True` 时强制检查 `repo_id`，设为 False 直接跳过该检查。

---

## HuggingFace 用户名大小写敏感导致数据集/模型路径 404
**日期**: 2026-03-23
**分类**: HuggingFace Hub / 环境配置
**状态**: ✅ 已解决

### 1. 问题情况
- **触发条件**: 在命令行中输入 HF 用户名时使用了全小写（如 `atticuxz/`），实际账号名含大写字母（如 `Atticuxz/`）
- **具体表现**: 数据集下载、`push_to_hub`、`hf download` 等操作返回 404 或找不到对应资源，报错信息不够直接

### 2. 排查方式
```bash
hf auth whoami
# 输出真实用户名（含大小写）
```

### 3. 根本原因
HuggingFace Hub 的 repo_id 路径中用户名**区分大小写**。`atticuxz/dataset` 和 `Atticuxz/dataset` 是不同路径，小写版本不存在就 404。

### 4. 解决方案
使用前先确认真实用户名：
```bash
hf auth whoami
```
所有 `--dataset.repo_id`、`--policy.repo_id`、`hf download` 参数中的用户名严格按 `whoami` 输出填写。

**为何有效**: HF Hub URL 路径大小写敏感，用户名必须与注册时完全一致。
