# AutoDL 数据盘环境变量配置

在使用 AutoDL 等云服务器时，系统盘通常空间有限（如 30GB），而数据盘空间较大（如 200GB+）。训练过程中 HuggingFace 模型缓存和 WandB 日志/artifact 会占用大量空间，需要配置环境变量将其转移到数据盘。

## 问题现象

- `OSError: [Errno 28] No space left on device` 在训练保存 checkpoint 时
- WandB artifact 暂存目录 `/root/.local/share/wandb/artifacts/staging/` 爆满
- HuggingFace 缓存目录 `~/.cache/huggingface` 占用系统盘空间

## 解决方案

在 `~/.zshrc` 或 `~/.bashrc` 中添加以下环境变量：

```bash
# HuggingFace 配置 - 模型和数据集缓存移到数据盘
export HF_HOME=/root/autodl-tmp/huggingface
export HF_HUB_CACHE=/root/autodl-tmp/huggingface/hub
export HF_LEROBOT_HOME=/root/autodl-tmp/lerobot

# WandB 配置 - 运行目录、缓存、数据、artifact 全部移到数据盘
export WANDB_DIR=/root/autodl-tmp/wandb
export WANDB_CACHE_DIR=/root/autodl-tmp/wandb/cache
export WANDB_DATA_DIR=/root/autodl-tmp/wandb/data
export WANDB_ARTIFACT_DIR=/root/autodl-tmp/wandb/artifacts
```

## 创建目录

```bash
mkdir -p /root/autodl-tmp/huggingface/hub
mkdir -p /root/autodl-tmp/lerobot
mkdir -p /root/autodl-tmp/wandb/{cache,data,artifacts}
```

## 环境变量说明

| 变量 | 作用 | 默认值 |
|------|------|--------|
| `HF_HOME` | HuggingFace 所有下载内容的根目录 | `~/.cache/huggingface` |
| `HF_HUB_CACHE` | 模型文件缓存目录（transformers, diffusers 等） | `$HF_HOME/hub` |
| `HF_LEROBOT_HOME` | LeRobot 数据集和模型存储目录 | `$HF_HOME/lerobot` |
| `WANDB_DIR` | WandB 运行目录（run 目录的父目录） | `./wandb` |
| `WANDB_CACHE_DIR` | WandB 缓存目录 | `~/.cache/wandb` |
| `WANDB_DATA_DIR` | WandB 数据目录 | `~/.local/share/wandb` |
| `WANDB_ARTIFACT_DIR` | WandB artifact 暂存目录 | `./artifacts` |

## 应用配置

```bash
# 如果是 zsh
source ~/.zshrc

# 如果是 bash
source ~/.bashrc
```

## 验证配置生效

```python
import os
import wandb.env as wandb_env
from huggingface_hub.constants import HF_HOME, HF_HUB_CACHE
from lerobot.utils.constants import HF_LEROBOT_HOME

# 检查 HuggingFace
print("HF_HOME:", HF_HOME)
print("HF_HUB_CACHE:", HF_HUB_CACHE)
print("HF_LEROBOT_HOME:", HF_LEROBOT_HOME)

# 检查 WandB
print("WANDB_DIR:", os.environ.get("WANDB_DIR"))
print("WANDB_CACHE_DIR:", wandb_env.get_cache_dir())
print("WANDB_DATA_DIR:", wandb_env.get_data_dir())
print("WANDB_ARTIFACT_DIR:", wandb_env.get_artifact_dir())
```

## 注意事项

1. **配置时机**：在启动任何训练之前配置好，否则可能已经写入默认位置
2. **存量迁移**：如果已有缓存需要迁移，将对应目录内容复制到新位置
3. **磁盘监控**：即使移到数据盘，也要定期清理不必要的 artifact 和模型缓存
4. **持久化**：AutoDL 关机后数据盘内容保留，但最好将重要模型备份到 HuggingFace Hub
5. **扩容建议**：AutoDL 数据盘默认大小可能仍不够支撑大型项目，建议根据需要手动增加数据盘大小

## 清理命令

```bash
# 清理 WandB 缓存
wandb artifact cache cleanup 10GB

# 查看 WandB 磁盘使用
wandb artifact cache list

# 手动删除旧缓存
rm -rf /root/autodl-tmp/wandb/cache/*

# 清理 HuggingFace 缓存（谨慎使用）
rm -rf /root/autodl-tmp/huggingface/hub/models--*/snapshots/*
```
