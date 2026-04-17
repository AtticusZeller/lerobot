# Bug Journal
> 开发过程中遇到的值得记录的问题与解决方案。

---

## Dev Container：nvm/UV 安装的工具在非交互 shell 中找不到
**日期**: 2026-04-12
**分类**: Docker / Dev Container
**状态**: ✅ 已解决

### 1. 问题情况
- nvm 安装 Node.js 后，实际路径为 `/root/.nvm/versions/node/v24.14.1/bin/node`（含具体版本号），`ENV PATH` 写死 `v24` 导致 `command not found`
- UV 安装 Python 3.12 后路径为 `/root/.local/share/uv/python/cpython-3.12-linux-x86_64-gnu/bin/python3.12`，同样不在 `PATH` 上

### 2. 解决方案
在 Dockerfile 安装完成后立即创建 symlink：
```dockerfile
# Python
RUN uv python install 3.12 \
    && ln -sf $(uv python find 3.12) /usr/local/bin/python3 \
    && ln -sf $(uv python find 3.12) /usr/local/bin/python

# Node
RUN . "$NVM_DIR/nvm.sh" && nvm install 24 \
    && ln -sf "$(which node)" /usr/local/bin/node \
    && ln -sf "$(which npm)" /usr/local/bin/npm
```
`/usr/local/bin` 在所有 shell 模式下均在 `PATH` 上，避免依赖 shell 初始化脚本。

---

## Dev Container：gh CLI 在容器中无法使用 keyring 认证
**日期**: 2026-04-12
**分类**: Docker / Dev Container / GitHub CLI
**状态**: ✅ 已解决

### 1. 问题情况
容器内运行 `gh auth status` 报错：
```text
failed to migrate config: cowardly refusing to continue with multi account migration:
couldn't find oauth token for "github.com": exec: "dbus-launch": executable file not found in $PATH
```
原因：笔记本上 `gh auth` 将 token 存在系统 keyring（dbus），容器内无 dbus 服务。

### 2. 解决方案
用 `GH_TOKEN` 环境变量绕过 keyring，gh CLI 优先读取该变量：
- `sync.sh` 中用 `gh auth token` 导出 token，写入 `dotfiles-private/.config/gh/hosts.yml`
- `post-create.sh` 从 hosts.yml 提取 token 并 `export GH_TOKEN=...` 写入 `~/.zshrc`

```bash
# sync.sh 关键片段
GH_TOKEN=$(gh auth token)
cat > "$REPO_DIR/.config/gh/hosts.yml" <<EOF
github.com:
    git_protocol: https
    users:
        ${GH_USER}:
            oauth_token: ${GH_TOKEN}
    user: ${GH_USER}
EOF
```

---

## Dev Container：VS Code postCreateCommand 收不到 PAT_TOKEN
**日期**: 2026-04-12
**分类**: Docker / Dev Container / VS Code
**状态**: ✅ 已解决

### 1. 问题情况
VS Code 打开容器时 `post-create.sh` 报 `PAT_TOKEN not set`，无法 clone dotfiles。

### 2. 根本原因
VS Code Dev Containers 的 `postCreateCommand` 在容器内运行，无法自动继承宿主机 shell 环境变量。

### 3. 解决方案
`devcontainer.json` 用 `${localEnv:VAR}` 语法从宿主机环境读取并注入：
```json
"containerEnv": {
    "PAT_TOKEN": "${localEnv:PAT_TOKEN}"
}
```
宿主机 `~/.zshrc` 加一行：
```bash
export PAT_TOKEN=$GITHUB_TOKEN
```
**注意**：VS Code 必须从已加载 `.zshrc` 的终端用 `code .` 启动，才能读到该变量；桌面图标启动的 VS Code 不继承 zsh 环境。

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

---

## Pi0.5 expert-only 训练 batch_size=4 无法收敛
**日期**: 2026-04-14
**分类**: 训练超参 / Pi0.5
**状态**: ✅ 已解决

### 1. 问题情况
- **触发条件**: 使用 `pi05_expert_so101_table_cleanup.yaml` 训练 Pi0.5 expert-only，`batch_size=4`，数据集 ~47k 帧 / 80 episodes
- **具体表现**: loss 不下降，训练无法收敛

### 2. 根本原因
Pi0.5 Action Expert 使用 Flow Matching（连续扩散）生成动作序列，每步训练从随机时间步采样并预测去噪方向。`batch_size=4` 时：
- 每个 batch 内时间步采样覆盖稀疏，梯度估计的**方差极大**
- 有效信噪比过低，参数更新方向近似随机游走，无法稳定收敛

这与图像扩散模型类似——小 batch 下扩散/flow 目标的梯度噪声远大于普通监督学习。

### 3. 解决方案
将 `batch_size` 从 4 提升到 16（32GB 显存，expert-only + bf16 足够）：

```yaml
batch_size: 16    # 32GB 显存，expert-only + bf16
steps: 4000       # 47513 / 16 ≈ 2970 steps/epoch ≈ 1.3 epochs → 留足步数
save_freq: 1000
```

同时开启 `gradient_checkpointing: true` 以确保显存够用。

**为何有效**: 更大的 batch 使每步梯度中时间步采样更均匀，降低方差，Flow Matching 目标的梯度估计更准确，训练得以收敛。

> **经验法则**: Flow Matching / Diffusion 类策略的 batch_size 建议 ≥ 16，小于 8 几乎不可能收敛。

---

## LeRobot Dataset 默认拉取 tag 而非 main 分支，导致元数据不一致
**日期**: 2026-04-14
**分类**: HuggingFace Hub / 数据集下载
**状态**: ✅ 已解决

### 1. 问题情况
- **触发条件**: 在 Hub 上修改了数据集 `Atticuxz/so101-table-cleanup` 的 `info.json`（修正 `total_frames` 从 47513 → 46963），push 到 main 分支后，训练机下载的数据集 `info.json` 仍是旧值 47513。
- **具体表现**: 训练第 1 步 DataLoader worker 抛出 `IndexError: Invalid key: 47207 is out of bounds for size 46963`。

### 2. 关键日志
```text
IndexError: Invalid key: 47207 is out of bounds for size 46963
```

### 3. 排查过程
对比 Hub 上不同 revision 的 `info.json`：

| Revision | `total_frames` |
|----------|---------------|
| `main` 分支 | 46963 ✅ (已修正) |
| `v3.0` tag | 47513 ❌ (旧值) |

LeRobot 的数据集加载链：`LeRobotDataset` → `LeRobotDatasetMetadata` → 根据 `codebase_version`（如 `v3.0`）查找对应 tag → 按 tag revision 下载文件。代码不会拉 `main` 分支，而是拉 `info.json` 中 `codebase_version` 对应的 tag。

### 4. 根本原因
Hub 上的 `v3.0` tag 指向旧 commit（修改 `info.json` 之前），LeRobot 按 tag 拉取，所以拿到的还是旧的 `info.json`。

### 5. 解决方案
将 `v3.0` tag 移到 main 的最新 commit：
```python
from huggingface_hub import HfApi

api = HfApi()
repo = "Atticuxz/so101-table-cleanup"
main_commit = "9eae4c5b07ca8553c2fcb8c953daff745a71ddc0"

api.delete_tag(repo_id=repo, tag="v3.0", repo_type="dataset")
api.create_tag(repo_id=repo, tag="v3.0", revision=main_commit, repo_type="dataset")
```

然后在训练机上清除缓存重新下载：
```bash
rm -rf ~/.cache/huggingface/hub/datasets--Atticuxz--so101-table-cleanup
```

> **经验法则**: 修改 Hub 上数据集的元数据后，必须同步更新对应 `codebase_version` 的 tag（如 `v3.0`），否则 LeRobot 仍会拉到旧版本。main 分支的改动不会自动反映到已有 tag。

---

## 相机访问：用户组权限与 RealSense 依赖缺失
**日期**: 2026-04-15
**分类**: 硬件 / Linux 权限 / 相机
**状态**: ✅ 已解决

### 1. 问题情况
运行 `lerobot-find-cameras` 时出现两类错误：

**(a) RealSense D435i 无法打开 / 检测失败**
```text
ERROR: Failed to connect or configure RealSense camera 244622071377:
Failed to open RealSenseCamera(244622071377).
NameError: name 'rs' is not defined
```

**(b) 部分 USB 相机读帧超时**
```text
Failed to connect or configure OpenCV camera /dev/video4:
Timed out waiting for frame from camera OpenCVCamera(/dev/video4) after 1000 ms.
```

### 2. 根本原因

| 现象 | 根本原因 |
|------|----------|
| RealSense `rs` 未定义 | `pyrealsense2` 库未安装（相机检测脚本 `try: import pyrealsense2 as rs`，失败时置 `rs=None`） |
| 相机设备归属 `plugdev` 组 | USB 相机 udev 规则将 `/dev/videoN` 分配给 `plugdev` 组，用户未在该组中 |
| `/dev/video4` 读帧超时 | 同一物理相机（icSpring）向内核注册多个 video node，只有其中一个是可读的真实视频流 |

设备组归属确认：
```bash
$ ls -l /dev/video*
crw-rw-rw-+ 1 root video   81, 0 /dev/video0   # RealSense/OpenCV
crw-rw-rw-+ 1 root plugdev 81, 4 /dev/video4   # icSpring (不可读 node)
crw-rw-rw-+ 1 root plugdev 81, 6 /dev/video6   # icSpring (可读 node)

$ groups
atticux dialout sudo video docker   # ← 缺 plugdev
```

### 3. 解决方案

**(a) 安装 `pyrealsense2`（uv 项目必须用 `uv add`，不能 `pip install`）：**
```bash
uv add pyrealsense2 --optional realsense
```

**(b) 将用户加入 `plugdev` 和 `video` 组（重新登录生效）：**
```bash
sudo usermod -aG plugdev,video $USER
# 退出并重新登录（或重启），使组变更生效
groups   # 确认包含 plugdev video
```

**(c) 多 video node 的识别：** 对同一物理相机注册多个 `/dev/videoN`（常见于 UVC 相机），用 `lerobot-find-cameras` 实际读帧验证，超时的那个 node 跳过使用。本项目中 icSpring 相机 `/dev/video4` 超时、`/dev/video6` 可读，故 `dev.sh` 的 `CAMERAS` 配置锁定为 `/dev/video6`（top）与 `/dev/video0`（wrist）。

### 4. 经验
- **优先用 `lerobot-find-cameras` 实机验证**，不要只靠 `ls /dev/video*` 判断，同一相机的多 node 只有一个能读帧。
- **RealSense 额外依赖**：必须装 `pyrealsense2`，并确保 `librealsense` udev 规则已安装（通常随包提供），否则需要 root 才能访问设备。
- **权限变更后必须重新登录**，`usermod -aG` 的新组只对新会话生效。
