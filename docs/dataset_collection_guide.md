# LeRobot 数据集采集与标注指南（SO-ARM101 遥操作）

本文档聚焦 **SO-ARM101 + Leader Arm 遥操作** 场景，覆盖数据集格式、采集流程、清洗标准与标注能力。

> **适用范围**
> - 机器人：SO-ARM101（ `so101_follower` + `so101_leader` ）
> - 采集方式：**Leader Arm 遥操作**（不含 HIL、手机遥操、仿真）
> - 目标策略：SmolVLA（基于 Pi0，**无需子任务标注**）/ Pi0.5（**需要子任务标注**）

---

## 目录

1. [数据集格式详解（v3.0）](#1-数据集格式详解v30)
2. [自动生成字段含义](#2-自动生成字段含义)
3. [采集过程的要求](#3-采集过程的要求官方最佳实践)
4. [代码库中的配置要求](#4-代码库中的配置要求)
5. [遥操作录制命令](#5-遥操作录制命令)
6. [数据清洗与质量标准](#6-数据清洗与质量标准)
7. [数据标注能力（Pi0.5 需要 / SmolVLA 不需要）](#7-数据标注能力)
8. [可视化与验证](#8-可视化与验证)
9. [快速参考流程](#9-快速参考流程)

---

## 1. 数据集格式详解（v3.0）

### 1.1 目录结构

```
dataset/
├── meta/
│   ├── info.json                    # 数据集 schema、特征定义、FPS、路径模板
│   ├── stats.json                   # 全局特征统计（mean/std/min/max）
│   ├── tasks.parquet                # 任务描述 → task_index 映射
│   └── episodes/
│       └── chunk-000/
│           └── file-000.parquet     # 每 episode 元信息（长度、任务、起止偏移）
├── data/
│   └── chunk-000/
│       └── file-000.parquet         # 帧级数据（多个 episode 合并在同一 parquet）
└── videos/
    └── <camera_key>/                # 按摄像头分目录，如 front/wrist 等
        └── chunk-000/
            └── file-000.mp4         # 多个 episode 视频合并为一个 mp4 分片
```

### 1.2 与 v2.x 的差异

* **合并存储**：v3.0 将多个 episode 的 parquet / mp4 合并为 "chunk + file" 分片，减少小文件数量。
* **episode 元数据 Parquet 化**：`meta/episodes/` 取代 v2.x 的 `episodes.jsonl`。
* **tasks.parquet**：取代 `tasks.jsonl`，支持列式读取。
* **流式视频编码**：采集时可选 `--dataset.streaming_encoding=true`，边录边编码，大幅降低磁盘 IO。

### 1.3 三大文件的分工

| 文件 | 作用 | 读取频率 |
|------|------|---------|
| `meta/info.json` | 数据集的"说明书"：features schema、fps、chunk 大小、视频参数 | 每次加载必读 |
| `data/chunk-*/file-*.parquet` | 每一帧的状态/动作/时间戳（视频帧以索引引用） | 训练时随机采样 |
| `videos/<cam>/chunk-*/file-*.mp4` | 图像数据（单独压缩存储，避免 parquet 膨胀） | 训练时按 frame_index 解码 |

---

## 2. 自动生成字段含义

每一帧 Parquet 行都会包含以下字段，其中加粗的是 **采集过程自动生成、无需配置**：

| 字段 | 类型 | 含义 | 由谁生成 |
|------|------|------|---------|
| `observation.state` | float32[N_motors] | 当前时刻 follower 的关节位置（度或归一化值） | 机器人硬件读取 |
| `action` | float32[N_motors] | 当前时刻 leader 下发给 follower 的目标关节指令 | 遥操作设备 |
| `observation.images.<camera_key>` | video / image | 对应摄像头画面（按 `frame_index` 索引进 mp4） | 摄像头 |
| **`timestamp`** | float32 | 相对 episode 起点的秒数 | **自动** |
| **`frame_index`** | int64 | 当前帧在 **本 episode 内** 的索引（从 0 开始） | **自动** |
| **`episode_index`** | int64 | 本 episode 在数据集内的序号 | **自动** |
| **`index`** | int64 | 全局帧索引（整个数据集唯一） | **自动** |
| **`task_index`** | int64 | 指向 `meta/tasks.parquet` 的任务 ID | **自动**（根据 `--dataset.single_task` 写入） |
| `subtask_index` | int64 | 子任务 ID（**仅 Pi0.5 / SARM 需要**，后处理阶段追加） | 标注流程生成 |

**要点：**

* 遥操作采集时你 **只需要关心 `observation.state` / `action` / `observation.images.*`**；其他字段由 `LeRobotDataset.add_frame()` 自动填充。
* `task_index` 是在 `save_episode()` 时根据 CLI 里的 `--dataset.single_task` 自动写入的。
* `subtask_index` 默认不存在，**只有跑了 VLM 标注后才会追加到 parquet 中**。

---

## 3. 采集过程的要求（官方最佳实践）

参考 [HuggingFace 官方博客](https://huggingface.co/blog/lerobot-datasets#what-makes-a-good-dataset)。

### 3.0 采集者视角：盯摄像头，不盯机械臂

> **核心原则**：采集时，演示者的注意力应放在 **摄像头画面**（实时预览）上，而非直接盯着机械臂。

原因：

* **模拟机器臂的主观视角**：模型推理时只能看到摄像头画面，如果演示者用肉眼直视机械臂来判断位置关系，录出的轨迹依赖了摄像头以外的信息，策略很难泛化。
* **即时质检视频流**：通过看摄像头预览，采集过程中就能发现曝光过度、遮挡、抖动、焦距错误等问题，而不是录完后才发现整批废片。

**实操建议**：

1. 采集前在旁边开一个显示器/笔记本，用本节 §8.3 的 SSH Rerun 方案实时显示摄像头画面
2. 每条 episode 开始前，先看预览确认画面清晰、无遮挡再开始操作
3. 操作过程中余光监视预览，发现画面异常（遮手、失焦、掉帧）立即按 **←** 取消重录

### 3.1 图像质量

* ✅ 推荐 **双摄像头视角**（例如 `front` + `wrist`）
* ✅ 视频 **无抖动**、 **曝光一致**、 **焦点清晰**
* ✅ 光照 **中性稳定**，避免偏黄/偏蓝
* ✅ 画面中 **不应出现 Leader Arm**
* ✅ **唯一运动物体** 是 Follower Arm 与被操作对象（避免人手、人身入镜）
* ✅ 背景 **静态、无干扰**，或使用可控变化
* ✅ 分辨率至少 **480×640（720p）**，帧率 **30 FPS**

### 3.2 Episode 内容

* ✅ 轨迹平滑、动作连贯，避免剧烈抖动
* ✅ 起始姿态一致，结束姿态稳定
* ✅ **黄金法则**：仅凭摄像头画面就能完成任务（不依赖人类视角）
* ✅ 起始 episode 数量：**至少 50 个**，每个位置变体至少 10 个

### 3.3 Feature 命名规范

**格式**： `<modality>.<location>`

| ✅ 推荐 | ❌ 避免 |
|---------|---------|
| `observation.images.front` | `observation.images.laptop` |
| `observation.images.top` | `observation.images.phone` |
| `observation.images.wrist.right` | `observation.images.cam1` |

### 3.4 任务描述

* ✅ 清晰、具体：`"Pick the yellow lego block and put it in the box"`
* ✅ 长度控制在 **25-50 字符**
* ❌ 避免：`"task1"`、`"demo2"`、空字符串、"Hold"、"Up" 等

---

## 4. 代码库中的配置要求

### 4.1 前置条件

1. **硬件连接**：SO-ARM101 Follower 和 Leader 通过 USB 串口连接
2. **电机校准**：首次使用运行 `lerobot-calibrate`
3. **摄像头发现**：`lerobot-find-cameras opencv` / `realsense`
4. **HuggingFace 登录**（如需上传）：

```bash
hf auth login --token ${HUGGINGFACE_TOKEN} --add-to-git-credential
HF_USER=$(NO_COLOR=1 hf auth whoami | awk -F': *' 'NR==1 {print $2}')
```

### 4.2 关键 CLI 参数

| 参数 | 说明 | 推荐值 |
|------|------|--------|
| `--dataset.repo_id` | HF Hub 仓库 ID | `${HF_USER}/so101-<task_name>` |
| `--dataset.single_task` | 任务描述（写入 `tasks.parquet` ） | 25-50 字符具体描述 |
| `--dataset.num_episodes` | 录制 episode 数量 | ≥ 50 |
| `--dataset.fps` | 采集帧率 | **30** |
| `--dataset.episode_time_s` | 每个 episode 时长（秒） | 20-60（根据任务） |
| `--dataset.reset_time_s` | 重置环境时长（秒） | 5-10 |
| `--dataset.streaming_encoding` | 流式视频编码（v3.0） | **true** |
| `--dataset.push_to_hub` | 录制完自动上传 | true |
| `--display_data` | 实时可视化（rerun） | true（调试时） |

### 4.3 摄像头配置

**单摄像头**：

```bash
--robot.cameras="{ front: {type: opencv, index_or_path: 0, width: 1280, height: 720, fps: 30}}"
```

**双摄像头（推荐）**：

```bash
--robot.cameras='{
  front: {type: opencv, index_or_path: "/dev/video0", width: 1280, height: 720, fps: 30},
  wrist: {type: opencv, index_or_path: "/dev/video2", width: 640, height: 480, fps: 30}
}'
```

> **Linux 权限**：RealSense 可能需要 `sudo` 或配置 udev 规则。

---

## 5. 遥操作录制命令

### 5.0 正式录制前：上机试遥操

录制前先用 `lerobot-teleoperate` 空跑，确认电机、摄像头、Leader-Follower 映射均正常，再切换到 §5.1 的 `lerobot-record`。

**第一步：终端 A — 起 Rerun Web Viewer**

```bash
# 同时开 gRPC:9876 + Web:9090，浏览器访问 http://localhost:9090
rerun --web-viewer
```

> 保持该终端不关闭；浏览器打开 `http://localhost:9090` 即可看到实时画面。

**第二步：终端 B — 启动遥操作**

```bash
lerobot-teleoperate \
    --robot.type=so101_follower \
    --robot.port=/dev/ttyACM0 \
    --robot.id=my_awesome_follower_arm \
    --robot.cameras='{
      front: {type: opencv, index_or_path: /dev/video6, width: 640, height: 480, fps: 30},
      wrist: {type: opencv, index_or_path: /dev/video0, width: 640, height: 480, fps: 30}
    }' \
    --teleop.type=so101_leader \
    --teleop.port=/dev/ttyACM1 \
    --teleop.id=my_awesome_leader_arm \
    --display_data=true \
    --display_ip=127.0.0.1 \
    --display_port=9876
```

**检查清单**（跑起来后逐项核对）：

| 检查项 | 预期现象 | 异常处理 |
|--------|---------|---------|
| Follower 上电后锁力 | 关节可抵抗外力，保持姿态 | 检查 `--robot.port` 或重新 `lerobot-calibrate` |
| Leader 拖动后 Follower 跟随 | 延迟 < 100ms，无抖动 | 检查波特率/固件版本 |
| 浏览器显示 `front` / `wrist` 两路画面 | 双流同时出现，画面清晰 | 检查 `/dev/video6`、`/dev/video0` 权限或索引 |
| 关节运动范围正常 | 无死区、无超限报警 | 重新运行 `lerobot-calibrate` |

> **Ctrl+C** 退出试遥操，不保存任何数据。确认一切正常后再运行下方 §5.1 的 `lerobot-record`。

### 5.1 完整命令

```bash
lerobot-record \
    --robot.type=so101_follower \
    --robot.port=/dev/ttyACM0 \
    --robot.id=my_so101 \
    --robot.cameras='{
      front: {type: opencv, index_or_path: "/dev/video0", width: 1280, height: 720, fps: 30},
      wrist: {type: opencv, index_or_path: "/dev/video2", width: 640, height: 480, fps: 30}
    }' \
    --teleop.type=so101_leader \
    --teleop.port=/dev/ttyACM1 \
    --teleop.id=my_leader \
    --display_data=true \
    --dataset.repo_id=${HF_USER}/so101-table-cleanup \
    --dataset.num_episodes=50 \
    --dataset.single_task="Pick the yellow lego block and put it in the box" \
    --dataset.fps=30 \
    --dataset.episode_time_s=30 \
    --dataset.reset_time_s=10 \
    --dataset.streaming_encoding=true
```

### 5.2 键盘控制

* **→（右箭头）**：提前结束当前 episode，进入 reset
* **←（左箭头）**：取消当前 episode，重录
* **ESC**：停止录制，编码视频并上传

### 5.3 断点续录

```bash
lerobot-record \
    --dataset.repo_id=${HF_USER}/so101-table-cleanup \
    --resume=true \
    --dataset.num_episodes=20    # 额外追加 20 个，非总数
```

---

## 6. 数据清洗与质量标准

> 本节的核验工作流基于 **[data_analysis_guide.md](./data_analysis_guide.md)**，后者对每项指标的物理含义、判定阈值和理论依据有完整说明，请配合阅读。

### 6.0 可视化核验工具

**在线（推荐）**：

<https://huggingface.co/spaces/lerobot/visualize_dataset> — 输入 `repo_id` 直接浏览。

**本地 Docker（数据集未公开时）**：

```bash
docker run -it -p 7860:7860 --platform=linux/amd64 --gpus all \
    registry.hf.space/lerobot-visualize-dataset:latest
```

启动后访问 `http://localhost:7860`，输入本地数据集路径或 HF `repo_id`。

### 6.1 核验流程（按顺序执行）

#### Step 1 — Dataset Statistics（`data_analysis_guide.md §2.1`）

打开 **Dataset Statistics** 面板，确认：

* **Total Episodes ≥ 50**，**FPS = 30**
* **Camera Resolutions** 与采集配置一致（如 1280×720）
* **Total Recording Time** = Total Frames / FPS，判断总录制时长是否合理

#### Step 2 — Episode Length Distribution（`data_analysis_guide.md §2.1`）

在 Dataset Statistics → Episode Lengths 区块：

* 要求 **Mean ≈ Median**——单类任务操作节奏应一致，否则说明存在离群回合
* 打开 Filtering → **Episode Length Filter**，用双滑块排除 < 5s 和 > 120s 的回合
* 视频确认后 flag 并删除离群回合

#### Step 3 — Lowest-Movement Episodes（`data_analysis_guide.md §2.2`）

在 Filtering → **Lowest-Movement** 列表：

* 🔴 红色（totalMovement < max 的 15%）→ **直接 flag 删除**（录制中机器人卡住或演示者未操作）
* 🟡 黄色（15–40%）→ 看视频确认，若中途卡顿建议删除，或考虑首尾裁剪
* 首尾裁剪方案：`filter_idle_frames.py` 输出 `keep_ranges.json` → `rebuild_trimmed_dataset.py`（**目前仍在验证中，使用前确认输出符合预期**）

#### Step 4 — Action Velocity · Most Jerky Episodes（`data_analysis_guide.md §2.5`）

在 Action Insights → **Action Velocity** 面板（切换到 All Episodes 模式）：

* 若 **Overall Verdict = Jerky** 或 **Moderate**，查看 Most Jerky Episodes 列表
* 标准：σ/max σ ≥ 70% 的**非夹爪**维度 → 看视频，酌情 flag 删除（遥操手抖、传感器跳变）
* 夹爪维度（标有 `discrete` tag）自动排除，无需处理

#### Step 5 — Demonstrator Speed Variance（`data_analysis_guide.md §2.6`）

**单人采集可跳过**：单人录制的数据集 CV 通常 < 0.2，无需速度归一化。

#### Step 6 — Cross-Episode Action Variance Heatmap（`data_analysis_guide.md §2.7`）

在 Action Insights → **Cross-Episode Action Variance Heatmap**（All Episodes 模式）：

* 重点检查：是否有某个维度**全程冷色（低方差）** → 该关节从未被充分激活 → 检查任务是否需要该维度，考虑补录更多 coverage
* 暖色区域说明数据多样性正常，SmolVLA 可直接处理

> **注意**：**State–Action Temporal Alignment（§2.4）** 和 **Action Autocorrelation（§2.3）** 是训练/推理阶段的调优指标，不属于采集阶段质量检查。详见 [so101_pipeline.md §3](./so101_pipeline.md) 和 [inference.md §3.1](./inference.md)。

### 6.2 常用清洗命令

**查看数据集统计**：

```bash
lerobot-info --repo-id ${HF_USER}/so101-table-cleanup
```

**删除低质量 Episode**：

```bash
lerobot-edit-dataset \
    --repo_id ${HF_USER}/so101-table-cleanup \
    --operation.type delete_episodes \
    --operation.episode_indices "[3, 7, 15]"
```

**修改任务描述**（批量）：

```bash
lerobot-edit-dataset \
    --repo_id ${HF_USER}/so101-table-cleanup \
    --operation.type modify_tasks \
    --operation.new_task "Pick the yellow lego block and put it in the box"
```

**移除错误摄像头**：

```bash
lerobot-edit-dataset \
    --repo_id ${HF_USER}/so101-table-cleanup \
    --operation.type remove_feature \
    --operation.feature_names "['observation.images.laptop']"
```

**批量过滤 idle 帧**（起始/结束静止段、中途卡顿段）：

参考 openpi 的 DROID idle-range 过滤思路，项目内置脚本 `src/lerobot/data_processing/filter_idle_frames.py` ，**一次扫描整个数据集的所有 episode**，输出：

* `keep_ranges.json` — `episode_index → [[start, end], ...]` 保留的帧区间
* `report.json` — 每 episode 的 `total_frames / kept_frames / keep_ratio`，以及建议删除的 episode 列表

```bash
python -m lerobot.data_processing.filter_idle_frames \
    --repo-id ${HF_USER}/so101-table-cleanup \
    --output-dir outputs/idle_filter \
    --signal observation.state \
    --idle-threshold 1e-3 \
    --min-idle-len 7 \
    --min-non-idle-len 16 \
    --trim-tail 10
```

**参数说明**：

| 参数 | 含义 | 默认值 |
|------|------|--------|
| `--signal` | 用于判断 idle 的特征列（ `observation.state` 或 `action` ） | `observation.state` |
| `--idle-threshold` | 连续两帧最大绝对差分 < 该阈值判定为 idle | `1e-3` |
| `--min-idle-len` | 连续 idle 帧 ≥ 该长度才会被过滤 | `7` |
| `--min-non-idle-len` | 保留段最短长度（< 则丢弃整段） | `16` （≈ 0.5s@30fps） |
| `--trim-tail` | 每段末尾额外裁掉的帧数（避免 chunk 动作尾部偏 idle） | `10` |
| `--delete-threshold` | `kept_frames / total_frames` 低于此值标记为待删除 | `0.3` |

脚本运行结束后会直接打印出可复制的 `lerobot-edit-dataset delete_episodes` 命令，用于一键删除几乎全 idle 的 episode。

**Replay 验证轨迹可重复性**：

```bash
lerobot-replay \
    --robot.type=so101_follower \
    --robot.port=/dev/ttyACM0 \
    --robot.id=my_so101 \
    --dataset.repo_id=${HF_USER}/so101-table-cleanup \
    --dataset.episode=0
```

### 6.3 常见问题与对应处理

| 问题 | 症状 | 处理 |
|------|------|------|
| 手动删 `.parquet` 破坏索引 | 加载报错、 `frame_index` 不连续 | 用 `lerobot-edit-dataset delete_episodes` 而非 `rm` |
| 摄像头命名混乱 | `images.laptop` 实际是腕部 | `remove_feature` 后重录，或用官方脚本重命名 |
| Action 维度不一致 | 合并数据集时报错 | 合并前用 `lerobot-info` 核对 `stats.json` |
| 任务描述空 / "Hold" | 下游训练 prompt 无信息 | `modify_tasks` 批量补写 |
| Episode 仅 1-2 帧 | 训练时被过滤但污染统计 | `delete_episodes` |

---

## 7. 数据标注能力

### 7.1 标注与策略的对应关系

| 策略 | 需要 Task 标签 | 需要 Subtask 标签 |
|------|---------------|------------------|
| ACT / Diffusion | ✅ | ❌ |
| **SmolVLA**（基于 Pi0） | ✅ | ❌ |
| **Pi0.5** | ✅ | ✅（sparse + dense） |
| SARM | ✅ | ✅（dual 模式） |

> **结论**：如果只训练 SmolVLA，**跳过整个 7.2 节**，只需保证 Task 标签质量即可。

### 7.2 Pi0.5 / SARM 的子任务标注（VLM 自动）

**位置**： `src/lerobot/data_processing/sarm_annotations/subtask_annotation.py`

**三种模式**：

| 模式 | 说明 | 适用策略 |
|------|------|---------|
| `single_stage` | 不标注子任务 | SmolVLA / ACT |
| `dense_only` | VLM 生成细粒度子任务 | - |
| `dual` | sparse（高层） + dense（细粒度） | **Pi0.5 / SARM** |

**运行命令**（Pi0.5 所需的 dual 标注）：

```bash
python src/lerobot/data_processing/sarm_annotations/subtask_annotation.py \
    --dataset_repo_id ${HF_USER}/so101-table-cleanup \
    --mode dual \
    --dense-subtasks "approach,grasp,lift,place,release" \
    --num-workers 2 \
    --gpu-ids 0,1
```

**生成内容**：

* 在 `data/chunk-*/file-*.parquet` 中追加：
  + `dense_subtask_names` / `dense_subtask_start_times` / `dense_subtask_end_frames`
  + `sparse_subtask_*`（dual 模式）
* `meta/subtasks.parquet`：子任务名 → 索引映射
* `meta/temporal_proportions_dense.json`：各子任务时间比例（SARM 论文公式 1）

### 7.3 Web 手动标注（Pi0.5 可选辅助）

* 在线：<https://huggingface.co/spaces/lerobot/annotate>
* 本地：<https://github.com/huggingface/lerobot-annotate>

浏览器中交互式标注子任务，完成后 Push to Hub。

---

## 8. 可视化与验证

### 8.1 在线可视化

<https://huggingface.co/spaces/lerobot/visualize_dataset> — 输入 `repo_id` 直接浏览。

### 8.2 本地可视化（rerun.io）

```bash
# 从 Hub
lerobot-dataset-viz --repo-id ${HF_USER}/so101-table-cleanup --episode-index 0

# 从本地
lerobot-dataset-viz --repo-id so101-table-cleanup --root ./local_data --mode local --episode-index 0
```

### 8.3 SSH 远程实时可视化（采集时推荐）

机器人主机通常无显示器，直接用 `--display_data=true` 会因为没有 GUI 报错。**推荐方案**：在本地机器上跑 Rerun server，机器人端连过去。

**第一步：本地机器（你的笔记本/工作站）**

```bash
# 安装 rerun（如果没有）
pip install rerun-sdk

# 启动 Rerun server，监听所有网卡
rerun --serve   # 默认端口 9876（gRPC）

# 或者用 Web Viewer（浏览器访问，无需安装桌面客户端）
rerun --web-viewer   # 同时开 gRPC:9876 + Web:9090
```

**第二步：机器人端（SSH 进去后）**

```bash
export LOCAL_IP=192.168.x.x   # 替换为你本机 IP

lerobot-record \
    --display_data=true \
    --display_ip=${LOCAL_IP} \
    --display_port=9876 \
    ... # 其他参数照常
```

> **说明**：远程模式下 `display_compressed_images` 会自动设为 `true`（JPEG 压缩后传输），节省带宽。Web Viewer 方式通过浏览器访问 `http://<本机IP>:9090` 查看，适合无法安装桌面客户端的场景。两者只是查看方式不同，机器人端命令完全相同。

### 8.3 子任务标注可视化（仅 Pi0.5 流程）

```bash
python src/lerobot/data_processing/sarm_annotations/subtask_annotation.py \
    --dataset_repo_id ${HF_USER}/so101-table-cleanup \
    --visualize-only \
    --num-visualizations 5
```

---

## 9. 快速参考流程

### 9.1 SmolVLA 流程（简单）

```
1. 硬件校准            lerobot-calibrate
2. 遥操作录制 ≥50 ep   lerobot-record --dataset.single_task=...
3. 可视化 & 质检       lerobot-dataset-viz / 在线 Visualizer
4. 清洗               lerobot-edit-dataset (delete_episodes / modify_tasks)
5. 训练               lerobot-train --policy.type=smolvla
```

### 9.2 Pi0.5 流程（含子任务标注）

```
1. 硬件校准            lerobot-calibrate
2. 遥操作录制 ≥50 ep   lerobot-record --dataset.single_task=...
3. 可视化 & 质检       lerobot-dataset-viz
4. 清洗               lerobot-edit-dataset
5. VLM 子任务标注      subtask_annotation.py --mode dual
6. 标注质检            subtask_annotation.py --visualize-only
7. 训练               lerobot-train --policy.type=pi05
```

---

## 参考

* 官方最佳实践：<https://huggingface.co/blog/lerobot-datasets#what-makes-a-good-dataset>
* 在线可视化：<https://huggingface.co/spaces/lerobot/visualize_dataset>
* 社区数据集标签：<https://huggingface.co/datasets?other=LeRobot>
* Discord：<https://discord.gg/ttk5CV6tUw>
