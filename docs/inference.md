# VLA 模型推理部署文档（SO-101 + LeRobot）

## 1. 概述

本文档描述如何将微调后的 Pi0.5 checkpoint 部署到 SO-101 机械臂上进行实机推理。

### 1.1 部署架构

LeRobot 使用 **gRPC 异步推理架构**，将模型推理与机器人控制解耦：

```
┌─────────────────────┐      gRPC         ┌─────────────────────┐      USB        ┌───────────┐
│   GPU 服务器        │ ◄────────────────► │   笔记本             │ ◄──────────────► │  SO-101   │
│                     │                     │                     │                │           │
│  - Policy Server    │  actions + obs     │  - Robot Client     │  servo cmd    │  - 6 DOF  │
│  - Pi0.5 checkpoint │ ◄───────────────── │  - 控制循环         │ ──────────────► │  - Gripper│
│  - gRPC 服务         │  ─────────────────► │  - 观测采集         │                │  - Camera │
│                     │                     │  - 录制             │                │           │
└─────────────────────┘                     └─────────────────────┘                └───────────┘
```

### 1.2 异步推理 vs 同步推理

| 特性 | 同步推理 | 异步推理 (Async) |
|------|---------|-----------------|
| 架构 | 推理-执行顺序交错 | 推理-执行并行解耦 |
| 闲置帧 | 存在（等待推理完成） | 无（提前计算下一chunk） |
| 适用场景 | 小模型、低延迟 | 大模型（VLA）、远程推理 |
| LeRobot 支持 | 部分策略 | 所有策略（包括 pi05） |

### 1.3 两种部署模式

| 模式 | 描述 | 适用场景 |
|------|------|----------|
| 本地部署 | 模型和控制在同一台机器 | 有本地 GPU（RTX 3090/4090+） |
| 远程部署 | 模型在 GPU 服务器，控制在笔记本 | 本地无足够 GPU，通过网络连接 |

---

## 2. 环境准备

### 2.1 GPU 服务器端（推理服务）

#### 硬件要求

- **GPU**: 至少 16GB 显存（Pi0.5 推理约需 14-16GB）
- **网络**: 与控制端在同一局域网，延迟 < 10ms（推荐有线连接）

#### 软件依赖

```bash
# 安装 LeRobot（含 async 推理支持）
uv sync --extra "async"

# 确认 gRPC 依赖已安装
# grpcio 已通过 [async] extra 自动安装
```

### 2.2 笔记本控制端（机器人控制）

#### 硬件要求

- **USB**: 连接 SO-101 控制板
- **相机**: USB 摄像头（确认 USB 带宽足够同时连接电机板和相机）
- **网络**: 与 GPU 服务器同局域网

#### 软件依赖

```bash
# 安装 LeRobot（含 async 推理支持）
uv sync --extra "async"

# 确认电机驱动已安装（Linux: udev rules; Windows: 驱动程序）
```

---

## 3. 推理服务部署

### 3.1 启动 Policy Server（GPU 服务器端）

#### 命令行方式

```bash
python -m lerobot.async_inference.policy_server \
     --host=0.0.0.0 \       # 监听所有网络接口（允许远程连接）
     --port=8080
```

**说明**:
- `host=0.0.0.0` 允许其他机器连接（本地测试用 `localhost` 或 `127.0.0.1`）
- `port=8080` 为默认端口，可根据需要修改
- Policy Server 启动时是"空容器"，具体策略在首次客户端握手时确定

#### API 方式

```python
from lerobot.async_inference.configs import PolicyServerConfig
from lerobot.async_inference.policy_server import serve

config = PolicyServerConfig(
    host="0.0.0.0",
    port=8080,
)
serve(config)
```

### 3.2 启动 Robot Client（笔记本控制端）

#### 命令行方式

```bash
python -m lerobot.async_inference.robot_client \
    --server_address=<GPU_IP>:8080 \
    --robot.type=so101_follower \
    --robot.port=/dev/ttyACM1 \
    --robot.id=so101_follower \
    --robot.cameras="{ top: {type: opencv, index_or_path: /dev/video10, width: 640, height: 480, fps: 30}, side: {type: intelrealsense, serial_number_or_name: 233522074606, width: 640, height: 480, fps: 30}}" \
    --task="把桌上的文具收到笔盒里" \
    --policy_type=pi05 \
    --pretrained_name_or_path=Atticuxz/pi05_expert_so101 \
    --policy_device=cuda \
    --actions_per_chunk=50 \
    --chunk_size_threshold=0.5 \
    --aggregate_fn_name=weighted_average \
    --debug_visualize_queue_size=True
```

#### API 方式

```python
import threading
from lerobot.robots.so_follower import SOFollowerConfig
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.async_inference.configs import RobotClientConfig
from lerobot.async_inference.robot_client import RobotClient
from lerobot.async_inference.helpers import visualize_action_queue_size

# 1. 配置相机（需与训练数据集的 camera key 一致）
camera_cfg = {
    "top": OpenCVCameraConfig(
        index_or_path="/dev/video10",
        width=640,
        height=480,
        fps=30
    ),
    "side": OpenCVCameraConfig(
        index_or_path="/dev/video11",
        width=640,
        height=480,
        fps=30
    )
}

# 2. 配置机器人
robot_cfg = SOFollowerConfig(
    port="/dev/ttyACM1",
    id="so101_follower",
    cameras=camera_cfg
)

# 3. 配置客户端
client_cfg = RobotClientConfig(
    robot=robot_cfg,
    server_address="<GPU_IP>:8080",
    policy_device="cuda",        # 服务端设备（cpu/cuda/mps/xpu）
    client_device="cpu",         # 客户端设备（通常为 cpu）
    policy_type="pi05",
    pretrained_name_or_path="Atticuxz/pi05_expert_so101",
    chunk_size_threshold=0.5,
    actions_per_chunk=50,
)

# 4. 创建并启动客户端
client = RobotClient(client_cfg)

# 5. 设置任务指令
task = "把桌上的文具收到笔盒里"

if client.start():
    # 启动 action 接收线程
    action_receiver_thread = threading.Thread(target=client.receive_actions, daemon=True)
    action_receiver_thread.start()

    try:
        # 运行控制循环
        client.control_loop(task)
    except KeyboardInterrupt:
        client.stop()
        action_receiver_thread.join()
        # 可选：绘制 action queue 大小变化
        visualize_action_queue_size(client.action_queue_size)
```

---

## 4. 推理参数配置

### 4.1 关键参数

| 参数 | 说明 | 默认值 | 推荐值 | 备注 |
|------|------|--------|--------|------|
| `actions_per_chunk` | 单次推理输出的动作数量 | 50 | 10-50 | 值越大空闲风险越小，但累积误差可能增加 |
| `chunk_size_threshold` | 队列阈值，低于此值时发送新观测 | 0.7 | 0.5-0.6 | 0.0 接近同步，1.0 每步都推理 |
| `policy_device` | 服务端推理设备 | - | `cuda` | CPU/cuda/mps/xpu |
| `client_device` | 客户端设备 | - | `cpu` | 处理观测、动作插值 |
| `aggregate_fn_name` | 重叠动作聚合函数 | - | `weighted_average` | weighted_average / max / min |

### 4.2 参数调优建议

#### `actions_per_chunk`

- **值越大**: 动作队列充足，不易空闲，但预测时间跨度长，累积误差可能增加
- **值越小**: 动作更精确，但需要更频繁推理，增加带宽和计算压力
- **建议**: 从默认 50 开始，如果经常空闲则增大，如果动作精度不足则减小

#### `chunk_size_threshold`

- **接近 0.0**: 类似同步推理，只在队列快空时才请求新推理
- **接近 1.0**: 每步都请求推理，高适应性但高带宽/计算压力
- **建议**: 0.5-0.6 通常是最佳平衡点

#### 调优方法

使用 `--debug_visualize_queue_size=True` 运行客户端，会实时绘制动作队列大小变化图：

- 队列持续接近 0 → 增大 `actions_per_chunk` 或降低控制帧率
- 队列持续接近上限 → 增大 `chunk_size_threshold`（更频繁更新）
- 队列在中间稳定震荡 → 参数合适

### 4.3 其他重要配置

| 配置项 | 说明 | 必须一致 |
|--------|------|---------|
| Camera keys | 相机名称（如 `top`, `side`） | 与数据集录制时一致 |
| Image resolution | 输入图像分辨率 | 与训练时一致 |
| Prompt | 任务指令文本 | 与训练时使用的一致 |
| Action space | 动作维度（SO-101 为 7 维） | 与配置一致 |

---

## 5. Checkpoint 管理

### 5.1 使用本地 Checkpoint

训练后，checkpoint 保存在 `outputs/<experiment_name>/checkpoints/`：

```
outputs/
└── pi05_expert_so101/
    ├── train_config.json
    ├── training_state/
    └── checkpoints/
        ├── 1000/
        │   └── pretrained_model/
        │       ├── config.json
        │       ├── model.safetensors
        │       ├── processor.json
        │       └── ...
        ├── 2000/
        └── last/                    # 最新检查点
            └── pretrained_model/
```

使用本地 checkpoint：

```bash
--pretrained_name_or_path=./outputs/pi05_expert_so101/checkpoints/last/pretrained_model
```

### 5.2 使用 HF Hub Checkpoint

```bash
--pretrained_name_or_path=Atticuxz/pi05_expert_so101
```

### 5.3 快速切换 Checkpoint（评估用）

评估阶段需要在多个 checkpoint 间快速切换：

```bash
# 1. 停止当前的 robot_client（Ctrl+C）
# 2. 在 robot_client 命令中修改 --pretrained_name_or_path
# 3. 重新启动 robot_client
```

Policy Server 无需重启 — 它在首次握手时加载策略，每次客户端连接都可以指定不同的 checkpoint。

---

## 6. 录制评估数据

### 6.1 使用 LeRobot 录制

```bash
python -m lerobot.async_inference.robot_client \
    --server_address=<GPU_IP>:8080 \
    --robot.type=so101_follower \
    --robot.port=/dev/ttyACM1 \
    --robot.id=so101_follower \
    --robot.cameras="{ top: {type: opencv, index_or_path: /dev/video10, width: 640, height: 480, fps: 30}}" \
    --task="把桌上的文具收到笔盒里" \
    --policy_type=pi05 \
    --pretrained_name_or_path=Atticuxz/pi05_expert_so101 \
    --dataset.repo_id=Atticuxz/eval_so101 \
    --dataset.single_task="收纳桌面文具到笔盒" \
    --dataset.streaming_encoding=true \
    --dataset.encoder_threads=2
```

**说明**:
- 添加 `--dataset.*` 参数会启用录制功能
- 录制内容包括：相机画面、关节状态、动作指令、时间戳
- 录制后的数据集格式与训练数据集一致，可用于后续分析

### 6.2 录制文件结构

```
eval_recordings/
├── checkpoint_step2000/
│   ├── episode_01/
│   │   ├── frame_00000.parquet
│   │   ├── videos/
│   │   │   └── top/episode_000000.mp4
│   │   └── eval_score.json        # 人工打分结果
│   ├── episode_02/
│   └── ...
├── checkpoint_step3000/
│   └── ...
└── eval_summary.csv                # 汇总表
```

### 6.3 评分记录模板

```json
{
  "checkpoint": "step-3000",
  "episode": 3,
  "task": "把桌上的文具收到笔盒里",
  "objects": ["铅笔", "橡皮", "尺子"],
  "score": {
    "铅笔": {"接近": 1, "抓取": 1, "运输": 1, "放置": 1},
    "橡皮": {"接近": 1, "抓取": 1, "运输": 0, "放置": 0},
    "尺子": {"接近": 0, "抓取": 0, "运输": 0, "放置": 0}
  },
  "total_score": 6,
  "max_score": 12,
  "progress_percent": 50,
  "duration_sec": 87,
  "failure_reason": "橡皮抓取后运输过程中掉落；尺子未尝试（超时）",
  "notes": "光照略暗于前两次"
}
```

---

## 7. 故障排查

### 7.1 连接问题

| 症状 | 可能原因 | 解决方案 |
|------|----------|---------|
| `Connection refused` | Policy Server 未启动 | 先启动 policy_server |
| `Connection timeout` | 网络不通或防火墙 | ping 测试，检查防火墙规则 |
| `Permission denied` | USB 端口权限问题 | 添加用户到 dialout 组或配置 udev rules |

### 7.2 机械臂不动或动作异常

| 症状 | 可能原因 | 解决方案 |
|------|----------|---------|
| 动作全为 0 或极小 | norm_stats 错误 | 检查 checkpoint 中的 norm_stats 是否正确 |
| 动作维度不匹配 | 配置错误 | 确认 action 维度为 7（6 DOF + 1 gripper） |
| 机械臂抖动 | 控制频率过高 | 降低相机 fps 或调整 chunk_size_threshold |
| 方向相反 | 电机 ID/标定不一致 | 确认标定文件与硬件一致 |

### 7.3 推理延迟过高

| 症状 | 可能原因 | 解决方案 |
|------|----------|---------|
| Action queue 频繁耗尽 | 推理速度跟不上 | 增大 `actions_per_chunk`，降低 fps |
| GPU 利用率低 | 数据加载瓶颈 | 检查相机配置，降低分辨率 |
| 网络延迟高 | 远程连接问题 | 使用有线网络，确保同一局域网 |

---

## 8. 仿真推理（LeIsaac + IsaacLab）

> 无需实机，通过 IsaacLab 仿真环境验证模型推理效果。

### 8.1 架构概述

```
┌─────────────────────┐      gRPC         ┌─────────────────────────────────────┐
│   GPU 服务器        │ ◄────────────────► │   IsaacLab 仿真（LeIsaac）          │
│                     │                     │                                     │
│  - Policy Server    │  actions + obs     │  - policy_inference.py              │
│  - Pi0.5 checkpoint │ ◄───────────────── │  - LeRobotServicePolicyClient       │
│  - gRPC 服务         │  ─────────────────► │  - SO-101 仿真 + 相机 + 物理引擎  │
│                     │                     │  - 自动成功/超时判定               │
└─────────────────────┘                     └─────────────────────────────────────┘
```

与实机部署的唯一区别：**笔记本 + 真机** 替换为 **IsaacLab 仿真环境**。Policy Server 侧完全不变。

### 8.2 环境准备

> 基于 [IsaacLab-uv](https://github.com/AtticusZeller/IsaacLab-uv)（uv 管理） + LeIsaac 扩展层。

#### 目录结构

```
你的工作目录/
├── IsaacLab-uv/           # 基础项目（uv 管理 IsaacSim + IsaacLab）
│   ├── pyproject.toml     # 基础依赖（torch, isaacsim, isaaclab 等）
│   ├── isaaclab/          # IsaacLab 子模块
│   └── source/            # IsaacLab 源码包
├── leisaac/               # LeIsaac 扩展（从 GitHub 克隆）
│   └── source/leisaac/    # leisaac 包源码
└── lerobot/               # LeRobot 开发版（已配置）
```

#### Step 1: 克隆 IsaacLab-uv（基础层）

```bash
git clone --recurse-submodules git@github.com:AtticusZeller/IsaacLab-uv.git
cd IsaacLab-uv

# 基础依赖：torch, isaacsim, isaaclab 等（由 IsaacLab-uv/pyproject.toml 管理）
uv sync --dev
```

#### Step 2: 克隆 LeIsaac（扩展层）

```bash
# 在 IsaacLab-uv 同级目录克隆
git clone git@github.com:LightwheelAI/leisaac.git
cd leisaac

# 下载 LeIsaac v0.3.0 资产（SO-101 USD + Toyroom 场景等）
# 从 GitIsaac Releases: https://github.com/LightwheelAI/leisaac/releases/tag/v0.3.0
# 下载 assets.zip，解压到 leisaac/assets/，覆盖现有目录

# 最终资产结构：
leisaac/assets/
├── robots/
│   └── so101_follower.usd   # SO-101 机器人模型
└── scenes/
    └── lightwheel_toyroom/   # Toyroom 仿真场景
        ├── scene.usd
        ├── assets/
        └── objects/
```

#### Step 3: 安装 LeIsaac + LeRobot 集成

```bash
# 回到 IsaacLab-uv 目录
cd ../IsaacLab-uv

# 安装 leisaac 包（含 lerobot-async 依赖：grpcio, protobuf）
# leisaac 的 pyproject.toml 在 source/leisaac/ 子目录
uv add --editable ../leisaac/source/leisaac --extras "lerobot-async"

# 覆盖 lerobot 版本：使用本地开发版（而非 pip 发布的 lerobot==0.4.2）
uv add --editable ../lerobot --extras "async"
```

#### Step 4: 验证兼容性

```bash
# 确认 IsaacSim 版本兼容
isaacsim isaacsim.exp.compatibility_check
```

#### 依赖版本对应表

| 依赖 | 版本 | 来源 |
|------|------|------|
| isaacsim | 5.1.0.0 | IsaacLab-uv pyproject.toml |
| torch | 2.7.0 | IsaacLab-uv pyproject.toml |
| isaaclab | 2.3.0 | IsaacLab-uv 子模块 |
| grpcio | 1.74.0 | leisaac lerobot-async extra |
| protobuf | 6.32.0 | leisaac lerobot-async extra（覆盖 IsaacLab 默认的 3.20.3） |

**⚠️ protobuf 版本冲突**：IsaacLab 默认安装 protobuf==3.20.3，leisaac 的 `lerobot-async` 需要 protobuf==6.32.0。`uv add --extras "lerobot-async"` 会自动升级 protobuf。

### 8.3 可用的 SO-101 仿真任务

| 任务 ID | 描述 | 成功条件 |
|---------|------|----------|
| `LeIsaac-SO101-CleanToyTable-v0` | 单臂桌面收纳（2 个物体放入盒子） | 物体均在盒内 + 手臂回 rest pose |
| `LeIsaac-SO101-PickOrange-v0` | 捡 3 个橙子放到盘子上 | 所有橙子在盘子范围内 + rest pose |
| `LeIsaac-SO101-LiftCube-v0` | 举起红色方块 20cm | 方块高度 > 基座 + 20cm |
| `LeIsaac-SO101-AssembleHamburger-v0` | 组装汉堡 | 任务特定条件 |

**任务选择建议**：如果训练数据是桌面收纳（table cleanup），选 `LeIsaac-SO101-CleanToyTable-v0`。

### 8.4 启动仿真推理

#### Step 1: 启动 Policy Server（与实机部署完全一致）

```bash
# 在 lerobot 项目目录下（lerobot conda 环境）
python -m lerobot.async_inference.policy_server \
    --host=0.0.0.0 \
    --port=8080
```

#### Step 2: 启动 LeIsaac 仿真推理

```bash
# 在 leisaac 项目目录下（leisaac conda 环境）
cd /path/to/leisaac

python scripts/evaluation/policy_inference.py \
    --task=LeIsaac-SO101-CleanToyTable-v0 \
    --policy_type=lerobot-pi05 \
    --policy_host=<POLICY_SERVER_IP> \
    --policy_port=8080 \
    --policy_checkpoint_path=Atticuxz/pi05_expert_so101 \
    --policy_action_horizon=50 \
    --policy_language_instruction="把桌上的文具收到笔盒里" \
    --policy_timeout_ms=15000 \
    --episode_length_s=60.0 \
    --eval_rounds=10 \
    --step_hz=60 \
    --enable_cameras
```

**参数说明**：

| 参数 | 说明 |
|------|------|
| `--task` | LeIsaac 注册的仿真任务 ID |
| `--policy_type=lerobot-pi05` | `lerobot-` 前缀 + 策略名（pi05/smolvla/act 等） |
| `--policy_host` | Policy Server 所在 IP（同机用 `localhost`） |
| `--policy_port` | 与 Policy Server 端口一致 |
| `--policy_checkpoint_path` | HF Hub repo ID 或本地 checkpoint 路径 |
| `--policy_action_horizon` | 每次推理产生的动作数（对应 `actions_per_chunk`） |
| `--policy_language_instruction` | 自然语言任务指令（应与训练时一致） |
| `--eval_rounds` | 自动评估的 episode 数（0 = 无超时，手动 R 键重置） |
| `--episode_length_s` | 每个 episode 最大时长（秒） |
| `--enable_cameras` | 必须启用，否则无相机观测 |
| `--headless` | 可选：无头模式运行（无 GUI 窗口，适合批量评估） |

#### 交互控制

- **R 键**: 手动重置当前 episode（`--eval_rounds=0` 时使用）
- 仿真窗口实时显示机械臂动作和物体状态

### 8.5 仿真推理的数据流

```
IsaacLab 环境                     LeRobotServicePolicyClient              LeRobot Policy Server
─────────────────                 ─────────────────────────               ────────────────────
env.step(action)
    │
    ▼
obs["policy"]
  ├─ joint_pos (rad)  ──────────► convert_leisaac_action_to_lerobot()
  │                                  │ (弧度 → 归一化 motor range)
  │                                  ▼
  │                               observation.state (6,)  ──────────────► gRPC SendObservations
  ├─ front (640×480 RGB) ────────► observation.images.front ────────────►
  ├─ wrist (640×480 RGB) ────────► observation.images.wrist ────────────►
  └─ task_description ───────────► task metadata ───────────────────────►
                                                                            │
                                                                            ▼
                                                                      Pi0.5 推理
                                                                            │
                                                                            ▼
                                  action_chunk (N, 6) ◄──────────────── gRPC GetActions
                                      │
                                      ▼
                                  convert_lerobot_action_to_leisaac()
                                      │ (归一化 → 弧度)
                                      ▼
                                  actions (N, 1, 6) tensor
                                      │
                                      ▼
                                  env.step(action[i])  ──► 物理仿真更新
```

### 8.6 关键兼容性说明

#### Camera Key 映射

| LeIsaac 仿真 | 训练数据集（实机录制） | 说明 |
|--------------|----------------------|------|
| `front` | `top` | 基座固定视角 |
| `wrist` | `side` | 腕部/侧面视角 |

**⚠️ 重要**：如果训练数据集的 camera key 是 `top`/`side`，而 LeIsaac 默认用 `front`/`wrist`，会导致策略无法正确匹配图像输入。

**解决方案**（选其一）：

1. **修改 LeIsaac 环境配置**（推荐用于验证已有 checkpoint）：
   在 `SingleArmTaskSceneCfg` 中将相机名称改为与训练数据一致：
   ```python
   # leisaac/tasks/template/single_arm_env_cfg.py
   # 将 "wrist" 改为 "side"，"front" 改为 "top"
   top: TiledCameraCfg = TiledCameraCfg(...)   # 原 front
   side: TiledCameraCfg = TiledCameraCfg(...)   # 原 wrist
   ```

2. **用仿真数据重新训练**（推荐用于仿真专用评估流程）：
   用 LeIsaac 采集仿真数据集（camera key 为 `front`/`wrist`），重新微调。

#### 动作空间

| 维度 | 关节 | LeIsaac 单位 | LeRobot 单位 |
|------|------|-------------|-------------|
| 0 | shoulder_pan | 弧度 (rad) | 归一化 (-100, 100) |
| 1 | shoulder_lift | 弧度 | 归一化 (-100, 100) |
| 2 | elbow_flex | 弧度 | 归一化 (-100, 100) |
| 3 | wrist_flex | 弧度 | 归一化 (-100, 100) |
| 4 | wrist_roll | 弧度 | 归一化 (-100, 100) |
| 5 | gripper | 弧度 | 归一化 (0, 100) |

转换由 `convert_leisaac_action_to_lerobot()` / `convert_lerobot_action_to_leisaac()` 自动处理，无需手动干预。

#### 关节限位映射

```
USD 关节限位 (degree)          Motor 限位 (归一化)
shoulder_pan:  (-110, 110)  ↔  (-100, 100)
shoulder_lift: (-100, 100)  ↔  (-100, 100)
elbow_flex:    (-100, 90)   ↔  (-100, 100)
wrist_flex:    (-95, 95)    ↔  (-100, 100)
wrist_roll:    (-160, 160)  ↔  (-100, 100)
gripper:       (-10, 100)   ↔  (0, 100)
```

### 8.7 批量自动评估

LeIsaac 的 `policy_inference.py` 内置了自动评估循环（`--eval_rounds`），每个 episode 结束后输出成功率。

#### 单 checkpoint 评估

```bash
python scripts/evaluation/policy_inference.py \
    --task=LeIsaac-SO101-CleanToyTable-v0 \
    --policy_type=lerobot-pi05 \
    --policy_host=localhost --policy_port=8080 \
    --policy_checkpoint_path=./outputs/pi05_expert_so101/checkpoints/last/pretrained_model \
    --policy_action_horizon=50 \
    --policy_language_instruction="把桌上的文具收到笔盒里" \
    --eval_rounds=10 \
    --headless \
    --enable_cameras
```

输出示例：
```
[Evaluation] Evaluating episode 1...
[Evaluation] Episode 1 is successful!
[Evaluation] now success rate: 1.000  [1/1]
[Evaluation] Evaluating episode 2...
[Evaluation] Episode 2 timed out!
[Evaluation] now success rate: 0.500  [1/2]
...
[Evaluation] Final success rate: 0.700  [7/10]
```

#### 多 checkpoint 批量对比（脚本示例）

```bash
#!/bin/bash
# batch_eval_sim.sh — 批量评估多个 checkpoint 的仿真成功率

POLICY_SERVER_HOST=localhost
POLICY_SERVER_PORT=8080
TASK="LeIsaac-SO101-CleanToyTable-v0"
INSTRUCTION="把桌上的文具收到笔盒里"
EVAL_ROUNDS=10
LEISAAC_DIR="/path/to/leisaac"

CHECKPOINTS=(
    "./outputs/pi05_expert_so101/checkpoints/1000/pretrained_model"
    "./outputs/pi05_expert_so101/checkpoints/2000/pretrained_model"
    "./outputs/pi05_expert_so101/checkpoints/3000/pretrained_model"
    "./outputs/pi05_expert_so101/checkpoints/last/pretrained_model"
    "Atticuxz/pi05_expert_so101"
)

echo "checkpoint,success_rate,success_count,total" > eval_results.csv

for ckpt in "${CHECKPOINTS[@]}"; do
    echo "=========================================="
    echo "Evaluating: $ckpt"
    echo "=========================================="

    result=$(cd "$LEISAAC_DIR" && python scripts/evaluation/policy_inference.py \
        --task="$TASK" \
        --policy_type=lerobot-pi05 \
        --policy_host="$POLICY_SERVER_HOST" \
        --policy_port="$POLICY_SERVER_PORT" \
        --policy_checkpoint_path="$ckpt" \
        --policy_action_horizon=50 \
        --policy_language_instruction="$INSTRUCTION" \
        --eval_rounds="$EVAL_ROUNDS" \
        --headless \
        --enable_cameras 2>&1 | tail -1)

    # 解析 "Final success rate: 0.700  [7/10]"
    rate=$(echo "$result" | grep -oP '[\d.]+(?=\s+\[)')
    counts=$(echo "$result" | grep -oP '\[\K[^\]]+')

    echo "$ckpt,$rate,$counts" >> eval_results.csv
    echo "Result: $result"
done

echo ""
echo "=========================================="
echo "Results saved to eval_results.csv"
cat eval_results.csv
```

**使用方式**：

```bash
# 1. 先启动 Policy Server（保持运行，所有 checkpoint 共用）
python -m lerobot.async_inference.policy_server --host=0.0.0.0 --port=8080

# 2. 运行批量评估脚本
bash batch_eval_sim.sh
```

> **注意**: 每次切换 checkpoint 时，`policy_inference.py` 会重新通过 `SendPolicyInstructions` 初始化策略，Policy Server 会自动加载新 checkpoint，无需重启。

### 8.8 仿真 vs 实机评估对比

| 维度 | 仿真评估（LeIsaac） | 实机评估 |
|------|---------------------|----------|
| 速度 | 快（可 headless 并行） | 慢（人工操作 + 观察） |
| 评分 | 自动（成功率/超时） | 人工（π0.5 子步骤 rubric） |
| 物理真实性 | 近似（IsaacSim 物理引擎） | 真实 |
| 环境一致性 | 完全一致（每次重置） | 有变化（光照/物体位置） |
| sim-to-real gap | 存在（材质/摩擦/相机差异） | 无 |
| 适用阶段 | 快速筛选候选方案 | 最终验证 Top 2-3 方案 |

**推荐流程**: 仿真批量筛选 → 选出 Top 2-3 → 实机少量验证

---

## 9. 参考链接

- LeRobot 异步推理官方文档: https://huggingface.co/docs/lerobot/async
- LeRobot 异步推理博客: https://huggingface.co/blog/async-robot-inference
- LeIsaac 项目: https://github.com/huggingface/LeIsaac
- LeIsaac 可用环境列表: `python scripts/environments/list_envs.py`
- 评估流程文档: [eval.md](./eval.md)
- 完整训练 Pipeline: [so101_pipeline.md](./so101_pipeline.md)

---

## 附录 A: 支持的策略和机器人

### 支持的策略 (SUPPORTED_POLICIES)

```python
SUPPORTED_POLICIES = ["act", "smolvla", "diffusion", "tdmpc", "vqbet", "pi0", "pi05", "groot"]
```

### 支持的机器人 (SUPPORTED_ROBOTS)

```python
SUPPORTED_ROBOTS = ["so100_follower", "so101_follower", "bi_so_follower", "omx_follower"]
```

SO-ARM101 在 LeRobot 中对应 `so101_follower` 类型。
