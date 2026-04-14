# VLA 模型推理部署文档（SO-101 + LeRobot）

## 1. 概述

本文档描述如何将微调后的 SmolVLA checkpoint 部署到 SO-101 机械臂上进行推理（实机 / 仿真）。

### 1.1 部署架构

LeRobot 使用 **gRPC 异步推理架构**，将模型推理与机器人控制解耦：

```
┌─────────────────────┐      gRPC         ┌─────────────────────┐      USB        ┌───────────┐
│   GPU 服务器        │ ◄────────────────► │   笔记本             │ ◄──────────────► │  SO-101   │
│                     │                     │                     │                │           │
│  - Policy Server    │  actions + obs     │  - Robot Client     │  servo cmd    │  - 6 DOF  │
│  - SmolVLA ckpt     │ ◄───────────────── │  - 控制循环         │ ──────────────► │  - Gripper│
│  - gRPC 服务         │  ─────────────────► │  - 观测采集         │                │  - Camera │
│                     │                     │  - 录制             │                │           │
└─────────────────────┘                     └─────────────────────┘                └───────────┘
```

### 1.2 异步推理 vs 同步推理

#### 同步推理

```
机器人控制循环 (单线程，每步阻塞等待推理):
┌─────────────────────────────────────────────────┐
│  获取 obs → 推理(等~100ms) → 执行 action 0      │
│  获取 obs → 推理(等~100ms) → 执行 action 0      │
│  获取 obs → 推理(等~100ms) → 执行 action 0      │
│  ...                                              │
└─────────────────────────────────────────────────┘
  每个周期 = obs采集 + 推理等待 + 执行
  实际频率 ≈ 1/(33ms + 100ms) ≈ 7.5 FPS
```

#### 异步推理 (Async Inference)

```
Client 控制循环 (~30 FPS):            Server 推理线程:
┌──────────────────────┐              ┌──────────────────┐
│ 执行 action 1        │              │                  │
│ 执行 action 2        │              │  推理中... ~100ms │
│ 执行 action 3        │  ── obs ──>  │                  │
│ ...                  │              │                  │
│ 执行 action 25       │  <── chunk ──│  返回50个 action  │
│ 执行 action 26       │              │  空闲等待下一个   │
│ ...                  │              │                  │
└──────────────────────┘              └──────────────────┘
  每个周期 = obs采集 + 执行 (无推理等待)
  实际频率 ≈ 30 FPS
```

#### 对比

| 特性 | 同步推理 | 异步推理 (Async) |
|------|---------|-----------------|
| 架构 | 推理-执行顺序交错 | 推理-执行并行解耦 |
| 控制频率 | 受限于推理延迟 (~7.5 FPS) | 机器人物理频率 (~30 FPS) |
| action 来源 | 每步推 1 个，基于最新 obs | 一次推 50 个，逐个消费 |
| obs 延迟 | 无（用最新 obs） | 有（消费的是旧 obs 推理的结果） |
| 闲置帧 | 存在（等待推理完成） | 无（提前计算下一 chunk） |
| 适用场景 | 小模型、低延迟 | 大模型（VLA）、远程推理 |

#### 核心机制：一次推理生成整个 action chunk

每次推理是 **一次性 forward** 生成全部 K 步动作（如 50 个），非逐个推理：

```
Client                               Server
  │                                    │
  │── 1. 发送 observation ──────────>  │
  │                                    │── 2. 入队 observation_queue (maxsize=1)
  │                                    │── 3. GetActions() 取出 obs
  │                                    │── 4. _predict_action_chunk(obs):
  │                                    │     ├─ 准备 observation
  │                                    │     ├─ preprocess (归一化/tokenize)
  │                                    │     ├─ policy.predict_action_chunk()  ← 一次前向推理
  │                                    │     ├─ postprocess (反归一化)
  │                                    │     └─ 转成 List[TimedAction]
  │  <── 5. 返回整个 action chunk ───  │
  │── 6. 逐个执行 action               │
```

**关键流程**：

1. Client 控制循环以 1/fps 频率从 action queue 逐个取出 action 执行
2. 当队列剩余 action 数量 ≤ `chunk_size_threshold × actions_per_chunk` 时，发送新 observation
3. Server 收到 obs 后一次 forward 生成整个 chunk，整批发回 client
4. Client 将新 chunk 放入队列，继续逐个消费

#### 重叠 action 聚合机制

新 chunk 到达时，可能与队列中旧 chunk 存在 timestep 重叠。Client 通过 `aggregate_fn` 合并重叠部分，避免误差积累：

```python
# 默认加权平均：新预测权重更高
"weighted_average": lambda old, new: 0.3 * old + 0.7 * new
```

可选聚合函数：`weighted_average`（默认）、`latest_only`、`average`、`conservative`（0.7*old + 0.3*new）。

**本质**：用旧的预测快速填充控制频率，同时不断用新推理结果修正，在实时性和准确性之间取平衡。默认 `chunk_size_threshold=0.5` 表示消耗一半 action 后即请求新推理，减少过时 action 数量。

### 1.3 两种部署模式

| 模式 | 描述 | 适用场景 |
|------|------|----------|
| 本地部署 | 模型和控制在同一台机器 | 有本地 GPU（RTX 3090/4090+） |
| 远程部署 | 模型在 GPU 服务器，控制在笔记本 | 本地无足够 GPU，通过网络连接 |

---

## 2. 推理服务部署

### 2.1 启动 Policy Server（GPU 服务器端）

#### 命令行方式

```bash
python -m lerobot.async_inference.policy_server \
     --host=0.0.0.0 \       # 监听所有网络接口（允许远程连接）
     --port=8080
```

**说明**:
* `host=0.0.0.0` 允许其他机器连接（本地测试用 `localhost` 或 `127.0.0.1`）
* `port=8080` 为默认端口，可根据需要修改
* Policy Server 启动时是"空容器"，具体策略在首次客户端握手时确定

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

### 2.2 启动 Robot Client（笔记本控制端）

#### 命令行方式

```bash
python -m lerobot.async_inference.robot_client \
    --server_address=<GPU_IP>:8080 \
    --robot.type=so101_follower \
    --robot.port=/dev/ttyACM1 \
    --robot.id=so101_follower \
    --robot.cameras="{ front: {type: opencv, index_or_path: /dev/video10, width: 640, height: 480, fps: 30}, wrist: {type: intelrealsense, serial_number_or_name: 233522074606, width: 640, height: 480, fps: 30}}" \
    --task="pick up the orange and place it on the plate" \
    --policy_type=smolvla \
    --pretrained_name_or_path=Atticuxz/smolvla_so101_table_cleanup \
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
    "front": OpenCVCameraConfig(
        index_or_path="/dev/video10",
        width=640,
        height=480,
        fps=30
    ),
    "wrist": OpenCVCameraConfig(
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
    policy_type="smolvla",
    pretrained_name_or_path="Atticuxz/smolvla_so101_table_cleanup",
    chunk_size_threshold=0.5,
    actions_per_chunk=50,
)

# 4. 创建并启动客户端
client = RobotClient(client_cfg)

# 5. 设置任务指令
task = "pick up the orange and place it on the plate"

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

## 3. 推理参数配置

### 3.1 关键参数

| 参数 | 说明 | 默认值 | 推荐值 | 备注 |
|------|------|--------|--------|------|
| `actions_per_chunk` | 单次推理输出的动作数量 | 50 | 10-50 | 值越大空闲风险越小，但累积误差可能增加 |
| `chunk_size_threshold` | 队列阈值，低于此值时发送新观测 | 0.7 | 0.5-0.6 | 0.0 接近同步，1.0 每步都推理 |
| `policy_device` | 服务端推理设备 | - | `cuda` | CPU/cuda/mps/xpu |
| `client_device` | 客户端设备 | - | `cpu` | 处理观测、动作插值 |
| `aggregate_fn_name` | 重叠动作聚合函数 | - | `weighted_average` | weighted_average / max / min |

### 3.2 参数调优建议

#### `actions_per_chunk`

* **值越大**: 动作队列充足，不易空闲，但预测时间跨度长，累积误差可能增加
* **值越小**: 动作更精确，但需要更频繁推理，增加带宽和计算压力
* **建议**: 从默认 50 开始，如果经常空闲则增大，如果动作精度不足则减小

#### `chunk_size_threshold`

* **接近 0.0**: 类似同步推理，只在队列快空时才请求新推理
* **接近 1.0**: 每步都请求推理，高适应性但高带宽/计算压力
* **建议**: 0.5-0.6 通常是最佳平衡点

#### 调优方法

使用 `--debug_visualize_queue_size=True` 运行客户端，会实时绘制动作队列大小变化图：

* 队列持续接近 0 → 增大 `actions_per_chunk` 或降低控制帧率
* 队列持续接近上限 → 增大 `chunk_size_threshold`（更频繁更新）
* 队列在中间稳定震荡 → 参数合适

### 3.3 相机参数配置

#### 相机参数由两部分组成

**Key（名称）** — 由训练数据集决定，不能修改。

**Value（硬件参数）** — 需要手动确认后填入。

#### Step 1: 查看数据集的相机 key

```bash
uv run python -c "
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
meta = LeRobotDatasetMetadata('Atticuxz/so101-table-cleanup')
for k, v in meta.features.items():
    dtype = v.get('dtype', '')
    if 'image' in dtype or 'video' in dtype:
        key = k.removeprefix('observation.images.')
        shape = v['shape']
        fps = v['info']['video.fps']
        print(f'  {key}: {shape[1]}x{shape[0]} @ {fps}fps')
"
```

输出示例：

```
  front: 640x480 @ 30fps
  wrist: 640x480 @ 30fps
```

这意味着推理时 `--robot.cameras` 的 key 必须是 `front` 和 `wrist`。

#### Step 2: 用 `lerobot-find-cameras` 确认硬件参数

```bash
lerobot-find-cameras
```

输出示例：

```
--- Detected Cameras ---
Camera #0:
  Type: OpenCV
  Id: /dev/video10
  ...
Camera #1:
  Type: RealSense
  Id: 233522074606
  ...
```

#### Step 3: 手动对应物理相机 → 数据集 key

根据物理位置判断哪个相机对应数据集中的哪个 key：

| 数据集 key | 物理含义 | 确认的硬件参数 |
|-----------|---------|-------------|
| `front` | 正面固定视角（俯视桌面） | type=opencv, `/dev/video10` |
| `wrist` | 腕部安装视角（随机械臂移动） | type=intelrealsense, serial=`233522074606` |

#### Step 4: 填入配置

```yaml
# --robot.cameras 的 value 需要填写：
{
  front: {
    type: opencv,                    # lerobot-find-cameras 输出的 Type
    index_or_path: /dev/video10,     # lerobot-find-cameras 输出的 Id
    width: 640,                      # 必须与数据集一致
    height: 480,                     # 必须与数据集一致
    fps: 30                          # 必须与数据集一致
  },
  wrist: {
    type: intelrealsense,
    serial_number_or_name: 233522074606,  # lerobot-find-cameras 输出的 Id
    width: 640,
    height: 480,
    fps: 30
  }
}
```

#### Value 各字段说明

| 字段 | 怎么确定 | 能否与数据集不同 |
|------|---------|---------------|
| `type` | `lerobot-find-cameras` 输出的 Type | N/A（硬件决定） |
| `index_or_path` | `lerobot-find-cameras` 输出的 Id（OpenCV 相机） | 不同机器可能不同 |
| `serial_number_or_name` | `lerobot-find-cameras` 输出的 Id（RealSense） | 固定（硬件序列号） |
| `width` / `height` | 必须与数据集一致 | **不能** |
| `fps` | 必须与数据集一致 | **不能** |

#### 映射链路

```
dev.sh CAMERAS key          数据集 info.json              Policy input_features
─────────────────           ────────────────              ─────────────────────
front ──────────────────► observation.images.front ──► SmolVLA input (resize to 512x512)
wrist ──────────────────► observation.images.wrist ──► SmolVLA input (resize to 512x512)

build_dataset_frame() 自动映射:
  key = "observation.images.front" → values["front"] = camera.read_latest()
  key = "observation.images.wrist" → values["wrist"] = camera.read_latest()
```

### 3.4 其他重要配置

| 配置项 | 说明 | 必须一致 |
|--------|------|---------|
| Image resolution | 输入图像分辨率（推理时自动 resize 到 512x512） | width/height 与数据集一致 |
| Prompt | 任务指令文本 | 与训练时使用的一致 |
| Action space | 动作维度（SO-101 为 7 维） | 与配置一致 |

---

## 4. Checkpoint 管理

### 4.1 使用本地 Checkpoint

训练后，checkpoint 保存在 `outputs/<experiment_name>/checkpoints/` ：

```
outputs/
└── smolvla_so101/
    ├── train_config.json
    ├── training_state/
    └── checkpoints/
        ├── 2000/
        │   └── pretrained_model/
        │       ├── config.json
        │       ├── model.safetensors
        │       ├── processor.json
        │       └── ...
        ├── 4000/
        └── last/                    # 最新检查点
            └── pretrained_model/
```

使用本地 checkpoint：

```bash
--pretrained_name_or_path=./outputs/smolvla_so101/checkpoints/last/pretrained_model
```

### 4.2 使用 HF Hub Checkpoint

```bash
--pretrained_name_or_path=Atticuxz/smolvla_so101_table_cleanup
```

### 4.3 快速切换 Checkpoint（评估用）

```bash
# 1. 停止当前的 robot_client（Ctrl+C）
# 2. 在 robot_client 命令中修改 --pretrained_name_or_path
# 3. 重新启动 robot_client
```

Policy Server 无需重启 — 它在首次握手时加载策略，每次客户端连接都可以指定不同的 checkpoint。

---

## 5. 录制评估数据

### 5.1 使用 LeRobot 录制

```bash
python -m lerobot.async_inference.robot_client \
    --server_address=<GPU_IP>:8080 \
    --robot.type=so101_follower \
    --robot.port=/dev/ttyACM1 \
    --robot.id=so101_follower \
    --robot.cameras="{ front: {type: opencv, index_or_path: /dev/video10, width: 640, height: 480, fps: 30}}" \
    --task="pick up the orange and place it on the plate" \
    --policy_type=smolvla \
    --pretrained_name_or_path=Atticuxz/smolvla_so101_table_cleanup \
    --dataset.repo_id=Atticuxz/eval_smolvla_pick_orange \
    --dataset.single_task="pick orange" \
    --dataset.streaming_encoding=true \
    --dataset.encoder_threads=2
```

**说明**:
* 添加 `--dataset.*` 参数会启用录制功能
* 录制内容包括：相机画面、关节状态、动作指令、时间戳
* 录制后的数据集格式与训练数据集一致，可用于后续分析

---

## 6. 仿真推理（LeIsaac + IsaacLab）

> 无需实机，通过 IsaacLab 仿真环境验证模型推理效果。

### 6.1 架构概述

```
┌─────────────────────┐      gRPC         ┌─────────────────────────────────────┐
│   GPU 服务器        │ ◄────────────────► │   IsaacLab 仿真（LeIsaac）          │
│                     │                     │                                     │
│  - Policy Server    │  actions + obs     │  - policy_inference.py              │
│  - SmolVLA ckpt     │ ◄───────────────── │  - LeRobotServicePolicyClient       │
│  - gRPC 服务         │  ─────────────────► │  - SO-101 仿真 + 相机 + 物理引擎  │
│                     │                     │  - 自动成功/超时判定               │
└─────────────────────┘                     └─────────────────────────────────────┘
```

与实机部署的唯一区别：**笔记本 + 真机** 替换为 **IsaacLab 仿真环境**。Policy Server 侧完全不变。

### 6.2 可用的 SO-101 仿真任务

| 任务 ID | 描述 | 成功条件 |
|---------|------|----------|
| `LeIsaac-SO101-PickOrange-v0` | 捡 3 个橙子放到盘子上 | 所有橙子在盘子范围内 + rest pose |
| `LeIsaac-SO101-CleanToyTable-v0` | 单臂桌面收纳（2 个物体放入盒子） | 物体均在盒内 + 手臂回 rest pose |
| `LeIsaac-SO101-LiftCube-v0` | 举起红色方块 20cm | 方块高度 > 基座 + 20cm |
| `LeIsaac-SO101-AssembleHamburger-v0` | 组装汉堡 | 任务特定条件 |

**本次使用**: 训练数据集为 `Atticuxz/so101-table-cleanup` ，对应任务 "Grab pens and place into box" 。

### 6.3 启动仿真推理

#### Step 1: 启动 Policy Server（与实机部署完全一致）

```bash
# 在 lerobot 项目目录下
cd /home/atticuszz/DevSpace/lerobot

python -m lerobot.async_inference.policy_server \
    --host=0.0.0.0 \
    --port=8080
```

#### Step 2: 启动 LeIsaac 仿真推理

**方式 A — 直接调用 python 脚本**：

```bash
cd /home/atticuszz/DevSpace/leisaac

python scripts/evaluation/policy_inference.py \
    --task=LeIsaac-SO101-PickOrange-v0 \
    --policy_type=lerobot-smolvla \
    --policy_host=<POLICY_SERVER_IP> \
    --policy_port=8080 \
    --policy_checkpoint_path=Atticuxz/smolvla_so101_table_cleanup \
    --policy_action_horizon=50 \
    --policy_language_instruction="pick up the orange and place it on the plate" \
    --policy_timeout_ms=15000 \
    --episode_length_s=60.0 \
    --eval_rounds=10 \
    --step_hz=60 \
    --enable_cameras
```

**方式 B — 使用 `run.sh` 封装脚本**：

```bash
cd /home/atticuszz/DevSpace/leisaac

./run.sh infer \
    --task LeIsaac-SO101-PickOrange-v0 \
    --policy_type lerobot-smolvla \
    --policy_host <POLICY_SERVER_IP> \
    --policy_port 8080 \
    --policy_checkpoint_path Atticuxz/smolvla_so101_table_cleanup \
    --policy_action_horizon 50 \
    --policy_language_instruction "pick up the orange and place it on the plate" \
    --eval_rounds 10
```

#### 参数说明

| 参数 | 说明 |
|------|------|
| `--task` | LeIsaac 注册的仿真任务 ID |
| `--policy_type=lerobot-smolvla` | `lerobot-` 前缀 + 策略名（pi05/smolvla/act 等） |
| `--policy_host` | Policy Server 所在 IP（同机用 `localhost` ） |
| `--policy_port` | 与 Policy Server 端口一致 |
| `--policy_checkpoint_path` | HF Hub repo ID 或本地 checkpoint 路径 |
| `--policy_action_horizon` | 每次推理产生的动作数（对应 `actions_per_chunk` ） |
| `--policy_language_instruction` | 自然语言任务指令（应与训练时一致） |
| `--eval_rounds` | 自动评估的 episode 数（0 = 无超时，手动 R 键重置） |
| `--episode_length_s` | 每个 episode 最大时长（秒） |
| `--enable_cameras` | 必须启用，否则无相机观测 |
| `--headless` | 可选：无头模式运行（无 GUI 窗口，适合批量评估） |

#### 交互控制

* **R 键**: 手动重置当前 episode（`--eval_rounds=0` 时使用）
* 仿真窗口实时显示机械臂动作和物体状态

### 6.4 仿真推理的数据流

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
                                                                      SmolVLA 推理
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

### 6.5 关键兼容性说明

#### Camera Key 映射

| LeIsaac 仿真 | 训练数据集 `so101-table-cleanup` | 说明 |
|--------------|----------------------------------|------|
| `front` | `front` | 正面固定视角（俯视桌面） |
| `wrist` | `wrist` | 腕部安装视角（随机械臂移动） |

当前训练数据集 `Atticuxz/so101-table-cleanup` 已使用 `front` / `wrist` 命名，与 LeIsaac 默认一致，无需额外映射。

**⚠️ 注意**：如果使用旧数据集（camera key 为 `top` / `side` ），需要在 LeIsaac 或推理配置中做映射。参见 3.3 节相机参数配置流程。

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

### 6.6 批量自动评估

LeIsaac 的 `policy_inference.py` 内置了自动评估循环（ `--eval_rounds` ），每个 episode 结束后输出成功率。

#### 单 checkpoint 评估

```bash
cd /home/atticuszz/DevSpace/leisaac

python scripts/evaluation/policy_inference.py \
    --task=LeIsaac-SO101-PickOrange-v0 \
    --policy_type=lerobot-smolvla \
    --policy_host=localhost --policy_port=8080 \
    --policy_checkpoint_path=./outputs/smolvla_so101/checkpoints/last/pretrained_model \
    --policy_action_horizon=50 \
    --policy_language_instruction="pick up the orange and place it on the plate" \
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
TASK="LeIsaac-SO101-PickOrange-v0"
INSTRUCTION="pick up the orange and place it on the plate"
EVAL_ROUNDS=10
LEISAAC_DIR="/home/atticuszz/DevSpace/leisaac"

CHECKPOINTS=(
    "./outputs/smolvla_so101/checkpoints/2000/pretrained_model"
    "./outputs/smolvla_so101/checkpoints/4000/pretrained_model"
    "./outputs/smolvla_so101/checkpoints/6000/pretrained_model"
    "./outputs/smolvla_so101/checkpoints/last/pretrained_model"
    "Atticuxz/smolvla_so101_table_cleanup"
)

echo "checkpoint,success_rate,success_count,total" > eval_results.csv

for ckpt in "${CHECKPOINTS[@]}"; do
    echo "=========================================="
    echo "Evaluating: $ckpt"
    echo "=========================================="

    result=$(cd "$LEISAAC_DIR" && python scripts/evaluation/policy_inference.py \
        --task="$TASK" \
        --policy_type=lerobot-smolvla \
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
cd /home/atticuszz/DevSpace/lerobot
python -m lerobot.async_inference.policy_server --host=0.0.0.0 --port=8080

# 2. 运行批量评估脚本
bash batch_eval_sim.sh
```

> **注意**: 每次切换 checkpoint 时， `policy_inference.py` 会重新通过 `SendPolicyInstructions` 初始化策略，Policy Server 会自动加载新 checkpoint，无需重启。

### 6.7 仿真 vs 实机评估对比

| 维度 | 仿真评估（LeIsaac） | 实机评估 |
|------|---------------------|----------|
| 速度 | 快（可 headless 并行） | 慢（人工操作 + 观察） |
| 评分 | 自动（成功率/超时） | 人工（子步骤 rubric） |
| 物理真实性 | 近似（IsaacSim 物理引擎） | 真实 |
| 环境一致性 | 完全一致（每次重置） | 有变化（光照/物体位置） |
| sim-to-real gap | 存在（材质/摩擦/相机差异） | 无 |
| 适用阶段 | 快速筛选候选方案 | 最终验证 Top 2-3 方案 |

**推荐流程**: 仿真批量筛选 → 选出 Top 2-3 → 实机少量验证

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
| Action queue 频繁耗尽 | 推理速度跟不上 | 增大 `actions_per_chunk` ，降低 fps |
| GPU 利用率低 | 数据加载瓶颈 | 检查相机配置，降低分辨率 |
| 网络延迟高 | 远程连接问题 | 使用有线网络，确保同一局域网 |

---

## 8. 参考链接

* LeRobot 异步推理官方文档: https://huggingface.co/docs/lerobot/async
* LeRobot 异步推理博客: https://huggingface.co/blog/async-robot-inference
* LeIsaac 项目: https://github.com/LightwheelAI/leisaac
* LeIsaac 可用环境列表: `python scripts/environments/list_envs.py`
* 训练配置: `experiments/smolvla_so101_table_cleanup.yaml`
* 训练数据集: `Atticuxz/so101-table-cleanup`
* 评估流程文档: [eval.md](./eval.md)
* 完整训练 Pipeline: [so101_pipeline.md](./so101_pipeline.md)

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
