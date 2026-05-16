# `lerobot.rltoken`

RL Token 仿真复现模块。完整实验设计与开发路线见 [`docs/rltoken_plan.md`](../../../docs/rltoken_plan.md)。

| 文件 | 阶段 | 状态 |
| --- | --- | --- |
| `train_token.py` | 阶段一：RL Token 编码器-解码器离线训练 | 已实现（per-task，2048D） |
| `train_online.py` | 阶段二：冻结编码器，块级 TD3 在线训练 | 已实现（单 LIBERO task） |
| `eval_throughput.py` | 评估吞吐率 / 平均步数 / 成功率 | 已实现 |

复用 LeRobot 现有模块：

- `lerobot.envs.libero` — LIBERO gym 封装
- `lerobot.policies.pi05` — π0.5 主干（冻结）
- `lerobot.processor` — LIBERO 与 π0.5 pre/post-processing

本目录直接迁移 rlt-openpi 的 RL Token / TD3 核心 PyTorch 代码，并用 `adapter.py` 与
`libero_chunk_env.py` 替换其 OpenPI/JAX + Franka/DROID 边界。
