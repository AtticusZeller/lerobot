# `lerobot.rltoken`

RL Token 仿真复现模块。完整实验设计与开发路线见 [`docs/rltoken_plan.md`](../../../docs/rltoken_plan.md)。

| 文件 | 阶段 | 状态 |
| --- | --- | --- |
| `train_token.py` | 阶段一：RL Token 编码器-解码器离线训练 | 占位（NotImplementedError） |
| `train_online.py` | 阶段二：冻结编码器，块级 TD3 在线训练 | 占位 |
| `eval_throughput.py` | 评估吞吐率 / 平均步数 / 成功率 | 占位 |

复用 LeRobot 现有模块：

- `lerobot.envs.libero` — LIBERO gym 封装
- `lerobot.rl.buffer` — replay buffer
- `lerobot.rl.learner` — actor-critic learner 骨架
- `lerobot.policies.pi05` — π0.5 主干（冻结）

参考代码：[rlt-openpi](https://github.com/yknxh/rlt-openpi)（主要参考）、[RLinf](https://github.com/) `rlinf/envs/libero` + `rlinf/workers/actor`（设计参考）。
