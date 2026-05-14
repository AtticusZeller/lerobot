"""RL Token 仿真复现模块。

包含三阶段实现：
- ``train_token``     — 阶段一：RL Token 编码器-解码器离线训练
- ``train_online``    — 阶段二：冻结编码器，块级 TD3 在线训练
- ``eval_throughput`` — 评估吞吐率（平均步数 / 吞吐率 / 成功率）

设计文档：``docs/rltoken_plan.md``
"""
