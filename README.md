# RL Token 仿真复现：基于冻结 π0.5 的残差强化学习微调

在 **LIBERO 仿真基准**上复现并扩展 Physical Intelligence 的 **RL Token** 方法（原文仅在闭源 π0.6 真机验证）。冻结开源 `lerobot/pi05_libero` 主干，仅训练轻量 RL Token 编码器 + 块级 TD3 actor-critic，目标是在维持 ≥97% 监督微调成功率的前提下，**显著降低任务完成步数 / 提升吞吐率**。

完整实验设计见 [`docs/rltoken_plan.md`](docs/rltoken_plan.md)。

## 研究问题

> 在监督微调基线已经很强（≈97%）的常见操作任务上，RL Token 的**残差编辑**能否显著提升任务执行效率（吞吐率），同时维持高成功率？

## 技术架构

| 模块 | 内容 |
| --- | --- |
| 冻结主干 | `lerobot/pi05_libero`（π0.5 在 LIBERO 上的官方 SFT 检查点） |
| 阶段一（离线） | RL Token 编码器-解码器（2048D，per-task，~5K 步） |
| 阶段二（在线） | 块级 TD3：actor ~170K 参数、双 Q 评论家、BC 正则 β=0.5、参考动作 50% 丢弃 |
| 仿真环境 | LIBERO（Spatial / Object / Goal / Libero-10），奖励 = `env.check_success()` 稀疏二值 |
| 评估指标 | 吞吐率（主）+ 成功率（辅），详见 `docs/rltoken_plan.md` §三、表 1-3 |

## 代码组织

| 路径 | 用途 |
| --- | --- |
| `src/lerobot/rltoken/` | RL Token 编码器/解码器、TD3 actor-critic、训练 entry（本项目新增） |
| `src/lerobot/envs/libero.py` | LIBERO gym 封装（LeRobot 自带，原样复用） |
| `src/lerobot/policies/pi05/` | π0.5 实现（不动） |
| `experiments/rltoken_pi05_libero.yaml` | 训练配置 |
| `docs/rltoken_plan.md` | 实验设计 V3（主线文档） |
| `plan_user.md` | 单任务 Stage 1/2 用户运行清单 |
| `docs/paper/` | RL Token / π_RL / π0.5 / π0.6 论文原文 |
| `docs/archive/so101/` | 前期 SO-101 真机阶段归档（不再维护） |

## 快速开始

```bash
uv sync --extra "pi"                    # 安装 π0.5 依赖
LIBERO_SUITE=libero_spatial bash dev.sh eval_baseline --env.task_ids='[0]'
LIBERO_SUITE=libero_spatial bash dev.sh train_token --dataset.task_index=0 --steps=5000
LIBERO_SUITE=libero_spatial bash dev.sh train_online --task_index=0 --rl_token_checkpoint PATH
bash dev.sh eval_throughput --ckpt PATH # 评估吞吐率
```

## 参考

- **原论文**：[`docs/paper/RL Token: Bootstrapping Online RL with Vision-Language-Action Models.md`](docs/paper/RL%20Token:%20Bootstrapping%20Online%20RL%20with%20Vision-Language-Action%20Models.md)
- **π0.5 原论文**：[`docs/paper/pi_0.5 a Vision-Language-Action Model with Open-World Generalization.md`](docs/paper/pi_0.5%20a%20Vision-Language-Action%20Model%20with%20Open-World%20Generalization.md)
- **代码参考**：[rlt-openpi](https://github.com/yknxh/rlt-openpi)（plan 主要参考）、[RLinf](https://github.com/) （LIBERO env + OpenPI 集成参考）
- **上游框架**：[HuggingFace LeRobot](https://huggingface.co/docs/lerobot/index)
