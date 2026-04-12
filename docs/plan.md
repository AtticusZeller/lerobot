# Development Plan

> 每日进展与计划记录。

---

## 2026-04-12 Daily Sync

### Completed Today

- [x] **GPU Dev Container 环境搭建**
  - 新建独立仓库 [gpu-devcontainer](https://github.com/AtticusZeller/gpu-devcontainer)：可复用 CUDA 12.4 + Ubuntu 22.04 镜像，适配智新云等裸机 GPU 服务器
  - 新建私有仓库 [dotfiles-private](https://github.com/AtticusZeller/dotfiles-private)：统一管理认证 token 和 shell 配置
  - `.devcontainer/` 已复制至当前 lerobot 仓库，VS Code 可直接 "Reopen in Container"

- [x] **镜像内容（8 层 Dockerfile）**
  - CUDA 12.4 devel + Ubuntu 22.04 基础层
  - Zsh + Oh My Zsh + Powerlevel10k + 插件（autosuggestions / syntax-highlighting / zsh-bat）
  - Tmux + Oh My Tmux
  - UV 0.11 + Python 3.12（UV 托管，symlink 至 `/usr/local/bin`）
  - UV 全局工具：nvitop、wandb、huggingface-hub（`hf` CLI）
  - Node.js 24（nvm，symlink 至 `/usr/local/bin`）+ Claude Code + cc-switch
  - GitHub CLI（gh）
  - JetBrainsMono Nerd Font
  - 镜像大小：8.49 GB

- [x] **Dotfiles 部署流程**
  - `sync.sh`：笔记本端一键收集 HF token、gh token（从 keyring 导出）、wandb .netrc、Claude Code 配置、tmux/git 配置到私有仓库
  - `post-create.sh`：容器首次启动时 clone dotfiles-private → symlink 配置文件 → 迁移认证
  - `PAT_TOKEN` 通过 `devcontainer.json` 的 `${localEnv:PAT_TOKEN}` 从宿主机 shell 环境注入，宿主机 `.zshrc` 中 `export PAT_TOKEN=$GITHUB_TOKEN`

- [x] **认证验证结果（容器内）**
  - GitHub CLI：`GH_TOKEN` 环境变量方案（gh keyring 在容器中无 dbus，不可用）✅
  - HuggingFace：`~/.cache/huggingface/token` ✅（user: Atticuxz）
  - wandb：`~/.netrc` (api.wandb.ai) ✅（user: atticux）
  - Claude Code：2.1.101 ✅
  - GPU：RTX 4060 Laptop 8GB，`--gpus all` 验证通过 ✅

### Next Steps

- [ ] **云端 GPU 服务器首次部署测试**
  - 在智新云等裸机服务器上 pull 镜像，走完完整 post-create.sh 流程
  - 验证 `uv sync --extra dev` 安装 lerobot 依赖正常
  - 验证 `lerobot-train` 可以正常调用 GPU

---

## 2026-03-29 Daily Sync

### 需求背景

目前没有实机，但需要快速横向对比多种微调方案的效果。逐个在真机上部署评估太慢，计划用 **LeIsaac + IsaacLab 仿真环境** 替代实机，实现自动化批量评估，大幅加速迭代。

### 核心任务：仿真批量评估 Pipeline

**目标**: 建立"微调 → 仿真推理 → 自动评分"的闭环，快速筛选最优方案后再上实机验证。

#### 1. 仿真推理环境搭建
- [ ] 确认 LeIsaac SO-101 仿真环境可用（`LeIsaac-SO101-PickOrange-v0` 或自定义任务）
- [ ] 验证 LeRobot Policy Server + LeIsaac `policy_inference.py` + `lerobot-pi05` 的端到端流程
- [ ] 确认 `LeRobotServicePolicyClient` 对 pi05 action 维度的兼容性

#### 2. 仿真数据集准备
- [ ] 准备与仿真环境对应的训练数据集（仿真采集 or 已有仿真数据集）
- [ ] 确保数据集格式、camera key、action 维度与仿真环境一致

#### 3. 多方案微调对比
- [ ] 定义待对比的微调方案矩阵（初步规划）：
  - 微调方式: `train_expert_only` vs `LoRA` vs 全参微调
  - 模型: `pi05` vs 其他候选（如 `smolvla`）
  - 超参: steps / batch_size / learning_rate 等
- [ ] 每种方案生成对应的 YAML 配置（`experiments/` 目录）

#### 4. 自动化评估脚本
- [ ] 编写批量评估脚本：遍历多个 checkpoint → 启动 Policy Server → 在仿真中跑 N 个 episode → 收集成功率/评分
- [ ] 评分指标：任务完成率、子步骤得分（接近/抓取/运输/放置）、平均耗时
- [ ] 输出汇总表（CSV/JSON），方便横向对比

#### 5. 结果分析与实机验证
- [ ] 根据仿真评估结果选出 Top 2-3 方案
- [ ] 有实机后在真机上做最终验证（少量评估即可）

### Next Steps

- [ ] 搭建 LeIsaac 仿真推理环境，跑通 SO-101 + pi05 端到端
- [ ] 编写自动化评估脚本框架
- [ ] 准备第一批对比实验的 YAML 配置

---

## 2026-03-28 Daily Sync

### Completed Today

- [x] **确立实机评估方案**
  - 建立两阶段评估流程：训练指标预筛（wandb loss/grad_norm） → 真机评估（π0.5 评分 rubric）
  - 评分 rubric 参考 π0.5 论文：按子步骤打分（接近/抓取/运输/放置，每步 1 分）
  - 评估方案文档: [eval.md](./eval.md)

- [x] **建立推理部署文档**
  - 基于 LeRobot async inference（gRPC 架构）整理推理部署流程
  - 推理部署文档: [inference.md](./inference.md)
  - 架构: GPU 服务器（policy_server）← gRPC → 笔记本（robot_client）← USB → SO-101

- [x] **精简 pipeline 文档**
  - 将 Section 4 推理部署内容精简为摘要+指向 inference.md
  - 删除失效的 Action Plotting 相关内容
  - 更新 Checkpoint 选择流程指向 eval.md

- [x] **参考资料补充**
  - 将 Seeed Studio LeRobot 文档和 HF 官方文档链接添加到 CLAUDE.md

### Next Steps

- [ ] **实机评估执行**
  - 从训练好的 checkpoint 中选取 3–5 个候选（平台期前/中/后段）
  - 按 eval.md 流程在真机上逐一评估，每个 checkpoint 评估 10 次
  - 根据 π0.5 评分结果选择最佳 checkpoint

---

## 2026-03-26 Daily Sync

### Completed Today

- [x] **YAML 配置启动改造**
  - 修改 `src/lerobot/configs/parser.py`，新增 `--yaml_config` 参数支持
  - draccus 原生支持 YAML 加载 + CLI 覆盖，改动最小化（3 行）
  - 创建 `experiments/` 目录，放置实验 YAML 配置
  - 基线配置: `experiments/pi05_lora_so101_table_cleanup.yaml`（复刻 03-23 训练参数）
  - **用法**: `lerobot-train --yaml_config=experiments/pi05_lora_so101_table_cleanup.yaml`
  - **CLI 覆盖**: `lerobot-train --yaml_config=... --steps=8000 --batch_size=8`（CLI 优先于 YAML）

### Next Steps

- [x] **切换至 train_expert_only 方案训练** ✅
  - 经 Gemini 讨论确认: `train_expert_only=true` 效果最好，**暂不使用 LoRA**
  - 冻结 PaliGemma VLM (~2.7B)，仅训练 Action Expert (~300M，~5-10% 参数)
  - 配置文件: `experiments/pi05_expert_so101_table_cleanup.yaml`
  - 启动命令: `lerobot-train --yaml_config=experiments/pi05_expert_so101_table_cleanup.yaml`
  - 训练已完成，loss 曲线正常收敛

---

## 2026-03-24 Daily Sync

### Completed Today

- [x] **训练验证机制调研**
  - 调研范围: LeRobot eval 体系、`lerobot-eval`、Pi0.5 loss 计算、checkpoint 选择
  - **核心结论**:
    - SO-ARM101 **没有仿真环境**（`envs/` 下无对应 env），因此 `eval_freq=0`，训练全程只记录 Training Loss
    - **没有内置的自动 checkpoint 选择机制** — 需要手动从 wandb loss 曲线判断
    - **`lerobot-eval` 无法用于 SO-ARM101** — 依赖 `gym.vector.VectorEnv`，只能用于pusht/aloha 等有仿真器的任务
    - **离线 Action Plotting 不存在内置脚本** — 需要自己写，从数据集批量推理，对比预测动作 vs 真实动作
    - Pi0.5 Loss 是 Flow Matching MSE: `loss = ||v_t - v_theta(x_t, t)||²`，其中 `x_t = t·noise + (1-t)·action`，`v_t = noise - action`
  - **收敛判断**: loss 应在训练全程稳定下降，前 1000 步下降最快，末期趋于平缓则表示收敛
  - **过拟合判断**: 训练 loss 持续下降但 dataset 本身数据量小时（< 500 episodes），需要用真实机器人验证
  - 文档已整理至 `docs/what_is_lora.md`（LoRA 基础概念）

### Next Steps

- [ ] **Checkpoint 横向对比验证（实机）**
  - 训练时每隔 1000 步保存一个 checkpoint（`save_freq: 1000`）
  - 从 loss 平台期选取 3–5 个候选 checkpoint（前段 / 中段 / 后段各 1–2 个）
  - 在真机上逐一部署测试，按 π0.5 评分 rubric 打分
  - 评分标准与流程详见 [eval.md](./eval.md)

- [ ] **实机部署验证**
  - 使用 LeRobot async inference（`policy_server` + `robot_client`）在真实 SO-ARM101 上部署
  - 按 π0.5 评分 rubric 逐子步骤打分（接近 → 抓取 → 运输 → 放置，每步 1 分）
  - 每个 checkpoint 评估 10 次，穿插执行控制环境变化（参考 π0.5 论文）
  - 评分定义见 [eval.md](./eval.md)，推理部署见 [inference.md](./inference.md)

---

## 2026-03-23 Daily Sync

### Completed Today

- [x] **数据集准备与格式对齐**
  - 将社区数据集 `youliangtan/so101-table-cleanup`（v2.1）转换为 v3.0 格式
  - 修复 `total_frames` 与实际 parquet 行数不一致问题（47513 vs 46963）
  - 上传至 `Atticuxz/so101-table-cleanup`

- [x] **Quantile 统计量计算**
  - 运行 `src/lerobot/scripts/augment_dataset_quantile_stats.py` 为数据集补充 Pi0.5 所需的 q01/q10/q50/q90/q99 分位统计量
  - **注意**: 必须先将数据集下载到本地（`hf download --repo-type dataset --local-dir ...`）再运行脚本，否则触发 `get_safe_version` 的 `huggingface_hub` 版本兼容 bug

- [x] **Pi0.5 LoRA 微调训练启动并完成**
  - 在单卡环境下成功启动并完成训练
  - 关键配置:
    ```
    --peft.method_type=LORA --peft.r=16 --policy.dtype=bfloat16
    --policy.gradient_checkpointing=true --steps=5000 --batch_size=4
    ```
  - 可训练参数 1.29M / 总参数 4.14B（0.03%）
  - 修复了两个启动 bug:
    - `--policy.push_to_hub=false` 必须显式设置
    - HF 用户名大小写敏感，需用 `hf auth whoami` 确认

- [x] **Pipeline 文档建立**
  - 新建 `docs/pi05_so101_lora_pipeline.md`（964 行）
  - 覆盖: LoRA 配置、SO-ARM101 数据集对齐、训练命令、gRPC 部署方案
  - 补充: `train_expert_only` 说明、社区全参微调命令对比、数据量与 steps 对照表、常见报错修复

- [x] **Bug Journal 更新**: `docs/bug.md` 新增两条记录

---

### Next Steps

- [x] **验证训练指标** ✅
  - **Loss 计算方式**: Pi0.5 使用 Flow Matching MSE loss（`modeling_pi05.py:730-783`）
    - 对每个 timestep t: `x_t = t·noise + (1-t)·action`，`v_t = noise - action`
    - Loss = MSE( `v_t` , `v_theta(x_t, t)` )，在所有 action 维度和时间维度上求均值
  - **收敛信号**: wandb 中 `loss` 曲线应全程稳定下降
    - 前 ~1000 步: 下降最快，loss 快速从高位回落
    - 中期 ~1000-3000 步: 下降速度减缓，曲线趋于平滑
    - 末期 ~3000+ 步: 接近平台，若 loss 仍有下降则可继续训练
    - `loss_per_dim` 数组: 各 action 维度的 loss 均值，应大致均衡，若某一维度持续远高于其他维度则需排查
  - **参数比例合理性**: wandb 会自动记录 `num_learnable_params / num_total_params`，LoRA 应为 ~0.03%~1%
  - **注意**: 无仿真环境下，Training Loss 是**唯一**训练期间的验证指标，无法区分过拟合和欠拟合
  - **上次训练结论**: 5000 步 loss 曲线正常收敛，验证通过

- [x] **关节/电机顺序三向对齐验证** ✅
  确认以下三者完全一致，防止训练数据与推理时动作空间错位:
  1. LeRobot 代码中 `SOFollowerRobotConfig` 定义的电机顺序（`shoulder_pan → gripper`）
  2. 物理 SO-ARM101 实际接线/ID 顺序
  3. 数据集 `meta/info.json` 中 `observation.state` 的 `names` 字段标注顺序

  **结论**: 三者关系已确认:
  - **基座模型 (Pi0.5)** 对 state/action 维度顺序**不敏感** — 预训练在异构多机器人数据上，通过投影层和微调适配不同机器人，不绑定特定维度语义
  - **数据集** `observation.state` / `action` 向量维度顺序由录制时的机器人硬件决定，与 `so_follower.py` 中电机字典插入顺序一致
  - **唯一隐患点**: 实机标定时电机 ID 必须与 `so_follower.py:53-59` 中硬编码的 Motor ID 一致。若标定 ID 与代码不匹配，会导致 state 向量维度语义全错，但训练本身不会报错
  - 摄像头 key 顺序同理：录制和推理时摄像头名称必须一致
  - 文档已更新：见 `docs/pi05_so101_lora_pipeline.md` 新增"⚠️ 电机和摄像头命名必须与代码一致"节

- [x] **完善微调 Pipeline 文档** ✅
  - LoRA 基础概念已整理至 `docs/what_is_lora.md`
  - Pi0.5 训练验证机制（wandb 曲线解读、收敛判断、过拟合识别）待补充到 `docs/pi05_so101_lora_pipeline.md`

- [x] **Checkpoint 使用方式整理** ✅
  - `factory.py:493-514` 的 `use_peft` 分支会自动从 `adapter_config.json` 读取 base model 并加载 adapter
  - `policy_server.py:152` 的 `from_pretrained()` 直接传入 checkpoint 路径即可，无需额外 `use_peft` 参数
  - 已补充到 `docs/pi05_so101_lora_pipeline.md`

- [ ] **实机部署验证**
  - 使用 `lerobot.async_inference.policy_server` + `robot_client` 在真实 SO-ARM101 上部署 checkpoint
  - 按 π0.5 评分 rubric 进行真机评估（见 [eval.md](./eval.md)）

---

### Notes / Blockers

- **训练有效性已验证**: train_expert_only 训练 loss 曲线正常收敛，Pi0.5 Flow Matching MSE 形态符合预期
- **剩余验证缺口**: 没有仿真环境的情况下，只能用真机测试来最终验证。wandb Training Loss 是必要但不充分条件
- **关节顺序对齐已确认**: 关键约束是**实机标定 ID 必须与代码一致**，其余由 LeRobot 框架自动保证
