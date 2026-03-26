# Development Plan

> 每日进展与计划记录。

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

- [ ] **切换至 train_expert_only 方案训练**
  - 经 Gemini 讨论确认: `train_expert_only=true` 效果最好，**暂不使用 LoRA**
  - 冻结 PaliGemma VLM (~2.7B)，仅训练 Action Expert (~300M，~5-10% 参数)
  - 配置文件: `experiments/pi05_expert_so101_table_cleanup.yaml`
  - 启动命令: `lerobot-train --yaml_config=experiments/pi05_expert_so101_table_cleanup.yaml`
  - 关注 wandb loss 曲线收敛情况，与之前 LoRA 实验对比

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

- [ ] **离线 Action Plotting 脚本开发**（待做）
  - 从数据集批量加载 batch，用 `model.select_action()` 或 `model.forward()` 推理
  - 对比: 预测动作轨迹 vs 数据集 ground-truth 动作轨迹（matplotlib 多维度折线图）
  - 可用于 checkpoint 之间的横向对比（不同 step 的 checkpoint 输出差异）
  - 脚本路径建议: `scripts/eval_offline_action_plot.py`

- [ ] **Checkpoint 横向对比验证**
  - 选出 2-3 个有代表性的 checkpoint（如: 训练前期、中期、最末期）
  - 用上述离线 Action Plotting 脚本对比不同 checkpoint 的动作输出
  - 选出动作轨迹最接近数据集动作分布的 checkpoint 作为最终选择

- [ ] **实机部署验证**
  - 使用 `policy_server.py` + `robot_client` 在真实 SO-ARM101 上运行 LoRA checkpoint
  - 验证动作输出是否平滑、合理

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

- [ ] **Checkpoint 使用方式整理**
  - `factory.py:493-514` 的 `use_peft` 分支会自动从 `adapter_config.json` 读取 base model 并加载 adapter
  - `policy_server.py:152` 的 `from_pretrained()` 直接传入 checkpoint 路径即可，无需额外 `use_peft` 参数
  - 待补充到 `docs/pi05_so101_lora_pipeline.md`

- [ ] **实机部署验证**
  - 使用 `lerobot.async_inference.policy_server` + `robot_client` 在真实 SO-ARM101 上跑训练好的 LoRA checkpoint
  - 验证动作输出是否合理

---

### Notes / Blockers

- **训练有效性已验证**: 5000 步训练 loss 曲线正常收敛，Pi0.5 Flow Matching MSE 形态符合预期
- **剩余验证缺口**: 没有仿真环境的情况下，只能用离线 Action Plotting + 实机测试来最终验证。wandb Training Loss 是必要但不充分条件
- **关节顺序对齐已确认**: 关键约束是**实机标定 ID 必须与代码一致**，其余由 LeRobot 框架自动保证
