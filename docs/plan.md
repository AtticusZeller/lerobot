# Development Plan

> 每日进展与计划记录。

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

- [ ] **验证训练指标**
  - 明确 wandb 中应关注的关键曲线（训练 loss 收敛趋势、`num_learnable_params` vs `num_total_params` 比例合理性）
  - 确认 Pi0.5 的 loss 计算方式（flow matching MSE，`modeling_pi05.py:1248-1283`）对应什么样的收敛形态

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

- [ ] **完善微调 Pipeline 文档**
  - 补充"训练结果验证"章节（如何从 wandb loss 曲线判断是否收敛、过拟合信号识别）

- [ ] **Checkpoint 使用方式整理**
  - 明确训练产出的 checkpoint 目录结构: `outputs/.../checkpoints/last/pretrained_model/`
  - LoRA adapter 权重文件位置: `adapter_model.safetensors` + `adapter_config.json`
  - 加载接口: `make_policy` 中 `cfg.use_peft=True` 分支（`factory.py:493-514`）
  - async inference 部署时 `--pretrained_name_or_path` 指向该路径的完整流程
  - 整理进 `docs/pi05_so101_lora_pipeline.md`

- [ ] **实机部署验证**
  - 使用 `lerobot.async_inference.policy_server` + `robot_client` 在真实 SO-ARM101 上跑训练好的 LoRA checkpoint
  - 验证动作输出是否合理

---

### Notes / Blockers

- 训练有效性尚未验证——当前只确认了训练流程跑通，loss 曲线是否收敛、模型行为是否正确仍待核实

- 关节顺序对齐已确认：关键约束是**实机标定 ID 必须与代码一致**，其余由 LeRobot 框架自动保证
