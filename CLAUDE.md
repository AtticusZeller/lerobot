# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

LeRobot is a HuggingFace library for state-of-the-art machine learning for real-world robotics in PyTorch. It provides tools for data collection, training, and deployment of robot policies.

This fork's root `README.md` is now a human-facing Chinese project page for **RL Token 仿真复现**（冻结 π0.5 + LIBERO 仿真 + 块级 TD3）。前期的 SO-101 真机阶段（SmolVLA / X-VLA / π0.5 桌面清理任务）已**归档到 `docs/archive/so101/`、`experiments/archive/so101/`、`media/archive/so101/`**，不再维护。主线设计文档是 `docs/rltoken_plan.md`，相关论文原文在 `docs/paper/`。

## Commands

### Installation

```bash
# Install with uv (recommended) for development
uv sync --extra "dev" --extra "test"

# Install with specific robot/policy extras
uv sync --extra "aloha" --extra "pusht"  # simulation environments
uv sync --extra "feetech"               # Feetech motors
uv sync --extra "pi"                    # Pi0 policy
```

### Code Quality

```bash
# Run all pre-commit checks (lint, format, type check, security)
pre-commit run --all-files

# Run ruff linter only
ruff check src/ --fix

# Run ruff formatter
ruff format src/

# Run mypy (only enforced on select modules: envs, configs, optim, model, cameras, motors, transport)
mypy src/lerobot/
```

### Tests

```bash
# Run full test suite
pytest tests -vv

# Run a single test file
pytest tests/test_datasets.py -sv

# Run with max failures threshold (as used in CI)
pytest tests -vv --maxfail=10

# Tests require git-lfs artifacts:
git lfs install && git lfs pull
```

### CLI Entry Points

The package exposes these commands after installation:

```bash
lerobot-train          # Train a policy
lerobot-eval           # Evaluate a trained policy
lerobot-record         # Record robot demonstrations
lerobot-replay         # Replay recorded episodes
lerobot-teleoperate    # Teleoperate a robot
lerobot-calibrate      # Calibrate robot motors
lerobot-find-cameras   # Detect connected cameras
lerobot-find-port      # Find robot serial port
lerobot-setup-motors   # Configure motors
lerobot-dataset-viz    # Visualize a dataset
lerobot-info           # Show dataset/model info
lerobot-edit-dataset   # Edit dataset episodes
```

## Architecture

### Configuration System ( `src/lerobot/configs/` )

Configs are dataclasses using [draccus](https://github.com/dlwh/draccus), which parses CLI args into nested dataclasses. Key classes:

* `TrainPipelineConfig` (`configs/train.py`) — top-level training config, composes `DatasetConfig`,  `PreTrainedConfig`,  `OptimizerConfig`, etc.
* `PreTrainedConfig` (`configs/policies.py`) — base for all policy configs; uses `draccus.ChoiceRegistry` for dynamic dispatch by policy name. Each policy registers itself with `@PreTrainedConfig.register_subclass("act")` etc.
* `DatasetConfig` (`configs/default.py`) — dataset repo ID, episodes subset, image transforms, video backend.
* `configs/parser.py` — draccus-based CLI parser with plugin discovery support.

### Policy System ( `src/lerobot/policies/` )

All policies inherit from:
* `PreTrainedConfig` — dataclass config registered in the draccus choice registry
* `PreTrainedPolicy(nn.Module)` — wraps HuggingFace Hub integration; implements `select_action()`,  `forward()`, and `get_optim_params()`

Available policies: `act` , `diffusion` , `tdmpc` , `vqbet` , `pi0` , `pi05` , `smolvla` , `groot` , `sarm` , `wall_x` , `xvla` , `sac` (RL), `rtc`

`policies/factory.py` — `make_policy()` instantiates a policy from config + dataset metadata, wiring input/output features automatically.

### Dataset System ( `src/lerobot/datasets/` )

* `LeRobotDataset` — PyTorch `Dataset` wrapping Parquet + video files, stored locally or on the HF Hub. Manages episodes, frames, and delta timestamps for temporal context.
* `LeRobotDatasetMetadata` — lightweight metadata-only view (no frames loaded).
* `MultiLeRobotDataset` — concatenates multiple datasets with a `dataset_index` key.
* `StreamingLeRobotDataset` — streaming version for large remote datasets.
* `datasets/factory.py` — `make_dataset()` creates the right dataset type from config.

Dataset storage layout: `data/chunk-XXX/episode_XXXXXXXX.parquet` for frames, `videos/chunk-XXX/<camera_key>/episode_XXXXXXXX.mp4` for video.

### Hardware Abstractions

* `motors/` — `MotorsBus` base class with implementations for `dynamixel`,  `feetech`,  `damiao`,  `robstride` motor protocols.
* `cameras/` — `Camera` base class with `opencv`,  `realsense`,  `zmq`, and `reachy2_camera` backends.
* `robots/` — `Robot` base class; robot-specific implementations under subdirectories (`koch_follower`,  `lekiwi`,  `so_follower`, etc.).
* `teleoperators/` — teleoperation devices.

### Training Pipeline ( `src/lerobot/scripts/lerobot_train.py` )

Uses HuggingFace `accelerate` for distributed training. Training loop: dataset → `EpisodeAwareSampler` → policy forward → optimizer step → optional eval via `lerobot_eval.py` . Checkpoints saved as safetensors + config JSON.

### Async Inference ( `src/lerobot/async_inference/` + `transport/` )

gRPC-based client-server architecture for decoupled policy inference. `transport/` contains protobuf definitions ( `services.proto` ), generated stubs ( `services_pb2*.py` ), and the `PolicyServer` / `RobotClient` pair. Requires `grpcio-dep` extra.

### Processor Pipeline ( `src/lerobot/processor/` )

Pre/post-processing pipeline that runs between raw observations and policy inputs, and between policy outputs and robot commands. Handles normalization, image resizing, tokenization for VLM-based policies.

## Key Conventions

* **`draccus` CLI pattern**: main scripts use `@parser.wrap()` decorator on `main(cfg: SomeConfig)`. CLI args map directly to config field paths (e.g.,  `--dataset.repo_id=...`).
* **Policy registration**: New policies must register their config with `@PreTrainedConfig.register_subclass("name")` and implement `PreTrainedPolicy`.
* **Feature types** (`configs/types.py`): `FeatureType.VISUAL`,  `STATE`,  `ACTION`,  `ENV` — used to wire dataset features to policy inputs/outputs.
* **mypy enforcement**: Only enforced on `envs`,  `configs`,  `optim`,  `model`,  `cameras`,  `motors`,  `transport` modules. Other modules have `ignore_errors = true`.
* **Line length**: 110 characters (ruff).
* **Python**: ≥3.12 required; use `pyupgrade --py312-plus` conventions (e.g.,  `X | Y` union types).

## Python / Deep Learning Code Style

### Comments
- Tensor ops: annotate shape inline as `# [B, T, C] -> [B, T, num_heads, head_dim]`
- Use `[B, C, H, W]` notation. B=batch, T=time/sequence, C=channels.
- Comment the WHY of design choices (loss function selection, freezing strategy),
  not the WHAT (don't restate code in English).
- No boilerplate comments like `# forward pass`, `# import libraries`.

### Docstrings
- Google style. Include tensor shapes in Args/Returns.
- Training scripts and one-off experiments: minimal docstrings, focus on inline comments.
- Reusable modules (model definitions, dataset classes): full docstrings.

### Naming
- Models/modules: PascalCase (`FlowMatchingHead`, `SO101Dataset`)
- Config keys: snake_case, match HuggingFace / LeRobot conventions
- Dimensions: spell out in variable names when ambiguous (`action_dim`, `num_heads`)

### Structure
- One model per file. Dataset class colocated with its transform logic.
- Training script: config at top as dataclass or dict, not scattered magic numbers.
- Wandb logging: group related metrics (`loss/flow`, `loss/total`, `lr/schedule`)

## Docs

`docs/` 下部分文件属于项目维护者的个人文档系统：

* `docs/source/` — 官方文档站源码（`.mdx` 文件），由 HuggingFace 官方维护，不要修改
* `docs/README.md` — 官方文档站入口
* `docs/rltoken_plan.md` — **主线设计文档**（V3），RL Token + π0.5 + LIBERO 仿真复现的完整实验设计、技术架构、开发路线图（冻结，不追加历史）
* `docs/plan.md` — **执行日志**，记录每个里程碑实际产出 / 命令 / 路径 / 待办；增量更新
* `plan_user.md` — **用户运行清单**，记录单任务 Stage 1/2 验证命令、期望产物和排障顺序
* `docs/paper/` — 原始论文 markdown：RL Token、π0、π0.5、π0.6、π_RL（实现时优先查阅）
* `docs/archive/so101/` — 前期 SO-101 真机阶段归档（SmolVLA / X-VLA / π0.5 桌面清理），**不再维护**，仅作历史参考

**协作约定**：被问及 docs/ 下的文档时，优先读 `rltoken_plan.md`、`paper/` 下论文原文，进度问题读 `plan.md`。`docs/archive/so101/` 仅在用户明确询问真机历史时引用；`docs/source/` 下的 `.mdx` 参考其内容但不要修改（官方文档）。

**输出路径约定**：设 `LEROBOT_OUTPUT_ROOT=<前缀>`（AutoDL 上通常 `/root/autodl-tmp/outputs`）后，`dev.sh` 的四个子命令在用户未显式传 `--output_dir` / `--save_dir` 时都会自动落到该前缀下：

| 子命令 | 默认输出（`LEROBOT_OUTPUT_ROOT` 未设） | 注入后 |
|---|---|---|
| `eval_baseline` | `outputs/eval/<日期>/<时间>_<job_name>/` | `<前缀>/eval/...`（由 `EvalPipelineConfig` 直接读环境变量） |
| `train_token` | `outputs/rltoken/encoder_decoder/` | `<前缀>/rltoken/encoder_decoder/`（`dev.sh` 注入 `--output_dir`） |
| `train_online` | `outputs/rltoken/online/` | `<前缀>/rltoken/online/`（`dev.sh` 注入 `--save_dir`） |
| `eval_throughput` | `outputs/baseline/` | `<前缀>/baseline/`（`dev.sh` 注入 `--output_dir`） |

显式传 `--output_dir=PATH` / `--save_dir=PATH` 时优先使用 PATH，不再注入。

**已知上游数据 bug**：`HuggingFaceVLA/libero` v3.0 的 `meta/episodes/*.parquet` 的 `data/file_index` 列与实际数据 parquet 布局不一致（[community issue #5](https://huggingface.co/datasets/HuggingFaceVLA/libero/discussions/5)）。结果是按 `episodes=[...]` 单 task 过滤会下到错文件、`Dataset.from_parquet` 返回 0 行。修复在 `experiments/dataset_overrides/HuggingFaceVLA_libero_v3.0_episodes_fix.json`（committed 1693 项映射），运行时由 `src/lerobot/rltoken/dataset_repair.py` 在 `LeRobotDatasetMetadata` 实例化后改写本地 parquet，原文件备份为 `*.broken.bak`。新增 / 修改的 hook 已挂到 `train_token._resolve_episode_filter`；如果上游推新 revision，重跑 `scripts/generate_libero_episode_fix.py --revision <new>` 重建 JSON。

## RL Token 多分支工作流（并行设计 → 单独细查 → 精修 → 合并）

为了在用户验证某阶段的同时不阻塞下一阶段编码，本项目采用 **git worktree + 多分支并行** 模式。当前活跃分支：

* `rltoken` (主) — 基线 eval、Stage 1 RL Token 训练、Stage 2 块级 TD3 核心代码均已在 `src/lerobot/rltoken/`。当前重点是按 `plan_user.md` 跑通 LIBERO-Spatial task 0；用户验证前避免继续扩展新功能。
* `rltoken_p2` (副) — 历史副工作树分支；Stage 2 核心已迁回主分支后，不再作为新功能默认落点。

**Workflow 模式**（这是项目级的 Claude Code 协作约定，不仅限于 RL Token 阶段）：

1. **分开设计**：用户验证阶段 N 时，新 worktree 上写阶段 N+1 骨架（仅新增文件、不动已 commit 文件）
2. **逐个细查**：阶段 N 验证后用户可能改 N 的实现；副分支用 `TODO(reconcile):` 标记需要回头同步的位置
3. **精修各自**：两边稳定后各自再精修（性能、文档、测试覆盖）
4. **最终合并**：`git merge --no-ff <副分支>` 回主，冲突通常仅在 `docs/plan.md`（人工合并进度状态）
5. **清理 worktree**：`git worktree remove ../lerobot-<副分支>` + `git branch -d <副分支>`

**新建副 worktree 的命令模板**：

```bash
git worktree add -b <副分支名> ../lerobot-<副分支名> <主分支 HEAD SHA>
cd ../lerobot-<副分支名>
# 此后所有编辑都在新工作树绝对路径下进行
```

**接手已有副 worktree 时必做**：先 `git worktree list` 看现有 worktree；再各自 `git log --oneline -5` 对齐两边 HEAD；最后读两边 `docs/plan.md` 的差异（`diff <主>/docs/plan.md <副>/docs/plan.md`）。

## References

* RL Token 原论文: `docs/paper/RL Token: Bootstrapping Online RL with Vision-Language-Action Models.md`
* π0.5 原论文: `docs/paper/pi_0.5 a Vision-Language-Action Model with Open-World Generalization.md`
* 代码参考: [rlt-openpi](https://github.com/yknxh/rlt-openpi)（主要参考）、`~/DevSpace/RLinf/`（LIBERO env + OpenPI 集成参考）
* HuggingFace LeRobot 官方文档: https://huggingface.co/docs/lerobot/index
