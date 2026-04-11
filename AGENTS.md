# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

LeRobot is a HuggingFace library for state-of-the-art machine learning for real-world robotics in PyTorch. It provides tools for data collection, training, and deployment of robot policies.

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

## Docs

`docs/` 下部分文件属于项目维护者的个人文档系统：

* `docs/source/` — 官方文档站源码（`.mdx` 文件），由 HuggingFace 官方维护
* `docs/README.md` — 官方文档站入口
* `docs/so101_pipeline.md` — **主文档**，SO-ARM101 微调完整 Pipeline（通用），模型子文档：`so101_pi05.md` / `so101_smolvla.md`
* `docs/plan.md` — 日常开发计划与任务追踪
* `docs/bug.md` — 开发过程中遇到的 bug 与解决方案记录
* `docs/eval.md` — 实机评估 SOP（π0.5 评分 rubric）
* `docs/inference.md` — gRPC 异步推理部署文档

**协作约定**：被问及 docs/ 下的文档时，优先读 `so101_pipeline.md` / `so101_pi05.md` / `so101_smolvla.md` / `plan.md` / `bug.md` / `eval.md` / `inference.md`。`docs/source/` 下的 `.mdx` 参考其内容但不要修改（官方文档）。

## References

* Seeed Studio LeRobot SO-ARM101 文档: https://wiki.seeedstudio.com/lerobot_so100m_new/#train-and-evaluate
* HuggingFace LeRobot 官方文档: https://huggingface.co/docs/lerobot/index
