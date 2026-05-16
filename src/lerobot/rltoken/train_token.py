"""Stage 1: train the RL Token encoder-decoder on LIBERO demos."""

from __future__ import annotations

import logging
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from pprint import pformat
from typing import Any

import draccus
import torch
import yaml
from safetensors.torch import save_file
from torch import Tensor
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.rltoken.rl_token import VLM_HIDDEN_DIM, RLTokenModel, extract_vlm_embeddings
from lerobot.utils.device_utils import get_safe_torch_device
from lerobot.utils.import_utils import register_third_party_plugins
from lerobot.utils.random_utils import set_seed
from lerobot.utils.utils import init_logging


@dataclass
class DatasetCfg:
    repo_id: str = "lerobot/libero"
    batch_size: int = 32
    num_workers: int = 4
    episodes: list[int] | None = None
    task_index: int | None = None


@dataclass
class TokenArchCfg:
    embedding_dim: int = VLM_HIDDEN_DIM
    encoder_layers: int = 2
    encoder_heads: int = 8
    decoder_layers: int = 2
    decoder_heads: int = 8


@dataclass
class Stage1Cfg:
    num_train_steps: int = 5000
    learning_rate: float = 1.0e-4
    weight_decay: float = 1.0e-5
    warmup_steps: int = 500
    max_grad_norm: float = 1.0
    log_every: int = 50
    ckpt_every: int = 1000


@dataclass
class TrainTokenConfig:
    pretrained: str = "lerobot/pi05_libero"
    output_dir: str = "outputs/rltoken/encoder_decoder"
    yaml_config: str | None = None
    suite: str | None = None
    steps: int | None = None
    seed: int = 42
    device: str | None = None
    use_amp: bool = False
    wandb_project: str | None = None
    wandb_enabled: bool = True
    dataset: DatasetCfg = field(default_factory=DatasetCfg)
    rltoken: TokenArchCfg = field(default_factory=TokenArchCfg)
    stage1: Stage1Cfg = field(default_factory=Stage1Cfg)


def _cli_has(path: str) -> bool:
    flag = f"--{path}"
    return any(arg == flag or arg.startswith(f"{flag}=") for arg in sys.argv[1:])


def _set_if_not_cli(cfg: Any, path: str, value: Any) -> None:
    if value is None or _cli_has(path):
        return
    obj = cfg
    parts = path.split(".")
    for part in parts[:-1]:
        obj = getattr(obj, part)
    setattr(obj, parts[-1], value)


def _merge_yaml_into_cfg(cfg: TrainTokenConfig) -> None:
    if cfg.yaml_config is None:
        return
    path = Path(cfg.yaml_config)
    if not path.exists():
        logging.warning("yaml_config %s does not exist; skipping YAML merge.", path)
        return

    data = yaml.safe_load(path.read_text()) or {}
    _set_if_not_cli(cfg, "pretrained", data.get("pretrained") or data.get("pretrained_pi05"))

    dataset = data.get("dataset", {})
    for key in ("repo_id", "batch_size", "num_workers", "task_index"):
        _set_if_not_cli(cfg, f"dataset.{key}", dataset.get(key))

    arch = data.get("rltoken", {})
    for key in ("embedding_dim", "encoder_layers", "encoder_heads", "decoder_layers", "decoder_heads"):
        _set_if_not_cli(cfg, f"rltoken.{key}", arch.get(key))

    stage1 = data.get("stage1", {})
    legacy_training = data.get("training", {})
    _set_if_not_cli(
        cfg,
        "stage1.num_train_steps",
        stage1.get("num_train_steps", legacy_training.get("stage1_steps")),
    )
    for key in ("learning_rate", "weight_decay", "warmup_steps", "max_grad_norm", "log_every", "ckpt_every"):
        _set_if_not_cli(cfg, f"stage1.{key}", stage1.get(key))

    wandb = data.get("wandb", {})
    _set_if_not_cli(cfg, "wandb_project", wandb.get("project"))
    _set_if_not_cli(cfg, "wandb_enabled", wandb.get("enabled"))


def _task_name_for_index(tasks, task_index: int) -> str:
    if "task_index" in tasks.columns:
        rows = tasks[tasks["task_index"] == task_index]
        if len(rows) > 0:
            return str(rows.index[0])
    if task_index < 0 or task_index >= len(tasks):
        raise ValueError(f"task_index {task_index} out of range [0, {len(tasks) - 1}].")
    return str(tasks.iloc[task_index].name)


def _episode_index(episode: dict[str, Any], fallback: int) -> int:
    return int(episode.get("episode_index", fallback))


def _episode_matches_task(episode: dict[str, Any], task_index: int, task_name: str) -> bool:
    if "task_index" in episode:
        value = episode["task_index"]
        if isinstance(value, Tensor):
            value = value.item()
        return int(value) == task_index

    episode_tasks = episode.get("tasks", [])
    if isinstance(episode_tasks, str):
        episode_tasks = [episode_tasks]
    return task_name in episode_tasks


def _resolve_episode_filter(ds_cfg: DatasetCfg) -> tuple[list[int] | None, str | None]:
    if ds_cfg.task_index is None:
        return ds_cfg.episodes, None

    meta = LeRobotDatasetMetadata(ds_cfg.repo_id)
    task_name = _task_name_for_index(meta.tasks, ds_cfg.task_index)
    task_episodes = [
        _episode_index(ep, ep_idx)
        for ep_idx, ep in enumerate(meta.episodes)
        if _episode_matches_task(ep, ds_cfg.task_index, task_name)
    ]
    if ds_cfg.episodes is not None:
        allowed = set(ds_cfg.episodes)
        task_episodes = [ep for ep in task_episodes if ep in allowed]
    if not task_episodes:
        raise ValueError(f"No episodes found for task_index={ds_cfg.task_index} ({task_name!r}).")
    return task_episodes, task_name


def _build_policy(cfg: TrainTokenConfig, ds_meta):
    policy_cfg = PreTrainedConfig.from_pretrained(cfg.pretrained)
    policy_cfg.pretrained_path = cfg.pretrained
    if hasattr(policy_cfg, "freeze_vision_encoder"):
        policy_cfg.freeze_vision_encoder = True
    if hasattr(policy_cfg, "train_expert_only"):
        policy_cfg.train_expert_only = True
    if cfg.device is not None:
        policy_cfg.device = cfg.device
    policy_cfg.use_amp = cfg.use_amp

    policy = make_policy(cfg=policy_cfg, ds_meta=ds_meta)
    policy.eval()
    for param in policy.parameters():
        param.requires_grad_(False)
    return policy, policy_cfg


def _collate_then_preprocess(items: list[dict[str, Any]], preprocessor) -> dict[str, Tensor]:
    batch: dict[str, Any] = {}
    for key in items[0]:
        values = [item[key] for item in items]
        batch[key] = torch.stack(values, dim=0) if isinstance(values[0], Tensor) else values
    return preprocessor(batch)


def _build_dataloader(ds_cfg: DatasetCfg, episodes: list[int] | None, preprocessor) -> DataLoader:
    dataset = LeRobotDataset(ds_cfg.repo_id, episodes=episodes)
    return DataLoader(
        dataset,
        batch_size=ds_cfg.batch_size,
        shuffle=True,
        num_workers=ds_cfg.num_workers,
        collate_fn=lambda items: _collate_then_preprocess(items, preprocessor),
        drop_last=True,
        persistent_workers=ds_cfg.num_workers > 0,
    )


def _checkpoint_dir(cfg: TrainTokenConfig) -> Path:
    out_dir = Path(cfg.output_dir)
    if cfg.suite:
        out_dir = out_dir / cfg.suite
    if cfg.dataset.task_index is not None:
        out_dir = out_dir / f"task_{cfg.dataset.task_index:02d}"
    return out_dir


def _save_checkpoint(model: RLTokenModel, cfg: TrainTokenConfig, step: int) -> Path:
    out_dir = _checkpoint_dir(cfg)
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / f"step_{step:06d}.safetensors"
    metadata = {
        "embedding_dim": str(cfg.rltoken.embedding_dim),
        "encoder_layers": str(cfg.rltoken.encoder_layers),
        "encoder_heads": str(cfg.rltoken.encoder_heads),
        "decoder_layers": str(cfg.rltoken.decoder_layers),
        "decoder_heads": str(cfg.rltoken.decoder_heads),
        "step": str(step),
    }
    save_file({k: v.detach().cpu() for k, v in model.state_dict().items()}, str(ckpt_path), metadata=metadata)
    return ckpt_path


@draccus.wrap()
def main(cfg: TrainTokenConfig) -> None:
    init_logging()
    register_third_party_plugins()
    _merge_yaml_into_cfg(cfg)
    if cfg.steps is not None:
        cfg.stage1.num_train_steps = cfg.steps

    device = get_safe_torch_device(cfg.device) if cfg.device else get_safe_torch_device("cuda")
    cfg.device = str(device)
    set_seed(cfg.seed)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True

    episodes, task_name = _resolve_episode_filter(cfg.dataset)
    logging.info("Stage 1 config:\n%s", pformat(asdict(cfg)))
    if task_name is not None:
        logging.info("Training task_index=%d task=%r episodes=%d", cfg.dataset.task_index, task_name, len(episodes))

    dataset_for_meta = LeRobotDataset(cfg.dataset.repo_id, episodes=episodes)
    ds_meta = dataset_for_meta.meta

    logging.info("Loading frozen π0.5 from %s", cfg.pretrained)
    policy, policy_cfg = _build_policy(cfg, ds_meta)
    preprocessor, _ = make_pre_post_processors(
        policy_cfg=policy_cfg,
        pretrained_path=cfg.pretrained,
        dataset_stats=ds_meta.stats,
        preprocessor_overrides={"device_processor": {"device": str(device)}},
    )
    loader = _build_dataloader(cfg.dataset, episodes, preprocessor)

    model = RLTokenModel(
        embedding_dim=cfg.rltoken.embedding_dim,
        encoder_layers=cfg.rltoken.encoder_layers,
        encoder_heads=cfg.rltoken.encoder_heads,
        decoder_layers=cfg.rltoken.decoder_layers,
        decoder_heads=cfg.rltoken.decoder_heads,
    ).to(device)
    num_params = sum(param.numel() for param in model.parameters())
    logging.info("RL Token trainable params: %.2fM", num_params / 1e6)

    optimizer = AdamW(model.parameters(), lr=cfg.stage1.learning_rate, weight_decay=cfg.stage1.weight_decay)
    warmup_steps = cfg.stage1.warmup_steps
    scheduler = LambdaLR(
        optimizer,
        lr_lambda=lambda step: 1.0 if warmup_steps <= 0 else min(1.0, (step + 1) / warmup_steps),
    )

    wandb_run = None
    if cfg.wandb_project and cfg.wandb_enabled:
        try:
            import wandb

            wandb_run = wandb.init(project=cfg.wandb_project, config=asdict(cfg))
        except Exception:
            logging.warning("wandb init failed; continuing without W&B.", exc_info=True)

    step = 0
    loss_ema: float | None = None
    start = time.time()
    data_iter = iter(loader)
    while step < cfg.stage1.num_train_steps:
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            batch = next(data_iter)

        z_vis, pad_vis = extract_vlm_embeddings(policy, batch)
        z_vis = z_vis.to(device)
        pad_vis = pad_vis.to(device)

        model.train()
        loss, z_rl, _z_hat = model(z_vis, pad_vis)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.stage1.max_grad_norm)
        optimizer.step()
        scheduler.step()

        step += 1
        loss_val = float(loss.item())
        loss_ema = loss_val if loss_ema is None else 0.9 * loss_ema + 0.1 * loss_val

        if step == 1 or step % cfg.stage1.log_every == 0:
            steps_per_s = step / max(time.time() - start, 1e-6)
            lr = scheduler.get_last_lr()[0]
            logging.info(
                "step=%6d | loss/L_ro=%.5f | loss/ema=%.5f | lr=%.3e | grad=%.3f | sps=%.2f",
                step,
                loss_val,
                loss_ema,
                lr,
                float(grad_norm.item()),
                steps_per_s,
            )
            if wandb_run is not None:
                wandb_run.log(
                    {
                        "loss/L_ro": loss_val,
                        "loss/ema": loss_ema,
                        "lr/schedule": lr,
                        "grad/norm": float(grad_norm.item()),
                        "train/steps_per_s": steps_per_s,
                        "token/z_rl_norm": float(z_rl.norm(dim=-1).mean().item()),
                    },
                    step=step,
                )

        if step % cfg.stage1.ckpt_every == 0 or step == cfg.stage1.num_train_steps:
            ckpt_path = _save_checkpoint(model, cfg, step)
            logging.info("Saved checkpoint: %s", ckpt_path)

    if wandb_run is not None:
        wandb_run.finish()
    logging.info("Stage 1 RL Token training done.")


if __name__ == "__main__":
    main()
