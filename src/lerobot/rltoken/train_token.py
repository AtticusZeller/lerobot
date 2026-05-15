"""阶段二a：RL Token 编码器-解码器离线训练。

设计：见 ``docs/rltoken_plan.md`` §2.3 与 §5.2 阶段二之 2a。

流程：
    1. ``make_policy(PI05Config(pretrained_path=...,  freeze_vision_encoder=True, train_expert_only=True))``
       — 加载冻结的 π0.5 主干
    2. ``LeRobotDataset(cfg.dataset.repo_id)`` + ``make_pre_post_processors`` — 演示数据 → 模型期望 batch
    3. ``extract_vlm_embeddings(policy, batch)`` 截取 paligemma ``last_hidden_state`` → ``z_{1:M}``
    4. ``z_rl = encoder(z)``、``z_hat = decoder(z_rl)``
    5. ``L_ro = MSE(z_hat, sg(z))``（pad-mask 掩蔽）、AdamW、5K 步
"""

from __future__ import annotations

import logging
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from pprint import pformat

import draccus
import torch
import yaml
from safetensors.torch import save_file
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader

from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.rltoken.rl_token import (
    VLM_HIDDEN_DIM,
    RLTokenDecoder,
    RLTokenEncoder,
    extract_vlm_embeddings,
    reconstruction_loss,
)
from lerobot.utils.device_utils import get_safe_torch_device
from lerobot.utils.import_utils import register_third_party_plugins
from lerobot.utils.random_utils import set_seed
from lerobot.utils.utils import init_logging


@dataclass
class DatasetCfg:
    repo_id: str = "lerobot/libero"
    batch_size: int = 4
    num_workers: int = 4
    episodes: list[int] | None = None


@dataclass
class TokenArchCfg:
    z_dim: int = 256
    d_model: int = 256
    encoder_layers: int = 2
    decoder_layers: int = 2
    n_heads: int = 8
    max_seq_len: int = 512


@dataclass
class Stage1Cfg:
    steps: int = 5000
    learning_rate: float = 3.0e-4
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    log_every: int = 50
    ckpt_every: int = 1000


@dataclass
class TrainTokenConfig:
    pretrained_pi05: str = "lerobot/pi05_libero"
    output_dir: str = "outputs/rltoken/encoder_decoder"
    yaml_config: str | None = None
    suite: str | None = None  # informational only; dataset already encodes the suite
    steps: int | None = None  # CLI override for stage1.steps
    seed: int = 42
    device: str | None = None
    use_amp: bool = False
    wandb_project: str | None = None
    dataset: DatasetCfg = field(default_factory=DatasetCfg)
    rltoken: TokenArchCfg = field(default_factory=TokenArchCfg)
    stage1: Stage1Cfg = field(default_factory=Stage1Cfg)


def _merge_yaml_into_cfg(cfg: TrainTokenConfig) -> None:
    """Pull a few well-known fields from ``experiments/rltoken_pi05_libero.yaml`` into ``cfg``.

    Only sets fields the user hasn't overridden via CLI (we can't tell here, so YAML simply provides
    defaults that lose to draccus CLI args because draccus parses CLI last). We keep this very small —
    full YAML→dataclass mapping is overkill for a single training script.
    """
    if cfg.yaml_config is None:
        return
    path = Path(cfg.yaml_config)
    if not path.exists():
        logging.warning(f"yaml_config {path} does not exist; skipping YAML merge.")
        return
    with path.open() as f:
        data = yaml.safe_load(f) or {}
    cfg.pretrained_pi05 = data.get("pretrained_pi05", cfg.pretrained_pi05)
    if "rltoken" in data:
        rl = data["rltoken"]
        cfg.rltoken.z_dim = rl.get("z_dim", cfg.rltoken.z_dim)
        cfg.rltoken.d_model = rl.get("encoder_hidden", cfg.rltoken.d_model)
    if "training" in data and "stage1_steps" in data["training"]:
        cfg.stage1.steps = data["training"]["stage1_steps"]
    if "dataset" in data:
        ds = data["dataset"]
        cfg.dataset.repo_id = ds.get("repo_id", cfg.dataset.repo_id)
        cfg.dataset.batch_size = ds.get("batch_size", cfg.dataset.batch_size)
        cfg.dataset.num_workers = ds.get("num_workers", cfg.dataset.num_workers)
    if "stage1" in data:
        st = data["stage1"]
        cfg.stage1.learning_rate = st.get("learning_rate", cfg.stage1.learning_rate)
        cfg.stage1.weight_decay = st.get("weight_decay", cfg.stage1.weight_decay)
        cfg.stage1.grad_clip = st.get("grad_clip", cfg.stage1.grad_clip)
        cfg.stage1.log_every = st.get("log_every", cfg.stage1.log_every)
        cfg.stage1.ckpt_every = st.get("ckpt_every", cfg.stage1.ckpt_every)
    if "wandb" in data and cfg.wandb_project is None:
        cfg.wandb_project = data["wandb"].get("project")


def _build_policy(cfg: TrainTokenConfig, ds_meta):
    policy_cfg = PreTrainedConfig.from_pretrained(cfg.pretrained_pi05)
    policy_cfg.pretrained_path = cfg.pretrained_pi05
    # Force-freeze: paper §2.3 keeps π0.5 frozen during RL Token training.
    if hasattr(policy_cfg, "freeze_vision_encoder"):
        policy_cfg.freeze_vision_encoder = True
    if hasattr(policy_cfg, "train_expert_only"):
        policy_cfg.train_expert_only = True
    if cfg.device is not None:
        policy_cfg.device = cfg.device
    policy_cfg.use_amp = cfg.use_amp

    policy = make_policy(cfg=policy_cfg, ds_meta=ds_meta)
    policy.eval()
    for p in policy.parameters():
        p.requires_grad = False
    return policy, policy_cfg


def _build_dataloader(ds_cfg: DatasetCfg, preprocessor) -> DataLoader:
    dataset = LeRobotDataset(ds_cfg.repo_id, episodes=ds_cfg.episodes)

    def collate_then_preprocess(items: list[dict]):
        # Stack items into a batch of tensors. LeRobotDataset returns python dicts per frame; we use a
        # minimal collator: stack tensors, list strings.
        batch: dict = {}
        keys = items[0].keys()
        for k in keys:
            vals = [it[k] for it in items]
            if isinstance(vals[0], torch.Tensor):
                batch[k] = torch.stack(vals, dim=0)
            else:
                batch[k] = vals
        return preprocessor(batch)

    return DataLoader(
        dataset,
        batch_size=ds_cfg.batch_size,
        shuffle=True,
        num_workers=ds_cfg.num_workers,
        collate_fn=collate_then_preprocess,
        drop_last=True,
        persistent_workers=ds_cfg.num_workers > 0,
    )


@draccus.wrap()
def main(cfg: TrainTokenConfig):
    init_logging()
    register_third_party_plugins()
    _merge_yaml_into_cfg(cfg)
    if cfg.steps is not None:
        cfg.stage1.steps = cfg.steps

    logging.info(pformat(asdict(cfg)))

    device = get_safe_torch_device(cfg.device) if cfg.device else get_safe_torch_device("cuda")
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    set_seed(cfg.seed)

    # Dataset first — its metadata feeds both make_policy (for feature shapes / stats) and the
    # preprocessor (for normalization).
    dataset_for_meta = LeRobotDataset(cfg.dataset.repo_id, episodes=cfg.dataset.episodes)
    ds_meta = dataset_for_meta.meta

    logging.info(f"Loading frozen π0.5 from {cfg.pretrained_pi05}")
    policy, policy_cfg = _build_policy(cfg, ds_meta)

    preprocessor, _ = make_pre_post_processors(
        policy_cfg=policy_cfg,
        pretrained_path=cfg.pretrained_pi05,
        dataset_stats=ds_meta.stats,
        preprocessor_overrides={"device_processor": {"device": str(device)}},
    )

    loader = _build_dataloader(cfg.dataset, preprocessor)

    encoder = RLTokenEncoder(
        vlm_hidden_dim=VLM_HIDDEN_DIM,
        z_dim=cfg.rltoken.z_dim,
        d_model=cfg.rltoken.d_model,
        n_layers=cfg.rltoken.encoder_layers,
        n_heads=cfg.rltoken.n_heads,
    ).to(device)
    decoder = RLTokenDecoder(
        vlm_hidden_dim=VLM_HIDDEN_DIM,
        z_dim=cfg.rltoken.z_dim,
        d_model=cfg.rltoken.d_model,
        n_layers=cfg.rltoken.decoder_layers,
        n_heads=cfg.rltoken.n_heads,
        max_seq_len=cfg.rltoken.max_seq_len,
    ).to(device)

    n_params = sum(p.numel() for p in encoder.parameters()) + sum(p.numel() for p in decoder.parameters())
    logging.info(f"RL Token encoder+decoder trainable params: {n_params / 1e6:.2f}M")

    optim = AdamW(
        list(encoder.parameters()) + list(decoder.parameters()),
        lr=cfg.stage1.learning_rate,
        weight_decay=cfg.stage1.weight_decay,
    )

    wandb_run = None
    if cfg.wandb_project:
        try:
            import wandb

            wandb_run = wandb.init(project=cfg.wandb_project, config=asdict(cfg))
        except ImportError:
            logging.warning("wandb not installed; skipping logging.")

    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    step = 0
    loss_ema = None
    t0 = time.time()
    data_iter = iter(loader)
    while step < cfg.stage1.steps:
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            batch = next(data_iter)

        z_tokens, pad_mask = extract_vlm_embeddings(policy, batch)
        z_tokens = z_tokens.to(device)
        pad_mask = pad_mask.to(device)
        seq_len = z_tokens.shape[1]

        if seq_len > cfg.rltoken.max_seq_len:
            raise RuntimeError(
                f"VLM sequence length {seq_len} exceeds decoder max_seq_len {cfg.rltoken.max_seq_len}."
            )

        encoder.train()
        decoder.train()
        z_rl = encoder(z_tokens, key_padding_mask=~pad_mask.bool())
        z_hat = decoder(z_rl, seq_len=seq_len)
        loss = reconstruction_loss(z_tokens, z_hat, pad_mask)

        optim.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(
            list(encoder.parameters()) + list(decoder.parameters()), cfg.stage1.grad_clip
        )
        optim.step()

        step += 1
        loss_val = loss.item()
        loss_ema = loss_val if loss_ema is None else 0.9 * loss_ema + 0.1 * loss_val

        if step % cfg.stage1.log_every == 0 or step == 1:
            steps_per_s = step / max(1e-6, time.time() - t0)
            logging.info(f"step={step:6d} | L_ro={loss_val:.4f} | ema={loss_ema:.4f} | sps={steps_per_s:.2f}")
            if wandb_run is not None:
                wandb_run.log({"L_ro": loss_val, "L_ro_ema": loss_ema, "steps_per_s": steps_per_s}, step=step)

        if step % cfg.stage1.ckpt_every == 0 or step == cfg.stage1.steps:
            ckpt_path = out_dir / f"step_{step:06d}.safetensors"
            save_file(
                {
                    **{f"encoder.{k}": v.detach().cpu() for k, v in encoder.state_dict().items()},
                    **{f"decoder.{k}": v.detach().cpu() for k, v in decoder.state_dict().items()},
                },
                str(ckpt_path),
            )
            logging.info(f"Saved checkpoint: {ckpt_path}")

    if wandb_run is not None:
        wandb_run.finish()
    logging.info("Stage 2a training done.")


if __name__ == "__main__":
    main()
