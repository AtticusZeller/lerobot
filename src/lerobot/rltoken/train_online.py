"""Stage 2: online chunk-level TD3 with frozen π0.5 and RL Token."""

import logging
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from pprint import pformat
from typing import Any

import draccus
import torch
import yaml

from lerobot.configs.policies import PreTrainedConfig
from lerobot.envs.configs import LiberoEnv as LiberoEnvConfig
from lerobot.envs.factory import make_env_pre_post_processors
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.rltoken.adapter import LeRobotPI05Adapter
from lerobot.rltoken.checkpoint import load_rl_token_model
from lerobot.rltoken.libero_chunk_env import LiberoChunkEnv
from lerobot.rltoken.online_trainer import OnlineRLConfig, OnlineRLTrainer
from lerobot.utils.device_utils import get_safe_torch_device
from lerobot.utils.import_utils import register_third_party_plugins
from lerobot.utils.random_utils import set_seed
from lerobot.utils.utils import init_logging


@dataclass
class TrainOnlineConfig(OnlineRLConfig):
    pretrained: str = "lerobot/pi05_libero_finetuned"
    yaml_config: str | None = None
    suite: str = "libero_spatial"
    task_index: int = 0
    rl_token_checkpoint: str = ""
    resume_checkpoint: str = ""
    dataset_repo_id: str = "HuggingFaceVLA/libero"
    episode_length: int | None = None
    init_states: bool = True
    camera_name: str = "agentview_image,robot0_eye_in_hand_image"
    control_mode: str = "relative"
    observation_height: int = 360
    observation_width: int = 360
    seed: int = 42
    device: str | None = None
    wandb_project: str | None = None
    wandb_enabled: bool = True


def _cli_has(path: str) -> bool:
    flag = f"--{path}"
    return any(arg == flag or arg.startswith(f"{flag}=") for arg in sys.argv[1:])


def _set_if_not_cli(cfg: Any, path: str, value: Any) -> None:
    if value is None or _cli_has(path):
        return
    setattr(cfg, path, value)


def _merge_yaml_into_cfg(cfg: TrainOnlineConfig) -> None:
    if cfg.yaml_config is None:
        return
    path = Path(cfg.yaml_config)
    if not path.exists():
        logging.warning("yaml_config %s does not exist; skipping YAML merge.", path)
        return

    data = yaml.safe_load(path.read_text()) or {}
    _set_if_not_cli(cfg, "pretrained", data.get("pretrained") or data.get("pretrained_pi05"))

    libero = data.get("libero", {})
    _set_if_not_cli(cfg, "suite", libero.get("suite"))
    _set_if_not_cli(cfg, "episode_length", libero.get("max_episode_steps"))

    dataset = data.get("dataset", {})
    _set_if_not_cli(cfg, "dataset_repo_id", dataset.get("repo_id"))
    _set_if_not_cli(cfg, "task_index", dataset.get("task_index"))

    arch = data.get("rltoken", {})
    _set_if_not_cli(cfg, "embedding_dim", arch.get("embedding_dim"))

    td3 = data.get("td3", {})
    mapping = {
        "chunk_length": td3.get("action_chunk_size"),
        "actor_hidden": td3.get("actor_hidden"),
        "actor_hidden_layers": td3.get("actor_hidden_layers"),
        "actor_noise_sigma": td3.get("actor_noise_sigma"),
        "bc_coef": td3.get("bc_coef"),
        "ref_action_dropout": td3.get("ref_action_dropout"),
        "gamma": td3.get("gamma"),
        "tau": td3.get("tau"),
        "update_to_data_ratio": td3.get("update_to_data_ratio"),
        "critic_updates_per_actor": td3.get("critic_updates_per_actor"),
        "target_noise_sigma": td3.get("target_noise_sigma"),
        "target_noise_clip": td3.get("target_noise_clip"),
        "actor_lr": td3.get("actor_lr"),
        "critic_lr": td3.get("critic_lr"),
    }
    for key, value in mapping.items():
        _set_if_not_cli(cfg, key, value)

    training = data.get("training", {})
    training_mapping = {
        "max_env_steps": training.get("stage2_steps"),
        "warmup_steps": training.get("warmup_steps"),
        "batch_size": training.get("batch_size"),
        "buffer_capacity": training.get("buffer_capacity"),
        "save_every": training.get("save_every"),
        "log_every": training.get("log_every"),
    }
    for key, value in training_mapping.items():
        _set_if_not_cli(cfg, key, value)

    wandb = data.get("wandb", {})
    _set_if_not_cli(cfg, "wandb_project", wandb.get("project"))
    _set_if_not_cli(cfg, "wandb_enabled", wandb.get("enabled"))


def _build_policy_and_adapter(cfg: TrainOnlineConfig, device: torch.device) -> LeRobotPI05Adapter:
    env_cfg = LiberoEnvConfig(
        task=cfg.suite,
        task_ids=[cfg.task_index],
        episode_length=cfg.episode_length,
        obs_type="pixels_agent_pos",
        camera_name=cfg.camera_name,
        init_states=cfg.init_states,
        control_mode=cfg.control_mode,
        observation_height=cfg.observation_height,
        observation_width=cfg.observation_width,
    )
    policy_cfg = PreTrainedConfig.from_pretrained(cfg.pretrained)
    policy_cfg.pretrained_path = cfg.pretrained
    policy_cfg.device = str(device)
    policy = make_policy(cfg=policy_cfg, env_cfg=env_cfg)
    policy.eval()
    for param in policy.parameters():
        param.requires_grad_(False)

    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy_cfg,
        pretrained_path=cfg.pretrained,
        preprocessor_overrides={"device_processor": {"device": str(device)}},
    )
    env_preprocessor, _env_postprocessor = make_env_pre_post_processors(env_cfg=env_cfg, policy_cfg=policy_cfg)
    return LeRobotPI05Adapter(
        policy=policy,
        env_preprocessor=env_preprocessor,
        preprocessor=preprocessor,
        postprocessor=postprocessor,
        device=device,
        proprio_dim=cfg.proprio_dim,
    )


class WandbLogger:
    def __init__(self, cfg: TrainOnlineConfig) -> None:
        self.run = None
        if not cfg.wandb_project or not cfg.wandb_enabled:
            return
        try:
            import wandb

            self.run = wandb.init(project=cfg.wandb_project, name=cfg.run_name, config=asdict(cfg))
        except Exception:
            logging.warning("wandb init failed; continuing without W&B.", exc_info=True)

    def log(self, metrics: dict[str, float], step: int | None = None) -> None:
        if self.run is not None:
            self.run.log(metrics, step=step)

    def finish(self) -> None:
        if self.run is not None:
            self.run.finish()


@draccus.wrap()
def main(cfg: TrainOnlineConfig) -> None:
    init_logging()
    register_third_party_plugins()
    _merge_yaml_into_cfg(cfg)
    if not cfg.rl_token_checkpoint:
        raise ValueError("--rl_token_checkpoint is required for Stage 2 online training.")
    if not cfg.run_name or cfg.run_name == "run":
        cfg.run_name = f"{cfg.suite}_task_{cfg.task_index:02d}"

    device = get_safe_torch_device(cfg.device) if cfg.device else get_safe_torch_device("cuda")
    cfg.device = str(device)
    set_seed(cfg.seed)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True

    logging.info("Stage 2 config:\n%s", pformat(asdict(cfg)))
    vla = _build_policy_and_adapter(cfg, device)
    rl_token_model = load_rl_token_model(cfg.rl_token_checkpoint, device=device)
    env = LiberoChunkEnv(
        suite_name=cfg.suite,
        task_index=cfg.task_index,
        action_dim=cfg.action_dim,
        chunk_length=cfg.chunk_length,
        episode_length=cfg.episode_length,
        init_states=cfg.init_states,
        camera_name=cfg.camera_name,
        control_mode=cfg.control_mode,
        observation_height=cfg.observation_height,
        observation_width=cfg.observation_width,
    )

    trainer = OnlineRLTrainer(config=cfg, vla=vla, rl_token_model=rl_token_model, device=device)
    if cfg.resume_checkpoint:
        trainer.load(cfg.resume_checkpoint)

    logger = WandbLogger(cfg)
    try:
        trainer.train(env=env, log_fn=logger.log)
    finally:
        logger.finish()
        env.close()


if __name__ == "__main__":
    main()
