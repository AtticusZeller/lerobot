"""阶段二c：块级 TD3 在线训练主循环。

设计：见 ``docs/rltoken_plan.md`` §2.4 与 §5.2 阶段二之 2b-2c。

流程：
    1. 加载冻结 π0.5 + 加载阶段二a encoder ckpt（filter ``encoder.`` prefix）
    2. 构造单 LIBERO env → ChunkedLiberoEnv
    3. 主循环：obs → π0.5 prefix forward → z_rl + ref_chunk → actor → env.step → buffer.add
    4. ``update_to_data_ratio`` 次梯度更新：critic loss / actor loss / Polyak

注意：
    - 真实端到端 run 需要 LIBERO 仿真器 + π0.5 权重；本入口先确保 import + draccus parse OK，
      smoke 通过 ``--stage2.smoke=True`` 走纯 TD3 单元 loop 验证 update 路径不崩。
    - 演示预填 replay buffer 的功能留 TODO(reconcile)，回灌后实现。
"""

# NOTE: deliberately no `from __future__ import annotations` — draccus.wrap() introspects the
# function signature via `dataclasses.fields(...)` and PEP-563 stringified annotations break it.

import copy
import logging
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from pprint import pformat

import draccus
import torch
from safetensors.torch import load_file, save_file
from torch.optim import AdamW

from lerobot.rltoken.block_env import ACTION_DIM, ChunkedLiberoEnv
from lerobot.rltoken.td3 import (
    TD3Actor,
    TD3Critic,
    soft_update_target,
    td3_actor_loss,
    td3_critic_loss,
)
from lerobot.utils.import_utils import register_third_party_plugins
from lerobot.utils.random_utils import set_seed
from lerobot.utils.utils import init_logging


@dataclass
class StageTwoCfg:
    suite: str = "libero_spatial"
    task_id: int = 0
    chunk_size: int = 8
    z_dim: int = 256
    prop_dim: int = 9  # joints.pos (7) + gripper.qpos (2)
    hidden: int = 256
    actor_lr: float = 3.0e-4
    critic_lr: float = 3.0e-4
    gamma: float = 0.99
    tau: float = 0.005
    beta_bc: float = 0.5
    ref_dropout: float = 0.5
    residual_scale: float = 0.1
    batch_size: int = 64
    buffer_capacity: int = 100_000
    warmup_steps: int = 200
    update_to_data_ratio: int = 5
    steps: int = 10_000
    log_every: int = 50
    ckpt_every: int = 5000
    smoke: bool = False  # synthetic loop, skip LIBERO + π0.5


@dataclass
class TrainOnlineConfig:
    pretrained_pi05: str = "lerobot/pi05_libero"
    encoder_ckpt: str = "outputs/rltoken/encoder_decoder/step_5000.safetensors"
    output_dir: str = "outputs/rltoken/stage2"
    device: str | None = None
    seed: int = 42
    wandb_project: str | None = None
    stage2: StageTwoCfg = field(default_factory=StageTwoCfg)


def _load_encoder_state(encoder, ckpt_path: str) -> None:
    """Load the encoder slice of a stage-2a safetensors checkpoint."""
    state = load_file(ckpt_path)
    enc_state = {k.removeprefix("encoder."): v for k, v in state.items() if k.startswith("encoder.")}
    if not enc_state:
        raise ValueError(f"No 'encoder.*' keys found in {ckpt_path}; not a stage-2a checkpoint?")
    missing, unexpected = encoder.load_state_dict(enc_state, strict=False)
    if missing or unexpected:
        logging.warning(f"encoder load_state_dict missing={missing} unexpected={unexpected}")


def _flatten_prop(obs: dict, device: torch.device) -> torch.Tensor:
    """Concatenate ``robot_state.joints.pos`` (7) + ``robot_state.gripper.qpos`` (2) → (1, 9)."""
    rs = obs["robot_state"]
    joints_pos = torch.as_tensor(rs["joints"]["pos"], dtype=torch.float32, device=device)
    grip_qpos = torch.as_tensor(rs["gripper"]["qpos"], dtype=torch.float32, device=device)
    return torch.cat([joints_pos, grip_qpos]).unsqueeze(0)


def _build_pi05_batch(
    obs: dict,
    task_description: str,
    preprocessor,
    env_preprocessor,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    """Convert single-env LIBERO observation into a policy-ready batch (B=1)."""
    from lerobot.envs.utils import preprocess_observation

    batched = preprocess_observation(obs)
    batched["task"] = [task_description]
    batched = env_preprocessor(batched)
    return preprocessor(batched)


def _make_chunk_env(cfg: StageTwoCfg):
    """Build a single-task ChunkedLiberoEnv. Imports LIBERO lazily."""
    from lerobot.envs.configs import LiberoEnv as LiberoEnvCfg
    from lerobot.envs.factory import make_env

    env_cfg = LiberoEnvCfg(task=cfg.suite, task_ids=[cfg.task_id])
    envs_dict = make_env(env_cfg, n_envs=1, use_async_envs=False)
    vec_env = envs_dict[cfg.suite][cfg.task_id]
    inner_env = vec_env.envs[0]  # SyncVectorEnv exposes sub-envs as .envs
    return ChunkedLiberoEnv(inner_env, chunk_size=cfg.chunk_size), env_cfg, vec_env


def _build_pi05(pretrained: str, env_cfg, device: torch.device):
    """Lazy import to keep --stage2.smoke / --help free of π0.5 heavy deps."""
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.factory import make_policy, make_pre_post_processors

    policy_cfg = PreTrainedConfig.from_pretrained(pretrained)
    policy_cfg.pretrained_path = pretrained
    if hasattr(policy_cfg, "freeze_vision_encoder"):
        policy_cfg.freeze_vision_encoder = True
    if hasattr(policy_cfg, "train_expert_only"):
        policy_cfg.train_expert_only = True
    policy_cfg.device = str(device)

    policy = make_policy(cfg=policy_cfg, env_cfg=env_cfg)
    policy.eval()
    for p in policy.parameters():
        p.requires_grad_(False)

    preprocessor, _ = make_pre_post_processors(
        policy_cfg=policy_cfg,
        pretrained_path=pretrained,
        preprocessor_overrides={"device_processor": {"device": str(device)}},
    )
    from lerobot.envs.factory import make_env_pre_post_processors

    env_preprocessor, _env_postprocessor = make_env_pre_post_processors(
        env_cfg=env_cfg, policy_cfg=policy_cfg
    )
    return policy, policy_cfg, preprocessor, env_preprocessor


def _build_replay_buffer(cfg: StageTwoCfg, device: torch.device):
    from lerobot.rl.buffer import ReplayBuffer

    return ReplayBuffer(
        capacity=cfg.buffer_capacity,
        device=str(device),
        storage_device="cpu",
        state_keys=["z_rl", "prop", "ref_chunk"],
        use_drq=False,
    )


def _td3_update_step(
    cfg: StageTwoCfg,
    buffer,
    actor: TD3Actor,
    actor_target: TD3Actor,
    critic: TD3Critic,
    critic_target: TD3Critic,
    opt_a: AdamW,
    opt_c: AdamW,
    device: torch.device,
) -> dict[str, float]:
    """Single critic+actor gradient step + target soft update. Returns scalar losses."""
    tr = buffer.sample(cfg.batch_size)
    z = tr["state"]["z_rl"]
    prop = tr["state"]["prop"]
    ref_flat = tr["state"]["ref_chunk"]
    a_flat = tr["action"]
    z_n = tr["next_state"]["z_rl"]
    prop_n = tr["next_state"]["prop"]
    ref_n_flat = tr["next_state"]["ref_chunk"]

    ref = ref_flat.view(-1, cfg.chunk_size, ACTION_DIM)
    a = a_flat.view(-1, cfg.chunk_size, ACTION_DIM)
    ref_n = ref_n_flat.view(-1, cfg.chunk_size, ACTION_DIM)

    loss_c = td3_critic_loss(
        critic,
        critic_target,
        actor_target,
        z,
        prop,
        ref,
        a,
        z_n,
        prop_n,
        ref_n,
        tr["reward"],
        tr["done"].float(),
        gamma=cfg.gamma,
        chunk_size=cfg.chunk_size,
    )
    opt_c.zero_grad(set_to_none=True)
    loss_c.backward()
    opt_c.step()

    drop = torch.rand(z.size(0), device=device) < cfg.ref_dropout
    loss_a = td3_actor_loss(actor, critic, z, prop, ref, beta=cfg.beta_bc, drop_ref=drop)
    opt_a.zero_grad(set_to_none=True)
    loss_a.backward()
    opt_a.step()

    soft_update_target(actor_target, actor, cfg.tau)
    soft_update_target(critic_target, critic, cfg.tau)
    return {"loss_critic": float(loss_c.item()), "loss_actor": float(loss_a.item())}


def _smoke_loop(cfg: TrainOnlineConfig, device: torch.device) -> None:
    """Pure-torch smoke loop: build TD3 + buffer, fill with synthetic data, run a few updates."""
    s2 = cfg.stage2
    actor = TD3Actor(
        z_dim=s2.z_dim,
        prop_dim=s2.prop_dim,
        action_dim=ACTION_DIM,
        chunk_size=s2.chunk_size,
        hidden=s2.hidden,
        ref_dropout=s2.ref_dropout,
        residual_scale=s2.residual_scale,
    ).to(device)
    critic = TD3Critic(
        z_dim=s2.z_dim,
        prop_dim=s2.prop_dim,
        action_dim=ACTION_DIM,
        chunk_size=s2.chunk_size,
        hidden=s2.hidden,
    ).to(device)
    actor_target = copy.deepcopy(actor).eval()
    critic_target = copy.deepcopy(critic).eval()
    for p in actor_target.parameters():
        p.requires_grad_(False)
    for p in critic_target.parameters():
        p.requires_grad_(False)
    opt_a = AdamW(actor.parameters(), lr=s2.actor_lr)
    opt_c = AdamW(critic.parameters(), lr=s2.critic_lr)

    buffer = _build_replay_buffer(s2, device)
    g = torch.Generator(device="cpu").manual_seed(cfg.seed)
    for _ in range(max(s2.batch_size + 1, 16)):
        state = {
            "z_rl": torch.randn(1, s2.z_dim, generator=g),
            "prop": torch.randn(1, s2.prop_dim, generator=g),
            "ref_chunk": torch.randn(1, s2.chunk_size * ACTION_DIM, generator=g).clamp(-0.5, 0.5),
        }
        next_state = {k: v + 0.01 for k, v in state.items()}
        action = state["ref_chunk"].clone()
        reward = float(torch.rand(1, generator=g).item() < 0.05)  # sparse
        done = bool(torch.rand(1, generator=g).item() < 0.05)
        buffer.add(state, action, reward, next_state, done, False)

    for step in range(s2.steps):
        for _ in range(s2.update_to_data_ratio):
            losses = _td3_update_step(
                s2,
                buffer,
                actor,
                actor_target,
                critic,
                critic_target,
                opt_a,
                opt_c,
                device,
            )
        if step % s2.log_every == 0 or step == s2.steps - 1:
            logging.info(f"[smoke] step={step} {losses}")

    logging.info("smoke loop done.")


def _online_loop(cfg: TrainOnlineConfig, device: torch.device) -> None:
    s2 = cfg.stage2
    chunk_env, env_cfg, _vec = _make_chunk_env(s2)
    policy, _policy_cfg, preprocessor, env_preprocessor = _build_pi05(cfg.pretrained_pi05, env_cfg, device)

    from lerobot.rltoken.rl_token import RLTokenEncoder, extract_vlm_embeddings

    encoder = RLTokenEncoder(z_dim=s2.z_dim, d_model=s2.hidden, n_layers=2).to(device)
    _load_encoder_state(encoder, cfg.encoder_ckpt)
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad_(False)

    actor = TD3Actor(
        z_dim=s2.z_dim,
        prop_dim=s2.prop_dim,
        action_dim=ACTION_DIM,
        chunk_size=s2.chunk_size,
        hidden=s2.hidden,
        ref_dropout=s2.ref_dropout,
        residual_scale=s2.residual_scale,
    ).to(device)
    critic = TD3Critic(
        z_dim=s2.z_dim,
        prop_dim=s2.prop_dim,
        action_dim=ACTION_DIM,
        chunk_size=s2.chunk_size,
        hidden=s2.hidden,
    ).to(device)
    actor_target = copy.deepcopy(actor).eval()
    critic_target = copy.deepcopy(critic).eval()
    for p in actor_target.parameters():
        p.requires_grad_(False)
    for p in critic_target.parameters():
        p.requires_grad_(False)
    opt_a = AdamW(actor.parameters(), lr=s2.actor_lr)
    opt_c = AdamW(critic.parameters(), lr=s2.critic_lr)
    buffer = _build_replay_buffer(s2, device)

    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    task_description = chunk_env.task_description
    obs, _info = chunk_env.reset()
    t0 = time.time()

    def _state_from_obs(obs_dict: dict) -> dict[str, torch.Tensor]:
        batch = _build_pi05_batch(obs_dict, task_description, preprocessor, env_preprocessor, device)
        with torch.no_grad():
            z_tokens, pad = extract_vlm_embeddings(policy, batch)
            z_rl = encoder(z_tokens.to(device), key_padding_mask=~pad.to(device).bool())
            ref_chunk_full = policy.predict_action_chunk(batch)  # (1, 50, A)
            ref_chunk = ref_chunk_full[:, : s2.chunk_size]  # (1, C, A)
        prop = _flatten_prop(obs_dict, device)
        return {"z_rl": z_rl, "prop": prop, "ref_chunk": ref_chunk.flatten(1)}, ref_chunk

    state, ref_chunk = _state_from_obs(obs)

    for step in range(s2.steps):
        with torch.no_grad():
            if step < s2.warmup_steps:
                action_chunk = ref_chunk
            else:
                drop = torch.zeros(1, dtype=torch.bool, device=device)
                action_chunk = actor(state["z_rl"], state["prop"], ref_chunk, drop_ref=drop)

        next_obs, reward, terminated, _truncated, info = chunk_env.step(action_chunk.squeeze(0).cpu().numpy())
        next_state, next_ref_chunk = _state_from_obs(next_obs)
        buffer.add(
            state=dict(state),
            action=action_chunk.flatten(1),
            reward=float(reward),
            next_state=dict(next_state),
            done=bool(terminated),
            truncated=False,
        )

        losses: dict[str, float] = {}
        if len(buffer) >= max(s2.batch_size, s2.warmup_steps):
            for _ in range(s2.update_to_data_ratio):
                losses = _td3_update_step(
                    s2,
                    buffer,
                    actor,
                    actor_target,
                    critic,
                    critic_target,
                    opt_a,
                    opt_c,
                    device,
                )

        if step % s2.log_every == 0 or step == s2.steps - 1:
            sps = (step + 1) / max(1e-6, time.time() - t0)
            logging.info(
                f"step={step:6d} buffer={len(buffer):6d} reward_last={reward:.2f} "
                f"inner_steps={info.get('inner_steps')} success={info.get('is_success')} "
                f"{losses} sps={sps:.2f}"
            )

        if terminated:
            obs, _ = chunk_env.reset()
            state, ref_chunk = _state_from_obs(obs)
        else:
            state, ref_chunk = next_state, next_ref_chunk

        if (step + 1) % s2.ckpt_every == 0 or step == s2.steps - 1:
            ckpt_path = out_dir / f"td3_step_{step + 1:07d}.safetensors"
            save_file(
                {
                    **{f"actor.{k}": v.detach().cpu() for k, v in actor.state_dict().items()},
                    **{f"critic.{k}": v.detach().cpu() for k, v in critic.state_dict().items()},
                },
                str(ckpt_path),
            )
            logging.info(f"Saved TD3 checkpoint: {ckpt_path}")

    logging.info("online TD3 training done.")


@draccus.wrap()
def main(cfg: TrainOnlineConfig) -> None:
    init_logging()
    register_third_party_plugins()
    logging.info(pformat(asdict(cfg)))

    device_str = cfg.device or ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_str)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    set_seed(cfg.seed)

    if cfg.stage2.smoke:
        _smoke_loop(cfg, device)
    else:
        _online_loop(cfg, device)


if __name__ == "__main__":
    main()


# TODO(reconcile): Items deferred until rltoken main branch verification is in.
# - Demo prefill of replay buffer from `lerobot/libero` via stage-2a-style dataloader (so warmup
#   doesn't have to be raw π0.5 rollouts).
# - Wire stage-2 checkpoint into `lerobot.rltoken.eval_throughput` so we can compare TD3-edited
#   chunks against the SFT baseline throughput.
# - Multi-task / multi-suite: currently single (suite, task_id). Generalise once the single-task
#   loop is stable.
# - Profile and possibly batch the π0.5 forward across multiple parallel envs.
