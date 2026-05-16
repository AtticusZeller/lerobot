"""Online RL trainer for frozen π0.5 + RL Token + chunk-level TD3."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from lerobot.rltoken.actor import Actor
from lerobot.rltoken.critic import TwinQCritic
from lerobot.rltoken.intervention import InterventionManager
from lerobot.rltoken.replay_buffer import ReplayBuffer
from lerobot.rltoken.rl_token import VLM_HIDDEN_DIM, RLTokenModel
from lerobot.rltoken.rollout_worker import RolloutWorker
from lerobot.rltoken.td3_utils import actor_loss, compute_td_target, critic_loss

log = logging.getLogger(__name__)


@dataclass
class OnlineRLConfig:
    embedding_dim: int = VLM_HIDDEN_DIM
    proprio_dim: int = 8
    action_dim: int = 7
    chunk_length: int = 8
    actor_hidden: int = 256
    actor_hidden_layers: int = 2
    actor_noise_sigma: float = 0.1
    ref_action_dropout: float = 0.5
    gamma: float = 0.99
    tau: float = 0.005
    update_to_data_ratio: int = 5
    bc_coef: float = 0.5
    critic_updates_per_actor: int = 2
    target_noise_sigma: float = 0.2
    target_noise_clip: float = 0.5
    actor_lr: float = 3.0e-4
    critic_lr: float = 3.0e-4
    buffer_capacity: int = 100_000
    batch_size: int = 256
    warmup_steps: int = 1000
    max_env_steps: int = 200_000
    save_dir: str = "outputs/rltoken/online"
    run_name: str = "run"
    save_every: int = 50
    log_every: int = 1

    @property
    def state_dim(self) -> int:
        return self.embedding_dim + self.proprio_dim

    @property
    def action_chunk_dim(self) -> int:
        return self.chunk_length * self.action_dim


class OnlineRLTrainer:
    """Run chunk-level TD3 with frozen VLA and frozen RL Token encoder."""

    def __init__(
        self,
        config: OnlineRLConfig,
        vla,
        rl_token_model: RLTokenModel,
        device: torch.device | str = "cuda",
    ) -> None:
        self.config = config
        self.device = torch.device(device)
        self.vla = vla
        self.rl_token_model = rl_token_model.eval()
        for param in self.rl_token_model.parameters():
            param.requires_grad_(False)

        self.actor = Actor(
            state_dim=config.state_dim,
            action_chunk_dim=config.action_chunk_dim,
            hidden_dim=config.actor_hidden,
            num_hidden_layers=config.actor_hidden_layers,
            sigma=config.actor_noise_sigma,
            ref_dropout=config.ref_action_dropout,
        ).to(self.device)
        self.critic = TwinQCritic(
            state_dim=config.state_dim,
            action_chunk_dim=config.action_chunk_dim,
            hidden_dim=config.actor_hidden,
            num_hidden_layers=config.actor_hidden_layers,
        ).to(self.device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=config.actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=config.critic_lr)
        self.replay_buffer = ReplayBuffer(
            capacity=config.buffer_capacity,
            state_dim=config.state_dim,
            action_chunk_dim=config.action_chunk_dim,
            chunk_length=config.chunk_length,
        )
        self._total_env_steps = 0
        self._total_updates = 0
        self._total_episodes = 0

    def _create_worker(self, env: Any, intervention_mgr: InterventionManager | None) -> RolloutWorker:
        return RolloutWorker(
            env=env,
            vla=self.vla,
            rl_token_model=self.rl_token_model,
            actor=self.actor,
            replay_buffer=self.replay_buffer,
            intervention_mgr=intervention_mgr or InterventionManager(),
            chunk_length=self.config.chunk_length,
            action_dim=self.config.action_dim,
            device=self.device,
        )

    def _update_step(self, update_idx: int) -> dict[str, float]:
        cfg = self.config
        batch = self.replay_buffer.sample(cfg.batch_size, device=self.device)
        x = batch["x"]
        a = batch["a"]
        a_tilde = batch["a_tilde"]
        rewards = batch["rewards"]
        next_x = batch["next_x"]
        dones = batch["dones"]

        td_target = compute_td_target(
            rewards=rewards,
            dones=dones,
            next_x=next_x,
            next_a_tilde=a_tilde,
            actor=self.actor,
            critic=self.critic,
            gamma=cfg.gamma,
            chunk_length=cfg.chunk_length,
            target_noise_sigma=cfg.target_noise_sigma,
            target_noise_clip=cfg.target_noise_clip,
        )
        q1, q2 = self.critic(x, a)
        c_loss = critic_loss(q1, q2, td_target)
        self.critic_optimizer.zero_grad(set_to_none=True)
        c_loss.backward()
        self.critic_optimizer.step()

        metrics = {
            "loss/critic": float(c_loss.item()),
            "q/q1_mean": float(q1.mean().item()),
            "q/q2_mean": float(q2.mean().item()),
        }
        if update_idx % cfg.critic_updates_per_actor == 0:
            self.actor.train()
            a_actor = self.actor(x, a_tilde)
            q_value = self.critic.q_min(x, a_actor)
            a_loss = actor_loss(q_value, a_actor, a_tilde, cfg.bc_coef)
            self.actor_optimizer.zero_grad(set_to_none=True)
            a_loss.backward()
            self.actor_optimizer.step()
            metrics["loss/actor"] = float(a_loss.item())

        self.critic.update_targets(cfg.tau)
        self._total_updates += 1
        return metrics

    @staticmethod
    def _log(log_fn, metrics: dict[str, float], step: int) -> None:
        if log_fn is None:
            return
        try:
            log_fn(metrics, step=step)
        except TypeError:
            log_fn(metrics)

    def train(self, env: Any, intervention_mgr: InterventionManager | None = None, log_fn=None) -> None:
        cfg = self.config
        worker = self._create_worker(env, intervention_mgr)
        start = time.time()

        if self.replay_buffer.size == 0 and cfg.warmup_steps > 0:
            stored, env_steps = worker.collect_warmup(cfg.warmup_steps)
            self._total_env_steps += env_steps
            log.info("Warmup stored %d transitions (%d env steps).", stored, env_steps)

        while self._total_env_steps < cfg.max_env_steps:
            self.actor.eval()
            stats = worker.collect_episode()
            self._total_episodes += 1
            self._total_env_steps += stats.num_steps

            update_metrics: dict[str, float] = {}
            for update_idx in range(cfg.update_to_data_ratio):
                update_metrics = self._update_step(update_idx)

            metrics = {
                "episode/reward": stats.total_reward,
                "episode/success": float(bool(stats.extra.get("success", False))),
                "episode/chunks": float(stats.num_chunks),
                "episode/steps": float(stats.num_steps),
                "train/env_steps": float(self._total_env_steps),
                "train/episodes": float(self._total_episodes),
                "train/updates": float(self._total_updates),
                "replay/size": float(self.replay_buffer.size),
                "time/seconds": time.time() - start,
                **update_metrics,
            }
            if self._total_episodes % cfg.log_every == 0:
                log.info(
                    "ep=%d env_steps=%d reward=%.2f success=%s buffer=%d",
                    self._total_episodes,
                    self._total_env_steps,
                    stats.total_reward,
                    bool(stats.extra.get("success", False)),
                    self.replay_buffer.size,
                )
                self._log(log_fn, metrics, self._total_env_steps)

            if self._total_episodes % cfg.save_every == 0:
                self.save()

        self.save()

    def save(self, path: str | None = None, save_buffer: bool = True) -> Path:
        save_dir = Path(path or self.config.save_dir) / self.config.run_name
        save_dir.mkdir(parents=True, exist_ok=True)
        ckpt_path = save_dir / f"online_rl_ep{self._total_episodes}.pt"
        payload: dict[str, Any] = {
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
            "total_env_steps": self._total_env_steps,
            "total_updates": self._total_updates,
            "total_episodes": self._total_episodes,
            "config": self.config,
        }
        if save_buffer:
            payload["replay_buffer"] = self.replay_buffer.state_dict()
        torch.save(payload, ckpt_path)
        log.info("Saved online RL checkpoint: %s", ckpt_path)
        return ckpt_path

    def load(self, ckpt_path: str) -> None:
        ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=False)
        self.actor.load_state_dict(ckpt["actor"])
        self.critic.load_state_dict(ckpt["critic"])
        self.actor_optimizer.load_state_dict(ckpt["actor_optimizer"])
        self.critic_optimizer.load_state_dict(ckpt["critic_optimizer"])
        self._total_env_steps = int(ckpt["total_env_steps"])
        self._total_updates = int(ckpt["total_updates"])
        self._total_episodes = int(ckpt["total_episodes"])
        if "replay_buffer" in ckpt:
            self.replay_buffer.load_state_dict(ckpt["replay_buffer"])
        log.info("Loaded online RL checkpoint: %s", ckpt_path)
