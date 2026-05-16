"""Rollout worker for chunk-level RL Token data collection."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray

from lerobot.rltoken.actor import Actor
from lerobot.rltoken.intervention import InterventionManager, InterventionResult
from lerobot.rltoken.replay_buffer import ReplayBuffer
from lerobot.rltoken.rl_token import RLTokenModel


@dataclass
class EpisodeStats:
    total_reward: float = 0.0
    num_chunks: int = 0
    num_steps: int = 0
    done: bool = False
    interventions: int = 0
    extra: dict[str, Any] = field(default_factory=dict)


class RolloutWorker:
    """Collect warmup and actor rollouts for online RL."""

    def __init__(
        self,
        env: Any,
        vla,
        rl_token_model: RLTokenModel,
        actor: Actor,
        replay_buffer: ReplayBuffer,
        intervention_mgr: InterventionManager,
        chunk_length: int,
        action_dim: int,
        device: torch.device | str = "cuda",
    ) -> None:
        self.env = env
        self.vla = vla
        self.rl_token_model = rl_token_model
        self.actor = actor
        self.replay_buffer = replay_buffer
        self.intervention_mgr = intervention_mgr
        self.chunk_length = chunk_length
        self.action_dim = action_dim
        self.device = torch.device(device)

    @torch.no_grad()
    def _extract_rl_state(self, obs: dict[str, Any]) -> tuple[NDArray, NDArray]:
        batch = self.vla.preprocess_obs(obs, task=self.env.task)
        z, pad_mask = self.vla.extract_embeddings(batch)
        z_rl = self.rl_token_model.encode(z, pad_mask)  # [1, D]
        proprio = self.vla.proprioception(batch)  # [1, proprio_dim]
        x = torch.cat([z_rl, proprio], dim=-1)  # [1, state_dim]

        a_tilde = self.vla.get_rl_chunk_reference(batch, self.chunk_length)  # [1, C, action_dim]
        a_tilde_flat = a_tilde.reshape(1, -1)  # [1, C * action_dim]
        return x.squeeze(0).cpu().numpy(), a_tilde_flat.squeeze(0).cpu().numpy()

    @torch.no_grad()
    def _get_warmup_action(self, obs: dict[str, Any]) -> NDArray:
        batch = self.vla.preprocess_obs(obs, task=self.env.task)
        a_tilde = self.vla.get_rl_chunk_reference(batch, self.chunk_length)
        return a_tilde.squeeze(0).cpu().numpy()

    @torch.no_grad()
    def _get_actor_action(self, x: NDArray, a_tilde_flat: NDArray) -> NDArray:
        x_t = torch.as_tensor(x, dtype=torch.float32, device=self.device).unsqueeze(0)
        a_tilde_t = torch.as_tensor(a_tilde_flat, dtype=torch.float32, device=self.device).unsqueeze(0)
        a_flat = self.actor(x_t, a_tilde_t)  # [1, C * action_dim]
        return a_flat.squeeze(0).cpu().numpy().reshape(self.chunk_length, self.action_dim)

    def collect_warmup(self, num_chunks: int) -> tuple[int, int]:
        """Run VLA-only chunks and store transitions.

        Returns:
            ``(stored_transitions, env_steps)``.
        """
        stored = 0
        env_steps = 0
        obs = self.env.reset()
        for _ in range(num_chunks):
            action_chunk = self._get_warmup_action(obs)
            x, a_tilde_flat = self._extract_rl_state(obs)
            next_obs, rewards, done, info = self.env.step(action_chunk)
            next_x, _ = self._extract_rl_state(next_obs)
            self.replay_buffer.add(
                x=x,
                a=action_chunk.reshape(-1),
                a_tilde=a_tilde_flat,
                rewards=rewards,
                next_x=next_x,
                done=float(done),
            )
            stored += 1
            env_steps += int(info.get("steps_executed", self.chunk_length))
            obs = self.env.reset() if done else next_obs
        return stored, env_steps

    def collect_episode(self, store_transitions: bool = True) -> EpisodeStats:
        stats = EpisodeStats()
        obs = self.env.reset()

        while True:
            x, a_tilde_flat = self._extract_rl_state(obs)

            intervention: InterventionResult | None = None
            if self.intervention_mgr.check_intervention():
                intervention = self.intervention_mgr.get_human_action(self.action_dim, self.chunk_length)

            if intervention is not None:
                action_chunk = intervention.action_chunk
                next_obs = intervention.next_obs
                rewards = intervention.rewards
                done = intervention.done
                info = intervention.info
                stats.interventions += 1
            else:
                action_chunk = self._get_actor_action(x, a_tilde_flat)
                next_obs, rewards, done, info = self.env.step(action_chunk)

            if store_transitions:
                next_x, _ = self._extract_rl_state(next_obs)
                self.replay_buffer.add(
                    x=x,
                    a=action_chunk.reshape(-1),
                    a_tilde=a_tilde_flat,
                    rewards=rewards,
                    next_x=next_x,
                    done=float(done),
                )

            stats.total_reward += float(np.asarray(rewards).sum())
            stats.num_chunks += 1
            stats.num_steps += int(info.get("steps_executed", self.chunk_length))
            if done:
                stats.done = True
                stats.extra = info
                break
            obs = next_obs

        return stats
