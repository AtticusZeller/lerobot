"""TD3 losses and targets for chunk-level RL Token training."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from lerobot.rltoken.actor import Actor
from lerobot.rltoken.critic import TwinQCritic


@torch.no_grad()
def compute_td_target(
    rewards: Tensor,
    dones: Tensor,
    next_x: Tensor,
    next_a_tilde: Tensor,
    actor: Actor,
    critic: TwinQCritic,
    gamma: float,
    chunk_length: int,
    target_noise_sigma: float = 0.2,
    target_noise_clip: float = 0.5,
) -> Tensor:
    """Compute macro-action TD target ``sum gamma^k r_k + gamma^C Q'``."""
    discount_powers = gamma ** torch.arange(chunk_length, device=rewards.device, dtype=rewards.dtype)
    chunk_return = (rewards * discount_powers).sum(dim=-1, keepdim=True)  # [B, 1]

    was_training = actor.training
    actor.eval()
    next_a = actor(next_x, next_a_tilde)
    if was_training:
        actor.train()

    noise = (torch.randn_like(next_a) * target_noise_sigma).clamp(-target_noise_clip, target_noise_clip)
    next_a = (next_a + noise).clamp(-1.0, 1.0)
    bootstrap = (gamma**chunk_length) * (1.0 - dones) * critic.target_q_min(next_x, next_a)
    return chunk_return + bootstrap


def critic_loss(q1: Tensor, q2: Tensor, q_target: Tensor) -> Tensor:
    return nn.functional.mse_loss(q1, q_target) + nn.functional.mse_loss(q2, q_target)


def actor_loss(q_value: Tensor, a: Tensor, a_tilde: Tensor, beta: float) -> Tensor:
    return -q_value.mean() + beta * nn.functional.mse_loss(a, a_tilde)
