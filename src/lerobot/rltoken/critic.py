"""TD3 twin Q critics for chunk-level RL Token training."""

from __future__ import annotations

import copy

import torch
from torch import Tensor, nn

from lerobot.rltoken.networks import MLP


class QNetwork(nn.Module):
    """Single Q-network mapping ``(state, action_chunk)`` to scalar Q."""

    def __init__(
        self,
        state_dim: int,
        action_chunk_dim: int,
        hidden_dim: int = 256,
        num_hidden_layers: int = 2,
    ) -> None:
        super().__init__()
        self.mlp = MLP(
            input_dim=state_dim + action_chunk_dim,
            output_dim=1,
            hidden_dim=hidden_dim,
            num_hidden_layers=num_hidden_layers,
        )

    def forward(self, x: Tensor, a: Tensor) -> Tensor:
        return self.mlp(torch.cat([x, a], dim=-1))


class TwinQCritic(nn.Module):
    """Twin Q networks with Polyak-updated target copies."""

    def __init__(
        self,
        state_dim: int,
        action_chunk_dim: int,
        hidden_dim: int = 256,
        num_hidden_layers: int = 2,
    ) -> None:
        super().__init__()
        self.q1 = QNetwork(state_dim, action_chunk_dim, hidden_dim, num_hidden_layers)
        self.q2 = QNetwork(state_dim, action_chunk_dim, hidden_dim, num_hidden_layers)
        self.q1_target = copy.deepcopy(self.q1)
        self.q2_target = copy.deepcopy(self.q2)
        for param in self.q1_target.parameters():
            param.requires_grad_(False)
        for param in self.q2_target.parameters():
            param.requires_grad_(False)

    def forward(self, x: Tensor, a: Tensor) -> tuple[Tensor, Tensor]:
        return self.q1(x, a), self.q2(x, a)

    def q_min(self, x: Tensor, a: Tensor) -> Tensor:
        q1, q2 = self.forward(x, a)
        return torch.min(q1, q2)

    @torch.no_grad()
    def target_q_min(self, x: Tensor, a: Tensor) -> Tensor:
        return torch.min(self.q1_target(x, a), self.q2_target(x, a))

    @torch.no_grad()
    def update_targets(self, tau: float) -> None:
        for online, target in ((self.q1, self.q1_target), (self.q2, self.q2_target)):
            for online_param, target_param in zip(online.parameters(), target.parameters(), strict=True):
                target_param.data.lerp_(online_param.data, tau)
