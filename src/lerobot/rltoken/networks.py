"""Shared neural network blocks for RL Token TD3."""

from __future__ import annotations

from torch import Tensor, nn


class MLP(nn.Module):
    """LayerNorm MLP used by actor and critic networks.

    Args:
        input_dim: Input feature width.
        output_dim: Output feature width.
        hidden_dim: Hidden layer width.
        num_hidden_layers: Number of hidden ``Linear -> ReLU`` blocks.
    """

    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int, num_hidden_layers: int) -> None:
        super().__init__()
        layers: list[nn.Module] = [nn.LayerNorm(input_dim)]
        prev_dim = input_dim
        for _ in range(num_hidden_layers):
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)
