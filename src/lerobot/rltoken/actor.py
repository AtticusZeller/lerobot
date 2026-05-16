"""TD3 actor conditioned on VLA reference action chunks."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from lerobot.rltoken.networks import MLP


class Actor(nn.Module):
    """Residual actor with reference-action dropout.

    Args:
        state_dim: RL state width, usually ``z_rl`` plus proprioception.
        action_chunk_dim: Flattened action chunk width, ``C * action_dim``.
        hidden_dim: MLP hidden layer width.
        num_hidden_layers: Number of hidden MLP layers.
        sigma: Gaussian exploration noise std used in training mode.
        ref_dropout: Probability of zeroing the reference action input.
    """

    def __init__(
        self,
        state_dim: int,
        action_chunk_dim: int,
        hidden_dim: int = 256,
        num_hidden_layers: int = 2,
        sigma: float = 0.1,
        ref_dropout: float = 0.5,
    ) -> None:
        super().__init__()
        self.action_chunk_dim = action_chunk_dim
        self.sigma = sigma
        self.ref_dropout = ref_dropout
        self.mlp = MLP(
            input_dim=state_dim + action_chunk_dim,
            output_dim=action_chunk_dim,
            hidden_dim=hidden_dim,
            num_hidden_layers=num_hidden_layers,
        )

        last_linear = [module for module in self.mlp.net if isinstance(module, nn.Linear)][-1]
        nn.init.zeros_(last_linear.weight)
        nn.init.zeros_(last_linear.bias)

    def forward(self, x: Tensor, a_tilde: Tensor) -> Tensor:
        """Return a residual-refined action chunk.

        Args:
            x: RL state, shape ``[B, state_dim]``.
            a_tilde: VLA reference action chunk, shape ``[B, C * action_dim]``.

        Returns:
            Clamped action chunk, shape ``[B, C * action_dim]``.
        """
        a_tilde_input = self._apply_ref_dropout(a_tilde)
        residual = self.mlp(torch.cat([x, a_tilde_input], dim=-1))
        action = a_tilde + residual
        if self.training:
            action = action + torch.randn_like(action) * self.sigma
        return action.clamp(-1.0, 1.0)

    def _apply_ref_dropout(self, a_tilde: Tensor) -> Tensor:
        if not self.training or self.ref_dropout == 0.0:
            return a_tilde
        keep_mask = torch.rand(a_tilde.shape[0], 1, device=a_tilde.device) >= self.ref_dropout
        return a_tilde * keep_mask
