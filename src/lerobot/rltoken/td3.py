"""阶段二c：块级 TD3 网络 + 损失 + Polyak 软更新。

设计：``docs/rltoken_plan.md`` §2.4。

接口约定：
    z_rl     (B, Z)        — RL Token 编码器输出（256D）
    prop     (B, P)        — 本体感知状态（拼接 joints.pos + gripper.qpos = 9D 默认）
    ref_chunk (B, C, A)    — π0.5 参考动作块（前 C 步）
    action_chunk (B, C, A) — 实际执行的动作块（actor 输出，可能 = ref + residual）

Actor 是残差头零初始化：训练初期 ``actor(z, prop, ref) ≈ ref``，避免破坏 π0.5 先验。
Critic 双 Q：取 min 计算 Bellman target，缓解过估计。
Bellman 跨 ``C`` 步：``y = r + (1-done) * γ^C * min_j Q'_j(s_{t+C}, actor_target(s_{t+C}, ref_{t+C}))``。
"""

from __future__ import annotations

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn


def _mlp(in_dim: int, hidden: int, out_dim: int, n_layers: int = 2) -> nn.Sequential:
    layers: list[nn.Module] = [nn.Linear(in_dim, hidden), nn.ReLU()]
    for _ in range(n_layers - 1):
        layers += [nn.Linear(hidden, hidden), nn.ReLU()]
    layers.append(nn.Linear(hidden, out_dim))
    return nn.Sequential(*layers)


class TD3Actor(nn.Module):
    """Residual chunk actor with zero-initialised head.

    forward returns ``ref_chunk + residual``, clipped to ``[action_low, action_high]``. With the
    head zero-init, the first forward exactly reproduces the π0.5 reference and the gradient flow
    only edits via the residual delta. ``ref_dropout`` masks the reference input (sets it to zero
    and toggles the ``ref_mask`` flag) so the actor learns to also operate without π0.5's hint.
    """

    def __init__(
        self,
        z_dim: int = 256,
        prop_dim: int = 9,
        action_dim: int = 7,
        chunk_size: int = 8,
        hidden: int = 256,
        ref_dropout: float = 0.5,
        action_low: float = -1.0,
        action_high: float = 1.0,
        residual_scale: float = 0.1,
    ):
        super().__init__()
        in_dim = z_dim + prop_dim + chunk_size * action_dim + 1  # +1 = ref_mask flag
        self.trunk = _mlp(in_dim, hidden, hidden, n_layers=2)
        self.residual_head = nn.Linear(hidden, chunk_size * action_dim)
        nn.init.zeros_(self.residual_head.weight)
        nn.init.zeros_(self.residual_head.bias)
        self.chunk_size = chunk_size
        self.action_dim = action_dim
        self.ref_dropout = ref_dropout
        self.action_low = action_low
        self.action_high = action_high
        self.residual_scale = residual_scale

    def forward(
        self,
        z_rl: Tensor,
        prop: Tensor,
        ref_chunk: Tensor,
        drop_ref: Tensor | None = None,
    ) -> Tensor:
        """Inputs: ``z_rl (B, Z), prop (B, P), ref_chunk (B, C, A), drop_ref (B,) bool``."""
        b = z_rl.shape[0]
        if drop_ref is None:
            if self.training:
                drop_ref = torch.rand(b, device=z_rl.device) < self.ref_dropout
            else:
                drop_ref = torch.zeros(b, dtype=torch.bool, device=z_rl.device)

        keep = (~drop_ref).to(ref_chunk.dtype).view(b, 1, 1)
        ref_masked = ref_chunk * keep
        ref_flag = keep.view(b, 1)  # 1.0 = ref kept, 0.0 = ref dropped

        x = torch.cat([z_rl, prop, ref_masked.flatten(1), ref_flag], dim=-1)
        h = self.trunk(x)
        residual = self.residual_head(h).view(b, self.chunk_size, self.action_dim)
        residual = torch.tanh(residual) * self.residual_scale
        action_chunk = ref_chunk + residual
        return action_chunk.clamp(self.action_low, self.action_high)


class TD3Critic(nn.Module):
    """Double-Q critic over flattened action chunk."""

    def __init__(
        self,
        z_dim: int = 256,
        prop_dim: int = 9,
        action_dim: int = 7,
        chunk_size: int = 8,
        hidden: int = 256,
    ):
        super().__init__()
        in_dim = z_dim + prop_dim + chunk_size * action_dim
        self.q1 = _mlp(in_dim, hidden, 1, n_layers=2)
        self.q2 = _mlp(in_dim, hidden, 1, n_layers=2)

    def forward(self, z_rl: Tensor, prop: Tensor, action_chunk: Tensor) -> Tensor:
        """Returns ``(B, 2)``."""
        x = torch.cat([z_rl, prop, action_chunk.flatten(1)], dim=-1)
        return torch.stack([self.q1(x).squeeze(-1), self.q2(x).squeeze(-1)], dim=-1)


@torch.no_grad()
def soft_update_target(target: nn.Module, online: nn.Module, tau: float = 0.005) -> None:
    """Polyak: ``target ← (1 - tau) * target + tau * online``."""
    for tp, p in zip(target.parameters(), online.parameters(), strict=True):
        tp.data.mul_(1.0 - tau).add_(p.data, alpha=tau)


def td3_critic_loss(
    critic: TD3Critic,
    critic_target: TD3Critic,
    actor_target: TD3Actor,
    z: Tensor,
    prop: Tensor,
    ref_chunk: Tensor,
    action_chunk: Tensor,
    z_next: Tensor,
    prop_next: Tensor,
    ref_chunk_next: Tensor,
    reward_chunk: Tensor,
    done: Tensor,
    gamma: float = 0.99,
    chunk_size: int = 8,
) -> Tensor:
    """Block-level Bellman: ``y = r + (1-done) * γ^C * min_j Q'_j(s_{t+C}, a')``.

    ``reward_chunk`` is the chunk-level scalar reward already accumulated by the env wrapper
    (1.0 if any inner step succeeded, else 0.0 — see ``block_env``). ``done`` is the chunk-level
    terminated flag.
    """
    with torch.no_grad():
        b = z_next.shape[0]
        a_next = actor_target(
            z_next,
            prop_next,
            ref_chunk_next,
            drop_ref=torch.zeros(b, dtype=torch.bool, device=z.device),
        )
        q_next = critic_target(z_next, prop_next, a_next)  # (B, 2)
        min_q_next, _ = q_next.min(dim=-1)  # (B,)
        y = reward_chunk + (1.0 - done) * (gamma**chunk_size) * min_q_next
    q_pred = critic(z, prop, action_chunk)  # (B, 2)
    return F.mse_loss(q_pred[:, 0], y) + F.mse_loss(q_pred[:, 1], y)


def td3_actor_loss(
    actor: TD3Actor,
    critic: TD3Critic,
    z: Tensor,
    prop: Tensor,
    ref_chunk: Tensor,
    beta: float = 0.5,
    drop_ref: Tensor | None = None,
) -> Tensor:
    """``L_actor = -Q1(s, π(s)).mean() + β * ||π(s) - ref||^2``.

    Critic params are not detached here; the optimiser must only step on ``actor.parameters()``.
    Standard TD3 uses Q1 only in the actor objective.
    """
    a_pi = actor(z, prop, ref_chunk, drop_ref=drop_ref)
    q1 = critic(z, prop, a_pi)[:, 0]
    bc = (a_pi - ref_chunk).pow(2).mean(dim=(1, 2))
    return (-q1 + beta * bc).mean()
