"""LeRobot π0.5 adapter exposing the interface used by RL Token rollouts."""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor

from lerobot.envs.utils import preprocess_observation
from lerobot.rltoken.rl_token import extract_vlm_embeddings
from lerobot.utils.constants import OBS_STATE


class LeRobotPI05Adapter:
    """Wrap a LeRobot ``PI05Policy`` for rlt-openpi-style online RL code."""

    def __init__(
        self,
        policy,
        env_preprocessor,
        preprocessor,
        postprocessor,
        device: torch.device | str,
        proprio_dim: int = 8,
    ) -> None:
        self.policy = policy
        self.env_preprocessor = env_preprocessor
        self.preprocessor = preprocessor
        self.postprocessor = postprocessor
        self.device = torch.device(device)
        self.proprio_dim = proprio_dim

    def preprocess_obs(self, obs: dict[str, Any], task: str) -> dict[str, Tensor]:
        """Convert raw LIBERO obs into a π0.5-ready batch."""
        batch = preprocess_observation(obs)
        batch["task"] = [task]
        batch = self.env_preprocessor(batch)
        return self.preprocessor(batch)

    @torch.no_grad()
    def extract_embeddings(self, batch: dict[str, Tensor]) -> tuple[Tensor, Tensor]:
        z_vis, pad_vis = extract_vlm_embeddings(self.policy, batch)
        return z_vis.to(self.device), pad_vis.to(self.device)

    @torch.no_grad()
    def get_rl_chunk_reference(self, batch: dict[str, Tensor], chunk_length: int) -> Tensor:
        """Return postprocessed VLA reference actions, shape ``[B, C, action_dim]``."""
        action_chunk = self.policy.predict_action_chunk(batch)[:, :chunk_length]
        action_chunk = self.postprocessor(action_chunk)
        return action_chunk.to(self.device, dtype=torch.float32)

    def proprioception(self, batch: dict[str, Tensor]) -> Tensor:
        """Return normalized proprioception used in the RL state, shape ``[B, proprio_dim]``."""
        return batch[OBS_STATE][:, : self.proprio_dim].to(self.device, dtype=torch.float32)
