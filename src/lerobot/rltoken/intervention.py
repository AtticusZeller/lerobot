"""No-op intervention interface kept compatible with rlt-openpi rollout code."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from numpy.typing import NDArray


@dataclass
class InterventionResult:
    action_chunk: NDArray
    next_obs: dict[str, Any]
    rewards: NDArray
    done: bool
    info: dict[str, Any] = field(default_factory=dict)


class InterventionManager:
    """Stub intervention manager for simulation runs."""

    def check_intervention(self) -> bool:
        return False

    def get_human_action(self, action_dim: int, chunk_length: int) -> InterventionResult | None:
        return None
