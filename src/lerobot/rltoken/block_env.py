"""阶段二b：块级 LIBERO 环境 wrapper。

把 ``LiberoEnv`` 的单步接口折叠为块级接口：一次 ``step(action_chunk)`` 内部执行 ``chunk_size``
个 inner step，accumulate 一个稀疏二值成功奖励，对齐 ``docs/rltoken_plan.md`` §2.4 块级 Bellman
设定。

设计要点：
    - 内步 reward 始终 0（LIBERO 不做奖励 shaping），只用 ``info["is_success"]`` 触发成功；
      因此 chunk reward = 1.0 iff 任一 inner step 成功，否则 0.0。
    - LiberoEnv.step 在 ``terminated=True`` 时**会自动在内部 reset 到下一回合**（line 361）；
      为避免在新回合的初始状态上继续执行 chunk 中剩余的动作，本 wrapper **遇到 terminated 即
      break**，info 报告 ``inner_steps``。
    - 因此返回的 ``next_obs`` 可能是新回合的 reset obs；上层 RL 训练应根据 ``terminated`` 决定
      bootstrap 时是否乘 ``(1 - done)``。
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

if TYPE_CHECKING:
    from lerobot.envs.libero import LiberoEnv

# Mirror constants from lerobot.envs.libero so that this module — and its smoke tests — can be
# imported without the heavyweight `libero` simulator dependency. Keep these in sync.
ACTION_DIM = 7
ACTION_LOW = -1.0
ACTION_HIGH = 1.0

CHUNK_REWARD_SUCCESS = 1.0
CHUNK_REWARD_DEFAULT = 0.0


class ChunkedLiberoEnv(gym.Env):
    """Wrap a single ``LiberoEnv`` so that one outer step consumes a length-``C`` action chunk.

    The action space becomes a ``Box`` of shape ``(chunk_size, action_dim)``. The reward is a single
    scalar per chunk (binary: 1.0 if any inner step's ``is_success`` flipped True, else 0.0). The
    returned ``info`` includes ``inner_infos`` (list of per-step infos), ``inner_steps`` (how many
    inner steps were actually executed before early-terminate), ``is_success`` (aggregate), and
    ``task_id``.
    """

    metadata = {"render_modes": ["rgb_array"], "render_fps": 80}

    def __init__(self, env: LiberoEnv, chunk_size: int = 8):
        super().__init__()
        if chunk_size <= 0:
            raise ValueError(f"chunk_size must be positive, got {chunk_size}")
        self.env = env
        self.chunk_size = chunk_size
        self.observation_space = env.observation_space
        self.action_space = spaces.Box(
            low=ACTION_LOW,
            high=ACTION_HIGH,
            shape=(chunk_size, ACTION_DIM),
            dtype=np.float32,
        )

    @property
    def task(self) -> str:
        return self.env.task

    @property
    def task_id(self) -> int:
        return self.env.task_id

    @property
    def task_description(self) -> str:
        return self.env.task_description

    @property
    def _max_episode_steps(self) -> int:
        return self.env._max_episode_steps

    def reset(self, seed: int | None = None, **kwargs) -> tuple[dict, dict[str, Any]]:
        return self.env.reset(seed=seed, **kwargs)

    def render(self):
        return self.env.render()

    def step(self, action: np.ndarray) -> tuple[dict, float, bool, bool, dict[str, Any]]:
        """Execute ``chunk_size`` inner steps; collapse to a single chunk transition.

        ``action`` shape: ``(chunk_size, action_dim)``. Named ``action`` (not ``action_chunk``) to
        match ``gym.Env.step``'s base signature.
        """
        action_chunk = np.asarray(action, dtype=np.float32)
        if action_chunk.shape != (self.chunk_size, ACTION_DIM):
            raise ValueError(
                f"Expected action_chunk shape ({self.chunk_size}, {ACTION_DIM}), got {action_chunk.shape}"
            )
        action_chunk = np.clip(action_chunk, ACTION_LOW, ACTION_HIGH)

        inner_infos: list[dict[str, Any]] = []
        any_success = False
        terminated = False
        last_obs: dict = {}
        executed = 0
        for i in range(self.chunk_size):
            last_obs, _inner_reward, terminated, _truncated, info = self.env.step(action_chunk[i])
            inner_infos.append(info)
            executed = i + 1
            if info.get("is_success", False):
                any_success = True
            if terminated:
                # LiberoEnv.step auto-resets on terminate; stop dispatching more chunk actions
                # into what would be a fresh episode.
                break

        chunk_reward = CHUNK_REWARD_SUCCESS if any_success else CHUNK_REWARD_DEFAULT
        info_out: dict[str, Any] = {
            "inner_infos": inner_infos,
            "inner_steps": executed,
            "is_success": any_success,
            "task": self.task,
            "task_id": self.task_id,
        }
        return last_obs, float(chunk_reward), bool(terminated), False, info_out
