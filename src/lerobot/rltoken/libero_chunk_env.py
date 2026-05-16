"""Single-task LIBERO wrapper that executes chunked action sequences."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from lerobot.envs.libero import LiberoEnv, _get_suite


class LiberoChunkEnv:
    """Wrap one LIBERO task so ``step`` consumes ``[C, action_dim]`` chunks."""

    def __init__(
        self,
        suite_name: str,
        task_index: int,
        action_dim: int,
        chunk_length: int,
        episode_length: int | None = None,
        init_states: bool = True,
        camera_name: str = "agentview_image,robot0_eye_in_hand_image",
        control_mode: str = "relative",
        observation_height: int = 360,
        observation_width: int = 360,
    ) -> None:
        self.suite_name = suite_name
        self.task_index = task_index
        self._action_dim = action_dim
        self._chunk_length = chunk_length
        suite = _get_suite(suite_name)
        self.env = LiberoEnv(
            task_suite=suite,
            task_id=task_index,
            task_suite_name=suite_name,
            episode_length=episode_length,
            camera_name=camera_name,
            init_states=init_states,
            observation_height=observation_height,
            observation_width=observation_width,
            obs_type="pixels_agent_pos",
            control_mode=control_mode,
        )
        self.task = self.env.task_description

    @property
    def action_dim(self) -> int:
        return self._action_dim

    @property
    def chunk_length(self) -> int:
        return self._chunk_length

    def reset(self) -> dict[str, Any]:
        obs, _info = self.env.reset()
        return obs

    def step(self, action_chunk: NDArray) -> tuple[dict[str, Any], NDArray, bool, dict[str, Any]]:
        action_chunk = np.asarray(action_chunk, dtype=np.float32)
        expected_shape = (self._chunk_length, self._action_dim)
        if action_chunk.shape != expected_shape:
            raise ValueError(f"Expected action_chunk shape {expected_shape}, got {action_chunk.shape}.")

        rewards = np.zeros(self._chunk_length, dtype=np.float32)
        done = False
        info: dict[str, Any] = {}
        next_obs = None
        for step_idx in range(self._chunk_length):
            next_obs, reward, terminated, truncated, info = self.env.step(action_chunk[step_idx])
            success = bool(info.get("is_success", False))
            rewards[step_idx] = 1.0 if success else float(reward)
            done = bool(terminated or truncated or success)
            if done:
                break

        info = dict(info)
        info["success"] = bool(info.get("is_success", False))
        info["steps_executed"] = step_idx + 1
        assert next_obs is not None
        return next_obs, rewards, done, info

    def close(self) -> None:
        self.env.close()
