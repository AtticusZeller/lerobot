"""Smoke tests for ChunkedLiberoEnv. No LIBERO simulator dependency."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from gymnasium import spaces

from lerobot.rltoken.block_env import (
    ACTION_DIM,
    ACTION_HIGH,
    ACTION_LOW,
    CHUNK_REWARD_DEFAULT,
    CHUNK_REWARD_SUCCESS,
    ChunkedLiberoEnv,
)


class _FakeLiberoEnv:
    """Minimal stand-in mimicking the slice of LiberoEnv interface ChunkedLiberoEnv touches.

    ``success_at`` (None = never): inner-step index at which ``is_success`` flips True (env auto-
    terminates on that step, matching LiberoEnv.step's ``terminated = done OR is_success``).
    """

    def __init__(self, success_at: int | None = None, action_dim: int = ACTION_DIM):
        self.task = "fake_task"
        self.task_id = 0
        self.task_description = "do the thing"
        self._max_episode_steps = 100
        self.observation_space = spaces.Dict({"x": spaces.Box(low=-1, high=1, shape=(3,), dtype=np.float32)})
        self.action_dim = action_dim
        self.success_at = success_at
        self.steps = 0
        self.actions: list[np.ndarray] = []

    def reset(self, seed=None, **kw):
        self.steps = 0
        self.actions.clear()
        return ({"x": np.zeros(3, dtype=np.float32)}, {"is_success": False})

    def step(self, action: np.ndarray) -> tuple[dict, float, bool, bool, dict[str, Any]]:
        assert action.shape == (self.action_dim,), action.shape
        self.actions.append(action.copy())
        self.steps += 1
        is_success = self.success_at is not None and (self.steps - 1) == self.success_at
        terminated = is_success
        info = {"is_success": is_success, "task": self.task, "task_id": self.task_id}
        return ({"x": np.zeros(3, dtype=np.float32)}, 0.0, terminated, False, info)


def _make_chunk(c: int = 8) -> np.ndarray:
    return np.full((c, ACTION_DIM), 0.5, dtype=np.float32)


def test_no_success_runs_full_chunk_with_zero_reward():
    env = ChunkedLiberoEnv(_FakeLiberoEnv(success_at=None), chunk_size=8)
    env.reset()
    obs, reward, terminated, truncated, info = env.step(_make_chunk(8))
    assert reward == CHUNK_REWARD_DEFAULT
    assert not terminated
    assert not truncated
    assert info["inner_steps"] == 8
    assert info["is_success"] is False
    assert len(info["inner_infos"]) == 8


def test_success_mid_chunk_truncates_and_gives_reward():
    fake = _FakeLiberoEnv(success_at=3)
    env = ChunkedLiberoEnv(fake, chunk_size=8)
    env.reset()
    _, reward, terminated, _, info = env.step(_make_chunk(8))
    assert reward == CHUNK_REWARD_SUCCESS
    assert terminated
    assert info["inner_steps"] == 4  # success on step index 3 → executed 4 inner steps
    assert info["is_success"] is True
    # Ensure we did not dispatch the trailing 4 actions into a fresh episode.
    assert len(fake.actions) == 4


def test_action_clipping():
    fake = _FakeLiberoEnv(success_at=None)
    env = ChunkedLiberoEnv(fake, chunk_size=4)
    env.reset()
    chunk = np.full((4, ACTION_DIM), 5.0, dtype=np.float32)
    chunk[1] = -7.0
    env.step(chunk)
    for a in fake.actions:
        assert (a <= ACTION_HIGH + 1e-6).all()
        assert (a >= ACTION_LOW - 1e-6).all()


def test_shape_mismatch_raises():
    env = ChunkedLiberoEnv(_FakeLiberoEnv(success_at=None), chunk_size=8)
    env.reset()
    with pytest.raises(ValueError):
        env.step(np.zeros((7, ACTION_DIM), dtype=np.float32))  # wrong C
    with pytest.raises(ValueError):
        env.step(np.zeros((8, ACTION_DIM + 1), dtype=np.float32))  # wrong action_dim


def test_observation_space_forwarded():
    inner = _FakeLiberoEnv()
    env = ChunkedLiberoEnv(inner, chunk_size=8)
    assert env.observation_space is inner.observation_space
    assert env.action_space.shape == (8, ACTION_DIM)


def test_invalid_chunk_size():
    with pytest.raises(ValueError):
        ChunkedLiberoEnv(_FakeLiberoEnv(), chunk_size=0)
