"""Chunk-level replay buffer for online RL Token training."""

from __future__ import annotations

import numpy as np
import torch
from numpy.typing import NDArray


class ReplayBuffer:
    """Fixed-capacity circular replay buffer for chunk transitions."""

    def __init__(self, capacity: int, state_dim: int, action_chunk_dim: int, chunk_length: int) -> None:
        self.capacity = capacity
        self.chunk_length = chunk_length
        self._ptr = 0
        self._size = 0
        self._x = np.zeros((capacity, state_dim), dtype=np.float32)
        self._a = np.zeros((capacity, action_chunk_dim), dtype=np.float32)
        self._a_tilde = np.zeros((capacity, action_chunk_dim), dtype=np.float32)
        self._rewards = np.zeros((capacity, chunk_length), dtype=np.float32)
        self._next_x = np.zeros((capacity, state_dim), dtype=np.float32)
        self._dones = np.zeros((capacity, 1), dtype=np.float32)

    @property
    def size(self) -> int:
        return self._size

    def add(
        self,
        x: NDArray,
        a: NDArray,
        a_tilde: NDArray,
        rewards: NDArray,
        next_x: NDArray,
        done: float,
    ) -> None:
        self._x[self._ptr] = x
        self._a[self._ptr] = a
        self._a_tilde[self._ptr] = a_tilde
        self._rewards[self._ptr] = rewards
        self._next_x[self._ptr] = next_x
        self._dones[self._ptr] = done
        self._ptr = (self._ptr + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def state_dict(self) -> dict[str, object]:
        size = self._size
        return {
            "ptr": self._ptr,
            "size": size,
            "x": self._x[:size].copy(),
            "a": self._a[:size].copy(),
            "a_tilde": self._a_tilde[:size].copy(),
            "rewards": self._rewards[:size].copy(),
            "next_x": self._next_x[:size].copy(),
            "dones": self._dones[:size].copy(),
        }

    def load_state_dict(self, state: dict[str, object]) -> None:
        size = int(state["size"])
        self._ptr = int(state["ptr"])
        self._size = size
        self._x[:size] = state["x"]
        self._a[:size] = state["a"]
        self._a_tilde[:size] = state["a_tilde"]
        self._rewards[:size] = state["rewards"]
        self._next_x[:size] = state["next_x"]
        self._dones[:size] = state["dones"]

    def sample(self, batch_size: int, device: str | torch.device = "cpu") -> dict[str, torch.Tensor]:
        if self._size == 0:
            raise RuntimeError("Cannot sample from an empty replay buffer.")
        indices = np.random.randint(0, self._size, size=batch_size)
        return {
            "x": torch.as_tensor(self._x[indices], device=device),
            "a": torch.as_tensor(self._a[indices], device=device),
            "a_tilde": torch.as_tensor(self._a_tilde[indices], device=device),
            "rewards": torch.as_tensor(self._rewards[indices], device=device),
            "next_x": torch.as_tensor(self._next_x[indices], device=device),
            "dones": torch.as_tensor(self._dones[indices], device=device),
        }
