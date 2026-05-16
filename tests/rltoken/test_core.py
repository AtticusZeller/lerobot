"""Tests for migrated RL Token core modules."""

from __future__ import annotations

import numpy as np
import torch

from lerobot.rltoken.actor import Actor
from lerobot.rltoken.critic import TwinQCritic
from lerobot.rltoken.online_trainer import OnlineRLConfig, OnlineRLTrainer
from lerobot.rltoken.replay_buffer import ReplayBuffer
from lerobot.rltoken.rl_token import RLTokenModel
from lerobot.rltoken.td3_utils import actor_loss, compute_td_target, critic_loss


def test_rl_token_model_shapes_and_stop_gradient() -> None:
    model = RLTokenModel(embedding_dim=32, encoder_layers=1, encoder_heads=4, decoder_layers=1, decoder_heads=4)
    z = torch.randn(4, 10, 32, requires_grad=True)
    pad_mask = torch.ones(4, 10, dtype=torch.bool)
    pad_mask[:, -2:] = False

    loss, z_rl, z_hat = model(z, pad_mask)
    loss.backward()

    assert loss.shape == ()
    assert z_rl.shape == (4, 32)
    assert z_hat.shape == (4, 10, 32)
    assert z.grad is None


def test_actor_critic_and_td3_utils() -> None:
    state_dim = 34
    action_chunk_dim = 6
    chunk_length = 3
    batch_size = 8
    actor = Actor(state_dim=state_dim, action_chunk_dim=action_chunk_dim, ref_dropout=0.0)
    critic = TwinQCritic(state_dim=state_dim, action_chunk_dim=action_chunk_dim)

    x = torch.randn(batch_size, state_dim)
    a_tilde = torch.randn(batch_size, action_chunk_dim)
    a = actor(x, a_tilde)
    q1, q2 = critic(x, a)
    rewards = torch.randn(batch_size, chunk_length)
    dones = torch.zeros(batch_size, 1)
    target = compute_td_target(rewards, dones, x, a_tilde, actor, critic, gamma=0.99, chunk_length=chunk_length)

    assert a.shape == (batch_size, action_chunk_dim)
    assert q1.shape == (batch_size, 1)
    assert q2.shape == (batch_size, 1)
    assert target.shape == (batch_size, 1)
    assert critic_loss(q1, q2, target).shape == ()
    assert actor_loss(critic.q_min(x, a), a, a_tilde, beta=0.5).shape == ()


def test_replay_buffer_sample() -> None:
    buffer = ReplayBuffer(capacity=8, state_dim=5, action_chunk_dim=6, chunk_length=3)
    for _ in range(4):
        buffer.add(
            x=np.zeros(5, dtype=np.float32),
            a=np.zeros(6, dtype=np.float32),
            a_tilde=np.zeros(6, dtype=np.float32),
            rewards=np.ones(3, dtype=np.float32),
            next_x=np.ones(5, dtype=np.float32),
            done=0.0,
        )

    batch = buffer.sample(batch_size=6, device="cpu")
    assert buffer.size == 4
    assert batch["x"].shape == (6, 5)
    assert batch["rewards"].shape == (6, 3)


class MockVLA:
    def __init__(self, embedding_dim: int, action_dim: int, chunk_length: int, proprio_dim: int) -> None:
        self.embedding_dim = embedding_dim
        self.action_dim = action_dim
        self.chunk_length = chunk_length
        self.proprio_dim = proprio_dim

    def preprocess_obs(self, obs, task: str):
        return obs

    def extract_embeddings(self, batch):
        return torch.randn(1, 5, self.embedding_dim), torch.ones(1, 5, dtype=torch.bool)

    def get_rl_chunk_reference(self, batch, chunk_length: int):
        return torch.zeros(1, chunk_length, self.action_dim)

    def proprioception(self, batch):
        return torch.zeros(1, self.proprio_dim)


class DummyChunkEnv:
    task = "dummy task"

    def __init__(self, action_dim: int, chunk_length: int) -> None:
        self.action_dim = action_dim
        self.chunk_length = chunk_length
        self._chunks = 0

    def reset(self):
        self._chunks = 0
        return {"state": np.zeros(1, dtype=np.float32)}

    def step(self, action_chunk):
        self._chunks += 1
        rewards = np.zeros(self.chunk_length, dtype=np.float32)
        done = self._chunks >= 2
        if done:
            rewards[-1] = 1.0
        return {"state": np.zeros(1, dtype=np.float32)}, rewards, done, {
            "success": done,
            "steps_executed": self.chunk_length,
        }


def test_online_trainer_smoke(tmp_path) -> None:
    cfg = OnlineRLConfig(
        embedding_dim=32,
        proprio_dim=2,
        action_dim=2,
        chunk_length=3,
        actor_hidden=32,
        actor_hidden_layers=1,
        ref_action_dropout=0.0,
        buffer_capacity=32,
        batch_size=4,
        warmup_steps=2,
        max_env_steps=12,
        update_to_data_ratio=2,
        save_dir=str(tmp_path),
        run_name="smoke",
        save_every=999,
    )
    vla = MockVLA(cfg.embedding_dim, cfg.action_dim, cfg.chunk_length, cfg.proprio_dim)
    rl_token = RLTokenModel(embedding_dim=32, encoder_layers=1, encoder_heads=4, decoder_layers=1, decoder_heads=4)
    trainer = OnlineRLTrainer(cfg, vla, rl_token, device="cpu")
    env = DummyChunkEnv(cfg.action_dim, cfg.chunk_length)

    logged = []
    trainer.train(env, log_fn=logged.append)

    assert trainer.replay_buffer.size >= cfg.warmup_steps
    assert trainer._total_updates > 0
    assert logged
