"""Smoke tests for TD3 actor / critic / losses / Polyak update."""

from __future__ import annotations

import copy

import torch

from lerobot.rltoken.td3 import (
    TD3Actor,
    TD3Critic,
    soft_update_target,
    td3_actor_loss,
    td3_critic_loss,
)

# Compact dims so the tests run fast on CPU.
B = 5
Z = 16
P = 9
A = 7
C = 4
HIDDEN = 32


def _rand_inputs(seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    z = torch.randn(B, Z, generator=g)
    prop = torch.randn(B, P, generator=g)
    ref = torch.randn(B, C, A, generator=g).clamp(-0.5, 0.5)
    return z, prop, ref


def test_actor_forward_shape_and_dtype():
    actor = TD3Actor(z_dim=Z, prop_dim=P, action_dim=A, chunk_size=C, hidden=HIDDEN)
    z, prop, ref = _rand_inputs()
    drop = torch.zeros(B, dtype=torch.bool)
    out = actor(z, prop, ref, drop_ref=drop)
    assert out.shape == (B, C, A)
    assert out.dtype == torch.float32


def test_actor_zero_init_reproduces_reference():
    """Residual head zero-init ⇒ initial action == reference (drop_ref disabled)."""
    actor = TD3Actor(z_dim=Z, prop_dim=P, action_dim=A, chunk_size=C, hidden=HIDDEN)
    actor.eval()
    z, prop, ref = _rand_inputs()
    drop = torch.zeros(B, dtype=torch.bool)
    with torch.no_grad():
        out = actor(z, prop, ref, drop_ref=drop)
    # action bounds are [-1, 1] and ref is in [-0.5, 0.5] so clipping never kicks in here.
    assert torch.allclose(out, ref, atol=1e-6)


def test_actor_dropped_ref_zero_init_is_zero():
    """With ref dropped + residual head zero-init, ``ref_chunk + 0 == 0`` (since ref_input is zeroed
    on the actor side, but ref_chunk is still added at the end). So the output equals ref_chunk
    regardless of drop_ref — the dropout only hides the ref from the trunk input, not from the
    residual base. This documents and pins the intended semantics.
    """
    actor = TD3Actor(z_dim=Z, prop_dim=P, action_dim=A, chunk_size=C, hidden=HIDDEN)
    actor.eval()
    z, prop, ref = _rand_inputs()
    drop = torch.ones(B, dtype=torch.bool)
    with torch.no_grad():
        out = actor(z, prop, ref, drop_ref=drop)
    assert torch.allclose(out, ref, atol=1e-6)


def test_critic_forward_shape():
    critic = TD3Critic(z_dim=Z, prop_dim=P, action_dim=A, chunk_size=C, hidden=HIDDEN)
    z, prop, ref = _rand_inputs()
    a = ref.clone()
    out = critic(z, prop, a)
    assert out.shape == (B, 2)
    assert out.dtype == torch.float32


def test_critic_loss_grad_flow():
    critic = TD3Critic(z_dim=Z, prop_dim=P, action_dim=A, chunk_size=C, hidden=HIDDEN)
    critic_target = copy.deepcopy(critic).eval()
    for p in critic_target.parameters():
        p.requires_grad_(False)
    actor_target = TD3Actor(z_dim=Z, prop_dim=P, action_dim=A, chunk_size=C, hidden=HIDDEN).eval()
    for p in actor_target.parameters():
        p.requires_grad_(False)

    z, prop, ref = _rand_inputs(seed=1)
    z_n, prop_n, ref_n = _rand_inputs(seed=2)
    a = ref.clone()
    reward = torch.zeros(B)
    reward[0] = 1.0
    done = torch.zeros(B)
    done[1] = 1.0

    loss = td3_critic_loss(
        critic,
        critic_target,
        actor_target,
        z,
        prop,
        ref,
        a,
        z_n,
        prop_n,
        ref_n,
        reward,
        done,
        gamma=0.99,
        chunk_size=C,
    )
    assert loss.dim() == 0 and torch.isfinite(loss)
    loss.backward()

    q1_grad = critic.q1[0].weight.grad
    assert q1_grad is not None and q1_grad.abs().sum() > 0
    # Target networks must stay frozen.
    for p in critic_target.parameters():
        assert p.grad is None
    for p in actor_target.parameters():
        assert p.grad is None


def test_actor_loss_grad_flow_and_critic_untouched():
    """At init the residual head is zero so trunk grad is zero (gradient blocked through
    the zero linear); but the residual head itself receives grad from -Q1(a_pi).
    After one optimizer step the head is non-zero and trunk grad becomes non-zero."""
    actor = TD3Actor(z_dim=Z, prop_dim=P, action_dim=A, chunk_size=C, hidden=HIDDEN)
    critic = TD3Critic(z_dim=Z, prop_dim=P, action_dim=A, chunk_size=C, hidden=HIDDEN)

    z, prop, ref = _rand_inputs(seed=3)
    drop = torch.zeros(B, dtype=torch.bool)

    # Snapshot critic weight to assert it doesn't move when we only step the actor.
    critic_w0 = critic.q1[0].weight.detach().clone()

    opt_a = torch.optim.SGD(actor.parameters(), lr=0.1)
    loss = td3_actor_loss(actor, critic, z, prop, ref, beta=0.5, drop_ref=drop)
    assert torch.isfinite(loss)
    opt_a.zero_grad()
    loss.backward()

    # Residual head receives gradient from -Q1(a_pi); trunk grad is zero at init by design.
    head_grad = actor.residual_head.weight.grad
    assert head_grad is not None and head_grad.abs().sum() > 0
    trunk_grad_init = actor.trunk[0].weight.grad
    assert trunk_grad_init is None or trunk_grad_init.abs().sum() == 0

    opt_a.step()  # head now non-zero
    # Second backward unblocks the trunk.
    loss2 = td3_actor_loss(actor, critic, z, prop, ref, beta=0.5, drop_ref=drop)
    opt_a.zero_grad()
    loss2.backward()
    trunk_grad = actor.trunk[0].weight.grad
    assert trunk_grad is not None and trunk_grad.abs().sum() > 0

    # Actor stepped; critic must not have moved because the optimiser only owns actor params.
    assert torch.allclose(critic.q1[0].weight, critic_w0)


def test_soft_update_midpoint():
    online = TD3Critic(z_dim=Z, prop_dim=P, action_dim=A, chunk_size=C, hidden=HIDDEN)
    target = copy.deepcopy(online)
    # Perturb online so we can detect motion.
    with torch.no_grad():
        for p in online.parameters():
            p.add_(1.0)

    snapshot_target = [p.detach().clone() for p in target.parameters()]
    snapshot_online = [p.detach().clone() for p in online.parameters()]

    soft_update_target(target, online, tau=0.5)

    for tp, t0, o0 in zip(target.parameters(), snapshot_target, snapshot_online, strict=True):
        expected = 0.5 * t0 + 0.5 * o0
        assert torch.allclose(tp.data, expected, atol=1e-6)
