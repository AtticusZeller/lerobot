"""RL Token 编码器-解码器 + VLM 嵌入提取。

设计：见 ``docs/rltoken_plan.md`` §2.3。

- 编码器：M 个 2048D 词元 → 单个 256D ``z_rl``（CLS-pooling Transformer）
- 解码器：``z_rl`` → 重建 M 个 2048D 词元（learnable queries + cross-attention）
- ``extract_vlm_embeddings(policy, batch)``：跑 π0.5 prefix forward，截取 paligemma 最后一层
  ``last_hidden_state``（shape ``(B, M, vlm_hidden_dim)``）。冻结主干，no_grad。
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import Tensor, nn

if TYPE_CHECKING:
    from lerobot.policies.pi05.modeling_pi05 import PI05Policy

VLM_HIDDEN_DIM = 2048  # gemma_2b width — see PI05Pytorch.__init__ paligemma_config.width


class RLTokenEncoder(nn.Module):
    """Compress VLM embeddings ``z_{1:M} (B, M, d_vlm)`` into a single bottleneck ``z_rl (B, z_dim)``.

    Architecture: project ``d_vlm → d_model``, concat a learnable CLS token, run ``n_layers``
    Transformer encoder layers, take the CLS slot, project ``d_model → z_dim``. The CLS pooling
    is equivalent to cross-attention from a single query into the M tokens (n_layers=2 is enough
    given the input is already strongly contextualized by paligemma).
    """

    def __init__(
        self,
        vlm_hidden_dim: int = VLM_HIDDEN_DIM,
        z_dim: int = 256,
        d_model: int = 256,
        n_layers: int = 2,
        n_heads: int = 8,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.input_proj = nn.Linear(vlm_hidden_dim, d_model)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.normal_(self.cls_token, std=0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.output_proj = nn.Linear(d_model, z_dim)

    def forward(self, z_tokens: Tensor, key_padding_mask: Tensor | None = None) -> Tensor:
        """``z_tokens``: ``(B, M, vlm_hidden_dim)``. ``key_padding_mask``: ``(B, M)`` bool, True = pad."""
        b = z_tokens.shape[0]
        h = self.input_proj(z_tokens)
        cls = self.cls_token.expand(b, -1, -1)
        h = torch.cat([cls, h], dim=1)
        if key_padding_mask is not None:
            cls_pad = torch.zeros(b, 1, dtype=key_padding_mask.dtype, device=key_padding_mask.device)
            key_padding_mask = torch.cat([cls_pad, key_padding_mask], dim=1)
        h = self.encoder(h, src_key_padding_mask=key_padding_mask)
        return self.output_proj(h[:, 0])


class RLTokenDecoder(nn.Module):
    """Reconstruct ``z_{1:M}`` from ``z_rl``.

    Architecture: learnable query embedding of length ``max_seq_len`` at ``d_model`` width, run
    ``n_layers`` Transformer decoder layers with cross-attention into a single memory slot that
    holds the projected ``z_rl``, project ``d_model → d_vlm``. Only the first ``M`` outputs (for
    the actual sequence length) are used in the reconstruction loss.
    """

    def __init__(
        self,
        vlm_hidden_dim: int = VLM_HIDDEN_DIM,
        z_dim: int = 256,
        d_model: int = 256,
        n_layers: int = 2,
        n_heads: int = 8,
        max_seq_len: int = 512,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.max_seq_len = max_seq_len
        self.query_embed = nn.Parameter(torch.zeros(1, max_seq_len, d_model))
        nn.init.normal_(self.query_embed, std=0.02)
        self.memory_proj = nn.Linear(z_dim, d_model)
        layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.decoder = nn.TransformerDecoder(layer, num_layers=n_layers)
        self.output_proj = nn.Linear(d_model, vlm_hidden_dim)

    def forward(self, z_rl: Tensor, seq_len: int) -> Tensor:
        """``z_rl``: ``(B, z_dim)``. Returns ``(B, seq_len, vlm_hidden_dim)``."""
        if seq_len > self.max_seq_len:
            raise ValueError(f"seq_len {seq_len} exceeds decoder max_seq_len {self.max_seq_len}")
        b = z_rl.shape[0]
        queries = self.query_embed[:, :seq_len].expand(b, -1, -1)
        memory = self.memory_proj(z_rl).unsqueeze(1)  # (B, 1, d_model)
        h = self.decoder(queries, memory)
        return self.output_proj(h)


@torch.no_grad()
def extract_vlm_embeddings(policy: PI05Policy, batch: dict[str, Tensor]) -> tuple[Tensor, Tensor]:
    """Run π0.5's prefix forward and return ``last_hidden_state`` from the paligemma language model.

    Returns ``(z_tokens, pad_mask)``:
      - ``z_tokens``: ``(B, M, VLM_HIDDEN_DIM)`` — the $z_{1:M}$ from plan §2.3
      - ``pad_mask``: ``(B, M)`` bool, True where the position is real (non-padding)
    """
    from lerobot.policies.pi05.modeling_pi05 import (
        OBS_LANGUAGE_ATTENTION_MASK,
        OBS_LANGUAGE_TOKENS,
        make_att_2d_masks,
    )

    policy.eval()
    pi05_model = policy.model
    images, img_masks = policy._preprocess_images(batch)  # noqa: SLF001
    tokens = batch[OBS_LANGUAGE_TOKENS]
    masks = batch[OBS_LANGUAGE_ATTENTION_MASK]

    prefix_embs, prefix_pad_masks, prefix_att_masks = pi05_model.embed_prefix(
        images, img_masks, tokens, masks
    )
    prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
    position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
    attn_4d = pi05_model._prepare_attention_masks_4d(prefix_att_2d_masks)  # noqa: SLF001

    (prefix_output, _), _ = pi05_model.paligemma_with_expert.forward(
        attention_mask=attn_4d,
        position_ids=position_ids,
        past_key_values=None,
        inputs_embeds=[prefix_embs, None],
        use_cache=False,
    )
    return prefix_output.detach().to(torch.float32), prefix_pad_masks.detach()


def reconstruction_loss(z_tokens: Tensor, z_hat: Tensor, pad_mask: Tensor) -> Tensor:
    """Masked MSE: ``L_ro = E_{i: pad_mask_i=1} || z_hat_i - sg(z_i) ||^2``."""
    err = (z_hat - z_tokens.detach()).pow(2).mean(dim=-1)  # (B, M)
    mask = pad_mask.to(err.dtype)
    return (err * mask).sum() / mask.sum().clamp_min(1.0)
