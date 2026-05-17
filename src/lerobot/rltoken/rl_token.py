"""RL Token encoder-decoder and π0.5 visual embedding extraction.

The encoder/decoder mirrors the rlt-openpi Stage 1 implementation. The
LeRobot-specific delta is ``extract_vlm_embeddings``: per V3 plan §2.3.1,
per-task training drops the fixed language tokens and trains on visual prefix
embeddings only.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import Tensor, nn

if TYPE_CHECKING:
    from lerobot.policies.pi05.modeling_pi05 import PI05Policy

VLM_HIDDEN_DIM = 2048


class RLTokenEncoder(nn.Module):
    """Encode VLA visual embeddings into a single RL token.

    Args:
        embedding_dim: VLA hidden width. For π0.5 Gemma-2B this is 2048.
        num_layers: Number of transformer encoder layers.
        num_heads: Number of attention heads.
    """

    def __init__(
        self,
        embedding_dim: int = VLM_HIDDEN_DIM,
        num_layers: int = 2,
        num_heads: int = 8,
    ) -> None:
        super().__init__()
        self.e_rl = nn.Parameter(torch.randn(1, 1, embedding_dim) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=4 * embedding_dim,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, z: Tensor, pad_mask: Tensor) -> Tensor:
        """Encode embeddings into ``z_rl``.

        Args:
            z: Visual VLA embeddings, shape ``[B, M, D]``.
            pad_mask: Boolean mask, shape ``[B, M]``. True means valid token.

        Returns:
            RL token, shape ``[B, D]``.
        """
        batch_size = z.shape[0]
        e_rl = self.e_rl.expand(batch_size, -1, -1)
        tokens = torch.cat([z, e_rl], dim=1)  # [B, M + 1, D]

        rl_mask = torch.ones(batch_size, 1, dtype=torch.bool, device=z.device)
        extended_pad_mask = torch.cat([pad_mask.bool(), rl_mask], dim=1)  # [B, M + 1]
        ignore_mask = ~extended_pad_mask

        out = self.transformer(tokens, src_key_padding_mask=ignore_mask)
        return out[:, -1, :]  # [B, D]


class RLTokenDecoder(nn.Module):
    """Reconstruct visual VLA embeddings from ``z_rl``.

    The decoder uses teacher-forced autoregression: target input is
    ``[z_rl, z_1, ..., z_{M-1}]`` and output position ``i`` predicts ``z_i``.

    Args:
        embedding_dim: VLA hidden width.
        num_layers: Number of transformer decoder layers.
        num_heads: Number of attention heads.
    """

    def __init__(
        self,
        embedding_dim: int = VLM_HIDDEN_DIM,
        num_layers: int = 2,
        num_heads: int = 8,
    ) -> None:
        super().__init__()
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=4 * embedding_dim,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.transformer = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.h_phi = nn.Linear(embedding_dim, embedding_dim)

    def forward(self, z_rl: Tensor, z: Tensor, pad_mask: Tensor) -> Tensor:
        """Reconstruct embeddings from an RL token.

        Args:
            z_rl: RL token, shape ``[B, D]``.
            z: Original visual VLA embeddings, shape ``[B, M, D]``.
            pad_mask: Boolean mask, shape ``[B, M]``. True means valid token.

        Returns:
            Reconstructed embeddings, shape ``[B, M, D]``.
        """
        tgt = torch.cat([z_rl.unsqueeze(1), z[:, :-1, :]], dim=1)  # [B, M, D]
        seq_len = tgt.shape[1]
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, dtype=torch.bool, device=tgt.device),
            diagonal=1,
        )
        memory = z_rl.unsqueeze(1)  # [B, 1, D]

        out = self.transformer(
            tgt,
            memory,
            tgt_mask=causal_mask,
            tgt_key_padding_mask=~pad_mask.bool(),
        )
        return self.h_phi(out)  # [B, M, D]


class RLTokenModel(nn.Module):
    """Combined RL token encoder-decoder for Stage 1 training.

    Args:
        embedding_dim: VLA hidden width.
        encoder_layers: Number of encoder transformer layers.
        encoder_heads: Number of encoder attention heads.
        decoder_layers: Number of decoder transformer layers.
        decoder_heads: Number of decoder attention heads.
    """

    def __init__(
        self,
        embedding_dim: int = VLM_HIDDEN_DIM,
        encoder_layers: int = 2,
        encoder_heads: int = 8,
        decoder_layers: int = 2,
        decoder_heads: int = 8,
    ) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.encoder_layers = encoder_layers
        self.encoder_heads = encoder_heads
        self.decoder_layers = decoder_layers
        self.decoder_heads = decoder_heads
        self.encoder = RLTokenEncoder(embedding_dim, encoder_layers, encoder_heads)
        self.decoder = RLTokenDecoder(embedding_dim, decoder_layers, decoder_heads)

    def forward(self, z: Tensor, pad_mask: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Encode, decode, and compute masked reconstruction loss.

        Args:
            z: Visual VLA embeddings, shape ``[B, M, D]``.
            pad_mask: Boolean mask, shape ``[B, M]``. True means valid token.

        Returns:
            ``(loss, z_rl, z_hat)`` where loss is scalar, ``z_rl`` is
            ``[B, D]``, and ``z_hat`` is ``[B, M, D]``.
        """
        z = z.detach()
        pad_mask = pad_mask.bool()

        z_rl = self.encoder(z, pad_mask)
        z_hat = self.decoder(z_rl, z, pad_mask)

        mse = (z_hat - z).pow(2).mean(dim=-1)  # [B, M]
        mask = pad_mask.to(mse.dtype)
        loss = (mse * mask).sum() / mask.sum().clamp_min(1.0)
        return loss, z_rl, z_hat

    @torch.no_grad()
    def encode(self, z: Tensor, pad_mask: Tensor) -> Tensor:
        """Extract ``z_rl`` for inference.

        Args:
            z: Visual VLA embeddings, shape ``[B, M, D]``.
            pad_mask: Boolean mask, shape ``[B, M]``. True means valid token.

        Returns:
            RL token, shape ``[B, D]``.
        """
        return self.encoder(z, pad_mask.bool())


@torch.no_grad()
def extract_vlm_embeddings(policy: PI05Policy, batch: dict[str, Tensor]) -> tuple[Tensor, Tensor]:
    """Run π0.5 prefix forward and return image-only hidden states.

    ``PI05Pytorch.embed_prefix`` lays tokens out as image patches followed by
    language tokens. Per-task RL Token training keeps the instruction fixed, so
    the language suffix is removed before training the encoder.

    Args:
        policy: Frozen π0.5 policy in eval mode.
        batch: Preprocessed LeRobot batch containing images and language tokens.

    Returns:
        ``(z_vis, pad_vis)`` where ``z_vis`` is ``[B, M_vis, 2048]`` and
        ``pad_vis`` is ``[B, M_vis]`` with True for valid visual tokens.
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
    num_lang_tokens = tokens.shape[1]
    num_visual_tokens = prefix_embs.shape[1] - num_lang_tokens
    if num_visual_tokens <= 0:
        raise ValueError(
            f"Expected at least one visual token, got prefix_len={prefix_embs.shape[1]} "
            f"and language_len={num_lang_tokens}."
        )

    prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
    position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
    language_model = pi05_model.paligemma_with_expert.paligemma.model.language_model
    language_model.config._attn_implementation = "eager"
    model_dtype = language_model.layers[0].self_attn.q_proj.weight.dtype
    if prefix_embs.dtype != model_dtype:
        prefix_embs = prefix_embs.to(dtype=model_dtype)
    attn_4d = pi05_model._prepare_attention_masks_4d(prefix_att_2d_masks).to(dtype=model_dtype)  # noqa: SLF001

    (prefix_output, _), _ = pi05_model.paligemma_with_expert.forward(
        attention_mask=attn_4d,
        position_ids=position_ids,
        past_key_values=None,
        inputs_embeds=[prefix_embs, None],
        use_cache=False,
    )

    z_vis = prefix_output[:, :num_visual_tokens, :].detach().to(torch.float32)  # [B, M_vis, D]
    pad_vis = prefix_pad_masks[:, :num_visual_tokens].detach().bool()  # [B, M_vis]
    return z_vis, pad_vis
