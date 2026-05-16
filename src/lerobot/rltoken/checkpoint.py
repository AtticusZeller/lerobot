"""Checkpoint helpers for RL Token Stage 1 and Stage 2."""

from __future__ import annotations

import logging

import torch
from safetensors import safe_open

from lerobot.rltoken.rl_token import VLM_HIDDEN_DIM, RLTokenModel

log = logging.getLogger(__name__)


def _metadata_int(metadata: dict[str, str] | None, key: str, default: int) -> int:
    if metadata is None or key not in metadata:
        return default
    return int(metadata[key])


def load_rl_token_model(ckpt_path: str, device: str | torch.device = "cuda") -> RLTokenModel:
    """Load a frozen ``RLTokenModel`` from safetensors or rlt-openpi ``.pt``."""
    if ckpt_path.endswith(".safetensors"):
        with safe_open(ckpt_path, framework="pt", device="cpu") as handle:
            metadata = handle.metadata()
            state = {key: handle.get_tensor(key) for key in handle}
        model = RLTokenModel(
            embedding_dim=_metadata_int(metadata, "embedding_dim", VLM_HIDDEN_DIM),
            encoder_layers=_metadata_int(metadata, "encoder_layers", 2),
            encoder_heads=_metadata_int(metadata, "encoder_heads", 8),
            decoder_layers=_metadata_int(metadata, "decoder_layers", 2),
            decoder_heads=_metadata_int(metadata, "decoder_heads", 8),
        )
        model.load_state_dict(state)
        step = metadata.get("step", "?") if metadata else "?"
    else:
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        saved_config = ckpt["config"]
        model = RLTokenModel(
            embedding_dim=saved_config.embedding_dim,
            encoder_layers=saved_config.encoder_layers,
            encoder_heads=saved_config.encoder_heads,
            decoder_layers=saved_config.decoder_layers,
            decoder_heads=saved_config.decoder_heads,
        )
        model.load_state_dict(ckpt["model"])
        step = ckpt.get("step", "?")
        del ckpt

    model = model.to(device)
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    log.info("Loaded RL token model from %s (step=%s)", ckpt_path, step)
    return model
