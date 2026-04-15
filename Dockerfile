# =============================================================================
# LeRobot Inference Container — standalone, no .devcontainer needed
# Multi-stage: builder (devel) → runtime (small)
# Ref: https://docs.astral.sh/uv/guides/integration/docker/
# =============================================================================

# ---------------------------------------------------------------------------
# Stage 1: Builder
# ---------------------------------------------------------------------------
FROM nvidia/cuda:12.4.0-devel-ubuntu22.04 AS builder

ARG DEBIAN_FRONTEND=noninteractive

# ── System tools ──
RUN apt-get update && apt-get install -y --no-install-recommends \
        git git-lfs curl wget build-essential ca-certificates \
        locales ffmpeg xz-utils \
    && locale-gen en_US.UTF-8 \
    && rm -rf /var/lib/apt/lists/*

ENV LANG=en_US.UTF-8
ENV LC_ALL=en_US.UTF-8

# ── uv (official distroless copy, pinned version) ──
# Ref: https://docs.astral.sh/uv/guides/integration/docker/#installing-uv
COPY --from=ghcr.io/astral-sh/uv:0.11.6 /uv /uvx /bin/

# ── uv environment ──
# Ref: https://docs.astral.sh/uv/guides/integration/docker/#using-the-environment
ENV UV_LINK_MODE=copy
# Ref: https://docs.astral.sh/uv/guides/integration/docker/#compiling-bytecode
ENV UV_COMPILE_BYTECODE=1

# ── Python 3.12 ──
RUN uv python install 3.12

WORKDIR /workspace/lerobot

# ── Install dependencies (cached unless deps change) ──
# Ref: https://docs.astral.sh/uv/guides/integration/docker/#intermediate-layers
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --extra smolvla --extra feetech --extra async \
    --no-install-project --locked --no-editable

# ── Copy source + install project ──
COPY README.md src/ src/
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --extra smolvla --extra feetech --extra async \
    --locked --no-editable

# ---------------------------------------------------------------------------
# Stage 2: Runtime
# ---------------------------------------------------------------------------
FROM nvidia/cuda:12.4.0-runtime-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
        bash ca-certificates locales ffmpeg curl \
    && locale-gen en_US.UTF-8 \
    && rm -rf /var/lib/apt/lists/*

ENV LANG=en_US.UTF-8
ENV LC_ALL=en_US.UTF-8

WORKDIR /workspace/lerobot

# ── Copy uv-managed Python + built venv (includes lerobot) ──
COPY --from=builder /root/.local/share/uv /root/.local/share/uv
COPY --from=builder /workspace/lerobot/.venv .venv

# ── Entrypoint ──
COPY dev.sh dev.sh
RUN chmod +x dev.sh

# Ref: https://docs.astral.sh/uv/guides/integration/docker/#using-the-environment
ENV PATH="/workspace/lerobot/.venv/bin:$PATH"

# HF auth: pass at runtime via -e HF_TOKEN=hf_xxx
# HF cache: mount a volume to persist downloaded models
#   docker run -v hf-cache:/root/.cache/huggingface ...
ENV HF_HOME=/root/.cache/huggingface

EXPOSE 8080

CMD ["bash", "dev.sh", "serve", "--host", "0.0.0.0", "--port", "8080"]
