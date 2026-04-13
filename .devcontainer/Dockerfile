FROM atticux/gpu-devcontainer:latest

WORKDIR /workspace/lerobot

RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --extra dev --extra smolvla --extra feetech \
    --no-install-project --locked

# FROM atticux/lerobot-dev:latest
