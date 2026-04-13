FROM atticux/gpu-devcontainer:latest

WORKDIR /workspace/lerobot

COPY pyproject.toml uv.lock ./
RUN uv sync --extra dev --extra smolvla --extra feetech --no-install-project --frozen
