FROM atticux/gpu-devcontainer:latest

WORKDIR /workspace/lerobot

# Add project-specific dependencies below, e.g.:
COPY pyproject.toml uv.lock ./
RUN uv sync --extra dev --extra smolvla --extra feetech --no-install-project --frozen

# FROM atticux/lerobot-dev:latest
