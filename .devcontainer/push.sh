#!/usr/bin/env bash
# Build and push the project dev image to Docker Hub
# Usage: bash .devcontainer/push.sh [tag]
set -euo pipefail

IMAGE="atticux/lerobot-dev"
TAG="${1:-latest}"

echo "Building $IMAGE:$TAG ..."
docker build -t "$IMAGE:$TAG" -f .devcontainer/Dockerfile .

echo "Pushing $IMAGE:$TAG ..."
docker push "$IMAGE:$TAG"

echo "Done: $IMAGE:$TAG"
