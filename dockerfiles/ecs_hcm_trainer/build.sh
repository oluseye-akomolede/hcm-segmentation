#!/bin/bash

# Build script for ECS HCM Trainer Docker image

set -e

# Configuration
IMAGE_NAME="ecs-hcm-trainer"
IMAGE_TAG=${1:-"latest"}
DOCKERFILE_PATH="dockerfiles/ecs_hcm_trainer/Dockerfile"
CONTEXT_PATH="."

# Git revision of the code being baked into the image. Falls back to "unknown"
# when not in a git checkout.
GIT_SHA=$(git rev-parse HEAD 2>/dev/null || echo "unknown")
GIT_SHA_SHORT=$(git rev-parse --short HEAD 2>/dev/null || echo "unknown")

# Default the tag to the short SHA (so images are traceable to a revision);
# an explicit first argument still wins.
if [[ -z "${1:-}" ]]; then
    IMAGE_TAG="${GIT_SHA_SHORT}"
fi

echo "Building Docker image: ${IMAGE_NAME}:${IMAGE_TAG}"
echo "Git SHA: ${GIT_SHA}"

# Build the image
docker build \
    -t "${IMAGE_NAME}:${IMAGE_TAG}" \
    --build-arg GIT_SHA="${GIT_SHA}" \
    -f "${DOCKERFILE_PATH}" \
    "${CONTEXT_PATH}"

echo "Build completed successfully!"
echo "Image: ${IMAGE_NAME}:${IMAGE_TAG}"

# Optional: Test the image
if [[ "${2}" == "--test" ]]; then
    echo "Testing the image..."
    docker run --rm "${IMAGE_NAME}:${IMAGE_TAG}" --help || echo "Test completed"
fi

# Show image info
echo "Image size:"
docker images "${IMAGE_NAME}:${IMAGE_TAG}" --format "table {{.Repository}}\t{{.Tag}}\t{{.Size}}\t{{.CreatedAt}}"
