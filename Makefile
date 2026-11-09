# Makefile for ECS HCM Trainer Docker Image

# Configuration
REGISTRY := registry.example.com
PROJECT := torchx_deploy
IMAGE_NAME := ecs_hcm_trainer
TAG ?= latest
FULL_IMAGE_NAME := $(REGISTRY)/$(PROJECT)/$(IMAGE_NAME):$(TAG)

# Docker build context and dockerfile paths
DOCKERFILE := dockerfiles/ecs_hcm_trainer/Dockerfile
CONTEXT := .

# Default target
.PHONY: all
all: build

# Build the Docker image
.PHONY: build
build:
	@echo "Building Docker image: $(FULL_IMAGE_NAME)"
	DOCKER_BUILDKIT=1 docker build \
		--build-arg BUILDKIT_INLINE_CACHE=1 \
		--compression=gzip \
		--cache-from $(FULL_IMAGE_NAME) \
		-t $(FULL_IMAGE_NAME) \
		-f $(DOCKERFILE) \
		$(CONTEXT)
	@echo "Build completed successfully!"
	@echo "Image: $(FULL_IMAGE_NAME)"

# Build with buildx (faster, better compression)
.PHONY: buildx
buildx:
	@echo "Building Docker image with buildx: $(FULL_IMAGE_NAME)"
	docker buildx build \
		--platform linux/amd64 \
		--cache-from type=registry,ref=$(FULL_IMAGE_NAME) \
		--cache-to type=inline \
		--compress \
		--load \
		-t $(FULL_IMAGE_NAME) \
		-f $(DOCKERFILE) \
		$(CONTEXT)
	@echo "Buildx completed successfully!"
	@echo "Image: $(FULL_IMAGE_NAME)"

# Push the image to Harbor registry
.PHONY: push
push: build
	@echo "Pushing image to Harbor registry: $(FULL_IMAGE_NAME)"
	docker push $(FULL_IMAGE_NAME)
	@echo "Push completed successfully!"

# Build with buildx and push directly (fastest for push)
.PHONY: buildx-push
buildx-push:
	@echo "Building and pushing with buildx: $(FULL_IMAGE_NAME)"
	docker buildx build \
		--platform linux/amd64 \
		--cache-from type=registry,ref=$(FULL_IMAGE_NAME) \
		--cache-to type=inline \
		--push \
		-t $(FULL_IMAGE_NAME) \
		-f $(DOCKERFILE) \
		$(CONTEXT)
	@echo "Buildx push completed successfully!"

# Build and push in one command
.PHONY: deploy
deploy: build push
	@echo "Deployment completed: $(FULL_IMAGE_NAME)"

# Fast deploy using buildx
.PHONY: deploy-fast
deploy-fast: buildx-push
	@echo "Fast deployment completed: $(FULL_IMAGE_NAME)"

# Login to Harbor registry (interactive)
.PHONY: login
login:
	@echo "Logging in to Harbor registry: $(REGISTRY)"
	docker login $(REGISTRY)

# Tag image with additional tags
.PHONY: tag
tag:
	@if [ -z "$(NEW_TAG)" ]; then \
		echo "Usage: make tag NEW_TAG=<new_tag>"; \
		exit 1; \
	fi
	@echo "Tagging image with new tag: $(NEW_TAG)"
	docker tag $(FULL_IMAGE_NAME) $(REGISTRY)/$(PROJECT)/$(IMAGE_NAME):$(NEW_TAG)
	@echo "Tagged as: $(REGISTRY)/$(PROJECT)/$(IMAGE_NAME):$(NEW_TAG)"

# Push with specific tag
.PHONY: push-tag
push-tag: tag
	@if [ -z "$(NEW_TAG)" ]; then \
		echo "Usage: make push-tag NEW_TAG=<new_tag>"; \
		exit 1; \
	fi
	@echo "Pushing tagged image: $(REGISTRY)/$(PROJECT)/$(IMAGE_NAME):$(NEW_TAG)"
	docker push $(REGISTRY)/$(PROJECT)/$(IMAGE_NAME):$(NEW_TAG)

# Clean up local images
.PHONY: clean
clean:
	@echo "Cleaning up local images..."
	-docker rmi $(FULL_IMAGE_NAME)
	@echo "Cleanup completed"

# Remove all related images (including different tags)
.PHONY: clean-all
clean-all:
	@echo "Removing all related images..."
	-docker images $(REGISTRY)/$(PROJECT)/$(IMAGE_NAME) --format "{{.Repository}}:{{.Tag}}" | xargs docker rmi
	@echo "All images removed"

# Show image information
.PHONY: info
info:
	@echo "Image Configuration:"
	@echo "  Registry: $(REGISTRY)"
	@echo "  Project:  $(PROJECT)"
	@echo "  Image:    $(IMAGE_NAME)"
	@echo "  Tag:      $(TAG)"
	@echo "  Full:     $(FULL_IMAGE_NAME)"
	@echo ""
	@echo "Local images:"
	@docker images $(REGISTRY)/$(PROJECT)/$(IMAGE_NAME) 2>/dev/null || echo "  No local images found"

# Test the built image
.PHONY: test
test: build
	@echo "Testing the image..."
	@echo "Checking PyTorch installation..."
	docker run --rm $(FULL_IMAGE_NAME) python -c "import torch; print('PyTorch version:', torch.__version__)"
	@echo "Checking CUDA availability..."
	docker run --rm --gpus all $(FULL_IMAGE_NAME) python -c "import torch; print('CUDA available:', torch.cuda.is_available())" 2>/dev/null || \
	docker run --rm $(FULL_IMAGE_NAME) python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
	@echo "Test completed successfully!"

# Interactive shell for debugging
.PHONY: shell
shell: build
	@echo "Starting interactive shell in container..."
	docker run --rm -it --gpus all $(FULL_IMAGE_NAME) /bin/bash 2>/dev/null || \
	docker run --rm -it $(FULL_IMAGE_NAME) /bin/bash

# Build with different tag
.PHONY: build-tag
build-tag:
	@if [ -z "$(NEW_TAG)" ]; then \
		echo "Usage: make build-tag NEW_TAG=<new_tag>"; \
		exit 1; \
	fi
	@echo "Building with tag: $(NEW_TAG)"
	docker build -t $(REGISTRY)/$(PROJECT)/$(IMAGE_NAME):$(NEW_TAG) -f $(DOCKERFILE) $(CONTEXT)

# Show help
.PHONY: help
help:
	@echo "Available targets:"
	@echo "  build          - Build the Docker image (BuildKit)"
	@echo "  buildx         - Build with buildx (faster, better compression)"
	@echo "  push           - Build and push the image to Harbor"
	@echo "  buildx-push    - Build and push with buildx (FASTEST)"
	@echo "  deploy         - Build and push (standard)"
	@echo "  deploy-fast    - Build and push with buildx (RECOMMENDED)"
	@echo "  login          - Login to Harbor registry"
	@echo "  tag NEW_TAG=x  - Tag image with additional tag"
	@echo "  push-tag NEW_TAG=x - Tag and push with new tag"
	@echo "  test           - Test the built image"
	@echo "  shell          - Start interactive shell in container"
	@echo "  clean          - Remove local image"
	@echo "  clean-all      - Remove all related local images"
	@echo "  info           - Show image configuration and local images"
	@echo "  help           - Show this help message"
	@echo ""
	@echo "Variables:"
	@echo "  TAG            - Image tag (default: latest)"
	@echo "  NEW_TAG        - New tag for tagging operations"
	@echo ""
	@echo "Examples:"
	@echo "  make build"
	@echo "  make push"
	@echo "  make build TAG=v1.0.0"
	@echo "  make push TAG=production"
	@echo "  make tag NEW_TAG=v1.0.0"
	@echo "  make push-tag NEW_TAG=staging"
