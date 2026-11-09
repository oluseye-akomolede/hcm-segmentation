# ============================================
# Stage 1: Builder - Install dependencies
# ============================================
FROM nvidia/cuda:12.1.1-devel-ubuntu22.04 as builder

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Install build dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-dev \
    python3-venv \
    build-essential \
    git \
    curl \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Create symbolic link for python
RUN ln -s /usr/bin/python3 /usr/bin/python

# Upgrade pip and install build tools
RUN python -m pip install --no-cache-dir --upgrade pip setuptools wheel

# Create virtual environment to isolate dependencies
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Split the venv into three parts of at most 2.5 GB each. Copied as one 5 GB
# layer, Harbor's blob verification outlasted docker's ~47 s PUT timeout and
# pushes needed several tries. The runtime stage copies the parts back onto
# /opt/venv, so the tree is unchanged; only the layering is. The script fails
# the build if the venv outgrows 3 x 2.5 GB, rather than shipping a big layer.
COPY dockerfiles/ecs_hcm_trainer/split_venv.py /tmp/split_venv.py

# Install everything, clean up and split in ONE RUN. The split moves files with
# os.rename; a file created by an earlier RUN lives in a lower overlayfs layer,
# so renaming it forces a full copy-up first. With the installs in their own
# RUNs that copied ~5 GB and took 960 s; inside one RUN every rename is a
# metadata change. pip's download cache is a BuildKit cache mount instead of a
# layer, so a dependency change still reuses the downloaded torch wheels.
#
# PyTorch with CUDA support first, from its own index. torchaudio is
# deliberately absent: nothing here imports it and no kept package requires it,
# and its wheel carries its own CUDA libs.
#
# The second list is the transitive closure of the image's training path --
# main.py and ecs_hcm_model/ minus its eval/ -- plus what those imports require at
# runtime. It is deliberately not the closure of everything the image ships:
# jetson/ and tools/ come along for the eval entry points, whose module
# ecs_hcm_model/eval/seg.py imports jetson.input_size and jetson.model_shape, and
# other modules under jetson/ import packages this image does not install --
# jetson/realtime_demo.py imports flask.
# timm is on the training path now (ecs_hcm_model/core/backbones.py builds the
# backbone from it), pinned to the version the model code was verified against:
# its own requirements are torch/torchvision/pyyaml (pinned above), safetensors
# (already installed, transformers 4.35.0 requires >=0.3.1) and huggingface_hub
# 0.17.0, which exposes every name timm 0.9.16 imports. Importing timm or
# building a model downloads nothing; only a pretrained backbone does, at model
# construction, into the hub cache.
# Dropped as unimported and unrequired: wandb, tensorboard, seaborn,
# torchmetrics, jsonlines, boto3 (mlflow only wants boto3 under its optional
# "extras"/"databricks"/"gateway"/"genai" extras), and torchaudio (imported
# nowhere). Kept despite not being imported directly, because a kept package
# hard-requires them: matplotlib and scikit-learn (mlflow), pandas (mlflow and
# datasets). pycocotools appeared twice and is listed once.
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install \
    torch==2.1.0 \
    torchvision==0.16.0 \
    --index-url https://download.pytorch.org/whl/cu121 \
    && \
    pip install \
    # ML/DL frameworks
    pytorch-lightning==2.1.0 \
    transformers==4.35.0 \
    timm==0.9.16 \
    datasets==2.14.0 \
    huggingface-hub==0.17.0 \
    pycocotools==2.0.7 \
    # Data processing
    numpy==1.24.3 \
    pandas==2.0.3 \
    pillow==10.0.0 \
    opencv-python-headless==4.8.0.76 \
    scikit-learn==1.3.0 \
    matplotlib==3.7.1 \
    tqdm==4.66.0 \
    colorama==0.4.6 \
    # MLflow and logging
    mlflow==2.8.0 \
    # Cloud storage
    minio==7.1.17 \
    s3fs==2023.9.0 \
    fsspec==2023.9.0 \
    pyarrow==13.0.0 \
    # Utilities
    pyyaml==6.0 \
    requests==2.31.0 \
    && \
    # Clean up immediately after install to reduce layer size
    # Each tolerant find is grouped: an ungrouped 'a && b || true && c' runs c
    # even when a failed, so a failed pip install would still go green.
    { find /opt/venv -type d -name "tests" -exec rm -rf {} + 2>/dev/null || true; } && \
    { find /opt/venv -type d -name "test" -exec rm -rf {} + 2>/dev/null || true; } && \
    { find /opt/venv -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true; } && \
    find /opt/venv -name "*.pyc" -delete && \
    find /opt/venv -name "*.pyo" -delete && \
    { find /opt/venv -type d -name "*.dist-info" -exec sh -c 'rm -rf {}/RECORD {}/INSTALLER' \; 2>/dev/null || true; } && \
    rm -rf /opt/venv/lib/python*/site-packages/pip /opt/venv/lib/python*/site-packages/setuptools && \
    python /tmp/split_venv.py /opt/venv /opt/venv-parts --parts 3 --cap-bytes 2500000000

# ============================================
# Stage 2: Runtime - Minimal production image
# ============================================
# The cu121 torch wheel bundles the CUDA math libraries the trainer actually
# uses inside torch/lib (cuBLASLt, cuDNN, nvrtc, and its own cudart), so the
# runtime base's copies are duplication: that base carried 2.01 GB of CUDA
# libraries plus a 1.8 GB /usr/local/cuda-12.1 tree.
#
# base and runtime ship an IDENTICAL /usr/local/cuda-12.1/compat (the
# driver-facing libcuda stub, libnvidia-nvvm, libnvidia-ptxjitcompiler); the
# only thing runtime adds is the CUDA math libraries. Neither ships nvidia-smi
# or a real driver -- those come from the host via the container runtime.
#
# torch/lib does NOT win by itself: libtorch_cuda.so's DT_NEEDED names
# (libcublas.so.12, libcublasLt.so.12) are resolved by the loader, and with the
# default search path it picked them up from /usr/local/cuda/lib64. So the
# runtime stage below puts torch/lib FIRST on LD_LIBRARY_PATH; with that, both
# resolve from the wheel (verified: ldd shows torch/lib for both, and the
# bundled libcublas loads). Without it, this image would fail to import torch
# CUDA. Do not drop that LD_LIBRARY_PATH.
FROM nvidia/cuda:12.1.1-base-ubuntu22.04 as runtime

# Build-time git SHA, baked into the image so training runs can record the
# exact code revision. build.sh passes this via --build-arg.
ARG GIT_SHA=unknown
ENV GIT_SHA=${GIT_SHA}

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=/opt/venv/bin:${CUDA_HOME}/bin:${PATH}
# torch/lib FIRST: the wheel's bundled cuBLAS/cuDNN/nvrtc are what libtorch_cuda
# loads against, and the base image has no /usr/local/cuda/lib64 to fall back to.
# See the note above the FROM line -- this is load-bearing, not cosmetic.
# The interpreter version is globbed rather than hardcoded: the builder stage
# pins it, and a hardcoded path that stopped matching would drop the wheel off
# the search path silently, which is the exact failure this line prevents.
ENV LD_LIBRARY_PATH=/opt/venv/lib/torch-cuda-libs:${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}
ENV PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
ENV GIT_PYTHON_REFRESH=quiet

# Install only runtime dependencies (no build tools)
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    python3 \
    python3-distutils \
    libjpeg62 \
    libpng16-16 \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Create symbolic link for python
RUN ln -s /usr/bin/python3 /usr/bin/python

# Copy virtual environment from builder (these layers rarely change), as the
# three parts split_venv.py made in the builder, so no layer exceeds ~2.5 GB.
# Using --link for better performance with BuildKit
COPY --from=builder --link /opt/venv-parts/part0/ /opt/venv/
COPY --from=builder --link /opt/venv-parts/part1/ /opt/venv/
COPY --from=builder --link /opt/venv-parts/part2/ /opt/venv/

# Stable path to torch's bundled CUDA libs, for LD_LIBRARY_PATH above. Resolved
# by glob at build time so it cannot go stale if the builder's interpreter
# version changes. Fails the build rather than shipping an image whose library
# order is quietly wrong.
RUN set -eu; \
    src=$(ls -d /opt/venv/lib/python*/site-packages/torch/lib); \
    test -n "$src"; \
    ln -sfn "$src" /opt/venv/lib/torch-cuda-libs; \
    test -e /opt/venv/lib/torch-cuda-libs/libcublas.so.12

# Create a non-root user for security (do this before copying code)
RUN useradd --create-home --shell /bin/bash trainer

# Set working directory
WORKDIR /app

# Set Python path to include the application modules
ENV PYTHONPATH=/app:${PYTHONPATH}

# Copy only necessary Python code (most frequently changing layer at the end)
COPY --chown=trainer:trainer ecs_hcm_model/ /app/ecs_hcm_model/
# The eval-only DAG's entry points live in tools/ (eval_seg.py, fetch_seg_val.py),
# and the module they call, ecs_hcm_model/eval/seg.py, imports jetson.input_size
# and jetson.model_shape. tools/ therefore ships as the entry points' own
# directory and jetson/ for what they import; neither is on the training path.
COPY --chown=trainer:trainer jetson/ /app/jetson/
COPY --chown=trainer:trainer tools/ /app/tools/
COPY --chown=trainer:trainer *.py /app/
COPY --chown=trainer:trainer requirements.txt /app/

# Switch to non-root user
USER trainer

