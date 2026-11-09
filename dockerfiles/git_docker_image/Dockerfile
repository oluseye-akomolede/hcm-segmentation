FROM ghcr.io/pytorch/torchx:0.8.0dev0

# Install git
RUN apt-get update && apt-get install -y git && \
    apt-get update && apt-get install -y python3 && \
    apt-get update && apt-get install ffmpeg libsm6 libxext6 -y && \
    apt-get update && apt-get install build-essential -y
    

RUN pip3 install --no-cache-dir --root-user-action=ignore \
torch \
torchvision \
torchaudio \
transformers \
ultralytics \
datasets \ 
opencv-python \
pynvml \
colorama

ENV CC=/usr/bin/gcc 

# Set working directory
WORKDIR /app

# Copy your project files
COPY . /app