# ECS HCM Trainer Docker Image

This Docker image is designed for training Hierarchical Clustering Model (HCM) on ImageNet1K dataset, specifically built to be orchestrated by Apache Airflow with command-line arguments.

## Features

- **CUDA Support**: Built on NVIDIA CUDA 12.1 base image for GPU acceleration
- **PyTorch 2.1.0**: Latest stable PyTorch with CUDA support
- **MLflow Integration**: Built-in experiment tracking and model management
- **Cloud Storage**: MinIO and S3 support for model artifacts and datasets
- **Command Line Interface**: Flexible parameter passing via command line arguments
- **Security**: Runs as non-root user
- **Health Checks**: Built-in container health monitoring

## Dependencies

### Core ML/DL Libraries
- PyTorch 2.1.0 (with CUDA 12.1 support)
- PyTorch Lightning 2.1.0
- Transformers 4.35.0
- Datasets 2.14.0
- Hugging Face Hub 0.17.0

### Data Processing
- NumPy, Pandas, Pillow
- OpenCV (headless)
- PyArrow for efficient data serialization

### Experiment Tracking
- MLflow 2.7.0
- TensorBoard 2.14.0
- Weights & Biases (WandB)

### Cloud Storage
- MinIO client
- S3FS for S3-compatible storage
- Boto3 for AWS services

## Usage

### Building the Image

```bash
# Build with default tag
docker build -t ecs-hcm-trainer:latest -f dockerfiles/ecs_hcm_trainer/Dockerfile .

# Or use the build script
chmod +x dockerfiles/ecs_hcm_trainer/build.sh
./dockerfiles/ecs_hcm_trainer/build.sh latest
```

### Running the Container

#### Basic Usage
```bash
docker run --gpus all ecs-hcm-trainer:latest
```

#### With Custom Parameters
```bash
docker run --gpus all ecs-hcm-trainer:latest \
    --new_model true \
    --environment production \
    --finetune false \
    --max_epochs 10 \
    --batch_size 16
```

#### From Airflow DAG
```python
from airflow.providers.docker.operators.docker import DockerOperator

train_task = DockerOperator(
    task_id='train_hcm_model',
    image='ecs-hcm-trainer:latest',
    command=[
        '--new_model', 'true',
        '--environment', 'production', 
        '--finetune', 'false',
        '--max_epochs', '20',
        '--batch_size', '32'
    ],
    device_requests=[
        docker.types.DeviceRequest(device_ids=["0"], capabilities=[["gpu"]])
    ],
    auto_remove=True,
    dag=dag
)
```

## Command Line Arguments

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--new_model` | boolean | true | Whether to create a new model instance |
| `--environment` | string | development | Environment for MLflow tracking URI |
| `--finetune` | boolean | false | Whether to enable fine-tuning |
| `--max_epochs` | integer | 5 | Maximum number of training epochs |
| `--batch_size` | integer | 8 | Batch size for training |

## Environment Variables

The container sets several important environment variables:

- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`: Optimizes CUDA memory allocation
- `GIT_PYTHON_REFRESH=quiet`: Suppresses Git Python warnings
- `PYTHONPATH=/app`: Ensures proper module imports
- `CUDA_VISIBLE_DEVICES`: Controls GPU visibility (can be overridden)

## GPU Requirements

- NVIDIA GPU with CUDA 12.1 support
- Minimum 8GB GPU memory (recommended 16GB+ for larger batch sizes)
- Docker with NVIDIA Container Toolkit installed

## Storage Requirements

- Container size: ~8-10GB
- Temporary storage for datasets and model checkpoints
- External storage (MinIO/S3) for persistent model artifacts

## Health Check

The container includes a health check that verifies:
- PyTorch installation
- CUDA availability
- Basic import functionality

## Security

- Runs as non-root user 'trainer'
- Minimal attack surface with only necessary dependencies
- No SSH or unnecessary services running

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**: Reduce batch size or use gradient checkpointing
2. **Import Errors**: Ensure PYTHONPATH includes `/app`
3. **GPU Not Available**: Check NVIDIA drivers and Docker GPU support
4. **Storage Issues**: Ensure sufficient disk space for model checkpoints

### Debugging

```bash
# Interactive shell for debugging
docker run --gpus all -it --entrypoint /bin/bash ecs-hcm-trainer:latest

# Check GPU availability
docker run --gpus all ecs-hcm-trainer:latest python -c "import torch; print(torch.cuda.is_available())"
```

## Development

To modify the training code:

1. Update the source code in the host directory
2. Rebuild the image
3. Test with new parameters

The image copies the entire application context, so any changes require a rebuild.
