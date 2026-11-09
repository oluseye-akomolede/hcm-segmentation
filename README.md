# ecs_hcm_model

A street-scene segmentation model that runs in real time on an NVIDIA Jetson Orin Nano, with the training pipeline, experiment tracking and deployment that put it there.

I designed this and directed an AI agent team that wrote it. The model architecture, the evaluation rules and the decision to refuse a bad export are mine. The implementation was written by the team against specs and reviews I ran.

## What it does

The model is a Hierarchical Clustering Model (HCM) for segmentation on a ConvNeXt V2 backbone. Training runs at 352-pixel input in two stages: COCO first, then a fine-tune on a mix of BDD100K street scenes and COCO (`--coco`, `--bdd`).

Three things run around it:

- **Training pipeline.** Airflow DAGs on Kubernetes launch one- or two-GPU PyTorch training with torchrun and DDP, log to MLflow, and write checkpoints to MinIO. `main.py` is the entry point; the warm-start path (`--init-from`) copies every compatible tensor from an earlier checkpoint into a new architecture and refuses to start if the source shares nothing with it.
- **Export and verification.** `jetson/export_trt.py` builds a TensorRT engine and verifies it against the PyTorch model before the engine is promoted. `jetson/promote_model.py` is the only path an engine takes into use.
- **Field capture.** A camera path on the Jetson captures frames, runs inference, and uploads only the frames the model is uncertain about (`jetson/uncertain_frames.py`), which turns a full video stream into a small set worth labelling.

## Validation

Validation IoU is the metric, and the rule that matters is in `docs/RUNBOOK.md`: an epoch checkpoint is `is_best` only when its validation IoU beats every earlier epoch in the same training process, and the single final `_best` is uploaded only after training ends. The runbook spells out why a resume starts that count again, and what to check in `metadata.json` before promoting a checkpoint. It exists because getting this wrong promotes the wrong weights and looks like success.

Engine export is gated too. `jetson/export_trt.py --verify` compares the TensorRT output against PyTorch and the export step refuses to produce a silently wrong engine, including refusing FP16 exports that produce NaN.

The real-time demo path measures agreement against a reference grid rather than asserting it: `jetson/verify_capture_windows.py` and `tests/test_camera_frames.py` pin the frame windows, the de-duplication rules and the uncertainty thresholds.

An evaluation harness (`ecs_hcm_model/eval/seg.py`) scores checkpoints on held-out street scenes, including a comparison against a public baseline, and can run over camera frames rather than a static set. The camera-frame measurement path pre-registers an IoU gap and refuses runs whose inputs do not match the pins.

## How to run it

Training, from a checkout:

```sh
python main.py train_hcm_imagenet1k --coco --input-size 352 --batch_size 2 --max_epochs <N>
```

Warm start from an earlier checkpoint:

```sh
python main.py train_hcm_imagenet1k --coco --input-size 352 --batch_size 2 \
    --max_epochs <N> --init-from <checkpoint-spec>
```

Read the startup line when it prints the warm-start tensor counts. Any skipped or newly initialised tensor means the source is not the checkpoint you think it is. Stop.

Export and verify on the Jetson:

```sh
PYTHONPATH=<checkout> python -m jetson.export_trt <checkpoint> --out <dir> --verify
```

The container image builds from `dockerfiles/ecs_hcm_trainer/` with `make build`. The runbook in `docs/RUNBOOK.md` is the operative document for training, resume and promotion rules.

## Status

The training and export paths work. The accuracy numbers this project quotes are not reproduced in this copy of the repository: the held-out IoU results live in MLflow runs and experiment notes that are not included here, so do not treat this repository as a source for them. What the repository does hold is the mechanism by which an accuracy number would be produced and checked.

The Jetson demo needs a Jetson with JetPack and a working CUDA toolchain; nothing here runs that path on a laptop. Some tests are skipped without a GPU, MinIO or a cluster.

## What is not here

No employer code, data or configuration. The BDD100K and COCO datasets are fetched by the pipeline, not shipped. Cluster paths, credentials and the container registry are placeholders.
