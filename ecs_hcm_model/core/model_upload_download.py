import os
import torch
import traceback
import json
import tempfile
from datetime import datetime
from pathlib import Path
from minio import Minio
from minio.error import S3Error
import pickle
from ecs_hcm_model.core.hcm_nn import (
    ArchMismatchError,
    arch_record_of,
    check_amodal_outline,
    check_backbone,
    check_decoder_arch,
    check_mask_stride,
    check_neck_channels,
    neck_argument,
)

def infer_segmentation_classes_from_state_dict(state_dict) -> int | None:
    w = state_dict.get("segmentation_head.weight", None)
    b = state_dict.get("segmentation_head.bias", None)
    if isinstance(w, torch.Tensor) and w.ndim == 4:
        return int(w.shape[0])  # out_channels = num classes
    if isinstance(b, torch.Tensor) and b.ndim == 1:
        return int(b.shape[0])
    return None


# The single source of truth for where each task's checkpoints live and what
# they are named. Every save path (per-epoch, best, last, interrupted, smoke)
# must go through this so a checkpoint's bucket and name always agree with its
# task -- previously everything was written to "models" under one name.
TASK_CHECKPOINTS = {
    "classification": {"bucket": "models", "model_name": "hcm_imagenet1k"},
    "coco_seg": {"bucket": "coco-models", "model_name": "hcm_coco_seg"},
    "cocoa_seg": {"bucket": "finetuned-models", "model_name": "hcm_cocoa_seg"},
}


def resolve_task(finetune: bool = False, coco: bool = False) -> str:
    """Map the trainer's flags to a checkpoint task key."""
    if finetune:
        return "cocoa_seg"
    if coco:
        return "coco_seg"
    return "classification"


def checkpoint_target(task: str) -> tuple[str, str]:
    """Return (bucket, model_name) for a task key."""
    if task not in TASK_CHECKPOINTS:
        raise ValueError(
            f"unknown task {task!r}; expected one of {sorted(TASK_CHECKPOINTS)}"
        )
    spec = TASK_CHECKPOINTS[task]
    return spec["bucket"], spec["model_name"]


# The task a checkpoint may be resumed into, as (checkpoint task, run task).
# The same task continues a lineage; the two cross-task steps are the
# pipeline's stages: ImageNet -> COCO (the segmentation head is added) and COCO
# -> COCOA (the refine). Anything else is a misconfiguration.
ALLOWED_RESUME_TRANSITIONS = frozenset({
    ("classification", "classification"),
    ("coco_seg", "coco_seg"),
    ("cocoa_seg", "cocoa_seg"),
    ("classification", "coco_seg"),
    ("coco_seg", "cocoa_seg"),
})

# The trainer flags that select each task, for errors that must name them.
_TASK_FLAGS = {
    "classification": "neither --coco nor --finetune",
    "coco_seg": "--coco",
    "cocoa_seg": "--finetune",
}


def task_for_bucket(bucket: str) -> str | None:
    """The task whose checkpoints live in ``bucket``, or None."""
    for task, spec in TASK_CHECKPOINTS.items():
        if spec["bucket"] == bucket:
            return task
    return None


def resume_bucket_for(task: str, model_base: str) -> str:
    """The bucket a ``task`` run resumes from, given ``--model-base``.

    ``--model-base`` names the bucket; it must hold one task's checkpoints and
    that task must be allowed to resume into ``task``
    (``ALLOWED_RESUME_TRANSITIONS``). Anything else raises, naming the flags,
    rather than falling back to "models".
    """
    source = task_for_bucket(model_base)
    if source is None:
        raise ValueError(
            f"--model-base {model_base!r} is not a checkpoint bucket; expected "
            f"one of {sorted(spec['bucket'] for spec in TASK_CHECKPOINTS.values())}"
        )
    if (source, task) not in ALLOWED_RESUME_TRANSITIONS:
        allowed = sorted(
            TASK_CHECKPOINTS[src]["bucket"]
            for src, dst in ALLOWED_RESUME_TRANSITIONS if dst == task
        )
        raise ValueError(
            f"a {task} run ({_TASK_FLAGS[task]}) cannot resume from "
            f"--model-base {model_base!r}, which holds {source} checkpoints; "
            f"use --model-base {' or '.join(allowed)}"
        )
    return model_base


def check_resume_metadata(custom: dict, task: str | None, input_size: int | None) -> None:
    """Check a resume's checkpoint metadata against the run.

    A task the checkpoint may not resume into raises (see
    ``ALLOWED_RESUME_TRANSITIONS``). A different input size is legitimate --
    fine-tuning at a new size, as 192 -> 352 did -- so it only warns, loudly.
    A legacy checkpoint without the field is not checked.
    """
    ckpt_task = custom.get("task")
    if task is not None and ckpt_task is not None:
        if (ckpt_task, task) not in ALLOWED_RESUME_TRANSITIONS:
            raise ValueError(
                f"the checkpoint was trained for {ckpt_task}, which a {task} run "
                f"({_TASK_FLAGS.get(task, task)}) may not resume from"
            )
    ckpt_size = custom.get("input_size")
    if input_size is not None and ckpt_size is not None and int(ckpt_size) != int(input_size):
        print("!" * 80)
        print(
            f"WARNING: resuming a checkpoint trained at input_size={ckpt_size} "
            f"into a run at input_size={input_size}. Fine if this is a deliberate "
            f"change of size; otherwise check --input-size."
        )
        print("!" * 80)


def is_resume_candidate(metadata: dict) -> bool:
    """Whether a stored checkpoint is a legitimate thing to resume training from.

    Every save path writes architecture "hcm" into the same bucket -- per-epoch,
    ``_best``, ``_last`` and smoke runs alike -- so "latest wins" would happily
    resume a real run from a 32-batch smoke checkpoint. Only ``is_best``
    checkpoints qualify, and a smoke checkpoint never does.

    Legacy checkpoints predate the is_best/is_smoke fields entirely; they are
    the only artifacts some buckets contain, so an absent ``is_best`` counts as
    eligible rather than as a rejection.
    """
    custom = metadata.get("custom_metadata") or {}
    if custom.get("is_smoke") is True:
        return False
    # Absent is_best => legacy checkpoint, keep it loadable.
    return bool(custom.get("is_best", True))


def select_resume_checkpoint(candidates: list[dict]) -> dict | None:
    """Pick the checkpoint to resume from, or None if there is no candidate.

    ``candidates`` are the parsed metadata.json documents (the full object, with
    ``upload_timestamp`` alongside ``custom_metadata``). Returns the chosen
    document, so the caller keeps whatever prefix/timestamp it needs.
    """
    eligible = [m for m in candidates if is_resume_candidate(m)]
    if not eligible:
        return None

    def _timestamp(m):
        raw = m.get("upload_timestamp")
        if not raw:
            # No timestamp sorts oldest, so a well-formed checkpoint always wins.
            return datetime.min
        try:
            return datetime.fromisoformat(raw)
        except (TypeError, ValueError):
            # A metadata.json we did read but whose timestamp is garbage is
            # corrupt, not old: sorting it oldest would quietly resume from
            # something else.
            raise ValueError(
                f"checkpoint {m.get('name')!r} has an unparseable "
                f"upload_timestamp {raw!r}"
            ) from None

    chosen = max(eligible, key=_timestamp)
    warn_if_better_passed_over(chosen, eligible)
    return chosen


def _recorded_val_iou(m: dict):
    """A candidate's recorded val_iou, or None if it didn't record a number."""
    value = ((m.get("custom_metadata") or {}).get("val_metrics") or {}).get("val_iou")
    return value if isinstance(value, (int, float)) else None


def warn_if_better_passed_over(chosen: dict, eligible: list[dict]) -> None:
    """Warn when an eligible candidate recorded a better val_iou than the chosen one.

    Newest still wins: a resume continues the lineage. The warning only makes
    the choice visible, naming both checkpoints, their epochs and metrics.
    """
    chosen_iou = _recorded_val_iou(chosen)
    if chosen_iou is None:
        return
    better = [m for m in eligible
              if m is not chosen and (_recorded_val_iou(m) or float("-inf")) > chosen_iou]
    if not better:
        return
    best = max(better, key=_recorded_val_iou)
    def _describe(m):
        custom = m.get("custom_metadata") or {}
        return (f"{m.get('name')!r} (epoch {custom.get('epoch', 'unknown')}, "
                f"val_iou {_recorded_val_iou(m)})")
    print("!" * 80)
    print(
        f"WARNING: resuming from the newest checkpoint {_describe(chosen)}, though "
        f"{_describe(best)} in the same bucket recorded a better val_iou. Newest "
        f"wins (a resume continues the lineage); check this is the run you meant."
    )
    print("!" * 80)


def prefix_timestamp(prefix: str) -> datetime | None:
    """The upload time encoded in a checkpoint prefix (``<name>/<%Y%m%d_%H%M%S>``).

    Readable or not, every checkpoint's prefix carries this, so it orders an
    unreadable metadata.json against the readable ones. None if the prefix
    isn't in that form.
    """
    try:
        return datetime.strptime(prefix.rstrip("/").rsplit("/", 1)[-1], "%Y%m%d_%H%M%S")
    except ValueError:
        return None


# Every save path builds hcm_nn with out_stride=16, and it cannot be inferred
# from the weights (dilation has no parameters).
DEFAULT_OUT_STRIDE = 16


def build_model_from_info(model_info: dict, state_dict: dict):
    """Rebuild a model from ``model_info.pkl`` and load ``state_dict`` into it.

    Checkpoints uploaded before hcm_nn recorded its own config carry
    ``out_stride: None`` and ``num_segmentation_classes: None``. The key is
    present, so ``.get(key, default)`` returned None, and hcm_nn(out_stride=None)
    failed the backbone's bare assert. A None value is treated as absent here.
    """
    model_class = model_info['model_class']
    model_cfg = model_info.get("model_config") or {}

    if model_class.__name__ == "hcm_nn":
        num_seg = model_cfg.get("num_segmentation_classes")
        if num_seg is None:
            num_seg = infer_segmentation_classes_from_state_dict(state_dict)
        out_stride = model_cfg.get("out_stride")
        recorded = out_stride is not None
        # The decoder version: recorded since it was introduced; older
        # checkpoints are read from their keys (only v2 has a fuse conv),
        # which makes them "v1", the architecture they were trained in. A
        # recorded version the keys contradict raises.
        decoder_arch = check_decoder_arch(model_cfg.get("decoder_arch"), state_dict)
        # The v2 architecture: recorded since it was introduced, so a checkpoint
        # from before it means resnet50 with the pre-change 2048 widening and a
        # stride-4 mask -- which is what the weights' keys say as well. Every
        # recorded value is checked against the weights rather than trusted, and
        backbone = check_backbone(model_cfg.get("backbone"), state_dict)
        neck = check_neck_channels(model_cfg.get("neck_channels"), state_dict)
        mask_stride = check_mask_stride(model_cfg.get("mask_stride"), state_dict)
        amodal_outline = check_amodal_outline(model_cfg.get("amodal_outline"), state_dict)
        if not recorded:
            out_stride = DEFAULT_OUT_STRIDE
        model = model_class(enable_finetuning=bool(model_cfg.get("enable_finetuning")),
                            num_segmentation_classes=num_seg,
                            out_stride=out_stride,
                            decoder_arch=decoder_arch,
                            backbone=backbone,
                            neck_channels=neck_argument(neck),
                            mask_stride=mask_stride,
                            amodal_outline=amodal_outline,
                            # The weights are about to be loaded over whatever
                            # the backbone starts as, so never fetch pretrained
                            # ones here: a rebuild is not a training start and
                            # must not need the network.
                            backbone_pretrained=False)
        # Whether the stride above came from the checkpoint or is the default
        # stand-in: load_hcm_model gives a stand-in to the caller's stride.
        model.out_stride_recorded = recorded
    else:
        # fallback for other models
        model = model_class()

    # Non-strict so an older checkpoint still loads into a newer architecture
    # (the one caller loads non-strict too), but a mismatch is never silent.
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        print(f"Warning: checkpoint/model key mismatch: "
              f"{len(missing)} missing {missing[:10]}, "
              f"{len(unexpected)} unexpected {unexpected[:10]}")
    # Kept on the model so load_hcm_model can tell a head the checkpoint
    # carried from one that was freshly initialised here.
    model.checkpoint_missing_keys = list(missing)
    return model


class ModelUploaderAndDownloader:
    def __init__(self, bucket_name="models"):
        """
        Initialize ModelUploaderAndDownloader for MinIO instance.

        :param bucket_name: Name of the MinIO bucket to store models
        """
        self.minio_endpoint = "minio.minio.svc.cluster.local:9000"
        self.bucket_name = bucket_name
        
        # Get credentials from environment variables
        self.access_key = os.getenv('MINIO_USER')
        self.secret_key = os.getenv('MINIO_PASSWORD')
        
        if not self.access_key or not self.secret_key:
            raise ValueError("MINIO_USER and MINIO_PASSWORD environment variables must be set")
        
        # Initialize MinIO client
        self.client = Minio(
            self.minio_endpoint,
            access_key=self.access_key,
            secret_key=self.secret_key,
            secure=False  # Use HTTP (since endpoint is http://)
        )
        
        # Ensure bucket exists
        self._ensure_bucket_exists()
    
    def _ensure_bucket_exists(self):
        """Create bucket if it doesn't exist."""
        try:
            if not self.client.bucket_exists(self.bucket_name):
                self.client.make_bucket(self.bucket_name)
                print(f"Created bucket: {self.bucket_name}")
        except S3Error as e:
            print(f"Error creating bucket: {e}")
            raise

    def upload_model(self, model, model_name, architecture=None, metadata=None):
        """
        Upload a PyTorch model to MinIO.
        
        :param model: PyTorch model to upload
        :param model_name: Name for the model
        :param architecture: Architecture tag (e.g., 'hcm')
        :param metadata: Additional metadata dictionary
        :return: Upload success status
        """
        try:
            # Create temporary directory for model files
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_path = Path(temp_dir)
                
                # Save model
                model_file = temp_path / "model.pth"
                torch.save(model.state_dict(), model_file)
                
                # Create metadata
                model_metadata = {
                    "name": model_name,
                    "architecture": architecture,
                    "upload_timestamp": datetime.now().isoformat(),
                    "pytorch_version": torch.__version__,
                    "model_class": model.__class__.__name__,
                    "git_commit": os.environ.get("GIT_SHA", "unknown"),
                    "custom_metadata": {**(metadata or {}), **arch_record_of(model)}
                }
                
                # Save metadata
                metadata_file = temp_path / "metadata.json"
                with open(metadata_file, 'w') as f:
                    json.dump(model_metadata, f, indent=2)
                
                # Save model architecture/class info if possible
                try:
                    model_info_file = temp_path / "model_info.pkl"
                    with open(model_info_file, 'wb') as f:
                        pickle.dump({
                            'model_class': model.__class__,
                            'model_state': model.state_dict(),
                            'model_config': {
                                "num_segmentation_classes": getattr(model, "num_segmentation_classes", None),
                                "out_stride": getattr(model, "out_stride", None),
                                "decoder_arch": getattr(model, "decoder_arch", None),
                                "enable_finetuning": getattr(model, "enable_finetuning", False),
                                # The v2 architecture. A loader that does not
                                # find these rebuilds the pre-change model,
                                # which is what a checkpoint from before them is.
                                **arch_record_of(model)
                            }
                        }, f)
                except Exception as e:
                    print(f"Warning: Could not save model class info: {e}")
                
                # Generate unique model path with timestamp
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                model_prefix = f"{model_name}/{timestamp}"
                
                # Upload files to MinIO. metadata.json goes LAST: the resume
                # search only lists metadata.json objects, so until it lands a
                # half-written checkpoint (weights without class info, or no
                # weights at all) is never a resume candidate.
                files_to_upload = [("model.pth", model_file)]
                if (temp_path / "model_info.pkl").exists():
                    files_to_upload.append(("model_info.pkl", temp_path / "model_info.pkl"))
                files_to_upload.append(("metadata.json", metadata_file))
                
                last_object_name = None
                for file_name, file_path in files_to_upload:
                    object_name = f"{model_prefix}/{file_name}"
                    last_object_name = object_name
                    self.client.fput_object(
                        self.bucket_name,
                        object_name,
                        str(file_path)
                    )
                    print(f"Uploaded {file_name} to {object_name}")
                
                print(f"Model '{model_name}' uploaded successfully to MinIO")
                return last_object_name
                
        except Exception as e:
            # Do NOT swallow: callers register a FINISHED run off this return
            # value, so a silent None here used to mark a broken upload as a
            # successful run. Let the failure propagate so the run ends FAILED.
            print(f"Error uploading model: {e}")
            print(traceback.format_exc())
            raise

    def search_and_download_model_by_architecture(self, architecture):
        """
        Search for the latest resumable model by architecture tag and download it.

        "Latest" means latest among *resume candidates* (see
        ``is_resume_candidate``), not simply latest uploaded: since every save
        path writes architecture "hcm" into the same bucket, taking the newest
        object would resume a real run from a smoke run's checkpoint.

        :param architecture: The architecture tag (e.g., 'hcm') to search for
        :return: Downloaded model or None. On None, ``last_search_error`` says
            why (empty bucket, nothing eligible, unreadable metadata, or a
            failed download).
        """
        self.last_search_error = None
        # The chosen checkpoint's metadata and prefix, for the caller's checks.
        self.resume_metadata = None
        self.resume_prefix = None
        try:
            # List all metadata files
            objects = self.client.list_objects(
                self.bucket_name,
                recursive=True
            )

            # Collect every matching checkpoint first, then choose. Reading all
            # of them up front is what lets the choice be explained below.
            matches = []  # (prefix, metadata, rejected_reason)
            # metadata.json objects that exist but could not be read or parsed.
            # A missing metadata.json is an upload in flight and never listed;
            # an unreadable one is a real checkpoint whose contents we don't know.
            unreadable = []  # (prefix, error)

            for obj in objects:
                if obj.object_name.endswith('metadata.json'):
                    try:
                        # Download and check metadata
                        with tempfile.NamedTemporaryFile() as temp_file:
                            self.client.fget_object(
                                self.bucket_name,
                                obj.object_name,
                                temp_file.name
                            )

                            with open(temp_file.name, 'r') as f:
                                metadata = json.load(f)

                            if metadata.get('architecture') == architecture:
                                # Extract prefix from object path
                                prefix_parts = obj.object_name.split('/')[:-1]
                                prefix = '/'.join(prefix_parts)
                                custom = metadata.get('custom_metadata') or {}
                                if is_resume_candidate(metadata):
                                    reason = None
                                elif custom.get('is_smoke') is True:
                                    reason = "smoke checkpoint"
                                else:
                                    reason = "not a best checkpoint"
                                matches.append((prefix, metadata, reason))

                    except Exception as e:
                        print(f"Error reading metadata from {obj.object_name}: {e}")
                        unreadable.append(
                            ('/'.join(obj.object_name.split('/')[:-1]), f"{type(e).__name__}: {e}")
                        )
                        continue

            chosen = select_resume_checkpoint([m for _, m, _ in matches])

            if chosen is None:
                # Distinguish "nothing there" from "only unsuited candidates":
                # the second case is a real misconfiguration worth naming.
                if matches:
                    rejected = ", ".join(
                        f"{p} ({r})" for p, _, r in matches if r
                    )
                    raise Exception(
                        f"No resumable model found with architecture "
                        f"'{architecture}'; all {len(matches)} candidate(s) were "
                        f"unsuitable: {rejected}"
                    )
                if unreadable:
                    raise Exception(
                        f"No readable model found with architecture "
                        f"'{architecture}'; {len(unreadable)} checkpoint(s) have "
                        f"unreadable metadata.json: "
                        + ", ".join(f"{p} ({err})" for p, err in unreadable)
                    )
                raise Exception(
                    f"the bucket has no checkpoint with architecture '{architecture}'"
                )

            latest_prefix = next(
                p for p, m, _ in matches if m is chosen
            )

            # An unreadable checkpoint newer than the one chosen might have been
            # the one to resume from; taking an older one would be silent.
            chosen_at = prefix_timestamp(latest_prefix)
            if chosen_at is not None:
                newer = [
                    (p, err) for p, err in unreadable
                    if (prefix_timestamp(p) or datetime.min) > chosen_at
                ]
                if newer:
                    raise Exception(
                        f"Refusing to resume from {latest_prefix}: "
                        f"{len(newer)} newer checkpoint(s) have unreadable "
                        f"metadata.json: "
                        + ", ".join(f"{p} ({err})" for p, err in newer)
                    )
            custom = chosen.get('custom_metadata') or {}
            print(
                f"Resuming from {latest_prefix} "
                f"(is_best={custom.get('is_best', 'legacy')}, "
                f"is_smoke={custom.get('is_smoke', False)}, "
                f"upload_timestamp={chosen.get('upload_timestamp')}, "
                f"task={custom.get('task', 'unknown')}, "
                f"input_size={custom.get('input_size', 'unknown')}, "
                f"coco_mix_ratio={custom.get('coco_mix_ratio', 'unknown')}); "
                f"considered {len(matches)} architecture-matching checkpoint(s), "
                f"{len(unreadable)} with unreadable metadata"
            )
            self.resume_metadata = chosen
            self.resume_prefix = latest_prefix
            result = self._download_model_from_prefix(latest_prefix)
            if result is None:
                self.last_search_error = (
                    f"downloading {latest_prefix} failed (see the error above)"
                )
            return result

        except ArchMismatchError:
            raise
        except Exception as e:
            print(f"Error searching and downloading model by architecture: {e}")
            print(traceback.format_exc())
            self.last_search_error = str(e)
            return None



    def _download_model_from_prefix(self, model_prefix, state_dict_only=False):
        """
        Download model files from MinIO given a prefix.

        :param model_prefix: Prefix path in MinIO bucket
        :param state_dict_only: return the checkpoint's own state dict, after
            the decoder-version check, instead of a rebuilt model. A warm start
            needs this: the non-strict rebuild would fill keys the source lacks
            with random init and drop the keys only the source has.
        :return: Loaded PyTorch model (or its state dict, see above)
        """
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_path = Path(temp_dir)
                
                # Download model files
                model_file = temp_path / "model.pth"
                metadata_file = temp_path / "metadata.json"
                
                self.client.fget_object(
                    self.bucket_name,
                    f"{model_prefix}/model.pth",
                    str(model_file)
                )
                
                self.client.fget_object(
                    self.bucket_name,
                    f"{model_prefix}/metadata.json",
                    str(metadata_file)
                )
                
                # Load metadata
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
                
                print(f"Loading model: {metadata.get('name', 'unknown')}")
                print(f"Architecture: {metadata.get('architecture', 'unknown')}")
                print(f"Upload timestamp: {metadata.get('upload_timestamp', 'unknown')}")
                
                # Load model state dict
                if torch.cuda.is_available():
                    state_dict = torch.load(model_file)
                else:
                    state_dict = torch.load(model_file, map_location=torch.device('cpu'))
                
                # Try to load model class info if available
                try:
                    model_info_file = temp_path / "model_info.pkl"
                    self.client.fget_object(
                        self.bucket_name,
                        f"{model_prefix}/model_info.pkl",
                        str(model_info_file)
                    )
                    
                    with open(model_info_file, 'rb') as f:
                        model_info = pickle.load(f)

                    if state_dict_only:
                        check_decoder_arch(
                            (model_info.get("model_config") or {}).get("decoder_arch"),
                            state_dict)
                        return state_dict

                    model = build_model_from_info(model_info, state_dict)

                except ArchMismatchError:
                    # Not degraded to "state dict only": the caller must see it.
                    raise
                except Exception as e:
                    # type + repr: the historical failure was a bare assert,
                    # whose str() is empty and left this line saying nothing.
                    print(f"Warning: Could not load model class info: "
                          f"{type(e).__name__}: {e!r}")
                    print("Returning state dict only. You'll need to manually load it into your model class.")
                    return state_dict
                
                return model
                
        except ArchMismatchError:
            raise
        except Exception as e:
            print(f"Error downloading model: {e}")
            print(traceback.format_exc())
            return None

    def list_models(self):
        """
        List all available models in the MinIO bucket.
        
        :return: List of model information dictionaries
        """
        try:
            models = []
            objects = self.client.list_objects(self.bucket_name, recursive=True)
            
            for obj in objects:
                if obj.object_name.endswith('metadata.json'):
                    try:
                        with tempfile.NamedTemporaryFile() as temp_file:
                            self.client.fget_object(
                                self.bucket_name,
                                obj.object_name,
                                temp_file.name
                            )
                            
                            with open(temp_file.name, 'r') as f:
                                metadata = json.load(f)
                            
                            models.append({
                                'name': metadata.get('name'),
                                'architecture': metadata.get('architecture'),
                                'upload_timestamp': metadata.get('upload_timestamp'),
                                'path': '/'.join(obj.object_name.split('/')[:-1])
                            })
                    
                    except Exception as e:
                        print(f"Error reading metadata from {obj.object_name}: {e}")
                        continue
            
            return sorted(models, key=lambda x: x['upload_timestamp'], reverse=True)
            
        except Exception as e:
            print(f"Error listing models: {e}")
            print(traceback.format_exc())
            return []
