import io
import os
import random
import traceback
import torch
from torch.utils.data import DataLoader, Dataset
from PIL import Image, PngImagePlugin, ImageDraw
from typing import Literal
import numpy as np
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode
try:
    from pycocotools import mask as mask_utils # pylint: disable=import-error pylance: disable=reportMissingModuleSource
except Exception as e:
    print(f"pycocotools not available for RLE decode: {e}")
    raise e

# Disable HuggingFace progress bars BEFORE importing datasets
os.environ['HF_HUB_DISABLE_PROGRESS_BARS'] = '1'
os.environ['HF_DATASETS_DISABLE_PROGRESS_BARS'] = '1'

from datasets import load_dataset, DownloadConfig, disable_progress_bar, Image as HFImage, IterableDataset as HFIterableDataset
import tempfile
from pathlib import Path
import warnings

# Suppress dataset library warnings about PIL Images in nested structures
warnings.filterwarnings('ignore', category=FutureWarning, module='datasets')
warnings.filterwarnings('ignore', message='.*PngImageFile.*')

# Disable progress bars programmatically
disable_progress_bar()

# pylint: disable=import-error disable=no-name-in-module
from ecs_hcm_model.core.model_upload_download import (
    ModelUploaderAndDownloader,
    infer_segmentation_classes_from_state_dict,
    check_resume_metadata,
    DEFAULT_OUT_STRIDE,
    resume_bucket_for,
    task_for_bucket,
) 
from ecs_hcm_model.core.GeneratorDataset import GeneratorDataset
from ecs_hcm_model.core.bdd import BDD_DATASET_NAME, BDD100kSegDataset, prepared_root
from functools import partial
from huggingface_hub import login, logout
from ecs_hcm_model.core.hcm_nn import (
    DECODER_V1,
    DECODER_V2,
    DEFAULT_BACKBONE,
    MASK_STRIDE_DEFAULT,
    hcm_nn,
    check_backbone,
    check_mask_stride,
    check_neck_channels,
    infer_decoder_arch,
    neck_argument,
)

MLFLOW_DEV = "http://10.0.1.102:9090"
MLFLOW_PROD = "http://10.0.1.86:9091"
PngImagePlugin.MAX_TEXT_CHUNK = 20000000

_MASK_STATS = {"poly": 0, "rle": 0, "bbox": 0, "empty": 0, "total": 0}


def download_coco_annotations():
    """
    Download COCO annotations from MinIO bucket to temporary directory.
    
    Returns:
        Path to the downloaded annotations file
    """
    try:
        print("Downloading COCO annotations from MinIO...")
        downloader = ModelUploaderAndDownloader(bucket_name="cocoa-annotations")
        
        # Create temporary directory for annotations
        temp_dir = tempfile.mkdtemp(prefix="coco_annotations_")
        temp_path = Path(temp_dir)
        annotations_file = temp_path / "cocoa_annotation.tar.gz"
        
        # List objects in the bucket to find the annotation file
        objects = list(downloader.client.list_objects(
            downloader.bucket_name,
            recursive=True
        ))
        
        if not objects:
            raise FileNotFoundError("No files found in cocoa-annotations bucket")
        
        # Download the first tar.gz file found (assuming there's one annotation file)
        annotation_object = None
        for obj in objects:
            if obj.object_name.endswith('.tar.gz'):
                annotation_object = obj.object_name
                break
        
        if not annotation_object:
            # If no tar.gz found, use the first file
            annotation_object = objects[0].object_name
        
        print(f"Downloading {annotation_object} to {annotations_file}")
        downloader.client.fget_object(
            downloader.bucket_name,
            annotation_object,
            str(annotations_file)
        )
        
        print(f"COCO annotations downloaded to: {annotations_file}")
        return str(annotations_file)
        
    except Exception as e:
        print(f"Error downloading COCO annotations: {e}")
        raise

def _warn_precedence(name, requested, from_checkpoint):
    """The checkpoint's architecture wins; say so when the caller asked otherwise."""
    msg = (f"load_hcm_model: {name}={requested} was requested but the checkpoint "
           f"has {name}={from_checkpoint}; using the checkpoint's {from_checkpoint}, "
           f"because its weights only load into that architecture.")
    print(f"WARNING: {msg}")
    warnings.warn(msg, stacklevel=3)


def _arch_kwargs(model):
    """The architecture fields of a model, for a rebuild that must keep them.

    Defaults are the pre-change model, so this is also right for a model that
    predates the fields.
    """
    return dict(backbone=getattr(model, "backbone_name", DEFAULT_BACKBONE),
                neck_channels=neck_argument(getattr(model, "neck_channels", None)),
                mask_stride=getattr(model, "mask_stride", MASK_STRIDE_DEFAULT),
                amodal_outline=getattr(model, "amodal_outline", False))


def load_hcm_model(
    new_model: bool = False,
    environment: Literal["development", "production"] = "development",
    finetune: bool = False,
    model_base: Literal["models", "finetuned_models"] = "models",
    device: torch.device = None,
    num_segmentation_classes: int | None = None,   # <-- add this
    out_stride: int | None = 16,                   # <-- keep consistent
    task: str | None = None,
    input_size: int | None = None,
    decoder_arch: str = DECODER_V1,
    backbone: str = DEFAULT_BACKBONE,
    neck_channels: int | None = None,
    mask_stride: int = MASK_STRIDE_DEFAULT,
    amodal_outline: bool = False,
) -> torch.nn.Module:

    print('*' * 80)
    print('*************** Loading HCM model')
    print('*' * 80)

    if out_stride is None:
        out_stride = DEFAULT_OUT_STRIDE
    # NECK_LEGACY is the helpers' spelling of the pre-change widening; the
    # constructor spells it None.
    neck_channels = neck_argument(neck_channels)

    if new_model:
        # decoder_arch only shapes a new model; a loaded one keeps the version its
        # weights were trained in. The backbone/neck/mask below likewise shape a
        # new model only: a loaded one keeps the architecture its weights are in.
        model = hcm_nn(enable_finetuning=finetune,
                       num_segmentation_classes=num_segmentation_classes,
                       out_stride=out_stride,
                       decoder_arch=decoder_arch,
                       backbone=backbone,
                       neck_channels=neck_channels,
                       mask_stride=mask_stride,
                       # Shapes a new model only, like the three above: a loaded
                       # one keeps the head its weights carry (or its record).
                       amodal_outline=amodal_outline,
                       # A new model is a training start: the v2 backbones come
                       # up on their pretrained ImageNet weights (the DAG's
                       # initialisation), which a loader never fetches.
                       backbone_pretrained=True)
        return model.to(device) if device else model

    tracking_uri = MLFLOW_DEV if environment == "development" else MLFLOW_PROD
    print(f"Tracking URI: {tracking_uri}")

    # The bucket comes from the run's task and --model-base; a combination that
    # doesn't map raises instead of quietly falling back to "models". Callers
    # that don't say their task get the bucket checked, not the transition.
    if task is not None:
        bucket = resume_bucket_for(task, model_base)
    elif task_for_bucket(model_base) is not None:
        bucket = model_base
    else:
        bucket = resume_bucket_for("classification", model_base)  # raises, naming it
    model_downloader = ModelUploaderAndDownloader(bucket_name=bucket)

    model_or_state = model_downloader.search_and_download_model_by_architecture("hcm")
    if model_or_state is None:
        reason = getattr(model_downloader, "last_search_error", None) or "unknown reason"
        raise ValueError(
            f"No HCM checkpoint to resume from in bucket "
            f"'{model_downloader.bucket_name}': {reason}"
        )
    resume_metadata = getattr(model_downloader, "resume_metadata", None) or {}
    check_resume_metadata(resume_metadata.get("custom_metadata") or {}, task, input_size)

    # ---- IMPORTANT: normalize return type ----
    # Precedence: the checkpoint decides the architecture (its weights only
    # load into it); the caller decides the mode.
    if isinstance(model_or_state, torch.nn.Module):
        model = model_or_state
        head = getattr(model, "segmentation_head", None)
        ckpt_stride = getattr(model, "out_stride", None)
        if not getattr(model, "out_stride_recorded", True):
            # A legacy checkpoint recorded no stride; the rebuild used the
            # default as a stand-in. The caller's stride applies, silently: the
            # stride has no weights, so the same state dict loads at any stride.
            if out_stride != ckpt_stride:
                ckpt_missing = getattr(model, "checkpoint_missing_keys", [])
                sd = model.state_dict()
                model = hcm_nn(enable_finetuning=finetune,
                               num_segmentation_classes=head.out_channels if head is not None else None,
                               out_stride=out_stride,
                               decoder_arch=getattr(model, "decoder_arch", DECODER_V1),
                               **_arch_kwargs(model),
                               backbone_pretrained=False)
                model.load_state_dict(sd, strict=True)
                model.checkpoint_missing_keys = ckpt_missing
                head = getattr(model, "segmentation_head", None)
            ckpt_stride = out_stride
        elif ckpt_stride is not None and out_stride != ckpt_stride:
            _warn_precedence("out_stride", out_stride, ckpt_stride)
        if head is not None:
            if num_segmentation_classes is not None and num_segmentation_classes != head.out_channels:
                _warn_precedence("num_segmentation_classes",
                                 num_segmentation_classes, head.out_channels)
        elif num_segmentation_classes is not None:
            # The checkpoint has no head and the caller wants one: recreate at
            # the checkpoint's out_stride (modules can't be "auto-added" from a
            # checkpoint cleanly).
            ckpt_missing = getattr(model, "checkpoint_missing_keys", [])
            sd = model.state_dict()
            model = hcm_nn(enable_finetuning=finetune,
                           num_segmentation_classes=num_segmentation_classes,
                           out_stride=ckpt_stride or out_stride,
                           decoder_arch=getattr(model, "decoder_arch", DECODER_V1),
                           **_arch_kwargs(model),
                           backbone_pretrained=False)
            missing, unexpected = model.load_state_dict(sd, strict=False)
            print("Recreated model to add segmentation head.")
            if missing: print("Missing keys:", missing)
            if unexpected: print("Unexpected keys:", unexpected)
            model.checkpoint_missing_keys = ckpt_missing
        ckpt_missing = getattr(model, "checkpoint_missing_keys", [])

    elif isinstance(model_or_state, dict) or str(type(model_or_state)).endswith("OrderedDict'>"):
        state_dict = model_or_state

        # The checkpoint's head size wins: a different one would not load.
        inferred = infer_segmentation_classes_from_state_dict(state_dict)
        if inferred is not None:
            if num_segmentation_classes is not None and num_segmentation_classes != inferred:
                _warn_precedence("num_segmentation_classes", num_segmentation_classes, inferred)
            elif num_segmentation_classes is None:
                print(f"Inferred num_segmentation_classes={inferred} from checkpoint.")
            num_segmentation_classes = inferred

        # A bare state dict carries no out_stride, so the caller's is used; its
        # keys do say the decoder version.
        model = hcm_nn(enable_finetuning=finetune,
                       num_segmentation_classes=num_segmentation_classes,
                       out_stride=out_stride,
                       decoder_arch=infer_decoder_arch(state_dict),
                       # No record to consult: below are the "predates the
                       # fields" rules, weights first, and their defaults (the
                       # resnet50 / widening / stride-4 model) when the weights
                       # are silent.
                       backbone=check_backbone(None, state_dict),
                       neck_channels=neck_argument(check_neck_channels(None, state_dict)),
                       mask_stride=check_mask_stride(None, state_dict),
                       # The state dict is loaded over the backbone below, so
                       # nothing is fetched: loading a checkpoint is offline.
                       backbone_pretrained=False)

        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if unexpected:
            print(f"Warning: Unexpected keys in checkpoint: {unexpected[:10]}{'...' if len(unexpected)>10 else ''}")
        if missing:
            print(f"Warning: Missing keys in checkpoint: {missing[:10]}{'...' if len(missing)>10 else ''}")
        ckpt_missing = missing

    else:
        raise TypeError(f"search_and_download_model_by_architecture returned unsupported type: {type(model_or_state)}")

    if not finetune and any(k.startswith("roi_classifier.") for k in ckpt_missing):
        raise ValueError(
            f"classification was requested (finetune=False) but the checkpoint "
            f"from bucket {model_downloader.bucket_name!r} has no classification "
            f"head (no roi_classifier weights); its classifier would be untrained. "
            f"Load a classification checkpoint, or pass finetune=True."
        )
    model.enable_finetuning = bool(finetune)
    # The parent checkpoint, so every checkpoint this run uploads records what
    # it continued from (a fresh model has no such attribute: null).
    prefix = getattr(model_downloader, "resume_prefix", None)
    model.resumed_from = f"{model_downloader.bucket_name}/{prefix}" if prefix else None

    return model.to(device) if device else model


def warm_start_state(model, source_state: dict) -> dict:
    """Copy every tensor of ``source_state`` whose name and shape match into ``model``.

    The warm start from a checkpoint of another decoder version: what the two
    share is copied, a tensor whose shape changed is skipped and keeps its
    init, and a tensor only the model has stays at init. The exception is a v1
    source into a v2 model: each rrd fuse conv is initialised to [I | 0] with
    zero bias (``initialised``), so it passes the normal branch through and the
    copied RCAB/conv2d see what they were trained on.
    Returns the key lists and the fraction of the model's elements copied.
    """
    target = getattr(model, "module", model)
    target_state = target.state_dict()
    copied, skipped, new = [], [], []
    loadable = {}
    for name, tensor in target_state.items():
        src = source_state.get(name)
        if src is None:
            new.append(name)
        elif tuple(src.shape) != tuple(tensor.shape):
            skipped.append(name)
        else:
            loadable[name] = src
            copied.append(name)
    # Source tensors with no counterpart in the model: reported, never loaded.
    dropped = [name for name in source_state if name not in target_state]
    initialised = []
    if (infer_decoder_arch(source_state) == DECODER_V1
            and infer_decoder_arch(target_state) == DECODER_V2):
        for name in list(new):
            if not (name.startswith("rrd_list.") and ".fuse." in name):
                continue
            tensor = target_state[name]
            if name.endswith(".weight"):
                # (out, 2*out, 1, 1): identity on the normal half, zero on the reversed.
                out_ch = tensor.shape[0]
                value = torch.zeros_like(tensor)
                value[:, :out_ch, 0, 0] = torch.eye(out_ch, dtype=tensor.dtype)
            else:
                value = torch.zeros_like(tensor)
            loadable[name] = value
            new.remove(name)
            initialised.append(name)
    target.load_state_dict(loadable, strict=False)

    total = sum(t.numel() for t in target_state.values())
    copied_numel = sum(target_state[name].numel() for name in copied)
    return {
        "copied": copied,
        "skipped": skipped,
        "new": new,
        "initialised": initialised,
        "dropped": dropped,
        "copied_fraction": (copied_numel / total) if total else 0.0,
    }


def parse_checkpoint_spec(spec: str) -> tuple:
    """``(bucket, prefix)`` from a checkpoint named as ``<bucket>/<prefix>``."""
    bucket, _, prefix = str(spec).strip().strip("/").partition("/")
    if not bucket or not prefix:
        raise ValueError(
            f"a checkpoint is named <bucket>/<prefix>, e.g. "
            f"coco-models/hcm_coco_seg_best/20261001_035743 (got {spec!r})"
        )
    return bucket, prefix


def download_checkpoint_state(spec: str) -> dict:
    """The checkpoint's own state dict, as saved, after the decoder-version check.

    Never a rebuilt model's state dict: a non-strict rebuild would report its
    random init as source weights and lose the keys only the source has.
    """
    bucket, prefix = parse_checkpoint_spec(spec)
    loaded = ModelUploaderAndDownloader(bucket_name=bucket)._download_model_from_prefix(
        prefix, state_dict_only=True)
    if loaded is None:
        raise ValueError(f"could not download checkpoint {bucket}/{prefix}")
    return loaded




class SmallDictDataset(Dataset):
    def __init__(self, images, labels):
        self.images = images
        self.labels = labels

    def __len__(self):
        return self.images.size(0)

    def __getitem__(self, idx):
        return {
            "pixel_values": self.images[idx],
            "label": self.labels[idx],
        }


def make_small_train_loader_from_stream(train_loader, num_batches=32):
    images_list = []
    labels_list = []

    for i, batch in enumerate(train_loader):
        images_list.append(batch["pixel_values"])
        labels_list.append(batch["label"])
        if i + 1 >= num_batches:
            break

    images = torch.cat(images_list, dim=0)
    labels = torch.cat(labels_list, dim=0)

    small_dataset = SmallDictDataset(images, labels)

    small_loader = DataLoader(
        small_dataset,
        batch_size=train_loader.batch_size,
        shuffle=True,
        num_workers=0,       # simpler for debug/overfit test
        pin_memory=True,
    )
    return small_loader

def convert_bytes_to_image(x):
    try:
        if isinstance(x, Image.Image):
            return x

        if isinstance(x, (bytes, bytearray)):
            return Image.open(io.BytesIO(x)).convert("RGB")

        if isinstance(x, str):
            # HuggingFace datasets can give pseudo-paths like "zip:/..."
            if x.startswith("zip:"):
                return None
            if os.path.exists(x):
                return Image.open(x).convert("RGB")
            return None

        if isinstance(x, dict):
            if "bytes" in x and x["bytes"] is not None:
                return Image.open(io.BytesIO(x["bytes"])).convert("RGB")

            if "path" in x and isinstance(x["path"], str):
                p = x["path"]
                if p.startswith("zip:"):
                    return None
                if os.path.exists(p):
                    return Image.open(p).convert("RGB")

    except Exception as e:  # pylint: disable=broad-exception-caught
        print(f"Error converting to image: {e}")

    return None


def train_transforms(examples):
    normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    nn_transforms = T.Compose([
        T.RandomResizedCrop(192), 
        T.RandomHorizontalFlip(), 
        T.ToTensor(), 
        normalize
    ])

    try:
        examples["pixel_values"] = [
            nn_transforms(convert_bytes_to_image(image_data["bytes"]).convert("RGB"))
            for image_data in examples["image"]
        ]
        return examples
    except Exception: # pylint: disable=broad-exception-caught
        try:
            examples["pixel_values"] = [
                nn_transforms(convert_bytes_to_image(examples["image"]).convert("RGB"))
            ]
            return examples
        except Exception as e: # pylint: disable=broad-exception-caught
            print(f"Transform error: {e}")
            return None

def val_transforms(examples):
    normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    nn_transforms = T.Compose([
        T.Resize(256), 
        T.CenterCrop(192),
        T.ToTensor(), 
        normalize
    ])

    try:
        examples["pixel_values"] = [
            nn_transforms(convert_bytes_to_image(image_data["bytes"]).convert("RGB"))
            for image_data in examples["image"]
        ]
        return examples
    except Exception: # pylint: disable=broad-exception-caught
        try:
            examples["pixel_values"] = [
                nn_transforms(convert_bytes_to_image(examples["image"]).convert("RGB"))
            ]
            return examples
        except Exception as e: # pylint: disable=broad-exception-caught
            print(f"Transform error: {e}")
            return None

def collate_fn(batch):
    if batch:
        return {
            'pixel_values': torch.stack([x['pixel_values'][0] for x in batch]),
            'label': torch.stack([x['label'] for x in batch])
        }
    return None



def _poly_to_mask_cocoa(poly_xy, H, W, debug=False):
    """
    poly_xy: (N,2) float/int array of x,y in image coords
    returns uint8 mask (H,W) in {0,1}

    Rasterised by pycocotools, the same rasteriser behind the COCOA validation
    masks: PIL's polygon fill drew about a pixel wider, so the train target ran
    1-2% larger than the val target for the same object.
    """
    pts = []
    for x, y in poly_xy:
        x = float(x); y = float(y)
        if np.isnan(x) or np.isnan(y) or np.isinf(x) or np.isinf(y):
            continue
        pts.extend((x, y))

    if debug:
        xs = poly_xy[:,0]; ys = poly_xy[:,1]
        print("poly min/max x:", xs.min(), xs.max(), "W:", W)
        print("poly min/max y:", ys.min(), ys.max(), "H:", H)

    # Need at least 3 points for a polygon
    if len(pts) < 6:
        return np.zeros((H, W), dtype=np.uint8)

    rle = mask_utils.merge(mask_utils.frPyObjects([pts], H, W))
    return (mask_utils.decode(rle) > 0).astype(np.uint8)


def _decode_rle(rle, H, W, debug=False):
    """
    rle: dict with keys 'counts' and 'size' (COCO format)
    returns uint8 mask (H,W) in {0,1} or None if cannot decode
    """


    try:
        m = mask_utils.decode(rle)
        # decode can return (H,W) or (H,W,1)
        if m.ndim == 3:
            m = m[..., 0]
        # COCO sometimes stores size in rle; trust decoded size but ensure matches
        if m.shape != (H, W):
            if debug:
                print(f"RLE decoded shape {m.shape} != expected {(H,W)}; will try to fit via rle['size']")
        return (m > 0).astype(np.uint8)
    except Exception as e: # pylint: disable=broad-exception-caught
        if debug:
            print(f"Failed to decode RLE: {e} (rle keys: {list(rle.keys())})")
        return None

def build_binary_mask_coco_all(sample, *, include_crowd=True, use_bbox_fallback=True, debug=False, stats=None,
                               with_outline=False):
    """
    Mask builder for every COCO/COCOA transform: ALWAYS returns the full (union) binary mask over ALL instances.

    ``stats``, when a dict, receives ``dropped_regions``: the COCOA regions (one
    per exploded instance, however many parts it has) of which no part reached
    the target, for any reason: near-full fill, unusable shape, a polygon field
    with no usable polygon, an unsupported or None part, no segmentation at all,
    or a region that is not a dict. ``partial_regions`` counts regions that kept
    some parts and lost others. Stuff and background regions are excluded on
    purpose and not counted. The transforms carry ``dropped_regions`` per sample
    so a drop is counted, never silent.

    Supports:
      A) COCOA-style: annotations is dict with "regions"  (uses explode_cocoa_region + visible_mask preference)
      B) COCO-style: annotations is list/dict of instance dicts

    Returns: np.ndarray [H,W] int64 values {0,1}

    ``with_outline=True``: returns ``(visible, outline)`` instead, both [H,W]
    int64. ``outline`` is the union of the *full* (amodal) polygons of exactly
    the regions the visible builder keeps, then ``| visible`` so the full label
    contains the visible one. A kept region whose full polygon is missing or
    unusable makes the whole sample's outline label unknown: the outline is
    all-zero (never built from the visible mask) and ``stats`` gets
    ``outline_unknown_images += 1``. Non-COCOA samples (COCO instances, BDD
    precomputed masks) have no amodal labels at all and return an all-zero
    outline. With the flag off the return is exactly today's array.
    """
    import numpy as np
    from PIL import Image

    img = sample.get("image", None)
    if img is None:
        return None

    if isinstance(img, Image.Image):
        W, H = img.size
    else:
        W = getattr(img, "width", None)
        H = getattr(img, "height", None)
        if W is None or H is None:
            return None

    ann = sample.get("annotations", None)
    out = np.zeros((H, W), dtype=np.uint8)
    outline = np.zeros((H, W), dtype=np.uint8)
    _outline_state = {"has_full": True}

    def _finish(visible, has_full=True):
        """Today's array, or ``(visible, outline)`` when outline targets are on."""
        if not with_outline:
            return visible.astype(np.int64)
        if not has_full:
            return visible.astype(np.int64), np.zeros((H, W), dtype=np.int64)
        return visible.astype(np.int64), (outline | visible).astype(np.int64)

    # -------------------------
    # Case A: COCOA-style (dict with regions)  (ALL instances, robust handling)
    # -------------------------
    if isinstance(ann, dict) and isinstance(ann.get("regions", None), list):
        if debug:
            print("cocoa-style annotations detected (ALL instances)")

        # local stats (optional)
        _stats = {
            "regions_in": 0,
            "regions_exploded": 0,
            "stuff_skipped": 0,
            "bgname_skipped": 0,
            "invalid_region": 0,
            "seg_none": 0,
            "pil": 0,
            "dense": 0,
            "rle": 0,
            "poly": 0,
            "poly_reject_span": 0,
            "poly_reject_area": 0,
            "poly_reject_fill": 0,
            "bad_shape": 0,
            "poly_unparsed": 0,
            "unknown_type": 0,
            "seg_part_none": 0,
            "added": 0,
            # Per region (one exploded instance), not per part: a region is
            # dropped when none of its parts reach the target, partial when
            # some do and some are left out.
            "dropped_regions": 0,
            "partial_regions": 0,
        }

        def _nearest():
            return Image.Resampling.NEAREST if hasattr(Image, "Resampling") else Image.NEAREST  # pylint: disable=no-member

        def _as_hw(mask):
            """Ensure mask is 2D and shaped (H,W) if possible, else None."""
            if mask is None:
                return None
            m = ensure_mask_hw(mask, img)
            if m is None:
                return None
            m = np.asarray(m)
            if m.ndim != 2:
                return None
            if m.shape == (H, W):
                return m
            if m.shape == (W, H):
                return m.T
            return None

        def _union_mask(m, tag=""):
            """Sanitize + union into out, with shape checks."""
            nonlocal out
            m = _as_hw(m)
            if m is None:
                _stats["bad_shape"] += 1
                if debug:
                    print(f"[cocoa_all] skip {tag}: bad shape")
                return False
            m = sanitize_instance_mask(m, debug=debug, tag=tag)
            if m is None:
                _stats["bad_shape"] += 1
                return False
            # hard guardrail: avoid "near-full-image" artifacts from bad polys/masks
            frac = float(m.mean())
            if frac > 0.995:
                _stats["poly_reject_fill"] += 1
                if debug:
                    print(f"[cocoa_all] reject {tag}: near-full fill frac={frac:.4f}")
                return False
            out |= m.astype(np.uint8)
            _stats["added"] += 1
            return True

        def _report():
            if isinstance(stats, dict):
                stats["dropped_regions"] = (
                    stats.get("dropped_regions", 0) + _stats["dropped_regions"])
                stats["partial_regions"] = (
                    stats.get("partial_regions", 0) + _stats["partial_regions"])
                if with_outline and not _outline_state["has_full"]:
                    stats["outline_unknown_images"] = (
                        stats.get("outline_unknown_images", 0) + 1)

        def _full_outline_parts(r):
            """The region's *full* parts as binary masks, or None.

            ``segmentation`` is COCOA's amodal polygon; ``visible_mask`` (when
            present) is what the visible builder preferred. Same part forms the
            visible loop handles, decoded by the same helpers; a part this
            cannot decode returns None rather than guessing.
            """
            segs = r.get("segmentation", None)
            if segs is None:
                return None
            if not isinstance(segs, list) or (
                segs and isinstance(segs[0], (int, float, np.number))
            ):
                segs = [segs]
            if not segs:
                # An empty polygon field is no full label, not an empty one.
                return None
            masks = []
            for seg in segs:
                if seg is None:
                    return None
                if isinstance(seg, Image.Image):
                    m = seg
                    if m.mode not in ("L", "1"):
                        m = m.convert("L")
                    if m.size != (W, H):
                        m = m.resize((W, H), resample=_nearest())
                    m_arr = np.asarray(m)
                    if m_arr.ndim == 3:
                        m_arr = m_arr[..., 0]
                    masks.append((m_arr > 0).astype(np.uint8))
                    continue
                if isinstance(seg, np.ndarray) and seg.ndim == 2:
                    masks.append((seg > 0).astype(np.uint8))
                    continue
                if isinstance(seg, dict) and ("counts" in seg) and ("size" in seg):
                    m, _ = decode_coco_rle_robust(
                        rle=seg, expected_h=H, expected_w=W, debug=False)
                    if m is None:
                        return None
                    masks.append(m)
                    continue
                if isinstance(seg, (list, tuple, np.ndarray)):
                    polys = _seg_to_polys(seg)
                    if not polys:
                        return None
                    for poly in polys:
                        masks.append(_poly_to_mask_cocoa(poly, H, W, debug=False))
                    continue
                return None
            return masks

        def _union_outline(masks):
            """Union the full parts into ``outline``; False if one is unusable."""
            for m in masks:
                m = _as_hw(m)
                if m is None:
                    return False
                m = sanitize_instance_mask(m, debug=debug, tag="outline")
                if m is None:
                    return False
                # The same ceiling the visible builder applies, so the full
                # label cannot be rejected for a reason the visible one was
                # not, and full ⊇ visible holds for every kept region.
                if float(m.mean()) > 0.995:
                    return False
                outline[:] |= m.astype(np.uint8)
            return True

        try:
            regions = ann["regions"]
            _stats["regions_in"] = len(regions)

            for r0 in regions:
                # explode_cocoa_region returns [] for a non-dict, which would
                # lose the region without a count.
                if not isinstance(r0, dict):
                    _stats["invalid_region"] += 1
                    _stats["dropped_regions"] += 1
                    continue
                for r in explode_cocoa_region(r0):
                    _stats["regions_exploded"] += 1

                    if not isinstance(r, dict):
                        _stats["invalid_region"] += 1
                        _stats["dropped_regions"] += 1
                        continue

                    # Skip stuff and explicit background-ish labels
                    if cocoa_region_is_stuff(r):
                        _stats["stuff_skipped"] += 1
                        continue
                    nm = str(r.get("name", "")).strip().lower()
                    if nm == "background" or "background" in nm:
                        _stats["bgname_skipped"] += 1
                        continue

                    # Prefer visible_mask over segmentation (this was key for COCOA)
                    segs = r.get("visible_mask", None) or r.get("segmentation", None)
                    if segs is None:
                        _stats["seg_none"] += 1
                        _stats["dropped_regions"] += 1
                        continue
                    # A flat polygon [x1, y1, ...] is itself a list: wrap it too, or
                    # the loop walks its coordinates and drops the region. This is
                    # the shape COCOA's decode_rle=False config yields per region.
                    if not isinstance(segs, list) or (
                        segs and isinstance(segs[0], (int, float, np.number))
                    ):
                        segs = [segs]

                    kept = lost = 0

                    def _tally(ok):
                        nonlocal kept, lost
                        if ok:
                            kept += 1
                        else:
                            lost += 1

                    for seg in segs:
                        if seg is None:
                            _stats["seg_part_none"] += 1
                            lost += 1
                            continue

                        # 0) PIL image mask
                        if isinstance(seg, Image.Image):
                            _stats["pil"] += 1
                            m = seg
                            if m.mode not in ("L", "1"):
                                m = m.convert("L")
                            if m.size != (W, H):
                                m = m.resize((W, H), resample=_nearest())
                            m_arr = np.asarray(m)
                            if m_arr.ndim == 3:
                                m_arr = m_arr[..., 0]
                            _tally(_union_mask((m_arr > 0).astype(np.uint8), tag="cocoa PIL"))
                            continue

                        # 1) Dense 2D mask
                        if isinstance(seg, np.ndarray) and seg.ndim == 2:
                            _stats["dense"] += 1
                            _tally(_union_mask((seg > 0).astype(np.uint8), tag="cocoa dense"))
                            continue

                        # 2) RLE dict
                        if isinstance(seg, dict) and ("counts" in seg) and ("size" in seg):
                            _stats["rle"] += 1
                            m, info = decode_coco_rle_robust(
                                rle=seg,
                                expected_h=H,
                                expected_w=W,
                                debug=debug,
                            )
                            if debug:
                                print(f"[cocoa_all] RLE decode info: {info}")
                            _tally(_union_mask(m, tag="cocoa RLE"))
                            continue

                        # 3) Polygon(s)
                        if isinstance(seg, (list, tuple, np.ndarray)):
                            polys = _seg_to_polys(seg)
                            if not polys:
                                # A region with a polygon field that holds no
                                # usable polygon is left out: count it.
                                _stats["poly_unparsed"] += 1
                                lost += 1
                                continue

                            for poly in polys:
                                _stats["poly"] += 1

                                # No span or area guard: an amodal or visible outline
                                # can legitimately cover most of the frame, and the val
                                # path (pycocotools masks) never applied one, so the
                                # two targets disagreed. _union_mask's near-full fill
                                # check is the one guard both paths share.
                                m = _poly_to_mask_cocoa(poly, H, W, debug=debug)
                                _tally(_union_mask(m, tag="cocoa poly"))
                            continue

                        # Unknown seg type -> skip, counted
                        _stats["unknown_type"] += 1
                        lost += 1
                        if debug:
                            print(f"[cocoa_all] skipping seg unsupported type: {type(seg)}")

                    if kept == 0:
                        _stats["dropped_regions"] += 1
                    elif lost:
                        _stats["partial_regions"] += 1

                    if with_outline and kept > 0:
                        # A region the visible target keeps must contribute its
                        # full label. If any part of it has none, the sample's
                        # whole outline target is unknown (zeroed in _finish):
                        # a lower target built from some regions only would be
                        # trained on as if it were complete.
                        parts = _full_outline_parts(r)
                        if parts is None or not _union_outline(parts):
                            _outline_state["has_full"] = False

            if debug:
                print("[cocoa_all] stats:", _stats, "out_fg_frac=", float(out.mean()))

            _report()
            return _finish(out, _outline_state["has_full"])

        except Exception as e:  # pylint: disable=broad-exception-caught
            if debug:
                print(f"Error processing COCOA-style annotations: {e}")
            return None

    # -------------------------
    # Case B: COCO-style instances (list-of-dicts or dict-of-lists)
    # -------------------------
    if ann is None:
        return _finish(out, False)

    # Normalize annotations into list-of-dicts
    if isinstance(ann, dict):
        # dict-of-lists -> list-of-dicts (HF sometimes)
        if "segmentation" in ann and isinstance(ann["segmentation"], list) and (
            len(ann["segmentation"]) > 0 and isinstance(ann["segmentation"][0], (list, dict))
        ):
            keys = list(ann.keys())
            n = len(ann["segmentation"])
            ann_list = [{k: ann.get(k, [None] * n)[i] for k in keys} for i in range(n)]
        else:
            ann_list = [ann]
    elif isinstance(ann, (list, tuple)):
        ann_list = list(ann)
    else:
        return _finish(out, False)

    def _clip_poly(poly):
        poly = np.asarray(poly, dtype=np.float32)
        if poly.ndim != 1 or poly.size < 6:
            return None
        xs = np.clip(poly[0::2], 0, W - 1)
        ys = np.clip(poly[1::2], 0, H - 1)
        outp = np.empty_like(poly)
        outp[0::2] = xs
        outp[1::2] = ys
        return outp.tolist()

    def _poly_to_mask(polys):
        if not polys:
            return None

        # single polygon flat list -> wrap
        if isinstance(polys, list) and len(polys) > 0 and isinstance(polys[0], (int, float)):
            polys = [polys]

        cleaned = []
        for p in polys:
            if isinstance(p, list):
                cp = _clip_poly(p)
                if cp is not None:
                    cleaned.append(cp)

        if not cleaned:
            return None

        rles = mask_utils.frPyObjects(cleaned, H, W)
        rle = mask_utils.merge(rles)
        m = mask_utils.decode(rle)  # HxW uint8
        return m

    def _bbox_to_mask(bbox):
        if not bbox or not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
            return None
        x, y, bw, bh = bbox
        x1 = int(max(0, np.floor(x)))
        y1 = int(max(0, np.floor(y)))
        x2 = int(min(W, np.ceil(x + bw)))
        y2 = int(min(H, np.ceil(y + bh)))
        if x2 <= x1 or y2 <= y1:
            return None
        m = np.zeros((H, W), dtype=np.uint8)
        m[y1:y2, x1:x2] = 1
        return m

    used = 0
    for a in ann_list:
        if not isinstance(a, dict):
            continue

        iscrowd = int(a.get("iscrowd", 0) or 0)
        if (not include_crowd) and iscrowd == 1:
            continue

        seg = a.get("segmentation", None)
        m = None

        if isinstance(seg, list):
            m = _poly_to_mask(seg)
        elif isinstance(seg, dict):
            W, H = img.size
            m, info = decode_coco_rle_robust(rle = seg, expected_h=H, expected_w=W, debug=debug)
            if debug:
                print("RLE decode mode:", info)
            m_bin = sanitize_instance_mask(m, debug=debug, tag="coco RLE")
            m = m_bin

        if m is None and use_bbox_fallback:
            m = _bbox_to_mask(a.get("bbox", None))

        if m is None:
            continue

        if m.shape != (H, W):
            if m.shape == (W, H):
                m = m.T
            else:
                continue

        out |= (m > 0).astype(np.uint8)
        used += 1

    if debug and out.sum() == 0:
        n = len(ann_list)
        print(f"[COCO MASK DEBUG][ALL] HxW={H}x{W} anns={n} used={used} out_sum={int(out.sum())}")

    return _finish(out, False)


def sanitize_instance_mask(m, debug=False, tag=""):
    """
    Return a binary uint8 mask {0,1} or None.
    Handles:
      - bool
      - {0,1} uint8/int
      - {0,255} uint8
      - soft/grayscale uint8 (0..255 with many unique values)
    """
    if m is None:
        return None

    # Convert torch -> numpy if needed
    try:
        if isinstance(m, torch.Tensor):
            m = m.detach().cpu().numpy()
    except Exception: # pylint: disable=broad-exception-caught
        pass

    m = np.asarray(m)

    if m.ndim != 2:
        if debug:
            print(f"[sanitize_instance_mask] {tag}: expected 2D mask, got {m.shape}")
        return None

    # bool -> {0,1}
    if m.dtype == np.bool_:
        return m.astype(np.uint8)

    # Force to uint8 for uniform handling
    if m.dtype != np.uint8:
        # If it's float in [0,1], scale up; else clip
        if np.issubdtype(m.dtype, np.floating):
            if m.max() <= 1.0 + 1e-6:
                m_u8 = (m * 255.0).clip(0, 255).astype(np.uint8)
            else:
                m_u8 = m.clip(0, 255).astype(np.uint8)
        else:
            m_u8 = m.clip(0, 255).astype(np.uint8)
        m = m_u8

    # Examine unique values (cheap)
    uniq = np.unique(m)
    uniq_n = len(uniq)

    # Already binary {0,1}
    if uniq_n <= 2 and set(uniq.tolist()).issubset({0, 1}):
        out = m

    # Classic {0,255}
    elif uniq_n <= 2 and set(uniq.tolist()).issubset({0, 255}):
        out = (m == 255).astype(np.uint8)

    else:
        # Soft / grayscale mask: use HIGH threshold.
        # grayscale / soft mask
        hi = (m >= 128).astype(np.uint8)
        lo = (m <= 127).astype(np.uint8)

        frac_hi = hi.mean()
        frac_lo = lo.mean()

        # Pick the one that looks like "object": usually smaller than background
        # (tweak bounds as you like)
        if frac_hi > 0.90 and frac_lo < 0.10:
            out = lo   # inverted polarity case
            if debug:
                print(f"[sanitize_instance_mask] {tag}: using low-values as FG (frac_hi={frac_hi:.3f})")
        else:
            out = hi

        # optional extra: if out is still absurdly full, drop it
        if out.mean() > 0.95:
            if debug:
                print(f"[sanitize_instance_mask] {tag}: mask too full (frac={out.mean():.3f})")
            # return None

    # Final sanity: drop empty masks
    if out.sum() == 0:
        return None

    return out

def ensure_mask_hw(mask_np, img, image_id=None):
    """
    mask_np: numpy array
    img: PIL image (or anything with .size -> (W,H))
    Ensures mask is HxW.
    """

    if mask_np is None:
        return None

    if mask_np.ndim == 3 and mask_np.shape[-1] == 1:
        mask_np = mask_np[..., 0]
    if mask_np.ndim != 2:
        raise ValueError(f"mask must be 2D, got {mask_np.shape} (image_id={image_id})")

    W, H = img.size  # PIL gives (W,H)

    # correct
    if mask_np.shape == (H, W):
        return mask_np

    # common swapped case
    if mask_np.shape == (W, H):
        return mask_np.T

    # if it's neither, fail loudly (or log + return None)
    raise ValueError(
        f"mask shape {mask_np.shape} does not match image {(H, W)} or swapped {(W, H)} "
        f"(image_id={image_id})"
    )


def decode_coco_rle_robust(rle: dict, *, expected_h=None, expected_w=None, debug=False):
    """
    Robustly decode COCO RLE that may arrive as:
    - counts: list[int] run-lengths (uncompressed)
    - counts: list[int] bytes (compressed but JSON-ified)
    - counts: str (compressed)
    - counts: bytes (compressed)
    Returns: (mask_uint8 {0,1}, chosen_mode_str) or (None, reason)
    """
    if not isinstance(rle, dict) or "counts" not in rle:
        return None, "bad_rle_dict"

    # size
    if "size" in rle and rle["size"] is not None:
        h, w = int(rle["size"][0]), int(rle["size"][1])
    elif expected_h is not None and expected_w is not None:
        h, w = int(expected_h), int(expected_w)
    else:
        return None, "missing_size"

    counts = rle["counts"]

    # Helper: try decode with a prepared rr
    def _try(rr, mode):
        try:
            m = mask_utils.decode(rr)
            if m is None:
                return None
            if m.ndim == 3:
                m = np.any(m, axis=2)
            m = (m > 0).astype(np.uint8)
            return m
        except Exception as e: # pylint: disable=broad-exception-caught
            if debug:
                print("decode failed", mode, e)
            return None

    # Case A: counts is already bytes or str => just decode
    if isinstance(counts, (bytes, str)):
        rr = {"counts": counts, "size": [h, w]}
        m = _try(rr, "compressed_direct")
        return (m, "compressed_direct") if m is not None else (None, "compressed_direct_failed")

    # Case B: counts is list => could be uncompressed run-lengths OR byte-values of compressed
    if isinstance(counts, list):
        # Flatten if nested lists (rare but happens)
        if len(counts) > 0 and isinstance(counts[0], list):
            flat = []
            for sub in counts:
                flat.extend(sub)
            counts = flat

        # If it looks like real run-lengths, sum should equal h*w exactly
        all_int = all(isinstance(x, (int, np.integer)) for x in counts)
        if all_int:
            s = int(np.sum(counts)) if len(counts) else 0
            if s == h * w:
                # pycocotools.decode only takes compressed counts; a list of
                # run lengths (raw COCO crowd segmentations) must go through
                # frPyObjects first, or decode raises, this returns None and
                # the caller falls back to the annotation's bbox.
                rr = mask_utils.frPyObjects(
                    {"counts": [int(x) for x in counts], "size": [h, w]}, h, w)
                m = _try(rr, "uncompressed_rl")
                return (m, "uncompressed_rl") if m is not None else (None, "uncompressed_rl_failed")

            # Otherwise, very likely this is JSON-ified bytes (0..255)
            # (byte-lengths like ~100-2000 are common; sum won't match h*w)
            if all(0 <= int(x) <= 255 for x in counts):
                rr = {"counts": bytes(counts), "size": [h, w]}
                m = _try(rr, "compressed_bytes_from_list")
                return (m, "compressed_bytes_from_list") if m is not None else (None, "compressed_bytes_from_list_failed")

        return None, "counts_list_unrecognized"

    return None, f"counts_type_{type(counts)}_unsupported"

def poly_span_reject(poly_xy, H, W, *, span_thresh=0.98):
    poly_xy = np.asarray(poly_xy, dtype=np.float32)
    xs = poly_xy[:, 0]; ys = poly_xy[:, 1]
    xspan = (xs.max() - xs.min()) / max(W - 1, 1)
    yspan = (ys.max() - ys.min()) / max(H - 1, 1)
    return (xspan >= span_thresh) and (yspan >= span_thresh)

def poly_area(poly_xy):
    p = np.asarray(poly_xy, dtype=np.float32)
    x = p[:,0]; y = p[:,1]
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def _seg_to_polys(seg):
    """
    Returns list[np.ndarray (N,2)] where N>=3
    Handles:
      - [x1,y1, x2,y2, ...]
      - [[x1,y1,...], [...]]  (multi-poly)
      - np.ndarray (N,2)
    """

    if seg is None:
        return []

    # already Nx2
    if isinstance(seg, np.ndarray) and seg.ndim == 2 and seg.shape[1] == 2 and seg.shape[0] >= 3:
        return [seg.astype(np.float32)]

    # single poly: list of numbers
    if isinstance(seg, (list, tuple)) and len(seg) >= 6 and isinstance(seg[0], (int, float, np.number)):
        arr = np.asarray(seg, dtype=np.float32)
        if arr.size % 2 != 0:
            return []
        poly = arr.reshape(-1, 2)
        return [poly] if poly.shape[0] >= 3 else []

    # multi-poly: list of lists
    if isinstance(seg, (list, tuple)) and len(seg) > 0 and isinstance(seg[0], (list, tuple)):
        polys = []
        for p in seg:
            if not isinstance(p, (list, tuple)) or len(p) < 6:
                continue
            if not isinstance(p[0], (int, float, np.number)):
                continue
            arr = np.asarray(p, dtype=np.float32)
            if arr.size % 2 != 0:
                continue
            poly = arr.reshape(-1, 2)
            if poly.shape[0] >= 3:
                polys.append(poly)
        return polys

    return []

def explode_cocoa_region(r: dict):
    """
    COCOA sometimes packs regions like:
      r["name"] = [..N..]
      r["segmentation"] = [..N..]
      r["is_stuff"] = [..N..]
      ...
    This returns a list of per-instance region dicts.
    """
    if not isinstance(r, dict):
        return []

    names = r.get("name", None)
    segs  = r.get("segmentation", None)

    # If not packed, return as a single region
    if not isinstance(names, (list, tuple)) or not isinstance(segs, (list, tuple)):
        return [r]

    n = min(len(names), len(segs))
    out = []
    for i in range(n):
        ri = {
            "name": names[i],
            "segmentation": segs[i],
        }
        # carry aligned optional fields if present
        for k in ("area", "is_stuff", "occlude_rate", "order", "visible_mask", "invisible_mask"):
            v = r.get(k, None)
            if isinstance(v, (list, tuple)) and len(v) > i:
                ri[k] = v[i]
            elif not isinstance(v, (list, tuple)) and v is not None:
                # sometimes scalar; attach it
                ri[k] = v
        out.append(ri)
    return out


def _ensure_list(x, n=None):
    # Convert scalar -> [scalar]
    if isinstance(x, list):
        return x
    if x is None and n is not None:
        return [None] * n
    return [x]


# The model input size, in one place. Every COCO/COCOA segmentation transform
# builds its tensors at this size (overridable per run via --input-size), and the
# uploaded checkpoint records the size it was trained at so a consumer reads it.
DEFAULT_INPUT_SIZE = 192

# The shorter-side resize the default was built with, as a ratio rather than a
# second literal: 256 for a 192 input. Kept as a ratio so changing the input size
# scales the resize with it (352 -> ~469) instead of silently changing the crop's
# field of view.
_RESIZE_SHORT_RATIO = 256 / 192


def _resize_short_for(input_size: int) -> int:
    """The shorter-side resize that matches ``input_size`` at the default ratio."""
    return int(round(int(input_size) * _RESIZE_SHORT_RATIO))


def _apply_color_jitter(img: Image.Image, strength: float) -> Image.Image:
    """Brightness, contrast and saturation by +-strength, hue by +-strength/4.

    Image only: the caller never passes the mask here, and no mask is ever
    colour-transformed.

    The factors come from `random`, not from `torchvision.transforms.ColorJitter`:
    ColorJitter draws from the torch RNG, and the crop, the flip and the scale
    draw all use `random`. Keeping the augmentation on one stream is what lets a
    seeded run reproduce a batch with the jitter on exactly as it does with the
    jitter off, and it leaves the torch RNG state untouched for the model.
    """
    if strength <= 0:
        return img
    img = TF.adjust_brightness(img, random.uniform(1 - strength, 1 + strength))
    img = TF.adjust_contrast(img, random.uniform(1 - strength, 1 + strength))
    img = TF.adjust_saturation(img, random.uniform(1 - strength, 1 + strength))
    return TF.adjust_hue(img, random.uniform(-strength / 4, strength / 4))


def _fg_centered_crop_params(mask_pil: Image.Image, out_h: int, out_w: int):
    """
    Pick a crop window (top,left,out_h,out_w) that contains at least one FG pixel if possible.
    mask_pil: mode "L" mask with FG>0
    """
    m = np.array(mask_pil)
    ys, xs = np.where(m > 0)

    #print(f"FG pixels found: {len(xs)}")

    H, W = m.shape[:2]

    # If no FG, fall back to random crop (or center crop if too small)
    if len(xs) == 0:
        if W == out_w and H == out_h:
            return 0, 0, out_h, out_w
        return T.RandomCrop.get_params(mask_pil, (out_h, out_w))

    # Choose a random FG pixel as an anchor
    k = random.randrange(len(xs))
    cy, cx = int(ys[k]), int(xs[k])

    # Choose top/left so that (cy,cx) lies within the crop
    top_min = cy - out_h + 1
    top_max = cy
    left_min = cx - out_w + 1
    left_max = cx

    # Clamp valid ranges to image bounds
    top_min = max(top_min, 0)
    left_min = max(left_min, 0)
    top_max = min(top_max, H - out_h)
    left_max = min(left_max, W - out_w)

    # If bounds are degenerate (should be rare if padded beforehand)
    if top_max < top_min or left_max < left_min:
        # fall back to a crop that fits
        top = min(max(cy - out_h // 2, 0), max(H - out_h, 0))
        left = min(max(cx - out_w // 2, 0), max(W - out_w, 0))
        return int(top), int(left), out_h, out_w

    top = random.randint(top_min, top_max)
    left = random.randint(left_min, left_max)
    return int(top), int(left), out_h, out_w

def binary_mask_for_sample(img, annotations, precomputed=None, *, debug=False, stats=None,
                           with_outline=False):
    """One sample's binary target: a supplied mask verbatim, else the COCO polygons.

    COCO rows carry instance annotations and the mask is the union of their
    polygons. BDD rows carry a prepared binary mask already (the seg label turned
    into objects vs background), so a transform handed ``binary_mask`` must use it
    rather than look for annotations that are not there. ``precomputed=None`` --
    every COCO row -- is the unchanged COCO path.

    The supplied mask is thresholded at >0 so an 8-bit 0/255 PNG and a 0/1 array
    both arrive as the same 0/1 target, and its shape must match the image: a
    mask built for another image is a data error, not a reason to resize it.

    ``with_outline=True`` returns ``(visible, outline)``; a prepared BDD mask has
    no amodal label at all, so its outline is all zero. That is not an *unknown*
    label (``_outline_flags`` reports ``has_full_outline=False`` for such a row
    and counts nothing as unknown), so the zero is never trained on.
    """
    if precomputed is not None:
        arr = np.asarray(precomputed)
        if arr.shape[:2] != (img.height, img.width):
            raise ValueError(
                f"binary mask is {arr.shape[:2]} but the image is "
                f"{(img.height, img.width)}"
            )
        if stats is not None:
            stats.setdefault("dropped_regions", 0)
        visible = (arr > 0).astype(np.uint8)
        if not with_outline:
            return visible
        return visible.astype(np.int64), np.zeros(visible.shape, dtype=np.int64)
    return build_binary_mask_coco_all(
        {"image": img, "annotations": annotations}, debug=debug, stats=stats,
        with_outline=with_outline,
    )


def coco_train_transforms(batch, *, input_size: int = DEFAULT_INPUT_SIZE,
                          with_outline: bool = False):
    """
    HuggingFace `set_transform` / `map` transform for COCO-style samples.

    IMPORTANT:
    - Must work when HF calls it with a SINGLE example dict
      (batch["image"] is a PIL.Image)
    - Must also work when HF calls it with a BATCH dict-of-lists
      (batch["image"] is a list of PIL.Image)

    Returns ONLY python lists (length n) for each key so HF batched indexing
    doesn't crash with "bool is not subscriptable".

    ``with_outline=True`` also returns ``outline_mask`` (the full label, moved
    by exactly the same crop, flip and resize as ``segmentation_mask``),
    ``has_full_outline`` and ``outline_unknown_images``. With it off no key is
    added and every output is what it was.
    """

    # ---- helpers ----
    def _ensure_list(x, n=None):
        if isinstance(x, list):
            return x
        if x is None and n is not None:
            return [None] * n
        return [x]

    def _to_pil_rgb(x):
        # Most of the time, after cast_column("image", Image(decode=True)), x is already PIL.
        if isinstance(x, Image.Image):
            return x.convert("RGB")
        # Defensive fallbacks:
        if isinstance(x, dict) and "path" in x:
            return Image.open(x["path"]).convert("RGB")
        if isinstance(x, dict) and "bytes" in x:
            from io import BytesIO
            return Image.open(BytesIO(x["bytes"])).convert("RGB")
        # last resort
        return Image.fromarray(np.array(x)).convert("RGB")

    # ---- you can tune these ----
    # The one setting: the model input size. The shorter-side resize keeps the
    # established 256/192 ratio so 352 gives ~469.
    target_size = int(input_size)
    resize_short = _resize_short_for(target_size)

    normalize = T.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )

    # ---- unpack batch safely ----
    images = _ensure_list(batch.get("image", None))
    n = len(images)

    annotations = _ensure_list(batch.get("annotations", None), n=n)
    # Prepared rows (BDD) hand the transform the target directly; COCO rows
    # have no such field and are built from annotations as before.
    precomputed_masks = _ensure_list(batch.get("binary_mask", None), n=n)
    #widths  = _ensure_list(batch.get("width", None), n=n)

    pixel_values_list = []
    segmentation_masks = []
    outline_masks = []      # parallel to the masks when with_outline
    has_full_list = []
    unknown_list = []
    valid_list = []
    dropped_list = []  # COCOA regions left out of each sample's target

    for i in range(n):
        has_full, unknown = False, 0
        try:
            img = _to_pil_rgb(images[i])
            ann = annotations[i]

            # Build binary mask from ann. You already have this helper in your codebase.
            # It should return a HxW np.uint8 array of {0,1}.
            # The union of ALL instances: the training target must be the same
            # question validation asks, or the model is scored on objects that
            # were not in its target. (The old K-sampled builder was deleted.)
            _mask_stats = {}
            if with_outline:
                binary_mask, outline_np = binary_mask_for_sample(
                    img, ann, precomputed_masks[i], debug=False, stats=_mask_stats,
                    with_outline=True,
                )
                has_full, unknown = _outline_flags(True, ann, _mask_stats)
            else:
                binary_mask = binary_mask_for_sample(
                    img, ann, precomputed_masks[i], debug=False, stats=_mask_stats
                )
            m = binary_mask  # np array 0/1
            _audit_mask(m, "input mask", debug=False)
            if m is None:
                raise RuntimeError("binary_mask is None")

            fg = int(m.sum())
            if fg == 0:
                # dump minimal info to see WHY
                anns = ann if isinstance(ann, list) else []
                seg_types = []
                for a in (anns[:5] if anns else []):
                    s = a.get("segmentation", None)
                    seg_types.append(type(s).__name__)
                print(f"[COCO MASK DEBUG] fg=0 image_id={batch.get('image_id')} "
                    f"n_anns={len(anns)} seg_types={seg_types} "
                    f"iscrowd={[a.get('iscrowd') for a in (anns[:5] if anns else [])]}")

            if binary_mask is None:
                binary_mask = np.zeros((img.height, img.width), dtype=np.uint8)

            mask = Image.fromarray((binary_mask * 255).astype(np.uint8), mode="L")
            if with_outline:
                # Same dtype and size as the visible mask, so every geometry op
                # below can be applied to both with the same parameters.
                outline_pil = Image.fromarray(
                    ((outline_np > 0) * 255).astype(np.uint8), mode="L")

            # ---- augmentation (train) ----
            # 1) resize (short side)
            img  = TF.resize(img,  resize_short, interpolation=InterpolationMode.BILINEAR)
            mask = TF.resize(mask, resize_short, interpolation=InterpolationMode.NEAREST)
            if with_outline:
                outline_pil = TF.resize(
                    outline_pil, resize_short, interpolation=InterpolationMode.NEAREST)

            # 2) random crop
            # If image smaller than target after resize (rare), pad first.
            if img.width < target_size or img.height < target_size:
                pad_w = max(0, target_size - img.width)
                pad_h = max(0, target_size - img.height)
                # pad right/bottom
                img  = TF.pad(img,  [0, 0, pad_w, pad_h], fill=0)
                mask = TF.pad(mask, [0, 0, pad_w, pad_h], fill=0)
                if with_outline:
                    outline_pil = TF.pad(outline_pil, [0, 0, pad_w, pad_h], fill=0)

            i_top, j_left, h, w = T.RandomCrop.get_params(img, (target_size, target_size))
            img  = TF.crop(img,  i_top, j_left, h, w)
            mask = TF.crop(mask, i_top, j_left, h, w)
            if with_outline:
                outline_pil = TF.crop(outline_pil, i_top, j_left, h, w)

            # 3) random horizontal flip
            if random.random() < 0.5:
                img  = TF.hflip(img)
                mask = TF.hflip(mask)
                if with_outline:
                    outline_pil = TF.hflip(outline_pil)

            # ---- tensorize ----
            pv = normalize(TF.to_tensor(img))  # (3, H, W)
            sm = torch.from_numpy((np.array(mask) > 0).astype(np.int64))  # (H, W) int64

            pixel_values_list.append(pv)
            segmentation_masks.append(sm)
            if with_outline:
                outline_masks.append(
                    torch.from_numpy((np.array(outline_pil) > 0).astype(np.int64)))
                has_full_list.append(bool(has_full))
                unknown_list.append(int(unknown))
            valid_list.append(True)
            dropped_list.append(int(_mask_stats.get("dropped_regions", 0)))

        except Exception: # pylint: disable=broad-exception-caught
            # Keep shapes consistent so collate can stack
            pixel_values_list.append(torch.zeros(3, target_size, target_size, dtype=torch.float32))
            segmentation_masks.append(torch.zeros(target_size, target_size, dtype=torch.int64))
            if with_outline:
                outline_masks.append(torch.zeros(target_size, target_size, dtype=torch.int64))
                has_full_list.append(False)
                unknown_list.append(0)
            valid_list.append(False)
            dropped_list.append(0)

    # Read once, before the return: this used to be assigned inside the loop
    # after the appends, so a batch whose every sample failed returned through
    # the except arm with the name never bound and raised UnboundLocalError
    # instead of handing back the failed batch it had just built.
    image_ids = _ensure_list(batch.get("image_id", None), n=n)

    # CRITICAL: every returned value is a LIST of length n
    out = {
        "pixel_values": pixel_values_list,
        "segmentation_mask": segmentation_masks,
        "valid": valid_list,
        "dropped_regions": dropped_list,
        "image_id": [int(x) if x is not None else -1 for x in image_ids],
    }
    if with_outline:
        # Only a batch that asked gets the second target: HF turns every
        # returned key into a column, and the plain path must not grow one.
        out["outline_mask"] = outline_masks
        out["has_full_outline"] = has_full_list
        out["outline_unknown_images"] = unknown_list
    return out

_SAMPLES = {"total": 0, "all_bg": 0}

def sanity_check_mask(mask_np, name="mask"):
    # mask_np: HxW {0,1}

    h, w = mask_np.shape
    frac = mask_np.mean()
    ys, xs = np.where(mask_np > 0)
    if len(xs) == 0:
        print(name, "EMPTY")
        return

    x1, x2 = xs.min(), xs.max()
    y1, y2 = ys.min(), ys.max()
    box_area = (x2 - x1 + 1) * (y2 - y1 + 1)
    print(
        f"{name}: fg_frac={frac:.3f} bbox=({x1},{y1})-({x2},{y2}) "
        f"bbox_area_frac={box_area/(h*w):.3f}"
    )

    # simple heuristic: if fg_frac is huge AND bbox covers almost whole image -> suspicious
    row_cov = (mask_np > 0).any(axis=1).mean()  # fraction of rows with any FG
    col_cov = (mask_np > 0).any(axis=0).mean()  # fraction of cols with any FG
    print(f"{name}: row_cov={row_cov:.3f} col_cov={col_cov:.3f}")

    if frac > 0.85:
        warn = "likely inverted (too much FG)"
    elif frac > 0.40 and row_cov > 0.98 and col_cov > 0.98:
        warn = "FG touches almost all rows/cols (union of many instances or speckle)"
    else:
        warn = None

    if warn is not None:
        print(f"*** WARNING: {name} mask looks suspicious: {warn} ***")

def _audit_mask(arr, tag, debug=False):
    if not debug or arr is None:
        return
    arr = np.asarray(arr)
    u = np.unique(arr)
    u_head = u[:20]
    print(f"[MASK AUDIT] {tag}: shape={arr.shape} dtype={arr.dtype} "
          f"min={arr.min()} max={arr.max()} uniq_head={u_head} uniq_n={len(u)} "
          f"mean={arr.mean():.4f} frac>0={float((arr>0).mean()):.4f} frac>127={float((arr>127).mean()):.4f}")

def _outline_flags(with_outline, ann, stats):
    """``(has_full_outline, outline_unknown_images)`` for one sample.

    Only a COCOA row (the builder's case-A test, a dict with a ``regions``
    list) can carry a full-outline label, and its label is complete unless the
    builder met a kept region with no usable full polygon. A COCO or prepared
    (BDD) row has no amodal label at all, so ``has_full_outline`` is False --
    but nothing is *unknown* about it, so the count stays 0.
    """
    if not with_outline:
        return False, 0
    unknown = int(stats.get("outline_unknown_images", 0) or 0)
    cocoa = isinstance(ann, dict) and isinstance(ann.get("regions", None), list)
    return (cocoa and unknown == 0), unknown


def _as_bool(x) -> bool:
    # robust parse for COCOA-style fields that may be bool/int/str
    if isinstance(x, bool):
        return x
    if x is None:
        return False
    if isinstance(x, (int, float)):
        return int(x) != 0
    if isinstance(x, str):
        s = x.strip().lower()
        return s in ("1", "true", "t", "yes", "y")
    return False


def cocoa_region_is_stuff(r: dict) -> bool:
    if not isinstance(r, dict):
        return True

    # normalize name (str or list)
    nm = r.get("name", "")
    if isinstance(nm, (list, tuple)):
        names = [str(x).lower() for x in nm]
    else:
        names = [str(nm).lower()]

    # hard skip background region(s)
    if any("background" == x or "background" in x for x in names):
        return True

    # skip "stuff" only if it is truly stuff (not the string "False")
    if _as_bool(r.get("is_stuff", False)):
        return True

    return False

def coco_train_transforms_fg_centered(batch, *, input_size: int = DEFAULT_INPUT_SIZE, debug=False,
                                     scale_jitter: tuple[float, float] | None = None,
                                     color_jitter: float | None = None,
                                     with_outline: bool = False):
    """
    Same behavior as your current transform, but the crop is FG-centered whenever possible:
      - Build binary mask (the union of ALL instances -- see below)
      - Resize (short side)
      - Pad to at least target_size
      - Crop window chosen to include >=1 FG pixel (if any exist), else random crop
      - Random hflip (paired)
      - Tensorize + normalize

    ``with_outline=True`` also carries the full-outline target through the same
    resize, pad, crop and flip (the crop window is the visible mask's), and
    returns ``outline_mask``, ``has_full_outline`` and
    ``outline_unknown_images``. With it off nothing changes.

    The anchor instance the crop centres on is chosen by looking at a foreground
    pixel of the union mask; it changes the crop window only, never the mask.

    Two augmentation steps are opt-in and off by default, so a run that does not
    ask for them produces the batches it always did:
      - scale_jitter=(LO, HI): the short-side resize is drawn per image from
        [LO * resize_short, HI * resize_short], so one batch sees a range of
        scales. The mask is resized by the same factor, NEAREST. Drawn here,
        before the pad and the crop, so the configured output size and the
        foreground-centred crop are exactly what they were.
      - color_jitter=S: brightness/contrast/saturation +- S, hue +- S/4, image
        only. See _apply_color_jitter for why it is not ColorJitter.
    """

    target_size = int(input_size)
    resize_short = _resize_short_for(target_size)

    normalize = T.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )

    def _ensure_list(x, n=None):
        if x is None:
            return [None] * (n or 0)
        if isinstance(x, (list, tuple)):
            if n is None:
                return list(x)
            if len(x) == n:
                return list(x)
            # broadcast scalar-ish or truncate/pad
            if len(x) == 1:
                return list(x) * n
            return list(x)[:n] + [None] * max(0, n - len(x))
        return [x] * (n or 1)

    def _to_pil_rgb(img):
        if isinstance(img, Image.Image):
            return img.convert("RGB")
        # Try numpy/torch
        if isinstance(img, np.ndarray):
            if img.ndim == 3 and img.shape[2] in (1, 3, 4):
                return Image.fromarray(img[..., :3]).convert("RGB")
            return Image.fromarray(img).convert("RGB")
        if torch.is_tensor(img):
            arr = img.detach().cpu()
            if arr.ndim == 3 and arr.shape[0] in (1, 3):
                arr = arr.permute(1, 2, 0).numpy()
            else:
                arr = arr.numpy()
            arr = np.clip(arr, 0, 255).astype(np.uint8)
            return Image.fromarray(arr[..., :3]).convert("RGB")
        raise TypeError(f"Unsupported image type: {type(img)}")

    # ---- unpack batch safely ----
    images = _ensure_list(batch.get("image", None))
    n = len(images)

    annotations = _ensure_list(batch.get("annotations", None), n=n)
    image_ids   = _ensure_list(batch.get("image_id", None), n=n)
    # Prepared rows (BDD) carry the target themselves; COCO rows do not.
    precomputed_masks = _ensure_list(batch.get("binary_mask", None), n=n)

    pixel_values_list = []
    segmentation_masks = []
    outline_masks = []      # parallel to the masks when with_outline
    has_full_list = []
    unknown_list = []
    valid_list = []
    dropped_list = []  # COCOA regions left out of each sample's target

    for i in range(n):
        has_full, unknown = False, 0
        try:
            img = _to_pil_rgb(images[i])
            ann = annotations[i]

            # Build binary mask from ann (HxW np.uint8 0/1).
            # The union of ALL instances. The crop is centred on foreground by
            # _fg_centered_crop_params below, which reads this mask but does not
            # narrow it: the anchor picks the window, the target stays the union.
            _mask_stats = {}
            if with_outline:
                binary_mask, outline_np = binary_mask_for_sample(
                    img, ann, precomputed_masks[i], debug=debug, stats=_mask_stats,
                    with_outline=True,
                )
                has_full, unknown = _outline_flags(True, ann, _mask_stats)
            else:
                binary_mask = binary_mask_for_sample(
                    img, ann, precomputed_masks[i], debug=debug, stats=_mask_stats
                )

            if binary_mask is None:
                binary_mask = np.zeros((img.height, img.width), dtype=np.uint8)

            mask = Image.fromarray((binary_mask * 255).astype(np.uint8), mode="L")
            if with_outline:
                outline_pil = Image.fromarray(
                    ((outline_np > 0) * 255).astype(np.uint8), mode="L")
            if debug:
                sanity_check_mask(mask_np=binary_mask, name=f"input mask image_id={image_ids[i]}")
                _audit_mask(arr=binary_mask, tag=f"input mask image_id={image_ids[i]}", debug=debug)

            # ---- augmentation (train) ----
            # 1) resize (short side)
            # IMPORTANT: resize_short should be int for "short side" semantics
            resize_here = resize_short
            if scale_jitter is not None:
                lo, hi = scale_jitter
                # Per image, not per batch: the point of the option is a range
                # of scales inside one batch. Drawn from `random` so it shares
                # the stream with the crop and the flip (see _apply_color_jitter).
                resize_here = max(1, int(round(resize_short * random.uniform(lo, hi))))
            img  = TF.resize(img,  resize_here, interpolation=InterpolationMode.BILINEAR)
            mask = TF.resize(mask, resize_here, interpolation=InterpolationMode.NEAREST)
            if with_outline:
                outline_pil = TF.resize(
                    outline_pil, resize_here, interpolation=InterpolationMode.NEAREST)
            # 1b) colour jitter (image only; the mask never enters this call)
            if color_jitter:
                img = _apply_color_jitter(img, float(color_jitter))

            if debug:
                sanity_check_mask(mask_np=np.array(mask) > 0, name=f"resized mask image_id={image_ids[i]}")
            # 2) pad to ensure >= target_size
            if img.width < target_size or img.height < target_size:
                pad_w = max(0, target_size - img.width)
                pad_h = max(0, target_size - img.height)
                img  = TF.pad(img,  [0, 0, pad_w, pad_h], fill=0)
                mask = TF.pad(mask, [0, 0, pad_w, pad_h], fill=0)
                if with_outline:
                    outline_pil = TF.pad(outline_pil, [0, 0, pad_w, pad_h], fill=0)

            # 3) FG-centered crop (paired) or random crop (paired)
            if (np.array(mask) > 0).any() and random.random() < 0.8:
                i_top, j_left, h, w = _fg_centered_crop_params(mask, target_size, target_size)
            else:
                i_top, j_left, h, w = T.RandomCrop.get_params(img, (target_size, target_size))
            img  = TF.crop(img,  i_top, j_left, h, w)
            mask = TF.crop(mask, i_top, j_left, h, w)
            if with_outline:
                outline_pil = TF.crop(outline_pil, i_top, j_left, h, w)

            if debug:
                sanity_check_mask(mask_np=np.array(mask) > 0, name=f"post-crop mask image_id={image_ids[i]}")

            # 4) random horizontal flip (paired)
            if random.random() < 0.5:
                img  = TF.hflip(img)
                mask = TF.hflip(mask)
                if with_outline:
                    outline_pil = TF.hflip(outline_pil)

            if debug:
                sanity_check_mask(mask_np=np.array(mask) > 0, name=f"post-flip mask image_id={image_ids[i]}")

            # ---- tensorize ----
            pv = normalize(TF.to_tensor(img))  # (3,H,W)
            sm = torch.from_numpy((np.array(mask) > 0).astype(np.int64))  # (H,W)

            if debug:
                _audit_mask(sm.numpy(), tag=f"post-crop mask image_id={image_ids[i]}", debug=debug)

            if debug and i < 2:
                fg_frac = float(sm.float().mean().item())
                print(f"[FG-CROP DEBUG] image_id={image_ids[i]} fg_frac_after_crop={fg_frac:.4f}")

            if debug:
                gt_fg_frac = (sm == 1).float().mean().item()
                _SAMPLES["total"] += 1

                if gt_fg_frac == 0.0:
                    _SAMPLES["all_bg"] += 1
                
                all_bg_fraction = _SAMPLES["all_bg"] / _SAMPLES["total"]
                print(f"[FG-CROP STATS] total_samples={_SAMPLES['total']} all_bg_samples={_SAMPLES['all_bg']} all_bg_fraction={all_bg_fraction:.4f}")

            pixel_values_list.append(pv)
            segmentation_masks.append(sm)
            if with_outline:
                outline_masks.append(
                    torch.from_numpy((np.array(outline_pil) > 0).astype(np.int64)))
                has_full_list.append(bool(has_full))
                unknown_list.append(int(unknown))
            valid_list.append(True)
            dropped_list.append(int(_mask_stats.get("dropped_regions", 0)))

        except Exception as e: # pylint: disable=broad-exception-caught
            print(f"[FG-CROP ERROR] idx={i} image_id={image_ids[i]} err={repr(e)}")
            print(traceback.format_exc())
            pixel_values_list.append(torch.zeros(3, target_size, target_size, dtype=torch.float32))
            segmentation_masks.append(torch.zeros(target_size, target_size, dtype=torch.int64))
            if with_outline:
                outline_masks.append(torch.zeros(target_size, target_size, dtype=torch.int64))
                has_full_list.append(False)
                unknown_list.append(0)
            valid_list.append(False)
            dropped_list.append(0)

    out = {
        "pixel_values": pixel_values_list,
        "segmentation_mask": segmentation_masks,
        "valid": valid_list,
        "dropped_regions": dropped_list,
        "image_id": [int(x) if x is not None else -1 for x in image_ids],
    }
    if with_outline:
        out["outline_mask"] = outline_masks
        out["has_full_outline"] = has_full_list
        out["outline_unknown_images"] = unknown_list
    return out




def coco_val_transforms(batch, *, input_size: int = DEFAULT_INPUT_SIZE,
                        with_outline: bool = False):
    """Deterministic val transform; ``with_outline`` adds the full-outline
    target and its two flags, moved by the same resize and centre crop."""
    target_size = int(input_size)
    resize_short = _resize_short_for(target_size)

    normalize = T.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )

    images = batch.get("image", None)
    images = _ensure_list(images)
    n = len(images)

    annos = batch.get("annotations", None)
    annos = _ensure_list(annos, n=n)  # if None -> [None]*n
    # Prepared rows (BDD) carry the target themselves; COCO rows do not.
    precomputed_masks = _ensure_list(batch.get("binary_mask", None), n=n)

    pixel_values_list = []
    segmentation_masks = []
    outline_masks = []      # parallel to the masks when with_outline
    has_full_list = []
    unknown_list = []
    valid_list = []  # IMPORTANT: must be list for batched path
    dropped_list = []  # COCOA regions left out of each sample's target

    for i in range(n):
        has_full, unknown = False, 0
        try:
            img = images[i]
            ann = annos[i]

            if img is None:
                raise ValueError("image is None")

            # ensure PIL RGB
            if not isinstance(img, Image.Image):
                # handle HF Image dicts/bytes/paths if you still need it
                if isinstance(img, dict) and "bytes" in img:
                    img = convert_bytes_to_image(img["bytes"])
                elif isinstance(img, dict) and "path" in img:
                    img = Image.open(img["path"])
                else:
                    img = convert_bytes_to_image(img)

            img = img.convert("RGB")

            # build mask
            _mask_stats = {}
            if with_outline:
                binary_mask, outline_np = binary_mask_for_sample(
                    img, ann, precomputed_masks[i], debug=False, stats=_mask_stats,
                    with_outline=True,
                )
                has_full, unknown = _outline_flags(True, ann, _mask_stats)
            else:
                binary_mask = binary_mask_for_sample(
                    img, ann, precomputed_masks[i], debug=False, stats=_mask_stats
                )
            if binary_mask is None:
                binary_mask = np.zeros((img.height, img.width), dtype=np.uint8)

            mask = Image.fromarray((binary_mask * 255).astype(np.uint8), mode="L")
            if with_outline:
                outline_pil = Image.fromarray(
                    ((outline_np > 0) * 255).astype(np.uint8), mode="L")

            # deterministic resize/crop
            img  = TF.resize(img, resize_short, interpolation=InterpolationMode.BILINEAR)
            mask = TF.resize(mask, resize_short, interpolation=InterpolationMode.NEAREST)
            if with_outline:
                outline_pil = TF.resize(
                    outline_pil, resize_short, interpolation=InterpolationMode.NEAREST)
            img  = TF.center_crop(img, (target_size, target_size))
            mask = TF.center_crop(mask, (target_size, target_size))
            if with_outline:
                outline_pil = TF.center_crop(outline_pil, (target_size, target_size))

            pv = normalize(TF.to_tensor(img))
            sm = torch.from_numpy((np.array(mask) > 0).astype(np.int64))

            pixel_values_list.append(pv)
            segmentation_masks.append(sm)
            if with_outline:
                outline_masks.append(
                    torch.from_numpy((np.array(outline_pil) > 0).astype(np.int64)))
                has_full_list.append(bool(has_full))
                unknown_list.append(int(unknown))
            valid_list.append(True)
            dropped_list.append(int(_mask_stats.get("dropped_regions", 0)))


        except Exception: # pylint: disable=broad-exception-caught
            pixel_values_list.append(torch.zeros(3, target_size, target_size))
            segmentation_masks.append(torch.zeros(target_size, target_size, dtype=torch.int64))
            if with_outline:
                outline_masks.append(torch.zeros(target_size, target_size, dtype=torch.int64))
                has_full_list.append(False)
                unknown_list.append(0)
            valid_list.append(False)
            dropped_list.append(0)

    # Read once, before the loop: assigning it inside the loop left it unbound
    # for a batch whose every sample failed, so the return below raised
    # UnboundLocalError instead of handing back the failed batch it had built.
    image_ids = _ensure_list(batch.get("image_id", None), n=n)

    # CRITICAL: all values MUST be lists of length n
    out = {
        "pixel_values": pixel_values_list,
        "segmentation_mask": segmentation_masks,
        "valid": valid_list,
        "dropped_regions": dropped_list,
        "image_id": [int(x) if x is not None else -1 for x in image_ids],
    }
    if with_outline:
        out["outline_mask"] = outline_masks
        out["has_full_outline"] = has_full_list
        out["outline_unknown_images"] = unknown_list
    return out





def _outline_tensor(sample):
    """The sample's full-outline target as an [H,W] int64 tensor.

    A per-example dict holds the tensor directly, or the length-1 list the
    batch transforms produce (``Dataset.with_transform`` feeds one sample as a
    batch of one). A target that is missing or None means the loader asked for
    outlines on a row that has none -- a loader bug, and one that must not reach
    the loss as a silently zeroed target.
    """
    x = sample.get("outline_mask", None)
    if isinstance(x, (list, tuple)):
        if len(x) != 1:
            raise ValueError(
                f"expected one outline target per sample, got {len(x)}")
        x = x[0]
    if x is None:
        raise ValueError(
            "a sample has no outline target; build the loader with "
            "with_outline=True only when every row carries one")
    if not torch.is_tensor(x):
        x = torch.as_tensor(np.asarray(x))
    return x.to(dtype=torch.int64)


def coco_collate_fn(batch, *, input_size: int = DEFAULT_INPUT_SIZE):
    target_size = int(input_size)
    # Did the loader ask for outlines? Read off the batch before the validity
    # filter, so an all-dropped batch still returns the three keys its loss
    # step expects instead of a KeyError.
    asked_outline = any(
        isinstance(b, dict) and "outline_mask" in b for b in batch if b is not None
    )
    good = []
    for b in batch:
        if b is None:
            continue
        v = b.get("valid", True)
        if isinstance(v, list):
            v = all(v)
        if not v:
            continue
        if "pixel_values" not in b or "segmentation_mask" not in b:
            continue
        good.append(b)

    if len(good) == 0:
        empty = {
            "pixel_values": torch.zeros(1, 3, target_size, target_size, dtype=torch.float32),
            "segmentation_mask": torch.zeros(1, target_size, target_size, dtype=torch.int64),
            "image_id": torch.tensor([-1], dtype=torch.int64),
            "valid": torch.tensor([False]),
            "dropped_regions": torch.tensor(0, dtype=torch.int64),
        }
        if asked_outline:
            empty["outline_mask"] = torch.zeros(
                1, target_size, target_size, dtype=torch.int64)
            empty["has_full_outline"] = torch.zeros(1, dtype=torch.bool)
            empty["outline_unknown_images"] = torch.tensor(0, dtype=torch.int64)
        return empty

    # image_id may be python int or tensor depending on transform path
    def _to_id(x):
        x = x.get("image_id", -1)
        if isinstance(x, torch.Tensor):
            return x.to(dtype=torch.int64).view(())
        return torch.tensor(int(x), dtype=torch.int64)

    return {
        "pixel_values": torch.stack([b["pixel_values"] for b in good], dim=0),
        "segmentation_mask": torch.stack([b["segmentation_mask"] for b in good], dim=0),
        # Only when the loader asked for it: this is what the training step
        # keys on to compute the outline loss, so an unasked batch must not
        # carry the key at all. A dropped image drops all three with it.
        **({"outline_mask": torch.stack(
                [_outline_tensor(b) for b in good], dim=0),
            "has_full_outline": torch.tensor(
                [bool(b.get("has_full_outline", False)) for b in good],
                dtype=torch.bool),
            "outline_unknown_images": torch.tensor(
                sum(int(b.get("outline_unknown_images", 0) or 0) for b in good),
                dtype=torch.int64)}
           if asked_outline else {}),
        "image_id": torch.stack([_to_id(b) for b in good], dim=0),
        "valid": torch.ones(len(good), dtype=torch.bool),
        # COCOA regions the mask builder left out of these samples' targets,
        # summed over the batch so the trainer can log it instead of losing it.
        "dropped_regions": torch.tensor(
            sum(int(b.get("dropped_regions", 0) or 0) for b in good), dtype=torch.int64
        ),
    }


def coco_train_transform_example(example: dict, *, input_size: int = DEFAULT_INPUT_SIZE,
                                 with_outline: bool = False) -> dict:
    """The streaming (per-example) train transform. ``with_outline`` adds
    ``outline_mask``, ``has_full_outline`` and ``outline_unknown_images``,
    moved by the same crop and flip as ``segmentation_mask``."""
    target_size = int(input_size)

    normalize = T.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225])

    try:
        img = example.get("image", None)
        ann = example.get("annotations", None)

        # ---- decode image ----
        if not isinstance(img, Image.Image):
            if isinstance(img, dict) and "bytes" in img:
                img = convert_bytes_to_image(img["bytes"])
            elif isinstance(img, dict) and "path" in img:
                img = convert_bytes_to_image(img["path"])
            else:
                img = convert_bytes_to_image(img)

        if img is None:
            raise ValueError("image decode returned None")

        img = img.convert("RGB")

        # ---- build binary mask at original resolution ----
        # The union of ALL instances; never a sampled subset.
        _mask_stats = {}
        # A prepared row (BDD) already carries its binary target; a COCO row is
        # built from polygons here.
        if with_outline:
            binary_mask, outline_np = binary_mask_for_sample(
                img, ann, example.get("binary_mask", None), debug=False,
                stats=_mask_stats, with_outline=True,
            )
            has_full, unknown = _outline_flags(True, ann, _mask_stats)
        else:
            binary_mask = binary_mask_for_sample(
                img, ann, example.get("binary_mask", None), debug=False, stats=_mask_stats
            )
        if binary_mask is None:
            binary_mask = np.zeros((img.height, img.width), dtype=np.uint8)

        mask = Image.fromarray((binary_mask * 255).astype(np.uint8), mode="L")
        if with_outline:
            outline_pil = Image.fromarray(
                ((outline_np > 0) * 255).astype(np.uint8), mode="L")

        # ---- random resized crop (same params for img + mask) ----
        top, left, h, w = T.RandomResizedCrop.get_params(
            img, scale=(0.7, 1.0), ratio=(3/4, 4/3)
        )
        img  = TF.resized_crop(img,  top, left, h, w, (target_size, target_size), InterpolationMode.BILINEAR)
        mask = TF.resized_crop(mask, top, left, h, w, (target_size, target_size), InterpolationMode.NEAREST)
        if with_outline:
            outline_pil = TF.resized_crop(
                outline_pil, top, left, h, w, (target_size, target_size),
                InterpolationMode.NEAREST)

        # ---- random flip ----
        if random.random() < 0.5:
            img = TF.hflip(img)
            mask = TF.hflip(mask)
            if with_outline:
                outline_pil = TF.hflip(outline_pil)

        pixel_values = normalize(TF.to_tensor(img))
        segmentation_mask = torch.from_numpy((np.array(mask) > 0).astype(np.int64))  # [target_size, target_size]

        image_id = example.get("image_id", -1)
        out = {
            "pixel_values": pixel_values,
            "segmentation_mask": segmentation_mask,
            "image_id": torch.tensor(int(image_id), dtype=torch.int64),
            "valid": True,
            "dropped_regions": int(_mask_stats.get("dropped_regions", 0)),
        }
        if with_outline:
            out["outline_mask"] = torch.from_numpy(
                (np.array(outline_pil) > 0).astype(np.int64))
            out["has_full_outline"] = bool(has_full)
            out["outline_unknown_images"] = int(unknown)
        return out

    except Exception: # pylint: disable=broad-exception-caught
        image_id = example.get("image_id", -1)
        out = {
            "pixel_values": torch.zeros(3, target_size, target_size, dtype=torch.float32),
            "segmentation_mask": torch.zeros(target_size, target_size, dtype=torch.int64),
            "image_id": torch.tensor(int(image_id), dtype=torch.int64),
            "valid": False,
            "dropped_regions": 0,
        }
        if with_outline:
            out["outline_mask"] = torch.zeros(
                target_size, target_size, dtype=torch.int64)
            out["has_full_outline"] = False
            out["outline_unknown_images"] = 0
        return out

def coco_val_transform_example(example: dict, *, input_size: int = DEFAULT_INPUT_SIZE,
                               with_outline: bool = False) -> dict:
    """The streaming (per-example) val transform; ``with_outline`` as for
    ``coco_val_transforms``."""
    target_size = int(input_size)
    resize_short = _resize_short_for(target_size)

    normalize = T.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225])
    try:
        img = example.get("image", None)
        ann = example.get("annotations", None)

        if not isinstance(img, Image.Image):
            if isinstance(img, dict) and "bytes" in img:
                img = convert_bytes_to_image(img["bytes"])
            elif isinstance(img, dict) and "path" in img:
                img = convert_bytes_to_image(img["path"])
            else:
                img = convert_bytes_to_image(img)

        if img is None:
            raise ValueError("image decode returned None")

        img = img.convert("RGB")

        # The union of ALL instances; never a sampled subset.
        _mask_stats = {}
        # A prepared row (BDD) already carries its binary target; a COCO row is
        # built from polygons here.
        if with_outline:
            binary_mask, outline_np = binary_mask_for_sample(
                img, ann, example.get("binary_mask", None), debug=False,
                stats=_mask_stats, with_outline=True,
            )
            has_full, unknown = _outline_flags(True, ann, _mask_stats)
        else:
            binary_mask = binary_mask_for_sample(
                img, ann, example.get("binary_mask", None), debug=False, stats=_mask_stats
            )
        if binary_mask is None:
            binary_mask = np.zeros((img.height, img.width), dtype=np.uint8)

        mask = Image.fromarray((binary_mask * 255).astype(np.uint8), mode="L")
        if with_outline:
            outline_pil = Image.fromarray(
                ((outline_np > 0) * 255).astype(np.uint8), mode="L")

        img  = TF.resize(img, resize_short, interpolation=InterpolationMode.BILINEAR)
        mask = TF.resize(mask, resize_short, interpolation=InterpolationMode.NEAREST)
        if with_outline:
            outline_pil = TF.resize(
                outline_pil, resize_short, interpolation=InterpolationMode.NEAREST)

        img  = TF.center_crop(img, (target_size, target_size))
        mask = TF.center_crop(mask, (target_size, target_size))
        if with_outline:
            outline_pil = TF.center_crop(outline_pil, (target_size, target_size))

        pixel_values = normalize(TF.to_tensor(img))
        segmentation_mask = torch.from_numpy((np.array(mask) > 0).astype(np.int64))

        image_id = example.get("image_id", -1)
        out = {
            "pixel_values": pixel_values,
            "segmentation_mask": segmentation_mask,
            "image_id": torch.tensor(int(image_id), dtype=torch.int64),
            "valid": True,
            "dropped_regions": int(_mask_stats.get("dropped_regions", 0)),
        }
        if with_outline:
            out["outline_mask"] = torch.from_numpy(
                (np.array(outline_pil) > 0).astype(np.int64))
            out["has_full_outline"] = bool(has_full)
            out["outline_unknown_images"] = int(unknown)
        return out

    except Exception: # pylint: disable=broad-exception-caught
        image_id = example.get("image_id", -1)
        out = {
            "pixel_values": torch.zeros(3, target_size, target_size, dtype=torch.float32),
            "segmentation_mask": torch.zeros(target_size, target_size, dtype=torch.int64),
            "image_id": torch.tensor(int(image_id), dtype=torch.int64),
            "valid": False,
            "dropped_regions": 0,
        }
        if with_outline:
            out["outline_mask"] = torch.zeros(
                target_size, target_size, dtype=torch.int64)
            out["has_full_outline"] = False
            out["outline_unknown_images"] = 0
        return out

def ensure_image_id(ds):
    """
    Ensure dataset has an `image_id` column. Works for both map-style and IterableDataset.
    """
    cols = getattr(ds, "column_names", None)
    if cols and "image_id" in cols:
        return ds
    return ds.map(lambda ex, i: {"image_id": int(i)}, with_indices=True)


def _split_stream_by_rank(dataset, global_rank: int, world_size: int):
    """Give each rank a disjoint slice of a shuffled streaming dataset.

    `datasets.distributed.split_dataset_by_node` is the sanctioned split, and it
    is not the same thing as `.shard(num_shards=world_size, index=rank)`: `.shard`
    divides by the dataset's own `num_shards`, and HF's streaming `.shuffle()`
    collapses that to 1 (measured: 4 source files report 4 before `.shuffle` and
    1 after). A `.shard()` applied after `.shuffle()` therefore hands the whole
    stream to rank 0 and an empty one to every other rank. At a world size of 1
    the dataset is returned untouched, so the single-process stream is unchanged.
    """
    if world_size <= 1:
        return dataset
    try:
        from datasets.distributed import split_dataset_by_node
    except ImportError as exc:  # pragma: no cover - depends on the datasets build
        raise RuntimeError(
            "distributed streaming needs datasets.distributed.split_dataset_by_node, "
            "which this `datasets` build does not provide; upgrade `datasets` or run "
            "with world_size=1."
        ) from exc
    return split_dataset_by_node(dataset, rank=global_rank, world_size=world_size)


def _distributed_sampler_for(
    dataset, global_rank: int, world_size: int, shuffle: bool, seed: int
):
    """A per-rank sampler for a map-style dataset, or None at a world size of 1.

    Map-style data has an index, so the shard is the standard
    `DistributedSampler`: it partitions the index space across ranks and reshuffles
    per epoch through `set_epoch`. Returning None at a world size of 1 keeps the
    plain DataLoader(shuffle=...) path byte-identical for single-process runs.
    """
    if world_size <= 1:
        return None
    return torch.utils.data.DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=global_rank,
        shuffle=shuffle,
        seed=seed,
        drop_last=False,
    )


def _bdd_row_transform(row, *, transform, input_size: int, is_train: bool,
                       scale_jitter=None, color_jitter=None, with_outline=False):
    """One BDD row through the transform that handles that side of a run.

    The COCO transforms were written for the HuggingFace ``map`` contract: they
    take a dict of columns, handle several rows at once, and return the same
    shape (the val-side one is batched too, even though the COCO val loader
    calls it per row). A prepared dataset yields single rows, so the row becomes
    a one-element column dict and the one-element columns come back out.

    Note that the transforms turn a per-row exception into a zero mask with
    ``valid=False`` rather than raising. A mask this adapter could not build
    would therefore look like an empty image, not like an error -- which is why
    the mask shape is checked in ``binary_mask_for_sample``.

    BDD has no amodal labels, so with ``with_outline`` a row comes back with an
    all-zero ``outline_mask`` and ``has_full_outline=False``: a real "no label"
    sample, never a made-up one.
    """
    columns = {k: [row.get(k)] for k in
               ("image", "annotations", "image_id", "binary_mask")}
    if is_train:
        out = transform(
            columns, input_size=input_size,
            scale_jitter=scale_jitter, color_jitter=color_jitter,
            with_outline=with_outline,
        )
    else:
        out = transform(columns, input_size=input_size, with_outline=with_outline)
    return {k: (v[0] if isinstance(v, (list, tuple)) else v)
            for k, v in out.items()}


def _bdd_dataloader(*, split: str, bdd_data_root, batch_size: int, workers: int,
                    global_rank: int, world_size: int, shuffle: bool,
                    input_size: int, transform, shuffle_seed: int = 42,
                    scale_jitter=None, color_jitter=None,
                    with_outline: bool = False) -> DataLoader:
    """A map-style BDD100K loader.

    The prepared tree already holds images plus binary object masks, so the row
    is (image, mask) and the transform is the COCO map-style one: it sees
    ``binary_mask`` and uses it in place of the polygons it would build for a
    COCO row. Everything else -- batch shape, collate, rank shard -- matches the
    COCO map-style path so the trainer consumes either dataset unchanged.
    """
    root = prepared_root(bdd_data_root)
    dataset = BDD100kSegDataset(root, split=split)
    print(f"BDD100K {split}: {len(dataset)} rows from {root}")

    dataset.set_transform(partial(
        _bdd_row_transform,
        transform=transform,
        input_size=input_size,
        is_train=(split == "train"),
        scale_jitter=scale_jitter,
        color_jitter=color_jitter,
        with_outline=with_outline,
    ))

    sampler = _distributed_sampler_for(dataset, global_rank, world_size, shuffle, shuffle_seed)
    return torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle and sampler is None,
        sampler=sampler,
        num_workers=workers,
        collate_fn=lambda b, _s=input_size: coco_collate_fn(b, input_size=_s),
        pin_memory=True,
        drop_last=True,
        persistent_workers=False,
    )


def get_training_dataloader(
    dataset_name: str = "imagenet-1k",
    is_dataset_local: bool = False,
    batch_size: int = 32,
    workers: int = 4,
    global_rank: int = 0,
    world_size: int = 1,
    hf_token: str = os.getenv('HF_TOKEN'),
    shuffle: bool = True,
    shuffle_buffer_size: int = 10000,
    shuffle_seed: int = 42,
    input_size: int = DEFAULT_INPUT_SIZE,
    scale_jitter: tuple[float, float] | None = None,
    color_jitter: float | None = None,
    bdd_data_root: str | None = None,
    with_outline: bool = False,
) -> DataLoader:
    """
    Create training dataloader for HCM model training.

    Args:
        dataset_name: Name of dataset (HuggingFace format)
        is_dataset_local: Whether dataset is stored locally
        batch_size: Training batch size
        workers: Number of data loading workers
        global_rank: Global rank for distributed training
        world_size: World size for distributed training
        hf_token: HuggingFace token for dataset access
        shuffle: Whether to shuffle the training data. Map-style datasets get
            DataLoader(shuffle=True); streaming/IterableDatasets get
            .shuffle(buffer_size=..., seed=...).
        shuffle_buffer_size: Buffer size for streaming shuffle.
        shuffle_seed: Seed for streaming shuffle.
        input_size: Spatial size of the COCO/COCOA segmentation pipeline. The
            transforms build their tensors at this size. Defaults to 192, the
            established behaviour.
        scale_jitter: (LO, HI) range for the per-image short-side scale, or None
            (default) for the fixed resize. See coco_train_transforms_fg_centered.
        color_jitter: Colour-jitter strength, or None (default) for no colour
            augmentation. See _apply_color_jitter.

        with_outline: Build the amodal full-outline target too (COCOA rows;
            COCO and BDD rows come back unlabelled rather than guessed). Off by
            default, so an ordinary run's batches are exactly as before.

        Both augmentation options are off by default, so a run that does not ask
        for them gets the batches it always got. Only the map-style COCO/COCOA
        path can apply them; asking for either on any other path raises rather
        than quietly running without the augmentation the caller asked for.

    Returns:
        Training DataLoader
    """

    # Refuse up front rather than run without the augmentation that was asked
    # for: the local generator path and the non-COCO path have their own
    # transforms, and neither takes these options.
    if with_outline and (is_dataset_local or (
        "coco" not in dataset_name.lower() and dataset_name.lower() != BDD_DATASET_NAME
    )):
        raise ValueError(
            "with_outline is only implemented for the COCO/COCOA segmentation "
            f"pipeline; dataset_name={dataset_name!r}, "
            f"is_dataset_local={is_dataset_local} has no COCOA full-outline "
            "labels and no prepared-binary-mask rows."
        )

    if (scale_jitter is not None or color_jitter is not None) and (
        is_dataset_local or "coco" not in dataset_name.lower()
    ):
        raise ValueError(
            "scale_jitter/color_jitter are only implemented for the map-style "
            f"COCO/COCOA path; dataset_name={dataset_name!r}, "
            f"is_dataset_local={is_dataset_local} cannot apply them. "
            "Leave both unset for this dataset."
        )

    # Stays refused across ranks because it has no cross-rank test coverage and
    # no production caller (train_hcm_imagenet1k always passes
    # is_dataset_local=False), while the HuggingFace path has both. This is not
    # the duplicate-data defect it used to cite: that was GeneratorDataset
    # comparing a per-worker local index against world_size, and it is fixed
    # (the split is now rank * num_workers + worker_id over the total).
    if is_dataset_local and world_size > 1:
        raise ValueError(
            "the local GeneratorDataset path is not offered across ranks: it has "
            "no cross-rank test coverage and no production caller, unlike the "
            f"HuggingFace path. world_size={world_size}, rank={global_rank}. Use "
            "the HuggingFace path for distributed runs, or run with world_size=1."
        )

    if dataset_name.lower() == BDD_DATASET_NAME:
        return _bdd_dataloader(
            split="train",
            bdd_data_root=bdd_data_root,
            batch_size=batch_size,
            workers=workers,
            global_rank=global_rank,
            world_size=world_size,
            shuffle=shuffle,
            shuffle_seed=shuffle_seed,
            input_size=input_size,
            scale_jitter=scale_jitter,
            color_jitter=color_jitter,
            transform=coco_train_transforms_fg_centered,
            with_outline=with_outline,
        )

    if is_dataset_local:
        print(f"Using local dataset, batch size: {batch_size}")
        train_dataset = GeneratorDataset(rank=global_rank, world_size=world_size)
        train_dataset.worker_initializer(train_transforms, "train")
        
        train_loader = torch.utils.data.DataLoader(
            dataset=train_dataset,
            batch_size=1,  # Local dataset handles batching internally
            shuffle=False,  # IterableDataset: ordering is the dataset's business
            num_workers=workers,
            collate_fn=collate_fn,
            pin_memory=False,
            drop_last=True,
            worker_init_fn=lambda worker_id: train_dataset.worker_initializer(train_transforms, "train"),
            persistent_workers=False,
        )
    else:
        print(f"Loading HuggingFace dataset: {dataset_name}")
        d_cfg = DownloadConfig(resume_download=True, max_retries=500)
        
        login(token=hf_token)
        
        # Try loading without trust_remote_code first (for standard datasets)
        # If that fails, try with trust_remote_code=True (for datasets with custom code)
        try:
            if "shunk031/COCOA" in dataset_name:
                print("Loading COCO dataset with custom code")
                # Download annotations from MinIO
                annotations_path = download_coco_annotations()
                
                train_dataset = load_dataset(
                    path=dataset_name, 
                    name="COCO",
                    data_dir=annotations_path,
                    decode_rle=False,
                    split='train', 
                    streaming=False, 
                    token=hf_token,
                    download_config=d_cfg,
                )
                train_dataset = ensure_image_id(train_dataset)
            else:
                if "shunk031/MSCOCO" in dataset_name:
                    print("Loading COCO dataset with custom code")
                    train_dataset = load_dataset(
                        dataset_name, 
                        year=2017,
                        split='train', 
                        coco_task="instances",
                        revision="main",
                        decode_rle=False,
                        streaming=False, 
                        token=hf_token,
                        download_config=d_cfg,
                    )
                    train_dataset = ensure_image_id(train_dataset)
                else:
                    train_dataset = load_dataset(
                        dataset_name, 
                        split='train', 
                        streaming=True, 
                        token=hf_token,
                        download_config=d_cfg,
                    )
        except (ValueError, TypeError) as e:
            print(f"Standard load failed, trying with trust_remote_code=True: {e}")
            train_dataset = load_dataset(
                dataset_name, 
                split='train', 
                streaming=True, 
                token=hf_token,
                trust_remote_code=True,
                download_config=d_cfg,
            )
        
        logout()
        
        # Use COCO-specific transforms and collate for COCO dataset
        if "coco" in dataset_name.lower():
            print("Using COCO-specific transforms with binary mask generation")

            # Always decode images
            if "image" in train_dataset.column_names:
                train_dataset = train_dataset.cast_column("image", HFImage(decode=True))

            if isinstance(train_dataset, HFIterableDataset):
                # The per-example transform below has no seam for these options.
                if scale_jitter is not None or color_jitter is not None:
                    raise ValueError(
                        "scale_jitter/color_jitter are only implemented for the "
                        "map-style COCO/COCOA path, and this dataset loaded as a "
                        "streaming IterableDataset. Leave both unset, or load the "
                        "dataset with streaming=False."
                    )
                # ✅ per-example transform for streaming datasets. The input size
                # is bound here: .map(fn, fn_kwargs=...) is how HF passes the
                # keyword into a per-example function.
                train_dataset = train_dataset.map(
                    coco_train_transform_example,
                    fn_kwargs={"input_size": input_size,
                               "with_outline": with_outline},
                )

                # Streaming datasets have no index, so shuffle with a buffer
                # instead of DataLoader(shuffle=True).
                if shuffle:
                    train_dataset = train_dataset.shuffle(
                        buffer_size=shuffle_buffer_size, seed=shuffle_seed
                    )

                # Shard after the shuffle: each rank then reads its own slice of
                # the shuffled stream rather than all of it.
                train_dataset = _split_stream_by_rank(
                    train_dataset, global_rank, world_size
                )

                train_loader = torch.utils.data.DataLoader(
                    dataset=train_dataset,
                    batch_size=batch_size,
                    shuffle=False,
                    num_workers=workers,
                    collate_fn=lambda b, _s=input_size: coco_collate_fn(b, input_size=_s),
                    pin_memory=True,
                    drop_last=True,
                    persistent_workers=False,
                )
            else:
                # ✅ map-style dataset: keep your batched transform via set_transform
                train_dataset = train_dataset.with_format("python")
                train_dataset.set_transform(
                    lambda batch, _s=input_size, _sj=scale_jitter, _cj=color_jitter,
                    _w=with_outline:
                    coco_train_transforms_fg_centered(
                        batch, input_size=_s, scale_jitter=_sj, color_jitter=_cj,
                        with_outline=_w,
                    )
                )

                # A map-style dataset has an index, so its rank shard is a
                # DistributedSampler: each rank takes its own slice of the index
                # space instead of iterating the whole set (None at world_size=1,
                # which leaves the shuffle= path exactly as it was).
                sampler = _distributed_sampler_for(
                    train_dataset, global_rank, world_size, shuffle, shuffle_seed
                )

                train_loader = torch.utils.data.DataLoader(
                    dataset=train_dataset,
                    batch_size=batch_size,
                    shuffle=shuffle and sampler is None,
                    sampler=sampler,
                    num_workers=workers,
                    collate_fn=lambda b, _s=input_size: coco_collate_fn(b, input_size=_s),
                    pin_memory=True,
                    drop_last=True,
                    persistent_workers=False,
                )
        else:
            train_dataset = train_dataset.map(train_transforms)
            train_dataset = train_dataset.with_format("torch")

            if isinstance(train_dataset, HFIterableDataset):
                if shuffle:
                    train_dataset = train_dataset.shuffle(
                        buffer_size=shuffle_buffer_size, seed=shuffle_seed
                    )
                # Shard after the shuffle, so each rank reads a disjoint slice of
                # the shuffled stream (a no-op at world_size=1).
                train_dataset = _split_stream_by_rank(
                    train_dataset, global_rank, world_size
                )
                shuffle_arg = False
                sampler = None
            else:
                sampler = _distributed_sampler_for(
                    train_dataset, global_rank, world_size, shuffle, shuffle_seed
                )
                shuffle_arg = shuffle and sampler is None

            train_loader = torch.utils.data.DataLoader(
                dataset=train_dataset,
                batch_size=batch_size,
                shuffle=shuffle_arg,
                sampler=sampler,
                num_workers=workers,
                collate_fn=collate_fn,
                pin_memory=True,
                drop_last=True,
                persistent_workers=False,
            )
    
    print(f"Created training dataloader with batch_size={batch_size}, workers={workers}")
    return train_loader


def get_validation_dataloader(
    dataset_name: str = "imagenet-1k",
    is_dataset_local: bool = False,
    batch_size: int = 32,
    workers: int = 4,
    global_rank: int = 0,
    world_size: int = 1,
    hf_token: str = os.getenv('HF_TOKEN'),
    input_size: int = DEFAULT_INPUT_SIZE,
    bdd_data_root: str | None = None,
    with_outline: bool = False,
) -> DataLoader:
    """
    Create training dataloader for HCM model training.

    Args:
        dataset_name: Name of dataset (HuggingFace format)
        is_dataset_local: Whether dataset is stored locally
        batch_size: Training batch size
        workers: Number of data loading workers
        global_rank: Global rank for distributed training
        world_size: World size for distributed training
        hf_token: HuggingFace token for dataset access
        input_size: Spatial size of the COCO/COCOA segmentation pipeline, so the
            validation tensors match the ones training used. Defaults to 192.
        with_outline: Also return the amodal full-outline target and its flags,
            as ``get_training_dataloader`` does. Off by default.

    Returns:
        Training DataLoader
    """

    if with_outline and (is_dataset_local or (
        "coco" not in dataset_name.lower() and dataset_name.lower() != BDD_DATASET_NAME
    )):
        raise ValueError(
            "with_outline is only implemented for the COCO/COCOA segmentation "
            f"pipeline; dataset_name={dataset_name!r}, "
            f"is_dataset_local={is_dataset_local} has no COCOA full-outline "
            "labels and no prepared-binary-mask rows."
        )

    if dataset_name.lower() == BDD_DATASET_NAME:
        return _bdd_dataloader(
            split="val",
            bdd_data_root=bdd_data_root,
            batch_size=batch_size,
            workers=workers,
            global_rank=global_rank,
            world_size=world_size,
            shuffle=False,
            input_size=input_size,
            transform=coco_val_transforms,
            with_outline=with_outline,
        )

    if is_dataset_local:
        print(f"Using local dataset, batch size: {batch_size}")
        val_dataset = GeneratorDataset(rank=global_rank, world_size=world_size)
        val_dataset.worker_initializer(val_transforms, "validation")
        
        val_loader = DataLoader(
            dataset=val_dataset,
            batch_size=1,  # Local dataset handles batching internally
            shuffle=None,
            num_workers=workers,
            collate_fn=collate_fn,
            pin_memory=False,
            drop_last=True,
            worker_init_fn=lambda worker_id: val_dataset.worker_initializer(val_transforms, "validation"),
            persistent_workers=False,
        )
    else:
        print(f"Loading HuggingFace dataset: {dataset_name}")
        d_cfg = DownloadConfig(resume_download=True, max_retries=500)
        
        login(token=hf_token)
        
        # Try loading without trust_remote_code first (for standard datasets)
        # If that fails, try with trust_remote_code=True (for datasets with custom code)
        try:
            if "shunk031/COCOA" in dataset_name:
                print("Loading COCO dataset with custom code")
                # Download annotations from MinIO
                annotations_path = download_coco_annotations()
                
                val_dataset = load_dataset(
                    path=dataset_name, 
                    name="COCO",
                    data_dir=annotations_path,
                    decode_rle=True,
                    split='validation', 
                    streaming=False, 
                    token=hf_token,
                    download_config=d_cfg,
                )
                val_dataset = ensure_image_id(val_dataset)
            else:
                if "shunk031/MSCOCO" in dataset_name:
                    print("Loading COCO dataset with custom code")
                    val_dataset = load_dataset(
                        dataset_name, 
                        year=2017,
                        coco_task="instances",
                        revision="main",
                        decode_rle=False,
                        split='validation', 
                        streaming=False, 
                        token=hf_token,
                        download_config=d_cfg,
                    )
                    val_dataset = ensure_image_id(val_dataset)
                else:
                    val_dataset = load_dataset(
                        dataset_name, 
                        split='validation', 
                        streaming=True, 
                        token=hf_token,
                        download_config=d_cfg,
                    )
        except (ValueError, TypeError) as e:
            print(f"Standard load failed, trying with trust_remote_code=True: {e}")
            val_dataset = load_dataset(
                dataset_name, 
                split='validation', 
                streaming=True, 
                token=hf_token,
                trust_remote_code=True,
                download_config=d_cfg,
            )
        
        logout()
        
        # Shuffle the dataset for random sampling during validation
        # Use buffer_size only for IterableDataset (streaming=True)
        #if isinstance(val_dataset, HFIterableDataset):
        #    val_dataset = val_dataset.shuffle(seed=42, buffer_size=10000)
        #else:
        #    val_dataset = val_dataset.shuffle(seed=42)
        
        # Use COCO-specific transforms and collate for COCO dataset
        if "coco" in dataset_name.lower():
            print("Using COCO-specific validation transforms with binary mask generation (set_transform)")
            # Use set_transform instead of map for better tensor handling
            # This applies transforms on-the-fly when items are accessed

            if isinstance(val_dataset, HFIterableDataset):
                # For IterableDataset, set_transform applies to each item
                val_dataset = val_dataset.cast_column("image", HFImage(decode=True))
                val_dataset = val_dataset.map(
                    coco_val_transform_example,
                    fn_kwargs={"input_size": input_size,
                               "with_outline": with_outline},
                )
                #val_dataset = val_dataset.with_format("torch")
                val_loader = DataLoader(
                    dataset=val_dataset, 
                    batch_size=batch_size, 
                    shuffle=None, 
                    num_workers=0,
                    collate_fn=lambda b, _s=input_size: coco_collate_fn(b, input_size=_s),
                    pin_memory=True,
                    drop_last=True,    
                    persistent_workers=False,  
                )
            else:
                val_dataset = val_dataset.cast_column("image", HFImage(decode=True))
                val_dataset = val_dataset.with_format("python")
                val_dataset.set_transform(
                    lambda batch, _s=input_size, _w=with_outline: coco_val_transforms(
                        batch, input_size=_s, with_outline=_w
                    )
                )
            
                val_loader = DataLoader(
                    dataset=val_dataset, 
                    batch_size=batch_size, 
                    shuffle=None, 
                    num_workers=0,
                    collate_fn=lambda b, _s=input_size: coco_collate_fn(b, input_size=_s),
                    pin_memory=True,
                    drop_last=True,    
                    persistent_workers=False,  
                )
        else:
            val_dataset = val_dataset.map(val_transforms)
            val_dataset = val_dataset.with_format("torch")  
            
            val_loader = DataLoader(
                dataset=val_dataset, 
                batch_size=batch_size, 
                shuffle=None, 
                num_workers=workers,
                collate_fn=collate_fn,      
                pin_memory=True,
                drop_last=True,    
                persistent_workers=False,  
            )
    
    print(f"Created validation dataloader with batch_size={batch_size}, workers={workers}")
    return val_loader