"""Score segmentation checkpoints on COCO val2017 or the COCOA val split.

The score is the trainer's own: the trainer's validation transform, collate and
``evaluate()``, at the checkpoint's input size, with the trainer's validation
loader settings: ``drop_last=True`` and the training run's batch size. The
trainer's validation loader uses the run's ``--batch_size`` (main.py default 8),
so pass the batch size the run used; the default 2 matches the 352 runs. With
``drop_last`` the batch size decides which tail images are scored, so the
``RESULT`` line reports it and the number of tail images dropped. A number from
this tool is therefore the same kind of number as a training run's ``val_iou``.

The grid is the trainer's too. Today's model -- ``mask_stride`` 4, the live
ResNet-50 checkpoints -- scores on the masks' own grid: the 88x88 logits are
bilinearly upsampled to the 352x352 target and the IoU is computed there. A
stride-2 model scores on its own 176x176 grid instead, with the masks brought
down by ``nearest``, which is what makes its number the same kind of number as
its own training loss (see ``trainer.target_grid_follows_logits``). A number
from a stride-2 checkpoint is therefore on its model's grid, not on 352, and is
not comparable with a stride-4 run's ``val_iou``.

    python tools/eval_seg.py --dataset coco  --data-root ~/eval/coco  a.pth b.pth
    python tools/eval_seg.py --dataset cocoa --data-root ~/eval/cocoa a.pth --json
    python tools/eval_seg.py --dataset coco  --data-root ~/eval/coco  a.pth --by-size
    python tools/eval_seg.py --dataset coco  --data-root <coco val root>  --baseline yolo11n-seg

``--by-size`` adds a per-object-size breakdown of the foreground recall: what
fraction of the ground-truth pixels of the small (under 32x32), medium (under
96x96) and large instances the model calls foreground. It answers a question the
one pooled IoU cannot -- whether a change helps small objects and hurts large
ones by the same amount -- and it changes nothing about the score beside it: the
breakdown is computed from the very tensors the IoU is computed from, in the same
pass. Note the two aggregate differently, on purpose: ``val_iou`` is the mean of
per-image ratios, each with its own ``eps``, while the buckets pool pixels over
every image scored. COCO only: COCOA's regions have no COCO area, and the flag
needs ``--num-workers 0`` because it builds each image's bucket map in this
process.

``--baseline NAME`` scores a public segmentation model instead of a checkpoint,
through the same transform, collate, ``evaluate()`` and ``--by-size`` breakdown,
so a baseline's IoU and ours are the same measure applied to two models. The
adapters and what each declares live in ``ecs_hcm_model/eval/baselines.py``. A
baseline runs at the native input size its adapter declares, or at
``--input-size`` when one is given -- the same model "as published" and at our
own resolution -- and the result reports which size it ran at and where that
size came from, so no row can read as one resolution while having run at
another. The adapter imports its package when it is used, so this tool runs on a
box where no baseline is installed, and a baseline whose class set is not ours
carries its caveat on the result rather than in a commit message.

Data layout under ``--data-root`` (``tools/fetch_seg_val.py`` builds it):

    coco:  annotations/instances_val2017.json, val2017/<file_name>
    cocoa: annotations/COCO_amodal_val2014.json, val2014/<file_name>, COCOA.py

COCO rows are built from the official annotation file rather than the HF
loader, which would download all of COCO. COCOA rows come from the
shunk031/COCOA builder's own example generator and features, with
``decode_rle=True`` as the trainer loads its validation split, fed only the
val2014 images the annotation file references.
"""

import sys

# The Jetson's torchaudio build is broken: importing it fails inside its native
# library. Nothing here uses audio, but torchmetrics (pulled in through the
# trainer) imports torchaudio whenever it is installed. A None entry makes the
# import system report it as absent, so torchmetrics takes its no-audio path.
# Set before any project import, and only when torchaudio is not loaded already.
if "torchaudio" not in sys.modules:
    sys.modules["torchaudio"] = None

import argparse
import importlib.util
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF

from ecs_hcm_model.core.hcm_nn import hcm_nn, neck_argument
from ecs_hcm_model.core.bdd import BDD100kSegDataset
# ``_resize_short_for`` and ``build_binary_mask_coco_all`` are the transform's own
# geometry and its own per-instance mask builder. Importing them (rather than
# restating either) is what lets --by-size put a label map through exactly the
# resize and crop the target went through, and bucket exactly the instances the
# target was built from.
from ecs_hcm_model.core.model_utils import (
    _resize_short_for,
    build_binary_mask_coco_all,
    coco_collate_fn,
    coco_val_transform_example,
)
from ecs_hcm_model.core.trainer import evaluate
from ecs_hcm_model.eval import baselines
from jetson.input_size import InputSizeError, resolve_input_size
from jetson.model_shape import (
    ModelShapeError,
    resolve_amodal_outline,
    resolve_backbone,
    resolve_demo_out_stride,
    resolve_decoder_arch,
    resolve_mask_stride,
    resolve_neck_channels,
)

# The trainer's validation loader settings, so a score here matches val_iou.
DEFAULT_BATCH_SIZE = 2
# The trainer's evaluate() runs the forward under torch.cuda.amp.autocast
# (float16 on CUDA), so fp16 is what every eval so far has measured.
PRECISIONS = ("fp16", "fp32")
DEFAULT_PRECISION = "fp16"
DROP_LAST = True

# Object-size buckets: COCO's convention, applied to the instance's pixel count
# in the ORIGINAL image (COCO: area < 32^2 small, < 96^2 medium, else large). The
# count is taken before the transform's resize, so an object the resize enlarges
# is still bucketed as the small object it is.
SMALL_MAX_PIXELS = 32 * 32
MEDIUM_MAX_PIXELS = 96 * 96
SIZE_BUCKETS = ("small", "medium", "large")
# 0 is background: no instance covers that pixel.
BUCKET_ID = {"small": 1, "medium": 2, "large": 3}
# The rule, as it is reported. Compact and space-free so it survives a RESULT
# line: the bucket is the instance's pixel count in the original image, and a
# pixel covered by several instances is charged to the smallest.
SIZE_RULE = "origpx:small<32^2,medium<96^2,smallest-wins"

COCO_ANNOTATIONS = Path("annotations") / "instances_val2017.json"
COCO_IMAGES = "val2017"
COCOA_ANNOTATIONS = Path("annotations") / "COCO_amodal_val2014.json"
COCOA_BUILDER = "COCOA.py"


class EvalError(Exception):
    """A condition that makes the requested score impossible or meaningless."""


# datasets 2.19.0 began EXIF-transposing images in Image.decode_example. The
# trainer pins 2.14, which does not, so a newer release would score images in
# an orientation the model never trained on. 2.18.0 is the last without it.
EXIF_TRANSPOSE_DATASETS = (2, 19, 0)


def _version_tuple(text):
    """The leading numeric release of a version string: '2.19.0.dev0' -> (2, 19, 0)."""
    parts = []
    for piece in str(text).split("."):
        digits = ""
        for ch in piece:
            if not ch.isdigit():
                break
            digits += ch
        if not digits:
            break
        parts.append(int(digits))
    return tuple(parts)


def _datasets_version():
    """The installed ``datasets`` version, read without importing it; None if absent."""
    try:
        from importlib.metadata import PackageNotFoundError, version
    except ImportError:
        return None
    try:
        return version("datasets")
    except PackageNotFoundError:
        return None


def check_datasets_version(installed=None):
    """Refuse a ``datasets`` release that EXIF-transposes on decode."""
    installed = _datasets_version() if installed is None else installed
    if installed is None:
        return
    if _version_tuple(installed) >= EXIF_TRANSPOSE_DATASETS:
        floor = ".".join(map(str, EXIF_TRANSPOSE_DATASETS))
        raise EvalError(
            f"datasets {installed} is installed; releases from {floor} on EXIF-transpose "
            f"images when decoding, unlike the datasets==2.14.0 the trainer pins, so "
            f"scores would not match val_iou. Install datasets==2.14.0 for the eval.")


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

class TransformedRows(Dataset):
    """Rows in the trainer's example format, through the trainer's val transform."""

    def __init__(self, rows, input_size):
        self.rows = rows
        self.input_size = int(input_size)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        return coco_val_transform_example(self.rows[i], input_size=self.input_size)


class CocoRows:
    """COCO val2017 rows from the official instances file, image ids ascending.

    Images without annotations are skipped, as the HF validation split the
    trainer scores (shunk031/MSCOCO) drops them: 4952 of val2017's 5000 images
    are kept. ``skipped_empty`` counts the ones left out.
    """

    def __init__(self, data_root):
        root = Path(data_root)
        ann_path = root / COCO_ANNOTATIONS
        if not ann_path.is_file():
            raise EvalError(f"COCO annotations not found: {ann_path}")
        with open(ann_path) as f:
            data = json.load(f)
        self.anns = {}
        for a in data["annotations"]:
            self.anns.setdefault(a["image_id"], []).append(a)
        images = sorted(data["images"], key=lambda i: i["id"])
        self.images = [i for i in images if self.anns.get(i["id"])]
        self.skipped_empty = len(images) - len(self.images)
        # The size each image declares, for --by-size: it buckets an instance by
        # its pixel count in the original image, which needs the original size
        # and not another decode. (None where an annotation file omits them; the
        # by-size path refuses there, and nothing else reads this.)
        self.sizes = {i["id"]: (i.get("width"), i.get("height")) for i in self.images}
        print(f"[eval] COCO: skipped {self.skipped_empty} images with no annotations, "
              f"scoring {len(self.images)}", file=sys.stderr)
        self.image_dir = root / COCO_IMAGES

    def bucket_map(self, image_id, input_size):
        """``(labels, instances)`` for one image, for ``--by-size``."""
        width, height = self.sizes.get(image_id, (None, None))
        if not width or not height:
            raise EvalError(
                f"image {image_id} declares no width/height in {COCO_ANNOTATIONS.name}; "
                f"--by-size buckets an instance by its pixel count in the original image "
                f"and cannot do that without them"
            )
        return bucket_map_for_row(self.anns.get(image_id, []), width, height, input_size)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, i):
        info = self.images[i]
        img = Image.open(self.image_dir / info["file_name"]).convert("RGB")
        return {
            "image": img,
            "annotations": self.anns.get(info["id"], []),
            "image_id": info["id"],
        }


def load_cocoa_builder(data_root):
    """The COCOA builder from ``<data_root>/COCOA.py``, configured as the trainer's val split.

    The builder class is imported and constructed directly instead of through
    ``datasets.load_dataset_builder``, which newer ``datasets`` releases no
    longer allow for script datasets. The result is the same builder object.
    """
    root = Path(data_root)
    script = root / COCOA_BUILDER
    if not script.is_file():
        raise EvalError(f"COCOA builder script not found: {script}")
    name = "ecs_hcm_cocoa_builder"
    spec = importlib.util.spec_from_file_location(name, script)
    module = importlib.util.module_from_spec(spec)
    # Registered before exec: the script's dataclasses look their module up.
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module.CocoaDataset(
        config_name="COCO", decode_rle=True, cache_dir=str(root / "hf_cache")
    )


def _field_names(feature):
    """Field names of a ``Sequence({...})`` feature, as any ``datasets`` release declares it."""
    inner = getattr(feature, "feature", feature)
    return list(inner) if isinstance(inner, dict) else []


def _columns(items, names):
    """A list of dicts as one dict of lists, the form ``Sequence({...})`` stores."""
    return {k: [item[k] for item in items] for k in names}


def as_stored_example(example, features):
    """The builder's example with its ``Sequence({...})`` fields already in columns.

    The builder yields ``annotations`` and each annotation's ``regions`` as
    lists of dicts. ``datasets`` 2.14 (the version the project pins) stores
    either form as columns; 5.x accepts only columns. Transposing here gives
    the same stored row on both.
    """
    ann_feature = features["annotations"]
    ann_names = _field_names(ann_feature)
    region_names = _field_names(getattr(ann_feature, "feature", ann_feature)["regions"])
    anns = [dict(a, regions=_columns(a["regions"], region_names)) for a in example["annotations"]]
    return dict(example, annotations=_columns(anns, ann_names))


def cocoa_rows(data_root, builder=None):
    """COCOA val rows: the builder's examples, stored through its own features.

    The round trip through ``Dataset.from_generator`` with the builder's
    features is what gives a row the shape the trainer's HF loader hands the
    transform (for instance ``annotations`` as a dict of per-region lists).
    """
    import datasets

    root = Path(data_root)
    ann_path = root / COCOA_ANNOTATIONS
    if not ann_path.is_file():
        raise EvalError(f"COCOA annotations not found: {ann_path}")
    if builder is None:
        builder = load_cocoa_builder(root)

    features = builder.info.features

    def gen():
        # The builder reads <base_image_dir>/val2014/<file_name>.
        for _key, example in builder._generate_examples(
            split="val", base_image_dir=str(root), amodal_annotation_path=str(ann_path)
        ):
            yield as_stored_example(example, features)

    return datasets.Dataset.from_generator(
        gen, features=features, cache_dir=str(root / "hf_cache")
    )


def build_rows(dataset, data_root):
    if dataset == "coco":
        return CocoRows(data_root)
    if dataset == "cocoa":
        return cocoa_rows(data_root)
    if dataset == "bdd":
        # The prepared BDD val split: each row is already (image, binary object
        # mask), which is the same target the trainer scored, so the eval and
        # the trainer measure the same thing. --data-root is the prepared tree.
        return BDD100kSegDataset(data_root, split="val")
    raise EvalError(f"unknown dataset {dataset!r}")


# ---------------------------------------------------------------------------
# Object-size buckets (--by-size)
# ---------------------------------------------------------------------------

class RowSize:
    """Just the size a mask builder reads off a decoded image.

    ``build_binary_mask_coco_all`` looks at the image only to learn H and W, and
    the COCO annotation file records both for every image it ships, so a bucket
    map costs no second JPEG decode. Should the two ever disagree, the coverage
    check in ``SizeRecall`` fails on that image and names it.
    """

    __slots__ = ("width", "height", "size")

    def __init__(self, width, height):
        self.width = int(width)
        self.height = int(height)
        self.size = (self.width, self.height)


def bucket_of(pixels):
    """COCO's size bucket for an instance of ``pixels`` pixels."""
    if pixels < SMALL_MAX_PIXELS:
        return "small"
    if pixels < MEDIUM_MAX_PIXELS:
        return "medium"
    return "large"


def bucket_map_for_row(annotations, width, height, input_size):
    """Per-pixel bucket of this image's ground truth, on the target's own grid.

    Returns ``(labels, instances)``: an ``input_size`` square uint8 array holding
    ``BUCKET_ID[bucket]`` where an instance covers the pixel and 0 elsewhere, and
    the number of instances bucketed into each bucket.

    Every instance is built by the transform's own builder, one instance per
    call, so the annotation-to-mask step is the code the target was built from
    rather than a second implementation of it. Instances are painted largest
    first, so where instances overlap the smallest one covering the pixel owns
    it, whatever order they arrived in (a stable sort, and instances of equal
    pixel count are in the same bucket anyway). The label map is then put
    through the transform's own resize and centre crop: nearest-neighbour maps a
    pixel to a pixel, so the resized map's nonzero pixels are exactly the
    resized target's foreground pixels.
    """
    size = RowSize(width, height)
    found = []
    for index, ann in enumerate(annotations):
        mask = build_binary_mask_coco_all({"image": size, "annotations": [ann]})
        if mask is None:
            continue
        pixels = int((mask > 0).sum())
        if pixels == 0:
            continue
        found.append((pixels, index, mask))

    instances = dict.fromkeys(SIZE_BUCKETS, 0)
    for pixels, _index, _mask in found:
        instances[bucket_of(pixels)] += 1

    labels = np.zeros((int(height), int(width)), dtype=np.uint8)
    for pixels, _index, mask in sorted(found, key=lambda f: -f[0]):
        labels[mask > 0] = BUCKET_ID[bucket_of(pixels)]

    cropped = TF.center_crop(
        TF.resize(Image.fromarray(labels), _resize_short_for(int(input_size)),
                  interpolation=InterpolationMode.NEAREST),
        (int(input_size), int(input_size)),
    )
    return np.array(cropped), instances


def foreground_prediction(logits, thresh=0.5):
    """The metric's own foreground rule, for the shapes ``iou_dice_fg_from_logits`` takes.

    Mirrors that function's probability branch rather than thresholding an
    argmax. The recall below has to be the IoU's own foreground, or the two
    numbers printed side by side would be two different measurements.
    """
    if logits.ndim == 4 and logits.size(1) == 2:
        p_fg = torch.softmax(logits, dim=1)[:, 1]
    elif logits.ndim == 4 and logits.size(1) == 1:
        p_fg = torch.sigmoid(logits[:, 0])
    elif logits.ndim == 3:
        p_fg = torch.sigmoid(logits)
    else:
        raise EvalError(f"unexpected logits shape: {tuple(logits.shape)}")
    return p_fg > thresh


class SizeRecall:
    """Per-object-size foreground recall, pooled over the images that were scored.

    Fed by ``evaluate()``'s ``on_metric_inputs`` hook, i.e. from the very tensors
    the IoU and Dice are computed from, in the same pass. Nothing here runs the
    model again. Those tensors are on whichever grid the metric was scored on --
    the target's own for a stride-4 model, the model's own 176x176 for a stride-2
    one -- so the label map comes down to that grid with the same nearest
    downsample the masks went through and the coverage check below is still an
    exact comparison.
    """

    def __init__(self):
        self.gt = dict.fromkeys(SIZE_BUCKETS, 0)
        self.hit = dict.fromkeys(SIZE_BUCKETS, 0)
        self.instances = dict.fromkeys(SIZE_BUCKETS, 0)
        self.images = 0

    def add_batch(self, logits, target, image_ids, bucket_map_fn):
        """Add one scored batch; ``bucket_map_fn(image_id)`` gives its label map."""
        pred_fg = foreground_prediction(logits)
        tgt_fg = target if target.dtype == torch.bool else (target == 1)
        if pred_fg.shape != tgt_fg.shape:
            raise EvalError(
                f"--by-size: logits {tuple(pred_fg.shape)} and target "
                f"{tuple(tgt_fg.shape)} are on different grids"
            )
        ids = [int(i) for i in torch.as_tensor(image_ids).flatten().tolist()]
        if len(ids) != pred_fg.shape[0]:
            raise EvalError(
                f"--by-size: {len(ids)} image ids for {pred_fg.shape[0]} scored samples"
            )
        for i, image_id in enumerate(ids):
            bucket_map, instances = bucket_map_fn(image_id)
            labels = torch.from_numpy(np.ascontiguousarray(bucket_map)).to(tgt_fg.device)
            gt = tgt_fg[i]
            if labels.shape != gt.shape:
                # The metric is on the model's own grid (a stride-2 model scores
                # its 176x176 output): the label map follows it down with
                # nearest, the same downsample the loss applied to the same
                # full-resolution mask, so the coverage check stays exact.
                labels = torch.nn.functional.interpolate(
                    labels[None, None].float(), size=tuple(gt.shape),
                    mode="nearest")[0, 0].long()
            covered = labels > 0
            if covered.shape != gt.shape or not torch.equal(covered, gt):
                raise EvalError(
                    f"--by-size: image {image_id}: the instances bucketed here do not "
                    f"add up to the target the metric scored -- they cover "
                    f"{int(covered.sum())} of its {int(gt.sum())} foreground pixels. "
                    f"The breakdown would be measured against a different ground truth."
                )
            for name in SIZE_BUCKETS:
                bucket = (labels == BUCKET_ID[name]) & gt
                self.gt[name] += int(bucket.sum())
                self.hit[name] += int((bucket & pred_fg[i]).sum())
                self.instances[name] += instances[name]
            self.images += 1

    def result(self):
        out = {}
        for name in SIZE_BUCKETS:
            gt = self.gt[name]
            out[name] = {
                # None, not 0.0: a bucket with no ground-truth pixels has no recall.
                "recall": (self.hit[name] / gt) if gt else None,
                "gt_pixels": gt,
                "instances": self.instances[name],
            }
        out["rule"] = SIZE_RULE
        out["images"] = self.images
        return out


def _size_callback(recall, bucket_map, input_size):
    """``SizeRecall`` in the shape ``evaluate()`` calls it."""

    def on_metric_inputs(logits, target, batch):
        recall.add_batch(
            logits, target, batch["image_id"],
            lambda image_id: bucket_map(image_id, input_size),
        )

    return on_metric_inputs


def format_by_size(by_size):
    """The buckets of ``--by-size``, as one RESULT-line token."""
    parts = []
    for name in SIZE_BUCKETS:
        b = by_size[name]
        recall = "n/a" if b["recall"] is None else f"{b['recall']:.3f}"
        parts.append(f"{name}:{recall}({b['gt_pixels']}px,{b['instances']})")
    return f"by_size={','.join(parts)} rule={by_size['rule']}"


# ---------------------------------------------------------------------------
# Checkpoints
# ---------------------------------------------------------------------------

def metadata_candidates(ckpt_path):
    """Sidecar files that may describe ``ckpt_path``, most specific first."""
    p = Path(ckpt_path)
    return [
        p.with_suffix(".json"),               # model.pth -> model.json
        p.with_suffix(".meta.json"),          # model.pth -> model.meta.json
        p.parent / "metadata.json",           # the demo's artifact-dir layout
    ]


def find_metadata(ckpt_path, explicit=None):
    """Return ``(meta, path)`` for a checkpoint, or ``(None, None)`` if it has none.

    A sidecar that exists but cannot be parsed is an error, not an absence:
    falling back to the default size would score a 352 model at 192 and still
    print a number.
    """
    paths = [Path(explicit)] if explicit else metadata_candidates(ckpt_path)
    for path in paths:
        if path.is_file():
            try:
                with open(path) as f:
                    return json.load(f), path
            except (OSError, ValueError) as e:
                raise EvalError(f"metadata {path} could not be read: {e}") from e
    if explicit:
        raise EvalError(f"metadata file not found: {explicit}")
    return None, None


def resolve_size(ckpt_path, override=None, explicit_meta=None):
    """Return ``(input_size, source)`` for a checkpoint.

    ``--input-size`` wins; otherwise the checkpoint's metadata through the
    demo's resolver; otherwise the default. The demo's environment override is
    deliberately not consulted: an exported demo variable must not change what
    an evaluation measures.
    """
    if override is not None:
        if int(override) <= 0:
            raise EvalError(f"--input-size must be positive, got {override}")
        return int(override), "--input-size"
    meta, path = find_metadata(ckpt_path, explicit_meta)
    try:
        size, source = resolve_input_size(meta, {})
    except InputSizeError as e:
        raise EvalError(f"metadata {path}: {e}") from e
    if path is not None:
        source = f"{source} ({path})"
    return size, source


def unwrap_state_dict(ckpt):
    if isinstance(ckpt, dict):
        for k in ("state_dict", "model_state_dict", "model"):
            if isinstance(ckpt.get(k), dict):
                return ckpt[k]
    return ckpt


def num_classes_from_state(sd):
    for k, v in sd.items():
        if k.endswith("segmentation_head.weight") and hasattr(v, "shape"):
            return int(v.shape[0])
    raise EvalError("checkpoint has no segmentation_head.weight: not a segmentation checkpoint")


def load_checkpoint(path, device, *, meta=None, out_stride=None, log=None):
    """Build the model and load the checkpoint exactly (strict).

    A state dict is built at the stride the demos would use: ``--out-stride``,
    then the sidecar's recorded stride, then the checkpoint's own, then 16. As
    with the input size, the demos' environment override is not consulted.

    The ladder, the neck and the mask branch come from the same artifacts and
    the same rule (``resolve_backbone``/``resolve_neck_channels``/
    ``resolve_mask_stride``): the sidecar when it states them, otherwise the
    weights' own keys, so a checkpoint from before the fields existed is the
    resnet50 / 2048-widening / stride-4 model it was trained as. A stated value
    the weights contradict raises rather than building a model that cannot load.
    """
    ckpt = torch.load(path, map_location="cpu")
    if isinstance(ckpt, torch.nn.Module):
        model = ckpt
    else:
        sd = unwrap_state_dict(ckpt)
        # Before anything reads the weights' architecture: a checkpoint
        # with no segmentation head is refused for that, not for how its
        # (absent) backbone keys look.
        num_classes = num_classes_from_state(sd)
        try:
            stride, source, warning = resolve_demo_out_stride(out_stride, {}, meta, ckpt)
        except ModelShapeError as e:
            raise EvalError(f"{path}: {e}") from e
        if log is not None:
            if warning:
                log(f"warning: {path}: {warning}")
            log(f"{path}: out_stride {stride} from {source}")
        try:
            decoder_arch, arch_source = resolve_decoder_arch(meta, sd)
            backbone, backbone_source = resolve_backbone(meta, ckpt)
            neck_channels, neck_source = resolve_neck_channels(meta, ckpt)
            mask_stride, mask_source = resolve_mask_stride(meta, ckpt)
            amodal_outline, outline_source = resolve_amodal_outline(meta, ckpt)
        except ModelShapeError as e:
            raise EvalError(f"{path}: {e}") from e
        if log is not None:
            log(f"{path}: decoder_arch {decoder_arch} from {arch_source}")
            log(f"{path}: backbone {backbone} from {backbone_source}")
            log(f"{path}: neck_channels {neck_channels} from {neck_source}")
            log(f"{path}: mask_stride {mask_stride} from {mask_source}")
            log(f"{path}: amodal_outline {amodal_outline} from {outline_source}")
        model = hcm_nn(
            enable_finetuning=True,
            num_segmentation_classes=num_classes,
            out_stride=stride,
            decoder_arch=decoder_arch,
            backbone=backbone,
            neck_channels=neck_argument(neck_channels),
            mask_stride=mask_stride,
            amodal_outline=amodal_outline,
            # The state dict is loaded over the backbone just below, so nothing
            # here reaches the network for pretrained weights.
            backbone_pretrained=False,
        )
        model.load_state_dict(sd, strict=True)
    return model.to(device)


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

class CountingLoader:
    """Iterate a loader, count what is scored and what failed, and drop placeholders.

    ``coco_collate_fn`` drops the samples that failed the transform, and
    replaces a batch whose every sample failed with one zero image marked
    ``valid=False``; ``evaluate()`` would score that blank as if it were data.
    Such batches are skipped here and their samples counted as failed.

    The batch limit is enforced here rather than by ``evaluate()``, which only
    checks it after taking the next batch: counted at that point, the batch it
    then discards would be reported as scored.
    """

    def __init__(self, loader, batch_size, max_batches=0):
        self.loader = loader
        self.batch_size = int(batch_size)
        self.max_batches = int(max_batches)
        self.images = 0
        self.failed = 0
        # Seconds spent waiting for the loader to produce a batch (decode,
        # transform, collate). The rest of the scoring time is the model's:
        # evaluate() reads loss.item() each batch, so the device has finished.
        self.data_seconds = 0.0

    def __iter__(self):
        scored = 0
        it = iter(self.loader)
        while True:
            t0 = time.perf_counter()
            try:
                batch = next(it)
            except StopIteration:
                return
            finally:
                self.data_seconds += time.perf_counter() - t0
            if self.max_batches and scored >= self.max_batches:
                return
            valid = batch.get("valid") if isinstance(batch, dict) else None
            if valid is not None and not bool(torch.as_tensor(valid).any()):
                self.failed += self.batch_size
                continue
            n = int(batch["pixel_values"].shape[0])
            self.images += n
            self.failed += self.batch_size - n
            scored += 1
            yield batch


def effective_precision(requested, device):
    """The precision the forward actually runs at.

    The trainer's autocast is torch.cuda.amp, which is inert off CUDA: on any
    other device an fp16 request still runs, and must be reported, as fp32.
    """
    return "fp16" if requested == "fp16" and str(device).startswith("cuda") else "fp32"


def score(model, rows, input_size, device, batch_size=DEFAULT_BATCH_SIZE,
          num_workers=0, max_batches=0, precision=DEFAULT_PRECISION, by_size=False):
    """Score one model on one set of rows. Returns the result dict."""
    ds = TransformedRows(rows, input_size)
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=DROP_LAST,
        collate_fn=lambda b, _s=int(input_size): coco_collate_fn(b, input_size=_s),
    )
    recall = None
    on_metric_inputs = None
    if by_size:
        bucket_map = getattr(rows, "bucket_map", None)
        if bucket_map is None:
            raise EvalError(
                f"--by-size buckets COCO instances by their area in the original image; "
                f"{type(rows).__name__} has no such instances"
            )
        recall = SizeRecall()
        on_metric_inputs = _size_callback(recall, bucket_map, input_size)
    counted = CountingLoader(loader, batch_size, max_batches)
    t0 = time.perf_counter()
    with torch.no_grad():
        metrics = evaluate(model, counted, device, is_segmentation=True,
                           use_amp=(precision == "fp16"), on_metric_inputs=on_metric_inputs)
    scoring = time.perf_counter() - t0

    total = len(ds)
    result = {
        "iou": float(metrics["val_iou"]),
        "dice": float(metrics["val_dice"]),
        "loss": float(metrics["val_loss"]),
        "n_images": counted.images,
        "n_rows": total,
        "n_dropped_tail": total - (total // batch_size) * batch_size,
        "n_failed": counted.failed,
        "input_size": int(input_size),
        "batch_size": int(batch_size),
        "precision": effective_precision(precision, device),
        "precision_requested": precision,
        "score_seconds": scoring,
        "data_wait_seconds": counted.data_seconds,
        "model_seconds": max(scoring - counted.data_seconds, 0.0),
        "images_per_s": counted.images / scoring if scoring > 0 else 0.0,
    }
    if recall is not None:
        # Only when asked for: without the flag the result is what it always was.
        result["by_size"] = recall.result()
    return result


def format_result(r):
    line = (
        f"RESULT {r['checkpoint']}: dataset={r['dataset']} iou={r['iou']:.2f} "
        f"dice={r['dice']:.2f} loss={r['loss']:.4f} n_images={r['n_images']} "
        f"input_size={r['input_size']} ({r['input_size_source']}) "
        f"batch_size={r['batch_size']} n_dropped_tail={r['n_dropped_tail']} "
        f"precision={r['precision']} seconds={r['seconds']:.0f} "
        f"images_per_s={r['images_per_s']:.2f} data_wait_s={r['data_wait_seconds']:.0f} "
        f"model_s={r['model_seconds']:.0f}"
    )
    if r["n_failed"]:
        line += f" FAILED_IMAGES={r['n_failed']}"
    if "by_size" in r:
        line += " " + format_by_size(r["by_size"])
    if "caveat" in r:
        line += f" caveat={r['caveat']}"
    return line


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Score segmentation checkpoints on COCO val2017 or COCOA val "
        "with the trainer's own validation transform and evaluate()."
    )
    p.add_argument("--dataset", required=True, choices=("coco", "cocoa", "bdd"))
    p.add_argument("--data-root", required=True, help="directory laid out as tools/fetch_seg_val.py builds it")
    p.add_argument("checkpoints", nargs="*", help="checkpoint .pth files")
    p.add_argument("--baseline", default=None,
                   help=("score a public segmentation model instead of a checkpoint, through "
                         "the same transform, collate, evaluate() and --by-size breakdown. "
                         f"Known baselines: {', '.join(baselines.names())}. The model runs at "
                         "the native input size its adapter declares, or at --input-size."))
    p.add_argument("--input-size", type=int, default=None,
                   help=("override the size read from each checkpoint's metadata; for a "
                         "baseline, the size to score it at instead of its adapter's native one"))
    p.add_argument("--out-stride", type=int, default=None,
                   help="override the backbone stride read from each checkpoint's metadata")
    p.add_argument("--metadata", default=None,
                   help="metadata JSON for every checkpoint (default: a sidecar next to each)")
    p.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE,
                   help=(f"the training run's batch size (--batch_size; default "
                         f"{DEFAULT_BATCH_SIZE} = the 352 runs)"))
    p.add_argument("--num-workers", type=int, default=0,
                   help="DataLoader workers (default 0: with 3, COCOA ran out of memory "
                   "at batch 0 on the Jetson's 8 GB)")
    p.add_argument("--precision", choices=PRECISIONS, default=DEFAULT_PRECISION,
                   help=("fp16 (default): the forward runs under the trainer's CUDA "
                         "autocast, as training validation does. fp32: autocast off, "
                         "as a cross-check. On CPU both are fp32."))
    p.add_argument("--max-batches", type=int, default=0, help="stop after N batches (0: all)")
    p.add_argument("--by-size", action="store_true",
                   help=("also report the foreground recall per object size (COCO only): "
                         "small under 32^2 original pixels, medium under 96^2, else large, "
                         "with an overlapping pixel charged to the smallest instance. The "
                         "score is unchanged. Needs --num-workers 0."))
    p.add_argument("--device", default=None, help="default: cuda if available, else cpu")
    p.add_argument("--json", action="store_true",
                   help="print one JSON object per checkpoint instead of RESULT lines")
    return p.parse_args(argv)


def _resolve_baseline_request(args):
    """``(spec, size, source)`` when a baseline was asked for, None for checkpoints.

    Every refusal here happens before any image is read: a run that would not mean
    what it says should cost seconds, not a pass over the validation set.
    """
    if not args.baseline:
        if not args.checkpoints:
            raise EvalError(
                "no checkpoint given; pass one or more checkpoint .pth files, or "
                f"--baseline NAME (known baselines: {', '.join(baselines.names())})"
            )
        return None
    conflicts = [
        name for name, given in (
            ("a checkpoint path", bool(args.checkpoints)),
            ("--metadata", bool(args.metadata)),
            ("--out-stride", args.out_stride is not None),
        ) if given
    ]
    if conflicts:
        raise EvalError(
            f"--baseline {args.baseline} scores a public model, which has none of: "
            f"{', '.join(conflicts)}. Drop them."
        )
    try:
        return baselines.resolve(args.baseline, args.input_size)
    except baselines.BaselineError as e:
        raise EvalError(str(e)) from e


def _report(r, what, device, log, as_json):
    """Print one row, with the notes that belong with it."""
    if r["precision"] != r["precision_requested"]:
        log(f"note: {what}: --precision {r['precision_requested']} ran as "
            f"{r['precision']} on {device} (autocast is CUDA-only)")
    if r["n_failed"]:
        log(f"warning: {what}: {r['n_failed']} images failed the transform and were not scored")
    print(json.dumps(r) if as_json else format_result(r), flush=True)


def _score_baseline(args, spec, size, source, rows, device, log):
    """Score one baseline through the same ``score()`` a checkpoint goes through."""
    t0 = time.time()
    try:
        model = baselines.build(spec.name, device, size)
        r = score(model, rows, size, device, batch_size=args.batch_size,
                  num_workers=args.num_workers, max_batches=args.max_batches,
                  precision=args.precision, by_size=args.by_size)
    except (EvalError, baselines.BaselineError, RuntimeError) as e:
        log(f"error: baseline {spec.name}: {e}")
        return 1
    r.update(checkpoint=f"baseline:{spec.name}", dataset=args.dataset,
             input_size_source=source, seconds=time.time() - t0)
    if spec.caveat:
        # Not the same measurement as ours: the caveat travels with the number,
        # as a space-free id on the RESULT line and with its sentence in the JSON.
        r["caveat"] = spec.caveat
        r["caveat_note"] = spec.caveat_note
    _report(r, f"baseline {spec.name}", device, log, args.json)
    return 0


def main(argv=None):
    args = parse_args(argv)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    log = lambda *a: print(*a, file=sys.stderr, flush=True)

    if os.environ.get("PYTORCH_NO_CUDA_MEMORY_CACHING"):
        log("PYTORCH_NO_CUDA_MEMORY_CACHING is set: the CUDA caching allocator is off")

    try:
        check_datasets_version()
        # Refuse a breakdown that could not mean what it says, before reading
        # any image: COCOA has no COCO instances to bucket, and the bucket map is
        # built here rather than shipped from worker processes.
        if args.by_size and args.dataset != "coco":
            raise EvalError(
                f"--by-size buckets COCO instances by their area in the original image; "
                f"{args.dataset} regions have no such area. Score --dataset coco instead."
            )
        if args.by_size and args.num_workers:
            raise EvalError(
                f"--by-size builds each image's bucket map in this process, so it needs "
                f"--num-workers 0 (got {args.num_workers})."
            )
        baseline = _resolve_baseline_request(args)
        sizes = metas = None
        if baseline is None:
            # Resolve every size before the slow part, so a bad sidecar fails in seconds.
            sizes = {c: resolve_size(c, args.input_size, args.metadata) for c in args.checkpoints}
            metas = {c: find_metadata(c, args.metadata)[0] for c in args.checkpoints}
        rows = build_rows(args.dataset, args.data_root)
    except EvalError as e:
        log(f"error: {e}")
        return 2
    log(f"{args.dataset}: {len(rows)} rows, device={device}")

    if baseline is not None:
        spec, size, source = baseline
        log(f"baseline {spec.name}: input size {size} from {source}")
        if spec.caveat:
            log(f"note: baseline {spec.name}: {spec.caveat}: {spec.caveat_note}")
        return _score_baseline(args, spec, size, source, rows, device, log)

    status = 0
    for ckpt in args.checkpoints:
        size, source = sizes[ckpt]
        log(f"{ckpt}: input size {size} from {source}")
        t0 = time.time()
        try:
            model = load_checkpoint(ckpt, device, meta=metas[ckpt],
                                    out_stride=args.out_stride, log=log)
            r = score(model, rows, size, device, batch_size=args.batch_size,
                      num_workers=args.num_workers, max_batches=args.max_batches,
                      precision=args.precision, by_size=args.by_size)
        except (EvalError, RuntimeError) as e:
            log(f"error: {ckpt}: {e}")
            status = 1
            continue
        r.update(checkpoint=str(ckpt), dataset=args.dataset,
                 input_size_source=source, seconds=time.time() - t0)
        _report(r, str(ckpt), device, log, args.json)
        del model
        if device.startswith("cuda"):
            torch.cuda.empty_cache()
    return status


if __name__ == "__main__":
    sys.exit(main())
