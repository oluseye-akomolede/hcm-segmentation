"""Public segmentation baselines, presented to the eval tool as the model it scores.

The eval tool already knows how to score a model: the trainer's validation
transform, the trainer's collate, `evaluate()`, and the metric's own tensors. An
adapter here plugs a public model into that path unchanged, so the only thing
that differs between a baseline's row and ours is the model.

What is identical: the transform and its geometry, the collate, the batch size,
the metric, the pass the score and the `--by-size` breakdown come from.
What is not: the model, the input size it runs at (each adapter declares its
native size, which is the default; `--input-size` scores it at another one on
purpose, so the same model can be reported "as published" and at our own
resolution -- every result reports the size it actually ran at), its class set
(declared as a caveat when it is not ours), and the `loss` field, which is the
trainer's own training objective evaluated on the baseline's probability -- the
comparable numbers are the IoU and the Dice.

Nothing here imports a baseline's package at module scope: the import happens in
`build()`, so this module loads on a box where no baseline is installed, and a
missing package is one actionable error instead of an import failure of the tool.
The packages are called, never vendored: `ultralytics` is AGPL-3.0, and neither
its source nor its weights belong in this repository.

Two properties of the transform matter here and are worth stating once:

- It normalizes with ImageNet mean/std, so an adapter whose model wants 0-1
  pixels must undo that itself (`denormalize`).
- It resizes the short side and then centre-crops to a square, so a baseline runs
  at our crop geometry at the size it is handed. No adapter can reproduce a
  non-square published geometry (the torchvision weights' recipe resizes the
  short side to 520 and does not crop); that is the deliberate price of one
  shared transform, and the size each row ran at is reported.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

# The transform's normalization (coco_val_transform_example), so an adapter can
# undo it for a model that wants pixels.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# ultralytics' own default confidence: the model runs as its authors ship it, and
# our metric's 0.5 threshold then applies to the union of the surviving masks.
YOLO_CONF = 0.25
# The weights file the package downloads on first use; never committed.
YOLO_WEIGHTS = "yolo11n-seg.pt"
# The published COCO/VOC-labels weights, by the name torchvision resolves.
TORCHVISION_WEIGHTS = "COCO_WITH_VOC_LABELS_V1"

# logit(p) is +-inf at p = 0 and 1. The clamp keeps it finite at a probability the
# metric's `p_fg > 0.5` still classifies the same way (sigmoid(+-13.8) ~ 1e-6).
_LOGIT_EPS = 1e-6


class BaselineError(Exception):
    """A baseline that cannot be scored as asked."""


# ---------------------------------------------------------------------------
# The metric's own interface
# ---------------------------------------------------------------------------

def foreground_logits(p_fg):
    """``[B,2,H,W]`` logits whose metric foreground probability is ``p_fg``.

    The metric takes ``softmax(logits, dim=1)[:, 1]`` for a two-channel output, so
    channel 0 is zero and channel 1 is ``logit(p_fg)``: the round-trip returns the
    model's own probability, and the metric's 0.5 threshold is the model's own
    0.5 rather than a second, unstated knob.
    """
    p = p_fg.clamp(min=_LOGIT_EPS, max=1.0 - _LOGIT_EPS)
    z = torch.log(p) - torch.log1p(-p)
    return torch.stack([torch.zeros_like(z), z], dim=1)


def denormalize(images):
    """The transform's ImageNet normalization undone, clamped back to 0-1 RGB."""
    mean = torch.tensor(IMAGENET_MEAN, device=images.device, dtype=images.dtype).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=images.device, dtype=images.dtype).view(1, 3, 1, 1)
    return (images * std + mean).clamp(0.0, 1.0)


def image_foreground(result, images):
    """One ultralytics result as ``[1,H,W]`` foreground probability on the input's grid.

    A pixel is foreground if any instance covers it, so the union is per pixel the
    largest instance probability. No instance at all is background everywhere,
    shaped like the non-empty case so the caller stacks either without a special case.
    """
    h, w = int(images.shape[-2]), int(images.shape[-1])
    masks = getattr(result, "masks", None)
    data = getattr(masks, "data", None) if masks is not None else None
    if data is None or data.numel() == 0:
        return images.new_zeros(1, h, w)
    up = F.interpolate(data.unsqueeze(1).float(), size=(h, w),
                       mode="bilinear", align_corners=False)
    return up.max(dim=0).values


# ---------------------------------------------------------------------------
# Adapters -- the object `evaluate()` calls
# ---------------------------------------------------------------------------

class ForegroundModel:
    """What `evaluate()` calls: `eval()`, then `__call__(images)` -> metric logits.

    A subclass supplies `foreground(images)` -> ``[B,H,W]`` probability. The logit
    round-trip is here, once, so no adapter can have its own.
    """

    def __init__(self, spec, module, device, size=None):
        self.spec = spec
        self.module = module
        self.device = device
        # The size the model is scored at, resolved once by `resolve`: the
        # adapter's native size unless one was asked for, so an adapter runs at
        # the size its result reports and never quietly at its own.
        self.size = int(size if size is not None else spec.native_size)
        self.eval()

    def _network(self):
        return self.module

    def eval(self):
        self._network().eval()
        return self

    def parameters(self):
        return self._network().parameters()

    def __call__(self, images):
        with torch.no_grad():
            p_fg = self.foreground(images)
        if p_fg.dim() != 3:
            raise BaselineError(
                f"{self.spec.name}'s adapter returned {tuple(p_fg.shape)}: `foreground` must "
                f"return [B,H,W], one foreground probability per pixel, and the [B,2,H,W] "
                f"logits the metric takes are built from it here"
            )
        return foreground_logits(p_fg)

    def foreground(self, images):
        raise NotImplementedError


class TorchvisionForeground(ForegroundModel):
    """The model's own decision per pixel: foreground is ``argmax != background``.

    A hard 0/1 probability, so the number says what the model decides rather than
    how confident it is; it goes through the same logit round-trip as every
    adapter and the metric thresholds at 0.5, so its mask is exactly the argmax
    rule and not a second, unstated threshold.
    """

    def foreground(self, images):
        out = self.module(images)["out"]
        return (out.argmax(dim=1) != 0).to(torch.float32)


class YoloForeground(ForegroundModel):
    """The union of the instance masks the package returns, on the input's grid."""

    def _network(self):
        # The YOLO object is not a module; the network it wraps is.
        return self.module.model

    def foreground(self, images):
        # The package wants 0-1 pixels and its own size, both stated here rather
        # than left to a default that could silently letterbox a wrong size.
        results = self.module.predict(source=denormalize(images),
                                      imgsz=self.size, conf=YOLO_CONF,
                                      device=str(self.device), verbose=False)
        if len(results) != int(images.shape[0]):
            raise BaselineError(
                f"{self.spec.name} returned {len(results)} results for a batch of "
                f"{int(images.shape[0])}: scoring it would silently score the wrong images"
            )
        return torch.cat([image_foreground(r, images) for r in results], dim=0)


# ---------------------------------------------------------------------------
# The registry
# ---------------------------------------------------------------------------

class BaselineSpec:
    """One baseline, declared. Constructing it imports nothing."""

    def __init__(self, name, native_size, size_source, wrapper, load, package, install,
                 caveat=None, caveat_note=None, timed_note=""):
        self.name = name
        self.native_size = int(native_size)
        self.size_source = size_source
        self.wrapper = wrapper
        self.load = load
        self.package = package
        self.install = install
        self.caveat = caveat
        self.caveat_note = caveat_note
        # What a bench's timed call actually covers, so a latency next to ours is
        # not read as the same measurement when the package's own pre/post
        # processing sits inside it.
        self.timed_note = timed_note


def _load_yolo(device):
    from ultralytics import YOLO
    return YOLO(YOLO_WEIGHTS)


def _load_torchvision(device):
    # weights=... also sets weights_backbone=None inside torchvision's builder, so
    # the COCO checkpoint is the only file fetched (no ImageNet backbone too).
    from torchvision.models.segmentation import lraspp_mobilenet_v3_large

    return lraspp_mobilenet_v3_large(weights=TORCHVISION_WEIGHTS).to(device)


# yolo11n-seg is COCO instance segmentation at the size the package serves: its
# 80 classes are ours, so it has nothing to declare but its size. It is AGPL-3.0,
# which is why it is called and never vendored.
#
# LRASPP-MobileNetV3-Large is the small model the row asks for beside ours (3.2M
# params); BSD code, but its published weights are trained on COCO with VOC's 21
# labels (torchvision reports their metric as "COCO-val2017-VOC-labels"), so its
# foreground is a different question from ours and the row carries that with it.
BASELINES = {
    "yolo11n-seg": BaselineSpec(
        name="yolo11n-seg",
        native_size=640,
        size_source="adapter yolo11n-seg (native 640)",
        wrapper=YoloForeground,
        load=_load_yolo,
        package="ultralytics",
        install="pip install ultralytics",
        timed_note=("the package's predict() at the scored size: its own preprocessing, NMS "
                    "and mask assembly are inside the timed call"),
    ),
    "torchvision-lraspp": BaselineSpec(
        name="torchvision-lraspp",
        native_size=520,
        size_source="adapter torchvision-lraspp (native 520)",
        wrapper=TorchvisionForeground,
        load=_load_torchvision,
        package="torchvision",
        install="pip install torchvision  # the weights download on first use",
        caveat="voc21-argmax",
        caveat_note=(
            "its weights are trained on COCO with VOC's 21 labels (background + 20 "
            "classes), not COCO's 80 instance classes, and its foreground is the "
            "argmax over those 21 being non-background: this is a different question "
            "from ours, not our question answered by another model"
        ),
        timed_note=("the model's forward at the scored size from a prepared tensor, plus "
                    "the adapter's argmax conversion"),
    ),
}


def names():
    """The baseline names the tool knows, sorted."""
    return sorted(BASELINES)


def get(name):
    """The spec for ``name``, or a refusal naming the ones that exist."""
    try:
        return BASELINES[name]
    except KeyError:
        raise BaselineError(
            f"unknown baseline {name!r}; known baselines: {', '.join(names())}"
        ) from None


def resolve(name, input_size=None):
    """``(spec, size, source)`` for a baseline: native unless one is asked for.

    A baseline is not a checkpoint, so there is no metadata to disagree with: its
    adapter declares the size the model was written for, and that is the default.
    `--input-size` runs it at another one *deliberately* (the same model "as
    published" and at our own resolution), and the source says which of the two
    happened, so no row can be read as having run at a size it did not run at.

    A size nothing can be built at is refused here, as ``resolve_size`` refuses it
    for a checkpoint: the caller resolves sizes before it reads an image, so a
    ``--input-size 0`` stops the run in seconds instead of after the val set is
    read and the model is loaded.
    """
    spec = get(name)
    if input_size is None:
        return spec, spec.native_size, spec.size_source
    size = int(input_size)
    if size <= 0:
        raise BaselineError(f"--input-size must be positive, got {input_size}")
    return (spec, size,
            f"--input-size {size} (adapter {name}'s native size is "
            f"{spec.native_size})")


def build(name, device, size=None):
    """The model object for ``name``, importing its package now and not before."""
    spec = get(name)
    try:
        module = spec.load(device)
    except ImportError as e:
        raise BaselineError(
            f"baseline {name} needs the {spec.package} package, which is not importable "
            f"({e}). Install it with: {spec.install}"
        ) from e
    return spec.wrapper(spec, module, device, size)
