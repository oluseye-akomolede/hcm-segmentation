"""The opt-in COCO train augmentation (openspec seg-aug-and-grad-accum, item 1).

``--aug-scale-jitter LO,HI`` draws each image's short-side resize from [LO,HI]
times the configured size, resizing the mask with it; ``--aug-color-jitter S``
jitters brightness/contrast/saturation by +-S and hue by +-S/4 on the image
only. Both are off by default, and off has to mean *off*: the batch must be
byte-identical to the pre-flag transform at a fixed seed, or every existing
comparison against an un-augmented run would move.

The transform is exercised directly on a synthetic COCOA row, as
test_cocoa_region_guards does, so no dataset is downloaded.
"""

import random

import numpy as np
import pytest
import torch

from ecs_hcm_model.core.model_utils import (
    coco_train_transforms_fg_centered,
    coco_val_transform_example,
    coco_val_transforms,
    get_training_dataloader,
)
from ecs_hcm_model.core.trainer import AugmentConfigError, validate_augmentation

from test_coco_seg_pipeline import _bright_image
from test_cocoa_region_guards import FULL_FRAME, _hf_row_with

SIZE = 512
JITTER = dict(scale_jitter=(0.75, 1.5), color_jitter=0.3)


def _batch(n=2):
    """One map-style COCOA batch: the same two samples every call.

    The image carries a bright square rather than being uniform: a constant
    image survives any resize unchanged, so it could not show what the scale
    jitter does.
    """
    row = _hf_row_with([FULL_FRAME])
    row["image"] = _bright_image([(256, 256)])
    return {k: [row[k] for _ in range(n)] for k in ("image", "annotations", "image_id")}


def _transform(**kw):
    """Run the train transform on that batch at a fixed seed.

    All three streams are pinned: the transform draws the crop window from
    torchvision's RNG (torch), and the flip and the scale factor from python's
    random, so seeding only one of them leaves the batch drifting between calls
    and every equality assertion here would pass or fail for the wrong reason.
    """
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    return coco_train_transforms_fg_centered(_batch(), **kw)


def _pixels(out):
    # The transform returns one tensor per sample in the batch, not a stacked
    # tensor; stacking here is what the collate would do.
    return torch.stack(out["pixel_values"])


def _masks(out):
    return torch.stack(out["segmentation_mask"]) > 0.5


# --------------------------------------------------------------------------
# 1. Off means off
# --------------------------------------------------------------------------

def test_the_defaults_are_no_jitter_and_the_batch_is_unchanged():
    """Flags off: an inert setting is a true no-op, pixels and mask both.

    Colour identity is the strong direction -- 0.0 is falsy, so the colour step
    is skipped outright (no conversion, no quantisation) and the batch has to
    come out byte-identical. An identity *scale* range cannot be tested the same
    way: a range is drawn per sample whenever one is given, the degenerate range
    included, so it consumes one random number per sample and shifts the
    crop/flip stream. That the off path itself draws nothing extra -- and so
    matches the pre-flag code at a fixed seed -- is
    test_the_transform_draws_no_randomness_when_the_flags_are_off.
    """
    off = _transform()
    inert = _transform(color_jitter=0.0)
    assert torch.equal(_pixels(off), _pixels(inert))
    assert torch.equal(_masks(off), _masks(inert))


def test_the_transform_draws_no_randomness_when_the_flags_are_off(monkeypatch):
    """Nothing new is drawn, so the seed's stream is the pre-flag stream.

    The pad/crop/hflip draws that already existed are untouched; what must not
    happen is an extra draw for a jitter that is off. If one were made, every
    downstream draw would shift and the batch would differ from the previous
    code at the same seed -- which is exactly what this rules out.
    """
    calls = []
    real = random.uniform
    monkeypatch.setattr(random, "uniform", lambda *a: (calls.append(a), real(*a))[1])
    _transform()
    assert calls == []


@pytest.mark.parametrize("kw,what", [
    (dict(scale_jitter=(0.75, 1.5)), "scale"),
    (dict(color_jitter=0.3), "colour"),
])
def test_the_flag_changes_the_batch(kw, what):
    """Each flag on its own has an effect, or the tests above prove nothing."""
    assert not torch.equal(_pixels(_transform(**kw)), _pixels(_transform())), what


def test_the_scale_range_is_the_one_that_is_drawn(monkeypatch):
    """One draw per sample, from exactly the requested range."""
    calls = []
    real = random.uniform
    monkeypatch.setattr(random, "uniform", lambda *a: (calls.append(a), real(*a))[1])
    out = _transform(scale_jitter=(0.75, 1.5))
    n = _pixels(out).shape[0]
    assert calls == [(0.75, 1.5)] * n


# --------------------------------------------------------------------------
# 2. What the flags must not do
# --------------------------------------------------------------------------

def test_the_colour_jitter_leaves_the_mask_alone():
    """Colour is an image-only augmentation: the target must not move.

    Asserted structurally rather than by value: with the scale jitter off, the
    crop is identical, so the mask has to come out byte-identical to the
    un-jittered batch while the pixels do not.
    """
    plain = _transform()
    tinted = _transform(color_jitter=0.3)
    assert torch.equal(_masks(plain), _masks(tinted))
    assert not torch.equal(_pixels(plain), _pixels(tinted))


def test_the_mask_stays_binary_and_aligned_with_the_image():
    out = _transform(**JITTER)
    pixels = _pixels(out)
    raw_masks = torch.stack(out["segmentation_mask"])
    assert pixels.shape[0] == raw_masks.shape[0]
    assert tuple(pixels.shape[-2:]) == tuple(raw_masks.shape[-2:])
    assert set(torch.unique(raw_masks).tolist()) <= {0, 1}


def test_the_val_transforms_take_no_jitter():
    """Val is the measurement, so it must never be jittered.

    A source check because the transforms are imported by name all over the
    trainer: if a jitter kwarg were added to the val path, a caller could pass
    one and silently make every val score incomparable.
    """
    for fn in (coco_val_transforms, coco_val_transform_example):
        code = fn.__code__
        # Keyword-only arguments sit past co_argcount, so both counts are needed
        # or a jitter added as a keyword-only parameter would slip through.
        names = code.co_varnames[: code.co_argcount + code.co_kwonlyargcount]
        assert not any("jitter" in name for name in names), f"{fn.__name__} {names}"


# --------------------------------------------------------------------------
# 3. The settings are refused up front where they cannot mean anything
# --------------------------------------------------------------------------

@pytest.mark.parametrize("kw", [
    dict(scale_jitter=(1.5, 0.75)),   # LO above HI
    dict(scale_jitter=(-1.0, 1.0)),   # non-positive
    dict(scale_jitter=(1.0, float("inf"))),
    dict(color_jitter=-0.1),
])
def test_a_jitter_that_cannot_mean_anything_is_a_usage_error(kw):
    with pytest.raises(AugmentConfigError):
        validate_augmentation(kw.get("scale_jitter"), kw.get("color_jitter"), trains_coco=True)


def test_jitter_on_the_imagenet_path_is_refused_not_ignored():
    """The jitter exists only in the COCO train transform.

    An ImageNet run would accept the flag and silently train without it, so the
    combination is a usage error rather than a no-op.
    """
    with pytest.raises(AugmentConfigError, match="COCO"):
        validate_augmentation((0.75, 1.5), None, trains_coco=False)


def test_the_validator_returns_what_it_validated():
    assert validate_augmentation((0.75, 1.5), 0.3, trains_coco=True) == ((0.75, 1.5), 0.3)
    assert validate_augmentation(None, None, trains_coco=False) == (None, None)


# --------------------------------------------------------------------------
# 4. The streaming paths cannot take the jitter
# --------------------------------------------------------------------------

@pytest.mark.parametrize("dataset_name,local", [
    ("imagenet-1k", False),   # the ImageNet transform has no jitter
    ("coco-seg", True),       # the local generator path makes its own batches
], ids=["imagenet", "local-generator"])
def test_the_loader_refuses_the_jitter_where_it_cannot_apply_it(dataset_name, local):
    """Refused, not ignored: a run that asked for augmentation must not silently
    train without it.

    The trainer checks the COCO/COCOA combination up front (see
    test_jitter_on_the_imagenet_path_is_refused_not_ignored); this is the
    backstop in the loader itself, for any caller that reaches it directly.
    """
    with pytest.raises(ValueError, match="only implemented for the map-style"):
        get_training_dataloader(dataset_name, is_dataset_local=local,
                                scale_jitter=(0.75, 1.5), workers=0)
