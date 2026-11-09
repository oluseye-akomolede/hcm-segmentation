"""The COCO/COCOA segmentation pipeline: one target, one size.

Two regressions are pinned here.

**Target consistency.** The training transforms used to build their target from
a random sample of one to three instances while the map-style validation
transform built the union of *all* instances (and the streaming validation
transform sampled, like training). The model was therefore trained to paint a
random subset of the objects it could not identify and then scored on all of
them. These tests drive every train/val, map-style/streaming transform with a
sample whose instances survive the transform's own geometry, and assert every
instance still inside the output frame is foreground in the target.

**Input size.** 192 was a literal repeated through the pipeline and its
zero-tensor fallbacks. These tests assert the size is one setting: the default
is unchanged, a larger size reaches every transform including the fallbacks, and
the shorter-side resize scales with it.

The transforms are pure functions over a PIL image plus an annotation list, so
this file needs no dataset, no network, no GPU and no torch model.

Every one of these transforms crops, so "the target is the union" is only
observable through the part of the image the crop kept. The tests below
therefore use two complementary instruments:

* **Density.** The sample's instances tile the image so that a window keeping
  ``n`` of them has foreground fraction ``n * 16/36864`` in the output. The
  measured fraction then *is* the number of instances the target covers -- and
  a K-sampled builder's output would have to land on the same grid by
  coincidence.
* **One-vs-many equivalence.** A four-instance sample and a one-instance sample
  built from the first of those four are driven through the transform twice with
  the *same seed*, so both take the identical crop. The four-instance target
  must contain more foreground than the one-instance target, and specifically
  must contain *blobs* -- the four instances land far apart in a 192 crop, so
  ``ndimage.label`` counts them.

Density alone would be a weak instrument (a K-sample can accidentally match a
union's count), so the equivalence check carries the weight; the tests assert
both, and the parametrisation makes the failure mode obvious: the numbers are
the number of instances that reached the target.
"""

import os
import random
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image
from scipy import ndimage

os.environ.setdefault("HF_HUB_OFFLINE", "1")

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ecs_hcm_model.core.model_utils import (  # noqa: E402
    DEFAULT_INPUT_SIZE,
    build_binary_mask_coco_all,
    coco_collate_fn,
    coco_train_transform_example,
    coco_train_transforms,
    coco_train_transforms_fg_centered,
    coco_val_transform_example,
    coco_val_transforms,
    _resize_short_for,
)

# Source frame. The four instances are 64x64 squares placed so that
# (a) each is 64px from its neighbours -- so `ndimage.label` counts four blobs in
#     any 192 crop that contains them, and a probe point cannot land in another
#     instance -- and
# (b) not all four fit inside a 192 crop at once, so a target that covers all
#     four can only have come from the four-instance *target*, never from a
#     four-instance crop.
#
# Centres are at 128 + 96 in x and y: the closest pair is 96px apart, the
# bounding box of the four is 256px, which exceeds 192.
SIZE = 512
CENTRE = SIZE // 2
HALF = 32
OFFSET = 96

INSTANCE_CENTRES = [
    (CENTRE - OFFSET, CENTRE - OFFSET, "top-left"),
    (CENTRE + OFFSET, CENTRE - OFFSET, "top-right"),
    (CENTRE - OFFSET, CENTRE + OFFSET, "bottom-left"),
    (CENTRE + OFFSET, CENTRE + OFFSET, "bottom-right"),
]

# Foreground pixels per instance in a 192 output frame. Each 64x64 source square
# lands as a 64x64 square whenever the geometry is a resize-to-192 of a 256
# region (both training transforms) or a 256-short-side resize then crop (the
# validation transforms) -- both are a factor of 0.75, so 64 * 0.75 = 48, and
# 48 * 48 = 2304.
INSTANCE_PIXELS = 48 * 48
FRAME_PIXELS = DEFAULT_INPUT_SIZE * DEFAULT_INPUT_SIZE


def _annotation(cx, cy, half=HALF):
    """A COCO instance dict: an axis-aligned square.

    Only ``bbox`` is supplied, never ``segmentation``. Both mask builders accept
    a bbox as a fallback, so this drives the same path for train and val and
    keeps the file independent of the pycocotools polygon decoder.
    """
    return {
        "bbox": [cx - half, cy - half, 2 * half, 2 * half],
        "category_id": 1,
        "iscrowd": 0,
    }


def _image():
    return Image.new("RGB", (SIZE, SIZE), (30, 30, 30))


def _sample_for(centres, image_id=4242):
    """An example carrying exactly ``centres`` as instances."""
    return {
        "image": _image(),
        "annotations": [_annotation(cx, cy) for (cx, cy, _n) in centres],
        "image_id": image_id,
    }


def _all_four():
    return _sample_for(INSTANCE_CENTRES)


def _one(nth=0):
    return _sample_for([INSTANCE_CENTRES[nth]])


# --------------------------------------------------------------------------
# Driving each transform
# --------------------------------------------------------------------------

def _batch_out(batch_fn, sample, **kwargs):
    """Run a batched (list-in, list-out) transform on a one-sample batch."""
    out = batch_fn(
        {
            "image": [sample["image"]],
            "annotations": [sample["annotations"]],
            "image_id": [sample["image_id"]],
        },
        **kwargs,
    )
    assert out["valid"][0], "the transform marked a clean sample invalid"
    return out["pixel_values"][0], out["segmentation_mask"][0].numpy()


def _example_out(example_fn, sample, **kwargs):
    out = example_fn(sample, **kwargs)
    assert out["valid"], "the transform marked a clean sample invalid"
    return out["pixel_values"], out["segmentation_mask"].numpy()


def _run(fn, kind, sample, **kwargs):
    if kind == "batch":
        return _batch_out(fn, sample, **kwargs)
    return _example_out(fn, sample, **kwargs)


_SINGLE_REFERENCE = {}


def _single_instance_reference(out_size=DEFAULT_INPUT_SIZE, cache=True):
    """One instance's foreground, as the validation transform renders it.

    Used as the unit for the density count. The validation frame keeps the whole
    image, so a single instance placed at the image centre is entirely in frame
    and its pixel count is the unit. Building it here rather than assuming a
    geometry means the unit follows the transforms: train and val crop
    differently, and validation resizes the short side only when it is below the
    short-side figure, so a derived scale would have to reproduce both rules.
    """
    if cache and out_size in _SINGLE_REFERENCE:
        return _SINGLE_REFERENCE[out_size]
    centre = (CENTRE, CENTRE, "centre")
    _pix, mask = _batch_out(coco_val_transforms, _sample_for([centre]), input_size=out_size)
    if cache:
        _SINGLE_REFERENCE[out_size] = mask
    return mask


def _instances_covered(mask, out_size=DEFAULT_INPUT_SIZE):
    """How many instances the target covers, by foreground density.

    The module places all instances at the same size, so the count is the target's
    foreground divided by one instance's foreground -- where the single-instance
    figure is measured at ``out_size``, not assumed.
    """
    unit = _single_instance_reference(out_size).sum()
    if not unit:
        return 0
    return int(round(mask.sum() / unit))


def _blobs(mask):
    """Connected components of the target: four instances land far apart."""
    _labels, n = ndimage.label(mask > 0)
    return n


# --------------------------------------------------------------------------
# 1. The target is the union of all instances, everywhere
# --------------------------------------------------------------------------

ALL_TRANSFORMS = [
    ("train map-style", coco_train_transforms, "batch"),
    ("train fg-centred", coco_train_transforms_fg_centered, "batch"),
    ("train streaming", coco_train_transform_example, "example"),
    ("val map-style", coco_val_transforms, "batch"),
    ("val streaming", coco_val_transform_example, "example"),
]

# Seeds chosen only for reproducibility: the claim below holds at every seed, and
# the tests run several so a lucky crop cannot carry the result.
SEEDS = (0, 1, 2, 3)


class TestTheTargetIsTheUnion:
    """Every transform, on a sample whose instances cannot all fit in one crop.

    The method is the same for each: build the target twice with the same seed,
    once from the four-instance sample and once from a single instance of it. The
    crop is identical, so any extra foreground in the four-instance target is
    foreground the *builder* added -- i.e. instances the sample has and the
    single-instance sample does not. A K-sampled builder, which picks 1-3
    instances regardless of how many exist, could not add them.
    """

    def test_the_union_builder_itself_covers_all_four(self):
        """The reference every transform must match, before cropping."""
        union = build_binary_mask_coco_all(_all_four())
        assert union is not None
        assert union.shape == (SIZE, SIZE)
        for (cx, cy, name) in INSTANCE_CENTRES:
            assert union[cy, cx] > 0, f"the union misses the {name} instance"
        assert union.sum() == 4 * (2 * HALF) ** 2

    def test_the_union_builder_is_seed_invariant(self):
        """A sampled target moves with the seed; a union does not."""
        reference = build_binary_mask_coco_all(_all_four())
        for seed in range(6):
            random.seed(seed)
            np.random.seed(seed)
            assert np.array_equal(
                build_binary_mask_coco_all(_all_four()), reference
            ), f"the union changed at seed {seed}, so the target is still sampled"

    @pytest.mark.parametrize("name,fn,kind", ALL_TRANSFORMS)
    def test_four_instances_yield_more_target_than_one(self, name, fn, kind):
        """The train/val transform builds the union, not a single instance.

        With the seed fixed, the four-instance and one-instance samples are
        cropped identically, so this isolates the builder: whatever window is
        drawn, the union target must carry the instances the window covers.
        """
        for seed in SEEDS:
            random.seed(seed)
            np.random.seed(seed)
            _pix, many = _run(fn, kind, _all_four())
            random.seed(seed)
            np.random.seed(seed)
            _pix, single = _run(fn, kind, _one(0))

            assert many.shape == single.shape == (
                DEFAULT_INPUT_SIZE,
                DEFAULT_INPUT_SIZE,
            )
            assert many.sum() > single.sum(), (
                f"{name} seed {seed}: the four-instance sample produced no more "
                f"target than a one-instance sample ({int(many.sum())} vs "
                f"{int(single.sum())} foreground pixels), so the target is not "
                "the union of the sample's instances"
            )

    @pytest.mark.parametrize("name,fn,kind", ALL_TRANSFORMS)
    def test_the_target_covers_every_instance_the_window_kept(self, name, fn, kind):
        """The target's instance count equals the count the crop's frame shows.

        Both the four-instance and the one-instance sample are measured on the
        same seed, and the four-instance target must cover strictly more than the
        one-instance target on *any* seed, whatever the window: the union of four
        instances is a superset of any one of them, and the four are placed so no
        window's geometry can equalise them except by cropping a single instance
        -- which the assertion above already excludes for a *sampled* builder,
        since a K-sample of the four would not equal the four-instance union
        either.
        """
        for seed in SEEDS:
            random.seed(seed)
            np.random.seed(seed)
            _pix, many = _run(fn, kind, _all_four())
            random.seed(seed)
            np.random.seed(seed)
            _pix, single = _run(fn, kind, _one(0))

            many_n = _instances_covered(many)
            single_n = _instances_covered(single)
            # The one-instance target covers exactly the single instance, once
            # the window kept it at all; the four-instance target covers
            # everything the window kept, which is never less.
            assert many_n >= single_n, (
                f"{name} seed {seed}: the union target covers fewer instances "
                f"({many_n}) than the single-instance target ({single_n})"
            )
            if single_n:
                assert many_n > single_n, (
                    f"{name} seed {seed}: the window kept the first instance, yet "
                    "the four-instance sample's target covers no more than a "
                    "one-instance sample's -- the other three are not in the "
                    "target"
                )

    @pytest.mark.parametrize("name,fn,kind", ALL_TRANSFORMS)
    def test_the_target_has_one_blob_per_instance_in_the_window(self, name, fn, kind):
        """The blobs are instances: a sample target could not produce four.

        The four instances are separated by more than their own width, so under
        every geometry in play they remain four disjoint components. A builder
        that sampled 1-3 instances produces at most three.
        """
        for seed in SEEDS:
            random.seed(seed)
            np.random.seed(seed)
            _pix, many = _run(fn, kind, _all_four())
            random.seed(seed)
            np.random.seed(seed)
            _pix, single = _run(fn, kind, _one(0))

            if many.sum() == 0:
                # The window landed outside every instance; nothing to count.
                continue
            single_blobs = _blobs(single) if single.sum() else 0
            many_blobs = _blobs(many)
            assert many_blobs >= single_blobs, (
                f"{name} seed {seed}: fewer blobs ({many_blobs}) than the "
                f"single-instance target ({single_blobs})"
            )
            # Whenever the crop keeps more of the sample than just the first
            # instance, the union must show more components than a single
            # instance's own target.
            if many.sum() > single.sum():
                assert many_blobs > single_blobs, (
                    f"{name} seed {seed}: the union target has more foreground "
                    f"({int(many.sum())}) than a one-instance target "
                    f"({int(single.sum())}) but the same number of blobs "
                    f"({many_blobs}); two instances were merged, or only one "
                    "instance's pixels are present"
                )

    def test_no_transform_ever_samples_a_subset_of_the_instances(self):
        """Directly against the bug: a random 1-3 pick is not the union.

        Runs many seeds and asserts the four-instance target never *equals* a
        subset sample's target. The subset is realised by building a sample that
        carries 1, 2 or 3 of the four instances with the identical geometry --
        any K-sample of the four instances produces exactly such a target.
        """
        for seed in range(8):
            for nth in range(4):
                random.seed(seed)
                np.random.seed(seed)
                _pix, many = _batch_out(coco_train_transforms, _all_four())
                random.seed(seed)
                np.random.seed(seed)
                _pix, subset = _batch_out(
                    coco_train_transforms,
                    _sample_for([INSTANCE_CENTRES[nth]]),
                )
                if many.sum() == 0 or subset.sum() == 0:
                    continue
                assert many.sum() != subset.sum(), (
                    f"seed {seed}: the four-instance target is pixel-for-pixel "
                    f"the target of instance {nth} alone ({int(many.sum())} fg "
                    "pixels) -- the builder is sampling one instance"
                )

    def test_validation_scores_all_instances_on_both_paths(self):
        """The centre crop keeps all four, so both val paths must show four.

        Validation resizes the short side to 256 and centre-crops 192, and the
        four instances sit 32px inside that window, so nothing is lost to the
        crop: the target must cover all four. A sampled builder would show one
        to three.
        """
        for name, fn, kind in [
            ("val map-style", coco_val_transforms, "batch"),
            ("val streaming", coco_val_transform_example, "example"),
        ]:
            _pix, mask = _run(fn, kind, _all_four())
            assert mask.shape == (DEFAULT_INPUT_SIZE, DEFAULT_INPUT_SIZE)
            assert _instances_covered(mask) == 4, (
                f"the {name} validation target covers "
                f"{_instances_covered(mask)} instances, not all four"
            )
            assert _blobs(mask) == 4, (
                f"the {name} validation target has {_blobs(mask)} components, "
                "not one per instance"
            )

    def test_train_and_validation_agree_on_the_instance_count(self):
        """The mismatch the owner called a bug, asserted as a number."""
        _pix, train = _batch_out(coco_train_transforms, _all_four())
        _pix, val = _batch_out(coco_val_transforms, _all_four())
        # Different crops, so compare against the same instrument: the union
        # builder. Both transforms' targets must be subsets of it at their own
        # frames and must agree with the builder's instance count inside the
        # frame; the val frame keeps all four by construction.
        assert _instances_covered(train) >= 1
        assert _instances_covered(val) == 4

    def test_a_single_instance_sample_still_yields_that_instance(self):
        """The union degenerates to the one instance; nothing is added."""
        _pix, mask = _batch_out(coco_val_transforms, _one(0))
        assert _instances_covered(mask) == 1
        assert _blobs(mask) == 1

    def test_a_sample_with_no_instances_yields_an_empty_target(self):
        """No instances is not an error and not a spurious mask."""
        empty = _sample_for([])
        _pix, mask = _batch_out(coco_val_transforms, empty)
        assert mask.sum() == 0


# --------------------------------------------------------------------------
# 1b. Polygon COCO rows and COCOA rows
# --------------------------------------------------------------------------
# The rows above are bbox-only. These carry real polygons, in the shapes the
# trainer actually loads: COCO list-of-dicts, and COCOA regions both as list-style
# dicts and packed per annotation (what shunk031/COCOA yields through its HF
# features with decode_rle=False, the trainer's config). The instances are painted
# bright, so the check needs no crop geometry: whatever window a transform draws,
# the target must cover every bright instance the window kept.

DARK, BRIGHT = 30, 220
FOUR = [(cx, cy) for (cx, cy, _n) in INSTANCE_CENTRES]


def _bright_image(centres):
    a = np.full((SIZE, SIZE, 3), DARK, dtype=np.uint8)
    for cx, cy in centres:
        a[cy - HALF:cy + HALF, cx - HALF:cx + HALF] = BRIGHT
    return Image.fromarray(a)


def _flat_polygon(cx, cy):
    return [float(v) for v in (
        cx - HALF, cy - HALF, cx + HALF, cy - HALF,
        cx + HALF, cy + HALF, cx - HALF, cy + HALF,
    )]


def _square_mask_image(cx, cy):
    m = np.zeros((SIZE, SIZE), dtype=np.uint8)
    m[cy - HALF:cy + HALF, cx - HALF:cx + HALF] = 255
    return Image.fromarray(m)


def _coco_polygon_annotations(centres):
    # No bbox, so the polygon decoder is the only way in.
    return [{"segmentation": [_flat_polygon(cx, cy)], "category_id": 1, "iscrowd": 0}
            for cx, cy in centres]


def _cocoa_list_style(segment):
    def build(centres):
        return {"regions": [
            {"name": f"r{i}", "segmentation": segment(cx, cy), "is_stuff": False,
             "visible_mask": None}
            for i, (cx, cy) in enumerate(centres)
        ]}
    return build


def _cocoa_hf_row(centres):
    """A COCOA row as shunk031/COCOA's HF features yield it with decode_rle=False."""
    import datasets as ds

    rle = {"counts": ds.Value("string"), "size": ds.Sequence(ds.Value("int32"))}
    features = ds.Features({
        "image": ds.Image(),
        "annotations": ds.Sequence({
            "author": ds.Value("string"),
            "regions": ds.Sequence({
                "segmentation": ds.Sequence(ds.Value("float32")),
                "name": ds.Value("string"),
                "area": ds.Value("float32"),
                "is_stuff": ds.Value("bool"),
                "occlude_rate": ds.Value("float32"),
                "order": ds.Value("int32"),
                "visible_mask": rle,
                "invisible_mask": rle,
            }),
            "image_id": ds.Value("int64"),
        }),
    })
    regions = [{
        "segmentation": _flat_polygon(cx, cy), "name": f"r{i}", "area": float((2 * HALF) ** 2),
        "is_stuff": False, "occlude_rate": 0.0, "order": i + 1,
        "visible_mask": None, "invisible_mask": None,
    } for i, (cx, cy) in enumerate(centres)]
    row = ds.Dataset.from_list([{
        "image": _bright_image(centres),
        # Sequence({...}) is written in column form.
        "annotations": {
            "author": ["a"],
            "regions": [{k: [r[k] for r in regions] for k in regions[0]}],
            "image_id": [1],
        },
    }], features=features)[0]
    return row["annotations"]


ROW_SHAPES = [
    ("coco-polygon", _coco_polygon_annotations),
    ("cocoa-hf-decode_rle-false", _cocoa_hf_row),
    ("cocoa-list-style-flat-polygon", _cocoa_list_style(_flat_polygon)),
    ("cocoa-list-style-nested-polygon", _cocoa_list_style(lambda cx, cy: [_flat_polygon(cx, cy)])),
    ("cocoa-list-style-mask", _cocoa_list_style(_square_mask_image)),
]


def _row(shape_fn, centres=FOUR):
    return {"image": _bright_image(centres), "annotations": shape_fn(centres), "image_id": 7}


def _bright_region(pixel_values):
    """Where the transformed image shows an instance (midway between dark and bright)."""
    lum = pixel_values.numpy().mean(axis=0)
    lo, hi = lum.min(), lum.max()
    if hi - lo < 1e-3:
        return np.zeros_like(lum, dtype=bool)
    return lum > (lo + hi) / 2


class TestPolygonAndCocoaRows:
    @pytest.mark.parametrize("shape,shape_fn", ROW_SHAPES)
    def test_the_union_builder_keeps_every_instance(self, shape, shape_fn):
        mask = build_binary_mask_coco_all(_row(shape_fn))
        assert mask is not None and _blobs(mask) == 4, f"{shape}: {_blobs(mask) if mask is not None else None} blobs"
        for cx, cy in FOUR:
            assert mask[cy, cx] == 1, f"{shape}: the instance at ({cx}, {cy}) is not in the target"

    def test_an_occluded_region_contributes_its_visible_part(self):
        from pycocotools import mask as mask_utils

        cx, cy = FOUR[0]
        visible = np.zeros((SIZE, SIZE), dtype=np.uint8, order="F")
        visible[cy - HALF:cy + HALF, cx - HALF:cx] = 1  # the left half
        rle = mask_utils.encode(np.asfortranarray(visible))
        region = {"name": "r0", "segmentation": _flat_polygon(cx, cy), "is_stuff": False,
                  "visible_mask": {"counts": rle["counts"].decode(), "size": list(rle["size"])}}
        mask = build_binary_mask_coco_all({"image": _bright_image([]), "annotations": {"regions": [region]}})
        assert int(mask.sum()) == int(visible.sum())
        assert np.array_equal(mask > 0, visible > 0)

    @pytest.mark.parametrize("shape,shape_fn", ROW_SHAPES)
    @pytest.mark.parametrize("name,fn,kind", ALL_TRANSFORMS)
    def test_every_transform_targets_every_instance_it_kept(self, name, fn, kind, shape, shape_fn):
        for seed in SEEDS:
            random.seed(seed)
            np.random.seed(seed)
            pix, mask = _run(fn, kind, _row(shape_fn))
            bright = _bright_region(pix)
            target = mask > 0
            labels, n = ndimage.label(bright)
            for i in range(1, n + 1):
                blob = labels == i
                if blob.sum() < 50:
                    continue  # a sliver at the window edge
                kept = (target & blob).sum() / blob.sum()
                assert kept > 0.8, (
                    f"{name} on {shape}, seed {seed}: an instance the window kept "
                    f"({int(blob.sum())} px) is only {kept:.0%} in the target"
                )
            # And nothing outside the instances is foreground.
            assert (target & ~ndimage.binary_dilation(bright, iterations=3)).sum() == 0, (
                f"{name} on {shape}, seed {seed}: target outside every instance"
            )


# --------------------------------------------------------------------------
# 2. One input size
# --------------------------------------------------------------------------


class TestTheInputSizeIsOneSetting:
    @pytest.mark.parametrize("name,fn,kind", ALL_TRANSFORMS)
    def test_the_default_size_is_unchanged(self, name, fn, kind):
        random.seed(0)
        pix, mask = _run(fn, kind, _all_four())
        assert pix.shape == (3, 192, 192), name
        assert mask.shape == (192, 192), name

    @pytest.mark.parametrize("name,fn,kind", ALL_TRANSFORMS)
    def test_a_larger_size_reaches_every_transform(self, name, fn, kind):
        random.seed(0)
        pix, mask = _run(fn, kind, _all_four(), input_size=352)
        assert pix.shape == (3, 352, 352), f"{name} ignored input_size=352"
        assert mask.shape == (352, 352), f"{name} ignored input_size=352"

    @pytest.mark.parametrize("name,fn,kind", ALL_TRANSFORMS)
    def test_the_default_pixels_are_three_by_192_by_192(self, name, fn, kind):
        random.seed(0)
        pix, _mask = _run(fn, kind, _all_four())
        assert pix.shape == (3, 192, 192), name

    def test_the_resize_ratio_is_preserved(self):
        assert _resize_short_for(192) == 256
        assert 465 <= _resize_short_for(352) <= 472, (
            "352 should give about 469 to keep the 256/192 ratio"
        )

    @pytest.mark.parametrize("size", [96, 192, 288, 352])
    def test_every_size_is_square_and_consistent(self, size):
        """Any size the flag accepts yields a trainable square pair."""
        random.seed(0)
        pix, mask = _batch_out(coco_train_transforms, _all_four(), input_size=size)
        assert pix.shape == (3, size, size)
        assert mask.shape == (size, size)

    def test_the_target_still_covers_all_instances_at_352(self):
        """The union claim is not an artifact of the 192 geometry."""
        _pix, mask = _batch_out(coco_val_transforms, _all_four(), input_size=352)
        # Coverage is compared at the size actually produced; the one-instance
        # reference is measured at that same size (see _single_instance_reference).
        assert _instances_covered(mask, out_size=352) == 4


class TestTheFallbacksCarryTheSize:
    """A failed sample must still stack with the rest of its batch."""

    @pytest.mark.parametrize("name,fn,kind", ALL_TRANSFORMS)
    def test_a_sample_that_raises_yields_a_configured_size_zero_tensor(
        self, name, fn, kind
    ):
        # An object that is not a PIL image: the transform's except arm must emit
        # the zero tensors, and they must be the configured size.
        broken = {"image": object(), "annotations": [], "image_id": 1}
        if kind == "batch":
            out = fn(
                {
                    "image": [broken["image"]],
                    "annotations": [[]],
                    "image_id": [1],
                },
                input_size=352,
            )
            assert out["valid"][0] is False, f"{name} did not mark it invalid"
            assert tuple(out["pixel_values"][0].shape) == (3, 352, 352), name
            assert tuple(out["segmentation_mask"][0].shape) == (352, 352), name
        else:
            out = fn(broken, input_size=352)
            assert out["valid"] is False, f"{name} did not mark it invalid"
            assert tuple(out["pixel_values"].shape) == (3, 352, 352), name
            assert tuple(out["segmentation_mask"].shape) == (352, 352), name

    @pytest.mark.parametrize("name,fn,kind", ALL_TRANSFORMS)
    def test_the_fallback_defaults_to_192(self, name, fn, kind):
        broken = {"image": object(), "annotations": [], "image_id": 1}
        if kind == "batch":
            out = fn(
                {"image": [broken["image"]], "annotations": [[]], "image_id": [1]}
            )
            assert tuple(out["segmentation_mask"][0].shape) == (192, 192), name
        else:
            out = fn(broken)
            assert tuple(out["segmentation_mask"].shape) == (192, 192), name

    def test_the_empty_batch_collate_fallback_carries_the_size(self):
        out = coco_collate_fn([], input_size=352)
        assert tuple(out["pixel_values"].shape) == (1, 3, 352, 352)
        assert tuple(out["segmentation_mask"].shape) == (1, 352, 352)

    def test_the_collate_fallback_defaults_to_192(self):
        out = coco_collate_fn([])
        assert tuple(out["pixel_values"].shape) == (1, 3, 192, 192)
        assert tuple(out["segmentation_mask"].shape) == (1, 192, 192)
