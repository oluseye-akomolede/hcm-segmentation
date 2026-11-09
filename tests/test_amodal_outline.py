"""The amodal outline channel: head, target, geometry, loss, metrics, loading.

One channel beside the segmentation head predicts the *full* extent of every
object the visible mask marks. The claims this file pins:

- the default model is byte-for-byte the pre-change one: no head, one return
  value, and a state dict the old loaders accept;
- the flag adds exactly two keys, ``outline_head.weight/bias``, and the stored
  flag and the weights agree or the load refuses;
- the outline target is the union of the kept regions' full polygons ORed with
  the visible target, so it covers the visible mask on every sample, and a
  sample whose full labels are incomplete is *unknown*, never quietly partial;
- every augmentation moves the outline with the visible mask by the same
  parameters, so the cover still holds after the crop and the flip;
- the loss trains only on labelled samples and still reaches the head when none
  is labelled, and the metrics score the head against the visible prediction's
  own baselines.

CPU-only and offline (the resnet50 build is stubbed as in ``test_hcm_v2_arch``).
"""

import gc

import numpy as np
import pytest
import torch
from PIL import Image
from torch import nn

from ecs_hcm_model.core import hcm_nn as hcm_mod
from ecs_hcm_model.core.hcm_nn import (
    ArchMismatchError,
    check_amodal_outline,
    infer_amodal_outline,
)
from ecs_hcm_model.core.model_upload_download import build_model_from_info
from ecs_hcm_model.core.model_utils import (
    _arch_kwargs,
    build_binary_mask_coco_all,
    coco_collate_fn,
    coco_train_transform_example,
    coco_train_transforms,
    coco_train_transforms_fg_centered,
    coco_val_transform_example,
    coco_val_transforms,
    warm_start_state,
)
from ecs_hcm_model.core.trainer import (
    evaluate,
    merge_mixed_val_metrics,
    outline_loss_from_logits,
    outline_metrics_from_logits,
    require_outline_labels,
)
from ecs_hcm_model.eval.seg import load_checkpoint as load_eval_checkpoint
from jetson.demo_model import build_demo_model
from jetson.model_shape import ModelShapeError, resolve_amodal_outline

from conftest import TinySegmenter, make_segmentation_batches

pytestmark = pytest.mark.filterwarnings("ignore::UserWarning")


# ---------------------------------------------------------------------------
# Helpers


def _offline(build):
    """Build resnet50 without reaching the network (as test_hcm_v2_arch does)."""
    from torchvision import models as tv_models

    real = tv_models.resnet50

    def offline_resnet50(weights=None, replace_stride_with_dilation=None):
        return real(weights=None, replace_stride_with_dilation=replace_stride_with_dilation)

    tv_models.resnet50 = offline_resnet50
    try:
        return build()
    finally:
        tv_models.resnet50 = real


def _model(**kwargs):
    kwargs.setdefault("enable_finetuning", True)
    kwargs.setdefault("num_segmentation_classes", 2)
    kwargs.setdefault("out_stride", 16)
    kwargs.setdefault("backbone_pretrained", False)
    torch.manual_seed(0)
    return _offline(lambda: hcm_mod.hcm_nn(**kwargs)).eval()


def _keys(model):
    return set(model.state_dict())


def _keys_of(**kwargs):
    """The key set of one model, freed before the caller builds another.

    Two live resnet50s do not fit in the 4 GiB virtual-memory cap these tests
    run under (torch's CPU arena reserves address space, not just weights), so
    every comparison here holds one model at a time.
    """
    model = _model(**kwargs)
    try:
        return set(model.state_dict())
    finally:
        del model
        gc.collect()


def _state(**kwargs):
    """A model's state dict, with the model freed and the record alongside."""
    model = _model(**kwargs)
    try:
        return model.state_dict(), model.arch_record()
    finally:
        del model
        gc.collect()


# A COCOA-ish row in the list form the builder reads directly: ``segmentation``
# is the amodal polygon, ``visible_mask`` (a PIL mask) what the camera/target
# saw. Bright on dark so a wrong window is visible, not a tie.
SIZE = 128
DARK, BRIGHT = 30, 220
FULL_BOX = (16, 16, 112, 112)      # x0, y0, x1, y1 -- the amodal extent
VISIBLE_BOX = (48, 48, 80, 80)     # the part that was not occluded


def _box_polygon(box):
    x0, y0, x1, y1 = box
    return [float(v) for v in (x0, y0, x1, y0, x1, y1, x0, y1)]


def _box_mask(box):
    m = np.zeros((SIZE, SIZE), np.uint8)
    x0, y0, x1, y1 = box
    m[y0:y1, x0:x1] = 255
    return Image.fromarray(m)


def _cocoa_row(full_box=FULL_BOX, visible_box=VISIBLE_BOX, visible=True,
               segmentation=True):
    """One regulation-size COCOA row, image and annotations."""
    img = np.full((SIZE, SIZE, 3), DARK, np.uint8)
    region = {"name": "r0", "is_stuff": False,
              "segmentation": _box_polygon(full_box) if segmentation else None,
              "visible_mask": _box_mask(visible_box) if visible else None}
    return {"image": Image.fromarray(img),
            "annotations": {"regions": [region]}, "image_id": 7}


def _as_array(x):
    return np.asarray(x.detach().cpu() if torch.is_tensor(x) else x)


def _mask2d(x):
    """A per-sample target as [H,W], whatever channel axis the path added."""
    a = _as_array(x)
    return a.reshape(a.shape[-2], a.shape[-1])


# ---------------------------------------------------------------------------
# 1. The head


class TestTheHead:
    def test_the_default_model_has_none_and_returns_one_value(self):
        model = _model()
        assert model.outline_head is None
        assert not any(k.startswith("outline_head.") for k in _keys(model))
        assert infer_amodal_outline(model.state_dict()) is False
        assert model.arch_record()["amodal_outline"] is False
        out = model(torch.randn(1, 3, 64, 64))
        assert torch.is_tensor(out) and out.shape[1] == 2

    def test_the_flag_adds_exactly_two_keys(self):
        plain = _keys_of()
        outl = _keys_of(amodal_outline=True)
        assert outl - plain == {"outline_head.weight", "outline_head.bias"}
        assert plain - outl == set()
        weights, record = _state(amodal_outline=True)
        assert infer_amodal_outline(weights) is True
        assert record["amodal_outline"] is True
        # The storage names the loader reads, spelled by the module itself.
        assert sorted(k for k in weights if k.startswith("outline_head.")) == \
            ["outline_head.bias", "outline_head.weight"]

    def test_the_outline_is_one_channel_on_the_mask_grid(self):
        model = _model(amodal_outline=True)
        x = torch.randn(1, 3, 64, 64)
        seg, outline = model(x, return_outline=True)
        assert tuple(seg.shape[1:]) == (2, 16, 16)      # stride 16
        assert tuple(outline.shape) == (1, 1, 16, 16)

    def test_the_plain_forward_is_unchanged_when_the_head_exists(self):
        model = _model(amodal_outline=True)
        x = torch.randn(1, 3, 64, 64)
        alone = model(x)
        with_head = model(x, return_outline=True)[0]
        assert torch.equal(alone, with_head)

    def test_a_head_with_no_segmentation_head_is_refused(self):
        with pytest.raises(ValueError, match="needs a segmentation head"):
            _model(amodal_outline=True, num_segmentation_classes=None,
                   enable_finetuning=False)

    def test_return_outline_on_a_headless_model_refuses(self):
        model = _model()
        with pytest.raises(RuntimeError, match="no outline head"):
            model(torch.randn(1, 3, 64, 64), return_outline=True)

    def test_return_outline_off_the_segmentation_path_refuses(self):
        model = _model(amodal_outline=True, enable_finetuning=False)
        with pytest.raises(RuntimeError, match="segmentation"):
            model(torch.randn(1, 3, 64, 64), return_outline=True)


class TestTheRecordAndTheWeights:
    def test_a_matching_record_and_weights_both_ways(self):
        for flag in (False, True):
            weights, _record = _state(amodal_outline=flag)
            assert check_amodal_outline(flag, weights) is flag
            del weights
            gc.collect()

    def test_with_nothing_recorded_the_weights_decide(self):
        headed, _ = _state(amodal_outline=True)
        assert check_amodal_outline(None, headed) is True
        del headed
        gc.collect()
        headless, _ = _state()
        assert check_amodal_outline(None, headless) is False

    def test_a_record_the_weights_contradict_refuses(self):
        headed, _ = _state(amodal_outline=True)
        with pytest.raises(ArchMismatchError, match="have an outline head"):
            check_amodal_outline(False, headed)
        del headed
        gc.collect()
        headless, _ = _state()
        with pytest.raises(ArchMismatchError, match="have no outline head"):
            check_amodal_outline(True, headless)

    def test_a_non_boolean_record_refuses(self):
        weights, _ = _state()
        with pytest.raises(ArchMismatchError, match="not a boolean"):
            check_amodal_outline("yes", weights)

    def test_a_state_dict_with_neither_head_says_nothing(self):
        weights, _ = _state()
        weights = {k: v for k, v in weights.items()
                   if not k.startswith("segmentation_head.")}
        assert infer_amodal_outline(weights) is None
        # And the load then takes the record at its word.
        assert check_amodal_outline(True, weights) is True
        assert check_amodal_outline(None, weights) is False

    def test_the_arch_kwargs_rebuild_the_head(self):
        model = _model(amodal_outline=True)
        kwargs = _arch_kwargs(model)
        del model
        gc.collect()
        assert kwargs["amodal_outline"] is True
        assert _keys_of(**kwargs) == _keys_of(amodal_outline=True)


# ---------------------------------------------------------------------------
# 2. The target


class TestTheTarget:
    def test_the_outline_covers_the_visible_part_and_the_hidden_ring(self):
        stats = {}
        visible, outline = build_binary_mask_coco_all(
            _cocoa_row(), stats=stats, with_outline=True)
        v = _as_array(visible) > 0
        o = _as_array(outline) > 0
        assert v.any() and o.any()
        assert ((o | v) == o).all(), "the outline does not cover the visible target"
        # The ring between the visible box and the full box is in the outline.
        x0, y0, x1, y1 = FULL_BOX
        assert o[y0 + 2, x0 + 2] and not v[y0 + 2, x0 + 2]
        # The visible part is a strict subset: the hidden part is real.
        assert (o & ~v).sum() > 0
        assert stats.get("outline_unknown_images", 0) == 0

    def test_a_kept_region_with_no_full_polygon_makes_the_sample_unknown(self):
        stats = {}
        visible, outline = build_binary_mask_coco_all(
            _cocoa_row(segmentation=False), stats=stats, with_outline=True)
        assert _as_array(visible).sum() > 0
        assert int(_as_array(outline).sum()) == 0
        assert stats["outline_unknown_images"] == 1

    def test_a_region_whose_full_label_covers_the_frame_is_unknown_too(self):
        stats = {}
        _visible, outline = build_binary_mask_coco_all(
            _cocoa_row(full_box=(0, 0, SIZE, SIZE)), stats=stats, with_outline=True)
        assert int(_as_array(outline).sum()) == 0
        assert stats["outline_unknown_images"] == 1

    def test_without_the_flag_the_builder_returns_todays_array(self):
        stats = {}
        mask = build_binary_mask_coco_all(_cocoa_row(), stats=stats)
        assert isinstance(mask, np.ndarray) and mask.ndim == 2
        assert mask.dtype == np.int64
        assert "outline_unknown_images" not in stats

    def test_both_values_share_one_dtype_and_frame(self):
        visible, outline = build_binary_mask_coco_all(_cocoa_row(), with_outline=True)
        assert visible.shape == outline.shape == (SIZE, SIZE)
        assert visible.dtype == outline.dtype == np.int64

    def test_a_sample_with_no_kept_region_is_empty_and_not_unknown(self):
        stats = {}
        row = _cocoa_row(visible=False, segmentation=False)
        visible, outline = build_binary_mask_coco_all(row, stats=stats, with_outline=True)
        assert int(_as_array(visible).sum()) == 0
        assert int(_as_array(outline).sum()) == 0
        # Nothing was kept, so nothing is missing from the outline.
        assert stats.get("outline_unknown_images", 0) == 0

    def test_the_unknown_decision_leaves_the_dropped_count_alone(self):
        """The spec's dropped-region scenario, on a sample that really drops one.

        A near-full region is dropped by the visible builder; the sample is
        also unknown for its outline because its one kept region has no full
        label. The unknown decision must not touch the dropped count, so both
        runs report the drop the visible builder found.
        """
        row = _cocoa_row(segmentation=False)
        row["annotations"]["regions"].append({
            "name": "r1", "is_stuff": False, "segmentation": None,
            "visible_mask": _box_mask((0, 0, SIZE, SIZE)),
        })
        off, on = {}, {}
        build_binary_mask_coco_all(row, stats=off)
        build_binary_mask_coco_all(row, stats=on, with_outline=True)
        assert off["dropped_regions"] == 1
        assert on["dropped_regions"] == off["dropped_regions"]
        assert on["outline_unknown_images"] == 1


# ---------------------------------------------------------------------------
# 3. Geometry: the augmentation moves both masks by the same parameters


TRANSFORMS = [coco_train_transforms, coco_train_transforms_fg_centered,
              coco_val_transforms]


class TestTheAugmentation:
    @pytest.mark.parametrize("fn", TRANSFORMS)
    @pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
    def test_the_augmented_outline_still_covers_its_visible_mask(self, fn, seed):
        row = _cocoa_row()
        batch = {k: [row[k]] for k in ("image", "annotations", "image_id")}
        torch.manual_seed(seed)
        np.random.seed(seed)
        out = fn(batch, input_size=64, with_outline=True)
        v = _mask2d(out["segmentation_mask"][0]) > 0
        o = _as_array(out["outline_mask"][0]) > 0
        assert v.shape == o.shape
        assert o.any() and ((o | v) == o).all(), (fn.__name__, seed)

    @pytest.mark.parametrize("fn", TRANSFORMS)
    def test_the_outline_key_is_absent_unless_asked(self, fn):
        row = _cocoa_row()
        batch = {k: [row[k]] for k in ("image", "annotations", "image_id")}
        out = fn(batch, input_size=64)
        for key in ("outline_mask", "has_full_outline", "outline_unknown_images"):
            assert key not in out

    @pytest.mark.parametrize("fn", TRANSFORMS)
    def test_a_loader_built_for_outlines_labels_every_sample(self, fn):
        row = _cocoa_row()
        batch = {k: [row[k]] * 3 for k in ("image", "annotations", "image_id")}
        out = fn(batch, input_size=64, with_outline=True)
        assert len(out["outline_mask"]) == 3
        assert all(_as_array(m).sum() > 0 for m in out["outline_mask"])
        assert list(out["has_full_outline"]) == [True, True, True]

    def test_the_streaming_transform_mirrors_the_outline(self):
        torch.manual_seed(0)
        out = coco_train_transform_example(_cocoa_row(), input_size=64,
                                           with_outline=True)
        v = _mask2d(out["segmentation_mask"]) > 0
        o = _as_array(out["outline_mask"]) > 0
        assert o.any() and ((o | v) == o).all()
        assert out["has_full_outline"] is True
        assert out["outline_unknown_images"] == 0

    def test_the_streaming_val_transform_is_deterministic_and_covered(self):
        out = coco_val_transform_example(_cocoa_row(), input_size=64, with_outline=True)
        v = _mask2d(out["segmentation_mask"]) > 0
        o = _as_array(out["outline_mask"]) > 0
        assert o.any() and ((o | v) == o).all()

    def test_an_unknown_target_stays_unknown_through_the_transform(self):
        row = _cocoa_row(segmentation=False)
        out = coco_val_transform_example(row, input_size=64, with_outline=True)
        assert int(_as_array(out["outline_mask"]).sum()) == 0
        assert out["has_full_outline"] is False
        assert out["outline_unknown_images"] == 1

    @pytest.mark.parametrize("fn", TRANSFORMS)
    @pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
    def test_the_outline_is_the_visible_mask_exactly_when_the_full_extent_is(self, fn, seed):
        """The transformed shift is the transformed shift, pixel for pixel.

        Both targets are built from the same polygon here, so they start
        identical; a crop, flip or resize applied to only one of them, or with
        different parameters, leaves a difference after the transform. The
        containment sweep above cannot see that, because containment survives a
        shift.
        """
        row = _cocoa_row(visible=False, full_box=VISIBLE_BOX)
        batch = {k: [row[k]] for k in ("image", "annotations", "image_id")}
        torch.manual_seed(seed)
        np.random.seed(seed)
        out = fn(batch, input_size=64, with_outline=True)
        v = _mask2d(out["segmentation_mask"][0]) > 0
        o = _as_array(out["outline_mask"][0]) > 0
        assert v.any()
        assert np.array_equal(o, v), (fn.__name__, seed)

    @pytest.mark.parametrize("fn", [coco_train_transform_example,
                                    coco_val_transform_example])
    def test_the_streaming_outline_is_the_visible_mask_exactly_when_full_is(self, fn):
        row = _cocoa_row(visible=False, full_box=VISIBLE_BOX)
        torch.manual_seed(0)
        out = fn(row, input_size=64, with_outline=True)
        v = _mask2d(out["segmentation_mask"]) > 0
        o = _as_array(out["outline_mask"]) > 0
        assert v.any()
        assert np.array_equal(o, v)

    @pytest.mark.parametrize("fn", TRANSFORMS)
    def test_a_prepared_bdd_row_is_an_unlabelled_sample(self, fn):
        """BDD has no amodal label: zeros and False, and not counted as unknown.

        The spec's "COCO or BDD in a mix" scenario: mixing plain data into a
        COCOA refine must not look like a broken COCOA row (which would raise
        the unknown count) nor like a labelled one (a full-black target would
        teach the head to predict nothing).
        """
        img = np.full((SIZE, SIZE, 3), DARK, np.uint8)
        mask = np.zeros((SIZE, SIZE), dtype=bool)
        x0, y0, x1, y1 = VISIBLE_BOX
        mask[y0:y1, x0:x1] = True
        row = {"image": Image.fromarray(img), "binary_mask": mask, "image_id": 3}
        batch = {k: [row[k]] for k in ("image", "binary_mask", "image_id")}
        torch.manual_seed(0)
        np.random.seed(0)
        out = fn(batch, input_size=64, with_outline=True)
        assert int(_as_array(out["outline_mask"][0]).sum()) == 0
        assert list(out["has_full_outline"]) == [False]
        assert list(out["outline_unknown_images"]) == [0]
        # the visible target is still the prepared mask, not zeroed with it
        assert _mask2d(out["segmentation_mask"][0]).sum() > 0


# ---------------------------------------------------------------------------
# 4. Collate and the loaders' contract


class TestTheCollate:
    def _samples(self, n=2, with_outline=True):
        return [coco_val_transform_example(_cocoa_row(), input_size=64,
                                           with_outline=with_outline)
                for _ in range(n)]

    def test_it_stacks_the_three_keys(self):
        batch = coco_collate_fn(self._samples())
        assert tuple(batch["outline_mask"].shape) == (2, 64, 64)
        assert batch["outline_mask"].dtype == torch.int64
        assert batch["has_full_outline"].dtype == torch.bool
        assert int(batch["outline_unknown_images"]) == 0

    def test_an_unasked_batch_carries_none_of_them(self):
        batch = coco_collate_fn(self._samples(with_outline=False))
        for key in ("outline_mask", "has_full_outline", "outline_unknown_images"):
            assert key not in batch

    def test_an_all_dropped_batch_keeps_the_keys(self):
        samples = self._samples()
        for s in samples:
            s["valid"] = False
        batch = coco_collate_fn(samples, input_size=64)
        assert tuple(batch["outline_mask"].shape) == (1, 64, 64)
        assert not bool(batch["has_full_outline"].any())
        assert int(batch["outline_unknown_images"]) == 0

    def test_the_unknown_count_is_summed_over_the_batch(self):
        samples = [coco_val_transform_example(
            _cocoa_row(segmentation=(i != 0)), input_size=64, with_outline=True)
            for i in range(3)]
        batch = coco_collate_fn(samples)
        assert int(batch["outline_unknown_images"]) == 1
        assert list(batch["has_full_outline"]) == [False, True, True]


# ---------------------------------------------------------------------------
# 5. The loss and the metrics


def _outline_loss(logits, target, has_full):
    loss, labelled = outline_loss_from_logits(
        logits, target, torch.tensor(has_full, dtype=torch.bool))
    return float(loss), labelled


class TestTheLoss:
    def test_a_perfect_head_loses_less_than_a_flipped_one(self):
        t = torch.zeros(1, 1, 8, 8)
        t[..., 2:6, 2:6] = 1
        good = torch.where(t > 0, 10.0, -10.0)
        bad = -good
        assert _outline_loss(good, t, [True])[0] < _outline_loss(bad, t, [True])[0]

    def test_an_unlabelled_sample_contributes_nothing(self):
        t = torch.zeros(2, 8, 8)
        t[0, 2:6, 2:6] = 1
        t[1, 1:3, 1:3] = 1
        logits = torch.randn(2, 1, 8, 8)
        both, n = _outline_loss(logits, t, [True, False])
        only_first, n1 = _outline_loss(logits[:1], t[:1], [True])
        assert n == 1 and n1 == 1
        assert both == pytest.approx(only_first, rel=1e-6)

    def test_no_labelled_sample_is_a_zero_that_still_reaches_the_head(self):
        t = torch.zeros(2, 8, 8)
        logits = torch.zeros(2, 1, 8, 8, requires_grad=True)
        loss, labelled = outline_loss_from_logits(
            logits, t, torch.tensor([False, False], dtype=torch.bool))
        assert labelled == 0 and float(loss) == 0.0
        # The head is in the graph: DDP will not report it unused, and a later
        # labelled batch trains it.
        assert loss.requires_grad


def _ring_target():
    """[B,H,W] amodal and visible targets: full box, inner box."""
    full = torch.zeros(2, 8, 8, dtype=torch.int64)
    full[:, 1:7, 1:7] = 1
    visible = torch.zeros(2, 8, 8, dtype=torch.int64)
    visible[:, 3:5, 3:5] = 1
    return full, visible


def _logits_around(masks, on=10.0):
    """Two-class visible logits that predict these masks exactly."""
    out = torch.full((masks.size(0), 2, 8, 8), -on)
    out[:, 1] = torch.where(masks > 0, on, -on)
    out[:, 0] = -out[:, 1]
    return out


def _outline_logits(masks, on=10.0):
    return torch.where(masks > 0, on, -on).unsqueeze(1)


class TestTheMetrics:
    def test_the_head_that_fills_the_hidden_part_beats_the_baseline(self):
        full, visible = _ring_target()
        m = outline_metrics_from_logits(
            _outline_logits(full), _logits_around(visible), full, visible,
            torch.tensor([True, True]))
        assert m["val_full_iou"] == pytest.approx(100.0, abs=1e-3)
        assert m["val_hidden_iou"] == pytest.approx(100.0, abs=1e-3)
        # The visible prediction cannot reach the hidden ring: its dilation
        # only guesses one pixel around itself.
        assert m["val_hidden_iou_baseline"] < m["val_hidden_iou"]
        assert m["val_full_iou_baseline"] < m["val_full_iou"]
        assert m["val_hidden_images"] == 2

    def test_a_sample_with_no_hidden_part_is_not_counted(self):
        full, visible = _ring_target()
        full[0] = visible[0]                      # no hidden extent on sample 0
        m = outline_metrics_from_logits(
            _outline_logits(full), _logits_around(visible), full, visible,
            torch.tensor([True, True]))
        assert m["val_hidden_images"] == 1

    def test_an_unlabelled_sample_is_left_out_entirely(self):
        full, visible = _ring_target()
        m = outline_metrics_from_logits(
            _outline_logits(full), _logits_around(visible), full, visible,
            torch.tensor([True, False]))
        assert m["val_hidden_images"] == 1
        assert m["val_full_iou"] == pytest.approx(100.0, abs=1e-3)

    def test_no_labelled_sample_reports_nothing(self):
        full, visible = _ring_target()
        m = outline_metrics_from_logits(
            _outline_logits(full), _logits_around(visible), full, visible,
            torch.tensor([False, False]))
        assert m == {}


# ---------------------------------------------------------------------------
# 6. The training entry points


class _OutlineSegmenter(TinySegmenter):
    """TinySegmenter plus a real outline head over its foreground logits."""

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.outline_head = nn.Conv2d(1, 1, 1)   # an attribute the flag reads
        self.amodal_outline = True

    def forward(self, x, return_outline=False):
        out = super().forward(x)
        if not return_outline:
            return out
        return out, self.outline_head(out[:, 1:2])


def _outline_batches(n=2, batch_size=2, h=4, w=4, labelled=True):
    batches = []
    for _ in range(n):
        mask = (torch.rand(batch_size, 1, h, w) > 0.5).float()
        batches.append({
            "pixel_values": torch.randn(batch_size, 3, h, w),
            "segmentation_mask": mask.to(torch.int64).squeeze(1),
            "image_id": torch.arange(batch_size),
            "valid": torch.ones(batch_size, dtype=torch.bool),
            "dropped_regions": torch.tensor(0),
            "outline_mask": mask.to(torch.int64).squeeze(1),
            "has_full_outline": torch.full((batch_size,), labelled, dtype=torch.bool),
            "outline_unknown_images": torch.tensor(0),
        })
    return batches


class TestTheEntryPoints:
    def test_the_peek_accepts_a_loader_with_labels(self):
        require_outline_labels(_outline_batches())    # does not raise

    def test_the_peek_refuses_a_loader_without_the_key(self):
        batches = _outline_batches()
        for b in batches:
            del b["outline_mask"]
            del b["has_full_outline"]
            del b["outline_unknown_images"]
        with pytest.raises(RuntimeError, match="with_outline=True"):
            require_outline_labels(batches)

    def test_the_peek_refuses_a_loader_with_no_labelled_sample(self):
        with pytest.raises(RuntimeError, match="full-outline label"):
            require_outline_labels(_outline_batches(labelled=False))

    def test_evaluate_scores_the_head_and_counts_the_images(self):
        batches = _outline_batches()
        out = evaluate(_OutlineSegmenter(in_features=48).eval(), batches,
                       torch.device("cpu"), is_segmentation=True)
        for key in ("val_full_iou", "val_hidden_iou", "val_outline_loss",
                    "val_hidden_images"):
            assert key in out, key
        assert out["val_outline_labelled_images"] == 4
        assert out["val_outline_unknown_images"] == 0

    def test_evaluate_reports_no_outline_numbers_without_the_targets(self):
        batches = _outline_batches()
        for b in batches:
            del b["outline_mask"]
            del b["has_full_outline"]
            del b["outline_unknown_images"]
        out = evaluate(TinySegmenter(in_features=48), batches,
                       torch.device("cpu"), is_segmentation=True)
        assert "val_full_iou" not in out
        assert "val_outline_labelled_images" not in out
        assert "val_iou" in out

    def test_an_unlabelled_set_logs_the_counts_but_no_loss(self):
        """A COCO (or BDD) set's term is a masked zero, not a measurement.

        Writing it would let a mixed validation average it into the outline
        loss as if the set had a target; the counts still say zero.
        """
        out = evaluate(_OutlineSegmenter(in_features=48).eval(),
                       _outline_batches(labelled=False),
                       torch.device("cpu"), is_segmentation=True)
        assert out["val_outline_labelled_images"] == 0
        assert "val_outline_loss" not in out
        # And nothing to score, so no IoU numbers either.
        assert "val_full_iou" not in out


class TestTheMixedValidation:
    """A COCOA-primary run validates on COCOA (outline labels) plus COCO (none)."""

    def _merge(self, primary, mixed):
        return merge_mixed_val_metrics(dict(primary), dict(mixed))

    def test_a_set_with_no_outline_numbers_does_not_abort_the_merge(self):
        primary = {"val_iou": 70.0, "val_loss": 1.0, "val_dice": 0.8,
                   "val_failed_images": 0, "val_dropped_regions": 0}
        mixed = {"val_iou": 60.0, "val_loss": 2.0, "val_dice": 0.7,
                 "val_failed_images": 1, "val_dropped_regions": 2,
                 "val_full_iou": 55.0, "val_hidden_iou": 40.0,
                 "val_full_iou_baseline": 50.0, "val_hidden_iou_baseline": 30.0,
                 "val_hidden_images": 9, "val_outline_unknown_images": 3,
                 "val_outline_labelled_images": 12, "val_outline_loss": 0.5}
        merged = self._merge(primary, mixed)
        # The outline numbers come from the set that has them, unsuffixed and
        # per-set; the other set contributes nothing rather than a zero.
        assert merged["val_full_iou"] == 55.0
        assert merged["val_full_iou_coco"] == 55.0
        assert "val_full_iou_cocoa" not in merged
        assert merged["val_hidden_images"] == 9
        assert merged["val_outline_labelled_images"] == 12

    def test_two_sets_with_outlines_keep_both_and_a_mean_loss(self):
        primary = {"val_iou": 70.0, "val_loss": 1.0, "val_dice": 0.8,
                   "val_failed_images": 0, "val_dropped_regions": 0,
                   "val_full_iou": 45.0, "val_hidden_iou": 20.0,
                   "val_full_iou_baseline": 40.0, "val_hidden_iou_baseline": 10.0,
                   "val_hidden_images": 4, "val_outline_unknown_images": 0,
                   "val_outline_labelled_images": 10, "val_outline_loss": 0.7}
        mixed = {**primary, "val_iou": 60.0, "val_full_iou": 55.0,
                 "val_outline_loss": 0.5, "val_hidden_images": 9,
                 "val_outline_labelled_images": 12}
        merged = self._merge(primary, mixed)
        assert merged["val_full_iou_cocoa"] == 45.0
        assert merged["val_full_iou_coco"] == 55.0
        # Primary first, as the run is selected on its own set.
        assert merged["val_full_iou"] == 45.0
        assert merged["val_outline_loss"] == pytest.approx(0.6)
        # Counts are totals, as for the other counts.
        assert merged["val_hidden_images"] == 13
        assert merged["val_outline_labelled_images"] == 22


# ---------------------------------------------------------------------------
# 7. Loading: the flag reaches the model from the artifacts


class TestResolveAmodalOutline:
    def test_the_weights_decide_when_nothing_is_stated(self):
        headed, _ = _state(amodal_outline=True)
        assert resolve_amodal_outline(None, headed) == (True, "weights")
        del headed
        gc.collect()
        headless, _ = _state()
        assert resolve_amodal_outline({}, headless) == (False, "weights")

    def test_a_stated_flag_that_the_weights_contradict_refuses(self):
        headed, _ = _state(amodal_outline=True)
        with pytest.raises(ModelShapeError, match="have an outline head"):
            resolve_amodal_outline({"amodal_outline": False}, headed)

    def test_a_stated_flag_is_reported_with_its_source(self):
        meta = {"custom_metadata": {"amodal_outline": True}}
        headed, _ = _state(amodal_outline=True)
        assert resolve_amodal_outline(meta, headed) \
            == (True, "metadata.custom_metadata.amodal_outline")

    def test_nothing_stated_and_no_head_anywhere_is_the_old_model(self):
        weights, _ = _state()
        weights = {k: v for k, v in weights.items()
                   if not k.startswith("segmentation_head.")}
        assert resolve_amodal_outline(None, weights) == (False, "default")

    def test_a_non_boolean_statement_refuses(self):
        weights, _ = _state()
        with pytest.raises(ModelShapeError, match="not true or false"):
            resolve_amodal_outline({"amodal_outline": "on"}, weights)

    def test_a_wrapped_checkpoint_is_unwrapped(self):
        headed, _ = _state(amodal_outline=True)
        assert resolve_amodal_outline(None, {"state_dict": headed}) == (True, "weights")


# ---------------------------------------------------------------------------
# 2. The loading paths: the builders that read a checkpoint


def _saved_checkpoint(tmp_path, name, *, amodal_outline, meta=None):
    """Write a real checkpoint; returns ``(path, its key shapes)``, nothing live."""
    model = _model(amodal_outline=amodal_outline)
    state = {k: v.clone() for k, v in model.state_dict().items()}
    shapes = {k: tuple(v.shape) for k, v in state.items()}
    del model
    gc.collect()
    ckpt = {"state_dict": state}
    if meta is not None:
        ckpt["custom_metadata"] = dict(meta)
    path = tmp_path / name
    torch.save(ckpt, path)
    return path, shapes


class TestTheLoadingPaths:
    def test_the_eval_tool_loads_a_live_checkpoint_without_a_head(self, tmp_path):
        path, shapes = _saved_checkpoint(tmp_path, "live.pth", amodal_outline=False)
        model = load_eval_checkpoint(path, "cpu")
        assert model.outline_head is None
        assert {k: tuple(v.shape) for k, v in model.state_dict().items()} == shapes

    def test_the_eval_tool_builds_the_head_the_weights_carry(self, tmp_path):
        # No record: the weights' own key decides, as the pre-change fields do.
        path, _ = _saved_checkpoint(tmp_path, "outline.pth", amodal_outline=True)
        model = load_eval_checkpoint(path, "cpu")
        assert model.outline_head is not None
        assert model.amodal_outline is True

    def test_the_demo_builder_loads_an_outline_checkpoint_strictly(self, tmp_path):
        path, shapes = _saved_checkpoint(
            tmp_path, "outline.pth", amodal_outline=True,
            meta={"amodal_outline": True})
        ckpt = torch.load(path, map_location="cpu")
        model = build_demo_model(2, 16, hcm_mod.DECODER_V1, "cpu",
                                 state_dict=ckpt["state_dict"], amodal_outline=True)
        assert model.outline_head is not None
        assert {k: tuple(v.shape) for k, v in model.state_dict().items()} == shapes

    def test_the_rebuild_from_model_info_builds_the_head(self):
        model = _model(amodal_outline=True)
        info = {"model_class": hcm_mod.hcm_nn, "model_config": _arch_kwargs(model)}
        state = {k: v.clone() for k, v in model.state_dict().items()}
        del model
        gc.collect()
        rebuilt = build_model_from_info(info, state)
        assert rebuilt.outline_head is not None
        assert torch.equal(rebuilt.outline_head.weight, state["outline_head.weight"])

    @pytest.mark.parametrize("recorded,weights_have_head", [(False, True), (True, False)])
    def test_a_record_the_weights_contradict_is_refused_by_the_rebuild(
            self, recorded, weights_have_head):
        model = _model(amodal_outline=weights_have_head)
        cfg = _arch_kwargs(model)
        cfg["amodal_outline"] = recorded
        info = {"model_class": hcm_mod.hcm_nn, "model_config": cfg}
        state = {k: v.clone() for k, v in model.state_dict().items()}
        del model
        gc.collect()
        with pytest.raises(ArchMismatchError, match="amodal_outline"):
            build_model_from_info(info, state)

    def test_a_warm_start_reports_the_head_as_new(self):
        source, _ = _state(amodal_outline=False)
        model = _model(amodal_outline=True)
        try:
            warm = warm_start_state(model, source)
            assert sorted(warm["new"]) == ["outline_head.bias", "outline_head.weight"]
            assert set(warm["copied"]) == set(source)
            assert warm["skipped"] == [] and warm["dropped"] == []
            after = model.state_dict()
            assert all(torch.equal(after[k], source[k]) for k in warm["copied"])
        finally:
            del model
            gc.collect()
