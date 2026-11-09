"""COCOA train regions are no longer dropped by the polygon span/area guards.

COCOA train loads with decode_rle=False, so its regions arrive as flat polygons.
build_binary_mask_coco_all used to drop a polygon spanning >= 98% of the image on
both axes, or covering > 80% of it, and did so silently; COCOA val (PIL masks) only
ever met the 0.995 fill check, so train and val targets disagreed. Now the 0.995
fill check is the one guard both paths share, every region it drops is counted
from the mask builder through the transforms and collate into a metric, and the
polygon rasteriser draws exactly what pycocotools draws.
"""
import numpy as np
import pytest
import torch
from pycocotools import mask as mask_utils

from ecs_hcm_model.core import model_utils
from ecs_hcm_model.core import trainer as trainer_module
from ecs_hcm_model.core.model_utils import (
    _poly_to_mask_cocoa,
    build_binary_mask_coco_all,
    coco_collate_fn,
    coco_train_transform_example,
    coco_train_transforms,
    coco_train_transforms_fg_centered,
    coco_val_transform_example,
    coco_val_transforms,
)
from ecs_hcm_model.core.trainer import evaluate
from ecs_hcm_model.eval.cocoa_scan import scan

from conftest import RecordingLogger, TinySegmenter, make_segmentation_batches
from test_coco_mix import _clean_seams, _script_evaluate  # noqa: F401  (fixture)
from test_coco_seg_pipeline import _bright_image, _flat_polygon, _cocoa_hf_row

S = 512

# A right triangle over the whole frame: spans 100% on both axes, so the old span
# guard dropped it, yet it fills only half the image.
FULL_SPAN_TRIANGLE = [0.0, 0.0, 511.0, 0.0, 0.0, 511.0]
# A 460 px square: spans about 90% on each axis (under the span guard) but covers
# about 81% of the image, so the old area guard dropped it.
LARGE_SQUARE = [20.0, 20.0, 480.0, 20.0, 480.0, 480.0, 20.0, 480.0]
# The whole frame: fill > 0.995, the one guard that remains.
FULL_FRAME = [0.0, 0.0, 512.0, 0.0, 512.0, 512.0, 0.0, 512.0]


def _reference(poly, h=S, w=S):
    """What pycocotools (the val path's decoder) draws for this polygon.

    Rounded through float32 first: the HF features store COCOA outlines as
    float32, so that is what the train path rasterises. A coordinate like 50.2
    lands on pycocotools' rounding boundary (x5 = 251.0), where float64 and float32
    differ by a whole edge row.
    """
    pts = [float(v) for v in np.asarray(poly, dtype=np.float32).reshape(-1)]
    return mask_utils.decode(mask_utils.merge(mask_utils.frPyObjects([pts], h, w)))


def _hf_row_with(polys):
    """A shunk031/COCOA row (decode_rle=False HF features) with these region outlines."""
    row = _cocoa_hf_row([(256, 256)] * len(polys))
    regions = row["regions"][0]
    regions["segmentation"] = [list(p) for p in polys]
    return {"image": _bright_image([]), "annotations": row, "image_id": 7}


class TestLargeRegionsAreKept:
    @pytest.mark.parametrize("name,poly", [("full-span", FULL_SPAN_TRIANGLE), ("large-area", LARGE_SQUARE)])
    def test_the_region_is_in_the_train_target(self, name, poly):
        stats = {}
        mask = build_binary_mask_coco_all(_hf_row_with([poly]), stats=stats)
        ref = _reference(poly)
        assert mask is not None and int(mask.sum()) == int(ref.sum()), name
        assert np.array_equal(mask > 0, ref > 0), name
        assert stats["dropped_regions"] == 0

    def test_a_large_region_and_a_small_one_both_land(self):
        small = _flat_polygon(400, 400)
        mask = build_binary_mask_coco_all(_hf_row_with([FULL_SPAN_TRIANGLE, small]))
        assert np.array_equal(mask > 0, (_reference(FULL_SPAN_TRIANGLE) | _reference(small)) > 0)


class TestTheRemainingGuardIsCounted:
    def test_a_full_frame_region_is_dropped_and_counted(self):
        stats = {}
        mask = build_binary_mask_coco_all(_hf_row_with([FULL_FRAME]), stats=stats)
        assert int(mask.sum()) == 0
        assert stats["dropped_regions"] == 1

    def test_only_the_bad_region_is_dropped(self):
        small = _flat_polygon(256, 256)
        stats = {}
        mask = build_binary_mask_coco_all(_hf_row_with([FULL_FRAME, small]), stats=stats)
        assert np.array_equal(mask > 0, _reference(small) > 0)
        assert stats["dropped_regions"] == 1

    def test_counts_accumulate_across_calls(self):
        stats = {}
        build_binary_mask_coco_all(_hf_row_with([FULL_FRAME]), stats=stats)
        build_binary_mask_coco_all(_hf_row_with([FULL_FRAME, FULL_FRAME]), stats=stats)
        assert stats["dropped_regions"] == 3

    @pytest.mark.parametrize("fn", [coco_train_transforms, coco_train_transforms_fg_centered, coco_val_transforms])
    def test_the_batch_transforms_report_it_per_sample(self, fn):
        good, bad = _hf_row_with([_flat_polygon(256, 256)]), _hf_row_with([FULL_FRAME, FULL_FRAME])
        out = fn({k: [good[k], bad[k]] for k in ("image", "annotations", "image_id")})
        assert list(out["dropped_regions"]) == [0, 2]

    @pytest.mark.parametrize("fn", [coco_train_transform_example, coco_val_transform_example])
    def test_the_streaming_transforms_report_it(self, fn):
        assert fn(_hf_row_with([FULL_FRAME]))["dropped_regions"] == 1
        assert fn(_hf_row_with([_flat_polygon(256, 256)]))["dropped_regions"] == 0

    def test_collate_sums_it_over_the_batch(self):
        samples = [coco_val_transform_example(_hf_row_with(p)) for p in ([FULL_FRAME], [FULL_FRAME, FULL_FRAME], [_flat_polygon(256, 256)])]
        assert int(coco_collate_fn(samples)["dropped_regions"]) == 3

    def test_evaluate_reports_it(self):
        class Model(torch.nn.Module):
            def forward(self, x):
                return torch.zeros(x.shape[0], 2, x.shape[2], x.shape[3])

        batch = {
            "pixel_values": torch.ones(2, 3, 8, 8),
            "segmentation_mask": torch.ones(2, 8, 8, dtype=torch.int64),
            "valid": torch.ones(2, dtype=torch.bool),
            "dropped_regions": torch.tensor(5),
        }
        out = evaluate(Model(), [batch, dict(batch)], torch.device("cpu"), is_segmentation=True)
        assert out["val_dropped_regions"] == 10

    def test_the_loop_logs_train_and_val_counts_every_epoch(self, monkeypatch):
        _script_evaluate(monkeypatch, [
            {"val_loss": 1.0, "val_iou": 40.0, "val_dice": 50.0, "val_dropped_regions": 0},
            {"val_loss": 0.9, "val_iou": 41.0, "val_dice": 51.0, "val_dropped_regions": 3},
        ])
        train = make_segmentation_batches(n_batches=2, batch_size=2)
        for b, n in zip(train, (1, 2)):
            b["dropped_regions"] = torch.tensor(n)
        logger = RecordingLogger()
        trainer_module.train_model_for_epochs(
            model=TinySegmenter(), epochs=2, train_loader=train,
            val_loader=make_segmentation_batches(n_batches=1, batch_size=2),
            device=torch.device("cpu"), logger=logger, test_mode=True,
            finetune=True, coco=False, task="cocoa_seg",
        )
        logged = {k: [(v, s) for kk, v, s in logger.scalars if kk == k]
                  for k in ("train_dropped_regions", "val_dropped_regions")}
        assert logged["train_dropped_regions"] == [(3.0, 0), (3.0, 1)]
        assert logged["val_dropped_regions"] == [(0.0, 0), (3.0, 1)]


class TestTheRasteriserMatchesPycocotools:
    @pytest.mark.parametrize("poly", [
        _flat_polygon(256, 256),
        [10.5, 10.5, 200.25, 30.75, 120.0, 190.5],
        [100.0, 50.0, 300.0, 60.0, 310.0, 260.0, 220.0, 400.0, 90.0, 300.0],
        [100.5, 50.2, 300.1, 60.7, 310.3, 260.4, 220.8, 400.6, 90.2, 300.9],
        LARGE_SQUARE,
        FULL_SPAN_TRIANGLE,
    ])
    def test_pixel_for_pixel(self, poly):
        mine = _poly_to_mask_cocoa(np.asarray(poly, dtype=np.float32).reshape(-1, 2), S, S)
        assert np.array_equal(np.asarray(mine) > 0, _reference(poly) > 0)

    def test_a_non_square_frame(self):
        poly = [5.0, 5.0, 300.0, 5.0, 300.0, 150.0, 5.0, 150.0]
        mine = _poly_to_mask_cocoa(np.asarray(poly, dtype=np.float32).reshape(-1, 2), 200, 320)
        assert np.array_equal(np.asarray(mine) > 0, _reference(poly, 200, 320) > 0)


class TestTheDeprecatedBuilderIsGone:
    def test_it_no_longer_exists(self):
        assert not hasattr(model_utils, "build_binary_mask_coco")


class TestTheScan:
    def test_it_counts_each_old_guard_and_empty_segmentations(self):
        small = _flat_polygon(256, 256)
        # Spans 98.5% on both axes and fills ~97%: the span guard catches it first.
        near_full = [0.0, 0.0, 504.0, 0.0, 504.0, 504.0, 0.0, 504.0]
        data = {
            "images": [{"id": 1, "height": S, "width": S}, {"id": 2, "height": S, "width": S}],
            "annotations": [
                {"image_id": 1, "regions": [
                    {"segmentation": FULL_SPAN_TRIANGLE}, {"segmentation": LARGE_SQUARE},
                    {"segmentation": small}, {"segmentation": []},
                ]},
                {"image_id": 2, "regions": [{"segmentation": small}, {"segmentation": near_full},
                                            {"segmentation": FULL_FRAME}]},
                {"image_id": 99, "regions": [{"segmentation": small}]},
            ],
        }
        r = scan(data)
        assert r["images"] == 2 and r["images_without_size"] == 1
        assert r["regions"] == 7 and r["polygons"] == 6 and r["empty_segmentations"] == 1
        assert r["old_span_rejects"] == 3  # the triangle, near_full, the full frame
        assert r["old_area_rejects"] == 1  # the large square
        assert r["old_fill_rejects"] == 0
        assert r["old_rejected_regions"] == 4
        assert r["old_images_with_a_rejected_region"] == 2
        assert r["current_fill_rejects"] == 1  # only the full frame


class TestEverySkipIsCounted:
    """bb725a73 part 1: the three skips that used to leave a region out silently."""

    def test_a_polygon_field_with_no_usable_polygon(self):
        stats = {}
        mask = build_binary_mask_coco_all(_hf_row_with([[10.0, 20.0]]), stats=stats)
        assert int(mask.sum()) == 0
        assert stats["dropped_regions"] == 1

    def test_an_unsupported_segment_type(self):
        sample = _hf_row_with([_flat_polygon(256, 256)])
        sample["annotations"]["regions"][0]["segmentation"] = ["not a mask"]
        stats = {}
        mask = build_binary_mask_coco_all(sample, stats=stats)
        assert int(mask.sum()) == 0
        assert stats["dropped_regions"] == 1

    @pytest.mark.parametrize("dtype", [np.float32, np.int32])
    def test_numpy_number_coordinates_are_drawn_not_dropped(self, dtype):
        small = _flat_polygon(256, 256)
        sample = _hf_row_with([small])
        sample["annotations"]["regions"][0]["segmentation"] = [
            [dtype(v) for v in small]
        ]
        stats = {}
        mask = build_binary_mask_coco_all(sample, stats=stats)
        assert np.array_equal(mask > 0, _reference(np.asarray(small, dtype=dtype)) > 0)
        assert stats.get("dropped_regions", 0) == 0


class _SizedLoader:
    """A list of batches that reports a batch size, as a DataLoader does."""

    def __init__(self, batches, batch_size):
        self.batches, self.batch_size = batches, batch_size

    def __iter__(self):
        return iter(self.batches)

    def __len__(self):
        return len(self.batches)


def _seg_batch(n, dropped, valid=True):
    b = make_segmentation_batches(n_batches=1, batch_size=n)[0]
    b["dropped_regions"] = torch.tensor(dropped)
    b["valid"] = torch.tensor([valid] * n)
    return b


class TestTrainCountsPerDataset:
    """bb725a73 parts 2-3: train drops and failed images, per dataset under a mix."""

    def _run(self, monkeypatch, mix):
        _script_evaluate(monkeypatch, [
            {"val_loss": 1.0, "val_iou": 40.0, "val_dice": 50.0},
        ])
        # Primary (COCOA, batch size 2): a full batch, then one short by a failure.
        primary = _SizedLoader([_seg_batch(2, 1), _seg_batch(1, 2)], batch_size=2)
        # Mix (COCO, batch size 3): a full batch, then an all-failed placeholder.
        mixed = _SizedLoader([_seg_batch(3, 10), _seg_batch(1, 20, valid=False)], batch_size=3)
        logger = RecordingLogger()
        trainer_module.train_model_for_epochs(
            model=TinySegmenter(), epochs=1, train_loader=primary,
            val_loader=make_segmentation_batches(n_batches=1, batch_size=2),
            device=torch.device("cpu"), logger=logger, test_mode=True,
            finetune=True, coco=False, task="cocoa_seg",
            coco_mix_ratio=1.0 if mix else 0.0,
            mix_train_loader=mixed if mix else None,
        )
        return {k: v for k, v, s in logger.scalars if k.startswith("train_") and s == 0}

    def test_a_mixed_epoch_reports_each_dataset(self, monkeypatch):
        logged = self._run(monkeypatch, mix=True)
        assert logged["train_dropped_regions"] == 33.0
        assert logged["train_dropped_regions_cocoa"] == 3.0
        assert logged["train_dropped_regions_coco"] == 30.0
        assert logged["train_failed_images"] == 4.0
        assert logged["train_failed_images_cocoa"] == 1.0
        assert logged["train_failed_images_coco"] == 3.0

    def test_an_unmixed_epoch_reports_totals_only(self, monkeypatch):
        logged = self._run(monkeypatch, mix=False)
        assert logged["train_dropped_regions"] == 3.0
        assert logged["train_failed_images"] == 1.0
        assert not any(k.endswith(("_cocoa", "_coco")) for k in logged)

    def test_a_clean_epoch_logs_zero_failed_images(self, monkeypatch):
        _script_evaluate(monkeypatch, [{"val_loss": 1.0, "val_iou": 40.0, "val_dice": 50.0}])
        logger = RecordingLogger()
        trainer_module.train_model_for_epochs(
            model=TinySegmenter(), epochs=1,
            train_loader=make_segmentation_batches(n_batches=2, batch_size=2),
            val_loader=make_segmentation_batches(n_batches=1, batch_size=2),
            device=torch.device("cpu"), logger=logger, test_mode=True,
            finetune=True, coco=False, task="cocoa_seg",
        )
        assert [(v, s) for k, v, s in logger.scalars if k == "train_failed_images"] == [(0.0, 0)]


class TestTheScanReadsLikeTheTrainer:
    """bb725a73 part 4: visible_mask first, then the polygon."""

    def test_a_visible_mask_is_read_before_the_polygon(self):
        small = _flat_polygon(256, 256)
        full = np.ones((S, S), dtype=np.uint8, order="F")
        part = np.zeros((S, S), dtype=np.uint8, order="F")
        part[:100, :100] = 1
        data = {
            "images": [{"id": 1, "height": S, "width": S}],
            "annotations": [{"image_id": 1, "regions": [
                # Occluded: the trainer uses the visible mask, not the full-span outline.
                {"segmentation": FULL_SPAN_TRIANGLE, "visible_mask": mask_utils.encode(part)},
                # Its visible mask fills the frame: only the current guard applies.
                {"segmentation": small, "visible_mask": mask_utils.encode(full)},
                {"segmentation": small},
            ]}],
        }
        r = scan(data)
        assert r["visible_masks"] == 2
        assert r["polygons"] == 1
        assert r["old_span_rejects"] == 0 and r["old_rejected_regions"] == 0
        assert r["current_fill_rejects"] == 1


class TestDropsAreCountedPerRegion:
    """f4657a64: every reason is counted, once per region, not once per part."""

    @staticmethod
    def _with_segmentation(segmentation):
        """One unpacked region whose segmentation is a list of parts."""
        sample = _hf_row_with([_flat_polygon(256, 256)])
        sample["annotations"]["regions"] = [{
            "name": "r0", "is_stuff": False, "segmentation": segmentation,
        }]
        return sample

    def test_one_region_with_three_full_frame_parts_is_one_drop(self):
        stats = {}
        build_binary_mask_coco_all(self._with_segmentation([FULL_FRAME] * 3), stats=stats)
        assert stats["dropped_regions"] == 1
        assert stats["partial_regions"] == 0

    def test_a_region_that_keeps_one_part_is_partial_not_dropped(self):
        stats = {}
        mask = build_binary_mask_coco_all(
            self._with_segmentation([FULL_FRAME, _flat_polygon(256, 256)]), stats=stats)
        assert int(mask.sum()) > 0
        assert stats["dropped_regions"] == 0
        assert stats["partial_regions"] == 1

    def test_a_none_part_is_counted(self):
        stats = {}
        build_binary_mask_coco_all(self._with_segmentation([None]), stats=stats)
        assert stats["dropped_regions"] == 1

    def test_a_none_part_beside_a_good_one_is_partial(self):
        stats = {}
        build_binary_mask_coco_all(
            self._with_segmentation([None, _flat_polygon(256, 256)]), stats=stats)
        assert (stats["dropped_regions"], stats["partial_regions"]) == (0, 1)

    def test_a_region_with_no_segmentation_is_counted(self):
        sample = _hf_row_with([_flat_polygon(256, 256)])
        region = sample["annotations"]["regions"][0]
        region["segmentation"] = None
        region.pop("visible_mask", None)
        stats = {}
        build_binary_mask_coco_all(sample, stats=stats)
        assert stats["dropped_regions"] == 1

    def test_a_region_that_is_not_a_dict_is_counted(self):
        sample = _hf_row_with([_flat_polygon(256, 256)])
        sample["annotations"]["regions"] = ["not a region"]
        stats = {}
        build_binary_mask_coco_all(sample, stats=stats)
        assert stats["dropped_regions"] == 1

    def test_stuff_regions_are_excluded_not_dropped(self):
        sample = _hf_row_with([_flat_polygon(256, 256)])
        sample["annotations"]["regions"][0]["is_stuff"] = True
        stats = {}
        build_binary_mask_coco_all(sample, stats=stats)
        assert stats["dropped_regions"] == 0
