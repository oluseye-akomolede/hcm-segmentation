"""The COCO mix in a COCOA refine.

The refine trains on COCOA alone, and the 2026-09-30 run lost COCO doing it
(val IoU 69.8 -> 44.5). This covers the knob that mixes COCO back in: the
per-epoch schedule, the cycling of a short COCO loader, the mean of the two
validation sets that best selection and early stopping read, the refusal of the
flag outside a refine, and the batch shapes of the two datasets.

The training-loop tests drive the real ``train_model_for_epochs`` with the
fakes from ``conftest`` and script ``evaluate``, so the loop's own arithmetic --
the mean, the best-epoch comparison, the value handed to early stopping -- is
what is under test rather than a re-implementation of it.
"""

import inspect
import random

import numpy as np
import pytest
import torch
from PIL import Image

from ecs_hcm_model.core import trainer as trainer_module
from ecs_hcm_model.core.trainer import (
    CocoaMixRatioError,
    MIX_SOURCE_MIX,
    MIX_SOURCE_PRIMARY,
    dataset_name_for,
    merge_mixed_val_metrics,
    mix_batches_for,
    mix_schedule,
    mixed_epoch_iterator,
    train_hcm_imagenet1k,
    train_model_for_epochs,
)

from conftest import RecordingLogger, TinySegmenter, make_segmentation_batches

PRIMARY_TAG = "cocoa"
MIX_TAG = "coco"

# The two spellings of an annotation a mixed epoch draws from: COCO puts a list
# of instance dicts under "annotations", COCOA a dict of regions. Both walked by
# the same builder and the same transform, which is what the mix relies on.
SQUARE = (200, 200, 64, 64)  # x, y, w, h


class _CountingLoader:
    """A re-iterable, sized stand-in for a DataLoader that tags its batches."""

    def __init__(self, n, tag):
        self.n = n
        self.tag = tag
        self.iterations = 0

    def __len__(self):
        return self.n

    def __iter__(self):
        self.iterations += 1
        for i in range(self.n):
            yield (self.tag, i)


class _UnsizedLoader:
    """A loader with no ``len``, like an IterableDataset DataLoader."""

    def __iter__(self):
        yield (MIX_TAG, 0)


def _counts(batches):
    counts = {}
    for batch in batches:
        counts[batch[0]] = counts.get(batch[0], 0) + 1
    return counts


def _tags(batches):
    return [batch[0] for batch in batches]


# --- the training loop's seams ----------------------------------------------


_CHECKPOINTS = []
_STOPPERS = []


def _record_checkpoint(**kwargs):
    _CHECKPOINTS.append(kwargs)
    return "fake/uri"


class _RecordingStopper:
    """Stands in for EarlyStopping and keeps every loss it was stepped with."""

    def __init__(self, *args, **kwargs):
        self.patience = kwargs.get("patience", 5)
        self.steps = []
        _STOPPERS.append(self)

    def step(self, loss):
        self.steps.append(float(loss))
        return False


def _script_evaluate(monkeypatch, returns):
    """Return one scripted ``evaluate`` result per call, in call order."""
    pending = list(returns)

    def fake_evaluate(**kwargs):
        assert pending, "evaluate was called more often than the script provides"
        return dict(pending.pop(0))

    monkeypatch.setattr(trainer_module, "evaluate", fake_evaluate)


@pytest.fixture(autouse=True)
def _clean_seams(monkeypatch):
    _CHECKPOINTS.clear()
    _STOPPERS.clear()
    monkeypatch.setattr(trainer_module, "save_epoch_checkpoint", _record_checkpoint)
    monkeypatch.setattr(trainer_module, "EarlyStopping", _RecordingStopper)
    yield


def _run_epochs(epochs, *, coco_mix_ratio=0.0, mixed_val=True, mixed_train=True):
    """Run a cocoa_seg refine through the real loop, on the fakes."""
    model = TinySegmenter()
    logger = RecordingLogger()
    result = train_model_for_epochs(
        model=model,
        epochs=epochs,
        train_loader=make_segmentation_batches(n_batches=4, batch_size=2),
        val_loader=make_segmentation_batches(n_batches=1, batch_size=2),
        device=torch.device("cpu"),
        logger=logger,
        test_mode=True,
        finetune=True,
        coco=False,
        task="cocoa_seg",
        checkpoint_every_epoch=True,
        coco_mix_ratio=coco_mix_ratio,
        mix_train_loader=(
            make_segmentation_batches(n_batches=4, batch_size=2) if mixed_train else None
        ),
        mix_val_loader=(
            make_segmentation_batches(n_batches=1, batch_size=2) if mixed_val else None
        ),
    )
    return result, logger


# --- the schedule -----------------------------------------------------------


class TestTheEpochSchedule:
    def test_ratio_one_alternates_the_two_datasets(self):
        assert mix_schedule(4, 4) == [
            MIX_SOURCE_PRIMARY,
            MIX_SOURCE_MIX,
            MIX_SOURCE_PRIMARY,
            MIX_SOURCE_MIX,
            MIX_SOURCE_PRIMARY,
            MIX_SOURCE_MIX,
            MIX_SOURCE_PRIMARY,
            MIX_SOURCE_MIX,
        ]

    def test_ratio_half_spreads_a_mixed_batch_after_every_second_primary(self):
        assert mix_schedule(4, 2) == [
            MIX_SOURCE_PRIMARY,
            MIX_SOURCE_MIX,
            MIX_SOURCE_PRIMARY,
            MIX_SOURCE_PRIMARY,
            MIX_SOURCE_MIX,
            MIX_SOURCE_PRIMARY,
        ]

    @pytest.mark.parametrize("n_primary,n_mix", [(10, 3), (10, 10), (7, 2), (100, 50), (2, 1)])
    def test_every_primary_batch_is_kept_and_every_mixed_batch_is_placed(self, n_primary, n_mix):
        schedule = mix_schedule(n_primary, n_mix)
        assert len(schedule) == n_primary + n_mix
        assert schedule.count(MIX_SOURCE_PRIMARY) == n_primary
        assert schedule.count(MIX_SOURCE_MIX) == n_mix

    def test_the_mixed_batches_are_not_appended_at_the_end(self):
        # Appending would be one long COCOA pass followed by the COCO batches,
        # which is the sequential training this exists to avoid.
        schedule = mix_schedule(10, 2)
        first_mix = schedule.index(MIX_SOURCE_MIX)
        assert first_mix < 10, "the first mixed batch must arrive during the pass"
        assert schedule[-1] == MIX_SOURCE_PRIMARY, "10 primary batches, 2 mixed, ends primary"

    def test_the_gap_between_mixed_batches_is_never_wider_than_the_ratio(self):
        # ratio 0.5 over 10 batches: no run of more than 4 primary batches
        # without a mixed one, so the COCO signal does not fade out mid-epoch.
        schedule = mix_schedule(10, 5)
        gaps = []
        run = 0
        for source in schedule:
            if source == MIX_SOURCE_MIX:
                gaps.append(run)
                run = 0
            else:
                run += 1
        assert max(gaps) <= 2, gaps


class TestHowManyBatches:
    def test_zero_ratio_adds_nothing(self):
        assert mix_batches_for(100, 0.0) == 0
        assert mix_batches_for(100, -1.0) == 0

    @pytest.mark.parametrize(
        "n_primary,ratio,expected",
        [(100, 1.0, 100), (100, 0.5, 50), (100, 0.25, 25), (8, 0.5, 4), (200, 0.5, 100)],
    )
    def test_the_count_is_the_ratio_of_the_primary_pass(self, n_primary, ratio, expected):
        assert mix_batches_for(n_primary, ratio) == expected

    def test_a_ratio_too_small_to_yield_a_batch_yields_none(self):
        # Honest consequence of rounding: 0.01 of a 10-batch epoch is 0.1 of a
        # batch, which rounds to no mix at all rather than to one batch.
        assert mix_batches_for(10, 0.01) == 0


# --- the iterator -----------------------------------------------------------


class TestTheMixedIterator:
    def test_a_zero_ratio_is_the_plain_loader(self):
        primary = _CountingLoader(6, PRIMARY_TAG)
        mix = _CountingLoader(2, MIX_TAG)
        batches = list(mixed_epoch_iterator(primary, mix, 0.0))
        assert _tags(batches) == [PRIMARY_TAG] * 6
        assert mix.iterations == 0, "an untouched run must never read the COCO loader"

    def test_ratio_one_takes_a_batch_from_each(self):
        primary = _CountingLoader(4, PRIMARY_TAG)
        mix = _CountingLoader(4, MIX_TAG)
        batches = list(mixed_epoch_iterator(primary, mix, 1.0))
        assert _counts(batches) == {PRIMARY_TAG: 4, MIX_TAG: 4}
        assert _tags(batches)[:4] == [
            PRIMARY_TAG,
            MIX_TAG,
            PRIMARY_TAG,
            MIX_TAG,
        ]

    def test_ratio_half_takes_half_as_many_from_coco(self):
        primary = _CountingLoader(8, PRIMARY_TAG)
        mix = _CountingLoader(4, MIX_TAG)
        batches = list(mixed_epoch_iterator(primary, mix, 0.5))
        assert _counts(batches) == {PRIMARY_TAG: 8, MIX_TAG: 4}

    def test_the_primary_pass_is_complete(self):
        primary = _CountingLoader(5, PRIMARY_TAG)
        mix = _CountingLoader(3, MIX_TAG)
        batches = list(mixed_epoch_iterator(primary, mix, 1.0))
        seen = [batch for batch in batches if batch[0] == PRIMARY_TAG]
        assert seen == [(PRIMARY_TAG, i) for i in range(5)]

    def test_a_short_coco_loader_is_cycled_not_exhausted(self):
        # MSCOCO is the smaller set, and an epoch that ran dry when COCO ran out
        # would quietly become "whatever COCO had".
        primary = _CountingLoader(10, PRIMARY_TAG)
        mix = _CountingLoader(2, MIX_TAG)
        batches = list(mixed_epoch_iterator(primary, mix, 1.0))
        assert _counts(batches) == {PRIMARY_TAG: 10, MIX_TAG: 10}
        assert mix.iterations >= 5, "a 2-batch loader must be read again to reach 10"
        assert [b for b in batches if b[0] == MIX_TAG][:4] == [
            (MIX_TAG, 0),
            (MIX_TAG, 1),
            (MIX_TAG, 0),
            (MIX_TAG, 1),
        ]

    def test_an_empty_coco_loader_is_refused_rather_than_ignored(self):
        primary = _CountingLoader(4, PRIMARY_TAG)
        with pytest.raises(ValueError, match="empty"):
            mixed_epoch_iterator(primary, _CountingLoader(0, MIX_TAG), 0.5)

    def test_a_loader_with_no_length_is_refused_with_a_clear_error(self):
        with pytest.raises(TypeError, match="sized"):
            mixed_epoch_iterator(_UnsizedLoader(), _CountingLoader(2, MIX_TAG), 0.5)


# --- the two validation sets ------------------------------------------------


def _metrics(loss, iou, dice):
    return {"val_loss": loss, "val_iou": iou, "val_dice": dice}


class TestTheTwoValidationSets:
    def test_the_plain_names_are_the_mean_of_the_two_sets(self):
        merged = merge_mixed_val_metrics(_metrics(1.0, 40.0, 50.0), _metrics(2.0, 60.0, 70.0))
        assert merged["val_loss"] == pytest.approx(1.5)
        assert merged["val_iou"] == pytest.approx(50.0)
        assert merged["val_dice"] == pytest.approx(60.0)

    def test_both_sets_are_kept_under_their_own_names(self):
        merged = merge_mixed_val_metrics(_metrics(1.0, 40.0, 50.0), _metrics(2.0, 60.0, 70.0))
        assert merged["val_iou_cocoa"] == pytest.approx(40.0)
        assert merged["val_iou_coco"] == pytest.approx(60.0)
        assert merged["val_loss_cocoa"] == pytest.approx(1.0)
        assert merged["val_loss_coco"] == pytest.approx(2.0)
        assert merged["val_dice_cocoa"] == pytest.approx(50.0)
        assert merged["val_dice_coco"] == pytest.approx(70.0)

    def test_each_sets_failed_image_count_is_logged(self, monkeypatch):
        a = dict(_metrics(1.0, 40.0, 50.0), val_failed_images=2)
        b = dict(_metrics(2.0, 60.0, 70.0), val_failed_images=3)
        _script_evaluate(monkeypatch, [a, b])
        _result, logger = _run_epochs(1, coco_mix_ratio=0.5)

        logged = [(k, v) for k, v, _ in logger.scalars if k.startswith("val_failed_images")]
        assert ("val_failed_images", 5.0) in logged
        assert ("val_failed_images_cocoa", 2.0) in logged
        assert ("val_failed_images_coco", 3.0) in logged

    def test_both_sets_are_logged_and_the_mean_is_the_tracked_metric(self, monkeypatch):
        _script_evaluate(
            monkeypatch,
            [_metrics(1.0, 40.0, 50.0), _metrics(2.0, 60.0, 70.0)],
        )
        _result, logger = _run_epochs(1, coco_mix_ratio=0.5)

        tracked = logger.metric_values()
        assert tracked["val_iou"] == pytest.approx(50.0), "the tracked IoU is the mean"
        assert tracked["val_loss"] == pytest.approx(1.5)
        assert tracked["val_dice"] == pytest.approx(60.0)

        logged = logger.scalar_values()
        assert logged["val_iou_cocoa"] == pytest.approx(40.0)
        assert logged["val_iou_coco"] == pytest.approx(60.0)
        assert logged["val_dice_cocoa"] == pytest.approx(50.0)
        assert logged["val_dice_coco"] == pytest.approx(70.0)
        assert logged["val_loss_cocoa"] == pytest.approx(1.0)
        assert logged["val_loss_coco"] == pytest.approx(2.0)

    def test_the_checkpoint_metadata_carries_both_sets(self, monkeypatch):
        _script_evaluate(
            monkeypatch,
            [_metrics(1.0, 40.0, 50.0), _metrics(2.0, 60.0, 70.0)],
        )
        _run_epochs(1, coco_mix_ratio=0.5)

        saved = _CHECKPOINTS[-1]
        assert saved["coco_mix_ratio"] == pytest.approx(0.5)
        val_metrics = saved["val_metrics"]
        assert val_metrics["val_iou_cocoa"] == pytest.approx(40.0)
        assert val_metrics["val_iou_coco"] == pytest.approx(60.0)
        assert val_metrics["val_iou"] == pytest.approx(50.0)

    def test_best_selection_reads_the_mean_not_the_refine_dataset(self, monkeypatch):
        # Epoch 0 is the better COCOA epoch (40 vs 10); epoch 1 is the better
        # mean (45 vs 25). A selection reading val_iou_cocoa would keep epoch 0.
        _script_evaluate(
            monkeypatch,
            [
                _metrics(1.0, 40.0, 44.0),
                _metrics(3.0, 10.0, 12.0),
                _metrics(5.0, 10.0, 12.0),
                _metrics(3.0, 80.0, 88.0),
            ],
        )
        result, _logger = _run_epochs(2, coco_mix_ratio=0.5)

        assert result["best_epoch"] == 1, "the mean improved even though COCOA fell"
        assert result["best_score"] == pytest.approx(45.0), "(10 + 80) / 2"

    def test_early_stopping_is_stepped_with_the_mean_loss(self, monkeypatch):
        _script_evaluate(
            monkeypatch,
            [
                _metrics(1.0, 40.0, 44.0),
                _metrics(3.0, 10.0, 12.0),
                _metrics(5.0, 10.0, 12.0),
                _metrics(3.0, 80.0, 88.0),
            ],
        )
        _run_epochs(2, coco_mix_ratio=0.5)

        # (1, 3) -> 2.0 and (5, 3) -> 4.0: neither is a value either set held.
        assert _STOPPERS[-1].steps == [pytest.approx(2.0), pytest.approx(4.0)]

    def test_a_zero_ratio_is_the_run_without_the_mix(self, monkeypatch):
        _script_evaluate(monkeypatch, [_metrics(1.0, 40.0, 44.0)])
        result, logger = _run_epochs(1, coco_mix_ratio=0.0, mixed_val=False, mixed_train=False)

        assert logger.metric_values()["val_iou"] == pytest.approx(40.0)
        assert result["best_score"] == pytest.approx(40.0)
        for name in ("val_iou_cocoa", "val_iou_coco", "val_loss_coco", "val_dice_cocoa"):
            assert name not in logger.scalar_values()
            assert name not in logger.metric_values()

    def test_a_zero_ratio_checkpoint_still_records_the_ratio(self, monkeypatch):
        # Every checkpoint carries the key, so a reader never has to treat a
        # missing ratio as "probably zero".
        _script_evaluate(monkeypatch, [_metrics(1.0, 40.0, 44.0)])
        _run_epochs(1, coco_mix_ratio=0.0, mixed_val=False, mixed_train=False)

        saved = _CHECKPOINTS[-1]
        assert "coco_mix_ratio" in saved
        assert saved["coco_mix_ratio"] == 0.0


# --- the dataset label ------------------------------------------------------


class TestTheDatasetName:
    def test_a_plain_refine_is_still_cocoa(self):
        assert dataset_name_for("cocoa_seg", 0.0) == "COCOA"

    def test_a_mixed_refine_names_both_datasets(self):
        assert dataset_name_for("cocoa_seg", 0.5) == "COCOA+MSCOCO"
        assert dataset_name_for("cocoa_seg", 1.0) == "COCOA+MSCOCO"

    def test_the_other_tasks_are_unchanged(self):
        assert dataset_name_for("coco_seg", 0.0) == "MSCOCO"
        assert dataset_name_for("classification", 0.0) == "imagenet-1k"

    def test_an_unknown_task_is_unknown(self):
        assert dataset_name_for("not_a_task", 0.0) == "unknown"

    def test_an_unknown_task_with_a_mix_still_names_the_mix(self):
        # Unreachable -- a ratio requires finetune, which is cocoa_seg -- but
        # defined: the suffix marks the mixed-in dataset, whatever the primary
        # task was, rather than silently dropping it.
        assert dataset_name_for("not_a_task", 0.5) == "unknown+MSCOCO"


# --- the flag ---------------------------------------------------------------


def _explode(*args, **kwargs):
    raise AssertionError("the ratio must be refused before any run is created")


class TestTheFlagIsRefused:
    def test_the_default_is_no_mix(self):
        default = inspect.signature(train_hcm_imagenet1k).parameters["coco_mix_ratio"]
        assert default.default == 0.0

    def test_a_negative_ratio_is_refused(self):
        with pytest.raises(CocoaMixRatioError, match="negative"):
            train_hcm_imagenet1k(coco_mix_ratio=-0.1, finetune=True)

    @pytest.mark.parametrize("ratio", [float("nan"), float("inf"), float("-inf")])
    def test_a_non_finite_ratio_is_refused(self, ratio, monkeypatch):
        # nan slips past both sign checks; inf used to fail only after the COCO
        # download. Both must be refused before a run exists.
        monkeypatch.setattr(trainer_module, "MLFlowHCMLogger", _explode)
        with pytest.raises(CocoaMixRatioError, match="finite"):
            train_hcm_imagenet1k(coco_mix_ratio=ratio, finetune=True)

    def test_a_ratio_without_finetune_is_refused(self):
        with pytest.raises(CocoaMixRatioError, match="finetune"):
            train_hcm_imagenet1k(coco_mix_ratio=0.5, finetune=False)

    def test_a_ratio_on_the_coco_only_task_is_refused(self):
        with pytest.raises(CocoaMixRatioError, match="finetune"):
            train_hcm_imagenet1k(coco_mix_ratio=0.5, coco=True, finetune=False)

    def test_the_refusal_happens_before_any_run_is_created(self, monkeypatch):
        # A late check would have opened an MLflow run and started a run that
        # cannot mean anything; this catches that by exploding on construction.
        monkeypatch.setattr(trainer_module, "MLFlowHCMLogger", _explode)
        with pytest.raises(CocoaMixRatioError):
            train_hcm_imagenet1k(coco_mix_ratio=0.5, finetune=False)

    def test_every_run_logs_its_ratio_at_step_zero_even_when_zero(self, monkeypatch):
        # Recorded unconditionally, so "no mix" is a value on the run rather
        # than an absence. Stops at the first GPU probe, after the log.
        class _Stop(BaseException):
            pass

        logged = []

        class _Logger:
            def __init__(self, *args, **kwargs):
                pass

            def set_logger(self, *args, **kwargs):
                pass

            def log_scalar(self, name, value, step=None):
                logged.append((name, value, step))

            def _end_run(self, *args, **kwargs):
                pass

        def _stop():
            raise _Stop()

        monkeypatch.setattr(trainer_module, "MLFlowHCMLogger", _Logger)
        monkeypatch.setattr(trainer_module.torch.cuda, "is_available", _stop)
        with pytest.raises(_Stop):
            train_hcm_imagenet1k(coco_mix_ratio=0.0, finetune=False)
        assert ("coco_mix_ratio", 0.0, 0) in logged

    def test_a_ratio_of_zero_is_accepted_without_a_refine_check(self, monkeypatch):
        # Zero must stay the untouched path, so it must not demand finetune.
        monkeypatch.setattr(trainer_module, "MLFlowHCMLogger", _explode)
        with pytest.raises(AssertionError, match="run is created"):
            train_hcm_imagenet1k(coco_mix_ratio=0.0, finetune=False)


# --- the two datasets' shapes ----------------------------------------------


def _coco_sample(image):
    x, y, w, h = SQUARE
    return {
        "image": image,
        "annotations": [{"bbox": [x, y, w, h], "category_id": 1, "iscrowd": 0}],
        "image_id": 7,
    }


def _cocoa_sample(image):
    x, y, w, h = SQUARE
    # One square region, spelled the COCOA way: a dict of regions, each with a
    # name and a polygon. visible_mask is preferred over segmentation when both
    # are present, and absent here.
    return {
        "image": image,
        "annotations": {
            "regions": [
                {
                    "name": "thing",
                    "is_stuff": False,
                    "segmentation": [[x, y, x + w, y, x + w, y + h, x, y + h]],
                }
            ]
        },
        "image_id": 7,
    }


class TestBothDatasetsHaveTheSameShape:
    """A mixed epoch feeds COCO batches to a loop built for COCOA."""

    def _transform(self, sample, seed=0):
        from ecs_hcm_model.core.model_utils import coco_train_transforms

        random.seed(seed)
        torch.manual_seed(seed)
        return coco_train_transforms(
            {"image": [sample["image"]], "annotations": [sample["annotations"]],
             "image_id": [sample["image_id"]]},
            input_size=192,
        )

    def _image(self):
        array = np.zeros((512, 512, 3), dtype=np.uint8)
        x, y, w, h = SQUARE
        array[y : y + h, x : x + w] = 200
        return Image.fromarray(array)

    def test_the_two_spellings_come_out_with_the_same_keys_and_shapes(self):
        image = self._image()
        coco = self._transform(_coco_sample(image))
        cocoa = self._transform(_cocoa_sample(image))

        assert set(coco) == set(cocoa)
        for key in ("pixel_values", "segmentation_mask", "image_id", "valid"):
            assert len(coco[key]) == len(cocoa[key]) == 1, key
        assert coco["pixel_values"][0].shape == cocoa["pixel_values"][0].shape == (3, 192, 192)
        assert coco["segmentation_mask"][0].shape == cocoa["segmentation_mask"][0].shape == (
            192,
            192,
        )
        assert coco["pixel_values"][0].dtype == cocoa["pixel_values"][0].dtype
        assert coco["segmentation_mask"][0].dtype == cocoa["segmentation_mask"][0].dtype
        assert coco["valid"] == cocoa["valid"] == [True]

    def test_the_same_region_yields_the_same_mask_either_way(self):
        # Same crop, same square: a bbox annotation and the matching polygon
        # must produce one mask, or a mixed epoch would train on a different
        # target depending on which dataset the batch came from.
        image = self._image()
        coco = self._transform(_coco_sample(image))
        cocoa = self._transform(_cocoa_sample(image))
        assert np.array_equal(
            coco["segmentation_mask"][0].numpy(), cocoa["segmentation_mask"][0].numpy()
        )
        assert coco["segmentation_mask"][0].sum() > 0, "the square must survive the crop"
