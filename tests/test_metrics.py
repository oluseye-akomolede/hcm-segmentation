"""Item 1: metric naming, no aliasing, and the percentage range guard."""

import pytest

from ecs_hcm_model.core.model_logger import (
    data_obj,
    TASK_CLASSIFICATION,
    TASK_SEGMENTATION,
    PercentageOutOfRangeError,
    _check_percentage,
)


class TestDistinctKeys:
    """1a: train and val metrics must not collide on the same key."""

    def test_train_and_val_use_distinct_prefixes(self, logger):
        logger.track(data_obj(1.0, 50.0, 90.0, prefix="train",
                              task=TASK_CLASSIFICATION), epoch=0)
        logger.track(data_obj(0.5, 60.0, 95.0, prefix="val",
                              task=TASK_CLASSIFICATION), epoch=0)

        keys = logger.metric_keys()
        assert "train_loss" in keys
        assert "val_loss" in keys
        assert "train_acc1" in keys
        assert "val_acc1" in keys
        # The old collision: one bare "loss"/"acc1" key written twice per epoch.
        assert "loss" not in keys
        assert "acc1" not in keys

    def test_val_does_not_overwrite_train(self, logger):
        logger.track(data_obj(1.0, 50.0, 90.0, prefix="train",
                              task=TASK_CLASSIFICATION), epoch=0)
        logger.track(data_obj(0.5, 60.0, 95.0, prefix="val",
                              task=TASK_CLASSIFICATION), epoch=0)

        values = logger.metric_values()
        assert values["train_loss"] == 1.0
        assert values["val_loss"] == 0.5
        assert values["train_acc1"] == 50.0
        assert values["val_acc1"] == 60.0

    def test_batch_metrics_use_batch_prefix(self, logger):
        logger.track_batch(data_obj(0.3, 70.0, 99.0, prefix="batch",
                                    task=TASK_CLASSIFICATION), batch_idx=5)
        keys = {k for k, _, _ in logger.batches}
        assert keys == {"batch_loss", "batch_acc1", "batch_acc5"}


class TestNoAliasing:
    """1b: classification must not report IoU/Dice, segmentation must not report acc."""

    def test_classification_logs_no_iou_or_dice(self, logger):
        logger.track(data_obj(1.0, 50.0, 90.0, prefix="train",
                              task=TASK_CLASSIFICATION), epoch=0)
        keys = logger.metric_keys()
        assert not any("iou" in k for k in keys)
        assert not any("dice" in k for k in keys)

    def test_segmentation_logs_iou_and_dice_and_no_acc(self, logger):
        logger.track(data_obj(1.0, 55.0, 65.0, IoU=55.0, Dice=65.0,
                              prefix="train", task=TASK_SEGMENTATION), epoch=0)
        keys = logger.metric_keys()
        assert "train_iou" in keys
        assert "train_dice" in keys
        assert not any("acc" in k for k in keys)

    def test_iou_is_not_silently_aliased_to_acc1(self):
        """The old data_obj defaulted IoU to acc1; that must be gone."""
        obj = data_obj(1.0, 12.0, 34.0, task=TASK_SEGMENTATION)
        assert obj.IoU is None, "IoU must not default to acc1"
        assert obj.Dice is None, "Dice must not default to acc5"

    def test_as_metrics_omits_none_values_for_segmentation(self):
        # A segmentation data_obj without IoU/Dice yields no iou/dice entries
        # rather than fabricated ones.
        obj = data_obj(1.0, None, None, task=TASK_SEGMENTATION)
        metrics = obj.as_metrics()
        assert metrics["iou"] is None
        assert metrics["dice"] is None


class TestPercentageGuard:
    """1c: a logged percentage outside [0, 100] must raise."""

    @pytest.mark.parametrize("bad", [6006.0, -1.0, 100.0001, float("inf"), float("nan")])
    def test_out_of_range_raises(self, bad):
        with pytest.raises(PercentageOutOfRangeError):
            _check_percentage("val_acc1", bad)

    @pytest.mark.parametrize("good", [0.0, 50.0, 100.0])
    def test_in_range_passes(self, good):
        _check_percentage("val_acc1", good)  # must not raise

    def test_none_is_allowed(self):
        _check_percentage("val_acc1", None)  # must not raise

    def test_double_scaled_value_is_caught_via_logger(self, logger):
        """The real bug: acc1=6006 from a fraction scaled by 100 twice."""
        with pytest.raises(PercentageOutOfRangeError):
            logger.track(
                data_obj(1.0, 6006.0, 90.0, prefix="val", task=TASK_CLASSIFICATION),
                epoch=0,
            )

    def test_fraction_that_was_never_scaled_is_accepted_as_percent(self, logger):
        """A 0.85 fraction logged as a percentage is in-range but suspicious.

        We do not reject it (it is a legal percentage), but the guard exists so
        the *out*-of-range direction -- the one that actually happened -- fails.
        """
        logger.track(data_obj(1.0, 0.85, 0.99, prefix="val",
                              task=TASK_CLASSIFICATION), epoch=0)
        assert logger.metric_values()["val_acc1"] == 0.85

    def test_loss_is_not_percentage_checked(self, logger):
        # Loss may legitimately exceed 100 and must not trip the guard.
        logger.track(data_obj(500.0, 50.0, 90.0, prefix="train",
                              task=TASK_CLASSIFICATION), epoch=0)
        assert logger.metric_values()["train_loss"] == 500.0


class TestConcreteTaskNames:
    """Regression: the registry's concrete task names must read as segmentation.

    The trainer passes "coco_seg"/"cocoa_seg" (from resolve_task), while the
    generic task kind is "segmentation". Comparing the two directly is always
    False, which logged segmentation epochs under the accuracy keys -- found by
    an end-to-end trainer test, not by inspection.
    """

    @pytest.mark.parametrize("task", ["coco_seg", "cocoa_seg", "segmentation"])
    def test_concrete_names_are_segmentation(self, task):
        obj = data_obj(1.0, 41.0, 58.0, prefix="train", task=task)
        assert obj.is_segmentation is True
        assert set(obj.as_metrics()) == {"iou", "dice", "loss"}

    def test_registry_names_and_generic_kind_agree(self):
        from ecs_hcm_model.core.model_upload_download import TASK_CHECKPOINTS
        from ecs_hcm_model.core.model_logger import normalise_task
        for name in TASK_CHECKPOINTS:
            expected = TASK_SEGMENTATION if name != "classification" else TASK_CLASSIFICATION
            assert normalise_task(name) == expected

    def test_positional_segmentation_values_are_logged(self, logger):
        """The trainer passes IoU/Dice positionally as ac1/ac5."""
        logger.track(data_obj(0.5, 41.0, 58.0, prefix="val", task="coco_seg"), 0)
        values = logger.metric_values()
        assert values["val_iou"] == 41.0
        assert values["val_dice"] == 58.0
        assert not any("acc" in k for k in values)

    def test_explicit_keywords_win_over_positional(self, logger):
        logger.track(data_obj(0.5, 1.0, 2.0, IoU=41.0, Dice=58.0,
                              prefix="val", task="coco_seg"), 0)
        values = logger.metric_values()
        assert values["val_iou"] == 41.0
        assert values["val_dice"] == 58.0
