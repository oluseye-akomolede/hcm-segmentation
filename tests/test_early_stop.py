"""Which validation metric stops a run, and the defaults that must not move.

Early stopping used to be hardwired to validation loss with patience five. The
BDD100K refine selects its best epoch on validation IoU, so stopping on a loss
that keeps drifting is the wrong signal. The monitor is now a choice, and the
choice has to be the *metric actually handed to the stopper* -- not merely a
constructor argument -- which is why the loop-level tests script ``evaluate``
and read back what the stopper was stepped with.
"""

import pytest
import torch

from ecs_hcm_model.core import trainer as T
from ecs_hcm_model.core.trainer import (
    EARLY_STOP_MONITORS,
    EarlyStopping,
    train_model_for_epochs,
)

from conftest import RecordingLogger, TinySegmenter, make_segmentation_batches


class _Recorder:
    """Stands in for EarlyStopping, recording how it was built and stepped."""

    built = []
    last = None

    def __init__(self, *args, **kwargs):
        self.patience = kwargs.get("patience", 5)
        self.monitor = kwargs.get("monitor", "val_loss")
        self.steps = []
        _Recorder.built.append(self)
        _Recorder.last = self

    def step(self, value):
        self.steps.append(value)
        return False  # never stops: the test reads the sequence instead


@pytest.fixture(autouse=True)
def _clean_seams(monkeypatch):
    _Recorder.built = []
    _Recorder.last = None
    monkeypatch.setattr(T, "save_epoch_checkpoint", lambda **kw: None)
    yield


def _script_evaluate(monkeypatch, returns):
    """One scripted validation result per epoch, in call order."""
    pending = list(returns)

    def fake_evaluate(**kwargs):
        assert pending, "evaluate was called more often than the script provides"
        return dict(pending.pop(0))

    monkeypatch.setattr(T, "evaluate", fake_evaluate)
    return pending


def _run(epochs, *, monitor="val_loss", patience=5, task="segmentation",
         mix_val_loader=None, best_on_primary=False,
         primary_suffix=T.MIX_SUFFIX_PRIMARY):
    """Run the real loop on the conftest fakes.

    The data branch keys on ``finetune or coco``; the metric names key on
    ``task``. A segmentation epoch therefore needs both, exactly as the COCO
    mix tests do it.
    """
    segmentation = task == "segmentation"
    if segmentation:
        model = TinySegmenter()
        train_loader = make_segmentation_batches(n_batches=2, batch_size=2)
        val_loader = make_segmentation_batches(n_batches=1, batch_size=2)
    else:
        from conftest import TinyClassifier, make_classification_batches

        model = TinyClassifier(num_classes=10, in_features=8)
        train_loader = make_classification_batches(n_batches=2, batch_size=2)
        val_loader = make_classification_batches(n_batches=1, batch_size=2)
    return train_model_for_epochs(
        model=model,
        epochs=epochs,
        train_loader=train_loader,
        val_loader=val_loader,
        device=torch.device("cpu"),
        logger=RecordingLogger(),
        test_mode=True,
        finetune=segmentation,
        coco=False,
        task=task,
        mix_val_loader=mix_val_loader,
        best_on_primary=best_on_primary,
        primary_suffix=primary_suffix,
        early_stop_monitor=monitor,
        early_stop_patience=patience,
    )


def _seg_metrics(val_loss, val_iou):
    return {"val_loss": val_loss, "val_iou": val_iou, "val_dice": val_iou}


class TestTheDefaultsDoNotMove:
    def test_the_monitor_list_names_both_metrics(self):
        assert EARLY_STOP_MONITORS == ("val_loss", "val_iou")

    def test_the_loop_builds_a_val_loss_stopper_with_patience_five(self, monkeypatch):
        monkeypatch.setattr(T, "EarlyStopping", _Recorder)
        _script_evaluate(monkeypatch, [_seg_metrics(1.0, 70.0), _seg_metrics(0.9, 71.0)])
        _run(2)
        assert (_Recorder.last.patience, _Recorder.last.monitor) == (5, "val_loss")

    def test_the_default_monitor_is_stepped_with_the_validation_loss(self, monkeypatch):
        monkeypatch.setattr(T, "EarlyStopping", _Recorder)
        _script_evaluate(monkeypatch, [_seg_metrics(1.0, 70.0), _seg_metrics(0.8, 71.0)])
        _run(2)
        assert _Recorder.last.steps == [1.0, 0.8], "val_loss, not val_iou"

    def test_the_val_iou_monitor_is_stepped_with_the_iou(self, monkeypatch):
        monkeypatch.setattr(T, "EarlyStopping", _Recorder)
        _script_evaluate(monkeypatch, [_seg_metrics(1.0, 70.0), _seg_metrics(0.8, 71.5)])
        _run(2, monitor="val_iou")
        assert _Recorder.last.steps == [70.0, 71.5], "val_iou, not val_loss"
        assert _Recorder.last.monitor == "val_iou"


class TestUnderAMixTheStopperWatchesTheSeriesSelectionReads:
    """A BDD refine mixes COCO in, and is selected on BDD alone (``val_iou_bdd``).

    ``merge_mixed_val_metrics`` makes the unsuffixed ``val_iou`` the mean, so a
    stopper reading that name follows a curve the best epoch is not chosen on:
    COCO degrading could stop a refine whose own IoU was still rising.
    """

    @staticmethod
    def _bdd_run(monkeypatch, epochs_metrics, *, monitor):
        """Two validation calls per epoch: the primary's, then the mix's."""
        monkeypatch.setattr(T, "EarlyStopping", _Recorder)
        _script_evaluate(monkeypatch, epochs_metrics)
        _run(2, monitor=monitor, mix_val_loader=make_segmentation_batches(
            n_batches=1, batch_size=2), best_on_primary=True,
            primary_suffix=T.MIX_SUFFIX_BDD)
        return _Recorder.last

    def test_the_iou_monitor_is_stepped_with_the_bdd_iou_not_the_mean(self, monkeypatch):
        stopper = self._bdd_run(monkeypatch, [
            _seg_metrics(1.0, 60.0), _seg_metrics(0.5, 80.0),   # mean iou 70.0
            _seg_metrics(0.9, 62.0), _seg_metrics(0.5, 79.0),   # mean iou 70.5
        ], monitor="val_iou")
        assert stopper.monitor == "val_iou"
        assert stopper.steps == [60.0, 62.0], "val_iou_bdd, not the BDD/COCO mean"

    def test_the_default_loss_monitor_reads_the_bdd_loss_too(self, monkeypatch):
        stopper = self._bdd_run(monkeypatch, [
            _seg_metrics(1.0, 60.0), _seg_metrics(0.5, 80.0),   # mean loss 0.75
            _seg_metrics(0.9, 62.0), _seg_metrics(0.5, 79.0),   # mean loss 0.70
        ], monitor="val_loss")
        assert stopper.steps == [1.0, 0.9], "val_loss_bdd, not the mean"

    def test_a_cocoa_mix_still_stops_on_the_mean(self, monkeypatch):
        """The unchanged path: selected on both datasets, so both drive the stop."""
        monkeypatch.setattr(T, "EarlyStopping", _Recorder)
        _script_evaluate(monkeypatch, [
            _seg_metrics(1.0, 60.0), _seg_metrics(0.5, 80.0),
            _seg_metrics(0.9, 62.0), _seg_metrics(0.5, 79.0),
        ])
        _run(2, monitor="val_iou", mix_val_loader=make_segmentation_batches(
            n_batches=1, batch_size=2))
        assert _Recorder.last.steps == [70.0, 70.5], "the mean of the two sets"


def test_a_monitor_the_run_does_not_produce_names_the_key_and_the_metrics(monkeypatch):
    """A classification run has no IoU series; the loop says so instead of stopping never."""
    _script_evaluate(monkeypatch, [{"val_loss": 0.5, "val_acc1": 60.0, "val_acc5": 80.0}])
    with pytest.raises(ValueError) as refused:
        _run(1, monitor="val_iou", task="classification")
    message = str(refused.value)
    assert "val_iou" in message and "val_loss" in message


class TestTheStopperItself:
    def test_a_loss_monitor_falls_and_an_iou_monitor_rises(self):
        loss = EarlyStopping(patience=1, monitor="val_loss")
        assert loss.mode == "min"
        assert [loss.step(v) for v in (1.0, 0.9)] == [False, False]

        iou = EarlyStopping(patience=1, monitor="val_iou")
        assert iou.mode == "max"
        assert [iou.step(v) for v in (70.0, 71.0)] == [False, False]

    def test_a_rising_loss_monitor_stops(self):
        stopper = EarlyStopping(patience=2, monitor="val_loss")
        assert [stopper.step(v) for v in (1.0, 1.1, 1.2, 1.3)] == [
            False, False, True, True]

    def test_a_falling_iou_monitor_stops(self):
        stopper = EarlyStopping(patience=2, monitor="val_iou")
        assert [stopper.step(v) for v in (70.0, 69.0, 68.0, 67.0)] == [
            False, False, True, True]

    def test_a_flat_metric_counts_as_non_improvement(self):
        """A plateau is what a saturated model looks like; it must count."""
        stopper = EarlyStopping(patience=1, monitor="val_iou")
        assert [stopper.step(v) for v in (70.0, 70.0, 70.0)] == [False, True, True]

    def test_patience_zero_never_stops_whatever_the_metric_does(self):
        stopper = EarlyStopping(patience=0, monitor="val_iou")
        assert not any(stopper.step(v) for v in (70.0, 60.0, 50.0, 40.0))

    def test_an_unknown_monitor_is_refused(self):
        with pytest.raises(ValueError, match="val_iou"):
            EarlyStopping(patience=5, monitor="val_dice")


class TestTheLoopHonoursTheMonitor:
    def test_falling_iou_stops_a_run_whose_loss_is_still_improving(self, monkeypatch):
        # The exact situation the flag exists for: loss says "carry on", IoU
        # says the segmentation is getting worse.
        _script_evaluate(monkeypatch, [
            _seg_metrics(1.0, 70.0),
            _seg_metrics(0.9, 69.0),
            _seg_metrics(0.8, 68.0),
        ])
        result = _run(3, monitor="val_iou", patience=1)
        assert result["last_epoch"] == 1, "stopped after the first non-improvement"

    def test_the_same_run_under_val_loss_does_not_stop(self, monkeypatch):
        _script_evaluate(monkeypatch, [
            _seg_metrics(1.0, 70.0),
            _seg_metrics(0.9, 69.0),
            _seg_metrics(0.8, 68.0),
        ])
        result = _run(3, monitor="val_loss", patience=1)
        assert result["last_epoch"] == 2, "every val_loss epoch improved"

    def test_patience_zero_runs_every_epoch(self, monkeypatch):
        _script_evaluate(monkeypatch, [
            _seg_metrics(1.0, 70.0),
            _seg_metrics(0.9, 69.0),
            _seg_metrics(0.8, 68.0),
        ])
        result = _run(3, monitor="val_iou", patience=0)
        assert result["last_epoch"] == 2


class TestTheFlagsAreRefusedWhenTheyCannotWork:
    def test_an_unknown_monitor_is_refused_before_the_first_epoch(self):
        with pytest.raises(ValueError, match="val_loss"):
            _run(1, monitor="val_dice")

    def test_a_negative_patience_is_refused(self):
        with pytest.raises(ValueError, match="patience"):
            _run(1, patience=-1)

    def test_an_iou_monitor_on_a_classification_run_names_the_missing_metric(self, monkeypatch):
        # A classification evaluate() never returns val_iou; the flag must say
        # so instead of raising a bare KeyError deep in the epoch.
        _script_evaluate(monkeypatch, [
            {"val_loss": 1.0, "val_acc1": 50.0, "val_acc5": 80.0},
        ])
        with pytest.raises(ValueError, match="val_iou"):
            _run(1, monitor="val_iou", task="classification")