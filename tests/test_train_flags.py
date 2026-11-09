"""Item 4 of the row: the three new settings reach the trainer and are recorded.

Wiring the flags through ``main.py`` and the ``train_hcm_1k`` mapping is only
half of it -- a run's settings are part of what the run *is*, so they are logged
as MLflow run parameters, identity values included, on the same reasoning as the
mix ratio logged as a metric.
"""
from __future__ import annotations

import sys

import pytest
import torch

import main as cli
from ecs_hcm_model.core import trainer as T
from tests.conftest import (
    RecordingLogger,
    RecordingUploader,
    TinyClassifier,
    make_classification_batches,
)


# --- the command line ---------------------------------------------------------

def _parse(argv):
    return cli.setup_subpparsers().parse_args(argv)


def test_the_flags_default_to_off_and_one_step():
    args = _parse(["train_hcm_imagenet1k"])

    assert args.aug_scale_jitter is None
    assert args.aug_color_jitter is None
    assert args.grad_accum_steps == 1


def test_the_parser_reads_the_three_flags():
    args = _parse([
        "train_hcm_imagenet1k", "--coco",
        "--aug-scale-jitter", "0.75", "1.5",
        "--aug-color-jitter", "0.3",
        "--grad-accum-steps", "4",
    ])

    assert tuple(args.aug_scale_jitter) == (0.75, 1.5)
    assert args.aug_color_jitter == 0.3
    assert args.grad_accum_steps == 4


def test_the_mapping_forwards_the_three_settings(monkeypatch):
    seen = {}
    monkeypatch.setattr(cli, "run_train_hcm_imagenet1k",
                        lambda **kwargs: seen.update(kwargs))

    cli.train_hcm_1k(_parse([
        "train_hcm_imagenet1k", "--coco",
        "--aug-scale-jitter", "0.75", "1.5",
        "--aug-color-jitter", "0.3",
        "--grad-accum-steps", "4",
    ]))

    assert seen["scale_jitter"] == (0.75, 1.5)
    assert seen["color_jitter"] == 0.3
    assert seen["grad_accum_steps"] == 4


def test_no_jitter_is_forwarded_as_none(monkeypatch):
    seen = {}
    monkeypatch.setattr(cli, "run_train_hcm_imagenet1k",
                        lambda **kwargs: seen.update(kwargs))

    cli.train_hcm_1k(_parse(["train_hcm_imagenet1k"]))

    assert seen["scale_jitter"] is None
    assert seen["color_jitter"] is None
    assert seen["grad_accum_steps"] == 1


def _usage_error(monkeypatch, argv, capsys):
    monkeypatch.setattr(sys, "argv", ["main.py", *argv])
    with pytest.raises(SystemExit) as caught:
        cli.main()
    return caught.value.code, capsys.readouterr().err


def test_a_step_count_below_one_is_a_usage_error(monkeypatch, capsys):
    code, err = _usage_error(
        monkeypatch, ["train_hcm_imagenet1k", "--coco", "--grad-accum-steps", "0"], capsys)

    assert code == 2
    assert "grad-accum-steps" in err


def test_a_backwards_scale_range_is_a_usage_error(monkeypatch, capsys):
    # LO above HI can never be drawn from, so it is refused rather than clamped.
    code, err = _usage_error(
        monkeypatch,
        ["train_hcm_imagenet1k", "--coco", "--aug-scale-jitter", "1.5", "0.75"],
        capsys,
    )

    assert code == 2
    assert "aug-scale-jitter" in err


def test_a_jitter_on_the_imagenet_path_is_a_usage_error(monkeypatch, capsys):
    # The ImageNet loader takes no jitter, so accepting the flag would silently
    # run without it.
    code, err = _usage_error(
        monkeypatch, ["train_hcm_imagenet1k", "--aug-color-jitter", "0.3"], capsys)

    assert code == 2
    assert "imagenet" in err.lower() or "COCO" in err


# --- what the run records -----------------------------------------------------

def _one_tiny_epoch(monkeypatch, logger, **kwargs):
    class _NeverStops:
        def __init__(self, *a, **kw):
            pass

        def step(self, current_loss):
            return False

    monkeypatch.setattr(T, "EarlyStopping", _NeverStops)
    torch.manual_seed(0)
    model = TinyClassifier()
    return T.train_model_for_epochs(
        model,
        epochs=1,
        train_loader=make_classification_batches(4, batch_size=4),
        val_loader=make_classification_batches(2, batch_size=2),
        device="cpu",
        logger=logger,
        test_mode=False,
        warmup=False,
        task="classification",
        **kwargs,
    )


def test_the_settings_are_logged_as_run_parameters(monkeypatch):
    logger = RecordingLogger()

    _one_tiny_epoch(
        monkeypatch, logger,
        augmentation={
            "scale_jitter": (0.75, 1.5), "color_jitter": 0.3, "grad_accum_steps": 4,
        },
        grad_accum_steps=4,
    )

    assert logger.params["aug_scale_jitter_lo"] == 0.75
    assert logger.params["aug_scale_jitter_hi"] == 1.5
    assert logger.params["aug_color_jitter"] == 0.3
    assert logger.params["grad_accum_steps"] == 4


def test_a_run_with_the_flags_off_logs_identity_values(monkeypatch):
    # "Off" has to be in the run's record too, or a later reader cannot tell an
    # unjittered run from one whose settings were lost.
    logger = RecordingLogger()

    _one_tiny_epoch(monkeypatch, logger)

    assert logger.params["aug_scale_jitter_lo"] == 1.0
    assert logger.params["aug_scale_jitter_hi"] == 1.0
    assert logger.params["aug_color_jitter"] == 0.0
    assert logger.params["grad_accum_steps"] == 1


# --- the entry point builds the loaders with the flags ------------------------

class _LoaderSpy:
    """Records the kwargs each loader factory is called with."""

    def __init__(self):
        self.train = []
        self.val = []

    def train_loader(self, **kwargs):
        self.train.append(kwargs)
        return [torch.randn(1)]

    def val_loader(self, **kwargs):
        self.val.append(kwargs)
        return [torch.randn(1)]


def _fake_train_result():
    return {
        "best_epoch": 0,
        "best_score": 1.0,
        "best_state": TinyClassifier().state_dict(),
        "best_metrics": {"val_acc1": 1.0, "val_acc5": 2.0},
        "best_train_loss": 0.1,
        "last_epoch": 0,
    }


def _drive_entry(monkeypatch, spy, **flags):
    """Run train_hcm_imagenet1k with stub storage and no downloads."""
    uploader = RecordingUploader()
    monkeypatch.setattr(T, "set_env_vars", lambda *a, **k: None)
    monkeypatch.setattr(T, "logger_config", lambda **kw: {"fake": True})
    monkeypatch.setattr(T, "MLFlowHCMLogger", lambda **kw: RecordingLogger())
    monkeypatch.setattr(T, "ModelUploaderAndDownloader", lambda **kw: uploader)
    monkeypatch.setattr(T, "load_hcm_model", lambda **kw: TinyClassifier())
    monkeypatch.setattr(T, "train_model_for_epochs", lambda **kw: _fake_train_result())
    monkeypatch.setattr(T, "get_training_dataloader", spy.train_loader)
    monkeypatch.setattr(T, "get_validation_dataloader", spy.val_loader)

    T.train_hcm_imagenet1k(
        new_model=True, finetune=True, max_epochs=1, batch_size=2, **flags
    )
    return uploader


def test_the_jitter_reaches_the_training_loader_never_the_val_loader(monkeypatch):
    # The mix ratio adds a second training loader; both must carry the jitter,
    # and the val loaders must stay jitter-free or the metric stops comparing.
    spy = _LoaderSpy()

    _drive_entry(monkeypatch, spy, scale_jitter=(0.75, 1.5), color_jitter=0.3,
                 coco_mix_ratio=0.5)

    assert len(spy.train) >= 2, "finetune + mix build two training loaders"
    for kwargs in spy.train:
        assert kwargs["scale_jitter"] == (0.75, 1.5)
        assert kwargs["color_jitter"] == 0.3
    assert spy.val, "the entry point must build a val loader"
    for kwargs in spy.val:
        assert kwargs.get("scale_jitter") is None
        assert kwargs.get("color_jitter") is None


def test_a_run_with_the_flags_off_passes_no_jitter(monkeypatch):
    spy = _LoaderSpy()

    _drive_entry(monkeypatch, spy)

    assert spy.train, "the entry point must build a training loader"
    for kwargs in spy.train + spy.val:
        assert kwargs.get("scale_jitter") is None
        assert kwargs.get("color_jitter") is None


def test_the_best_upload_says_how_the_run_was_trained(monkeypatch):
    # The promoted model on the Jetson has to say what produced it, or a
    # comparison cannot tell a jittered run from a plain one.
    spy = _LoaderSpy()

    uploader = _drive_entry(monkeypatch, spy, scale_jitter=(0.75, 1.5),
                            color_jitter=0.3, grad_accum_steps=4)

    best = [u for u in uploader.uploads if u["model_name"].endswith("_best")]
    assert len(best) == 1
    assert best[0]["metadata"]["augmentation"] == {
        "scale_jitter": [0.75, 1.5],
        "color_jitter": 0.3,
        "grad_accum_steps": 4,
    }
