"""Items 2, 3, 5, 7: run lifecycle, upload failures, early stopping, OOM safety."""

import pytest
import torch

from ecs_hcm_model.core.trainer import (
    train_model_for_epochs,
    KilledException,
    EarlyStopping,
)
from ecs_hcm_model.core.model_upload_download import (
    resolve_task,
    checkpoint_target,
    TASK_CHECKPOINTS,
)

from conftest import (
    TinyClassifier,
    TinySegmenter,
    RecordingLogger,
    RecordingUploader,
    trainer_source as _trainer_source,
)


def _clf_setup():
    model = TinyClassifier(num_classes=10, in_features=8)
    train_batches = [
        (torch.randn(4, 8), torch.randint(0, 4, (4,))) for _ in range(2)
    ]
    val_batches = [
        (torch.randn(4, 8), torch.randint(0, 4, (4,))) for _ in range(2)
    ]
    return model, train_batches, val_batches


class TestEarlyStoppingOnValLoss:
    """5: early stopping must be driven by validation loss."""

    def test_step_uses_val_loss_not_train_loss(self):
        """With train loss falling but val loss rising, stopping must trigger."""
        stopper = EarlyStopping(patience=2)
        # train loss would say "keep going"; val loss is what we pass in.
        val_losses = [1.0, 1.1, 1.2, 1.3]
        stopped = [stopper.step(v) for v in val_losses]
        assert stopped[-1] is True, "rising val loss must eventually stop"

    def test_improving_val_loss_does_not_stop(self):
        stopper = EarlyStopping(patience=2)
        stopped = [stopper.step(v) for v in [1.0, 0.9, 0.8, 0.7]]
        assert not any(stopped)


class TestCheckpointTaskMapping:
    """4b: the task registry maps each task to its own bucket and name."""

    def test_resolve_task_flags(self):
        assert resolve_task() == "classification"
        assert resolve_task(coco=True) == "coco_seg"
        assert resolve_task(finetune=True) == "cocoa_seg"

    def test_finetune_wins_over_coco(self):
        # Both flags set: finetune is the more specific branch in the trainer.
        assert resolve_task(finetune=True, coco=True) == "cocoa_seg"

    def test_checkpoint_targets(self):
        assert checkpoint_target("classification") == ("models", "hcm_imagenet1k")
        assert checkpoint_target("coco_seg") == ("coco-models", "hcm_coco_seg")
        assert checkpoint_target("cocoa_seg") == ("finetuned-models", "hcm_cocoa_seg")

    def test_names_are_distinct_per_task(self):
        names = {spec["model_name"] for spec in TASK_CHECKPOINTS.values()}
        assert len(names) == len(TASK_CHECKPOINTS), "each task needs its own name"

    def test_unknown_task_raises(self):
        with pytest.raises(ValueError):
            checkpoint_target("nonsense")


class TestUploaderRaises:
    """3: upload failures must raise, not return None."""

    def test_upload_failure_raises(self):
        uploader = RecordingUploader(fail=True)
        model = TinyClassifier()
        with pytest.raises(RuntimeError, match="simulated upload failure"):
            uploader.upload_model(model, "m", architecture="hcm")

    def test_successful_upload_records_metadata(self):
        uploader = RecordingUploader()
        model = TinyClassifier()
        uri = uploader.upload_model(
            model, "m", architecture="hcm", metadata={"task": "classification"}
        )
        assert uri is not None
        assert uploader.uploads[0]["metadata"]["task"] == "classification"


class TestEpochCheckpointing:
    """4a/4c: per-epoch checkpoints, best tracking, metadata fields."""

    def test_returns_best_epoch_and_state(self, monkeypatch):
        model, train_batches, val_batches = _clf_setup()
        logger = RecordingLogger()

        monkeypatch.setattr(
            "ecs_hcm_model.core.trainer.evaluate",
            lambda **kw: {"val_loss": 0.5, "val_acc1": 42.0, "val_acc5": 88.0},
        )
        monkeypatch.setattr(
            "ecs_hcm_model.core.trainer.save_epoch_checkpoint",
            _record_checkpoint,
        )

        result = train_model_for_epochs(
            model=model,
            epochs=2,
            train_loader=train_batches,
            val_loader=val_batches,
            device=torch.device("cpu"),
            logger=logger,
            test_mode=True,
            finetune=False,
            coco=False,
            task="classification",
            checkpoint_every_epoch=True,
        )

        assert result["best_epoch"] == 0, "first epoch ties and must win (>)"
        assert result["best_score"] == 42.0
        assert result["best_state"] is not None
        assert result["last_epoch"] == 1
        assert len(_record_checkpoint.calls) == 2, "one checkpoint per epoch"
        # The best flag was set on the first epoch only.
        assert _record_checkpoint.calls[0]["is_best"] is True
        assert _record_checkpoint.calls[1]["is_best"] is False
        assert _record_checkpoint.calls[0]["is_last"] is False
        assert _record_checkpoint.calls[1]["is_last"] is True

    def test_checkpoint_metadata_has_all_required_fields(self, monkeypatch):
        model, train_batches, val_batches = _clf_setup()
        logger = RecordingLogger(run_id="run-abc")

        monkeypatch.setattr(
            "ecs_hcm_model.core.trainer.evaluate",
            lambda **kw: {"val_loss": 0.25, "val_acc1": 77.0, "val_acc5": 91.0},
        )

        captured = {}

        def fake_upload(self, model, model_name, architecture=None, metadata=None):
            captured["metadata"] = metadata
            captured["model_name"] = model_name
            captured["bucket"] = self.bucket_name
            return "some/uri"

        monkeypatch.setattr(RecordingUploader, "upload_model", fake_upload)
        monkeypatch.setattr(
            "ecs_hcm_model.core.trainer.ModelUploaderAndDownloader",
            RecordingUploader,
        )

        train_model_for_epochs(
            model=model,
            epochs=1,
            train_loader=train_batches,
            val_loader=val_batches,
            device=torch.device("cpu"),
            logger=logger,
            test_mode=False,
            finetune=False,
            coco=False,
            task="classification",
            checkpoint_every_epoch=True,
        )

        md = captured["metadata"]
        for field in [
            "mlflow_run_id", "task", "dataset_name", "epoch",
            "val_metrics", "train_loss", "val_loss", "git_commit",
            "is_best", "is_last", "is_smoke", "input_size",
        ]:
            assert field in md, f"metadata missing {field!r}"
        assert md["mlflow_run_id"] == "run-abc"
        assert md["task"] == "classification"
        assert md["dataset_name"] == "imagenet-1k"
        assert md["is_smoke"] is False
        assert captured["bucket"] == "models"
        assert captured["model_name"].startswith("hcm_imagenet1k")
        # A run that did not ask for a size records the default, so a consumer
        # reads the field rather than assuming 192.
        assert md["input_size"] == 192

    def test_checkpoint_records_the_configured_input_size(self, monkeypatch):
        """The uploaded model carries the size it trained at.

        The Jetson demo reads this instead of assuming 192, so the value has to
        be the configured one, not the default written unconditionally.
        """
        model, train_batches, val_batches = _clf_setup()
        logger = RecordingLogger(run_id="run-size")

        monkeypatch.setattr(
            "ecs_hcm_model.core.trainer.evaluate",
            lambda **kw: {"val_loss": 0.25, "val_acc1": 77.0, "val_acc5": 91.0},
        )

        captured = {}

        def fake_upload(self, model, model_name, architecture=None, metadata=None):
            captured["metadata"] = metadata
            return "some/uri"

        monkeypatch.setattr(RecordingUploader, "upload_model", fake_upload)
        monkeypatch.setattr(
            "ecs_hcm_model.core.trainer.ModelUploaderAndDownloader",
            RecordingUploader,
        )

        train_model_for_epochs(
            model=model,
            epochs=1,
            train_loader=train_batches,
            val_loader=val_batches,
            device=torch.device("cpu"),
            logger=logger,
            test_mode=False,
            finetune=False,
            coco=False,
            task="classification",
            checkpoint_every_epoch=True,
            input_size=352,
        )

        assert captured["metadata"]["input_size"] == 352

    def test_checkpoint_records_the_model_out_stride(self, monkeypatch):
        """The uploaded model carries its real backbone stride (9764159e).

        Without it the demos always report "16 (from default)". The value comes
        from the model, so a stride-8 model records 8, not the default.
        """
        model, train_batches, val_batches = _clf_setup()
        model.out_stride = 8
        logger = RecordingLogger(run_id="run-stride")

        monkeypatch.setattr(
            "ecs_hcm_model.core.trainer.evaluate",
            lambda **kw: {"val_loss": 0.25, "val_acc1": 77.0, "val_acc5": 91.0},
        )

        captured = {}

        def fake_upload(self, model, model_name, architecture=None, metadata=None):
            captured["metadata"] = metadata
            return "some/uri"

        monkeypatch.setattr(RecordingUploader, "upload_model", fake_upload)
        monkeypatch.setattr(
            "ecs_hcm_model.core.trainer.ModelUploaderAndDownloader",
            RecordingUploader,
        )

        train_model_for_epochs(
            model=model,
            epochs=1,
            train_loader=train_batches,
            val_loader=val_batches,
            device=torch.device("cpu"),
            logger=logger,
            test_mode=False,
            finetune=False,
            coco=False,
            task="classification",
            checkpoint_every_epoch=True,
        )

        assert captured["metadata"]["out_stride"] == 8

    def test_out_stride_of_unwraps_and_tolerates_absence(self):
        """A DDP-style wrapper is unwrapped; a model with no stride records None."""
        from ecs_hcm_model.core.trainer import out_stride_of

        inner = TinyClassifier(num_classes=10, in_features=8)
        inner.out_stride = 32
        wrapper = torch.nn.Module()
        wrapper.module = inner
        assert out_stride_of(wrapper) == 32
        assert out_stride_of(TinyClassifier(num_classes=10, in_features=8)) is None

    def test_smoke_run_uploads_and_flags_is_smoke(self, monkeypatch):
        """9: a test_mode (smoke) run must still exercise the upload path."""
        model, train_batches, val_batches = _clf_setup()
        logger = RecordingLogger()
        monkeypatch.setattr(
            "ecs_hcm_model.core.trainer.evaluate",
            lambda **kw: {"val_loss": 0.5, "val_acc1": 10.0, "val_acc5": 20.0},
        )
        captured = {}

        def fake_upload(self, model, model_name, architecture=None, metadata=None):
            captured["metadata"] = metadata
            return "uri"

        monkeypatch.setattr(RecordingUploader, "upload_model", fake_upload)
        monkeypatch.setattr(
            "ecs_hcm_model.core.trainer.ModelUploaderAndDownloader",
            RecordingUploader,
        )

        train_model_for_epochs(
            model=model, epochs=1,
            train_loader=train_batches, val_loader=val_batches,
            device=torch.device("cpu"), logger=logger,
            test_mode=True, finetune=False, coco=False,
            task="classification", checkpoint_every_epoch=True,
        )
        assert captured["metadata"]["is_smoke"] is True


class TestOOMSafety:
    """7: an OOM before `loss`/`output` exist must not raise UnboundLocalError."""

    def test_module_has_guarded_delete(self):
        """The unsafe `del loss, output` must be gone from both handlers."""
        # Resolved from the test file, not the CWD (see conftest.repo_path).
        src = _trainer_source()
        assert "del loss, output" not in src, (
            "unguarded `del loss, output` still present; an OOM before those "
            "names are bound would raise UnboundLocalError"
        )
        # And the guarded form must be present.
        assert "if 'loss' in locals()" in src

    def test_oom_raised_before_assignment_is_survivable(self, monkeypatch):
        """Simulate an OOM on the first forward, before loss/output are bound."""
        model = TinyClassifier(num_classes=10, in_features=8)
        logger = RecordingLogger()
        calls = {"n": 0}

        real_forward = TinyClassifier.forward

        def oom_then_ok(self, x):
            calls["n"] += 1
            if calls["n"] == 1:
                raise RuntimeError("CUDA out of memory")
            return real_forward(self, x)

        monkeypatch.setattr(TinyClassifier, "forward", oom_then_ok)
        monkeypatch.setattr(
            "ecs_hcm_model.core.trainer.evaluate",
            lambda **kw: {"val_loss": 0.5, "val_acc1": 10.0, "val_acc5": 20.0},
        )
        monkeypatch.setattr(
            "ecs_hcm_model.core.trainer.save_epoch_checkpoint",
            _record_checkpoint,
        )

        batches = [(torch.randn(4, 8), torch.randint(0, 4, (4,))) for _ in range(3)]

        # Must not raise: the OOM batch is skipped and the epoch continues.
        train_model_for_epochs(
            model=model, epochs=1,
            train_loader=batches, val_loader=batches,
            device=torch.device("cpu"), logger=logger,
            test_mode=True, finetune=False, coco=False,
            task="classification", checkpoint_every_epoch=False,
        )


class TestSegmentationPath:
    """The segmentation task logs iou/dice and reaches the right bucket."""

    def test_segmentation_epoch_logs_iou_dice(self, monkeypatch):
        model = TinySegmenter(in_features=8, h=4, w=4)
        logger = RecordingLogger()
        monkeypatch.setattr(
            "ecs_hcm_model.core.trainer.evaluate",
            lambda **kw: {"val_loss": 0.4, "val_iou": 61.0, "val_dice": 70.0},
        )
        monkeypatch.setattr(
            "ecs_hcm_model.core.trainer.save_epoch_checkpoint",
            _record_checkpoint,
        )
        seg_batches = [
            {
                "pixel_values": torch.randn(2, 8),
                "segmentation_mask": (torch.rand(2, 1, 4, 4) > 0.5).float(),
            }
            for _ in range(2)
        ]

        train_model_for_epochs(
            model=model, epochs=1,
            train_loader=seg_batches, val_loader=seg_batches,
            device=torch.device("cpu"), logger=logger,
            test_mode=True, finetune=False, coco=True,
            task="coco_seg", checkpoint_every_epoch=True,
        )

        keys = logger.metric_keys()
        assert "train_iou" in keys and "train_dice" in keys
        assert "val_iou" in keys and "val_dice" in keys
        assert not any("acc" in k for k in keys), "segmentation must not log accuracy"


# --- helpers -------------------------------------------------------------

def _record_checkpoint(**kwargs):
    _record_checkpoint.calls.append(kwargs)
    return "fake/uri"


_record_checkpoint.calls = []


@pytest.fixture(autouse=True)
def _clear_checkpoint_calls():
    _record_checkpoint.calls = []
    yield
    _record_checkpoint.calls = []


class TestInterruptedSaveIsSurvivable:
    """2b/4d: the interrupted-save path passes descriptive val_metrics entries.

    It used to build metadata with float() over every value, so the descriptive
    entries raised inside an active exception handler and masked the original
    HuggingFace error.
    """

    def test_descriptive_val_metrics_do_not_raise(self, monkeypatch, tmp_path):
        from ecs_hcm_model.core.trainer import save_epoch_checkpoint
        captured = {}

        def fake_upload(self, model, model_name, architecture=None, metadata=None):
            captured["metadata"] = metadata
            captured["model_name"] = model_name
            return "uri"

        monkeypatch.setattr(RecordingUploader, "upload_model", fake_upload)
        monkeypatch.setattr(
            "ecs_hcm_model.core.trainer.ModelUploaderAndDownloader",
            RecordingUploader,
        )

        uri = save_epoch_checkpoint(
            model=TinyClassifier(),
            task="classification",
            epoch=4,
            val_metrics={"of_epoch": 4, "interruption_reason": "HuggingFace HTTP error"},
            avg_loss=0.5,
            val_loss=None,
            run_id="run-x",
            is_best=False,
            is_last=True,
            is_smoke=False,
            lr_decay_start=2,
        )
        assert uri == "uri"
        md = captured["metadata"]
        assert md["val_metrics"]["interruption_reason"] == "HuggingFace HTTP error"
        assert md["val_loss"] is None
        assert md["epoch"] == 4
        assert md["is_last"] is True
        # The interrupted save records the start the run actually used, not the
        # function's default: a dropped argument here silently writes 5.
        assert md["lr_decay_start"] == 2
        assert captured["model_name"] == "hcm_imagenet1k_epoch4"

    def test_numeric_metrics_stay_numeric(self, monkeypatch):
        from ecs_hcm_model.core.trainer import save_epoch_checkpoint
        captured = {}
        monkeypatch.setattr(
            RecordingUploader, "upload_model",
            lambda self, model, model_name, architecture=None, metadata=None:
                captured.update(metadata=metadata) or "uri",
        )
        monkeypatch.setattr(
            "ecs_hcm_model.core.trainer.ModelUploaderAndDownloader",
            RecordingUploader,
        )
        save_epoch_checkpoint(
            model=TinyClassifier(), task="coco_seg", epoch=1,
            val_metrics={"val_loss": 0.4, "val_iou": 61.0},
            avg_loss=0.6, val_loss=0.4, is_last=True,
        )
        md = captured["metadata"]
        assert md["val_metrics"]["val_iou"] == 61.0
        assert md["train_loss"] == 0.6
        assert md["dataset_name"] == "MSCOCO"
