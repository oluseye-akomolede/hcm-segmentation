"""Item 2: the run lifecycle.

``train_hcm_imagenet1k`` must end its MLflow run exactly once, with the status
that matches what actually happened. The original code only reached
``register(...)`` on the success path, which is why 111 runs are stuck RUNNING.
"""

import pytest
import torch

from ecs_hcm_model.core.trainer import (
    train_hcm_imagenet1k,
    KilledException,
)

from conftest import (
    TinyClassifier,
    RecordingLogger,
    RecordingUploader,
    trainer_source as _trainer_source,
)


def _patch_trainer(monkeypatch, logger, uploader):
    """Wire the trainer to the fakes, and skip anything that needs a real env."""
    monkeypatch.setattr(
        "ecs_hcm_model.core.trainer.logger_config", lambda **kw: {"fake": True}
    )
    monkeypatch.setattr(
        "ecs_hcm_model.core.trainer.MLFlowHCMLogger", lambda **kw: logger
    )
    monkeypatch.setattr(
        "ecs_hcm_model.core.trainer.ModelUploaderAndDownloader",
        lambda **kw: uploader,
    )
    monkeypatch.setattr(
        "ecs_hcm_model.core.trainer.get_training_dataloader",
        lambda **kw: [torch.randn(1)],
    )
    monkeypatch.setattr(
        "ecs_hcm_model.core.trainer.get_validation_dataloader",
        lambda **kw: [torch.randn(1)],
    )
    monkeypatch.setattr(
        "ecs_hcm_model.core.trainer.load_hcm_model", lambda **kw: TinyClassifier()
    )
    monkeypatch.setattr("ecs_hcm_model.core.trainer.set_env_vars", lambda *a, **k: None)


def _fake_train_result():
    """A plausible train_model_for_epochs result whose best_state loads cleanly."""
    return {
        "best_epoch": 0,
        "best_score": 1.0,
        "best_state": TinyClassifier().state_dict(),
        "best_metrics": {"val_acc1": 1.0, "val_acc5": 2.0},
        "best_train_loss": 0.1,
        "last_epoch": 0,
    }


def _run(**kwargs):
    defaults = dict(
        max_epochs=1,
        batch_size=2,
        test_mode=True,
    )
    defaults.update(kwargs)
    return train_hcm_imagenet1k(**defaults)


class TestSuccessPath:
    def test_success_ends_finished(self, monkeypatch):
        logger = RecordingLogger()
        uploader = RecordingUploader()
        _patch_trainer(monkeypatch, logger, uploader)
        monkeypatch.setattr(
            "ecs_hcm_model.core.trainer.train_model_for_epochs",
            lambda **kw: _fake_train_result(),
        )

        _run()

        assert logger.finish_statuses() == ["FINISHED"], (
            "a clean run must end exactly once, as FINISHED"
        )

    def test_exception_ends_failed_not_running(self, monkeypatch):
        """The bug: the except path re-raised without ending the run."""
        logger = RecordingLogger()
        uploader = RecordingUploader()
        _patch_trainer(monkeypatch, logger, uploader)

        def boom(**kw):
            raise RuntimeError("training exploded mid-epoch")

        monkeypatch.setattr(
            "ecs_hcm_model.core.trainer.train_model_for_epochs", boom
        )

        with pytest.raises(RuntimeError, match="training exploded"):
            _run()

        assert logger.finish_statuses() == ["FAILED"], (
            f"an exception must end the run FAILED, got {logger.registered}"
        )
        assert logger.registered[0][2], "FAILED run must carry an exc_msg"
        assert "training exploded" in logger.registered[0][2]

    def test_killed_exception_ends_killed(self, monkeypatch):
        logger = RecordingLogger()
        uploader = RecordingUploader()
        _patch_trainer(monkeypatch, logger, uploader)

        def killed(**kw):
            raise KilledException("SIGTERM received")

        monkeypatch.setattr("ecs_hcm_model.core.trainer.train_model_for_epochs", killed)

        with pytest.raises(KilledException):
            _run()

        assert logger.finish_statuses() == ["KILLED"]

    def test_run_ends_exactly_once_on_success(self, monkeypatch):
        """register + finally must not double-end the run."""
        logger = RecordingLogger()
        uploader = RecordingUploader()
        _patch_trainer(monkeypatch, logger, uploader)
        monkeypatch.setattr(
            "ecs_hcm_model.core.trainer.train_model_for_epochs",
            lambda **kw: _fake_train_result(),
        )

        _run()
        assert len(logger.registered) == 1, (
            f"run ended {len(logger.registered)} times: {logger.registered}"
        )


class TestUploadFailurePath:
    """3: an upload failure must end the run FAILED, not FINISHED."""

    def test_upload_failure_ends_failed(self, monkeypatch):
        logger = RecordingLogger()
        uploader = RecordingUploader(fail=True)
        _patch_trainer(monkeypatch, logger, uploader)
        monkeypatch.setattr(
            "ecs_hcm_model.core.trainer.train_model_for_epochs",
            lambda **kw: _fake_train_result(),
        )

        with pytest.raises(RuntimeError, match="simulated upload failure"):
            _run()

        assert logger.finish_statuses() == ["FAILED"], (
            "a swallowed upload error used to register FINISHED with uri=None"
        )

    def test_swallowed_upload_never_registers_finished(self, monkeypatch):
        """Guard the specific old shape: FINISHED with model_uri=None."""
        logger = RecordingLogger()
        uploader = RecordingUploader(fail=True)
        _patch_trainer(monkeypatch, logger, uploader)
        monkeypatch.setattr(
            "ecs_hcm_model.core.trainer.train_model_for_epochs",
            lambda **kw: _fake_train_result(),
        )

        with pytest.raises(RuntimeError):
            _run()

        for uri, status, _ in logger.registered:
            assert not (status == "FINISHED" and uri is None), (
                "FINISHED with no artifact is exactly the old bug"
            )


class TestFinishStatusIsValidated:
    """2b: the invalid 'INTERRUPTED_SAVED' status raised inside the handler."""

    def test_no_invalid_finish_status_in_module(self):
        # Resolved from this file, not the CWD: running pytest from anywhere
        # other than the repo root used to fail with FileNotFoundError here.
        src = _trainer_source()
        assert "INTERRUPTED_SAVED" not in src, (
            "INTERRUPTED_SAVED is not a valid status and raised ValueError "
            "inside the error handler, masking the original exception"
        )


class TestBestUploadLeavesTheModelAlone:
    """ba2315eb: the best upload swaps in the best weights and must swap back.

    ``model.state_dict()`` returns the live tensors, so a "saved" current state
    taken that way is overwritten by the best-weights load and the restore is a
    no-op: the caller is left holding the best weights, not the last epoch's.
    """

    def _wire(self, monkeypatch):
        torch.manual_seed(0)
        model = TinyClassifier()
        last = {k: v.detach().clone() for k, v in model.state_dict().items()}
        best = {k: torch.full_like(v, 7.0) for k, v in last.items()}
        seen = {}

        class SnapshotUploader(RecordingUploader):
            def upload_model(self, model, model_name, architecture=None, metadata=None):
                seen[model_name] = {
                    k: v.detach().clone() for k, v in model.state_dict().items()
                }
                return super().upload_model(model, model_name, architecture, metadata)

        _patch_trainer(monkeypatch, RecordingLogger(), SnapshotUploader())
        monkeypatch.setattr(
            "ecs_hcm_model.core.trainer.load_hcm_model", lambda **kw: model
        )
        result = _fake_train_result()
        result["best_state"] = best
        monkeypatch.setattr(
            "ecs_hcm_model.core.trainer.train_model_for_epochs", lambda **kw: result
        )
        return model, last, best, seen

    def test_the_best_checkpoint_carries_the_best_weights(self, monkeypatch):
        _model, _last, best, seen = self._wire(monkeypatch)
        _run()
        uploaded = next(v for k, v in seen.items() if k.endswith("_best"))
        for k, v in best.items():
            assert torch.equal(uploaded[k], v), k

    def test_the_model_holds_the_last_weights_after_the_upload(self, monkeypatch):
        model, last, _best, _seen = self._wire(monkeypatch)
        _run()
        for k, v in model.state_dict().items():
            assert torch.equal(v, last[k]), (
                f"{k}: the restore reloaded the best weights instead of the last"
            )
