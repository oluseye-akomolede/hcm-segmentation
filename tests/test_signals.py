"""Item 1: SIGTERM/SIGINT must end the MLflow run KILLED, not strand it RUNNING.

Airflow deletes the pod on cancellation, which delivers SIGTERM. The default
disposition terminates the process immediately, so the run created moments
earlier stayed RUNNING forever -- the same class of leak as the 111 stuck runs,
arriving by a different route.
"""

import os
import signal

import pytest
import torch

from ecs_hcm_model.core.trainer import (
    KilledException,
    SignalKilled,
    install_kill_handlers,
    restore_kill_handlers,
    train_hcm_imagenet1k,
)

from conftest import TinyClassifier, RecordingLogger, RecordingUploader


def _patch_trainer(monkeypatch, logger, uploader):
    monkeypatch.setattr("ecs_hcm_model.core.trainer.logger_config", lambda **kw: {"fake": True})
    monkeypatch.setattr("ecs_hcm_model.core.trainer.MLFlowHCMLogger", lambda **kw: logger)
    monkeypatch.setattr(
        "ecs_hcm_model.core.trainer.ModelUploaderAndDownloader", lambda **kw: uploader
    )
    monkeypatch.setattr(
        "ecs_hcm_model.core.trainer.get_training_dataloader", lambda **kw: [torch.randn(1)]
    )
    monkeypatch.setattr(
        "ecs_hcm_model.core.trainer.get_validation_dataloader", lambda **kw: [torch.randn(1)]
    )
    monkeypatch.setattr("ecs_hcm_model.core.trainer.load_hcm_model", lambda **kw: TinyClassifier())
    monkeypatch.setattr("ecs_hcm_model.core.trainer.set_env_vars", lambda *a, **k: None)


@pytest.fixture(autouse=True)
def _restore_handlers():
    """Never leave a test's signal handler installed in the pytest process."""
    yield
    restore_kill_handlers()


class TestSignalKilledException:
    def test_carries_signal_name(self):
        exc = SignalKilled(signal.SIGTERM)
        assert exc.signame == "SIGTERM"
        assert "SIGTERM" in str(exc)
        assert "15" in str(exc)

    def test_is_a_killed_exception(self):
        """: the existing except KilledException path is what ends the run."""
        assert issubclass(SignalKilled, KilledException)


class TestHandlerInstallation:
    def test_installs_for_sigterm_and_sigint(self):
        previous = {s: signal.getsignal(s) for s in (signal.SIGTERM, signal.SIGINT)}
        try:
            assert install_kill_handlers() is True
            assert signal.getsignal(signal.SIGTERM) is not previous[signal.SIGTERM]
            assert signal.getsignal(signal.SIGINT) is not previous[signal.SIGINT]
        finally:
            restore_kill_handlers()

    def test_restore_puts_them_back(self):
        previous = {s: signal.getsignal(s) for s in (signal.SIGTERM, signal.SIGINT)}
        install_kill_handlers()
        restore_kill_handlers()
        assert signal.getsignal(signal.SIGTERM) is previous[signal.SIGTERM]
        assert signal.getsignal(signal.SIGINT) is previous[signal.SIGINT]


class TestSignalBeforeRunStarts:
    """(b) a signal before the run exists must exit cleanly, creating no run."""

    def test_signal_before_run_creates_no_run(self, monkeypatch):
        logger = RecordingLogger()
        uploader = RecordingUploader()
        _patch_trainer(monkeypatch, logger, uploader)

        # The signal lands before MLFlowHCMLogger is even constructed.
        def boom(**kw):
            raise SignalKilled(signal.SIGTERM)

        monkeypatch.setattr("ecs_hcm_model.core.trainer.logger_config", boom)

        with pytest.raises(SignalKilled):
            train_hcm_imagenet1k(max_epochs=1, batch_size=2, test_mode=True)

        assert logger.registered == [], (
            "a signal before the run started must not create or end a run"
        )


class TestSignalMidEpoch:
    """(a)/(c) mid-epoch: run ends KILLED exactly once, with the reason tagged."""

    def test_mid_epoch_signal_ends_killed_once(self, monkeypatch):
        logger = RecordingLogger()
        uploader = RecordingUploader()
        _patch_trainer(monkeypatch, logger, uploader)

        def killed(**kw):
            raise SignalKilled(signal.SIGTERM)

        monkeypatch.setattr("ecs_hcm_model.core.trainer.train_model_for_epochs", killed)

        with pytest.raises(SignalKilled):
            train_hcm_imagenet1k(max_epochs=1, batch_size=2, test_mode=True)

        assert logger.finish_statuses() == ["KILLED"], (
            f"expected exactly one KILLED end, got {logger.registered}"
        )
        assert logger.kill_reason == "SIGTERM", (
            "the KILLED run must record why it was killed"
        )

    def test_killed_is_not_downgraded_to_failed_by_finally(self, monkeypatch):
        """The finally calls _end_run('FAILED'); it must not overwrite KILLED."""
        logger = RecordingLogger()
        uploader = RecordingUploader()
        _patch_trainer(monkeypatch, logger, uploader)

        def killed(**kw):
            raise SignalKilled(signal.SIGINT)

        monkeypatch.setattr("ecs_hcm_model.core.trainer.train_model_for_epochs", killed)

        with pytest.raises(SignalKilled):
            train_hcm_imagenet1k(max_epochs=1, batch_size=2, test_mode=True)

        assert "FAILED" not in logger.finish_statuses()
        assert len(logger.registered) == 1, "no double end_run"

    def test_real_sigterm_to_self_ends_killed(self, monkeypatch):
        """Drive an actual os.kill so the handler wiring is exercised, not just
        the exception it raises."""
        logger = RecordingLogger()
        uploader = RecordingUploader()
        _patch_trainer(monkeypatch, logger, uploader)

        def killed_by_real_signal(**kw):
            os.kill(os.getpid(), signal.SIGTERM)
            raise AssertionError("SIGTERM handler did not raise")

        monkeypatch.setattr(
            "ecs_hcm_model.core.trainer.train_model_for_epochs", killed_by_real_signal
        )

        with pytest.raises(SignalKilled):
            train_hcm_imagenet1k(max_epochs=1, batch_size=2, test_mode=True)

        assert logger.finish_statuses() == ["KILLED"]
        assert logger.kill_reason == "SIGTERM"

    def test_handlers_are_restored_after_a_signal(self, monkeypatch):
        """A killed run must not leave the handler installed behind it."""
        logger = RecordingLogger()
        uploader = RecordingUploader()
        _patch_trainer(monkeypatch, logger, uploader)
        before = signal.getsignal(signal.SIGTERM)

        def killed(**kw):
            raise SignalKilled(signal.SIGTERM)

        monkeypatch.setattr("ecs_hcm_model.core.trainer.train_model_for_epochs", killed)

        with pytest.raises(SignalKilled):
            train_hcm_imagenet1k(max_epochs=1, batch_size=2, test_mode=True)

        assert signal.getsignal(signal.SIGTERM) is before


class TestCheckpointUploadFailureTag:
    """Item 3: failed checkpoint uploads are counted and tagged, not just printed."""

    def test_failures_are_counted_and_tagged(self, monkeypatch):
        from ecs_hcm_model.core.trainer import train_model_for_epochs
        logger = RecordingLogger()

        monkeypatch.setattr(
            "ecs_hcm_model.core.trainer.evaluate",
            lambda **kw: {"val_loss": 0.5, "val_acc1": 42.0, "val_acc5": 88.0},
        )
        monkeypatch.setattr(
            "ecs_hcm_model.core.trainer.save_epoch_checkpoint",
            lambda **kw: (_ for _ in ()).throw(RuntimeError("minio down")),
        )

        model = TinyClassifier()
        batches = [(torch.randn(4, 8), torch.randint(0, 10, (4,))) for _ in range(2)]

        train_model_for_epochs(
            model=model, epochs=2, train_loader=batches, val_loader=batches,
            device=torch.device("cpu"), logger=logger, test_mode=True,
            finetune=False, coco=False, task="classification",
            checkpoint_every_epoch=True,
        )

        assert logger.checkpoint_upload_failures == 2, (
            "both epochs failed to save; the run must record that"
        )

    def test_no_tag_when_nothing_failed(self, monkeypatch):
        from ecs_hcm_model.core.trainer import train_model_for_epochs
        logger = RecordingLogger()
        monkeypatch.setattr(
            "ecs_hcm_model.core.trainer.evaluate",
            lambda **kw: {"val_loss": 0.5, "val_acc1": 42.0, "val_acc5": 88.0},
        )
        monkeypatch.setattr(
            "ecs_hcm_model.core.trainer.save_epoch_checkpoint", lambda **kw: "uri"
        )
        model = TinyClassifier()
        batches = [(torch.randn(4, 8), torch.randint(0, 10, (4,))) for _ in range(2)]

        train_model_for_epochs(
            model=model, epochs=1, train_loader=batches, val_loader=batches,
            device=torch.device("cpu"), logger=logger, test_mode=True,
            finetune=False, coco=False, task="classification",
            checkpoint_every_epoch=True,
        )
        assert logger.checkpoint_upload_failures is None
