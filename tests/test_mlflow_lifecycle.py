"""Lifecycle assertions against a REAL local MLflow store.

The other suites fake the logger, which is what makes them fast and offline --
but a fake cannot observe the defect these tests exist for. ``mlflow.set_tag``
on an already-closed run is *silently* ignored: no exception, no warning, the
tag simply is not there. A ``RecordingLogger`` that assigns ``self.kill_reason``
records the call and cannot see that the write went nowhere, so a green suite
said nothing about the KILLED run losing its reason.

Here a real ``MLFlowHCMLogger`` drives a real MLflow file store in ``tmp_path``
and every assertion is read back with ``MlflowClient`` from persisted state.
No network, no MinIO, no GPU.
"""

import os
import signal

import mlflow
import pytest

# MLflow 3.x refuses the filesystem backend unless this is set; these tests are
# precisely about persisting to a local store, so opt in.
os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")

from ecs_hcm_model.core.model_logger import (  # noqa: E402
    MLFlowHCMLogger,
    logger_config,
)
from ecs_hcm_model.core.trainer import (  # noqa: E402
    SignalKilled,
    restore_kill_handlers,
)

from conftest import TinyClassifier  # noqa: E402


# Not 192, the trainer's default: a logger that wrote a constant instead of the
# configured size would pass a test run at the default.
_INPUT_SIZE = 352


def _real_logger(tmp_path):
    """A real logger on a real local store, with no run started yet."""
    mlflow.set_tracking_uri("file:" + str(tmp_path / "mlruns"))
    logger = MLFlowHCMLogger(
        logger_config(
            experiment_name=f"hcm-lifecycle-{tmp_path.name}",
            environment="development",
            dataset="imagenet-1k",
            max_epochs=1,
            batch_size=2,
            input_size=_INPUT_SIZE,
        )
    )
    # The constructor calls enable_system_metrics_logging(), which flips this
    # env var to True. This environment has no psutil, so MLflow would then
    # refuse to start a run -- unrelated to what is under test, so turn it back
    # off once the constructor has had its say.
    mlflow.disable_system_metrics_logging()
    return logger


def _start_logger(tmp_path):
    """A real logger with a run opened by this test, plus a reader for it."""
    logger = _real_logger(tmp_path)
    logger.set_logger(
        logger_config(
            experiment_name=f"hcm-lifecycle-{tmp_path.name}",
            environment="development",
            dataset="imagenet-1k",
            max_epochs=1,
            batch_size=2,
            input_size=_INPUT_SIZE,
        )
    )
    return logger, mlflow.MlflowClient(), logger.run_id


def _patch_trainer(monkeypatch, logger):
    """Wire the trainer onto a real logger, with everything else faked away.

    The dataloaders, the model load and ``train_model_for_epochs`` are replaced
    exactly as the other suites do -- this keeps the test offline and stops it
    reaching for an HF token. The logger is the one thing left real, because it
    is the thing under test.
    """
    monkeypatch.setattr(
        "ecs_hcm_model.core.trainer.logger_config",
        lambda **kw: logger_config(
            experiment_name=kw["experiment_name"],
            environment=kw["environment"],
            dataset=kw["dataset"],
            max_epochs=kw["max_epochs"],
            batch_size=kw["batch_size"],
            input_size=kw["input_size"],
        ),
    )
    monkeypatch.setattr("ecs_hcm_model.core.trainer.MLFlowHCMLogger", lambda **kw: logger)
    monkeypatch.setattr(
        "ecs_hcm_model.core.trainer.get_training_dataloader", lambda **kw: []
    )
    monkeypatch.setattr(
        "ecs_hcm_model.core.trainer.get_validation_dataloader", lambda **kw: []
    )
    monkeypatch.setattr(
        "ecs_hcm_model.core.trainer.load_hcm_model", lambda **kw: TinyClassifier()
    )
    monkeypatch.setattr("ecs_hcm_model.core.trainer.set_env_vars", lambda *a, **k: None)
    monkeypatch.setattr(
        "ecs_hcm_model.core.trainer.ModelUploaderAndDownloader",
        lambda **kw: _NeverCalled(),
    )


class _NeverCalled:
    """Any uploader use in these tests is a wiring mistake, not a test."""

    def upload_model(self, *a, **kw):
        raise AssertionError("the upload path is not part of these tests")


def _tags(client, run_id):
    return client.get_run(run_id).data.tags


def _status(client, run_id):
    return client.get_run(run_id).info.status


@pytest.fixture(autouse=True)
def _clean_tracking_and_signals():
    """Leave no run open and no handler installed, whatever a test did."""
    yield
    while mlflow.active_run() is not None:
        mlflow.end_run()
    restore_kill_handlers()


class TestKillReasonSurvivesTheKill:
    """The regression: tagging after ``kill()`` writes to a closed run.

    This drives the trainer's own ``KilledException`` arm rather than calling
    the logger directly. Calling ``tag_kill_reason()`` then ``kill()`` here
    would only prove that MLflow accepts a tag on an open run -- true of any
    ordering the test author picks, and blind to the defect, which lives in the
    order the *trainer* performs the two calls.
    """

    def test_trainer_kill_path_keeps_the_reason(self, tmp_path, monkeypatch):
        """A kill raised mid-training ends KILLED and keeps kill_reason."""
        import ecs_hcm_model.core.trainer as trainer

        logger = _real_logger(tmp_path)
        _patch_trainer(monkeypatch, logger)

        def die(**kw):
            raise SignalKilled(signal.SIGTERM)

        monkeypatch.setattr(trainer, "train_model_for_epochs", die)

        with pytest.raises(SignalKilled):
            trainer.train_hcm_imagenet1k(environment="development", max_epochs=1)

        run_id = logger.run_id
        client = mlflow.MlflowClient()
        assert _status(client, run_id) == "KILLED"
        assert _tags(client, run_id).get("kill_reason") == "SIGTERM", (
            "the KILLED run lost its reason: the tag was written after the run "
            "was closed, where MLflow drops it silently"
        )

    def test_tagging_after_the_kill_now_reaches_the_run(self, tmp_path):
        """The behaviour this file used to pin down as a loss is now fixed.

        The old tag path went through fluent ``mlflow.set_tag``, which drops a
        tag on a closed run silently -- so a ``kill_reason`` tagged after
        ``kill()`` vanished. Tags are addressed to the logger's own run id now,
        so they land either way. The ordering constraint the trainer works
        around is therefore no longer load-bearing, though the trainer's own
        ordering is unchanged.
        """
        logger, client, run_id = _start_logger(tmp_path)

        logger.kill()
        logger.tag_kill_reason("SIGTERM")

        assert _status(client, run_id) == "KILLED"
        assert _tags(client, run_id).get("kill_reason") == "SIGTERM", (
            "a tag written after the run closed was dropped: it is not being "
            "addressed to the run id"
        )


class TestFailureAndUploadTags:
    def test_failed_run_carries_the_error_tag(self, tmp_path):
        logger, client, run_id = _start_logger(tmp_path)

        logger.register(
            model_uri=None, finish_status="FAILED", exc_msg="simulated upload failure"
        )

        assert _status(client, run_id) == "FAILED"
        assert _tags(client, run_id).get("error") == "simulated upload failure"

    def test_checkpoint_upload_failures_tag_persists(self, tmp_path):
        """The count is written while the run is live, and survives to the end."""
        logger, client, run_id = _start_logger(tmp_path)

        logger.tag_checkpoint_upload_failures(0)
        logger.tag_checkpoint_upload_failures(3)
        logger.register(model_uri="models/hcm/ckpt", finish_status="FINISHED")

        assert _status(client, run_id) == "FINISHED"
        assert _tags(client, run_id).get("checkpoint_upload_failures") == "3"

    def test_finished_run_is_not_downgraded_by_the_finally_path(self, tmp_path):
        """The trainer's ``finally`` calls ``_end_run("FAILED")`` after a
        successful ``register``; idempotence must leave FINISHED alone."""
        logger, client, run_id = _start_logger(tmp_path)

        logger.register(model_uri="models/hcm/ckpt", finish_status="FINISHED")
        logger._end_run("FAILED")  # what the finally block does

        assert _status(client, run_id) == "FINISHED"


class TestTheRunRecordsItsInputSize:
    """The run's scale used to be batch size alone.

    Two runs of the same experiment at 192 and at 352 carry the same dataset,
    the same flags and the same batch size, so from MLflow they were the same
    run -- while the checkpoints they produced differed geometrically. The
    config in these tests is 352, not the default, so a logger that wrote a
    constant would pass neither of them.
    """

    def test_the_run_is_parametrised_and_tagged_with_the_size(self, tmp_path):
        logger, client, run_id = _start_logger(tmp_path)

        data = client.get_run(run_id).data
        assert data.params["input_size"] == "352"
        assert data.tags["input_size"] == "352"
        # The dict it was added to is still written whole: a param block that
        # broke would otherwise only show up as a missing sibling.
        assert data.params["batch_size"] == "2"
        assert data.tags["batch_size"] == "2"

    def test_the_entry_point_records_the_size_it_was_given(self, tmp_path, monkeypatch):
        """``--input-size`` reaches the run.

        The trainer must hand the size it trains at to the config; a call that
        left the field to a default would record every run as a 192 run.
        """
        import ecs_hcm_model.core.trainer as trainer

        logger = _real_logger(tmp_path)
        _patch_trainer(monkeypatch, logger)
        seen: dict = {}

        def _capture(**kw):
            seen.update(kw)
            # The real class: a field the trainer failed to fill raises here
            # rather than silently going unrecorded.
            return logger_config(**kw)

        monkeypatch.setattr(trainer, "logger_config", _capture)

        def _die(**kw):
            raise SignalKilled(signal.SIGTERM)

        # Aborted at the first training call -- set_logger has already run, and
        # the kill path is the one TestKillReasonSurvivesTheKill pins down, so
        # nothing here depends on the abort being graceful.
        monkeypatch.setattr(trainer, "train_model_for_epochs", _die)

        with pytest.raises(SignalKilled):
            trainer.train_hcm_imagenet1k(
                environment="development", max_epochs=1, input_size=_INPUT_SIZE
            )

        assert seen["input_size"] == _INPUT_SIZE
        recorded = mlflow.MlflowClient().get_run(logger.run_id).data
        assert recorded.params["input_size"] == "352"
        assert recorded.tags["input_size"] == "352"
