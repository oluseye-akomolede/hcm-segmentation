"""Per-run tags must land on the run that produced them, never on a stray run.

The defect these tests exist for is invisible to a recording fake. The trainer's
exit path tags the run *after* the run has been ended; the old implementation
wrote through the fluent ``mlflow.set_tag``, which targets whatever run is
active. With no active run fluent ``set_tag`` does not raise -- it silently
auto-starts a new run to hold the tag. Run ``real-a68e1a2`` therefore left its
``oom_skipped_total`` on a stray empty run (``dashing-worm-142``) while the run
that actually trained kept no tag at all.

A ``RecordingLogger`` records the call and can see none of that, so every
assertion below is read back with ``MlflowClient`` from a real local store, and
counts the runs in the experiment to prove no second one appeared.
"""

import os

import mlflow
import pytest

# MLflow 3.x refuses the filesystem backend unless this is set; these tests are
# precisely about persisting to a local store, so opt in.
os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")

from ecs_hcm_model.core.model_logger import (  # noqa: E402
    MLFlowHCMLogger,
    logger_config,
)
from ecs_hcm_model.core.trainer import restore_kill_handlers  # noqa: E402

from test_mlflow_lifecycle import (  # noqa: E402
    _NeverCalled,
    _patch_trainer,
    _real_logger,
    _start_logger,
    _status,
    _tags,
)


def _runs(client, experiment_name):
    """Every run in the named experiment, or [] if it was never created."""
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        return []
    return client.search_runs([experiment.experiment_id])


@pytest.fixture(autouse=True)
def _clean_tracking_and_signals():
    """Leave no run open and no handler installed, whatever a test did."""
    yield
    while mlflow.active_run() is not None:
        mlflow.end_run()
    restore_kill_handlers()


class TestTagAfterTheRunEnded:
    """The regression: a post-run tag must not create a run."""

    def test_oom_tag_lands_on_the_ended_run_and_creates_no_second_run(
        self, tmp_path
    ):
        logger, client, run_id = _start_logger(tmp_path)
        logger.register(model_uri=None, finish_status="FINISHED")

        assert _status(client, run_id) == "FINISHED"
        # What the trainer's ``finally`` does, after register() ended the run.
        logger.tag_oom_skipped_total(7)
        experiment_name = f"hcm-lifecycle-{tmp_path.name}"

        assert _tags(client, run_id).get("oom_skipped_total") == "7", (
            "the ended run lost its OOM tag: it was written through an API that "
            "picks the active run, and there was none"
        )
        assert len(_runs(client, experiment_name)) == 1, (
            "tagging an ended run created a second run -- exactly the stray "
            "dashing-worm-142 a finished run left behind"
        )

    def test_checkpoint_upload_failures_after_the_end_also_stays_put(self, tmp_path):
        logger, client, run_id = _start_logger(tmp_path)

        logger.tag_checkpoint_upload_failures(2)
        logger.register(model_uri=None, finish_status="FINISHED")
        logger.tag_checkpoint_upload_failures(3)

        assert _tags(client, run_id).get("checkpoint_upload_failures") == "3"
        assert len(_runs(client, f"hcm-lifecycle-{tmp_path.name}")) == 1

    def test_kill_reason_after_the_end_still_reaches_the_run(self, tmp_path):
        """The kill path tags before kill(); the new write works either way."""
        logger, client, run_id = _start_logger(tmp_path)

        logger.tag_kill_reason("SIGTERM")
        logger.kill()
        # A second tag after the run closed must still land.
        logger.tag_kill_reason("SIGINT")

        assert _tags(client, run_id).get("kill_reason") == "SIGINT"
        assert _status(client, run_id) == "KILLED"
        assert len(_runs(client, f"hcm-lifecycle-{tmp_path.name}")) == 1

    def test_the_tag_is_placed_on_the_run_that_produced_it(self, tmp_path):
        """Two logger objects, one run each: the tag must not cross over.

        The exact-run-id addressing is what makes this hold even when the other
        logger's run is the active one.
        """
        experiment_name = f"hcm-lifecycle-{tmp_path.name}"
        logger_a, client, run_id_a = _start_logger(tmp_path)
        mlflow.end_run()  # no active run, as after register()

        logger_b = _real_logger(tmp_path)
        experiment_name_b = f"hcm-lifecycle-b-{tmp_path.name}"
        client_b = mlflow.MlflowClient()
        mlflow.set_experiment(experiment_name_b)
        experiment_b = client_b.get_experiment_by_name(experiment_name_b)
        run_b = client_b.create_run(experiment_b.experiment_id)
        mlflow.start_run(run_id=run_b.info.run_id)
        logger_b.run = client_b.get_run(run_b.info.run_id)
        run_id_b = logger_b.run_id
        mlflow.end_run()
        assert run_id_a != run_id_b

        logger_a.register(model_uri=None, finish_status="FINISHED")
        logger_a.tag_oom_skipped_total(11)
        logger_b.tag_oom_skipped_total(22)

        assert _tags(client, run_id_a).get("oom_skipped_total") == "11"
        assert _tags(client, run_id_b).get("oom_skipped_total") == "22"
        assert len(_runs(client, experiment_name)) == 1
        assert len(_runs(client, experiment_name_b)) == 1


class TestNoWriteAfterTheEndCreatesARun:
    """R1: every logger write after the end stays on the run or is dropped.

    The tag path was fixed first; metrics, params, a kill and a second
    register() still went through fluent calls that start a new run once the
    run has ended.
    """

    def _ended(self, tmp_path, status="FINISHED"):
        logger, client, run_id = _start_logger(tmp_path)
        logger.register(model_uri=None, finish_status=status)
        assert mlflow.active_run() is None
        return logger, client, run_id

    def test_a_late_scalar_lands_on_the_ended_run(self, tmp_path):
        logger, client, run_id = self._ended(tmp_path)
        logger.log_scalar("train_isc_absmax", 6.0e4, step=3)
        assert client.get_run(run_id).data.metrics["train_isc_absmax"] == 6.0e4
        assert len(_runs(client, f"hcm-lifecycle-{tmp_path.name}")) == 1

    def test_a_late_epoch_metric_set_lands_on_the_ended_run(self, tmp_path):
        from ecs_hcm_model.core.model_logger import data_obj
        logger, client, run_id = self._ended(tmp_path)
        logger.track(data_obj(0.5, 61.0, 70.0, prefix="val", task="segmentation"), epoch=2)
        metrics = client.get_run(run_id).data.metrics
        assert metrics["val_iou"] == 61.0 and metrics["val_loss"] == 0.5
        assert len(_runs(client, f"hcm-lifecycle-{tmp_path.name}")) == 1

    def test_a_kill_after_register_changes_nothing(self, tmp_path):
        """SIGTERM during teardown: the run keeps FINISHED and gains no error."""
        logger, client, run_id = self._ended(tmp_path)
        logger.kill()
        assert _status(client, run_id) == "FINISHED"
        assert "error" not in client.get_run(run_id).data.params
        assert len(_runs(client, f"hcm-lifecycle-{tmp_path.name}")) == 1

    def test_a_second_register_changes_nothing(self, tmp_path):
        logger, client, run_id = self._ended(tmp_path)
        logger.register(model_uri="s3://x/y", finish_status="FAILED", exc_msg="late")
        run = client.get_run(run_id)
        assert run.info.status == "FINISHED"
        assert "error" not in run.data.params and "model_uri" not in run.data.tags
        assert len(_runs(client, f"hcm-lifecycle-{tmp_path.name}")) == 1

    def test_a_late_dataset_log_is_dropped(self, tmp_path):
        logger, client, _run_id = self._ended(tmp_path)
        logger.log_dataset(object())  # never reaches mlflow, so any object will do
        assert mlflow.active_run() is None
        assert len(_runs(client, f"hcm-lifecycle-{tmp_path.name}")) == 1

    def test_register_writes_its_uri_and_error_to_its_own_run(self, tmp_path):
        """The live path, now run-addressed, still records what it did."""
        logger, client, run_id = _start_logger(tmp_path)
        logger.register(model_uri="s3://b/k", finish_status="FAILED", exc_msg="boom")
        run = client.get_run(run_id)
        assert run.info.status == "FAILED"
        assert run.data.tags["model_uri"] == "s3://b/k"
        assert run.data.params["model_location"] == "s3://b/k"
        assert run.data.params["error"] == "boom" and run.data.tags["error"] == "boom"


class TestTagWithoutARun:
    """A logger that never had a run must write nothing and create nothing."""

    def test_no_run_started_means_no_tag_and_no_new_run(self, tmp_path):
        logger = _real_logger(tmp_path)  # constructed, set_logger() never called
        assert logger.run_id is None

        # None of these may raise, and none may start a run.
        logger.tag_kill_reason("SIGTERM")
        logger.tag_checkpoint_upload_failures(4)
        logger.tag_oom_skipped_total(5)
        logger.log_scalar("x", 1.0)
        logger.kill()

        assert mlflow.active_run() is None
        assert _runs(mlflow.MlflowClient(), f"hcm-lifecycle-{tmp_path.name}") == []


class TestTrainerFinishPath:
    """The whole path, not just the logger call: a finished run keeps its tag."""

    def test_a_finished_run_carries_oom_skipped_total_and_no_scrap_run(
        self, tmp_path, monkeypatch
    ):
        import ecs_hcm_model.core.trainer as trainer

        logger = _real_logger(tmp_path)
        _patch_trainer(monkeypatch, logger)

        # A training run that skipped some batches but completed normally.
        monkeypatch.setattr(
            trainer,
            "train_model_for_epochs",
            lambda **kw: {
                "best_epoch": 0,
                "best_score": 1.0,
                "best_state": None,
                "best_metrics": {},
                "best_train_loss": 0.5,
                "last_epoch": 0,
                "oom_skipped_total": 3,
            },
        )

        trainer.train_hcm_imagenet1k(environment="development", max_epochs=1)

        run_id = logger.run_id
        client = mlflow.MlflowClient()
        assert _status(client, run_id) == "FINISHED"
        assert _tags(client, run_id).get("oom_skipped_total") == "3", (
            "the finished run lost the OOM tag the exit path wrote"
        )
        assert len(_runs(client, "hcm_imagenet1k_experiment")) == 1, (
            "the exit path left a stray run holding only the tag"
        )
