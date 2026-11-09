import abc
import math
from typing import Literal
from dataclasses import dataclass
from torch.utils.data import (
    DataLoader,

)

import mlflow
from pytorch_lightning.loggers import MLFlowLogger


MLFLOW_DEV = "http://10.0.1.102:9090"
MLFLOW_PROD = "http://10.0.1.86:9091"

@dataclass
class logger_config:
    experiment_name : str
    environment : Literal["development", "production"]
    dataset : str
    max_epochs : int
    batch_size : int
    # The spatial size the run trains at. It is part of the run's identity, not a
    # detail of it: two runs of the same experiment at 192 and at 352 are the same
    # flags and the same data, and only this tells them apart from the MLflow side
    # (the checkpoint metadata has carried it all along).
    input_size : int

# Metric keys are namespaced by phase (train/val) and by task, so a train and a
# val value never land on the same key at the same step, and a classification run
# never reports IoU/Dice (nor a segmentation run acc1/acc5).
TASK_CLASSIFICATION = "classification"
TASK_SEGMENTATION = "segmentation"

# The task registry (model_upload_download.TASK_CHECKPOINTS) names the concrete
# segmentation tasks, while TASK_SEGMENTATION above is the generic task kind.
# Accept either spelling as "this is a segmentation run" -- comparing a concrete
# name against the generic one is always False, which silently logged
# segmentation epochs under the accuracy keys.
SEGMENTATION_TASK_NAMES = {"coco_seg", "cocoa_seg", TASK_SEGMENTATION}


def normalise_task(task: str) -> str:
    """Collapse a concrete task name ("coco_seg") onto its task kind."""
    return TASK_SEGMENTATION if task in SEGMENTATION_TASK_NAMES else TASK_CLASSIFICATION

# Percentage metrics must live in [0, 100]. Guarding here is what turns a
# double-scaled value (e.g. acc1=6006) into a loud failure instead of garbage
# that silently pollutes MLflow.
PERCENT_METRICS = {"acc1", "acc5", "iou", "dice"}


class PercentageOutOfRangeError(ValueError):
    """Raised when a metric that should be a 0-100 percentage is not."""


def _check_percentage(name: str, value: float) -> None:
    if value is None:
        return
    try:
        v = float(value)
    except (TypeError, ValueError):
        raise PercentageOutOfRangeError(f"{name} is not numeric: {value!r}")
    if not math.isfinite(v) or v < 0.0 or v > 100.0:
        raise PercentageOutOfRangeError(
            f"{name}={v} is outside the expected 0-100 percentage range. "
            f"This usually means a value was scaled by 100 twice (or not at all)."
        )


class data_obj():
    """Epoch- or batch-level metrics, tagged with the phase and task they belong to.

    ``prefix`` is "train" or "val"; ``task`` is TASK_CLASSIFICATION or
    TASK_SEGMENTATION. The positional signature is unchanged so existing
    construction sites keep working.
    """

    def __init__(self, lv, ac1, ac5, IoU=None, Dice=None,
                 prefix="train", task=TASK_CLASSIFICATION):
        self.loss_value = lv
        self.prefix = prefix
        self.task = task
        self.is_segmentation = normalise_task(task) == TASK_SEGMENTATION

        # acc1/acc5 carry accuracy for classification and IoU/Dice for
        # segmentation; the task decides which names they are logged under.
        # They are kept separate (no aliasing) so a reader can never mistake an
        # accuracy for an IoU.
        self.acc1 = ac1
        self.acc5 = ac5
        self.IoU = IoU
        self.Dice = Dice

    def as_metrics(self) -> dict:
        """Map to {metric_name: value}; only the names valid for this task."""
        if self.is_segmentation:
            # Callers differ in where they put the segmentation numbers: some
            # pass IoU/Dice by keyword, but the trainer's call sites pass them
            # positionally (ac1/ac5). Read whichever the caller actually used,
            # preferring the explicit keyword; a value of None means "not
            # measured", which the logger then omits rather than fabricating.
            iou = self.IoU if self.IoU is not None else self.acc1
            dice = self.Dice if self.Dice is not None else self.acc5
            metrics = {"iou": iou, "dice": dice}
        else:
            metrics = {"acc1": self.acc1, "acc5": self.acc5}
        metrics["loss"] = self.loss_value
        return metrics

class AbstractLogger(abc.ABC):
    @abc.abstractmethod
    def _setup_environment(
        self
        ):
        raise NotImplementedError

    @abc.abstractmethod
    def set_logger(
        self,
        config : logger_config,
        ) -> MLFlowLogger:
        raise NotImplementedError
    
    @abc.abstractmethod
    def register(
        self,
        model_uri: str | None,
        finish_status: Literal["FAILED","FINISHED","KILLED"],
        exc_msg: str | None = None,
        ):
        raise NotImplementedError

    @abc.abstractmethod
    def run_id(self) -> str | None:
        """The active MLflow run id, for stamping into checkpoint metadata."""
        raise NotImplementedError

    @abc.abstractmethod
    def tag_kill_reason(self, reason: str) -> None:
        """Record why a run was killed (e.g. the signal that arrived).

        Addressed to this logger's own run, so it lands whether the run is
        still open or has already ended, and never creates a run.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def tag_checkpoint_upload_failures(self, count: int) -> None:
        """Record how many per-epoch checkpoints failed to upload.

        Addressed to this logger's own run; never creates a run.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def tag_oom_skipped_total(self, count: int) -> None:
        """Record how many training batches were skipped for CUDA OOM.

        Addressed to this logger's own run; never creates a run.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def tag_failed_rank(self, rank: int) -> None:
        """Record which rank of a distributed run lost the race to failure.

        A distributed run fails as a whole, but only one rank usually finds the
        underlying error; the others trip over a broken collective. Naming the
        rank that failed first is what makes the run's FAILED state diagnosable.

        Addressed to this logger's own run; never creates a run.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def log_scalar(self, key: str, value: float, step: int = None) -> None:
        """Log a single scalar under its exact key (no percentage range guard)."""
        raise NotImplementedError


def set_tracking_environment(environment: Literal["development", "production"]) -> None:
    mlflow.set_tracking_uri(MLFLOW_DEV if environment == "development" else MLFLOW_PROD)

def _setup_environment(environment: Literal["development", "production"]) -> None:
    set_tracking_environment(environment)
    mlflow.enable_system_metrics_logging()

def _validate_environment(environment: str) -> None:
    valid_environments = {"development", "production"}
    if environment not in valid_environments:
        raise ValueError("Environment should be development or production.")


def _set_common_tags(config : logger_config) -> dict[str,str]:
    # The run's configuration, as the run list and the registered model see it.
    # Kept in step with the params ``set_logger`` writes: a field the run is
    # tagged with is one the run is parametrised with, so a reader of either
    # surface gets the same configuration.
    return{
        "batch_size": str(config.batch_size),
        "input_size": str(config.input_size),
        "max_epochs": str(config.max_epochs),
        "environment": config.environment,
        "dataset": config.dataset
    }

def get_registration_name(
    model_type: str, run_name: str
) -> str:
    return f"ml_{model_type}_{run_name}"

def _set_registered_model_tags(tags: dict[str, str], registration_name: str, run_id: str) -> None:
    client = mlflow.MlflowClient()

    # Set tags on the registered model (model-level)
    for key, value in tags.items():
        client.set_registered_model_tag(
            name=registration_name,
            key=key,
            value=value
        )

    # Get the latest model version from this run
    versions = client.search_model_versions(f"name='{registration_name}' and run_id='{run_id}'")
    for version in versions:
        for key, value in tags.items():
            client.set_model_version_tag(
                name=registration_name,
                version=version.version,
                key=key,
                value=value
            )


class MLFlowHCMLogger(AbstractLogger):
    VALID_FINISH_STATUS = {"FAILED", "FINISHED", "KILLED"}
    def __init__(
        self,
        config: logger_config

    ):
        _validate_environment(config.environment)
        self.environment = config.environment
        self.experiment_name = config.experiment_name

        self.experiment: mlflow.entities.Experiment | None = None
        self.run: mlflow.entities.Run | None = None
        self.logger: MLFlowLogger | None = None
        self.tags = dict[str]
        self.ckpt_name = None
        self._ended = False
        self._mlflow_client: mlflow.MlflowClient | None = None

        self._setup_environment()

    def _setup_environment(self) -> None:
        _setup_environment(self.environment)

    def track(
              self,
              data_inst: data_obj = None,
              epoch: int = None
            ):
        self._log_metrics(data_inst, data_inst.prefix, step=epoch)

    def track_batch(
            self,
            data_inst: data_obj = None,
            batch_idx: int = None
        ):
        self._log_metrics(data_inst, "batch", step=batch_idx)

    def _log_metrics(self, data_inst: data_obj, prefix: str, step: int) -> None:
        """Log one metric set under ``{prefix}_{name}`` keys."""
        if data_inst is None:
            return
        for name, value in data_inst.as_metrics().items():
            if value is None:
                continue
            if name in PERCENT_METRICS:
                _check_percentage(f"{prefix}_{name}", value)
            self._log_metric(f"{prefix}_{name}", value, step)

    def log_scalar(self, key: str, value: float, step: int = None) -> None:
        """Log a single non-percentage scalar under its exact key.

        Activation magnitudes (``train_isc_absmax`` and friends) are raw
        numbers, not 0-100 percentages, so they must NOT pass through the
        percentage guard in ``_log_metrics`` -- a value of 60000 would be
        rejected as an out-of-range percentage and the measurement, which is
        the whole point of tracking it, would be lost. The key is used verbatim.
        """
        if value is None:
            return
        try:
            value = float(value)
        except (TypeError, ValueError):
            return
        self._log_metric(key, value, step)

    def _client(self) -> mlflow.MlflowClient:
        if self._mlflow_client is None:
            self._mlflow_client = mlflow.MlflowClient()
        return self._mlflow_client

    def _log_metric(self, key: str, value: float, step: int | None) -> None:
        """Write one metric to THIS logger's run, by id.

        Same reason as ``_set_run_tag``: a fluent ``mlflow.log_metric`` after
        ``register()`` ended the run silently starts a new run to hold it. A
        late metric lands on the run it belongs to; without a run it is dropped.
        """
        if self.run is None:
            return
        self._client().log_metric(self.run.info.run_id, key, value, step=step)

    def set_logger(
        self,
        config : logger_config,
    ) -> None:
        self.experiment = mlflow.set_experiment(config.experiment_name)
        self.tags = _set_common_tags(
            config=config
        )



        client = mlflow.MlflowClient()
        run = client.create_run(self.experiment.experiment_id)
        run = mlflow.start_run(run_id = run.info.run_id, tags = self.tags)
        mlflow.log_params(
            {
                "dataset_name": config.dataset,
                "max epochs" : config.max_epochs,
                "batch_size": config.batch_size,
                # Written here rather than with the trainer's other settings
                # (log_run_param): the size is known before the run starts, so it
                # belongs to the run's own configuration record and is written
                # once, by the code that owns that record.
                "input_size": config.input_size,
                "environment": config.environment,

            }
        )

        self.run = run
        self.ckpt_name = f"ml_hcm_{run.info.run_name}"

    @property
    def run_id(self) -> str | None:
        return self.run.info.run_id if self.run is not None else None

    def register(
        self,
        model_uri: str | None,
        finish_status: Literal["FAILED","FINISHED","KILLED"],
        exc_msg: str | None = None,
    ) -> None:
        if finish_status not in self.VALID_FINISH_STATUS:
            raise ValueError("finish_status must be one of 'FAILED', 'FINISHED', 'KILLED'")
        if self._ended:
            # The run's outcome is already recorded; a second register() must
            # neither rewrite it nor, through a fluent call, start a new run.
            print(f"register({finish_status}) after the run ended: ignored")
            return

        # Log the model URI as both a tag and parameter
        if model_uri is not None:
            self._set_run_tag("model_uri", model_uri)
            self._log_run_param("model_location", model_uri)
            print(f"Logged model URI: {model_uri}")

        if finish_status == "FAILED":
            self._log_run_param("error", exc_msg)
            if exc_msg:
                self._set_run_tag("error", str(exc_msg)[:500])

        self._end_run(finish_status)

    def _end_run(self, status: str) -> None:
        """End the active run exactly once.

        Every exit path funnels through here so a run can never be left
        RUNNING, and so a ``finally`` after a successful ``register`` is a
        no-op rather than a double end (which MLflow warns on).
        """
        if self._ended:
            return
        self._ended = True
        mlflow.end_run(status=status)

    def log_dataset(
            self,
            data_loader: DataLoader,

    )->None:
        # log_input has no run-addressed form here, so it is only written while
        # this logger's run is the live one; afterwards it would start a new run.
        if self.run is None or self._ended:
            return
        mlflow.log_input(dataset=data_loader.dataset,context="training")

    def kill(self):
        # After register() the run already has its final status: a kill then
        # (SIGTERM during teardown) must not relabel it or create a run.
        if self._ended:
            return
        self._log_run_param("error", "Process was killed")
        self._end_run("KILLED")

    def log_run_param(self, key: str, value) -> None:
        """Write one param to THIS logger's run, by id.

        The same call ``_log_run_param`` makes, for a caller that wants to
        record a run setting after the run exists: params are what MLflow shows
        as the run's inputs, and a setting can only be added this way, not
        through the constructor's ``log_params``.
        """
        self._log_run_param(key, value)

    def _log_run_param(self, key: str, value) -> None:
        """Write one param to THIS logger's run, by id (see ``_set_run_tag``)."""
        if self.run is None:
            return
        self._client().log_param(self.run.info.run_id, key, value)

    def _set_run_tag(self, key: str, value) -> bool:
        """Write one tag to THIS logger's run. Returns whether it was written.

        Addressed to the run id rather than written through the fluent
        ``mlflow.set_tag``. The fluent call targets whatever run happens to be
        active: the trainer's exit path tags the run *after* ``register()`` has
        ended it, and with no active run the fluent call does not fail -- it
        silently auto-starts a NEW run to hold the tag. That is how run
        ``real-a68e1a2`` left its ``oom_skipped_total`` on a stray empty run
        (``dashing-worm-142``) instead of on itself.

        A logger that never had a run started has no id to address, so it
        writes nothing rather than letting the call attach itself to a new run.
        """
        if self.run is None:
            return False
        self._client().set_tag(self.run.info.run_id, key, str(value))
        return True

    def tag_kill_reason(self, reason: str) -> None:
        """Tag the run with the signal/condition that killed it.

        Addressed to this logger's own run, so it lands whether the run is
        still live or has already been ended, and never creates a run.
        """
        self._set_run_tag("kill_reason", str(reason)[:500])

    def tag_checkpoint_upload_failures(self, count: int) -> None:
        """Tag the run with the number of per-epoch checkpoints that failed.

        A printed warning is invisible in MLflow, so a run whose checkpoints
        all failed to persist looked healthy from the tracking side.
        """
        self._set_run_tag("checkpoint_upload_failures", int(count))

    def tag_oom_skipped_total(self, count: int) -> None:
        """Tag the run with the number of training batches skipped for OOM.

        Same reasoning as checkpoint_upload_failures: the per-batch OOM prints
        are invisible from MLflow, and a run that skipped every batch -- the
        real V100 double-booking -- otherwise looked like a completed epoch.
        """
        self._set_run_tag("oom_skipped_total", int(count))

    def tag_failed_rank(self, rank: int) -> None:
        """Tag the run with the rank that failed first in a distributed run.

        Written by the rank that owns the run, about a peer that has already
        died, so it goes through the same tag path as the others: addressed to
        this logger's own run, and dropped with a warning rather than raising if
        the run has already been closed.
        """
        self._set_run_tag("failed_rank", int(rank))