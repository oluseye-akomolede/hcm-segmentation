"""Shared fakes for the HCM trainer tests.

These tests run on CPU with no network, no GPU, no MinIO and no MLflow. The
logger, the uploader and the dataloaders are all faked, and a tiny linear model
stands in for ``hcm_nn`` (which is far too heavy to instantiate per test).
"""

import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn

# Make the repo importable when pytest is run from the repo root.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ecs_hcm_model.core.model_logger import (  # noqa: E402
    data_obj,
    TASK_CLASSIFICATION,
    TASK_SEGMENTATION,
    PERCENT_METRICS,
    _check_percentage,
)


class RecordingLogger:
    """Stands in for MLFlowHCMLogger, recording every metric it is asked to log."""

    def __init__(self, run_id="run-fake-123"):
        self.metrics = []          # list of (key, value, step)
        self.batches = []          # list of (key, value, step) for track_batch
        self.scalars = []          # list of (key, value, step) for log_scalar
        self.params = {}           # key -> value for log_run_param
        self.registered = []       # list of (model_uri, finish_status, exc_msg)
        self.killed = False
        self.kill_reason = None
        self.checkpoint_upload_failures = None
        self.oom_skipped_total = None
        self.failed_rank = None
        self._ended = False
        self._run_id = run_id

    @property
    def run_id(self):
        return self._run_id

    def set_logger(self, config=None):
        pass

    def track(self, data_inst, epoch):
        self._record(data_inst, self.metrics, epoch)

    def track_batch(self, data_inst, batch_idx):
        self._record(data_inst, self.batches, batch_idx)

    def _record(self, data_inst, sink, step):
        if data_inst is None:
            return
        for name, value in data_inst.as_metrics().items():
            if value is None:
                continue
            if name in PERCENT_METRICS:
                _check_percentage(f"{data_inst.prefix}_{name}", value)
            sink.append((f"{data_inst.prefix}_{name}", value, step))

    def register(self, model_uri, finish_status, exc_msg=None):
        self.registered.append((model_uri, finish_status, exc_msg))
        self._ended = True

    def kill(self):
        self.killed = True
        self.registered.append((None, "KILLED", None))
        self._ended = True

    def tag_kill_reason(self, reason):
        self.kill_reason = str(reason)

    def tag_checkpoint_upload_failures(self, count):
        self.checkpoint_upload_failures = int(count)

    def tag_oom_skipped_total(self, count):
        self.oom_skipped_total = int(count)

    def tag_failed_rank(self, rank):
        self.failed_rank = int(rank)

    def log_run_param(self, key, value):
        """Record a run param (the run's inputs, unlike a logged scalar)."""
        self.params[key] = value

    def run_params(self):
        return dict(self.params)

    def log_scalar(self, key, value, step=None):
        # Deliberately NOT percentage-checked: activation magnitudes are raw
        # numbers and must survive a value far above 100.
        self.scalars.append((key, float(value), step))

    def scalar_keys(self):
        return [k for k, _, _ in self.scalars]

    def scalar_values(self):
        return {k: v for k, v, _ in self.scalars}

    def _end_run(self, status):
        if self._ended:
            return
        self._ended = True
        self.registered.append((None, status, None))

    def finish_statuses(self):
        return [status for _, status, _ in self.registered]

    def metric_keys(self):
        return {k for k, _, _ in self.metrics}

    def metric_values(self):
        return {k: v for k, v, _ in self.metrics}


class RecordingUploader:
    """Stands in for ModelUploaderAndDownloader; records uploads, can be made to fail."""

    instances = []  # every uploader constructed, so tests can inspect them

    def __init__(self, bucket_name="models", fail=False):
        self.bucket_name = bucket_name
        self.fail = fail
        self.uploads = []
        RecordingUploader.instances.append(self)

    def upload_model(self, model, model_name, architecture=None, metadata=None):
        if self.fail:
            raise RuntimeError("simulated upload failure")
        self.uploads.append(
            {
                "model_name": model_name,
                "architecture": architecture,
                "metadata": metadata or {},
                "bucket": self.bucket_name,
            }
        )
        return f"{self.bucket_name}/{model_name}/model.pth"


class _ActivationMaxMixin:
    """The activation-monitoring surface the trainer reads from a real hcm_nn."""

    ACTIVATION_STAGES = ("isc", "csc", "isc3")

    def _init_activation_absmax(self):
        self.activation_absmax = {s: 0.0 for s in self.ACTIVATION_STAGES}

    def reset_activation_absmax(self):
        for stage in self.activation_absmax:
            self.activation_absmax[stage] = 0.0


class TinyClassifier(_ActivationMaxMixin, nn.Module):
    """Minimal stand-in for hcm_nn on the classification path.

    ``hcm_nn`` always exposes a segmentation head (the trainer's smoke-mode
    diagnostics read ``model.segmentation_head`` on both task paths), so the
    stand-in does too -- it is simply not used for classification.
    """

    def __init__(self, num_classes=10, in_features=8):
        super().__init__()
        self.backbone = nn.Linear(in_features, 16)
        self.head = nn.Linear(16, num_classes)
        # Unused on this path, but hcm_nn always has it and the trainer reads it.
        self.segmentation_head = nn.Linear(16, 1)
        self._init_activation_absmax()

    def forward(self, x):
        return self.head(self.backbone(x.view(x.size(0), -1)))


class TinySegmenter(_ActivationMaxMixin, nn.Module):
    """Minimal stand-in for hcm_nn on the segmentation path (binary masks).

    The real model emits two channels (background + foreground); the trainer's
    loss, its smoke-mode diagnostics and its dice metric all assume that.
    """

    def __init__(self, in_features=8, h=4, w=4, num_classes=2):
        super().__init__()
        self.backbone = nn.Linear(in_features, 32)
        self.segmentation_head = nn.Linear(32, num_classes * h * w)
        self.num_classes, self.h, self.w = num_classes, h, w
        # hcm_nn exposes this flag and the trainer reads it in smoke mode.
        self.enable_finetuning = True
        self._init_activation_absmax()

    def forward(self, x):
        out = self.segmentation_head(self.backbone(x.view(x.size(0), -1)))
        return out.view(x.size(0), self.num_classes, self.h, self.w)


def make_classification_batches(n_batches=2, batch_size=4, in_features=8, num_classes=10):
    """Yield (images, labels) tuples for the classification path."""
    return [
        (
            torch.randn(batch_size, in_features),
            torch.randint(0, num_classes, (batch_size,)),
        )
        for _ in range(n_batches)
    ]


def make_segmentation_batches(n_batches=2, batch_size=4, in_features=8, h=4, w=4):
    """Yield (images, masks) dicts for the segmentation path."""
    return [
        {
            "pixel_values": torch.randn(batch_size, in_features),
            "segmentation_mask": (torch.rand(batch_size, 1, h, w) > 0.5).float(),
        }
        for _ in range(n_batches)
    ]


def _selftest_repo_path_is_cwd_independent():
    """Guarded self-test, run from ``tests/test_source_paths.py``.

    Kept here so the helper and the thing it guarantees live together.
    """
    import os

    here = os.getcwd()
    try:
        os.chdir("/")
        assert repo_path("ecs_hcm_model", "core", "trainer.py").is_file()
    finally:
        os.chdir(here)


# --- Keeping the suite off the real MLflow server ---------------------------
# The dev MLflow server (MLFLOW_DEV, http://10.0.1.102:9090) is reachable
# from this VM, and MLFlowHCMLogger's constructor calls
# set_tracking_environment(), which sets the tracking URI to it. A test that
# points mlflow at a local file store and then builds a real logger therefore
# has its store silently replaced, and every run it opens -- including the
# KILLED ones the signal tests create on purpose -- lands in the production
# experiment. About 70 such runs accumulated before this was noticed.
#
# Two things stop it, because one is not enough:
#   1. the fixture redirects the module-level setters at a tmp file store, so
#      the constructor cannot reach the network even if a test forgets;
#   2. a guard on mlflow.set_tracking_uri fails loudly on any http(s) URI, so a
#      future path that bypasses (1) is a red test rather than a stray run.

# The only tracking schemes a test may use. Everything else -- http(s) above
# all -- reaches a network service and can create runs outside this machine.
# Written as an allow-list rather than a block-list of http(s): a scheme this
# file has never heard of must be refused, not waved through.
#
# No empty string in this tuple: every string startswith "", so a single ""
# entry would silently turn the whole check into a no-op.
_ALLOWED_TRACKING_SCHEMES = ("file:", "sqlite:")


def _assert_local_tracking_uri(uri) -> None:
    """Fail if anything tries to point mlflow at a non-local store."""
    if uri is None:
        return
    if not isinstance(uri, str):
        raise AssertionError(
            f"refusing a non-string mlflow tracking URI {uri!r}"
        )
    scheme = uri.strip().lower()
    if not any(scheme.startswith(s) for s in _ALLOWED_TRACKING_SCHEMES):
        raise AssertionError(
            f"refusing to point mlflow at {uri!r}: the test suite must not "
            f"talk to a real MLflow server (it would create runs in the "
            f"production experiment). Use a local store instead -- one of "
            f"{_ALLOWED_TRACKING_SCHEMES}."
        )


@pytest.fixture(autouse=True)
def _keep_mlflow_local(monkeypatch, tmp_path_factory):
    """Neutralise the real tracking URI and guard against setting one.

    Autouse and repo-wide on purpose: the leak came from a logger constructed
    deep inside a test that did not intend to touch the network, so opting in
    per-test would not have prevented it.
    """
    import mlflow

    import ecs_hcm_model.core.model_logger as model_logger

    store = tmp_path_factory.mktemp("mlflow-store")
    local_uri = "file:" + str(store / "mlruns")

    # (1) Redirect the module's own setters, so MLFlowHCMLogger.__init__ lands
    # on the tmp store however the test set things up.
    monkeypatch.setattr(
        model_logger, "set_tracking_environment", lambda environment: mlflow.set_tracking_uri(local_uri)
    )
    monkeypatch.setattr(
        model_logger, "_setup_environment", lambda environment: mlflow.set_tracking_uri(local_uri)
    )

    # (2) Guard every set_tracking_uri call, whoever makes it. `mlflow` is a
    # single module object, so patching it here is visible to the code under
    # test too.
    real_set = mlflow.set_tracking_uri

    def guarded_set(uri=None, *a, **kw):
        _assert_local_tracking_uri(uri)
        return real_set(uri, *a, **kw)

    monkeypatch.setattr(mlflow, "set_tracking_uri", guarded_set)

    yield


def repo_path(*parts):
    """Absolute path to something in the repo, independent of the CWD.

    Tests that read source files must not depend on where pytest was launched:
    a relative path works from the repo root and fails with FileNotFoundError
    from anywhere else, which is exactly what a CI with a different rootdir
    does. Anchored to this file's location (``tests/conftest.py``).
    """
    return REPO_ROOT.joinpath(*parts)


def trainer_source():
    """The trainer's source text, for the source-inspection tests."""
    return repo_path("ecs_hcm_model", "core", "trainer.py").read_text(
        encoding="utf-8"
    )


@pytest.fixture(autouse=True)
def _reset_uploader_registry():
    RecordingUploader.instances = []
    yield
    RecordingUploader.instances = []


@pytest.fixture
def logger():
    return RecordingLogger()


@pytest.fixture
def uploader():
    return RecordingUploader()
