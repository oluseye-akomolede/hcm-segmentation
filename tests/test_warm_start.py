"""Warm start: a fresh model initialised from a checkpoint of another version.

--init-from builds a fresh model (v2 by default) and copies in every tensor of
the named checkpoint whose name and shape still fit; the rest keep their init,
except that a v1 source initialises v2's rrd fuse convs to [I | 0].
It starts a new lineage: init_from names the source, resumed_from stays null,
and it can never be combined with a resume.

CPU-only and offline: the ResNet-50 fetch is stubbed as in test_decoder_arch.
"""

import inspect
import io
import json
import pickle

import pytest
import torch

import ecs_hcm_model.core.hcm_nn as hcm_mod
from ecs_hcm_model.core import model_utils
from ecs_hcm_model.core import trainer as trainer_module
from ecs_hcm_model.core.hcm_nn import DECODER_V1, DECODER_V2, DecoderArchMismatchError
from ecs_hcm_model.core.model_upload_download import ModelUploaderAndDownloader
from ecs_hcm_model.core.model_utils import download_checkpoint_state, warm_start_state
from ecs_hcm_model.core.trainer import InitFromError, train_hcm_imagenet1k, warm_start_from
from tests.conftest import RecordingLogger, RecordingUploader, TinyClassifier
from tests.test_decoder_arch import _Bucket

SOURCE = "coco-models/hcm_coco_seg_best/20261001_035743"


def _offline(build):
    from torchvision import models as tv_models

    real = tv_models.resnet50

    def offline_resnet50(weights=None, replace_stride_with_dilation=None):
        return real(weights=None, replace_stride_with_dilation=replace_stride_with_dilation)

    tv_models.resnet50 = offline_resnet50
    try:
        return build()
    finally:
        tv_models.resnet50 = real


def _model(arch, seed=0, classes=2):
    torch.manual_seed(seed)
    return _offline(lambda: hcm_mod.hcm_nn(enable_finetuning=True, num_segmentation_classes=classes,
                                           out_stride=16, decoder_arch=arch))


@pytest.fixture(scope="module")
def v1_state():
    return {k: v.clone() for k, v in _model(DECODER_V1, seed=1).state_dict().items()}


# --- copying by name and shape ------------------------------------------------


def test_v1_into_v2_copies_everything_and_initialises_the_fuse_convs(v1_state):
    model = _model(DECODER_V2, seed=2)
    warm = warm_start_state(model, v1_state)

    assert warm["new"] == [] and warm["skipped"] == [] and warm["dropped"] == []
    assert sorted(warm["copied"]) == sorted(v1_state)
    assert len(warm["initialised"]) == 8
    assert all(k.startswith("rrd_list.") and ".fuse." in k for k in warm["initialised"])
    after = model.state_dict()
    for k in warm["copied"]:
        assert torch.equal(after[k], v1_state[k]), k
    for rung in model.rrd_list:
        w, b = rung.fuse.weight, rung.fuse.bias
        n = w.shape[0]
        assert torch.equal(w[:, :n, 0, 0], torch.eye(n))
        assert not w[:, n:].any() and not b.any()
    assert 0.0 < warm["copied_fraction"] < 1.0


def test_a_v2_rung_starts_as_the_v1_rcab_and_conv_on_the_normal_branch(v1_state):
    """With fuse = [I | 0] the rung passes only the normal (sigmoid-gated)
    branch, which is what the copied RCAB and conv2d were trained on."""
    model = _model(DECODER_V2, seed=2).eval()
    warm_start_state(model, v1_state)
    source = _model(DECODER_V1, seed=0)
    source.load_state_dict(v1_state)
    source.eval()

    torch.manual_seed(3)
    f_s, p = torch.randn(1, 32, 12, 20), torch.randn(1, 32, 6, 10)
    for warm_rung, v1_rung in zip(model.rrd_list, source.rrd_list):
        with torch.no_grad():
            gate = torch.sigmoid(torch.nn.functional.interpolate(
                p, size=f_s.shape[-2:], mode="bilinear", align_corners=False))
            normal = torch.nn.functional.group_norm(f_s * gate, num_groups=8, eps=1e-5)
            conv = v1_rung.conv2d
            expected = torch.nn.functional.conv2d(
                v1_rung.RCAB(normal), conv.weight, conv.bias, padding=1)
            got = warm_rung(f_s, p)
        assert torch.allclose(got, expected, atol=1e-6), (got - expected).abs().max()


def test_a_v2_source_into_v2_copies_the_fuse_convs():
    source = _model(DECODER_V2, seed=1).state_dict()
    model = _model(DECODER_V2, seed=2)
    warm = warm_start_state(model, source)
    assert warm["initialised"] == [] and warm["new"] == []
    for rung_key in [k for k in source if ".fuse." in k]:
        assert torch.equal(model.state_dict()[rung_key], source[rung_key])


def test_a_v2_source_into_v2_through_the_trainer_reports_nothing_initialised(monkeypatch, capsys):
    """The --init-from path for a v2 source into a v2 model (item 3 of the row).

    A v2 source has the fuse convs the model has, so every tensor is copied and
    nothing is left at its init. The startup line is what an operator reads when
    the run starts, so it has to say exactly that.
    """
    source = {k: v.clone() for k, v in _model(DECODER_V2, seed=1).state_dict().items()}
    monkeypatch.setattr(trainer_module, "download_checkpoint_state", lambda spec: source)
    model = _model(DECODER_V2, seed=2)

    warm = warm_start_from(model, SOURCE, RecordingLogger())

    assert warm["initialised"] == [] and warm["new"] == [] and warm["skipped"] == []
    assert sorted(warm["copied"]) == sorted(source), "every tensor should have been copied"
    assert warm["copied_fraction"] == 1.0
    for key, value in model.state_dict().items():
        assert torch.equal(value, source[key]), key

    line = capsys.readouterr().out
    assert f"{len(source)} tensors copied, 0 skipped (shape changed), 0 initialised" in line, line
    assert "0 new (left at init), 0 unused from the source" in line, line


def test_a_tensor_whose_shape_changed_is_skipped_and_keeps_its_init(v1_state):
    # A 3-class source head into a 2-class model: same names, other shapes.
    source = _model(DECODER_V1, seed=1, classes=3).state_dict()
    model = _model(DECODER_V2, seed=2)
    init = {k: v.clone() for k, v in model.state_dict().items()}
    warm = warm_start_state(model, source)

    assert warm["skipped"], "the 3-class head should not fit a 2-class model"
    for k in warm["skipped"]:
        assert tuple(source[k].shape) != tuple(init[k].shape)
        assert torch.equal(model.state_dict()[k], init[k]), k


def test_source_tensors_the_model_lacks_are_reported_not_loaded():
    model = torch.nn.Linear(2, 2)
    warm = warm_start_state(model, {**model.state_dict(), "extra.weight": torch.zeros(3)})
    assert warm["dropped"] == ["extra.weight"]
    assert warm["copied_fraction"] == 1.0


# --- warm_start_from: the trainer's step --------------------------------------


def test_warm_start_marks_a_new_lineage_and_logs_the_counts(monkeypatch, v1_state):
    monkeypatch.setattr(trainer_module, "download_checkpoint_state",
                        lambda spec: v1_state if spec == SOURCE else None)
    model = _model(DECODER_V2, seed=2)
    model.resumed_from = "coco-models/something/else"
    logger = RecordingLogger()
    warm = warm_start_from(model, SOURCE, logger)

    assert model.init_from == SOURCE
    assert model.resumed_from is None
    logged = {name: value for name, value, _ in _scalars(logger)}
    assert logged["init_copied_keys"] == len(warm["copied"])
    assert logged["init_skipped_keys"] == 0
    assert logged["init_initialised_keys"] == len(warm["initialised"]) == 8
    assert logged["init_new_keys"] == len(warm["new"]) == 0
    assert logged["init_copied_fraction"] == warm["copied_fraction"]


def test_a_source_that_shares_nothing_is_refused(monkeypatch):
    monkeypatch.setattr(trainer_module, "download_checkpoint_state",
                        lambda spec: {"unrelated.weight": torch.zeros(1)})
    model = torch.nn.Linear(2, 2)
    with pytest.raises(InitFromError, match="shares no tensor"):
        warm_start_from(model, SOURCE, RecordingLogger())
    assert getattr(model, "init_from", None) is None


def test_a_wrapped_model_is_marked_on_the_inner_module(monkeypatch):
    inner = torch.nn.Linear(2, 2)
    monkeypatch.setattr(trainer_module, "download_checkpoint_state",
                        lambda spec: inner.state_dict())
    wrapper = torch.nn.Module()
    wrapper.module = inner
    warm_start_from(wrapper, SOURCE, RecordingLogger())
    assert trainer_module.init_from_of(wrapper) == SOURCE


def _scalars(logger):
    """(name, value, step) for every scalar RecordingLogger saw."""
    return logger.scalars


# --- the metadata records the source -------------------------------------------


class TestInitFromIsRecorded:
    def _capture(self, monkeypatch):
        captured = {}

        class Uploader:
            def __init__(self, bucket_name):
                pass

            def upload_model(self, model, model_name, architecture, metadata):
                captured.update(metadata)
                return "uri"

        monkeypatch.setattr(trainer_module, "ModelUploaderAndDownloader", Uploader)
        return captured

    def _save(self, model):
        trainer_module.save_epoch_checkpoint(model=model, task="coco_seg", epoch=0,
                                             val_metrics={"val_iou": 0.3},
                                             avg_loss=1.0, val_loss=1.0)

    def test_a_warm_started_model_records_its_source_and_no_parent(self, monkeypatch):
        captured = self._capture(monkeypatch)
        model = torch.nn.Linear(2, 2)
        model.init_from, model.resumed_from = SOURCE, None
        self._save(model)
        assert captured["init_from"] == SOURCE
        assert captured["resumed_from"] is None

    def test_any_other_model_records_null(self, monkeypatch):
        captured = self._capture(monkeypatch)
        self._save(torch.nn.Linear(2, 2))
        assert "init_from" in captured and captured["init_from"] is None


def test_the_final_uploads_record_the_source_and_build_the_asked_version(monkeypatch):
    """Through train_hcm_imagenet1k: the _best/_last uploads, which a later
    resume would pick, say where the weights came from and that they are not a
    continuation."""
    logger, uploader, built = RecordingLogger(), RecordingUploader(), {}

    def load(**kw):
        built.update(kw)
        return TinyClassifier()

    for name, value in (
        ("logger_config", lambda **kw: {"fake": True}),
        ("MLFlowHCMLogger", lambda **kw: logger),
        ("ModelUploaderAndDownloader", lambda **kw: uploader),
        ("get_training_dataloader", lambda **kw: [torch.randn(1)]),
        ("get_validation_dataloader", lambda **kw: [torch.randn(1)]),
        ("load_hcm_model", load),
        ("set_env_vars", lambda *a, **k: None),
        ("download_checkpoint_state", lambda spec: TinyClassifier().state_dict()),
        ("train_model_for_epochs", lambda **kw: {
            "best_epoch": 0, "best_score": 1.0, "best_state": TinyClassifier().state_dict(),
            "best_metrics": {"val_acc1": 1.0, "val_acc5": 2.0}, "best_train_loss": 0.1,
            "last_epoch": 0}),
    ):
        monkeypatch.setattr(trainer_module, name, value)

    train_hcm_imagenet1k(new_model=True, init_from=SOURCE, init_decoder_arch=DECODER_V1,
                         max_epochs=1, batch_size=2, test_mode=True)

    assert built["new_model"] is True and built["decoder_arch"] == DECODER_V1
    assert uploader.uploads, "the run uploaded nothing"
    for upload in uploader.uploads:
        assert upload["metadata"]["init_from"] == SOURCE, upload["model_name"]
        assert upload["metadata"]["resumed_from"] is None, upload["model_name"]
    assert logger.finish_statuses() == ["FINISHED"]


# --- refused before any run exists ---------------------------------------------


def _explode(*args, **kwargs):
    raise AssertionError("an unusable --init-from must be refused before any run is created")


class TestTheFlagIsRefused:
    def test_the_default_is_no_warm_start(self):
        params = inspect.signature(train_hcm_imagenet1k).parameters
        assert params["init_from"].default is None
        assert params["init_decoder_arch"].default == DECODER_V2

    def test_a_warm_start_cannot_be_a_resume(self, monkeypatch):
        monkeypatch.setattr(trainer_module, "MLFlowHCMLogger", _explode)
        with pytest.raises(InitFromError, match="new lineage"):
            train_hcm_imagenet1k(new_model=False, init_from=SOURCE)

    def test_an_unknown_decoder_version_is_refused(self, monkeypatch):
        monkeypatch.setattr(trainer_module, "MLFlowHCMLogger", _explode)
        with pytest.raises(InitFromError, match="init_decoder_arch"):
            train_hcm_imagenet1k(new_model=True, init_from=SOURCE, init_decoder_arch="v3")

    @pytest.mark.parametrize("spec", ["coco-models", "coco-models/", "/hcm_coco_seg_best"])
    def test_a_name_that_is_not_bucket_slash_prefix_is_refused(self, monkeypatch, spec):
        for name in ("MLFlowHCMLogger", "logger_config", "load_hcm_model"):
            monkeypatch.setattr(trainer_module, name, _explode)
        with pytest.raises(InitFromError, match="<bucket>/<prefix>"):
            train_hcm_imagenet1k(new_model=True, init_from=spec)


@pytest.mark.parametrize("spec", ["", "coco-models", "coco-models/", "/hcm_coco_seg_best"])
def test_a_source_that_is_not_bucket_slash_prefix_is_refused(spec):
    with pytest.raises(ValueError, match="<bucket>/<prefix>"):
        download_checkpoint_state(spec)


def test_the_source_is_read_from_its_own_bucket_and_prefix(monkeypatch):
    seen = {}

    class Store:
        def __init__(self, bucket_name):
            seen["bucket"] = bucket_name

        def _download_model_from_prefix(self, prefix, state_dict_only=False):
            seen["prefix"], seen["state_dict_only"] = prefix, state_dict_only
            return torch.nn.Linear(2, 2).state_dict()

    monkeypatch.setattr(model_utils, "ModelUploaderAndDownloader", Store)
    state = download_checkpoint_state(SOURCE)
    assert seen == {"bucket": "coco-models", "prefix": "hcm_coco_seg_best/20261001_035743",
                    "state_dict_only": True}
    assert set(state) == {"weight", "bias"}


def _serve(monkeypatch, state, recorded):
    """Serve ``state`` as SOURCE through the real _download_model_from_prefix,
    from a dict-backed bucket, with model_info recording ``recorded``."""
    prefix = SOURCE.split("/", 1)[1]
    bucket = _Bucket()
    buf = io.BytesIO()
    torch.save(state, buf)
    bucket.objects[f"{prefix}/model.pth"] = buf.getvalue()
    bucket.objects[f"{prefix}/metadata.json"] = json.dumps({"name": "hcm_coco_seg_best"}).encode()
    bucket.objects[f"{prefix}/model_info.pkl"] = pickle.dumps({
        "model_class": hcm_mod.hcm_nn,
        "model_config": {"enable_finetuning": True, "num_segmentation_classes": 2,
                         "out_stride": 16, "decoder_arch": recorded}})

    def store(bucket_name):
        s = object.__new__(ModelUploaderAndDownloader)
        s.bucket_name, s.client = bucket_name, bucket
        return s

    monkeypatch.setattr(model_utils, "ModelUploaderAndDownloader", store)


def test_the_source_state_dict_is_the_checkpoint_itself_not_a_rebuild(monkeypatch):
    """Through the real _download_model_from_prefix: a source without a head
    leaves the model's head new and at its init, and a source-only tensor is
    reported dropped. A non-strict rebuild would instead fill the head with
    random init, count it as copied, and lose the extra tensor."""
    full = _model(DECODER_V1, seed=1).state_dict()
    headless = {k: v for k, v in full.items() if not k.startswith("segmentation_head.")}
    assert len(headless) < len(full)
    source = {**headless, "extra.weight": torch.ones(3)}

    _serve(monkeypatch, source, recorded=DECODER_V1)
    state = _offline(lambda: download_checkpoint_state(SOURCE))
    model = _model(DECODER_V1, seed=2)
    init = {k: v.clone() for k, v in model.state_dict().items()}
    warm = warm_start_state(model, state)

    head = sorted(k for k in full if k.startswith("segmentation_head."))
    assert sorted(warm["new"]) == head
    for k in head:
        assert torch.equal(model.state_dict()[k], init[k]), k
    assert warm["dropped"] == ["extra.weight"]
    assert sorted(warm["copied"]) == sorted(headless)


# --- the command line ------------------------------------------------------------


def _cli(monkeypatch, argv):
    import main

    captured = {}
    monkeypatch.setattr(main, "run_train_hcm_imagenet1k", lambda **kw: captured.update(kw))
    parser = main.setup_subpparsers()
    args = parser.parse_args(["train_hcm_imagenet1k", *argv])
    args.func(args)
    return captured


def test_init_from_reaches_the_trainer_and_implies_a_fresh_model(monkeypatch):
    captured = _cli(monkeypatch, ["--coco", "--init-from", SOURCE])
    assert captured["init_from"] == SOURCE
    assert captured["init_decoder_arch"] == "v2"
    assert captured["new_model"] is True


def test_without_init_from_nothing_changes(monkeypatch):
    captured = _cli(monkeypatch, ["--coco"])
    assert captured["init_from"] is None
    assert captured["new_model"] is False


def test_the_service_passes_the_warm_start_through(monkeypatch):
    from ecs_hcm_model.core.services import trainer as service

    captured = {}
    monkeypatch.setattr(service, "train_hcm_imagenet1k", lambda **kw: captured.update(kw))
    service.run_train_hcm_imagenet1k(new_model=True, init_from=SOURCE, init_decoder_arch="v1")
    assert captured["init_from"] == SOURCE and captured["init_decoder_arch"] == "v1"


def test_a_source_recording_a_version_its_weights_contradict_is_refused(monkeypatch, v1_state):
    """The state-dict-only download keeps the decoder-version check."""
    _serve(monkeypatch, v1_state, recorded=DECODER_V2)
    with pytest.raises(DecoderArchMismatchError, match="'v2'.*'v1'"):
        download_checkpoint_state(SOURCE)


def test_a_v2_source_missing_one_rungs_fuse_leaves_it_new_not_initialised():
    """[I | 0] is for a v1 source only; a v2 source's missing fuse keys are new."""
    source = {k: v for k, v in _model(DECODER_V2, seed=1).state_dict().items()
              if not k.startswith("rrd_list.0.fuse.")}
    model = _model(DECODER_V2, seed=2)
    init = {k: v.clone() for k, v in model.state_dict().items()}
    warm = warm_start_state(model, source)

    assert warm["initialised"] == []
    assert sorted(warm["new"]) == ["rrd_list.0.fuse.bias", "rrd_list.0.fuse.weight"]
    for k in warm["new"]:
        assert torch.equal(model.state_dict()[k], init[k]), k
