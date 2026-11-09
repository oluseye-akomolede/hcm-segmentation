"""The class-info load path returns a built model, not the raw state dict.

Cluster runs logged "Could not load model class info: " with an empty message:
hcm_nn never recorded out_stride, the uploader pickled None, and
hcm_nn(out_stride=None) failed the backbone's bare assert (AssertionError(),
whose str() is empty). These tests drive the real upload and download code
through an in-memory stand-in for the MinIO client.
"""
import io
import pickle
from pathlib import Path

import pytest
import torch

from ecs_hcm_model.core.hcm_nn import hcm_nn
from ecs_hcm_model.core.model_upload_download import (
    ModelUploaderAndDownloader,
    build_model_from_info,
)

PREFIX = "hcm_coco_seg/20261001_000000"


class FakeMinio:
    """Just the three calls the uploader/downloader make, backed by a dict."""

    def __init__(self):
        self.objects = {}

    def fput_object(self, bucket, name, path):
        self.objects[name] = Path(path).read_bytes()

    def fget_object(self, bucket, name, path):
        if name not in self.objects:
            raise FileNotFoundError(name)
        Path(path).write_bytes(self.objects[name])


def _store():
    # Bypass __init__: it needs MinIO credentials and a live endpoint.
    store = object.__new__(ModelUploaderAndDownloader)
    store.bucket_name = "coco-models"
    store.client = FakeMinio()
    return store


def _put_checkpoint(store, state_dict, model_info=None):
    """Write a checkpoint the way upload_model lays one out."""
    objs = store.client.objects
    b = io.BytesIO()
    torch.save(state_dict, b)
    objs[f"{PREFIX}/model.pth"] = b.getvalue()
    objs[f"{PREFIX}/metadata.json"] = b'{"name": "hcm_coco_seg", "architecture": "hcm"}'
    if model_info is not None:
        objs[f"{PREFIX}/model_info.pkl"] = pickle.dumps(model_info)


def _same_weights(a, b):
    sa, sb = a.state_dict(), b.state_dict()
    return sa.keys() == sb.keys() and all(torch.equal(sa[k], sb[k]) for k in sa)


@pytest.fixture(scope="module")
def trained():
    torch.manual_seed(0)
    return hcm_nn(num_segmentation_classes=2, out_stride=16).eval()


class Unbuildable:
    """Fails construction the way the backbone did: a bare assert.

    Raised explicitly: pytest rewrites asserts in test modules and would give
    this one a message, and the point is an exception whose str() is empty.
    """

    def __init__(self):
        raise AssertionError()


def test_hcm_nn_records_its_config(trained):
    assert trained.out_stride == 16
    assert trained.num_segmentation_classes == 2


def test_a_legacy_none_config_still_loads_with_its_head(trained, capsys):
    store = _store()
    legacy_info = {
        "model_class": hcm_nn,
        "model_config": {"num_segmentation_classes": None, "out_stride": None,
                         "enable_finetuning": False},
    }
    _put_checkpoint(store, trained.state_dict(), legacy_info)

    model = store._download_model_from_prefix(PREFIX)

    assert isinstance(model, hcm_nn), capsys.readouterr().out
    assert model.segmentation_head.out_channels == 2
    assert model.out_stride == 16
    assert _same_weights(model, trained)
    out = capsys.readouterr().out
    assert "Could not load model class info" not in out
    assert "key mismatch" not in out


def test_a_new_upload_round_trips(trained, capsys):
    store = _store()
    store.upload_model(trained, "hcm_coco_seg", architecture="hcm")

    info_name = next(n for n in store.client.objects if n.endswith("model_info.pkl"))
    cfg = pickle.loads(store.client.objects[info_name])["model_config"]
    assert cfg["out_stride"] == 16
    assert cfg["num_segmentation_classes"] == 2

    model = store._download_model_from_prefix(info_name.rsplit("/", 1)[0])

    assert isinstance(model, hcm_nn)
    assert model.segmentation_head.out_channels == 2
    assert _same_weights(model, trained)
    assert "Could not load model class info" not in capsys.readouterr().out


def test_unusable_class_info_returns_the_state_dict_with_a_typed_warning(trained, capsys):
    store = _store()
    _put_checkpoint(store, trained.state_dict(), {"model_class": Unbuildable})

    result = store._download_model_from_prefix(PREFIX)

    assert isinstance(result, dict)
    assert result.keys() == trained.state_dict().keys()
    out = capsys.readouterr().out
    assert "Could not load model class info: AssertionError: AssertionError()" in out
    assert "Returning state dict only" in out


def test_absent_class_info_returns_the_state_dict(trained, capsys):
    store = _store()
    _put_checkpoint(store, trained.state_dict(), model_info=None)

    result = store._download_model_from_prefix(PREFIX)

    assert isinstance(result, dict)
    assert "FileNotFoundError" in capsys.readouterr().out


def test_a_key_mismatch_is_reported_not_silent(trained, capsys):
    state = dict(trained.state_dict())
    state["bogus.weight"] = torch.zeros(1)
    del state["segmentation_head.bias"]
    info = {"model_class": hcm_nn,
            "model_config": {"num_segmentation_classes": 2, "out_stride": 16}}

    model = build_model_from_info(info, state)

    assert isinstance(model, hcm_nn)
    out = capsys.readouterr().out
    assert "key mismatch: 1 missing ['segmentation_head.bias']" in out
    assert "1 unexpected ['bogus.weight']" in out
