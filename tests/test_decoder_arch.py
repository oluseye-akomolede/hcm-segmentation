"""The decoder architecture version: v1 (legacy, bit-identical) and v2 (fixed).

v1 had three geometry faults, all fixed under v2 only:
  A. hcm_rrd concatenated its two gated copies along the WIDTH and halved it
     back with a stride-(1,2) conv, so output column j came from input 2j.
  B. hcm_csc returned only an upsampled 16x16 attention map; no full-resolution
     detail reached the decoder.
  C. r4 was not resized to the working resolution with r2/r3/r5.

CPU-only and offline: the ResNet-50 fetch is stubbed as in test_fp16_overflow.
"""

import hashlib
import importlib.util
import io
import json
import pickle
from pathlib import Path

import pytest
import torch

from ecs_hcm_model.core import hcm_nn as hcm_mod
from ecs_hcm_model.core.hcm_nn import (
    DECODER_V1,
    DECODER_V2,
    hcm_csc,
    DecoderArchMismatchError,
    hcm_rrd,
    infer_decoder_arch,
)
from ecs_hcm_model.core.model_upload_download import (
    ModelUploaderAndDownloader,
    build_model_from_info,
)
from jetson.model_shape import ModelShapeError, resolve_decoder_arch

REPO = Path(__file__).resolve().parents[1]
C = hcm_mod.ISC_OUTPUT_CHANNEL_NUM


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


def _model(arch, seed=0):
    torch.manual_seed(seed)
    return _offline(lambda: hcm_mod.hcm_nn(enable_finetuning=True, num_segmentation_classes=2,
                                           out_stride=16, decoder_arch=arch)).eval()


def _column_profile(module, H=8, W=32, col=10, seed=0):
    """Per-output-column response of ``module`` to a delta column at ``col``."""
    torch.manual_seed(seed)
    p = torch.randn(1, C, H, W)
    base = torch.randn(1, C, H, W) * 0.1
    poked = base.clone()
    poked[..., col] += 5.0
    with torch.no_grad():
        diff = (module(poked, p) - module(base, p)).abs()
    return diff.sum(dim=(0, 1, 2))


# ---------------------------------------------------------------------------
# A. hcm_rrd keeps the horizontal geometry under v2
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("arch", [DECODER_V1, DECODER_V2])
def test_rrd_output_keeps_the_input_width(arch):
    torch.manual_seed(0)
    rrd = hcm_rrd(arch).eval()
    out = rrd(torch.randn(1, C, 8, 32), torch.randn(1, C, 8, 32))
    assert out.shape[-2:] == (8, 32)


def _peak(arch, col):
    torch.manual_seed(0)
    rrd = hcm_rrd(arch).eval()
    return int(_column_profile(rrd, col=col).argmax())


COLUMNS = (6, 10, 14, 18, 22)


def _offsets(arch):
    return {_peak(arch, c) - c for c in COLUMNS}


def test_rrd_is_shift_equivariant_under_v2():
    """The response follows the input column: one constant offset, within a pixel.

    The offset itself comes from the random 3x3 kernels (measured: +1), not from
    the geometry, so the test pins equivariance rather than an exact column.
    """
    offsets = _offsets(DECODER_V2)
    assert len(offsets) == 1 and abs(next(iter(offsets))) <= 1, offsets


def test_rrd_was_not_shift_equivariant_under_v1():
    """The fault the v2 fix removes: v1 answers at about half the column."""
    assert len(_offsets(DECODER_V1)) > 1


# ---------------------------------------------------------------------------
# B. hcm_csc passes full-resolution detail under v2
# ---------------------------------------------------------------------------

def _csc_detail_ratio(arch):
    """Response at a poked pixel over the mean response, for a single-pixel poke.

    Measured: v1 2.8 (the detail is smeared over the 16x16 attention map),
    v2 42.9 (the residual carries it at full resolution).
    """
    torch.manual_seed(0)
    csc = hcm_csc(arch).eval()
    H = W = 48
    f_s = torch.randn(1, C, H, W) * 0.1
    up = torch.randn(1, C, 12, 12) * 0.1
    poked = f_s.clone()
    poked[..., 20, 30] += 5.0
    with torch.no_grad():
        diff = (csc(poked, up) - csc(f_s, up)).abs().sum(dim=(0, 1))
    return float(diff[20, 30] / diff.mean())


def test_a_fine_detail_in_f_s_reaches_the_csc_output_under_v2():
    assert _csc_detail_ratio(DECODER_V2) > 10


def test_v1_csc_smeared_the_detail():
    assert _csc_detail_ratio(DECODER_V1) < 10


# ---------------------------------------------------------------------------
# C. all four ISC rungs run at one size under v2
# ---------------------------------------------------------------------------

def _isc_sizes(arch):
    model = _model(arch)
    sizes = []
    hooks = [m.register_forward_hook(lambda _m, _i, o: sizes.append(tuple(o.shape[-2:])))
             for m in model.isc_list]
    with torch.no_grad():
        model(torch.randn(1, 3, 192, 192))
    for h in hooks:
        h.remove()
    return sizes


def test_all_four_isc_outputs_share_one_size_under_v2():
    sizes = _isc_sizes(DECODER_V2)
    assert len(sizes) == 4 and len(set(sizes)) == 1, sizes


def test_v1_ran_isc2_at_another_size():
    sizes = _isc_sizes(DECODER_V1)
    assert len(set(sizes)) == 2, sizes


# ---------------------------------------------------------------------------
# Whole model
# ---------------------------------------------------------------------------

def test_v2_output_has_the_same_shape_as_v1():
    x = torch.randn(1, 3, 192, 192)
    shapes = []
    for arch in (DECODER_V1, DECODER_V2):
        with torch.no_grad():
            out = _model(arch)(x)
        out = out if torch.is_tensor(out) else out[0]
        shapes.append(tuple(out.shape))
    assert shapes[0] == shapes[1]


def test_v2_adds_only_the_fuse_convs():
    v1 = set(_model(DECODER_V1).state_dict())
    v2 = set(_model(DECODER_V2).state_dict())
    assert v1 <= v2
    added = v2 - v1
    assert added and all(k.startswith("rrd_list.") and ".fuse." in k for k in added)


def test_the_version_is_read_back_from_the_weights():
    assert infer_decoder_arch(_model(DECODER_V1).state_dict()) == DECODER_V1
    assert infer_decoder_arch(_model(DECODER_V2).state_dict()) == DECODER_V2


def test_an_unknown_version_is_refused():
    with pytest.raises(ValueError):
        hcm_rrd("v3")
    with pytest.raises(ValueError):
        hcm_csc("width")


# ---------------------------------------------------------------------------
# Old checkpoints: v1 is bit-identical to the committed model
# ---------------------------------------------------------------------------

# The oracle is frozen: a byte-for-byte copy of hcm_nn.py as of b005aaf, the
# last commit before the decoder was versioned. Reading it from git would need
# a .git (absent in CI) and, after merge, would compare v2 with itself.
LEGACY_HCM_NN = Path(__file__).parent / "fixtures" / "hcm_nn_legacy_b005aaf.py"
LEGACY_HCM_NN_SHA256 = "0ef4d22643cdd98bb95213d2c6181b596667f871a8d60e5c57ee76238b4d8f29"


def _committed_hcm_module():
    assert hashlib.sha256(LEGACY_HCM_NN.read_bytes()).hexdigest() == LEGACY_HCM_NN_SHA256, \
        "the frozen legacy hcm_nn.py was edited; it must stay the b005aaf bytes"
    spec = importlib.util.spec_from_file_location("_legacy_hcm_nn_b005aaf", LEGACY_HCM_NN)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# Segmentation (fine-tuning, 2 classes: the 1x1 segmentation head) and
# classification (no fine-tuning: forward_internal goes on to the ROI module and
# the 1000-way roi_classifier, which fine-tuning returns before).
@pytest.mark.parametrize("enable_finetuning,num_segmentation_classes,out_shape", [
    (True, 2, (1, 2, 48, 48)),
    (False, None, (1, 1000)),
], ids=["seg", "cls"])
def test_a_legacy_checkpoint_runs_bit_identically_under_v1(
        enable_finetuning, num_segmentation_classes, out_shape):
    old = _committed_hcm_module()
    torch.manual_seed(0)
    legacy = _offline(lambda: old.hcm_nn(enable_finetuning=enable_finetuning,
                                         num_segmentation_classes=num_segmentation_classes,
                                         out_stride=16)).eval()
    sd = legacy.state_dict()
    assert infer_decoder_arch(sd) == DECODER_V1

    rebuilt = _offline(lambda: build_model_from_info(
        {"model_class": hcm_mod.hcm_nn,
         "model_config": {"enable_finetuning": enable_finetuning,
                          "num_segmentation_classes": num_segmentation_classes,
                          "out_stride": 16}},
        sd))
    assert isinstance(rebuilt, torch.nn.Module)
    assert rebuilt.decoder_arch == DECODER_V1
    assert rebuilt.enable_finetuning is enable_finetuning
    # build_model_from_info loads non-strict; a renamed or dropped v1
    # parameter must fail here rather than leave a tensor at its init.
    rebuilt.load_state_dict(sd, strict=True)
    rebuilt.eval()

    x = torch.randn(1, 3, 192, 192)
    with torch.no_grad():
        a, b = legacy(x), rebuilt(x)
    a = a if torch.is_tensor(a) else a[0]
    b = b if torch.is_tensor(b) else b[0]
    assert tuple(a.shape) == out_shape
    assert torch.equal(a, b)


def test_a_new_upload_records_and_rebuilds_v2():
    trained = _model(DECODER_V2)
    rebuilt = _offline(lambda: build_model_from_info(
        {"model_class": hcm_mod.hcm_nn,
         "model_config": {"enable_finetuning": True, "num_segmentation_classes": 2,
                          "out_stride": 16, "decoder_arch": DECODER_V2}},
        trained.state_dict()))
    assert rebuilt.decoder_arch == DECODER_V2



# A recorded version the weights contradict raises; it is never loaded
# non-strict into the wrong decoder (fuse convs dropped, or left random).

def _info(arch):
    return {"model_class": hcm_mod.hcm_nn,
            "model_config": {"enable_finetuning": True, "num_segmentation_classes": 2,
                             "out_stride": 16, "decoder_arch": arch}}


@pytest.mark.parametrize("recorded,weights", [(DECODER_V1, DECODER_V2), (DECODER_V2, DECODER_V1)])
def test_a_contradicted_recorded_version_raises(recorded, weights):
    sd = _model(weights).state_dict()
    with pytest.raises(DecoderArchMismatchError, match=f"{recorded!r}.*{weights!r}"):
        _offline(lambda: build_model_from_info(_info(recorded), sd))


def test_an_unknown_recorded_version_is_named_as_unknown():
    sd = _model(DECODER_V1).state_dict()
    with pytest.raises(DecoderArchMismatchError, match=r"^unknown decoder_arch 'v9'"):
        _offline(lambda: build_model_from_info(_info("v9"), sd))


class _Bucket:
    """The two MinIO calls the download path makes, backed by a dict."""

    def __init__(self):
        self.objects = {}

    def fget_object(self, bucket, name, path):
        if name not in self.objects:
            raise FileNotFoundError(name)
        Path(path).write_bytes(self.objects[name])

    def list_objects(self, bucket, prefix=None, recursive=False):
        return [type("O", (), {"object_name": n})() for n in sorted(self.objects)
                if n.startswith(prefix or "")]


@pytest.mark.parametrize("recorded,weights", [(DECODER_V1, DECODER_V2), (DECODER_V2, DECODER_V1)])
def test_the_download_path_raises_instead_of_returning_a_state_dict(recorded, weights):
    """The downloader's handlers turn other rebuild errors into "state dict
    only" or None; a contradicted version must reach the caller."""
    store = object.__new__(ModelUploaderAndDownloader)
    store.bucket_name, store.client = "coco-models", _Bucket()
    prefix = "hcm_coco_seg_best/20261002_000000"
    buf = io.BytesIO()
    torch.save(_model(weights).state_dict(), buf)
    store.client.objects[f"{prefix}/model.pth"] = buf.getvalue()
    store.client.objects[f"{prefix}/model_info.pkl"] = pickle.dumps(_info(recorded))
    store.client.objects[f"{prefix}/metadata.json"] = json.dumps({
        "name": "hcm_coco_seg_best", "architecture": "hcm",
        "upload_timestamp": "2026-10-02T00:00:00",
        "custom_metadata": {"is_best": True, "decoder_arch": recorded}}).encode()
    with pytest.raises(DecoderArchMismatchError):
        _offline(lambda: store.search_and_download_model_by_architecture("hcm"))


# ---------------------------------------------------------------------------
# Demos and eval: the pure resolver
# ---------------------------------------------------------------------------

V1_KEYS = {"rrd_list.0.conv2d.weight": 0}
V2_KEYS = {"rrd_list.0.conv2d.weight": 0, "rrd_list.0.fuse.weight": 0}


def test_resolver_reads_an_unrecorded_checkpoint_as_v1():
    assert resolve_decoder_arch({}, V1_KEYS) == (DECODER_V1, "weights")
    assert resolve_decoder_arch(None, V2_KEYS) == (DECODER_V2, "weights")


def test_resolver_prefers_the_custom_metadata():
    meta = {"custom_metadata": {"decoder_arch": "v2"}, "decoder_arch": "v2"}
    assert resolve_decoder_arch(meta, V2_KEYS) == (DECODER_V2, "metadata.custom_metadata.decoder_arch")
    assert resolve_decoder_arch({"decoder_arch": "v2"}, V2_KEYS) == (DECODER_V2, "metadata.decoder_arch")


@pytest.mark.parametrize("meta,keys", [
    ({"decoder_arch": "v2"}, V1_KEYS),
    ({"custom_metadata": {"decoder_arch": "v1"}}, V2_KEYS),
    ({"decoder_arch": "width"}, V1_KEYS),
    # The two stated values disagree: the custom one agrees with the weights,
    # the top-level one does not. Both are checked, so this raises too.
    ({"custom_metadata": {"decoder_arch": "v2"}, "decoder_arch": "v1"}, V2_KEYS),
    ({"custom_metadata": {"decoder_arch": "v1"}, "decoder_arch": "v2"}, V1_KEYS),
])
def test_resolver_refuses_a_contradicted_or_unknown_version(meta, keys):
    with pytest.raises(ModelShapeError):
        resolve_decoder_arch(meta, keys)


@pytest.mark.parametrize("path", ["jetson/demo_model.py", "ecs_hcm_model/eval/seg.py"])
def test_every_consumer_builds_the_resolved_version(path):
    src = (REPO / path).read_text()
    assert "resolve_decoder_arch(meta, " in src
    assert "decoder_arch=decoder_arch" in src
