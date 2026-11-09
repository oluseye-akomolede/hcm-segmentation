"""HCM v2: the backbone ladder, the neck, the finer mask and the arch record.

With no new argument the model built here is the pre-change ResNet-50 one: the
same key set and the same forward, so a checkpoint from before these fields
existed loads, and behaves, exactly as it did. Most of this file is that claim,
plus the loaders that have to keep it true.

CPU-only and offline. The torchvision fetch is stubbed as in
``test_decoder_arch`` -- the factory calls through ``torchvision.models`` rather
than a from-import, so the stub reaches it. The timm cells skip when this host's
timm does not know the name (the CI runner image inherits timm 0.9.0, which lacks
EfficientViT; the trainer image this change pins is timm==0.9.16).
"""

import ast
import importlib.util
import json
from pathlib import Path

import pytest
import torch

from ecs_hcm_model.core import backbones as bb
from ecs_hcm_model.core import hcm_nn as hcm_mod
from ecs_hcm_model.core.backbones import (
    BACKBONE_CONVNEXTV2_NANO,
    BACKBONE_CONVNEXTV2_BASE,
    BACKBONE_CONVNEXTV2_TINY,
    BACKBONE_EFFICIENTVIT_B2,
    BACKBONE_RESNET50,
    DEFAULT_BACKBONE,
    backbone_names,
    backbone_spec,
    build_backbone,
)
from ecs_hcm_model.core.hcm_nn import (
    ArchMismatchError,
    DECODER_V2,
    ISC_OUTPUT_CHANNEL_NUM,
    MASK_REFINE_CHANNELS,
    MASK_STRIDE_DEFAULT,
    MASK_STRIDES,
    NECK_CHANNELS,
    NECK_LEGACY,
    arch_record_of,
    check_backbone,
    check_mask_stride,
    check_neck_channels,
    hcm_isc,
    infer_backbone,
    infer_mask_stride,
    infer_neck_channels,
    validate_arch,
)
from ecs_hcm_model.core.model_upload_download import build_model_from_info
from ecs_hcm_model.core.trainer import (
    calculate_segmentation_loss,
    iou_dice_fg_from_logits,
    target_grid_follows_logits,
)

REPO = Path(__file__).resolve().parents[1]
C = ISC_OUTPUT_CHANNEL_NUM

pytestmark = pytest.mark.filterwarnings("ignore::UserWarning")


def _offline(build):
    """Build resnet50 without reaching the network, as test_decoder_arch does."""
    from torchvision import models as tv_models

    real = tv_models.resnet50

    def offline_resnet50(weights=None, replace_stride_with_dilation=None):
        return real(weights=None, replace_stride_with_dilation=replace_stride_with_dilation)

    tv_models.resnet50 = offline_resnet50
    try:
        return build()
    finally:
        tv_models.resnet50 = real


def _model(**kwargs):
    kwargs.setdefault("enable_finetuning", True)
    kwargs.setdefault("num_segmentation_classes", 2)
    kwargs.setdefault("out_stride", 16)
    # The pretrained ImageNet fetch is the trainer's business, not a unit
    # test's: it is a network call (and a slow one, or a hang, where the hub is
    # unreachable). Only the training entry point asks for it.
    kwargs.setdefault("backbone_pretrained", False)
    torch.manual_seed(0)
    return _offline(lambda: hcm_mod.hcm_nn(**kwargs)).eval()


# ---------------------------------------------------------------------------
# The ladder: a registry, resnet50 by default, an unknown name refused.


def test_the_default_backbone_is_todays_resnet50():
    assert DEFAULT_BACKBONE == BACKBONE_RESNET50
    assert backbone_names()[0] == BACKBONE_RESNET50
    assert backbone_spec(DEFAULT_BACKBONE).rung_channels == (256, 512, 1024, 2048)


def test_an_unknown_backbone_is_refused_by_name():
    with pytest.raises(ValueError, match=r"unknown backbone 'nope'.*resnet50"):
        backbone_spec("nope")
    with pytest.raises(ValueError, match=r"unknown backbone 'nope'"):
        build_backbone("nope")


def test_a_model_built_with_no_backbone_argument_is_the_default_one():
    assert _model().backbone_name == DEFAULT_BACKBONE


def test_a_timm_backbone_refuses_an_out_stride_it_cannot_honour():
    # The resnet entry is the only one whose ladder is restride-able; silently
    # ignoring the request would build a model at the wrong stride.
    with pytest.raises(ValueError, match="fixed 4/8/16/32 ladder"):
        build_backbone(BACKBONE_EFFICIENTVIT_B2, out_stride=8)


def test_the_resnet_entry_refuses_a_weights_path():
    with pytest.raises(ValueError, match="weights_path is only used by the timm backbones"):
        build_backbone(BACKBONE_RESNET50, weights_path="/tmp/whatever.pth")


def test_a_missing_timm_is_named_in_the_error(monkeypatch):
    """A backbone that needs timm says so, naming the pin, rather than a bare ImportError."""
    import sys

    monkeypatch.setitem(sys.modules, "timm", None)  # makes ``import timm`` raise
    with pytest.raises(ImportError, match=r"timm==0\.9\.16"):
        build_backbone(BACKBONE_EFFICIENTVIT_B2)


# ---------------------------------------------------------------------------
# The neck: a configurable width, and the pre-change widening when unasked.


def test_the_default_model_keeps_the_pre_change_widening():
    """No new argument, no new keys: the pre-change neck is a bare 1x1."""
    model = _model()
    assert model.neck_channels is None
    keys = model.state_dict()
    assert "adapt_c2.weight" in keys
    assert "adapt_c2.0.weight" not in keys
    assert keys["adapt_c2.weight"].shape == (2048, 256, 1, 1)
    # The decoder reads the widened 2048, as it did.
    assert model.ASPP.conv1x1.in_channels == 2048
    assert model.isc_list[0].conv1_1_1x1.in_channels == 2048


def test_every_neck_rung_projects_to_neck_channels():
    model = _model(neck_channels=128)
    assert model.neck_channels == 128
    for rung in ("adapt_c2", "adapt_c3", "adapt_c4", "adapt_c5"):
        block = getattr(model, rung)
        assert block[0].out_channels == 128
        assert block[0].in_channels == backbone_spec(DEFAULT_BACKBONE).rung_channels[
            ("adapt_c2", "adapt_c3", "adapt_c4", "adapt_c5").index(rung)]
        # 1x1 then GroupNorm, in that order, as the spec delta says.
        assert isinstance(block[1], torch.nn.GroupNorm)
        assert block[1].num_channels == 128
    assert not any(k.endswith("adapt_c5.weight") for k in model.state_dict())


def test_a_default_neck_on_a_timm_backbone_is_the_configurable_width():
    """The pre-change widening is preserved for resnet50 only: nothing else had one."""
    assert validate_arch(BACKBONE_EFFICIENTVIT_B2, None, 4) == NECK_CHANNELS


def test_the_isc_and_aspp_take_the_neck_width():
    model = _model(neck_channels=128)
    assert model.ASPP.conv1x1.in_channels == 128
    assert model.ASPP.conv_rates[0].in_channels == 128
    for isc in model.isc_list:
        assert isc.conv1_1_1x1.in_channels == 128
        assert isc.conv1_2_1x1.in_channels == 128
        assert isc.conv1_3_1x1.in_channels == 128
    assert model.isc_list[0].conv2_1_3x3.in_channels == 128
    # ... and still emits the 32 channels the decoder is built around.
    with torch.no_grad():
        assert model.isc_list[0](torch.randn(1, 128, 8, 8)).shape == (1, C, 8, 8)
    with torch.no_grad():
        assert model.ASPP(torch.randn(1, 128, 8, 8)).shape[1:] == (C, 4, 4)


@pytest.mark.parametrize("neck,mid,inner", [(2048, 256, 64), (256, 256, 64),
                                            (128, 128, 64), (64, 64, 64), (16, 16, 16)])
def test_the_isc_inner_widths_scale_with_the_neck(neck, mid, inner):
    """The chosen rule: mid=min(neck,256), inner=min(mid,64) -- today's 256/64 at 2048."""
    isc = hcm_isc(neck)
    assert isc.mid_channels == mid
    assert isc.inner_channels == inner
    assert isc.conv1_2_1x1.out_channels == mid
    assert isc.conv2_1_3x3.out_channels == inner
    with torch.no_grad():
        assert isc(torch.randn(1, neck, 6, 6)).shape == (1, C, 6, 6)


def test_a_neck_width_that_groupnorm_cannot_divide_still_builds():
    model = _model(neck_channels=100)
    assert model.adapt_c2[1].num_channels == 100
    assert model.adapt_c2[1].num_groups <= 32
    assert 100 % model.adapt_c2[1].num_groups == 0


@pytest.mark.parametrize("bad", [0, -1, 1.5, "128", True])
def test_an_unusable_neck_width_is_refused(bad):
    with pytest.raises(ValueError, match="neck_channels"):
        validate_arch(DEFAULT_BACKBONE, bad, 4)


# ---------------------------------------------------------------------------
# The finer mask: stride 2 by request, stride 4 by default.


def test_the_default_mask_stride_is_four_and_the_keys_are_todays():
    model = _model()
    assert model.mask_stride == MASK_STRIDE_DEFAULT == 4
    assert model.mask_conv1 is None and model.mask_conv2 is None and model.mask_proj is None
    assert not [k for k in model.state_dict() if k.startswith("mask_")]


@pytest.mark.parametrize("size,stride4,stride2", [(352, 88, 176), (64, 16, 32)])
def test_the_mask_grid_is_the_input_over_the_mask_stride(size, stride4, stride2):
    x = torch.randn(1, 3, size, size)
    with torch.no_grad():
        assert tuple(_model()(x).shape[-2:]) == (stride4, stride4)
        assert tuple(_model(mask_stride=2)(x).shape[-2:]) == (stride2, stride2)


def test_the_stride_two_mask_is_built_from_the_backbones_half_resolution_tap():
    model = _model(mask_stride=2)
    spec = backbone_spec(DEFAULT_BACKBONE)
    assert model.mask_proj.in_channels == spec.tap_channels == 64
    # Two sources at the decoder's own width, then a 3x3 refinement back to it,
    # so the segmentation head is the one the stride-4 model already has.
    assert model.mask_conv1[0].in_channels == 2 * C
    assert model.mask_conv1[0].out_channels == MASK_REFINE_CHANNELS
    assert model.mask_conv2[0].in_channels == MASK_REFINE_CHANNELS
    assert model.mask_conv2[0].out_channels == C
    assert model.segmentation_head.in_channels == C


def test_a_stride_four_head_loads_into_the_stride_two_model():
    """The refinement returns to the pre-change head width, so the head is shared."""
    stride4 = _model()
    stride2 = _model(mask_stride=2)
    assert stride2.segmentation_head.weight.shape == stride4.segmentation_head.weight.shape
    head = {"segmentation_head." + k: v for k, v in stride4.segmentation_head.state_dict().items()}
    missing, unexpected = stride2.load_state_dict(head, strict=False)
    assert not unexpected
    # Every head tensor the stride-4 model has is consumed by the stride-2 model
    # (the rest of that model is simply not in this partial load).
    assert not [k for k in missing if k.startswith("segmentation_head")]
    assert [k for k in missing if k.startswith("mask_")], "the refinement's own keys"


def test_a_stride_two_model_is_not_the_stride_four_model():
    """The refinement has to change the mask, not merely exist."""
    x = torch.randn(1, 3, 64, 64)
    with torch.no_grad():
        torch.manual_seed(0)
        a = _model()(x)
        torch.manual_seed(0)
        b = _model(mask_stride=2)(x)
    assert a.shape != b.shape or not torch.allclose(a, b)


@pytest.mark.parametrize("bad", [0, 1, 3, 8, "2", None])
def test_an_unsupported_mask_stride_is_refused(bad):
    with pytest.raises(ValueError, match="mask_stride"):
        validate_arch(DEFAULT_BACKBONE, None, bad)


def test_the_mask_stride_without_the_head_is_still_refused():
    """enable_finetuning=False has no mask head; the stride is still validated."""
    torch.manual_seed(0)
    with pytest.raises(ValueError, match="mask_stride"):
        _offline(lambda: hcm_mod.hcm_nn(enable_finetuning=False, num_segmentation_classes=None,
                                        out_stride=16, mask_stride=3))


def test_a_backbone_that_loses_its_declared_tap_is_refused():
    model = _model(mask_stride=2)
    real = model.backbone.forward_with_tap

    def no_tap(x):
        rungs, _ = real(x)
        return rungs, None

    model.backbone.forward_with_tap = no_tap
    with pytest.raises(RuntimeError, match="declares a stride-2 tap but produced none"):
        with torch.no_grad():
            model(torch.randn(1, 3, 64, 64))


def test_a_tap_on_the_wrong_grid_is_refused():
    model = _model(mask_stride=2)
    real = model.backbone.forward_with_tap

    def wrong_tap(x):
        rungs, _ = real(x)
        return rungs, rungs[1]  # stride 8, not the half-resolution feature

    model.backbone.forward_with_tap = wrong_tap
    with pytest.raises(RuntimeError, match="is not this backbone's half-resolution feature"):
        with torch.no_grad():
            model(torch.randn(1, 3, 64, 64))


def test_the_roi_path_still_reads_the_stride_four_decoder_map():
    """The refinement is the segmentation head's business, not the classifier's."""
    model = _model(mask_stride=2)
    assert model.roi_classifier[0].output_size == (1, 1)
    assert model.differentiable_roi_module is not None
    x = torch.randn(1, 3, 64, 64)
    with torch.no_grad():
        assert model(x).shape[-2:] == (32, 32)
    # The classifier path consumes rrd_output[0] itself; the refinement never
    # sees it. Pin that by asking for the classifier: 1000 classes, pooled.
    with torch.no_grad():
        flat = _model(enable_finetuning=False, mask_stride=2)
        assert flat(x).shape == (1, 1000)


# ---------------------------------------------------------------------------
# timm backbones. The question is not whether timm imports but what it knows.
# The image the CI Job runs (ecs_hcm_trainer:b6e1977) inherits timm 0.9.0 from its
# NGC base: `import timm` succeeds there, and convnextv2_nano.fcmae_ft_in1k
# builds, but efficientvit_b2.r224_in1k is an unknown model (EfficientViT landed
# in timm 0.9.9). The pin in dockerfiles/ecs_hcm_trainer/Dockerfile is
# timm==0.9.16, so until a runner image is built from it the efficientvit cells
# skip (naming this host's timm version) and the convnextv2 cells still run.

HAS_TIMM = importlib.util.find_spec("timm") is not None
_TIMM_KNOWS = {}

needs_timm = pytest.mark.skipif(
    not HAS_TIMM, reason="the timm backbones need timm (the trainer image pins timm==0.9.16)")


def _timm_knows(name):
    """Whether this environment's timm can build the registry name `name`."""
    if name not in _TIMM_KNOWS:
        timm_name = getattr(backbone_spec(name), "timm_name", None)
        ok = False
        if HAS_TIMM and timm_name is not None:
            import timm

            try:
                timm.create_model(timm_name, pretrained=False, features_only=True)
                ok = True
            except Exception:
                ok = False
        _TIMM_KNOWS[name] = ok
    return _TIMM_KNOWS[name]


def _require_timm(name):
    """Skip a cell this host's timm cannot express, naming the version."""
    if not HAS_TIMM:
        pytest.skip("no timm in this environment (the trainer image pins timm==0.9.16)")
    if not _timm_knows(name):
        import timm

        pytest.skip(f"this timm ({timm.__version__}) does not know "
                    f"{backbone_spec(name).timm_name}; the trainer image pins timm==0.9.16")


def _buildable_backbones():
    """The registry names this host can actually build; resnet50 is always one."""
    return [n for n in backbone_names() if n == BACKBONE_RESNET50 or _timm_knows(n)]


@needs_timm
def test_a_name_this_timm_cannot_build_is_a_skip_not_a_failure(monkeypatch):
    """The gate asks the capability, not importability.

    This is the regression: the CI runner image's timm 0.9.0 answers `import timm`
    but not `create_model('efficientvit_b2.r224_in1k')`, so an importability gate
    let the cell run and the model raised. It must read as a skip naming the
    version, and the registry sweep must drop the name, not claim it.
    """
    import timm

    def unknown(*a, **kw):
        raise RuntimeError("Unknown model (efficientvit_b2)")

    monkeypatch.setattr(timm, "create_model", unknown)
    _TIMM_KNOWS.clear()
    try:
        assert _timm_knows(BACKBONE_EFFICIENTVIT_B2) is False
        assert _buildable_backbones() == [BACKBONE_RESNET50]
        with pytest.raises(pytest.skip.Exception, match="efficientvit"):
            _require_timm(BACKBONE_EFFICIENTVIT_B2)
    finally:
        _TIMM_KNOWS.clear()  # a cached False would skip the real cells below


@needs_timm
@pytest.mark.parametrize("name", [BACKBONE_EFFICIENTVIT_B2, BACKBONE_CONVNEXTV2_NANO,
                                   BACKBONE_CONVNEXTV2_TINY, BACKBONE_CONVNEXTV2_BASE])
def test_a_timm_backbone_reports_the_channels_its_spec_declares(name):
    _require_timm(name)
    spec = backbone_spec(name)
    model = build_backbone(name, pretrained=False)
    model.eval()
    with torch.no_grad():
        rungs, tap = model.forward_with_tap(torch.randn(1, 3, 64, 64))
    assert [tuple(r.shape[1:]) for r in rungs] == [(c, 64 // r, 64 // r)
                                                   for c, r in zip(spec.rung_channels,
                                                                   spec.rung_reductions)]
    if spec.tap_channels is None:
        assert tap is None
    else:
        assert tap.shape[1] == spec.tap_channels
        assert 64 // tap.shape[-2] == spec.tap_reduction == 2


@needs_timm
def test_a_backbone_whose_channels_do_not_match_the_spec_is_refused():
    """Library drift is named, not silently built at the wrong width."""
    import timm

    class Drifted(torch.nn.Module):
        # The registry states (8, 16, 32, 64) for B2; 999 is the drift.
        feature_info = [{"num_chs": 8}, {"num_chs": 16}, {"num_chs": 999}, {"num_chs": 64}]

        def forward(self, x):
            raise AssertionError("a drifted model must never be run")

    real = timm.create_model
    timm.create_model = lambda *a, **k: Drifted()
    try:
        with pytest.raises(ValueError, match="999"):
            build_backbone(BACKBONE_EFFICIENTVIT_B2)
    finally:
        timm.create_model = real


@needs_timm
def test_the_timm_models_channels_are_the_registrys():
    """The registry's stated ladder is what timm actually reports for each name."""
    import timm

    names = [n for n in (BACKBONE_EFFICIENTVIT_B2, BACKBONE_CONVNEXTV2_NANO,
                         BACKBONE_CONVNEXTV2_TINY, BACKBONE_CONVNEXTV2_BASE)
             if _timm_knows(n)]
    assert names, "this host's timm knows neither registry backbone: the check went vacuous"
    for name in names:
        spec = backbone_spec(name)
        model = timm.create_model(spec.timm_name, features_only=True, pretrained=False,
                                  out_indices=spec.out_indices)
        assert tuple(int(fi["num_chs"]) for fi in model.feature_info) == spec.rung_channels


@needs_timm
def test_the_tiny_backbone_is_a_352_segmentation_model_of_about_33m_parameters():
    """The mid-size arm: ConvNeXt V2-Tiny at the deployed 352 input size."""
    _require_timm(BACKBONE_CONVNEXTV2_TINY)
    spec = backbone_spec(BACKBONE_CONVNEXTV2_TINY)
    assert spec.rung_channels == (96, 192, 384, 768), "the published V2-Tiny ladder"
    assert spec.tap_channels is None, "ConvNeXt has no stride-two level"
    model = _model(backbone=BACKBONE_CONVNEXTV2_TINY, mask_stride=2)
    with torch.no_grad():
        assert tuple(model(torch.randn(1, 3, 352, 352)).shape[-2:]) == (176, 176)
    # The backbone alone is the published checkpoint's shape: pinning the exact
    # count names the weights the run starts from, not just a size class.
    assert sum(p.numel() for p in build_backbone(
        BACKBONE_CONVNEXTV2_TINY, pretrained=False).parameters()) == 27_864_960
    total = sum(p.numel() for p in model.parameters())
    # The whole model at neck 128 / mask stride 2 with the 1000-class head is
    # about 31.3M; the band leaves room for the head's class count.
    assert 30_000_000 <= total <= 33_000_000, f"{total:,} parameters"


@needs_timm
def test_the_base_backbone_is_a_352_segmentation_model_of_about_91m_parameters():
    """The large arm: ConvNeXt V2-Base at the deployed 352 input size."""
    _require_timm(BACKBONE_CONVNEXTV2_BASE)
    spec = backbone_spec(BACKBONE_CONVNEXTV2_BASE)
    assert spec.rung_channels == (128, 256, 512, 1024), "the published V2-Base ladder"
    assert spec.tap_channels is None, "ConvNeXt has no stride-two level"
    model = _model(backbone=BACKBONE_CONVNEXTV2_BASE, mask_stride=2)
    with torch.no_grad():
        assert tuple(model(torch.randn(1, 3, 352, 352)).shape[-2:]) == (176, 176)
    # The backbone alone is the published checkpoint's shape: pinning the exact
    # count names the weights the run starts from, not just a size class.
    assert sum(p.numel() for p in build_backbone(
        BACKBONE_CONVNEXTV2_BASE, pretrained=False).parameters()) == 87_690_752
    total = sum(p.numel() for p in model.parameters())
    # The whole model at neck 128 / mask stride 2 with the 1000-class head is
    # about 91.2M; the band leaves room for the head's class count.
    assert 89_000_000 <= total <= 94_000_000, f"{total:,} parameters"


@needs_timm
def test_a_backbone_with_no_stride_two_level_uses_its_first_rung():
    model = _model(backbone=BACKBONE_CONVNEXTV2_NANO, mask_stride=2)
    spec = backbone_spec(BACKBONE_CONVNEXTV2_NANO)
    assert spec.tap_channels is None
    assert model.mask_proj.in_channels == spec.rung_channels[0] == 80
    with torch.no_grad():
        assert tuple(model(torch.randn(1, 3, 64, 64)).shape[-2:]) == (32, 32)


@needs_timm
def test_a_v2_backbone_gets_the_configurable_neck_by_default():
    _require_timm(BACKBONE_EFFICIENTVIT_B2)
    model = _model(backbone=BACKBONE_EFFICIENTVIT_B2, neck_channels=None)
    assert model.neck_channels == NECK_CHANNELS
    rungs = backbone_spec(BACKBONE_EFFICIENTVIT_B2).rung_channels
    for rung, width in zip(("adapt_c2", "adapt_c3", "adapt_c4", "adapt_c5"), rungs):
        assert getattr(model, rung)[0].in_channels == width
        assert getattr(model, rung)[0].out_channels == NECK_CHANNELS


@needs_timm
def test_every_registered_backbone_builds_a_mask_at_either_stride():
    x = torch.randn(1, 3, 64, 64)
    names = _buildable_backbones()
    assert BACKBONE_RESNET50 in names, names  # the sweep must never go vacuous
    for name in names:
        for stride, grid in ((4, 16), (2, 32)):
            model = _model(backbone=name, mask_stride=stride)
            with torch.no_grad():
                assert tuple(model(x).shape) == (1, 2, grid, grid), (name, stride)


# ---------------------------------------------------------------------------
# The architecture record and the weights-derived checks.


def test_the_arch_record_names_the_three_fields():
    assert arch_record_of(_model(neck_channels=128, mask_stride=2)) == {
        "backbone": DEFAULT_BACKBONE, "neck_channels": 128, "mask_stride": 2,
        # The amodal-outline delta adds this one; a model built without the
        # head still records the decision, so a load never has to guess.
        "amodal_outline": False}


def test_a_legacy_record_is_the_pre_change_architecture():
    record = arch_record_of(_model())
    assert record == {"backbone": BACKBONE_RESNET50, "neck_channels": None,
                      "mask_stride": 4, "amodal_outline": False}


def test_the_arch_record_unwraps_a_ddp_model():
    wrapped = torch.nn.DataParallel(_model(neck_channels=128))
    assert arch_record_of(wrapped)["neck_channels"] == 128


def test_the_arch_record_of_a_foreign_object_is_empty():
    assert arch_record_of(object()) == {}


def test_legacy_weights_resolve_to_resnet50_with_no_neck_and_stride_four():
    sd = _model().state_dict()
    assert infer_backbone(sd) == BACKBONE_RESNET50
    assert infer_neck_channels(sd) == NECK_LEGACY
    assert infer_mask_stride(sd) == 4
    assert check_backbone(None, sd) == BACKBONE_RESNET50
    assert check_neck_channels(None, sd) == NECK_LEGACY
    assert check_mask_stride(None, sd) == 4


def test_the_weights_decide_a_neck_width_the_record_omits():
    sd = _model(neck_channels=96).state_dict()
    assert infer_neck_channels(sd) == 96
    assert check_neck_channels(None, sd) == 96


def test_a_recorded_backbone_the_weights_contradict_is_refused():
    sd = _model().state_dict()
    with pytest.raises(ArchMismatchError, match=BACKBONE_EFFICIENTVIT_B2):
        check_backbone(BACKBONE_EFFICIENTVIT_B2, sd)


def test_a_recorded_neck_the_weights_contradict_is_refused():
    sd = _model(neck_channels=128).state_dict()
    with pytest.raises(ArchMismatchError, match="128.*96|96.*128"):
        check_neck_channels(96, sd)
    with pytest.raises(ArchMismatchError, match=NECK_LEGACY):
        check_neck_channels(NECK_LEGACY, sd)


def test_a_recorded_mask_stride_the_weights_contradict_is_refused():
    with pytest.raises(ArchMismatchError, match="4.*2|2.*4"):
        check_mask_stride(2, _model().state_dict())
    with pytest.raises(ArchMismatchError, match="2.*4|4.*2"):
        check_mask_stride(4, _model(mask_stride=2).state_dict())


def test_a_recorded_architecture_the_weights_agree_with_passes():
    sd = _model(neck_channels=128, mask_stride=2).state_dict()
    assert check_backbone(BACKBONE_RESNET50, sd) == BACKBONE_RESNET50
    assert check_neck_channels(128, sd) == 128
    assert check_mask_stride(2, sd) == 2


def test_the_legacy_neck_is_not_spelled_as_a_string():
    with pytest.raises(ArchMismatchError, match="null"):
        check_neck_channels("legacy", _model().state_dict())


@pytest.mark.parametrize("name", [BACKBONE_EFFICIENTVIT_B2, BACKBONE_CONVNEXTV2_NANO,
                                   BACKBONE_CONVNEXTV2_TINY, BACKBONE_CONVNEXTV2_BASE])
def test_a_timm_state_dict_without_a_recorded_name_is_refused(name):
    """A timm backbone's own name is not recoverable from its keys."""
    _require_timm(name)
    model = _model(backbone=name)
    sd = model.state_dict()
    assert bb.REGISTRY  # the module that knows the names is reachable
    with pytest.raises(ArchMismatchError, match="timm"):
        infer_backbone(sd)


def test_a_timm_record_the_resnet_weights_contradict_is_refused():
    with pytest.raises(ArchMismatchError, match=BACKBONE_EFFICIENTVIT_B2):
        check_backbone(BACKBONE_EFFICIENTVIT_B2, _model().state_dict())


def test_validate_arch_returns_the_resolved_neck():
    assert validate_arch(DEFAULT_BACKBONE, None, 4) is None
    assert validate_arch(DEFAULT_BACKBONE, 128, 2) == 128
    assert validate_arch(BACKBONE_EFFICIENTVIT_B2, None, 2) == NECK_CHANNELS


# ---------------------------------------------------------------------------
# The loaders. Every file that builds an hcm_nn is enumerated below; a new one
# fails these tests rather than shipping un-wired.


def _model_build_sites():
    """Every place the repo constructs an hcm model, from the AST.

    Three forms: a direct ``hcm_nn(...)`` call, the downloader's
    ``build_model_from_info(...)`` factory (which calls the saved class), and the
    saved class itself via ``model_class(**model_config)``. The AST, not a text
    scan: prose about ``hcm_nn(neck_channels=...)`` in a docstring is not a call
    site, and a commented-out call is not one either.
    """
    builders = {"hcm_nn", "build_model_from_info", "model_class"}
    sites = []
    for path in sorted(REPO.rglob("*.py")):
        parts = path.parts
        if any(p in parts for p in (".git", ".venv", "node_modules", "build", "jupyter_notebooks")):
            continue
        try:
            tree = ast.parse(path.read_text())
        except (SyntaxError, UnicodeDecodeError):
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            name = func.id if isinstance(func, ast.Name) else getattr(func, "attr", None)
            if name in builders:
                sites.append((path.relative_to(REPO).as_posix(), node.lineno,
                              name, [kw.arg for kw in node.keywords]))
    return sites


# Every file allowed to build the model, and how each states the architecture.
CONSUMERS = {
    "ecs_hcm_model/core/hcm_nn.py": "the class itself (no construction: the class body is not a call)",
    "ecs_hcm_model/core/model_upload_download.py": "the downloader, from the saved config",
    "ecs_hcm_model/core/model_utils.py": "the loader, from the record or the weights",
    "ecs_hcm_model/eval/seg.py": "the scoring tool, from the sidecar or the weights",
    "jetson/demo_model.py": "the demos, from the sidecar or the weights",
}


def test_this_is_the_whole_enumeration_of_model_builders():
    found = {path for path, _, name, _ in _model_build_sites()
             if not path.startswith("tests/") and name != "model_class"}
    assert found == {k for k in CONSUMERS if k != "ecs_hcm_model/core/hcm_nn.py"}, (
        "a new hcm model construction site: wire it for backbone/neck_channels/"
        "mask_stride and add it here (and to test_every_builder_states_the_architecture)")


def test_every_builder_states_the_architecture():
    """No builder may rely on the constructor's defaults: it decides from its artifacts."""
    for path, lineno, name, keywords in _model_build_sites():
        if path.startswith("tests/") or path == "ecs_hcm_model/core/hcm_nn.py":
            continue
        if name == "build_model_from_info":
            assert "model_config" in (REPO / path).read_text(), f"{path}:{lineno}"
            continue
        if name == "model_class" and not keywords:
            # The downloader's "fallback for other models" branch: a generic
            # class, not an hcm one. It is the only such site in the repo, so a
            # new file cannot use it to build an hcm model un-stated.
            assert path == "ecs_hcm_model/core/model_upload_download.py", (path, lineno)
            continue
        assert None in keywords or {"backbone", "neck_channels", "mask_stride"} <= set(keywords), (
            f"{path}:{lineno} builds a model without stating its architecture: {keywords}")


@pytest.mark.parametrize("path,names", [
    ("ecs_hcm_model/core/model_upload_download.py",
     ("check_backbone", "check_neck_channels", "check_mask_stride")),
    ("ecs_hcm_model/core/model_utils.py",
     ("check_backbone", "check_neck_channels", "check_mask_stride")),
    ("ecs_hcm_model/eval/seg.py",
     ("resolve_backbone", "resolve_neck_channels", "resolve_mask_stride")),
    ("jetson/demo_model.py",
     ("resolve_backbone", "resolve_neck_channels", "resolve_mask_stride")),
])
def test_a_loader_resolves_before_it_builds(path, names):
    text = (REPO / path).read_text()
    for name in names:
        assert name in text, f"{path} never asks {name}"


def test_the_downloader_records_and_rebuilds_a_v2_checkpoint():
    """The saved config carries the architecture, and the rebuild uses it."""
    trained = _model(neck_channels=128, mask_stride=2)
    info = {"model_class": hcm_mod.hcm_nn,
            "model_config": {"enable_finetuning": True, "num_segmentation_classes": 2,
                             "out_stride": 16, "decoder_arch": trained.decoder_arch,
                             "backbone": BACKBONE_RESNET50, "neck_channels": 128,
                             "mask_stride": 2}}
    rebuilt = _offline(lambda: build_model_from_info(info, trained.state_dict()))
    assert rebuilt.backbone_name == BACKBONE_RESNET50
    assert rebuilt.neck_channels == 128
    assert rebuilt.mask_stride == 2


def test_the_downloader_rebuilds_a_legacy_checkpoint_with_nothing_recorded():
    """An older checkpoint has no such fields and must mean resnet50/legacy/4."""
    trained = _model()
    info = {"model_class": hcm_mod.hcm_nn,
            "model_config": {"enable_finetuning": True, "num_segmentation_classes": 2,
                             "out_stride": 16, "decoder_arch": trained.decoder_arch}}
    rebuilt = _offline(lambda: build_model_from_info(info, trained.state_dict()))
    assert rebuilt.neck_channels is None
    assert rebuilt.mask_stride == 4
    assert "adapt_c2.weight" in rebuilt.state_dict()


def test_the_downloader_refuses_a_recorded_neck_the_weights_contradict():
    trained = _model(neck_channels=128)
    info = {"model_class": hcm_mod.hcm_nn,
            "model_config": {"enable_finetuning": True, "num_segmentation_classes": 2,
                             "out_stride": 16, "decoder_arch": trained.decoder_arch,
                             "neck_channels": 96}}
    with pytest.raises(ArchMismatchError, match="96"):
        _offline(lambda: build_model_from_info(info, trained.state_dict()))


def test_the_downloader_writes_the_architecture_into_the_saved_config(tmp_path):
    """What the upload persists is what a later loader reads.

    The real ``upload_model`` is run against a client that keeps the uploaded
    files instead of sending them to MinIO, so the assertion is on the bytes
    that would be stored -- metadata.json and model_info.pkl.
    """
    import pickle

    from ecs_hcm_model.core.model_upload_download import ModelUploaderAndDownloader

    kept = {}

    class KeepingClient:
        def fput_object(self, bucket, object_name, path):
            kept[object_name.rsplit("/", 1)[-1]] = Path(path).read_bytes()

    uploader = object.__new__(ModelUploaderAndDownloader)  # no MinIO, no credentials
    uploader.bucket_name = "hcm"
    uploader.client = KeepingClient()

    model = _model(neck_channels=128, mask_stride=2)
    uploader.upload_model(model, "hcm_v2_test", architecture="hcm")

    record = {"backbone": BACKBONE_RESNET50, "neck_channels": 128, "mask_stride": 2}
    meta = json.loads(kept["metadata.json"])
    assert meta["model_class"] == "hcm_nn"
    assert record.items() <= meta["custom_metadata"].items()
    info = pickle.loads(kept["model_info.pkl"])
    assert record.items() <= info["model_config"].items()
    # and the rebuild from exactly those bytes is the model that was uploaded
    rebuilt = _offline(lambda: build_model_from_info(info, model.state_dict()))
    assert (rebuilt.backbone_name, rebuilt.neck_channels, rebuilt.mask_stride) == (
        BACKBONE_RESNET50, 128, 2)


def _uploader_and_bytes(model):
    """Run the real upload against a client that keeps the uploaded bytes.

    Returns the downloader and ``{object name: bytes}``, so the download path
    can be handed exactly what was stored -- a round trip over the real save and
    load code, with no MinIO.
    """
    from ecs_hcm_model.core.model_upload_download import ModelUploaderAndDownloader

    kept = {}

    class KeepingClient:
        def fput_object(self, bucket, object_name, path):
            kept[object_name] = Path(path).read_bytes()

    uploader = object.__new__(ModelUploaderAndDownloader)  # no MinIO, no credentials
    uploader.bucket_name = "hcm"
    uploader.client = KeepingClient()
    uploader.upload_model(model, "hcm_v2_test", architecture="hcm")
    return uploader, kept


def _serve(kept):
    """A client that hands back the stored bytes, for the download path."""
    from ecs_hcm_model.core.model_upload_download import ModelUploaderAndDownloader

    class ServingClient:
        def fget_object(self, bucket, object_name, path):
            Path(path).write_bytes(kept[object_name])

    downloader = object.__new__(ModelUploaderAndDownloader)
    downloader.bucket_name = "hcm"
    downloader.client = ServingClient()
    return downloader


def test_the_architecture_survives_an_upload_and_download():
    """The round trip over the real save and load code, bytes to bytes."""
    model = _model(neck_channels=128, mask_stride=2)
    _, kept = _uploader_and_bytes(model)
    prefix = next(iter(kept)).rsplit("/", 1)[0]
    rebuilt = _offline(lambda: _serve(kept)._download_model_from_prefix(prefix))
    assert (rebuilt.backbone_name, rebuilt.neck_channels, rebuilt.mask_stride) == (
        BACKBONE_RESNET50, 128, 2)
    for key, value in model.state_dict().items():
        assert torch.equal(rebuilt.state_dict()[key], value), key


def test_the_downloader_does_not_swallow_a_contradicted_architecture():
    """A resume must fail loudly: no bare state dict and no None."""
    import pickle

    model = _model(neck_channels=128, mask_stride=2)
    _, kept = _uploader_and_bytes(model)
    names = sorted(kept)
    prefix = names[0].rsplit("/", 1)[0]
    info_name = next(n for n in names if n.endswith("model_info.pkl"))
    info = pickle.loads(kept[info_name])
    info["model_config"]["neck_channels"] = 96  # the weights show 128
    kept[info_name] = pickle.dumps(info)
    with pytest.raises(ArchMismatchError, match="96"):
        _serve(kept)._download_model_from_prefix(prefix)


# ---------------------------------------------------------------------------
# jetson/model_shape.py: the torch-free mirror has to agree with the model.

def test_the_torch_free_mirror_lists_the_same_backbones():
    from jetson import model_shape as ms

    assert tuple(ms.BACKBONES) == backbone_names()
    assert ms.DEFAULT_BACKBONE == DEFAULT_BACKBONE
    assert tuple(ms.MASK_STRIDES) == MASK_STRIDES
    assert ms.MASK_STRIDE_DEFAULT == MASK_STRIDE_DEFAULT
    assert ms.NECK_LEGACY == NECK_LEGACY
    assert "import torch" not in (REPO / "jetson/model_shape.py").read_text()


def test_the_mirror_and_the_model_agree_on_a_legacy_checkpoint():
    """The two layers spell the pre-change widening differently, and mean the same.

    ``hcm_nn.infer_neck_channels`` answers ``NECK_LEGACY`` (a string: in its own
    signature ``None`` already means "resolve from the backbone"), while the
    torch-free mirror answers ``None`` -- the value ``hcm_nn(neck_channels=...)``
    takes for the widening, which is what its callers pass through. Both agree on
    the architecture; only the spelling differs.
    """
    from jetson import model_shape as ms

    ckpt = {"state_dict": _model().state_dict()}
    assert infer_backbone(ckpt["state_dict"]) == ms.resolve_backbone(None, ckpt)[0]
    assert infer_neck_channels(ckpt["state_dict"]) == NECK_LEGACY
    assert ms.resolve_neck_channels(None, ckpt)[0] is None
    assert NECK_LEGACY != None  # noqa: E711 -- the two spellings must stay distinct
    assert ms.resolve_mask_stride(None, ckpt)[0] == 4


def test_the_mirror_and_the_model_agree_on_a_v2_checkpoint():
    from jetson import model_shape as ms

    sd = _model(neck_channels=128, mask_stride=2).state_dict()
    meta = {"custom_metadata": {"backbone": BACKBONE_RESNET50, "neck_channels": 128,
                                "mask_stride": 2}}
    ckpt = {"state_dict": sd}
    # agreement, both spellings: the mirror's None-for-legacy is the model's NECK_LEGACY
    assert ms.resolve_backbone(meta, ckpt) == (BACKBONE_RESNET50,
                                               "metadata.custom_metadata.backbone")
    assert infer_backbone(sd) == ms.resolve_backbone(meta, ckpt)[0]
    assert infer_neck_channels(sd) == ms.resolve_neck_channels(meta, ckpt)[0] == 128
    assert ms.resolve_mask_stride(meta, ckpt)[0] == 2


def test_the_mirror_refuses_a_stated_architecture_the_weights_contradict():
    from jetson.model_shape import ModelShapeError
    from jetson import model_shape as ms

    ckpt = {"state_dict": _model().state_dict()}
    with pytest.raises(ModelShapeError, match="mask_stride"):
        ms.resolve_mask_stride({"mask_stride": 2}, ckpt)
    with pytest.raises(ModelShapeError, match="neck_channels"):
        ms.resolve_neck_channels({"neck_channels": 128}, ckpt)


def test_the_mirror_refuses_an_unknown_backbone_name():
    """A stated name the mirror does not list is refused, naming it and the list.

    The list is the mirror's whole vocabulary: an unlisted name has no known
    ladder or widths, so nothing downstream could build or reshape it.
    """
    from jetson.model_shape import ModelShapeError
    from jetson import model_shape as ms

    # A name no timm tag spells, so adding a family member can never quietly
    # turn this into a known backbone: the earlier fixture here was Base, and
    # registering Base is exactly what broke it.
    unknown = "not_a_backbone.definitely_unregistered"
    ckpt = {"state_dict": _model().state_dict()}
    with pytest.raises(ModelShapeError) as refused:
        ms.resolve_backbone({"backbone": unknown}, ckpt)
    message = str(refused.value)
    assert unknown in message, "the refused value"
    for accepted in ms.BACKBONES:
        assert accepted in message, f"the accepted list, missing {accepted}"


def test_the_mirror_reads_a_wrapped_or_pickled_checkpoint():
    """The three shapes a checkpoint arrives in: state dict, wrapper, torch.save(nn.Module)."""
    from jetson import model_shape as ms

    sd = _model().state_dict()
    assert ms.resolve_mask_stride(None, sd)[0] == 4
    assert ms.resolve_mask_stride(None, {"state_dict": sd})[0] == 4
    assert ms.resolve_mask_stride(None, {"model_state_dict": sd})[0] == 4
    assert ms.resolve_mask_stride(None, _model())[0] == 4  # a whole pickled module
    assert ms.resolve_backbone(None, _model())[0] == BACKBONE_RESNET50


def test_a_silent_mask_stride_is_attributed_to_the_default_not_the_weights():
    """Weights with no mask branch state no stride, so the source says "default".

    Reported as read from the weights, the loader's own fallback becomes a
    stride a caller can honestly believe the weights gave.
    """
    from jetson import model_shape as ms

    # Neither a mask branch nor a head: no evidence either way, on any of the
    # three shapes a checkpoint arrives in.
    assert ms.resolve_mask_stride(None, {"backbone.conv1.weight": object()}) == (
        MASK_STRIDE_DEFAULT, "default")
    assert ms.resolve_mask_stride(None, {"state_dict": {}}) == (
        MASK_STRIDE_DEFAULT, "default")
    # The directions where the weights *do* speak keep saying so.
    assert ms.resolve_mask_stride(None, {"segmentation_head.bias": object()}) == (4, "weights")
    assert ms.resolve_mask_stride(None, {"mask_proj.weight": object()}) == (2, "weights")


def test_the_decoder_resolver_always_names_the_weights_as_the_source():
    """Every state dict shows one decoder or the other: the fuse conv, or not.

    A decoder version is never silent -- unlike the mask stride, which needs the
    refinement's modules to speak -- so the resolver must not carry a branch that
    falls back to another field's default (15e85d0's dead ``from_keys is None``
    arm returned the mask stride's default 4 as a decoder version).
    """
    from jetson import model_shape as ms

    assert ms.resolve_decoder_arch(None, {}) == (hcm_mod.DECODER_V1, "weights")
    assert ms.resolve_decoder_arch(None, {"rrd_list.0.fuse.weight": object()}) == (
        hcm_mod.DECODER_V2, "weights")


def test_the_mirror_tolerates_weights_that_say_nothing_about_the_neck():
    """Silent weights leave the record (or the legacy default) to decide.

    Refusing here would replace the loader's own missing-key report with a
    worse message, and a checkpoint that lost one tensor is not a different
    architecture.
    """
    from jetson import model_shape as ms

    silent = {"backbone.conv1.weight": torch.zeros(1)}
    assert ms.resolve_neck_channels(None, {"state_dict": silent}) == (None, "weights")
    assert ms.resolve_neck_channels({"neck_channels": 128},
                                    {"state_dict": silent}) == (128, "metadata.neck_channels")
    # The positive disagreement still raises.
    from jetson.model_shape import ModelShapeError
    with pytest.raises(ModelShapeError, match="pre-change"):
        ms.resolve_neck_channels({"neck_channels": 128},
                                 {"state_dict": _model().state_dict()})


# ---------------------------------------------------------------------------
# The trainer: the target follows the logits when the model's mask is finer.


def test_the_default_training_grid_is_todays():
    """Stride 4 trains on the masks' grid, exactly as before."""
    assert target_grid_follows_logits(_model()) is False
    assert target_grid_follows_logits(_model(mask_stride=4)) is False


def test_a_stride_two_model_trains_on_its_own_grid():
    assert target_grid_follows_logits(_model(mask_stride=2)) is True
    assert target_grid_follows_logits(object()) is False  # no attribute, the default


def _image_batches(n, batch_size=2, size=64, classes=1000):
    return [(torch.randn(batch_size, 3, size, size), torch.randint(0, classes, (batch_size,)))
            for _ in range(n)]


def test_the_run_record_carries_the_architecture(monkeypatch):
    """The three fields are run *inputs*: a bake-off's arms must be told apart."""
    _require_timm(BACKBONE_CONVNEXTV2_NANO)
    from ecs_hcm_model.core import trainer as T
    from tests.conftest import RecordingLogger

    class _NeverStops:
        def __init__(self, *a, **kw):
            pass

        def step(self, current_loss):
            return False

    monkeypatch.setattr(T, "EarlyStopping", _NeverStops)
    logger = RecordingLogger()
    torch.manual_seed(0)
    model = _offline(lambda: hcm_mod.hcm_nn(enable_finetuning=False,
                                            num_segmentation_classes=None, out_stride=16,
                                            backbone=BACKBONE_CONVNEXTV2_NANO,
                                            neck_channels=192, mask_stride=2,
                                            backbone_pretrained=False))
    T.train_model_for_epochs(
        model, epochs=1, train_loader=_image_batches(2), val_loader=_image_batches(1),
        device="cpu", logger=logger, test_mode=False, warmup=False, task="classification")
    assert logger.params["backbone"] == BACKBONE_CONVNEXTV2_NANO
    assert logger.params["neck_channels"] == 192
    assert logger.params["mask_stride"] == 2


def test_the_run_record_spells_the_legacy_neck_as_none(monkeypatch):
    """As the checkpoint does: "none" is the pre-change 2048 widening, not 2048."""
    from ecs_hcm_model.core import trainer as T
    from tests.conftest import RecordingLogger

    class _NeverStops:
        def __init__(self, *a, **kw):
            pass

        def step(self, current_loss):
            return False

    monkeypatch.setattr(T, "EarlyStopping", _NeverStops)
    logger = RecordingLogger()
    torch.manual_seed(0)
    # The classification head, so the loop's classification path can score this
    # run: what is under test is the record, not the task.
    model = _offline(lambda: hcm_mod.hcm_nn(enable_finetuning=False,
                                            num_segmentation_classes=None, out_stride=16))
    T.train_model_for_epochs(
        model, epochs=1, train_loader=_image_batches(2), val_loader=_image_batches(1),
        device="cpu", logger=logger, test_mode=False, warmup=False, task="classification")
    assert logger.params["backbone"] == BACKBONE_RESNET50
    assert logger.params["neck_channels"] == "none"
    assert logger.params["mask_stride"] == 4


def test_the_default_grid_upsamples_the_logits_to_the_masks():
    logits = torch.randn(1, 2, 8, 8, requires_grad=True)
    masks = torch.randint(0, 2, (1, 32, 32))
    _, iou, _ = calculate_segmentation_loss(logits, masks, "cpu")
    assert 0.0 <= iou <= 1.0


def test_target_follows_logits_brings_the_masks_down_with_nearest():
    seen = {}
    logits = torch.randn(1, 2, 16, 16)
    masks = torch.randint(0, 2, (1, 64, 64))
    _, _, _ = calculate_segmentation_loss(logits, masks, "cpu", target_follows_logits=True,
                                          on_metric_inputs=lambda p, g: seen.update(
                                              pred=tuple(p.shape), gt=tuple(g.shape)))
    assert seen["pred"][-2:] == (16, 16)
    assert seen["gt"][-2:] == (16, 16), "the mask should follow the logits, not the reverse"


def test_a_downsampled_target_keeps_only_label_values():
    """nearest, not bilinear: an interpolated label would be a third class."""
    logits = torch.randn(1, 2, 8, 8)
    masks = torch.randint(0, 2, (1, 32, 32))
    seen = {}
    calculate_segmentation_loss(logits, masks, "cpu", target_follows_logits=True,
                                on_metric_inputs=lambda p, g: seen.update(gt=g))
    assert set(torch.unique(seen["gt"].long()).tolist()) <= {0, 1}


def test_nearest_and_bilinear_disagree_so_the_mode_is_pinned():
    """A discriminating target: the mode choice is observable in the metric."""
    masks = torch.zeros(1, 32, 32, dtype=torch.long)
    masks[0, 0, 0] = 1  # a single foreground pixel: nearest keeps ~1/16 of it,
    # bilinear-then-threshold would keep it under the usual 0.5 cut or smear it.
    logits = torch.full((1, 2, 8, 8), -4.0)
    logits[:, 1] = 4.0  # predict everything foreground
    seen = {}
    calculate_segmentation_loss(logits, masks, "cpu", target_follows_logits=True,
                                on_metric_inputs=lambda p, g: seen.update(gt=g))
    nearest_gt = seen["gt"].long()
    assert nearest_gt.sum().item() == 1
    assert nearest_gt.shape[-2:] == (8, 8)


def test_the_trainer_passes_the_grid_flag_from_the_model():
    """Both call sites read the flag, and the epoch loop's read unwraps.

    The loop holds a ``DistributedDataParallel`` on a multi-rank run, and DDP
    defines no ``__getattr__``, so the loop's read has to unwrap first -- or a
    stride-2 run trains and scores its loss and IoU on the masks' grid while its
    own ``evaluate`` (handed ``unwrap_or_self(model)``) scores the logits' grid.
    Pinned as source text because that is what tells a fix at the loop's call
    site apart from one applied only inside the helper, which this line is the
    only witness to. The validation call site reads a model its own caller has
    already unwrapped, so it needs no change.
    """
    text = (REPO / "ecs_hcm_model/core/trainer.py").read_text()
    loop = text[text.index("def train_model_for_epochs("):]
    # Whitespace-insensitive: the call is line-wrapped at this width.
    assert "target_follows_logits=target_grid_follows_logits(unwrap_or_self(model))" in (
        "".join(loop.split())), (
        "the epoch loop reads the grid flag off the raw model; on a two-rank run "
        "that is the DDP wrapper, whose attribute read falls back to stride 4")
    assert text.count("target_follows_logits=target_grid_follows_logits(") == 2, (
        "both call sites (the loop and evaluate) must still read the flag")


def test_the_grid_flag_reads_through_a_real_ddp_wrapper():
    """Item 2's read is total: a wrapper is unwrapped, not silently defaulted.

    A real ``DistributedDataParallel`` in a one-rank gloo group, not a stand-in:
    the property under test is DDP's own attribute lookup, and a fake that
    delegates ``__getattr__`` to ``module`` would test the fake. The read does
    not depend on the group's size (``tests/test_ddp_training.py`` covers the
    same wrapper with two real ranks).
    """
    dist = pytest.importorskip("torch.distributed")
    if not (dist.is_available() and dist.is_gloo_available()):
        pytest.skip("this torch was built without gloo: no CPU process group here")
    if dist.is_initialized():
        pytest.skip("a process group is already up in this process")

    import socket

    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    dist.init_process_group(
        backend="gloo", init_method=f"tcp://127.0.0.1:{port}", world_size=1, rank=0)
    try:
        wrapped_fine = torch.nn.parallel.DistributedDataParallel(
            _model(mask_stride=2), find_unused_parameters=True)
        wrapped_today = torch.nn.parallel.DistributedDataParallel(
            _model(mask_stride=MASK_STRIDE_DEFAULT), find_unused_parameters=True)

        # The failure mode itself: the wrapper exposes none of the model's own
        # record, so the read the fix replaces would return the stride-4 default.
        assert not hasattr(wrapped_fine, "mask_stride")

        assert target_grid_follows_logits(wrapped_fine) is True
        # The control: the same wrapper around a stride-4 model is still False,
        # so the assertion above cannot be met by a read that returns a constant.
        assert target_grid_follows_logits(wrapped_today) is False
    finally:
        dist.destroy_process_group()


def test_a_neck_width_that_does_not_divide_32_still_builds_and_round_trips():
    """The neck's GroupNorm clamps its group count instead of assuming 32 divides.

    ``neck_channels`` is a free parameter, so 33 -- which 32 groups cannot divide
    -- is the width that clamp exists for: ``_neck_norm`` walks down to 11. A
    forward is the assertion that matters (GroupNorm raises when its group count
    does not divide its channels), and the group count is part of the parameter's
    shape, so the strict reload cannot paper over a disagreement about it.
    """
    model = _model(neck_channels=33)
    for rung in ("adapt_c2", "adapt_c3", "adapt_c4", "adapt_c5"):
        norm = getattr(model, rung)[1]
        assert (norm.num_channels, norm.num_groups) == (33, 11)

    x = torch.randn(1, 3, 64, 64)
    with torch.no_grad():
        assert tuple(model(x).shape[-2:]) == (16, 16)

    reloaded = _model(neck_channels=33)
    reloaded.load_state_dict(model.state_dict())  # strict: same key set
    assert reloaded.adapt_c2[1].num_groups == 11


def test_the_trainer_takes_the_three_fields_and_forwards_them_to_the_loader():
    text = (REPO / "ecs_hcm_model/core/trainer.py").read_text()
    signature = text[text.index("def train_hcm_imagenet1k("):]
    signature = signature[:signature.index("):")]
    for name, declared in (("backbone", "str = DEFAULT_BACKBONE"),
                           ("neck_channels", "int | None = None"),
                           ("mask_stride", "int = MASK_STRIDE_DEFAULT")):
        assert f"{name}: {declared}" in signature, (
            f"train_hcm_imagenet1k does not take {name}: {declared}")
    call = text[text.index("model = load_hcm_model("):]
    call = call[:call.index("\n        )")]
    assert "backbone=backbone" in call
    assert "neck_channels=neck_channels" in call
    assert "mask_stride=mask_stride" in call


def test_the_trainer_refuses_a_bad_architecture_before_any_run_starts():
    """A mistyped backbone costs a stack trace, not an MLflow run and a GPU preflight."""
    from ecs_hcm_model.core.trainer import train_hcm_imagenet1k

    with pytest.raises(ValueError, match="unknown backbone"):
        train_hcm_imagenet1k(backbone="efficientvit-typo")
    with pytest.raises(ValueError, match="mask_stride"):
        train_hcm_imagenet1k(mask_stride=3)
    with pytest.raises(ValueError, match="neck_channels"):
        train_hcm_imagenet1k(neck_channels=-8)


def test_a_new_model_starts_from_the_imagenet_backbone():
    """Item 5: with no HCM init the run builds a fresh model on ImageNet weights."""
    from torchvision import models as tv_models
    from torchvision.models import ResNet50_Weights

    from ecs_hcm_model.core.model_utils import load_hcm_model

    seen = {}
    real = tv_models.resnet50

    def recording(weights=None, replace_stride_with_dilation=None):
        seen["weights"] = weights
        return real(weights=None, replace_stride_with_dilation=replace_stride_with_dilation)

    tv_models.resnet50 = recording
    try:
        model = load_hcm_model(new_model=True, finetune=True, num_segmentation_classes=2,
                               out_stride=16, decoder_arch=DECODER_V2,
                               backbone=BACKBONE_RESNET50, neck_channels=128, mask_stride=2)
    finally:
        tv_models.resnet50 = real
    assert seen.get("weights") is ResNet50_Weights.DEFAULT, (
        "a fresh run initialises from the pretrained ImageNet backbone")
    assert (model.backbone_name, model.neck_channels, model.mask_stride) == (
        BACKBONE_RESNET50, 128, 2)


def test_loading_a_checkpoint_never_fetches_pretrained_weights(tmp_path):
    """The demos and the scorer load weights, they do not initialise a model."""
    from torchvision import models as tv_models

    from ecs_hcm_model.eval.seg import load_checkpoint

    trained = _model()
    path = tmp_path / "model.pth"
    torch.save({"state_dict": trained.state_dict()}, path)

    seen = {}
    real = tv_models.resnet50

    def recording(weights=None, replace_stride_with_dilation=None):
        seen["weights"] = weights
        return real(weights=None, replace_stride_with_dilation=replace_stride_with_dilation)

    lines = []
    tv_models.resnet50 = recording
    try:
        model = load_checkpoint(path, "cpu", log=lines.append)
    finally:
        tv_models.resnet50 = real
    assert seen == {"weights": None}, "a load must not reach the network"
    assert model.mask_stride == 4 and model.neck_channels is None
    assert any("mask_stride 4 from weights" in ln for ln in lines), lines


def test_the_metrics_are_computed_on_the_grid_the_loss_uses():
    """The returned IoU/Dice are the ones for the tensors ``on_metric_inputs`` sees."""
    logits = torch.zeros(1, 2, 8, 8)
    logits[:, 1, :2, :2] = 4.0   # foreground in the top-left 2x2 of the 8x8 grid
    masks = torch.zeros(1, 32, 32, dtype=torch.long)
    masks[:, :8, :8] = 1         # whose nearest 4x downsample is exactly that 2x2
    seen = {}
    _, iou, dice = calculate_segmentation_loss(
        logits, masks, "cpu", target_follows_logits=True,
        on_metric_inputs=lambda p, g: seen.update(pred=p, gt=g))
    manual_iou, manual_dice = iou_dice_fg_from_logits(seen["pred"], seen["gt"])
    assert iou == pytest.approx(manual_iou)
    assert dice == pytest.approx(manual_dice)
    assert manual_iou == pytest.approx(1.0, abs=1e-6), "prediction and target agree here"


def test_the_old_grid_still_upsamples_the_logits():
    """The default is untouched: the same model scored on the masks' grid differs."""
    logits = torch.zeros(1, 2, 8, 8)
    logits[:, 1, :2, :2] = 4.0
    masks = torch.zeros(1, 32, 32, dtype=torch.long)
    masks[:, :8, :8] = 1
    seen = {}
    _, iou, _ = calculate_segmentation_loss(
        logits, masks, "cpu", on_metric_inputs=lambda p, g: seen.update(pred=p, gt=g))
    assert seen["pred"].shape[-2:] == (32, 32), "the logits follow the masks by default"
    assert seen["gt"].shape[-2:] == (32, 32)
    assert iou < 1.0, "a 4x bilinear upsample is not the masks' own grid"


# ---------------------------------------------------------------------------
# The image and the docs.


def test_the_trainer_image_pins_timm_exactly():
    text = (REPO / "dockerfiles/ecs_hcm_trainer/Dockerfile").read_text()
    assert "timm==0.9.16" in text
    # ... and the image is not where weights come from: nothing is fetched at
    # build time and no weights file is copied in. (curl/wget are installed as
    # tools; what must not appear is a download or a COPY of a checkpoint.)
    forbidden = ("snapshot_download", "hf_hub_download", "from_pretrained",
                 "torch.hub", ".pth", ".onnx", ".engine", ".safetensors")
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            continue
        for needle in forbidden:
            assert needle not in stripped, f"the image touches weights: {stripped}"


def test_the_trainer_image_lists_timm_once_and_pinned():
    """Every non-comment mention of timm is the pin, so nothing floats."""
    text = (REPO / "dockerfiles/ecs_hcm_trainer/Dockerfile").read_text()
    mentions = [ln.strip() for ln in text.splitlines()
                if not ln.strip().startswith("#") and "timm" in ln]
    assert mentions, "timm is not installed by the image"
    assert any("timm==0.9.16" in m for m in mentions)
    for mention in mentions:
        # the only unpinned allowance would be a transitive name like
        # ``huggingface-hub``; a bare ``timm`` or a range would be a floating pin
        assert "timm==" in mention or "huggingface" in mention, mention


def test_the_spec_declares_the_three_fields():
    """The live decoder spec declares the three v2 architecture fields.

    This read the HCM v2 change delta while that change was open. Archive moved
    the delta into the live spec, and CI ships openspec/ without changes/archive,
    so the live spec is the only home both share.
    """
    spec = (REPO / "openspec/specs/model-decoder/spec.md").read_text()
    for field in ("backbone", "neck_channels", "mask_stride"):
        assert field in spec, f"the decoder spec never mentions {field}"
