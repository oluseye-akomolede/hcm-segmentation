"""The backbones the HCM decoder can be built on, as one registry.

The decoder consumes four feature rungs in a fixed order and, at the finest
mask stride, one half-resolution tap. It never needs to know which family
produced them. This module is that boundary: a :class:`BackboneSpec` states the
rungs, the tap and the licence, and :func:`build_backbone` returns a module
whose ``forward_with_tap`` yields ``(rungs, tap)``.

``resnet50`` is the pre-change backbone, moved here unchanged: its parameter
names, its forward output and its ``out_stride`` behaviour are what they were,
so a model built through this registry with no other setting is byte-for-byte
the model that existed before. The timm entries present the fixed 4/8/16/32
ladder; ``out_stride`` is a property of the resnet entry and is not applied to
them.

Nothing here touches the network at import time, and
``build_backbone(..., pretrained=False)`` downloads nothing.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import torch
import torch.nn as nn
from torchvision import models as tv_models
from torchvision.models import ResNet50_Weights

BACKBONE_RESNET50 = "resnet50"
BACKBONE_EFFICIENTVIT_B2 = "efficientvit_b2.r224_in1k"
BACKBONE_CONVNEXTV2_NANO = "convnextv2_nano.fcmae_ft_in1k"
# The 22k-tag weights, not the 1k ones: the same FCMAE recipe pre-trained on
# ImageNet-22k and fine-tuned on 1k. timm 0.9.16 (the trainer image's pin)
# resolves this tag -- checked in the image, not assumed: create_model accepts
# convnextv2_tiny.fcmae_ft_in22k_in1k and its _384 variant.
BACKBONE_CONVNEXTV2_TINY = "convnextv2_tiny.fcmae_ft_in22k_in1k"
# The same 22k FCMAE tag one size up. Resolved, not assumed: timm 0.9.16 (the
# trainer image's pin) accepts it, and so does the Jetson venv's 0.9.0.
BACKBONE_CONVNEXTV2_BASE = "convnextv2_base.fcmae_ft_in22k_in1k"

DEFAULT_BACKBONE = BACKBONE_RESNET50


@dataclass(frozen=True)
class BackboneSpec:
    """What the decoder needs to know about a backbone, frozen.

    ``rung_channels`` and ``rung_reductions`` are the native ladder. The decoder
    resizes every rung to the first rung's grid, so a backbone whose ladder runs
    at different strides is still consumed correctly; these values are what the
    build checks the backbone against, not what the decoder assumes.

    ``tap_channels`` is the width of the half-resolution feature the finest mask
    stride concatenates, or ``None`` when the backbone's stem downsamples by 4
    and therefore offers no half-resolution signal at all.
    """

    name: str
    kind: str  # "resnet" | "timm"
    rung_channels: Tuple[int, int, int, int]
    rung_reductions: Tuple[int, int, int, int]
    tap_channels: Optional[int]
    tap_reduction: Optional[int] = None
    timm_name: Optional[str] = None
    out_indices: Tuple[int, ...] = (0, 1, 2, 3)
    license_note: str = ""


REGISTRY: dict[str, BackboneSpec] = {
    BACKBONE_RESNET50: BackboneSpec(
        name=BACKBONE_RESNET50,
        kind="resnet",
        rung_channels=(256, 512, 1024, 2048),
        rung_reductions=(4, 8, 16, 32),
        tap_channels=64,  # the post-ReLU stem, before the maxpool
        tap_reduction=2,
        license_note="BSD-3-Clause (torchvision)",
    ),
    BACKBONE_EFFICIENTVIT_B2: BackboneSpec(
        name=BACKBONE_EFFICIENTVIT_B2,
        kind="timm",
        rung_channels=(48, 96, 192, 384),
        rung_reductions=(4, 8, 16, 32),
        tap_channels=24,  # the stem's output, the input to the first stage
        tap_reduction=2,
        timm_name=BACKBONE_EFFICIENTVIT_B2,
        license_note="Apache-2.0",
    ),
    BACKBONE_CONVNEXTV2_NANO: BackboneSpec(
        name=BACKBONE_CONVNEXTV2_NANO,
        kind="timm",
        rung_channels=(80, 160, 320, 640),
        rung_reductions=(4, 8, 16, 32),
        tap_channels=None,  # the stem is a 4x downsample: no half-res signal
        tap_reduction=None,
        timm_name=BACKBONE_CONVNEXTV2_NANO,
        license_note="CC-BY-NC-4.0 (non-commercial: check before using in a product)",
    ),
    BACKBONE_CONVNEXTV2_TINY: BackboneSpec(
        name=BACKBONE_CONVNEXTV2_TINY,
        kind="timm",
        # The nano entry's ladder one size up: same four stages and strides,
        # wider. Same stem handling, so the same no-half-res-signal consequence.
        rung_channels=(96, 192, 384, 768),
        rung_reductions=(4, 8, 16, 32),
        tap_channels=None,
        tap_reduction=None,
        timm_name=BACKBONE_CONVNEXTV2_TINY,
        license_note="CC-BY-NC-4.0 (non-commercial: check before using in a product)",
    ),
    BACKBONE_CONVNEXTV2_BASE: BackboneSpec(
        name=BACKBONE_CONVNEXTV2_BASE,
        kind="timm",
        # The tiny entry's ladder one size up again: same four stages and
        # strides, wider. Same stem handling, so the same no-half-res-signal
        # consequence.
        rung_channels=(128, 256, 512, 1024),
        rung_reductions=(4, 8, 16, 32),
        tap_channels=None,
        tap_reduction=None,
        timm_name=BACKBONE_CONVNEXTV2_BASE,
        license_note="CC-BY-NC-4.0 (non-commercial: check before using in a product)",
    ),
}


def backbone_names() -> Tuple[str, ...]:
    """Every registered name, in registration order."""
    return tuple(REGISTRY)


def backbone_spec(name: str) -> BackboneSpec:
    """The spec for ``name``; an unregistered name raises, naming both sides."""
    try:
        return REGISTRY[name]
    except KeyError:
        raise ValueError(
            f"unknown backbone {name!r}; registered backbones are {backbone_names()}"
        ) from None


class ResNet50Backbone(nn.Module):
    """The pre-change backbone. Moved here unchanged; see the module docstring."""

    def __init__(self, out_stride=16, pretrained=True):
        super().__init__()
        assert out_stride in (8, 16, 32)

        # replace_stride_with_dilation corresponds to layer2, layer3, layer4
        # default OS=32 (no dilation)
        if out_stride == 32:
            dil = (False, False, False)
        elif out_stride == 16:
            dil = (False, False, True)   # keep layer4 resolution, use dilation there
        else:  # out_stride == 8
            dil = (False, True, True)    # keep layer3+layer4 resolution

        # Called through the torchvision.models module rather than a from-import
        # of resnet50, so a test that stubs torchvision.models.resnet50 to build
        # offline (test_decoder_arch.py, test_fp16_overflow.py) still intercepts
        # this build. A from-import would bind the real constructor at import
        # time and make those stubs silently ineffective, i.e. the suite would
        # quietly depend on a warm hub cache or the network.
        m = tv_models.resnet50(
            weights=ResNet50_Weights.DEFAULT if pretrained else None,
            replace_stride_with_dilation=dil
        )

        # "stem"
        self.conv1 = m.conv1
        self.bn1 = m.bn1
        self.relu = m.relu
        self.maxpool = m.maxpool

        # stages
        self.layer1 = m.layer1  # C2
        self.layer2 = m.layer2  # C3
        self.layer3 = m.layer3  # C4
        self.layer4 = m.layer4  # C5

    def forward_with_tap(self, x):
        """``(rungs, tap)``: the four stages and the pre-maxpool stem output."""
        stem = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(stem)
        c2 = self.layer1(x)   # ~1/4
        c3 = self.layer2(c2)  # ~1/8
        c4 = self.layer3(c3)  # ~1/16 (or ~1/8 if OS=8)
        c5 = self.layer4(c4)  # ~1/32 (or ~1/16/~1/8 if dilated)
        return (c2, c3, c4, c5), stem

    def forward(self, x):
        c2, c3, c4, c5 = self.forward_with_tap(x)[0]
        return c2, c3, c4, c5


def _feature_module_path(model):
    """The submodule timm says produces the first rung."""
    try:
        return model.feature_info[0]["module"]
    except (AttributeError, IndexError, KeyError, TypeError) as exc:
        raise ValueError(
            "the model reports no first-feature submodule, so there is nowhere "
            "to take the half-resolution tap from"
        ) from exc


class _TimmBackbone(nn.Module):
    """A timm backbone presented as four rungs plus an optional half-res tap.

    The tap is the *input* to the first rung's module, captured by a forward
    pre-hook: for a backbone whose stem is a 2x downsample that is the stem's
    output, and the hook needs no per-model knowledge of attribute names. Its
    width is checked against the spec on the first forward, so a drifted library
    stops the run instead of silently feeding the head a different shape.
    """

    def __init__(self, spec: BackboneSpec, *, pretrained: bool, weights_path: Optional[str]):
        super().__init__()
        try:
            # Imported here so this module imports, and the resnet50 model
            # builds, without timm installed.
            import timm
        except ImportError as exc:
            raise ImportError(
                f"{spec.name} needs timm, which the trainer image pins at "
                f"timm==0.9.16 (the package every other hcm model builds "
                f"without); install it to build this backbone"
            ) from exc

        self.spec = spec
        model = timm.create_model(
            spec.timm_name,
            features_only=True,
            pretrained=pretrained,
            out_indices=spec.out_indices,
        )
        reported = tuple(int(fi["num_chs"]) for fi in model.feature_info)
        if reported != spec.rung_channels:
            raise ValueError(
                f"{spec.name}: timm reports rung channels {reported} but the "
                f"registry states {spec.rung_channels}; the library and the "
                f"registry have drifted"
            )
        if weights_path is not None:
            if pretrained:
                raise ValueError(
                    "pass either pretrained=True (the environment's cache) or "
                    "weights_path (a mirrored file), not both"
                )
            timm.models.load_checkpoint(model, weights_path, strict=True)
        self.model = model
        self._tap = None
        if spec.tap_channels is not None:
            # feature_info reports the path as it is in the model timm built;
            # features_only=True wraps that model, and the wrapper flattens a
            # nested name ("stages.0" -> "stages_0"), so resolve both spellings.
            path = _feature_module_path(model)
            head, _, tail = path.rpartition(".")
            candidates = [path] + ([f"{head}_{tail}"] if head else [])
            for holder in (model, getattr(model, "body", None)):
                if holder is None:
                    continue
                for candidate in candidates:
                    try:
                        first = holder.get_submodule(candidate)
                    except AttributeError:
                        continue
                    first.register_forward_pre_hook(self._capture_tap)
                    return
            raise ValueError(
                f"{spec.name}: no first-rung submodule ({path!r}) to take the "
                f"half-resolution tap from; the library's module layout has changed"
            )

    def _capture_tap(self, module, args):
        # Must return None: a pre-hook's return value *replaces* the module's
        # arguments, and a returned tuple is splatted across them.
        if not args:
            raise RuntimeError(
                f"{self.spec.name}: the first rung was called without a "
                f"positional input, so there is no half-resolution tap"
            )
        self._tap = args[0]

    def forward_with_tap(self, x):
        self._tap = None
        rungs = tuple(self.model(x))
        tap = self._tap
        if self.spec.tap_channels is not None:
            if tap is None:
                raise RuntimeError(
                    f"{self.spec.name}: no half-resolution tap was captured at "
                    f"the first rung; the library's module layout has changed"
                )
            if int(tap.shape[1]) != self.spec.tap_channels:
                raise RuntimeError(
                    f"{self.spec.name}: the tap has {int(tap.shape[1])} channels "
                    f"but the registry states {self.spec.tap_channels}"
                )
        return rungs, tap

    def forward(self, x):
        return self.forward_with_tap(x)[0]


def build_backbone(
    name: str = DEFAULT_BACKBONE,
    *,
    out_stride: int = 16,
    pretrained: bool = False,
    weights_path: Optional[str] = None,
) -> nn.Module:
    """Build ``name`` with pretrained ImageNet weights disabled by default.

    ``pretrained=True`` fetches through the library's own cache, which honours
    ``HF_HOME``/``TORCH_HOME``; in-cluster that cache lives on the PVC, and
    ``weights_path`` loads a mirrored file instead. Neither reads a credential
    from code, and neither runs at import time.
    """
    spec = backbone_spec(name)
    if spec.kind == "resnet":
        if weights_path is not None:
            raise ValueError(
                f"{name}: weights_path is only used by the timm backbones; the "
                f"resnet entry takes pretrained=True"
            )
        return ResNet50Backbone(out_stride=out_stride, pretrained=pretrained)
    if out_stride != 16:
        raise ValueError(
            f"{name} presents a fixed 4/8/16/32 ladder, so out_stride only means "
            f"something for the resnet entry; got {out_stride!r}"
        )
    return _TimmBackbone(spec, pretrained=pretrained, weights_path=weights_path)
