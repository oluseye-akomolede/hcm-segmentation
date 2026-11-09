"""The architecture a demo builds before loading the weights into it.

The demos hardcoded ``out_stride=16`` and accepted whatever class count the
checkpoint's segmentation head had. The overlay only makes sense for a binary
(background / foreground) head, and a model trained at another stride would
either fail to load or, worse, load and draw at the wrong scale.

Like ``input_size.py``: no ``cv2``, no torch, no work at import, so the rules
are testable on a machine without a camera or a GPU.
"""

from collections.abc import Mapping

from jetson.input_size import _as_positive_int

DEFAULT_OUT_STRIDE = 16
ALLOWED_OUT_STRIDES = (8, 16, 32)
DEMO_NUM_CLASSES = 2

OUT_STRIDE_ENV_VAR = "ECS_HCM_DEMO_OUT_STRIDE"
SOURCE_OUT_STRIDE_CLI = "--out-stride"
SOURCE_OUT_STRIDE_ENV = f"env:{OUT_STRIDE_ENV_VAR}"


class ModelShapeError(ValueError):
    """The checkpoint is not one the demos can draw, or states an unusable shape."""


def _checkpoint_out_stride(ckpt):
    if not isinstance(ckpt, Mapping):
        return None
    if "out_stride" in ckpt:
        return ckpt["out_stride"]
    config = ckpt.get("model_config")
    if isinstance(config, Mapping):
        return config.get("out_stride")
    return None


def resolve_out_stride(meta, ckpt):
    """Return ``(out_stride, source)`` for the model the demo is about to build.

    The order: ``custom_metadata.out_stride``, a top-level ``out_stride`` in the
    sidecar, the stride the checkpoint itself recorded (``out_stride`` or
    ``model_config.out_stride`` in a wrapped checkpoint), then 16. A stated
    stride that is not one the backbone supports raises.
    """
    candidates = []
    if isinstance(meta, Mapping):
        custom = meta.get("custom_metadata")
        if isinstance(custom, Mapping):
            candidates.append(("metadata.custom_metadata.out_stride", custom.get("out_stride")))
        candidates.append(("metadata.out_stride", meta.get("out_stride")))
    candidates.append(("checkpoint", _checkpoint_out_stride(ckpt)))

    for source, value in candidates:
        stride = _usable_stride(value, source)
        if stride is not None:
            return stride, source
    return DEFAULT_OUT_STRIDE, "default"


def _usable_stride(value, source):
    """``value`` as a supported stride, ``None`` when unstated; raise when unusable."""
    if value is None:
        return None
    try:
        stride = _as_positive_int(value)
    except ValueError as exc:
        raise ModelShapeError(f"out_stride {value!r} from {source} is not usable") from exc
    if stride is not None and stride not in ALLOWED_OUT_STRIDES:
        raise ModelShapeError(
            f"out_stride {value!r} from {source} is not one of {ALLOWED_OUT_STRIDES}"
        )
    return stride


def resolve_demo_out_stride(cli_stride, env, meta, ckpt):
    """Return ``(out_stride, source, warning)`` for a demo, overrides included.

    The order, the same as the input size's: ``--out-stride`` (``cli_stride``,
    ``None`` when not given), then ``ECS_HCM_DEMO_OUT_STRIDE``, then the stride
    the model recorded (:func:`resolve_out_stride`: the sidecar, then the
    checkpoint), then 16. An empty environment value counts as unset.

    An override that disagrees with a recorded stride is honoured, with a
    warning rather than an error, as the loader does: the operator may know
    better than an old artifact, but the weights may then fail to load. An
    override also masks a recorded stride that is unusable, with a warning,
    since an override exists for exactly that kind of broken artifact.
    """
    override = None
    if cli_stride is not None:
        stride = _usable_stride(cli_stride, SOURCE_OUT_STRIDE_CLI)
        if stride is None:
            raise ModelShapeError(f"out_stride {cli_stride!r} from --out-stride is not usable")
        override = (stride, SOURCE_OUT_STRIDE_CLI)
    elif OUT_STRIDE_ENV_VAR in env:
        stride = _usable_stride(env[OUT_STRIDE_ENV_VAR], SOURCE_OUT_STRIDE_ENV)
        if stride is not None:
            override = (stride, SOURCE_OUT_STRIDE_ENV)

    if override is None:
        stride, source = resolve_out_stride(meta, ckpt)
        return stride, source, None

    stride, source = override
    try:
        recorded, recorded_source = resolve_out_stride(meta, ckpt)
    except ModelShapeError as exc:
        return stride, source, f"out_stride {stride} from {source} overrides an unusable recorded stride ({exc})"
    if recorded_source != "default" and recorded != stride:
        return stride, source, (
            f"out_stride {stride} from {source} overrides {recorded} recorded in "
            f"{recorded_source}; the weights may not load at this stride"
        )
    return stride, source, None


# The decoder architecture version (see hcm_nn.DECODER_*), spelled out here so
# this module stays torch-free.
DECODER_V1 = "v1"
DECODER_V2 = "v2"
DECODER_ARCHES = (DECODER_V1, DECODER_V2)


def _state_dict_decoder_arch(state_dict):
    """The version the weights were saved from: only v2 has a fuse conv."""
    for key in state_dict:
        if isinstance(key, str) and key.startswith("rrd_list.") and ".fuse." in key:
            return DECODER_V2
    return DECODER_V1


def resolve_decoder_arch(meta, state_dict):
    """Return ``(decoder_arch, source)`` for the decoder the demo is about to build.

    The sidecar's ``custom_metadata.decoder_arch`` or top-level ``decoder_arch``
    when stated; otherwise the weights' own keys, which makes a checkpoint from
    before the field existed "v1". A stated version that is unknown, or that
    the weights contradict, raises: the weights only load into one decoder.
    """
    from_keys = _state_dict_decoder_arch(state_dict)
    stated = []
    if isinstance(meta, Mapping):
        custom = meta.get("custom_metadata")
        if isinstance(custom, Mapping):
            stated.append(("metadata.custom_metadata.decoder_arch", custom.get("decoder_arch")))
        stated.append(("metadata.decoder_arch", meta.get("decoder_arch")))
    # Every stated value is checked, not just the first: a top-level value
    # that disagrees with an agreeing custom_metadata one still raises.
    found = None
    for source, value in stated:
        if value is None:
            continue
        if value not in DECODER_ARCHES:
            raise ModelShapeError(f"{source}={value!r} is not one of {DECODER_ARCHES}")
        if value != from_keys:
            raise ModelShapeError(
                f"{source}={value!r} but the weights are a {from_keys!r} decoder")
        if found is None:
            found = (value, source)
    if found is not None:
        return found
    # ``from_keys`` is v1 or v2, never None: the only thing that distinguishes
    # the two decoders is the fuse conv, so a state dict either shows it or is
    # the pre-change v1.
    return from_keys, "weights"


def require_two_classes(num_classes):
    """The overlay draws one foreground class; refuse any other head at load."""
    if num_classes != DEMO_NUM_CLASSES:
        raise ModelShapeError(
            f"the checkpoint's segmentation head has {num_classes} classes; the "
            f"demos draw a binary (background/foreground) head of {DEMO_NUM_CLASSES}"
        )


# The ladder and the mask branch, spelled out here so this module stays
# torch-free (see hcm_nn.DEFAULT_BACKBONE / backbones.REGISTRY / MASK_STRIDES).
DEFAULT_BACKBONE = "resnet50"
BACKBONES = ("resnet50", "efficientvit_b2.r224_in1k",
             "convnextv2_nano.fcmae_ft_in1k",
             "convnextv2_tiny.fcmae_ft_in22k_in1k",
             "convnextv2_base.fcmae_ft_in22k_in1k")
# The pre-change 2048 widening, which has no width to record: the model spells
# it neck_channels=None.
NECK_LEGACY = "legacy"
MASK_STRIDE_DEFAULT = 4
MASK_STRIDES = (2, 4)


def _weights_of(ckpt):
    """The state dict inside a wrapped checkpoint, or ``ckpt`` itself.

    Handles the three shapes a checkpoint arrives in: a plain state dict, a
    ``{"state_dict": ...}`` wrapper, and a whole pickled ``nn.Module`` (whose
    ``state_dict()`` is called, since iterating the module itself raises).
    """
    if isinstance(ckpt, Mapping):
        for key in ("state_dict", "model_state_dict", "model"):
            inner = ckpt.get(key)
            if isinstance(inner, Mapping):
                return inner
        return ckpt
    as_dict = getattr(ckpt, "state_dict", None)
    if callable(as_dict):
        try:
            return as_dict()
        except TypeError:
            pass
    return ckpt


def _state_dict_backbone(state_dict):
    """The family the weights came from, or None when they show no backbone.

    Only the family is recoverable: a timm backbone's own name is not in the
    keys, which is why :func:`resolve_backbone` refuses to invent one. Read from
    the ``backbone.`` prefix as a whole, not from one key: a checkpoint that lost
    ``backbone.conv1.weight`` still has the rest of its resnet, and refusing it
    here would replace the loader's own missing-key report with a worse message.
    """
    family = None
    for key in state_dict:
        if not isinstance(key, str) or not key.startswith("backbone."):
            continue
        if key.startswith("backbone.model."):
            return "timm"
        family = DEFAULT_BACKBONE
    return family


def _state_dict_neck(state_dict):
    """``(kind, channels)`` the weights show: ``None`` when they are silent.

    ``("legacy", None)`` is the pre-change widening, ``("neck", width)`` a
    projected neck, and ``(None, None)`` no evidence either way -- no
    ``adapt_c2`` block, or one whose tensors carry no readable shape.
    """
    for key, value in state_dict.items():
        if isinstance(key, str) and key.startswith("adapt_c2.0."):
            shape = getattr(value, "shape", None)
            if shape:
                return "neck", int(shape[0])
    for key in state_dict:
        if isinstance(key, str) and key == "adapt_c2.weight":
            return "legacy", None
    return None, None


def _state_dict_mask_stride(state_dict):
    """The mask stride the weights show: 2, 4, or None when they are silent.

    Any of the refinement's three modules means stride 2. A complete model with
    a segmentation head and none of them is the stride-4 model; anything else
    is no evidence, and the record or the default decides.
    """
    marks = ("mask_conv1.", "mask_conv2.", "mask_proj.")
    seen_head = False
    for key in state_dict:
        if not isinstance(key, str):
            continue
        if key.startswith(marks):
            return 2
        if key.startswith("segmentation_head."):
            seen_head = True
    return MASK_STRIDE_DEFAULT if seen_head else None


def _state_dict_amodal_outline(state_dict):
    """Whether the weights carry an outline head: True/False, or None if silent.

    ``outline_head.weight`` means True. A segmentation head without one is a
    complete model that has none, so False; showing neither head says nothing
    either way, and the record or the default decides.
    """
    seen_head = False
    for key in state_dict:
        if not isinstance(key, str):
            continue
        if key.startswith("outline_head."):
            return True
        if key.startswith("segmentation_head."):
            seen_head = True
    return False if seen_head else None


def _stated(meta, field):
    """Every value the artifacts state for ``field``, as ``(source, value)`` pairs.

    The sidecar first (``custom_metadata`` then top level), as for the decoder
    version, so a top-level value that disagrees with an agreeing
    ``custom_metadata`` one still raises. Any artifact that states nothing
    leaves the weights to decide.
    """
    stated = []
    if isinstance(meta, Mapping):
        custom = meta.get("custom_metadata")
        if isinstance(custom, Mapping):
            stated.append((f"metadata.custom_metadata.{field}", custom.get(field)))
        stated.append((f"metadata.{field}", meta.get(field)))
    return [(source, value) for source, value in stated if value is not None]


def resolve_backbone(meta, ckpt):
    """Return ``(backbone, source)`` for the model about to be built.

    A stated name is checked against the weights' family before it is used: the
    two are the only pair that can load. With nothing stated the weights decide
    -- a resnet-shaped state dict is ``resnet50``, which is what every
    checkpoint from before the field existed is -- and a timm-shaped one is
    refused, because its own name is not recoverable from the keys and guessing
    one would build a model the weights do not fit. Only a *positive*
    disagreement refuses: weights showing no backbone at all leave a stated
    name to decide (and the load reports what is missing).
    """
    state_dict = _weights_of(ckpt)
    from_keys = _state_dict_backbone(state_dict)
    found = None
    for source, value in _stated(meta, "backbone"):
        if value not in BACKBONES:
            raise ModelShapeError(f"{source}={value!r} is not one of {BACKBONES}")
        if from_keys == "timm" and value == DEFAULT_BACKBONE:
            raise ModelShapeError(
                f"{source}={value!r} but the weights are a timm backbone, not "
                f"{DEFAULT_BACKBONE}"
            )
        if from_keys is not None and from_keys != "timm" and value != DEFAULT_BACKBONE:
            raise ModelShapeError(
                f"{source}={value!r} but the weights are a {DEFAULT_BACKBONE}"
            )
        if found is None:
            found = (value, source)
    if found is not None:
        return found
    if from_keys == "timm":
        raise ModelShapeError(
            "the weights are a timm backbone but no artifact records which one; "
            "its name cannot be read off the keys. Record it in the model "
            "config (trainer saves 'backbone') and load with that sidecar."
        )
    if from_keys is None:
        raise ModelShapeError(
            "the weights show no backbone at all: expected resnet50's backbone.* "
            "keys or a timm backbone.model body, or a record of which backbone "
            "they are"
        )
    return DEFAULT_BACKBONE, "weights"


def resolve_neck_channels(meta, ckpt):
    """Return ``(neck_channels, source)``; ``None`` means the pre-change widening.

    ``None`` is the model's own spelling of the 2048 widening (NECK_LEGACY), so
    a caller passes it straight to ``hcm_nn(neck_channels=...)``. A stated width
    that the weights contradict raises; with nothing stated the weights decide,
    and a checkpoint with no neck of its own is the legacy one.
    """
    kind, from_keys = _state_dict_neck(_weights_of(ckpt))
    found = None
    for source, value in _stated(meta, "neck_channels"):
        if value == NECK_LEGACY:
            raise ModelShapeError(
                f"{source}={NECK_LEGACY!r} is the pre-change widening, which is "
                f"spelled null (neck_channels=None), not the string"
            )
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ModelShapeError(f"{source}={value!r} is not a positive integer")
        if kind == "legacy":
            raise ModelShapeError(
                f"{source}={value} but the weights are the pre-change 2048 "
                f"widening, which has no {value}-channel neck"
            )
        if kind == "neck" and value != from_keys:
            raise ModelShapeError(
                f"{source}={value} but the weights show a {from_keys}-channel neck")
        if found is None:
            found = (value, source)
    if found is not None:
        return found
    if kind is None:
        # No neck in the weights and nothing stated: the pre-change model, whose
        # widening has no width to record.
        return None, "weights"
    return from_keys, "weights"


def resolve_mask_stride(meta, ckpt):
    """Return ``(mask_stride, source)`` for the model about to be built.

    A stated stride that the weights contradict raises; with nothing stated the
    weights decide, and a checkpoint with no mask refinement is the stride-4
    model -- reported as from "default", because a stride read from the absence
    of a branch is not a stride the weights stated (``resolve_out_stride``
    reports its own fallback the same way). Only a *positive* disagreement
    refuses: weights that show no mask either way leave a stated stride to
    decide.
    """
    weights = _weights_of(ckpt)
    from_keys = _state_dict_mask_stride(weights)
    found = None
    for source, value in _stated(meta, "mask_stride"):
        if value not in MASK_STRIDES:
            raise ModelShapeError(f"{source}={value!r} is not one of {MASK_STRIDES}")
        if from_keys is not None and value != from_keys:
            raise ModelShapeError(
                f"{source}={value} but the weights are a stride-{from_keys} mask")
        if found is None:
            found = (value, source)
    if found is not None:
        return found
    if from_keys is None:
        # Weights that say nothing about a mask branch: the stride-4 model, and
        # the source says so rather than reporting a stride nobody stated as
        # read from the weights (``resolve_out_stride`` spells it the same).
        return MASK_STRIDE_DEFAULT, "default"
    return from_keys, "weights"


def resolve_amodal_outline(meta, ckpt):
    """Return ``(amodal_outline, source)`` for the model about to be built.

    A stated flag that the weights contradict raises; with nothing stated the
    weights decide (``outline_head.weight`` present -> True), and a checkpoint
    showing neither head is the pre-change model, reported as from "default"
    (``resolve_mask_stride`` spells its absence the same way). Only a
    *positive* disagreement refuses: weights showing no head at all leave a
    stated True to decide, and the strict load reports what is missing.
    """
    from_keys = _state_dict_amodal_outline(_weights_of(ckpt))
    found = None
    for source, value in _stated(meta, "amodal_outline"):
        if not isinstance(value, bool):
            raise ModelShapeError(f"{source}={value!r} is not true or false")
        if from_keys is not None and value != from_keys:
            raise ModelShapeError(
                f"{source}={value} but the weights "
                f"{'have an' if from_keys else 'have no'} outline head")
        if found is None:
            found = (value, source)
    if found is not None:
        return found
    if from_keys is None:
        return False, "default"
    return from_keys, "weights"
