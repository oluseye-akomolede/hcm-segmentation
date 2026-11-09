import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from ecs_hcm_model.core.backbones import (
    DEFAULT_BACKBONE,
    ResNet50Backbone,
    backbone_names,
    backbone_spec,
    build_backbone,
)

ISC_OUTPUT_CHANNEL_NUM = 32
ISC_INNER_CHANNEL_NUMBER = 64
ISC_INPUT_CHANNEL_NUMBER = 256
RESNET_OUTPUT_CHANNEL_NUMBER = 2048
RRD_OUTPUT_CHANNEL_NUMBER = 32
ASPP_RATES = [6,12,18]

# The v2 neck and mask stride.
#
# neck_channels narrows every pyramid rung to one width with 1x1 + GroupNorm.
# ``None`` is not "unset" but the pre-change neck: resnet50 keeps the 2048
# widening built from bare 1x1 convs, which is what makes a model built with no
# new argument byte-for-byte the model that existed before. The v2 backbones
# default to NECK_CHANNELS, the proposal's 128.
#
# mask_stride is the stride of the logits the segmentation head sees. 4 is the
# decoder's own grid (the pre-change model); 2 adds the refinement below.
NECK_CHANNELS = 128
MASK_REFINE_CHANNELS = 64
MASK_STRIDE_DEFAULT = 4
MASK_STRIDES = (2, 4)

# The decoder architecture version. "v1" is the original, kept bit-identical so
# old checkpoints load and behave exactly as trained. "v2" fixes three things:
#   A. each hcm_rrd concatenated its two gated copies of f_s along the WIDTH and
#      a stride-(1,2) conv halved it back, so output column j was built from
#      input column 2j: every rung squashed the image horizontally. v2 joins
#      them on channels (a 1x1 fuse back to 32) and keeps the geometry.
#   B. hcm_csc returned only the 16x16 attention map upsampled, so no
#      full-resolution detail reached the head; v2 adds added_input back.
#   C. r4 was the one pyramid level not resized to the working resolution, so
#      isc[2]/csc[2]/rrd[2] ran at a quarter of their neighbours' size.
# Old checkpoints record no version and are "v1"; new training builds "v2".
DECODER_V1 = "v1"
DECODER_V2 = "v2"
DECODER_ARCHES = (DECODER_V1, DECODER_V2)


def infer_decoder_arch(state_dict):
    """The decoder version a state dict was saved from: only v2 has a fuse conv."""
    for key in state_dict:
        if key.startswith("rrd_list.") and ".fuse." in key:
            return DECODER_V2
    return DECODER_V1


class ArchMismatchError(ValueError):
    """A recorded architecture the weights contradict (or do not name).

    The base of the family: a loader that must not degrade a mismatch to a bare
    state dict (the downloader) catches this one and covers every field.
    """


class DecoderArchMismatchError(ArchMismatchError):
    """A recorded decoder version the weights contradict (or do not name)."""


def check_decoder_arch(recorded, state_dict):
    """Return the version to build: the recorded one, which the weights must agree with.

    ``recorded`` None means the checkpoint predates the field; the weights'
    keys decide. Otherwise a value the keys contradict raises: non-strict
    loading would drop the fuse convs, or leave them randomly initialised.
    """
    from_keys = infer_decoder_arch(state_dict)
    if recorded is None:
        return from_keys
    if recorded not in DECODER_ARCHES:
        raise DecoderArchMismatchError(
            f"unknown decoder_arch {recorded!r} (known: {DECODER_ARCHES})")
    if recorded != from_keys:
        raise DecoderArchMismatchError(
            f"recorded decoder_arch={recorded!r} but the weights are a {from_keys!r} decoder")
    return recorded


# The neck as the *weights* show it, for the one state of affairs a width
# cannot name: the pre-change pyramid, where every rung is widened to 2048 by a
# bare 1x1 conv. It is spelled out here so a caller can tell "the weights say
# the old widening" from "the weights say nothing".
NECK_LEGACY = "legacy"


def backbone_family(state_dict):
    """Which family the keys show: ``resnet50``, ``'timm'``, or None if neither.

    The family is read from the ``backbone.`` prefix as a whole, not from one
    key: a checkpoint that lost ``backbone.conv1.weight`` still has the rest of
    its resnet, and refusing it here would replace the loader's own report of
    the missing key with a worse message.
    """
    family = None
    for key in state_dict:
        if not isinstance(key, str) or not key.startswith("backbone."):
            continue
        if key.startswith("backbone.model."):
            return "timm"
        family = "resnet50"
    return family


def infer_backbone(state_dict):
    """The backbone a bare state dict was saved from.

    A bare state dict carries no name and the keys only say the family, so a
    timm backbone raises here: rebuilding it as ResNet-50 would load a
    different architecture and the caller would only find out from the keys.
    """
    family = backbone_family(state_dict)
    if family == "timm":
        raise ArchMismatchError(
            "the weights are a timm backbone, but a bare state dict records no "
            "name and the keys do not carry one; load it through its checkpoint "
            "(model_info/metadata) instead")
    return DEFAULT_BACKBONE


def check_backbone(recorded, state_dict):
    """Return the backbone to build: the recorded one, which the weights must agree with.

    ``recorded`` None means the checkpoint predates the field, and the weights
    decide -- ResNet-50, the only backbone then. A timm backbone cannot be
    guessed from the keys, so an unnamed one raises rather than being rebuilt
    as a different model. Only a *positive* disagreement refuses: weights that
    show no backbone at all leave the record to decide (and the strict load
    reports what is missing).
    """
    family = backbone_family(state_dict)
    if recorded is None:
        if family == "timm":
            raise ArchMismatchError(
                "the weights are a timm backbone but the checkpoint records no "
                "backbone; the name cannot be recovered from the weights")
        return DEFAULT_BACKBONE
    if recorded not in backbone_names():
        raise ArchMismatchError(
            f"unknown backbone {recorded!r} (registered: {backbone_names()})")
    if family is not None:
        expected = "resnet50" if backbone_spec(recorded).kind == "resnet" else "timm"
        if expected != family:
            raise ArchMismatchError(
                f"recorded backbone={recorded!r} but the weights are a {family} backbone")
    return recorded


def infer_neck_channels(state_dict):
    """The neck the weights show: a width, NECK_LEGACY, or None if they are silent.

    None means no evidence either way -- no ``adapt_c2`` block, or one whose
    tensors carry no readable shape -- not "the widening".
    """
    for key, value in state_dict.items():
        if not isinstance(key, str) or not key.startswith("adapt_c2.0."):
            continue
        shape = getattr(value, "shape", None)
        if shape:
            return int(shape[0])
    for key in state_dict:
        if isinstance(key, str) and key == "adapt_c2.weight":
            return NECK_LEGACY
    return None


def check_neck_channels(recorded, state_dict):
    """Return the neck to build: the recorded width, which the weights must agree with.

    ``recorded`` None means the checkpoint predates the field; the weights
    decide. NECK_LEGACY is returned for the pre-change 2048 widening, which has
    no width to record (the model spells it ``neck_channels=None``).
    """
    from_keys = infer_neck_channels(state_dict)
    if recorded is None:
        return NECK_LEGACY if from_keys is None else from_keys
    if recorded == NECK_LEGACY:
        raise ArchMismatchError(
            "neck_channels 'legacy' is the pre-change 2048 widening, which is recorded "
            "as null (the model spells it neck_channels=None), not as the string")
    if isinstance(recorded, bool) or not isinstance(recorded, int) or recorded <= 0:
        raise ArchMismatchError(
            f"neck_channels {recorded!r} is not a positive integer")
    if from_keys is None:
        # Silent weights: the record decides, and the strict load names whatever
        # is missing. (A neck-less state dict from a pre-neck model would have
        # been caught by the family check long before this.)
        return recorded
    if from_keys == NECK_LEGACY:
        raise ArchMismatchError(
            f"recorded neck_channels={recorded} but the weights are the pre-change "
            f"2048 widening, which is not a {recorded}-channel neck")
    if recorded != from_keys:
        raise ArchMismatchError(
            f"recorded neck_channels={recorded} but the weights show a {from_keys}-channel neck")
    return recorded


def neck_argument(resolved):
    """The constructor's spelling of a resolved breadth.

    The helpers above speak NECK_LEGACY for the pre-change widening; the
    constructor spells the same thing ``None``. Every loader that hands a
    resolved value to ``hcm_nn(...)`` goes through here, so the sentinel never
    reaches the width validation.
    """
    return None if resolved == NECK_LEGACY or resolved is None else resolved


def infer_mask_stride(state_dict):
    """The mask stride the weights show: 2, 4, or None if they are silent.

    Any of the refinement's three modules means stride 2. A complete model with
    a segmentation head and none of them is the stride-4 model; anything else
    (no head, no readable keys) is no evidence, and the record or the default
    decides.
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


def infer_amodal_outline(state_dict):
    """Whether the weights carry an outline head: True/False, or None if silent.

    None means the state dict shows neither head, so it is no evidence either
    way; a segmentation head without an outline head is a *complete* model that
    has none, which is positive evidence of False.
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


def check_amodal_outline(recorded, state_dict):
    """Return whether to build the outline head: the recorded flag, checked.

    ``recorded`` None means the checkpoint predates the field; the weights
    decide (``outline_head.weight`` present -> True), and a model with neither
    head is not an outline model. A recorded value the weights contradict
    raises: loading a checkpoint over a bigger architecture would leave the
    outline head randomly initialised, and over a smaller one would silently
    drop it. Only a *positive* disagreement refuses -- weights showing no head
    at all leave the record to decide, and the load reports what is missing.
    """
    from_keys = infer_amodal_outline(state_dict)
    if recorded is None:
        return bool(from_keys)
    if not isinstance(recorded, bool) and recorded not in (0, 1):
        raise ArchMismatchError(
            f"recorded amodal_outline={recorded!r} is not a boolean")
    recorded = bool(recorded)
    if from_keys is not None and recorded != from_keys:
        raise ArchMismatchError(
            f"recorded amodal_outline={recorded} but the weights "
            f"{'have an' if from_keys else 'have no'} outline head")
    return recorded


def check_mask_stride(recorded, state_dict):
    """Return the mask stride to build: the recorded one, which the weights must agree with.

    ``recorded`` None means the checkpoint predates the field; the weights
    decide, and a checkpoint with no refinement is the stride-4 model.
    """
    from_keys = infer_mask_stride(state_dict)
    if recorded is None:
        return MASK_STRIDE_DEFAULT if from_keys is None else from_keys
    if recorded not in MASK_STRIDES:
        raise ArchMismatchError(
            f"unknown mask_stride {recorded!r} (known: {MASK_STRIDES})")
    if from_keys is not None and recorded != from_keys:
        raise ArchMismatchError(
            f"recorded mask_stride={recorded} but the weights are a stride-{from_keys} mask")
    return recorded


def arch_record_of(model):
    """The architecture fields of a model, under any DDP/DataParallel wrapper.

    ``{}`` for a model that does not declare them, so an uploader can spread the
    result into a metadata dict without caring what it was handed.
    """
    inner = getattr(model, "module", model)
    record = getattr(inner, "arch_record", None)
    return record() if callable(record) else {}


def validate_arch(backbone=DEFAULT_BACKBONE, neck_channels=None,
                  mask_stride=MASK_STRIDE_DEFAULT):
    """Check the three architecture choices, returning the resolved ``neck_channels``.

    Raises ``ValueError`` naming the offending value. The model constructor and
    the training entry point both call this, so a refused value is refused
    before an MLflow run, a GPU preflight and a download rather than at model
    construction, and the two callers cannot drift apart.

    ``neck_channels`` None is the backbone's own neck: resnet50 keeps the
    pre-change 2048 widening, so a model built with no new argument is the model
    that existed before and its checkpoints load unchanged. The v2 backbones
    have no pre-change model to preserve and default to NECK_CHANNELS.
    """
    if mask_stride not in MASK_STRIDES:
        raise ValueError(f"mask_stride {mask_stride!r} is not one of {MASK_STRIDES}")
    kind = backbone_spec(backbone).kind
    if neck_channels is None:
        return None if kind == "resnet" else NECK_CHANNELS
    if isinstance(neck_channels, bool) or not isinstance(neck_channels, int) or neck_channels <= 0:
        raise ValueError(f"neck_channels {neck_channels!r} is not a positive integer")
    return neck_channels


def _check(name, t):
    finite = torch.isfinite(t)
    if not finite.all():
        bad = t[~finite].flatten()[:5]
        tf = t[finite]
        mn = tf.min().item() if tf.numel() else float("nan")
        mx = tf.max().item() if tf.numel() else float("nan")
        print(f"[NaN/Inf] {name}: shape={tuple(t.shape)} finite_min={mn} finite_max={mx} bad_sample={bad}")
        raise RuntimeError(f"Non-finite detected in {name}")


# The largest magnitude that may be handed to code running under autocast/fp16.
# fp16 tops out at 65504, and the cast saturates silently: 70000.0 in fp32 is
# finite, and `.to(float16)` turns it into inf with no warning at all. Leave
# headroom below the ceiling so the boundary trips before a value is one
# rounding step away from saturating.
FP16_MAX_SAFE = 6.0e4


def _guard_fp16_boundary(name, t):
    """Refuse to hand a tensor of unsafe magnitude to fp16/autocast code.

    Raises rather than clamps: clamping would hide real activation growth from
    the owner, which is the thing this whole path exists to make visible. A
    value that trips this is a genuine numerical problem worth seeing, not
    something to quietly flatten.
    """
    mx = t.detach().abs().amax().item()
    if not math.isfinite(mx) or mx > FP16_MAX_SAFE:
        raise RuntimeError(
            f"{name} max|x|={mx:.6g} exceeds the safe fp16 boundary "
            f"{FP16_MAX_SAFE:.6g} (fp16 goes non-finite above 65504); "
            f"casting here would overflow to inf"
        )
    return t


# Stages whose activation magnitude is watched. These values are plain
# magnitudes, not percentages -- they must stay out of any 0-100 range guard.
#
# "isc3" is isc_output[3] specifically: it is monitored on its own because it is
# the only ISC output that reaches the decoder (rrd_list[3] takes it directly),
# so it is the stage whose growth the fp16 boundary actually depends on. The
# aggregate "isc" is the max over all four ISC outputs, which can hide stage 3
# sitting at the ceiling while the other three stay small.
ACTIVATION_STAGES = ("isc", "csc", "isc3")


def _record_absmax(module, stage, t):
    """Fold max|t| into the module's running max for ``stage``.

    Detached and host-side: this is a diagnostic, so it must not carry a
    gradient or hold the graph alive. The caller (the trainer) reads and resets
    it at each logging point, so the running max catches a single spike between
    logging points rather than only sampling the batch that happens to land on
    one.
    """
    try:
        mx = t.detach().abs().amax().item()
    except (RuntimeError, AttributeError):
        # Never let a diagnostic break training.
        return
    if not isinstance(getattr(module, "activation_absmax", None), dict):
        module.activation_absmax = {s: 0.0 for s in ACTIVATION_STAGES}
    if mx > module.activation_absmax.get(stage, 0.0):
        module.activation_absmax[stage] = float(mx)


def group_norm(channels, num_groups=32):
    return nn.GroupNorm(num_groups=min(num_groups, channels), num_channels=channels)

def modify_resnet50_stride(model,stride_layer=2):
    # Modify the first convolutional layer
    model.conv1 = torch.nn.Conv2d(3, 64, kernel_size=7, stride=1, padding=3, bias=False)
    
    # Modify the downsampling layers in the residual blocks
    strides = [1, 1, 1, 1]
    strides[stride_layer-1] = 1
    
    for i, layer in enumerate([model.layer1, model.layer2, model.layer3, model.layer4]):
        for block in layer:
            block.conv1.stride = (1, 1)
            block.conv2.stride = (1, 1)
            if block.downsample is not None:
                block.downsample[0].stride = (1, 1)
        if i < 4:
            layer[0].conv1.stride = (strides[i], strides[i])
            if layer[0].downsample is not None:
                layer[0].downsample[0].stride = (strides[i], strides[i])
    return model

def modify_resnet50_stride_and_channels(model,stride_layer=2):
    # Modify the first convolutional layer
    model.conv1 = torch.nn.Conv2d(2048, 64, kernel_size=7, stride=1, padding=3, bias=False)
    
    # Modify the downsampling layers in the residual blocks
    strides = [1, 1, 1, 1]
    strides[stride_layer-1] = 1
    
    for i, layer in enumerate([model.layer1, model.layer2, model.layer3, model.layer4]):
        for block in layer:
            block.conv1.stride = (1, 1)
            block.conv2.stride = (1, 1)
            if block.downsample is not None:
                block.downsample[0].stride = (1, 1)
        if i < 4:
            layer[0].conv1.stride = (strides[i], strides[i])
            if layer[0].downsample is not None:
                layer[0].downsample[0].stride = (strides[i], strides[i])
    return model

class hcm_isc(nn.Module):
    def __init__(self, in_channels=RESNET_OUTPUT_CHANNEL_NUMBER):
        super(hcm_isc,self).__init__()
        # The neck width. The ISC's own widths stay at their pre-change values
        # and cap, rather than follow, a narrower neck: at the legacy 2048 the
        # three are 256/256/64 exactly as before, and a 128-channel neck takes
        # them to 128/128/64. Scaling them *down* in proportion would leave a
        # 128-channel neck a 16-channel middle, and the ISC is the module the
        # whole decoder reads from.
        self.mid_channels = min(in_channels, ISC_INPUT_CHANNEL_NUMBER)
        self.inner_channels = min(self.mid_channels, ISC_INNER_CHANNEL_NUMBER)
        self.conv1_1_1x1 = nn.Conv2d(in_channels,ISC_OUTPUT_CHANNEL_NUM,kernel_size=1,padding=0)
        self.conv1_2_1x1 = nn.Conv2d(in_channels,self.mid_channels,kernel_size=1,padding=0)
        self.conv1_3_1x1 = nn.Conv2d(in_channels,self.mid_channels,kernel_size=1,padding=0)
        self.conv2_1_3x3 = nn.Conv2d(self.mid_channels,self.inner_channels,kernel_size=3,padding=1)
        self.conv2_2_5x5 = nn.Conv2d(self.mid_channels,self.inner_channels,kernel_size=5, padding=2)
        self.conv3_1_3x3 = nn.Conv2d(self.inner_channels,self.inner_channels,kernel_size=3,padding=1)
        self.conv3_2_5x5 = nn.Conv2d(self.inner_channels,self.inner_channels,kernel_size=5, padding=2)
        self.conv4_1_3x3 = nn.Conv2d(self.inner_channels,ISC_OUTPUT_CHANNEL_NUM,kernel_size=3,padding=1)

    def get_model_size(self):
        total_params = sum(p.numel() for p in self.parameters())
        total_size = total_params * 4  # 4 bytes per parameter (float32)
        return total_size / (1024 * 1024)  # Convert to MB

    def forward(self, x):
        _check("isc_in", x)
        x = x.float()

        out1_1 = self.conv1_1_1x1(x)
        out1_2 = self.conv1_2_1x1(x)
        out1_3 = self.conv1_3_1x1(x)

        out_2_1 = self.conv2_1_3x3(out1_2)
        _check("out_2_1", out_2_1)
        out_2_2 = self.conv2_2_5x5(out1_3)

        int_out_2 = out_2_1 + out_2_2
        _check("int_out_2", int_out_2)

        out3_1 = self.conv3_1_3x3(int_out_2)
        _check("out3_1", out3_1)
        out3_2 = self.conv3_2_5x5(int_out_2)
        _check("out3_2", out3_2)

        int_out_3 = out3_1 * out3_2
        int_out_3 = torch.nan_to_num(int_out_3, nan=0.0, posinf=1e4, neginf=-1e4)
        int_out_3 = int_out_3.clamp(-1e4, 1e4)

        int_out_3 = _gn32(int_out_3, groups=8)
        _check("int_out_3", int_out_3)

        out4_1 = self.conv4_1_3x3(int_out_3)
        _check("out4_1", out4_1)

        output = out1_1 + out4_1
        _check("output", output)

        # Return fp32, NOT the input dtype. `output` is unbounded after the
        # group_norm above, and casting it back to fp16 saturates silently once
        # |output| > 65504: a real run died here with a -inf in isc_out[0] that
        # `_check` had just passed, because the check ran on the fp32 value and
        # the very next statement is what produced the inf. The caller keeps
        # this in fp32 and guards the boundary where fp16 resumes.
        return output

class JointAttention(nn.Module):
    def __init__(self, in_channels, num_heads=8):
        super(JointAttention,self).__init__()
        self.in_channels = in_channels
        self.num_heads = num_heads
        self.query = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.key = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.value = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.out_proj = nn.Conv2d(in_channels, in_channels, kernel_size=1)

    def get_model_size(self):
        total_params = sum(p.numel() for p in self.parameters())
        total_size = total_params * 4
        return total_size / (1024 * 1024)

    def forward(self, x):
        B, C, H, W = x.size()
        if H > 16 or W > 16:
            x = F.adaptive_avg_pool2d(x, (16, 16))
        head_dim = C // self.num_heads
        assert C % self.num_heads == 0, f"C={C} not divisible by heads={self.num_heads}"

        q = self.query(x).view(B, self.num_heads, head_dim, H * W).transpose(2, 3)  # (B, h, HW, d)
        k = self.key(x).view(B, self.num_heads, head_dim, H * W).transpose(2, 3)    # (B, h, HW, d)
        v = self.value(x).view(B, self.num_heads, head_dim, H * W)                  # (B, h, d, HW)
        _check("JA_x", x)
        _check("JA_q", q)
        _check("JA_k", k)
        _check("JA_v", v)

        # ---- critical stability block: do scores + softmax in fp32 ----
        qf = q.float()
        kf = k.float()
        vf = v.float()

        scale = 1.0 / math.sqrt(head_dim)
        scores = torch.matmul(qf, kf.transpose(2, 3)) * scale          # (B, h, HW, HW)
        
        # stability trick used everywhere
        scores = scores - scores.max(dim=-1, keepdim=True).values

        attn = F.softmax(scores, dim=-1)                               # fp32 softmax
        out = torch.matmul(attn, vf.transpose(2, 3))                   # (B, h, HW, d)
        out = out.to(dtype=q.dtype)                                    # cast back (fp16/bf16)

        if not torch.isfinite(scores).all():
            print("scores min/max:", scores.min().item(), scores.max().item())
            raise RuntimeError("Non-finite scores")

        out = out.transpose(2, 3).contiguous().view(B, C, H, W)
        return self.out_proj(out)

class PONO(torch.nn.Module):
    def __init__(self, num_channels):
        super(PONO,self).__init__()
        self.num_channels = num_channels
        self.weight = torch.nn.Parameter(torch.ones(1, num_channels, 1, 1))
        self.bias = torch.nn.Parameter(torch.zeros(1, num_channels, 1, 1))
    
    def get_model_size(self):
        total_params = sum(p.numel() for p in self.parameters())
        total_size = total_params * 4
        return total_size / (1024 * 1024)

    def forward(self, x):
         # x: (B, C, H, W)
        mean = torch.mean(x, dim=(2, 3), keepdim=True)
        std = torch.std(x, dim=(2, 3), keepdim=True, unbiased=False)
        std = std.clamp_min(1e-5)
        normalized = (x - mean) / std
        return self.weight * normalized + self.bias

class hcm_csc(nn.Module):
    def __init__(self, decoder_arch=DECODER_V1):
        super(hcm_csc,self).__init__()
        if decoder_arch not in DECODER_ARCHES:
            raise ValueError(f"decoder_arch must be one of {DECODER_ARCHES}, got {decoder_arch!r}")
        self.decoder_arch = decoder_arch
        self.norm = nn.GroupNorm(num_groups=8, num_channels=ISC_OUTPUT_CHANNEL_NUM)
        self.pre_attn_norm = nn.GroupNorm(num_groups=8, num_channels=ISC_OUTPUT_CHANNEL_NUM)
        self.JA = JointAttention(in_channels=ISC_OUTPUT_CHANNEL_NUM)
        self.PN = PONO(ISC_OUTPUT_CHANNEL_NUM)
    
    def get_model_size(self):
        total_params = sum(p.numel() for p in self.parameters())
        total_size = total_params * 4
        return total_size / (1024 * 1024)

    def forward(self, f_s, f_splus1):
        _check("f_s", f_s)
        _check("f_splus1", f_splus1)
        f_s = self.norm(f_s)
        _check("normed_f_s", f_s)
        f_splus1 = self.norm(f_splus1)
        _check("normed_f_splus1", f_splus1)
        upsampled = F.interpolate(
            f_splus1,
            size=f_s.shape[-2:],     # ✅ match spatial dims exactly
            mode="bilinear",
            align_corners=False
        )
        _check("upsampled", upsampled)
        added_input = torch.add(f_s, upsampled)
        _check("added_input", added_input)
        # HARD CAP before attention — this is mandatory
        added_input_ds = F.adaptive_avg_pool2d(added_input, (16, 16))

        added_input_ds = self.pre_attn_norm(added_input_ds)

        _check("added_input_ds", added_input_ds)
        joint = self.JA(added_input_ds)
        _check("JA_out", joint)
        joint = F.interpolate(joint, size=added_input.shape[-2:], mode="bilinear", align_corners=False)
        normalized = self.PN(joint)
        _check("PONO_out", normalized)
        if self.decoder_arch == DECODER_V2:
            # The full-resolution input, so detail finer than the 16x16
            # attention grid still reaches the decoder.
            return added_input + normalized
        return normalized

class ResidualChannelAttentionBlock(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super(ResidualChannelAttentionBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(in_channels, in_channels // reduction_ratio)
        self.fc2 = nn.Linear(in_channels // reduction_ratio, in_channels)
        self.sigmoid = nn.Sigmoid()

    def get_model_size(self):
        total_params = sum(p.numel() for p in self.parameters())
        total_size = total_params * 4
        return total_size / (1024 * 1024)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)

        # Channel Attention
        attn = self.avg_pool(out)
        attn = attn.view(attn.size(0), -1)
        attn = self.fc1(attn)
        attn = self.relu(attn)
        attn = self.fc2(attn)
        attn = self.sigmoid(attn)
        attn = attn.view(attn.size(0), attn.size(1), 1, 1)
        
        out = out * attn
        out = out + residual
        return out
    
class hcm_rrd(nn.Module):
    def __init__(self, decoder_arch=DECODER_V1):
        super().__init__()
        if decoder_arch not in DECODER_ARCHES:
            raise ValueError(f"decoder_arch must be one of {DECODER_ARCHES}, got {decoder_arch!r}")
        self.decoder_arch = decoder_arch
        if decoder_arch == DECODER_V2:
            # 1x1 fuse of the two 32-channel copies back to 32, so the RCAB and
            # conv below keep v1's shapes and weights (+2080 params).
            self.fuse = nn.Conv2d(2 * ISC_OUTPUT_CHANNEL_NUM, ISC_OUTPUT_CHANNEL_NUM, kernel_size=1)
            stride = 1
        else:
            stride = (1, 2)
        self.RCAB = ResidualChannelAttentionBlock(ISC_OUTPUT_CHANNEL_NUM)
        self.conv2d = nn.Conv2d(ISC_OUTPUT_CHANNEL_NUM,RRD_OUTPUT_CHANNEL_NUMBER,kernel_size=3,padding=1,stride=stride)

    def get_model_size(self):
        total_params = sum(p.numel() for p in self.parameters())
        total_size = total_params * 4
        return total_size / (1024 * 1024)

    def forward(self,f_s:torch.Tensor,p_splus1:torch.Tensor):
        repeated_p_s = F.interpolate(
            p_splus1,
            size=f_s.shape[-2:],     # ✅ match f_s exactly
            mode="bilinear",
            align_corners=False
        )
        sig_p_s = F.sigmoid(repeated_p_s)
        fp_mult_normal = torch.mul(f_s,sig_p_s)
        fp_mult_normal = F.group_norm(fp_mult_normal, num_groups=8, eps=1e-5)

        sig_p_s_reversed = torch.sub(1,sig_p_s)
        fp_mult_reversed = torch.mul(f_s,sig_p_s_reversed)
        fp_mult_reversed = F.group_norm(fp_mult_reversed, num_groups=8, eps=1e-5)

        if self.decoder_arch == DECODER_V2:
            cat_tensor = self.fuse(torch.cat((fp_mult_normal,fp_mult_reversed),dim=1))
        else:
            cat_tensor = torch.cat((fp_mult_normal,fp_mult_reversed),dim=3)
        rcab_output = self.RCAB(cat_tensor)
        conv_output = self.conv2d(rcab_output)
        return conv_output
    

class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels, rates):
        super(ASPP, self).__init__()
        self.conv1x1 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.bn1x1 = group_norm(out_channels)

        self.conv_rates = nn.ModuleList()
        self.bn_rates = nn.ModuleList()
        for rate in rates:
            self.conv_rates.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=rate, dilation=rate))
            self.bn_rates.append(group_norm(out_channels))

        self.global_avg_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            group_norm(out_channels),
            nn.ReLU()
        )

        self.conv_concat = nn.Conv2d(out_channels * (len(rates) + 2), out_channels, kernel_size=1, stride=(2, 2))
        self.bn_concat = group_norm(out_channels)

    def get_model_size(self):
        total_params = sum(p.numel() for p in self.parameters())
        total_size = total_params * 4
        return total_size / (1024 * 1024)

    def forward(self, x):
        res = [F.relu(self.bn1x1(self.conv1x1(x)))]
        for conv, bn in zip(self.conv_rates, self.bn_rates):
            res.append(F.relu(bn(conv(x))))
        pool = F.interpolate(self.global_avg_pool(x), size=x.shape[2:], mode='bilinear', align_corners=True)
        res.append(pool)
        x = torch.cat(res, dim=1)
        x = F.relu(self.bn_concat(self.conv_concat(x)))
        return x

class DifferentiableROIModule(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Conv2d(in_channels, 1, kernel_size=1),
            nn.Sigmoid()  # [B, 1, H, W] — attention map
        )

    def get_model_size(self):
        total_params = sum(p.numel() for p in self.parameters())
        total_size = total_params * 4
        return total_size / (1024 * 1024)

    def forward(self, features):
        attn_map = self.attn(features)
        return attn_map

def _check_tensor(name, t):
    if not torch.is_tensor(t):
        print(f"{name}: not a tensor")
        return
    finite = torch.isfinite(t)
    if not finite.all():
        bad = t.flatten()[~finite.flatten()][:5]
        print(f"[NaN/Inf] {name}: shape={tuple(t.shape)} "
              f"min={t.min().item()} max={t.max().item()} bad_sample={bad}")
        raise RuntimeError(f"Non-finite detected in {name}")
    # optional: also print magnitude diagnostics
    t_absmax = t.abs().max().item()
    print(f"[OK] {name}: shape={tuple(t.shape)} min={t.min().item():.4g} "
          f"max={t.max().item():.4g} absmax={t_absmax:.4g} dtype={t.dtype}")

def _gn32(t, groups=32, eps=1e-5):
    # groups must divide channels
    c = t.shape[1]
    g = min(groups, c)
    while c % g != 0 and g > 1:
        g -= 1
    return F.group_norm(t, num_groups=g, eps=eps)

def _neck_norm(channels, num_groups=32):
    # GroupNorm needs its group count to divide its channel count, and the neck
    # width is configurable, so clamp instead of assuming the two divide.
    g = min(num_groups, channels)
    while channels % g != 0 and g > 1:
        g -= 1
    return nn.GroupNorm(g, channels)


def _neck_block(in_channels, out_channels):
    """One v2 neck rung: 1x1 to the neck width, then GroupNorm.

    Deliberately a different module from the pre-change widening, so a state
    dict from one neck cannot load into the other.
    """
    return nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=1),
                         _neck_norm(out_channels))


def _conv_norm_relu(in_channels, out_channels):
    """One step of the mask refinement: 3x3 conv, GroupNorm, ReLU."""
    return nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                         _neck_norm(out_channels),
                         nn.ReLU(inplace=True))

class hcm_nn(nn.Module):
    def __init__(self, enable_finetuning: bool = False, num_segmentation_classes: int = None, out_stride=16,
                 decoder_arch=DECODER_V1, backbone: str = DEFAULT_BACKBONE, neck_channels: int = None,
                 mask_stride: int = MASK_STRIDE_DEFAULT, backbone_pretrained: bool = True,
                 amodal_outline: bool = False):
        super().__init__()

        neck_channels = validate_arch(backbone, neck_channels, mask_stride)
        if amodal_outline and num_segmentation_classes is None:
            # The outline head is built beside the segmentation head and is read
            # through the same forward, so a model with neither head cannot have
            # one: otherwise the architecture record could say True while no
            # outline weights exist to load.
            raise ValueError(
                "amodal_outline=True needs a segmentation head; this model was "
                "built with num_segmentation_classes=None")
        self.amodal_outline = bool(amodal_outline)

        self.backbone_name = backbone
        self.neck_channels = neck_channels
        self.mask_stride = mask_stride
        spec = backbone_spec(backbone)
        self.spec = spec

        self.backbone = build_backbone(backbone, out_stride=out_stride,
                                       pretrained=backbone_pretrained)
        self.backbone = self.backbone.float()

        # The pyramid the ISC reads. The pre-change model widened every rung to
        # 2048 with a bare 1x1; a v2 neck narrows each rung to neck_channels with
        # 1x1 + GroupNorm. The two are distinct modules on purpose (see the spec
        # delta): a state dict from one must not load into the other.
        if neck_channels is None:
            self.adapt_c2 = nn.Conv2d(spec.rung_channels[0], RESNET_OUTPUT_CHANNEL_NUMBER, kernel_size=1)
            self.adapt_c3 = nn.Conv2d(spec.rung_channels[1], RESNET_OUTPUT_CHANNEL_NUMBER, kernel_size=1)
            self.adapt_c4 = nn.Conv2d(spec.rung_channels[2], RESNET_OUTPUT_CHANNEL_NUMBER, kernel_size=1)
            self.adapt_c5 = nn.Identity()  # the last rung is already 2048 wide
            neck_width = RESNET_OUTPUT_CHANNEL_NUMBER
        else:
            self.adapt_c2 = _neck_block(spec.rung_channels[0], neck_channels)
            self.adapt_c3 = _neck_block(spec.rung_channels[1], neck_channels)
            self.adapt_c4 = _neck_block(spec.rung_channels[2], neck_channels)
            self.adapt_c5 = _neck_block(spec.rung_channels[3], neck_channels)
            neck_width = neck_channels

        self.ASPP = ASPP(neck_width, ISC_OUTPUT_CHANNEL_NUM, ASPP_RATES)

        self.isc_list = nn.ModuleList([hcm_isc(neck_width) for _ in range(4)])
        self.csc_list = nn.ModuleList([hcm_csc(decoder_arch) for _ in range(3)])
        self.rrd_list = nn.ModuleList([hcm_rrd(decoder_arch) for _ in range(4)])
        self.differentiable_roi_module = DifferentiableROIModule(ISC_OUTPUT_CHANNEL_NUM)

        self.roi_classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),  # reduced spatial size
            nn.Flatten(),
            nn.Linear(ISC_OUTPUT_CHANNEL_NUM, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 1000)
        )

        #nn.init.kaiming_normal_(self.roi_classifier[2].weight, mode='fan_out', nonlinearity='relu')
        #nn.init.zeros_(self.roi_classifier[-1].bias)

        # The finest mask: rrd_output[0] is the decoder's stride-4 grid, so at
        # mask_stride 2 the head sees it upsampled and merged with a 32-channel
        # projection of the earliest half-resolution signal the backbone has --
        # its stride-2 tap, or, for a backbone whose stem downsamples 4x and so
        # genuinely has none (ConvNeXtV2-nano), a 2x-upsampled projection of its
        # first rung. The merged width is 2*32 in both cases, so the refinement
        # below has one shape. None of this exists at stride 4, so a stride-4
        # checkpoint's keys are exactly the keys it had before.
        if mask_stride == 2:
            tap_in = spec.tap_channels if spec.tap_channels is not None else spec.rung_channels[0]
            self.mask_proj = nn.Conv2d(tap_in, ISC_OUTPUT_CHANNEL_NUM, kernel_size=1)
            self.mask_conv1 = _conv_norm_relu(ISC_OUTPUT_CHANNEL_NUM * 2,
                                              MASK_REFINE_CHANNELS)
            self.mask_conv2 = _conv_norm_relu(MASK_REFINE_CHANNELS, ISC_OUTPUT_CHANNEL_NUM)
        else:
            self.mask_proj = None
            self.mask_conv1 = None
            self.mask_conv2 = None

        if num_segmentation_classes is not None:
            self.segmentation_head = nn.Conv2d(ISC_OUTPUT_CHANNEL_NUM, num_segmentation_classes, kernel_size=1)
        else:
            self.segmentation_head = None

        # The full (amodal) outline: one channel off the same feature map the
        # segmentation head reads, so the visible head and every consumer of it
        # keep their shape and meaning (a separate head, not a third class).
        if amodal_outline:
            self.outline_head = nn.Conv2d(ISC_OUTPUT_CHANNEL_NUM, 1, kernel_size=1)
        else:
            self.outline_head = None

        self.enable_finetuning = enable_finetuning
        # Recorded so the uploader's model_info.pkl carries real values; without
        # them it pickled None and the loader could not rebuild the model.
        self.out_stride = out_stride
        self.num_segmentation_classes = num_segmentation_classes
        self.decoder_arch = decoder_arch

        # Running max|activation| per stage since the last reset; the trainer
        # reads and clears this at each logging point. Plain magnitudes, not
        # percentages -- they must not meet the 0-100 metric guard.
        self.activation_absmax = {s: 0.0 for s in ACTIVATION_STAGES}

    def reset_activation_absmax(self):
        """Clear the running activation maxima (called after each log)."""
        for stage in self.activation_absmax:
            self.activation_absmax[stage] = 0.0

    def resnet_plus_norm(self, mod, inp):
        out = mod(inp)
        out = _gn32(out.float()).to(out.dtype)
        return out

    def forward_internal(self, x, return_outline=False):
        if return_outline:
            if self.outline_head is None:
                raise RuntimeError(
                    "return_outline=True but this model has no outline head; "
                    "build it with amodal_outline=True")
            if not self.enable_finetuning:
                raise RuntimeError(
                    "return_outline=True needs the segmentation path "
                    "(enable_finetuning=True); this model returns the ROI "
                    "classifier's logits")
        _check("model_in_x", x)
        d_type = x.dtype
        with torch.cuda.amp.autocast(enabled=False):
            x32 = x.float()
            (c2, c3, c4, c5), tap = self.backbone.forward_with_tap(x32)
            _check("c2", c2); _check("c3", c3); _check("c4", c4); _check("c5", c5)

        c2 = c2.to(d_type)
        c3 = c3.to(d_type)
        c4 = c4.to(d_type)
        c5 = c5.to(d_type)
        # The backbone's half-resolution feature, when it has one. Both paths
        # already compute it, so asking for it costs nothing here.
        if tap is not None:
            tap = tap.to(d_type)

        # Make a consistent pyramid, all as 2048-ch maps
        r2 = self.adapt_c2(c2); _check("r2", r2)
        r3 = self.adapt_c3(c3); _check("r3", r3)
        r4 = self.adapt_c4(c4); _check("r4", r4)
        r5 = self.adapt_c5(c5); _check("r5", r5)

        # Optional: pick a "working resolution" (e.g., r4 spatial size)
        target_hw = r2.shape[-2:]
        r2 = F.interpolate(r2, size=target_hw, mode="bilinear", align_corners=False)
        r3 = F.interpolate(r3, size=target_hw, mode="bilinear", align_corners=False)
        if self.decoder_arch == DECODER_V2:
            # v1 left r4 out, so its rung ran at a quarter of the others' size.
            r4 = F.interpolate(r4, size=target_hw, mode="bilinear", align_corners=False)
        r5 = F.interpolate(r5, size=target_hw, mode="bilinear", align_corners=False)

        # Now you have 4 depth-aligned feature maps that "work at all depths"
        res_output = [r2, r3, r4, r5]

        isc_output = []
        # The whole ISC->CSC chain runs outside autocast, in fp32. The CSC stage
        # must be INSIDE this block, not merely handed fp32 tensors: autocast
        # casts a conv's *inputs* down to fp16 even when the caller passes fp32,
        # so a CSC loop sitting just outside here would silently lose the range
        # one step later and overflow exactly as the ISC cast used to.
        with torch.cuda.amp.autocast(enabled=False):
            for i in range(4):
                t = self.isc_list[i](res_output[i].float())
                _check(f"isc_out[{i}]", t)
                _record_absmax(self, "isc", t)
                if i == 3:
                    # Tracked separately: this is the only ISC output that
                    # reaches the decoder, so it is the one the rrd boundary
                    # below has to survive.
                    _record_absmax(self, "isc3", t)
                isc_output.append(t)

            csc_output = [None, None, None]
            for i in range(2, -1, -1):
                if i != 2:
                    csc_output[i] = self.csc_list[i](isc_output[i], csc_output[i + 1])
                else:
                    csc_output[i] = self.csc_list[i](isc_output[i], isc_output[i + 1])
                _record_absmax(self, "csc", csc_output[i])

        aspp_output = self.ASPP(res_output[3])

        del res_output
        # The single guarded boundary. Everything above runs fp32; everything
        # from here on runs under the caller's autocast, so these fp32 values
        # get cast down to fp16 for the decoder convs. That cast is safe only
        # while the magnitudes are, so refuse loudly rather than let it saturate
        # into the inf that killed the run.
        #
        # rrd_list[3] is the one rung that consumes a RAW ISC output
        # (isc_output[3]); every other rung consumes a CSC output, which is
        # already small. The ISC outputs are the largest tensors in the model
        # (smoke runs saw ~41k against an fp16 ceiling of 65504), so this rung
        # must run outside autocast for the same reason the CSC loop above does:
        # autocast would cast the conv's *inputs* down to fp16 and lose the
        # range one step before the guard below could see it.
        rrd_output = [None, None, None, None]
        for i in range(3, -1, -1):
            if i != 3:
                rrd_output[i] = self.rrd_list[i](csc_output[i], rrd_output[i + 1])
            else:
                with torch.cuda.amp.autocast(enabled=False):
                    rrd_output[i] = self.rrd_list[i](
                        isc_output[i].float(), aspp_output.float()
                    )
            _guard_fp16_boundary(f"rrd_out[{i}]", rrd_output[i])

        del isc_output, csc_output, aspp_output

        output = rrd_output[0]

        if self.enable_finetuning:
            if self.mask_stride == 2:
                # The head sees half the stride-4 grid: the finer mask the
                # proposal asks for, which is the commented-out interpolate this
                # replaces, done with the backbone's own detail rather than
                # bilinear alone.
                output = self._refine_mask(output, c2, tap)
            seg_logits = self.segmentation_head(output) if self.segmentation_head else output
            if return_outline:
                # Same feature map, one more 1x1 conv: the decoder is computed
                # once, so the outline cannot drift from the visible mask.
                return seg_logits, self.outline_head(output)
            return seg_logits

        # The ROI path keeps consuming the decoder's own stride-4 map: the
        # refinement is a segmentation-head concern, and its pooled classifier
        # has no use for the finer grid.
        attn_map = self.differentiable_roi_module(output)
        weighted_features = output * attn_map
        pooled = self.roi_classifier[0](weighted_features)
        flattened = self.roi_classifier[1](pooled)
        logits = self.roi_classifier[2:](flattened)

        return logits

    def _refine_mask(self, output, first_rung, tap):
        """The stride-2 mask: the stride-4 decoder output, upsampled and merged.

        The merge partner is the finest signal the backbone has and is projected
        to the decoder's own ISC_OUTPUT_CHANNEL_NUM first: the stride-2 ``tap``
        when the backbone exposes one, otherwise its first rung (the only
        sub-stride-8 level a stem that downsamples 4x leaves), brought up to the
        same grid. Both give the refinement the same cached width, and it
        returns to ISC_OUTPUT_CHANNEL_NUM, so the segmentation head is the one
        the stride-4 model already has and a stride-4 head's weights load here.
        """
        up = F.interpolate(output, scale_factor=2, mode="bilinear", align_corners=False)
        if tap is not None:
            if tap.shape[-2:] != up.shape[-2:]:
                # Input sizes that are not a multiple of 4 can leave the two
                # grids a pixel apart. Anything further apart is not the
                # half-resolution feature, and resampling it would hide that.
                apart = max(abs(tap.shape[-2] - up.shape[-2]), abs(tap.shape[-1] - up.shape[-1]))
                if apart > 1:
                    raise RuntimeError(
                        f"the {self.backbone_name} tap is {tuple(tap.shape[-2:])} but the "
                        f"mask grid is {tuple(up.shape[-2:])}; the tap is not this "
                        f"backbone's half-resolution feature"
                    )
                tap = F.interpolate(tap, size=up.shape[-2:], mode="bilinear", align_corners=False)
            detail = self.mask_proj(tap)
        else:
            if self.spec.tap_channels is not None:
                raise RuntimeError(
                    f"{self.backbone_name} declares a stride-2 tap but produced none, "
                    f"so mask_stride=2 has no half-resolution feature to merge")
            detail = self.mask_proj(F.interpolate(
                first_rung, size=up.shape[-2:], mode="bilinear", align_corners=False))
        refined = self.mask_conv2(self.mask_conv1(torch.cat((up, detail), dim=1)))
        _guard_fp16_boundary("mask_out", refined)
        return refined

    def arch_record(self):
        """The architecture a loader needs to rebuild this model, as recorded."""
        return {
            "backbone": self.backbone_name,
            "neck_channels": self.neck_channels,
            "mask_stride": self.mask_stride,
            "amodal_outline": self.amodal_outline,
        }

    def forward(self, x, return_outline=False):
        """The segmentation logits; with ``return_outline``, ``(seg, outline)``.

        The flag is a keyword here rather than a second method so a DDP-wrapped
        model still runs it through DDP's own forward -- a separate method would
        skip gradient sync. ``outline`` is ``[B, 1, H, W]`` on the same grid as
        the segmentation logits.
        """
        return self.forward_internal(x, return_outline=return_outline)
    
    def get_backbone_names(self):
        return [name for name, _ in self.decoder_resnet50s.named_parameters()]
    
    @staticmethod
    def convert_batchnorm_to_groupnorm(module):
        for name, child in module.named_children():
            if isinstance(child, nn.BatchNorm2d):
                setattr(module, name, group_norm(child.num_features))
            else:
                hcm_nn.convert_batchnorm_to_groupnorm(child)


def checkpoint_residual_block(block):
    class CheckpointWrapper(nn.Module):
        def __init__(self, block):
            super().__init__()
            self.block = block

        def forward(self, x):
            return self.block(x)

    return CheckpointWrapper(block)




        