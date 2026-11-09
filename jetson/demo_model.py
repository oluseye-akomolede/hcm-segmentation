"""The model load every demo runs: realtime, video and image.

No cv2 here, so the load the demos run can be tested where cv2 is absent.
"""
import torch

from ecs_hcm_model.core.hcm_nn import hcm_nn
from jetson.model_shape import (
    DEFAULT_BACKBONE,
    MASK_STRIDE_DEFAULT,
    ModelShapeError,
    resolve_amodal_outline,
    resolve_backbone,
    resolve_decoder_arch,
    resolve_mask_stride,
    resolve_neck_channels,
    require_two_classes,
    resolve_demo_out_stride,
)


def unwrap_state_dict(ckpt):
    if isinstance(ckpt, dict):
        for k in ("state_dict", "model_state_dict", "model"):
            if k in ckpt and isinstance(ckpt[k], dict):
                return ckpt[k]
    return ckpt


def infer_num_classes_from_state(sd: dict) -> int:
    for k, v in sd.items():
        if k.endswith("segmentation_head.weight") and hasattr(v, "shape"):
            return int(v.shape[0])
    raise RuntimeError("Could not infer num_segmentation_classes (missing segmentation_head.weight in checkpoint).")


def load_demo_model(weights_path, meta, *, cli_stride, env, device, input_size, input_size_source):
    """Return ``(model, num_classes)``: built at the checkpoint's stride and
    decoder version, strictly loaded, in eval mode on ``device``."""
    ckpt = torch.load(str(weights_path), map_location="cpu")
    state_dict = unwrap_state_dict(ckpt)
    num_classes = infer_num_classes_from_state(state_dict)
    print("inferred num_segmentation_classes:", num_classes)
    # A non-binary head, or a stated stride the backbone cannot build, stops
    # the demo at load instead of drawing a wrong overlay.
    try:
        require_two_classes(num_classes)
        out_stride, out_stride_source, stride_warning = resolve_demo_out_stride(
            cli_stride, env, meta, ckpt)
        if stride_warning:
            print(f"WARNING: {stride_warning}")
        print(f"input size: {input_size} (from {input_size_source}), "
              f"out_stride: {out_stride} (from {out_stride_source})")
        decoder_arch, decoder_arch_source = resolve_decoder_arch(meta, state_dict)
        print(f"decoder: {decoder_arch} (from {decoder_arch_source})")
        backbone, backbone_source = resolve_backbone(meta, ckpt)
        print(f"backbone: {backbone} (from {backbone_source})")
        neck_channels, neck_source = resolve_neck_channels(meta, ckpt)
        print(f"neck_channels: {neck_channels} (from {neck_source})")
        mask_stride, mask_source = resolve_mask_stride(meta, ckpt)
        print(f"mask_stride: {mask_stride} (from {mask_source})")
        amodal_outline, outline_source = resolve_amodal_outline(meta, ckpt)
        print(f"amodal_outline: {amodal_outline} (from {outline_source})")
    except ModelShapeError as e:
        raise SystemExit(f"Cannot use this checkpoint: {e}") from e

    model = build_demo_model(num_classes, out_stride, decoder_arch, device,
                             state_dict=state_dict, backbone=backbone,
                             neck_channels=neck_channels, mask_stride=mask_stride,
                             amodal_outline=amodal_outline)
    return model, num_classes


def build_demo_model(num_classes, out_stride, decoder_arch, device, state_dict=None,
                     backbone=None, neck_channels=None, mask_stride=None,
                     amodal_outline=False):
    """The demo's model build: strictly loaded when ``state_dict`` is given
    (random weights otherwise, for benchmarking), eval mode, on ``device``.

    ``backbone`` / ``neck_channels`` / ``mask_stride`` / ``amodal_outline`` are
    the architecture the checkpoint recorded (``None`` / ``False`` keep each
    default, which is the pre-change model). ``backbone_pretrained=False``: the
    weights being loaded are the checkpoint's, so nothing here fetches anything,
    and a benchmark build (no state dict) must not either.
    """
    model = hcm_nn(enable_finetuning=True, num_segmentation_classes=num_classes, out_stride=out_stride,
                   decoder_arch=decoder_arch,
                   backbone=DEFAULT_BACKBONE if backbone is None else backbone,
                   neck_channels=neck_channels,
                   mask_stride=MASK_STRIDE_DEFAULT if mask_stride is None else mask_stride,
                   amodal_outline=amodal_outline,
                   backbone_pretrained=False)
    if state_dict is not None:
        model.load_state_dict(state_dict, strict=True)
    model.eval().to(device)
    model.backbone.float()
    return model


def require_fp16_device(fp16, device):
    """``--fp16`` is CUDA float16 autocast, the eval tool's fp16; refuse it
    elsewhere rather than run a mode nobody has measured."""
    if fp16 and torch.device(device).type != "cuda":
        raise ValueError(f"--fp16 needs a CUDA device (this run is on {device})")


def demo_forward(model, x, fp16=False):
    """The demos' forward: no_grad, fp32 by default. ``fp16`` (opt-in) runs it
    under float16 autocast, as the eval does, and hands back fp32 logits so
    the overlay code sees the same dtype either way."""
    with torch.no_grad():
        if fp16:
            with torch.autocast(device_type=x.device.type, dtype=torch.float16):
                return model(x).float()
        return model(x)
