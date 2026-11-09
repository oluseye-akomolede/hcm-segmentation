#!/usr/bin/env python3
"""
Infer on a single image and write an overlayed output image.

Usage:
  python infer_one_image.py \
    --image /path/to/input.png \
    --weights /home/user/models/model.pth \
    --out /path/to/output.png \
    --device auto

The input size comes from --input-size if given, else from the metadata.json
next to --weights (or ECS_HCM_DEMO_INPUT_SIZE), else 192; see
jetson/input_size.py. A metadata.json that is present but unreadable stops
the demo.
"""

import argparse
import json
import os
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from jetson.input_size import InputSizeError, load_metadata, resolve_with_cli
from jetson.demo_model import demo_forward, load_demo_model, require_fp16_device


# -----------------------
# Defaults / constants
# -----------------------
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

ALPHA = 0.55          # heat overlay
P_MIN = 0.02          # heatmap starts showing above this
P_THRESH = 0.25       # contour threshold
FG_CLASS = 1          # binary foreground class index


# -----------------------
# Model helpers
# -----------------------
def load_model(weights_path: Path, device: str, input_hw: int, meta,
               cli_stride=None, input_size_source="?", fp16=False) -> torch.nn.Module:
    model, num_classes = load_demo_model(
        weights_path, meta, cli_stride=cli_stride, env=os.environ, device=device,
        input_size=input_hw, input_size_source=input_size_source)

    # warmup
    dummy = torch.zeros(1, 3, input_hw, input_hw, device=device, dtype=torch.float32)
    _ = demo_forward(model, dummy, fp16=fp16)

    return model, num_classes

def preprocess_bgr(frame_bgr: np.ndarray, input_hw: int) -> torch.Tensor:
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)  # pylint: disable=no-member
    rgb = cv2.resize(rgb, (input_hw, input_hw), interpolation=cv2.INTER_LINEAR)  # pylint: disable=no-member

    x = rgb.astype(np.float32) / 255.0
    x = (x - IMAGENET_MEAN) / IMAGENET_STD
    x = np.transpose(x, (2, 0, 1))  # HWC -> CHW
    x = torch.from_numpy(x).unsqueeze(0)  # [1,3,H,W]
    return x

def overlay_heat_and_contours(frame_bgr: np.ndarray, logits: torch.Tensor, input_hw: int) -> np.ndarray:
    """
    logits: [1,2,h,w] (binary foreground/background)
    returns: frame-sized BGR image with heatmap + contours.
    """
    if logits.ndim != 4 or logits.shape[1] != 2:
        raise RuntimeError(f"Expected logits [1,2,h,w], got {tuple(logits.shape)}")

    # Upsample logits to model input resolution (input_hw,input_hw)
    if logits.shape[-2:] != (input_hw, input_hw):
        logits_up = F.interpolate(logits, size=(input_hw, input_hw), mode="bilinear", align_corners=False)
    else:
        logits_up = logits

    probs = torch.softmax(logits_up, dim=1)  # [1,2,H,W]
    p_fg = probs[0, FG_CLASS]                # [H,W]

    # Heat visualization
    p_vis = torch.clamp((p_fg - P_MIN) / (1.0 - P_MIN), 0.0, 1.0)
    p_u8 = (p_vis * 255.0).to(torch.uint8).cpu().numpy()
    p_u8 = cv2.resize(p_u8, (frame_bgr.shape[1], frame_bgr.shape[0]), interpolation=cv2.INTER_LINEAR)  # pylint: disable=no-member
    heat = cv2.applyColorMap(p_u8, cv2.COLORMAP_JET)  # pylint: disable=no-member
    out = cv2.addWeighted(frame_bgr, 1.0 - ALPHA, heat, ALPHA, 0.0)  # pylint: disable=no-member

    # Contours on confident foreground
    mask_u8 = (p_fg > P_THRESH).to(torch.uint8).cpu().numpy() * 255
    mask_u8 = cv2.resize(mask_u8, (frame_bgr.shape[1], frame_bgr.shape[0]), interpolation=cv2.INTER_NEAREST)  # pylint: disable=no-member

    kernel = np.ones((3, 3), np.uint8)
    mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_OPEN, kernel, iterations=1)   # pylint: disable=no-member
    mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, kernel, iterations=1)  # pylint: disable=no-member

    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # pylint: disable=no-member
    cv2.drawContours(out, contours, -1, (255, 255, 255), 2)  # pylint: disable=no-member

    #score = float(p_fg.max().detach().cpu())
    #cv2.putText(out, f"max p_fg: {score:.2f}", (10, 60),
    #            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)  # pylint: disable=no-member

    return out


# -----------------------
# Main
# -----------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True, help="Input image path (png/jpg)")
    ap.add_argument("--weights", required=True, help="Model checkpoint .pth")
    ap.add_argument("--out", required=True, help="Output image path")
    ap.add_argument("--input-size", type=int, default=None,
                    help="Model input H=W (default: the model's metadata.json, else 192)")
    ap.add_argument("--out-stride", type=int, default=None,
                    help="Backbone output stride (8/16/32); overrides "
                         "ECS_HCM_DEMO_OUT_STRIDE and the model's recorded stride")
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"], help="Device")
    ap.add_argument("--fp16", action="store_true",
                    help="float16 autocast inference (CUDA only; default fp32)")
    args = ap.parse_args()

    img_path = Path(args.image)
    weights = Path(args.weights)
    out_path = Path(args.out)

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    print("device:", device)
    try:
        require_fp16_device(args.fp16, device)
    except ValueError as e:
        raise SystemExit(str(e)) from e
    print("precision:", "fp16 (autocast)" if args.fp16 else "fp32")
    print("image:", img_path)
    print("weights:", weights)

    # The size the weights were trained at, unless overridden on the command
    # line; a stated size that cannot be used, or a metadata.json next to the
    # weights that is present but unreadable, stops the demo.
    try:
        meta = load_metadata(weights.with_name("metadata.json"))
        input_size, input_size_source = resolve_with_cli(args.input_size, meta, os.environ)
    except InputSizeError as e:
        raise SystemExit(f"Cannot resolve the model input size: {e}") from e

    frame = cv2.imread(str(img_path), cv2.IMREAD_COLOR)  # pylint: disable=no-member
    if frame is None:
        raise RuntimeError(f"Could not read image: {img_path}")

    model, _ = load_model(weights, device, input_size, meta,
                          cli_stride=args.out_stride, input_size_source=input_size_source,
                          fp16=args.fp16)

    x = preprocess_bgr(frame, input_size).to(device=device, dtype=torch.float32)

    logits = demo_forward(model, x, fp16=args.fp16)

    out = overlay_heat_and_contours(frame, logits, input_size)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(out_path), out)  # pylint: disable=no-member
    if not ok:
        raise RuntimeError(f"Failed to write output image: {out_path}")

    print("wrote:", out_path)


if __name__ == "__main__":
    main()
