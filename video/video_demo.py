import argparse
import os
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from jetson.input_size import InputSizeError, load_metadata, resolve_input_size
from jetson.demo_model import demo_forward, load_demo_model, require_fp16_device


# -----------------------
# Config
# -----------------------
# The input size is not fixed here: it is resolved from the model's metadata
# below, the same way jetson/realtime_demo.py does. See jetson/input_size.py.
# Each path is overridable from the environment; the defaults are the Jetson's.
def _parse_args(argv=None):
    """The flags; everything else is configured through ECS_HCM_DEMO_* env vars."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-stride", type=int, default=None,
                    help="Backbone output stride (8/16/32); overrides "
                         "ECS_HCM_DEMO_OUT_STRIDE and the model's recorded stride")
    ap.add_argument("--fp16", action="store_true",
                    help="float16 autocast inference (CUDA only; default fp32)")
    return ap.parse_args(argv)


ARGS = _parse_args()

ART_DIR = Path(os.environ.get("ECS_HCM_DEMO_ART_DIR", "/home/user/models")).expanduser()
WEIGHTS = ART_DIR / "model.pth"          # rename if needed
META    = ART_DIR / "metadata.json"      # optional

FRAMES_DIR = Path(os.environ.get("ECS_HCM_DEMO_FRAMES_DIR", "~/clustering/videos/frames")).expanduser()
OUT_DIR    = Path(os.environ.get("ECS_HCM_DEMO_OUT_DIR", "~/clustering/videos/frames_out")).expanduser()

ALPHA = 0.45
THRESH = 0.7
SHOW_FPS = True

# ImageNet normalization (ResNet50 pretrained expects this)
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


# -----------------------
# Helpers
# -----------------------
def preprocess_bgr(frame_bgr: np.ndarray, input_size: int) -> torch.Tensor:
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)  # pylint: disable=no-member
    rgb = cv2.resize(rgb, (input_size, input_size), interpolation=cv2.INTER_LINEAR)  # pylint: disable=no-member

    x = rgb.astype(np.float32) / 255.0
    x = (x - IMAGENET_MEAN) / IMAGENET_STD
    x = np.transpose(x, (2, 0, 1))  # HWC -> CHW
    x = torch.from_numpy(x).unsqueeze(0)  # [1,3,H,W]
    return x

def list_frames(frames_dir: Path):
    exts = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
    paths = [p for p in frames_dir.iterdir() if p.suffix.lower() in exts]
    paths.sort(key=lambda p: p.name)
    return paths


# -----------------------
# Load model
# -----------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", device)
try:
    require_fp16_device(ARGS.fp16, device)
except ValueError as e:
    raise SystemExit(str(e)) from e
print("precision:", "fp16 (autocast)" if ARGS.fp16 else "fp32")

# The size the weights were trained at; a stated size that cannot be used, or
# a metadata.json that is present but unreadable, stops the demo rather than
# silently falling back to 192. A missing metadata.json is still fine.
try:
    meta = load_metadata(META)
    if meta is not None:
        print("loaded metadata keys:", list(meta.keys())[:10])
    input_size, input_size_source = resolve_input_size(meta, os.environ)
except InputSizeError as e:
    raise SystemExit(f"Cannot resolve the model input size: {e}") from e

model, _ = load_demo_model(
    WEIGHTS, meta, cli_stride=ARGS.out_stride, env=os.environ, device=device,
    input_size=input_size, input_size_source=input_size_source)

# warmup
dummy = torch.zeros(1, 3, input_size, input_size, device=device, dtype=torch.float32)
_ = demo_forward(model, dummy, fp16=ARGS.fp16)


# -----------------------
# Inference loop over frames
# -----------------------
frames = list_frames(FRAMES_DIR)
if not frames:
    raise RuntimeError(f"No frames found in {FRAMES_DIR}")

print(f"Found {len(frames)} frames in {FRAMES_DIR}")
# Created only once there is something to write, not when the module loads.
OUT_DIR.mkdir(parents=True, exist_ok=True)
fps = 0.0
t_prev = time.time()

# Visualization knobs. P_MIN is 0.2 here and 0.5 in jetson/realtime_demo.py,
# so this demo's heat overlay starts fainter than the realtime one.
FG_CLASS = 1
P_MIN = 0.2
HEAT_ALPHA = 0.35
P_THRESH = 0.65

kernel = np.ones((3, 3), np.uint8)
skipped_read = skipped_write = 0

for idx, frame_path in enumerate(frames, start=1):
    frame = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)  # pylint: disable=no-member
    if frame is None:
        print("WARN: failed to read", frame_path)
        skipped_read += 1
        continue

    x = preprocess_bgr(frame, input_size).to(device=device, dtype=torch.float32)

    logits = demo_forward(model, x, fp16=ARGS.fp16)

    if logits.ndim != 4 or logits.shape[1] != 2:
        raise RuntimeError(f"Expected logits [1,2,h,w], got {tuple(logits.shape)}")

    if logits.shape[-2:] != (input_size, input_size):
        logits_up = F.interpolate(
            logits, size=(input_size, input_size), mode="bilinear", align_corners=False
        )
    else:
        logits_up = logits

    probs = torch.softmax(logits_up, dim=1)  # [1,2,H,W]
    p_fg = probs[0, FG_CLASS]  # [H,W]

    # Heatmap (confidence gated)
    p_vis = torch.clamp((p_fg - P_MIN) / (1.0 - P_MIN), 0.0, 1.0)
    p_u8 = (p_vis * 255.0).to(torch.uint8).cpu().numpy()  # [H,W]
    p_u8 = cv2.resize(p_u8, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_LINEAR)  # pylint: disable=no-member
    heat = cv2.applyColorMap(p_u8, cv2.COLORMAP_JET)  # pylint: disable=no-member
    out = cv2.addWeighted(frame, 1.0 - HEAT_ALPHA, heat, HEAT_ALPHA, 0.0)  # pylint: disable=no-member

    # Contours for confident regions
    mask_u8 = (p_fg > P_THRESH).to(torch.uint8).cpu().numpy() * 255
    mask_u8 = cv2.resize(mask_u8, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)  # pylint: disable=no-member
    mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_OPEN, kernel, iterations=1)   # pylint: disable=no-member
    mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, kernel, iterations=1)  # pylint: disable=no-member

    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # pylint: disable=no-member
    cv2.drawContours(out, contours, -1, (255, 255, 255), 2)  # pylint: disable=no-member

    score = float(p_fg.max().detach().cpu())
    cv2.putText(out, f"max p_fg: {score:.2f}", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)  # pylint: disable=no-member

    if SHOW_FPS:
        t_now = time.time()
        dt = max(t_now - t_prev, 1e-6)
        t_prev = t_now
        fps = 0.9 * fps + 0.1 * (1.0 / dt)
        cv2.putText(out, f"FPS: {fps:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)  # pylint: disable=no-member

    out_path = OUT_DIR / frame_path.name
    ok = cv2.imwrite(str(out_path), out)  # pylint: disable=no-member
    if not ok:
        print("WARN: failed to write", out_path)
        skipped_write += 1

    if idx % 100 == 0:
        print(f"Processed {idx}/{len(frames)} frames...")

print(f"Done. Wrote overlays to: {OUT_DIR} "
      f"({len(frames) - skipped_read - skipped_write}/{len(frames)} frames; "
      f"skipped {skipped_read} unreadable, {skipped_write} unwritten)")
