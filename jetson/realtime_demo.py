import argparse
import atexit
import os
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from jetson.capture import CameraReadError, RateLimitedWarning, read_with_retry
from jetson.capture_session import CaptureConfigError, build_capture
from jetson.demo_stop import StopRequest
from jetson.frame_stream import STREAM, mjpeg_part
from jetson.input_size import InputSizeError, load_metadata, resolve_input_size
from jetson.demo_model import demo_forward, load_demo_model, require_fp16_device
from jetson.trt_engine import EngineError, TrtEngine, sha256_file

# ---- at top ----
from flask import Flask, Response
import threading

app = Flask(__name__)

def gen():
    """Yield the multipart body, one part per newly produced frame.

    Reads from the shared stream rather than a module global, so a viewer
    blocks until the capture loop publishes a frame instead of re-sending the
    newest one in a busy loop. ``frames_since(0)`` starts a viewer at the first
    frame published after it connected.
    """
    for _seq, frame in STREAM.frames_since(0):
        yield mjpeg_part(frame)


@app.route("/video")
def video():
    return Response(gen(), mimetype="multipart/x-mixed-replace; boundary=frame")

def start_server():
    app.run(host="0.0.0.0", port=5000, debug=False, use_reloader=False)


# ---- before your capture loop ----
# The server thread is started by main(), not at import: importing this module
# must not bind a port, choose a device, load weights or open a camera.
print("Open http://<JETSON_IP>:5000/video")


# -----------------------
# Config
# -----------------------
CAM_INDEX = 0
# The input size is not fixed here: it is resolved at startup from the model's
# own metadata, so a checkpoint trained at another size is not silently fed 192
# pixel frames. See jetson/input_size.py.
# Overridable for a different deploy layout; the default is the Jetson's.
ART_DIR = Path(os.environ.get("ECS_HCM_DEMO_ART_DIR", "/home/user/models")).expanduser()
WEIGHTS = ART_DIR / "model.pth"          # rename if needed
META    = ART_DIR / "metadata.json"      # optional; used if you want

SHOW_FPS = True

# ImageNet normalization (ResNet50 pretrained expects this)
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# -----------------------
# Helpers
# -----------------------
def get_fourcc():
    """
    Jetson OpenCV sometimes lacks cv2.VideoWriter_fourcc.
    Use cv2.VideoWriter.fourcc if available; otherwise return None.
    """
    if hasattr(cv2, "VideoWriter") and hasattr(cv2.VideoWriter, "fourcc"): # pylint: disable=no-member
        return cv2.VideoWriter.fourcc(*"MJPG") # pylint: disable=no-member
    # fallback: don't set FOURCC
    return None

def preprocess_bgr(frame_bgr: np.ndarray, input_size: int) -> torch.Tensor:
    # BGR -> RGB
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)# pylint: disable=no-member
    # The whole frame is resized to a square: the demo shows the full field of
    # view, so it takes no crop, unlike the training pipeline's window.
    rgb = cv2.resize(rgb, (input_size, input_size), interpolation=cv2.INTER_LINEAR)# pylint: disable=no-member

    x = rgb.astype(np.float32) / 255.0
    x = (x - IMAGENET_MEAN) / IMAGENET_STD
    x = np.transpose(x, (2, 0, 1))  # HWC -> CHW
    x = torch.from_numpy(x).unsqueeze(0)  # [1,3,H,W]
    return x

def _parse_args(argv=None):
    """The flags; everything else is configured through ECS_HCM_DEMO_* env vars."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-stride", type=int, default=None,
                    help="Backbone output stride (8/16/32); overrides "
                         "ECS_HCM_DEMO_OUT_STRIDE and the model's recorded stride")
    ap.add_argument("--fp16", action="store_true",
                    help="float16 autocast inference (CUDA only; default fp32)")
    ap.add_argument("--engine", type=Path, default=None,
                    help="Dir written by jetson/export_trt.py; run its TensorRT engine "
                         "instead of the torch forward (CUDA only)")
    ap.add_argument("--capture-uncertain", type=Path, default=None, metavar="DIR",
                    help="Save raw frames whose foreground probability is undecided "
                         "into DIR and upload them to MinIO (off by default; needs "
                         "MINIO_ENDPOINT, MINIO_USER, MINIO_PASSWORD)")
    return ap.parse_args(argv)


def main():
    args = _parse_args()
    # A demo started as a background job inherits SIGINT ignored, and CPython
    # then installs no handler at all. Ours goes in before anything expensive,
    # so a stop that arrives during the model or engine load is held by the
    # interpreter and acted on when that call returns -- not discarded at
    # delivery, which is what a start-up that had to be killed looked like.
    stop = StopRequest().install()
    # Capture is built first, so a missing MinIO variable stops the demo before
    # it serves a feed whose frames would never leave the device.
    capture = None
    if args.capture_uncertain is not None:
        try:
            weights_sha = sha256_file(WEIGHTS) if WEIGHTS.exists() else None
            capture = build_capture(
                args.capture_uncertain,
                context={"model": WEIGHTS.name, "model_sha256": weights_sha,
                         "input_size": None},
            )
        except CaptureConfigError as e:
            raise SystemExit(f"Cannot start capture: {e}") from e
        # The queue must not be lost when the demo is interrupted.
        atexit.register(capture.stop)
    # Start serving before the model is built, so the port is reachable while
    # the first (slow) load happens and the browser sees the stream as soon as
    # the first frame exists.
    threading.Thread(target=start_server, daemon=True).start()

    # -----------------------
    # Load model
    # -----------------------
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device)
    try:
        require_fp16_device(args.fp16, device)
    except ValueError as e:
        raise SystemExit(str(e)) from e
    print("precision:", "fp16 (autocast)" if args.fp16 else "fp32")

    # The size the weights were trained at, from the environment first and then
    # the model's own metadata, so the frames this demo feeds the model match
    # the frames it was trained on. A stated size that cannot be used -- or a
    # metadata.json that is present but unreadable -- stops the demo rather
    # than being quietly replaced by the default: running at the wrong size
    # draws a plausible overlay computed at the wrong resolution. A missing
    # metadata.json is still fine.
    try:
        meta = load_metadata(META)
        if meta is not None:
            print("loaded metadata keys:", list(meta.keys())[:10])
        input_size, input_size_source = resolve_input_size(meta, os.environ)
    except InputSizeError as e:
        raise SystemExit(f"Cannot resolve the model input size: {e}") from e
    if capture is not None:
        capture.set_context(input_size=input_size)

    model, _ = load_demo_model(
        WEIGHTS, meta, cli_stride=args.out_stride, env=os.environ, device=device,
        input_size=input_size, input_size_source=input_size_source)

    #for name, m in model.named_children():
    #    if name != "backbone":
    #        m.half()

    # An engine, when one is given, replaces the forward only: the same input
    # size and the same checkpoint it was built from are required, so a stale or
    # wrong-size engine is refused rather than drawing a plausible overlay.
    engine = None
    if args.engine is not None:
        if device != "cuda":
            raise SystemExit(f"--engine needs a CUDA device (this run is on {device})")
        if args.fp16:
            raise SystemExit("--fp16 does not apply to --engine: the engine's precision is fixed at export")
        try:
            engine = TrtEngine.load(args.engine, input_size=input_size,
                                    source_sha256=sha256_file(WEIGHTS))
        except EngineError as e:
            raise SystemExit(str(e)) from e
        print(f"engine: {args.engine} (input size {engine.input_size}, {engine.precision})")

    # warmup
    dummy = torch.zeros(1, 3, input_size, input_size, device=device, dtype=torch.float32)
    if engine is not None:
        _ = engine.infer(dummy)
    else:
        _ = demo_forward(model, dummy, fp16=args.fp16)

    # -----------------------
    # Camera
    # -----------------------
    cap = cv2.VideoCapture(CAM_INDEX) # pylint: disable=no-member
    fourcc = get_fourcc()
    if fourcc is not None:
        cap.set(cv2.CAP_PROP_FOURCC, fourcc) # pylint: disable=no-member

    # Optional: request a smaller camera frame to reduce USB bandwidth / display load
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640) # pylint: disable=no-member
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480) # pylint: disable=no-member
    cap.set(cv2.CAP_PROP_FPS, 30) # pylint: disable=no-member

    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera index {CAM_INDEX}")

    fps = 0.0
    t_prev = time.time()

    encode_failed = RateLimitedWarning("JPEG encode failed; frame not served")

    print("Press 'q' or ESC to quit.")
    while not stop.requested:
        # A few bad reads are retried; a camera that stays gone exits nonzero
        # so a supervisor sees a failure, not a clean stop.
        try:
            frame = read_with_retry(cap.read)
        except CameraReadError as e:
            cap.release()
            raise SystemExit(f"Stopping: {e}") from e

        x = preprocess_bgr(frame, input_size).to(device=device, dtype=torch.float32)

        # The same forward bench_model.py times; the engine when one is given.
        if engine is not None:
            logits = engine.infer(x)
        else:
            logits = demo_forward(model, x, fp16=args.fp16)   # enable_finetuning=True => segmentation logits

        # ---- Heatmap + contour overlay (confidence-gated) ----
        if logits.ndim != 4 or logits.shape[1] != 2:
            raise RuntimeError(f"Expected logits [1,2,h,w], got {tuple(logits.shape)}")

        # Upsample logits to model input resolution
        if logits.shape[-2:] != (input_size, input_size):
            logits_up = F.interpolate(
                logits,
                size=(input_size, input_size),
                mode="bilinear",
                align_corners=False,
            )
        else:
            logits_up = logits

        probs = torch.softmax(logits_up, dim=1)  # [1,2,H,W]

        FG_CLASS = 1  # confirmed from your test
        p_fg = probs[0, FG_CLASS]  # [H,W] in [0,1]

        # Capture scores the frame it is about to draw on, from this same p_fg:
        # no second forward, and the score cannot disagree with the overlay.
        # `frame` is untouched at this point, so the saved file is the raw image.
        if capture is not None:
            capture.offer(frame, p_fg, fps=fps)

        # --- Option 1: suppress low-confidence areas for nicer visuals ---
        P_MIN = 0.5  # below this, heat fades to 0
        p_vis = torch.clamp((p_fg - P_MIN) / (1.0 - P_MIN), 0.0, 1.0)

        # Heatmap image (uint8) at camera resolution
        p_u8 = (p_vis * 255.0).to(torch.uint8).cpu().numpy()  # [H,W]
        p_u8 = cv2.resize(p_u8, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_LINEAR)  # pylint: disable=no-member

        heat = cv2.applyColorMap(p_u8, cv2.COLORMAP_JET)  # pylint: disable=no-member

        HEAT_ALPHA = 0.35
        out = cv2.addWeighted(frame, 1.0 - HEAT_ALPHA, heat, HEAT_ALPHA, 0.0)  # pylint: disable=no-member

        # --- Option 2: contour for confident foreground regions ---
        P_THRESH = 0.65
        mask_u8 = (p_fg > P_THRESH).to(torch.uint8).cpu().numpy() * 255  # [H,W] 0/255
        mask_u8 = cv2.resize(mask_u8, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)  # pylint: disable=no-member

        # Clean speckles
        kernel = np.ones((3, 3), np.uint8)
        mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_OPEN, kernel, iterations=1)   # pylint: disable=no-member
        mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, kernel, iterations=1)  # pylint: disable=no-member

        contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # pylint: disable=no-member
        cv2.drawContours(out, contours, -1, (255, 255, 255), 2)  # pylint: disable=no-member

        # Optional: label a “confidence” number for the frame
        score = float(p_fg.max().detach().cpu())
        cv2.putText(out, f"max p_fg: {score:.2f}", (10, 60),  # pylint: disable=no-member
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)  # pylint: disable=no-member

        # FPS
        if SHOW_FPS:
            t_now = time.time()
            dt = max(t_now - t_prev, 1e-6)
            t_prev = t_now
            fps = 0.9 * fps + 0.1 * (1.0 / dt)
            cv2.putText(out, f"FPS: {fps:.1f}", (10, 30), # pylint: disable=no-member
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2) # pylint: disable=no-member

        #cv2.imshow("HCM Realtime Segmentation Overlay", out) # pylint: disable=no-member
        #key = cv2.waitKey(1) & 0xFF # pylint: disable=no-member
        #if key in (ord("q"), 27):
        #    break

        ok_jpg, buf = cv2.imencode(".jpg", out, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        if ok_jpg:
            # Hand the encoded frame to the viewers. Each one receives it at
            # most once; the previous frame is dropped, so a viewer slower than
            # the camera sees the current frame and never a backlog.
            STREAM.publish(buf.tobytes())
        else:
            encode_failed()


    if stop.requested:
        # Leaving normally, not by exception: the capture session's atexit hook
        # then flushes its queue and prints the counters the harness reads.
        print(f"stopping on {stop.name}", flush=True)
    cap.release()
    cv2.destroyAllWindows() # pylint: disable=no-member


if __name__ == "__main__":
    main()

