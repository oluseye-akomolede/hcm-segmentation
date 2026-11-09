"""Export a demo checkpoint to ONNX and a TensorRT engine.

    PYTHONPATH=<repo> python -m jetson.export_trt <checkpoint dir> --out <dir> [--fp16]
        [--workspace-mb 1024] [--opset 17] [--verify] [--verify-images 4]
        [--verify-dir <image dir>] [--json]

The checkpoint is loaded exactly as the demo loads it (``jetson.demo_model``), so the
exported graph has the decoder version, stride and input size that checkpoint records. The
engine is built for a fixed batch of 1 at that size -- the geometry the demo runs. What is
written into ``--out`` is ``model.onnx``, ``model.engine`` and ``engine.json``; an existing
``--out`` is refused rather than overwritten.

``--verify`` runs seeded random input -- and the images in ``--verify-dir``, when one is
given, preprocessed exactly as the demo preprocesses a camera frame -- through both the
torch model and the engine, both in fp32: the engine is built with TF32 refused, and the
torch reference has its own TF32 turned off around it (see ``_fp32_reference``), so the
gate measures the engine rather than the reference's precision. It fails on any non-finite
engine output, when fewer than
99.5% of pixels agree on their argmax class, or when the maximum absolute logits
difference exceeds 0.25. The foreground-mask IoU is reported but does not gate: the
smallest mask measured is 6 of 7 union pixels, which score 0.8571 IoU however correct the
engine is, so IoU is noise on the small masks this model produces. An engine that is fast
but different is not an engine you can ship.

The ONNX package and the TensorRT Python bindings are imported inside the functions, so
this module imports on a host that has neither. ``export`` checks both before it touches
the output directory and refuses with ``EngineUnavailable`` (exit 2) naming the one that
is missing, so a failed export leaves nothing behind for the next run to trip over. An
export that refuses after it has begun writing -- a failed ``--verify``, or a failure in
the builder -- likewise leaves no ``--out``: the export writes into a temporary directory
beside ``--out`` and moves it into place only once every step has succeeded.

The legacy tracer loses the spatial extent of a tensor that has passed through GroupNorm,
so ``adaptive_avg_pool2d`` in the model has no input size to size its bins from and the
export fails with ``SymbolicValueError``. ``export_onnx`` therefore patches
``F.adaptive_avg_pool2d`` with a static-shape equivalent for the duration of the export
call only, and always restores the original (see ``_static_pool_for_export``).

TensorRT 10.3 builds a *wrong* fp32 engine for the v2 decoder. A per-tensor scan against
ONNX Runtime on the folded graph puts it in ``hcm_rrd``: everything up to the two
GroupNorm outputs matches to 5e-5, and the channel concat that joins them -- whose inputs
are exactly those two tensors -- is off by 6.18. The second GroupNorm chain is mis-written
when TensorRT places it in the concat buffer; everything downstream inherits the error.
``export_onnx`` therefore also rewrites the v2 rrd's pair of 8-group normalisations as one
16-group normalisation over the channel-concatenated tensor, for the duration of the
export call only. That is the same arithmetic, not an approximation: neither call has
affine parameters, and 16 groups of 4 channels over the 64-channel concat cut the tensor
in the same places as 8 groups of 4 over each 32-channel half. See
``_fused_rrd_groupnorm_for_export``.
"""
import argparse
import contextlib
import datetime
import io
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

import torch
import torch.nn.functional as F

from ecs_hcm_model.core.hcm_nn import DECODER_V2, hcm_rrd
from jetson import demo_model
from jetson.input_size import InputSizeError, load_metadata, resolve_input_size
from jetson.trt_engine import (
    ENGINE_FILE,
    ENGINE_JSON,
    ONNX_FILE,
    EngineError,
    EngineUnavailable,
    _import_trt,
    _torch_dtype,
    dtype_name,
    sha256_file,
)

INPUT_NAME = "input"
OUTPUT_NAME = "logits"
DEFAULT_OPSET = 17
DEFAULT_WORKSPACE_MB = 1024
VERIFY_IMAGES = 4
# The seed of the random half of --verify: fixed so two runs on the same engine report
# the same numbers.
VERIFY_SEED = 0
# What --verify gates on. The pixel-agreement floor is what catches a mask that moved;
# the logit ceiling catches an engine that is quietly wrong everywhere (the good engine
# measures ~0.04, the mis-compiled one 6.9).
PIXEL_AGREEMENT_FLOOR = 0.995
MAX_LOGIT_DIFF_CEILING = 0.25
# Reported for context only. A correct engine scores 0.8571 on the smallest mask measured
# (6 of 7 union pixels), so gating on 0.99 would reject a correct engine.
REFERENCE_MASK_IOU = 0.99
VERIFY_IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png")
# The v2 rrd normalises each 32-channel half in 8 groups; the fused rewrite normalises the
# 64-channel concat in twice as many, which is the same 4-channel groups.
RRD_GROUPS = 8


def _parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("checkpoint", type=Path, help="Dir with model.pth (and optionally metadata.json)")
    ap.add_argument("--out", type=Path, required=True,
                    help="Dir to write model.onnx/model.engine/engine.json into; must not exist")
    ap.add_argument("--fp16", action="store_true",
                    help="Build an fp16 engine (refused: this graph's fp16 engine "
                         "returns non-finite logits)")
    ap.add_argument("--workspace-mb", type=int, default=DEFAULT_WORKSPACE_MB,
                    help=f"TensorRT workspace memory pool limit in MB (default {DEFAULT_WORKSPACE_MB})")
    ap.add_argument("--opset", type=int, default=DEFAULT_OPSET,
                    help=f"ONNX opset (default {DEFAULT_OPSET})")
    ap.add_argument("--verify", action="store_true",
                    help="Compare the engine against the torch model and gate on the "
                         "agreement floors")
    ap.add_argument("--verify-images", type=int, default=VERIFY_IMAGES,
                    help=f"Random inputs to run through both models under --verify, and "
                         f"the most images to take from --verify-dir (default {VERIFY_IMAGES})")
    ap.add_argument("--verify-dir", type=Path, default=None,
                    help="Directory of jpg/png frames to verify on besides the random "
                         "inputs; each frame is preprocessed exactly as the demo "
                         "preprocesses a camera frame")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu",
                    help="Device for the torch model during --verify (default cuda)")
    ap.add_argument("--json", action="store_true", help="Print the engine.json record as one JSON object")
    return ap.parse_args(argv)


def _require_onnx() -> None:
    """Refuse when the ONNX package is absent: the graph it writes is the builder's input."""
    try:
        import onnx  # noqa: F401  pylint: disable=import-outside-toplevel,unused-import
    except ImportError as exc:
        raise EngineUnavailable(
            "the ONNX Python package is not importable on this host; the export builds the "
            f"engine from an ONNX graph it cannot write without it ({exc})"
        ) from exc


def _require_export_deps() -> None:
    """Refuse before anything is written when either half of the export is missing."""
    _require_onnx()
    _import_trt()


def _adaptive_bins(extent: int, out: int, *, dtype, device) -> torch.Tensor:
    """The ``(out, extent)`` averaging matrix for PyTorch's adaptive pooling bin rule.

    Bin ``i`` spans ``[floor(i*n/out), ceil((i+1)*n/out))`` and contributes the mean over
    that span, so for a non-divisible ``n`` the bins are uneven and neighbours can
    overlap. That rule is the operator's semantics; an interpolation or a fixed kernel
    would be a different function.

    Built in the input's ``dtype`` and on its ``device``: the means are exact in float64
    and rounded once in float32, where building in float32 and casting afterwards would
    round twice.
    """
    weights = torch.zeros(out, extent, dtype=dtype, device=device)
    for i in range(out):
        start = (i * extent) // out
        end = -((-(i + 1) * extent) // out)
        weights[i, start:end] = 1.0 / (end - start)
    return weights


def _static_adaptive_avg_pool2d(x: torch.Tensor, output_size) -> torch.Tensor:
    """``F.adaptive_avg_pool2d`` expressed with the input's extent as python ints.

    During the trace ``int(x.shape[-2])`` is concrete even where the tracer's symbolic
    shape is not, so the same per-bin means can be written as two matmuls:
    ``Ph @ x @ Pw^T``. Accepts an int or a ``(height, width)`` output size; the matrices
    follow ``x``'s dtype and device.
    """
    if isinstance(output_size, int):
        output_size = (output_size, output_size)
    out_h, out_w = output_size
    height, width = int(x.shape[-2]), int(x.shape[-1])
    pool_h = _adaptive_bins(height, out_h, dtype=x.dtype, device=x.device)
    pool_w = _adaptive_bins(width, out_w, dtype=x.dtype, device=x.device)
    return pool_h @ x @ pool_w.transpose(0, 1)


@contextlib.contextmanager
def _static_pool_for_export():
    """Swap in the static pool for the export call and always restore the original.

    ``hcm_nn`` reaches the operator as ``F.adaptive_avg_pool2d``, so patching the
    attribute on ``torch.nn.functional`` is what the tracer will call. The ``finally``
    keeps a raising export from leaving the process patched.
    """
    original = torch.nn.functional.adaptive_avg_pool2d
    torch.nn.functional.adaptive_avg_pool2d = _static_adaptive_avg_pool2d
    try:
        yield
    finally:
        torch.nn.functional.adaptive_avg_pool2d = original


@contextlib.contextmanager
def _fused_rrd_groupnorm_for_export():
    """Rewrite the v2 rrd's two GroupNorms as one, for the export call only.

    TensorRT 10.3 mis-compiles the second GroupNorm chain when it lands in the concat
    buffer, so the export traces the same arithmetic written as a single normalisation
    over the channel-concatenated tensor. The rewrite is exact -- see the module
    docstring -- and it is only *spelled differently* for the trace: the model, the
    training path and eager inference keep calling the original.

    ``nn.Module.__call__`` resolves ``self.forward`` on the type at call time, so setting
    the attribute on the class is what both the tracer and any eager caller would reach.
    The v1 decoder's forward is untouched: its concat is on the width, so the two
    normalisations it performs are not adjacent in channel order and cannot be fused.
    """
    original = hcm_rrd.forward

    def fused_forward(self, f_s, p_splus1):
        if self.decoder_arch != DECODER_V2:
            return original(self, f_s, p_splus1)
        repeated_p_s = F.interpolate(
            p_splus1, size=f_s.shape[-2:], mode="bilinear", align_corners=False)
        sig_p_s = F.sigmoid(repeated_p_s)
        # The original computes the two gated copies, normalises each, then concatenates.
        # Concatenating first and normalising once is the same thing because the group
        # boundaries fall between the two halves.
        both = torch.cat(
            (torch.mul(f_s, sig_p_s), torch.mul(f_s, torch.sub(1, sig_p_s))), dim=1)
        both = F.group_norm(both, num_groups=2 * RRD_GROUPS, eps=1e-5)
        return self.conv2d(self.RCAB(self.fuse(both)))

    hcm_rrd.forward = fused_forward
    try:
        yield
    finally:
        hcm_rrd.forward = original


def _load_checkpoint(checkpoint: Path):
    """``(model_on_cpu, size, source)`` -- the demo's own load, at the demo's size."""
    weights = checkpoint / "model.pth"
    if not weights.is_file():
        raise EngineError(f"no model.pth in {checkpoint}")
    try:
        meta = load_metadata(checkpoint / "metadata.json")
    except InputSizeError as exc:
        raise EngineError(str(exc)) from exc
    try:
        input_size, source = resolve_input_size(meta, os.environ)
    except InputSizeError as exc:
        raise EngineError(f"Cannot resolve the model input size: {exc}") from exc
    try:
        model, _ = demo_model.load_demo_model(
            weights, meta, cli_stride=None, env=os.environ, device="cpu",
            input_size=input_size, input_size_source=source)
    except SystemExit as exc:
        if isinstance(exc.code, str):
            raise EngineError(exc.code) from exc
        raise
    return model, weights, input_size


def export_onnx(model, input_size: int, onnx_path: Path, opset: int) -> None:
    """The legacy exporter: opset 17, fixed batch 1 / fixed size, no onnxscript needed.

    Two rewrites are installed for exactly this call and nothing else runs under them: the
    static pool (see the module docstring) and the fused v2 GroupNorm.
    """
    dummy = torch.zeros(1, 3, input_size, input_size, dtype=torch.float32)
    with _static_pool_for_export(), _fused_rrd_groupnorm_for_export():
        torch.onnx.export(
            model,
            dummy,
            str(onnx_path),
            opset_version=opset,
            dynamo=False,
            input_names=[INPUT_NAME],
            output_names=[OUTPUT_NAME],
            do_constant_folding=True,
            training=torch.onnx.TrainingMode.EVAL,
        )


def build_engine(onnx_path: Path, *, fp16: bool, workspace_mb: int):
    """``(engine_bytes, tensorrt_version)`` from the serialized ONNX."""
    trt = _import_trt()
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network()
    parser = trt.OnnxParser(network, logger)
    if not parser.parse(onnx_path.read_bytes()):
        errors = "; ".join(str(parser.get_error(i)) for i in range(parser.num_errors))
        raise EngineError(f"TensorRT could not parse {onnx_path}: {errors or 'unknown error'}")
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, int(workspace_mb) * (1 << 20))
    # TensorRT selects TF32 tactics for an fp32 network unless they are refused, and a TF32
    # tactic computes fp32 operands with a 10-bit mantissa -- an engine this record calls
    # "fp32" would not be fp32. Cleared unconditionally, FP16 or not: TF32 governs the fp32
    # arithmetic that is left in any build, and clearing it beside FP16 keeps one build path.
    config.clear_flag(trt.BuilderFlag.TF32)
    if fp16:
        config.set_flag(trt.BuilderFlag.FP16)
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise EngineError("TensorRT failed to build the engine")
    return bytes(serialized), trt.__version__


def engine_tensors(engine_bytes: bytes, *, input_name: str, input_size: int):
    """``(input_dtype, outputs)``, read from a context at the served shape.

    Each output carries its name, shape and dtype; the input dtype is the engine's own, so
    a consumer knows what it must feed a buffer of (an fp16 build wants fp16).
    """
    trt = _import_trt()
    runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
    engine = runtime.deserialize_cuda_engine(engine_bytes)
    if engine is None:
        raise EngineError("TensorRT could not deserialize the engine it just built")
    context = engine.create_execution_context()
    context.set_input_shape(input_name, (1, 3, input_size, input_size))
    input_dtype = dtype_name(_torch_dtype(trt, engine.get_tensor_dtype(input_name)))
    outputs = []
    for index in range(engine.num_io_tensors):
        name = engine.get_tensor_name(index)
        if engine.get_tensor_mode(name) == trt.TensorIOMode.OUTPUT:
            outputs.append({
                "name": name,
                "shape": [int(d) for d in context.get_tensor_shape(name)],
                "dtype": dtype_name(_torch_dtype(trt, engine.get_tensor_dtype(name))),
            })
    return input_dtype, outputs


def _fg_iou(a_logits: torch.Tensor, b_logits: torch.Tensor) -> float:
    """IoU of the two foreground masks (argmax over classes), as the demo thresholds them."""
    a = a_logits.argmax(dim=1).bool().flatten()
    b = b_logits.argmax(dim=1).bool().flatten()
    union = int((a | b).sum().item())
    if union == 0:
        return 1.0
    return int((a & b).sum().item()) / union


def _pixel_agreement(a_logits: torch.Tensor, b_logits: torch.Tensor) -> float:
    """The fraction of pixels whose argmax class is the same in both, as the demo reads them."""
    return float((a_logits.argmax(dim=1) == b_logits.argmax(dim=1)).float().mean().item())


def _nonfinite_count(t: torch.Tensor) -> int:
    """Elements of ``t`` that are NaN or infinite -- a poisoned output, not a small error."""
    return int((~torch.isfinite(t)).sum().item())


def _verify_image_paths(directory: Path, *, limit: int):
    """The first ``limit`` jpg/png files in ``directory``, sorted, for a stable report."""
    paths = sorted(
        path for path in Path(directory).iterdir()
        if path.is_file() and path.suffix.lower() in VERIFY_IMAGE_SUFFIXES)
    return paths[:limit]


def _load_verify_image(path: Path, *, input_size: int, device: str) -> torch.Tensor:
    """One image as the demo would have fed it to the model.

    Preprocessed by ``realtime_demo.preprocess_bgr`` rather than by a copy of it: the point
    of verifying on real frames is to verify the pipeline the demo actually runs. That
    module is imported here rather than at module scope because importing it pulls in flask
    and prints a banner, neither of which belongs in a tool whose ``--json`` output is
    parsed -- so the banner is captured, not printed.
    """
    try:
        import cv2  # pylint: disable=import-outside-toplevel
    except ImportError as exc:  # a host without OpenCV cannot decode frames at all
        raise EngineError(
            "--verify-dir needs OpenCV to decode images (pip install opencv-python), "
            "or drop the flag and verify on the random inputs"
        ) from exc
    with contextlib.redirect_stdout(io.StringIO()):
        from jetson import realtime_demo  # pylint: disable=import-outside-toplevel
    frame = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if frame is None:
        raise EngineError(f"--verify-dir: {path} could not be read as an image")
    x = realtime_demo.preprocess_bgr(frame, input_size)
    return x.to(device=device, dtype=torch.float32)


def _verify_inputs(*, input_size: int, images: int, verify_dir: Path, device: str):
    """``[(kind, label, tensor)]``: seeded random inputs first, then the directory's images.

    The random input is kept even when a directory is given: it is where a wildly wrong
    engine shows up as a logits difference, and it needs no fixture on the box.
    """
    inputs = []
    generator = torch.Generator().manual_seed(VERIFY_SEED)
    for index in range(images):
        x = torch.randn(
            1, 3, input_size, input_size, generator=generator, dtype=torch.float32)
        inputs.append(("random", f"random[{index}]", x.to(device)))
    if verify_dir is not None:
        paths = _verify_image_paths(verify_dir, limit=images)
        if not paths:
            raise EngineError(
                f"--verify-dir {verify_dir} holds no "
                f"{'/'.join(VERIFY_IMAGE_SUFFIXES)} images to verify on")
        for path in paths:
            inputs.append(
                ("image", path.name, _load_verify_image(path, input_size=input_size, device=device)))
    return inputs


def _verdict(pairs, *, verify_dir: Path) -> dict:
    """The verify report for ``[(label, ref_logits, got_logits)]``; raises when a gate fails.

    Split out from :func:`verify` so the gates can be exercised without an engine, a model
    or a CUDA device -- which is the only way they are testable off the box.
    """
    report = {
        "inputs": len(pairs),
        "random_inputs": sum(1 for kind, *_ in pairs if kind == "random"),
        "image_inputs": sum(1 for kind, *_ in pairs if kind == "image"),
        "verify_dir": str(verify_dir) if verify_dir is not None else None,
        "max_abs_logit_diff": 0.0,
        "max_logit_diff_allowed": MAX_LOGIT_DIFF_CEILING,
        "min_pixel_agreement": 1.0,
        "pixel_agreement_required": PIXEL_AGREEMENT_FLOOR,
        "min_mask_iou": 1.0,
        "mask_iou_reference": REFERENCE_MASK_IOU,
        "nonfinite_outputs": 0,
        "notes": [],
    }
    failures = []
    for kind, label, ref, got in pairs:
        # An infinite or NaN output is a failure the *numbers* cannot express: the diff is
        # NaN and the argmax is meaningless. Report the count, and measure the rest on a
        # zeroed copy so the report still carries a comparison a human can read.
        nonfinite = _nonfinite_count(got)
        report["nonfinite_outputs"] += nonfinite
        if nonfinite:
            failures.append(f"{label} ({kind}): {nonfinite} non-finite logits")
        ref, got = ref.float(), torch.nan_to_num(got.float(), nan=0.0, posinf=0.0, neginf=0.0)
        diff = float((ref - got).abs().max().item())
        agreement = _pixel_agreement(ref, got)
        report["max_abs_logit_diff"] = max(report["max_abs_logit_diff"], diff)
        report["min_pixel_agreement"] = min(report["min_pixel_agreement"], agreement)
        report["min_mask_iou"] = min(report["min_mask_iou"], _fg_iou(ref, got))
    if report["min_pixel_agreement"] < PIXEL_AGREEMENT_FLOOR:
        failures.append(
            f"only {report['min_pixel_agreement']:.4f} of pixels agree on their class, "
            f"need {PIXEL_AGREEMENT_FLOOR}")
    if report["max_abs_logit_diff"] > MAX_LOGIT_DIFF_CEILING:
        failures.append(
            f"max abs logit diff {report['max_abs_logit_diff']:.4f} is over "
            f"{MAX_LOGIT_DIFF_CEILING}")
    if verify_dir is None:
        report["notes"].append(
            "no --verify-dir: verified on seeded random input only, not on real images")
    report["passed"] = not failures
    if failures:
        error = EngineError("engine and torch model disagree: " + "; ".join(failures))
        # The numbers behind the refusal, for a caller that wants to print them: the
        # message alone says which gate tripped, not how far off the engine was.
        error.report = report
        raise error
    return report


@contextlib.contextmanager
def _fp32_reference():
    """Run torch with TF32 refused, so the reference is fp32 like the engine.

    ``build_engine`` clears ``BuilderFlag.TF32``, but torch leaves TF32 on for cuDNN --
    ``torch.backends.cudnn.allow_tf32`` is true by default -- so the model ``--verify``
    compares against computes its convolutions with a 10-bit mantissa. The gate would then
    blame the engine for a difference the reference contributed, and how far over it lands
    depends on the weights: a ResNet-50 engine measured 0.026 against a TF32 reference and
    passed, while two v2 backbones measured 0.30 and 0.72 against the same reference and
    were refused, though their engines were as faithful.

    The flags are restored to the values they held, not set back to true: their defaults
    differ between the two (cuDNN's is true, matmul's is false on recent torch), and a
    caller that chose otherwise keeps its own process.

    Not via ``NVIDIA_TF32_OVERRIDE``: TensorRT records that variable in the engine and
    refuses to run an engine whose build-time and run-time values differ, so an engine
    built under it is only servable by a process that sets it too.
    """
    previous = (torch.backends.cudnn.allow_tf32, torch.backends.cuda.matmul.allow_tf32)
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        yield
    finally:
        torch.backends.cudnn.allow_tf32, torch.backends.cuda.matmul.allow_tf32 = previous


def verify(engine, model, *, input_size: int, images: int, device: str,
           verify_dir: Path = None) -> dict:
    """Run both models over the same inputs and gate on the report from :func:`_verdict`."""
    if torch.device(device).type != "cuda":
        raise EngineError("--verify needs a CUDA device to run the engine (pass --device cuda)")
    model = model.to(device)
    pairs = []
    with _fp32_reference():
        for kind, label, x in _verify_inputs(
                input_size=input_size, images=images, verify_dir=verify_dir, device=device):
            ref = demo_model.demo_forward(model, x)
            pairs.append((kind, label, ref, engine.infer(x)))
    return _verdict(pairs, verify_dir=verify_dir)


def export(args) -> dict:
    # Before the output directory is even looked at: a refusal here must leave no trace,
    # or the next run is refused for overwriting the wreckage of the last one.
    _require_export_deps()
    if args.out.exists():
        raise EngineError(f"--out {args.out} already exists; refusing to overwrite it")
    if args.workspace_mb < 1:
        raise EngineError("--workspace-mb must be >= 1")
    if args.verify_images < 1:
        raise EngineError("--verify-images must be >= 1")
    if args.verify_dir is not None and not args.verify_dir.is_dir():
        raise EngineError(f"--verify-dir {args.verify_dir} is not a directory")
    if args.fp16:
        raise EngineError(
            "--fp16 is refused for this graph: an fp16 engine of this model returns "
            "non-finite logits while its fp32 engine is finite, so the export stays "
            "fp32 until that is diagnosed. Re-run without --fp16."
        )

    # Everything that writes goes into a directory of its own beside --out, moved into
    # place only once every step -- including --verify -- has succeeded. A refusal
    # halfway through (a failed verdict, a builder failure, a kill) therefore leaves no
    # --out at all, rather than the wreckage the overwrite check above would then refuse
    # to replace.
    args.out.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(dir=args.out.parent, prefix=f"{args.out.name}.partial-"))
    try:
        record = _export_into(staging, args)
        # mkdtemp creates 0700, and rename moves the inode, so without this the output
        # directory comes out owner-only where a plain mkdir gives 0o777 minus the umask
        # (the demos and other tools read these files). Done on the staging directory,
        # before the rename, so nothing after the atomic move can fail and leave --out.
        _chmod_like_mkdir(staging)
        staging.rename(args.out)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return record


def _chmod_like_mkdir(path: Path) -> None:
    """Give ``path`` the mode ``Path.mkdir`` would: 0o777 minus the process umask."""
    umask = os.umask(0)
    os.umask(umask)
    path.chmod(0o777 & ~umask)


def _export_into(out: Path, args) -> dict:
    model, weights, input_size = _load_checkpoint(args.checkpoint)
    source_sha = sha256_file(weights)

    onnx_path = out / ONNX_FILE
    engine_path = out / ENGINE_FILE

    export_onnx(model, input_size, onnx_path, args.opset)
    engine_bytes, trt_version = build_engine(
        onnx_path, fp16=args.fp16, workspace_mb=args.workspace_mb)
    engine_path.write_bytes(engine_bytes)
    input_dtype, outputs = engine_tensors(
        engine_bytes, input_name=INPUT_NAME, input_size=input_size)

    record = {
        "source_checkpoint": str(weights),
        "source_sha256": source_sha,
        "input_size": int(input_size),
        "input_name": INPUT_NAME,
        "input_dtype": input_dtype,
        "precision": "fp16" if args.fp16 else "fp32",
        # Whether the builder was allowed to use TF32 tactics; always false (see build_engine).
        # Descriptive, not a loader key: a record written before this field existed still loads.
        "tf32": False,
        "workspace_mb": int(args.workspace_mb),
        "opset": int(args.opset),
        "tensorrt_version": str(trt_version),
        "outputs": outputs,
        "decoder_arch": getattr(model, "decoder_arch", None),
        "onnx_sha256": sha256_file(onnx_path),
        "engine_sha256": sha256_file(engine_path),
        "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
    }

    verify_report = None
    if args.verify:
        engine = load_engine_from(engine_path, record)
        verify_report = verify(
            engine, model, input_size=input_size,
            images=args.verify_images, device=args.device, verify_dir=args.verify_dir)
        record["verify"] = verify_report

    (out / ENGINE_JSON).write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    return record


def load_engine_from(engine_path: Path, record: dict):
    """A TrtEngine from bytes already in hand, without re-reading the record."""
    trt = _import_trt()
    runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
    engine = runtime.deserialize_cuda_engine(engine_path.read_bytes())
    if engine is None:
        raise EngineError(f"TensorRT could not deserialize {engine_path}")
    from jetson.trt_engine import TrtEngine  # local import: keeps the module import cheap
    return TrtEngine(record, runtime=runtime, engine=engine,
                     context=engine.create_execution_context())


def format_report(record: dict) -> str:
    lines = [
        f"checkpoint {record['source_checkpoint']} (sha256 {record['source_sha256'][:12]}...)",
        f"input size {record['input_size']}, precision {record['precision']}, "
        f"opset {record['opset']}, TensorRT {record['tensorrt_version']}",
        f"input {record['input_name']} ({record['input_dtype']})",
        f"outputs: " + ", ".join(
            f"{o['name']}{tuple(o['shape'])} ({o['dtype']})" for o in record["outputs"]),
    ]
    if record.get("verify"):
        v = record["verify"]
        sources = f"{v['random_inputs']} random"
        if v["image_inputs"]:
            sources += f" + {v['image_inputs']} image"
        lines.append(
            f"verify: {sources} input(s), max abs logit diff "
            f"{v['max_abs_logit_diff']:.4f} (limit {v['max_logit_diff_allowed']}), "
            f"min pixel agreement {v['min_pixel_agreement']:.4f} "
            f"(need {v['pixel_agreement_required']}), "
            f"min mask IoU {v['min_mask_iou']:.4f} (reference {v['mask_iou_reference']}, "
            f"not gated), non-finite {v['nonfinite_outputs']}")
        for note in v.get("notes", []):
            lines.append(f"verify note: {note}")
    return "\n".join(lines)


def main(argv=None) -> int:
    args = _parse_args(argv)
    try:
        record = export(args)
    except EngineError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(record) if args.json else format_report(record))
    return 0


if __name__ == "__main__":
    sys.exit(main())
