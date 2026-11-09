"""Time the realtime demo's model at one input size, v1 or v2 decoder.

    PYTHONPATH=<repo> python -m jetson.bench_model --arch v2 [--input-size 352] [--checkpoint DIR]
                                         [--warmup 10] [--iters 100] [--device cuda] [--fp16] [--json]
    PYTHONPATH=<repo> python -m jetson.bench_model --baseline yolo11n-seg [--warmup 10] [--iters 100]

With --checkpoint and no --input-size, the size is the one the realtime demo
would use for that checkpoint (environment, then metadata.json, then the
demo's default); a flag that contradicts the metadata is refused.

The model is built and run through the demo's own path (``jetson.demo_model``:
``build_demo_model`` / ``load_demo_model`` and ``demo_forward``), so the
numbers are the demo's. No cv2: it runs where the camera stack is absent.

``--baseline NAME`` times a public segmentation model instead, through the same
clock, warmup and percentile code (``ecs_hcm_model.eval.baselines``), at the size
its adapter declares. Its report carries ``timed``: what the timed call covers,
because a package's own preprocessing and NMS sit inside its predict() and are
not the same measurement as a bare forward. The baseline's package is imported
only for that run, so this module loads where none is installed.
"""
import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import torch

from jetson import demo_model
from jetson.input_size import SOURCE_CLI, SOURCE_DEFAULT, InputSizeError, load_metadata, resolve_input_size
from jetson.trt_engine import EngineError, TrtEngine, sha256_file

from ecs_hcm_model.eval import baselines

ARCHES = ("v1", "v2")
DEFAULT_BENCH_INPUT_SIZE = 352

# The timing clock, a seam for tests (a fake clock gives known durations).
_clock = time.perf_counter


def _parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arch", choices=ARCHES, default=None,
                    help="Decoder version; required without --checkpoint, checked against it otherwise")
    ap.add_argument("--input-size", type=int, default=None,
                    help=f"Default {DEFAULT_BENCH_INPUT_SIZE}; with --checkpoint, the demo's size for it")
    ap.add_argument("--checkpoint", type=Path, default=None,
                    help="Dir with model.pth (and optionally metadata.json), loaded as the demo loads it")
    ap.add_argument("--out-stride", type=int, default=None,
                    help="Backbone stride; as the demo's flag with --checkpoint, else 16")
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--fp16", action="store_true",
                    help="float16 autocast (CUDA only), as the demos' --fp16")
    ap.add_argument("--baseline", default=None,
                    help="Time a public segmentation model instead of the demo's, at the size "
                         f"its adapter declares. Known: {', '.join(baselines.names())}")
    ap.add_argument("--engine", type=Path, default=None,
                    help="Dir written by jetson/export_trt.py: time its engine instead of the "
                         "torch forward (needs --checkpoint, for the validation)")
    ap.add_argument("--json", action="store_true", help="Print one JSON object")
    return ap.parse_args(argv)


class BenchError(Exception):
    """A refusal: printed as one line, exit 2."""


def _checkpoint_input_size(cli_size, meta, env):
    """``(size, source)`` for a checkpoint: the realtime demo's resolver when
    no size was given; a given size must agree with one the metadata states."""
    try:
        if cli_size is None:
            return resolve_input_size(meta, env)
        stated, source = resolve_input_size(meta, {})
    except InputSizeError as e:
        raise BenchError(f"Cannot resolve the model input size: {e}") from e
    if source != SOURCE_DEFAULT and stated != cli_size:
        raise BenchError(f"--input-size {cli_size} but the checkpoint's {source} is {stated}")
    return cli_size, SOURCE_CLI


def _build(args, device):
    """``(model, input_size)``."""
    if args.checkpoint is None:
        if args.arch is None:
            raise BenchError("--arch is required without --checkpoint")
        size = DEFAULT_BENCH_INPUT_SIZE if args.input_size is None else args.input_size
        return demo_model.build_demo_model(2, args.out_stride or 16, args.arch, device), size
    weights = args.checkpoint / "model.pth"
    if not weights.is_file():
        raise BenchError(f"no model.pth in {args.checkpoint}")
    try:
        meta = load_metadata(args.checkpoint / "metadata.json")
    except InputSizeError as e:
        raise BenchError(str(e)) from e
    size, source = _checkpoint_input_size(args.input_size, meta, os.environ)
    try:
        model, _ = demo_model.load_demo_model(
            weights, meta, cli_stride=args.out_stride, env=os.environ, device=device,
            input_size=size, input_size_source=source)
    except SystemExit as e:
        # The loader's refusals (a contradicted version, a non-binary head, a
        # bad stride) stop the demo with a message; here they are a refusal.
        if isinstance(e.code, str):
            raise BenchError(e.code) from e
        raise
    if args.arch is not None and model.decoder_arch != args.arch:
        raise BenchError(f"--arch {args.arch} but {args.checkpoint} is a {model.decoder_arch} model")
    return model, size


def _baseline_model(args, device):
    """``(model, input_size, spec, size_source)`` for ``--baseline``: a public model.

    The demo's own options describe a checkpoint or an engine, so they are refused
    rather than ignored: a baseline timed with a flag that did nothing is a number
    nobody can reproduce.
    """
    given = [name for name, value in (("--arch", args.arch), ("--checkpoint", args.checkpoint),
                                      ("--out-stride", args.out_stride), ("--engine", args.engine))
             if value is not None]
    if given:
        raise BenchError(f"--baseline {args.baseline} times a public model, which has no "
                         f"{', '.join(given)}")
    if args.fp16:
        raise BenchError(f"--baseline {args.baseline} runs its package's own fp32 inference; "
                         f"--fp16 does not apply to it")
    try:
        spec, size, source = baselines.resolve(args.baseline, args.input_size)
        model = baselines.build(spec.name, device, size)
    except baselines.BaselineError as e:
        raise BenchError(str(e)) from e
    return model, size, spec, source


def _load_engine(args, input_size, device):
    """The `--engine` engine, validated against the checkpoint this run serves.

    The engine is shape-specialised and built from one checkpoint, so it is only
    meaningful next to the checkpoint it came from: `--checkpoint` is required and the
    recorded sha256 must be its `model.pth`.
    """
    if args.engine is None:
        return None
    if device.type != "cuda":
        raise BenchError(f"--engine needs a CUDA device (this run is on {device})")
    if args.fp16:
        raise BenchError("--fp16 does not apply to --engine: the engine's precision is fixed at export")
    if args.checkpoint is None:
        raise BenchError("--engine needs --checkpoint: it is validated against the checkpoint it was built from")
    weights = args.checkpoint / "model.pth"
    if not weights.is_file():
        raise BenchError(f"no model.pth in {args.checkpoint}")
    try:
        return TrtEngine.load(args.engine, input_size=input_size, source_sha256=sha256_file(weights))
    except EngineError as e:
        raise BenchError(str(e)) from e


def _percentile(sorted_ms, q):
    """Nearest-rank percentile of an ascending list."""
    return sorted_ms[max(0, math.ceil(q * len(sorted_ms)) - 1)]


def _device_sync(device):
    """A callable that waits for ``device``'s queued work; a no-op off CUDA."""
    if device.type == "cuda":
        return lambda: torch.cuda.synchronize(device)
    return lambda: None


def time_forward(forward, *, warmup, iters, sync, clock=None):
    """Ascending per-iteration times in ms. Each timed iteration ends with
    ``sync()``, so queued GPU work is inside the iteration that queued it."""
    clock = clock or _clock
    for _ in range(warmup):
        forward()
    sync()
    times_ms = []
    for _ in range(iters):
        t0 = clock()
        forward()
        sync()
        times_ms.append((clock() - t0) * 1000)
    return sorted(times_ms)


def run(args):
    if args.warmup < 0 or args.iters < 1:
        raise BenchError("--warmup must be >= 0 and --iters >= 1")
    device = torch.device(args.device)
    spec = None
    if args.baseline:
        model, input_size, spec, size_source = _baseline_model(args, device)
    else:
        try:
            demo_model.require_fp16_device(args.fp16, device)
        except ValueError as e:
            raise BenchError(str(e)) from e
        model, input_size = _build(args, device)
    cuda = device.type == "cuda"
    engine = _load_engine(args, input_size, device)
    x = torch.zeros(1, 3, input_size, input_size, device=device, dtype=torch.float32)
    sync = _device_sync(device)

    if engine is not None:
        def forward():
            engine.infer(x)
    elif spec is not None:
        # Exactly what the eval tool's forward costs: the adapter call on a
        # prepared tensor, including whatever the package does inside it.
        def forward():
            model(x)
    else:
        def forward():
            demo_model.demo_forward(model, x, fp16=args.fp16)

    # Warmup first, so its allocations do not count toward the peak.
    time_forward(forward, warmup=args.warmup, iters=0, sync=sync)
    if cuda:
        torch.cuda.reset_peak_memory_stats(device)
    times_ms = time_forward(forward, warmup=0, iters=args.iters, sync=sync)
    if spec is not None:
        report = {
            "arch": None,
            "baseline": spec.name,
            "input_size": input_size,
            "input_size_source": size_source,
            "device": str(device),
            "backend": "torch",
            "dtype": "float32",
            "params": sum(p.numel() for p in model.parameters()),
            "batch_size": 1,
            "warmup": args.warmup,
            "iters": args.iters,
            "mean_ms": sum(times_ms) / len(times_ms),
            "p50_ms": _percentile(times_ms, 0.50),
            "p90_ms": _percentile(times_ms, 0.90),
            "peak_cuda_mem_mb": torch.cuda.max_memory_allocated(device) / 2**20 if cuda else None,
            "timed": spec.timed_note,
        }
        if spec.caveat:
            report["caveat"] = spec.caveat
            report["caveat_note"] = spec.caveat_note
        return report
    return {
        "arch": model.decoder_arch,
        "input_size": input_size,
        "device": str(device),
        "backend": "engine" if engine is not None else "torch",
        "dtype": engine.precision if engine is not None else ("float16-autocast" if args.fp16 else "float32"),
        "params": sum(p.numel() for p in model.parameters()),
        "warmup": args.warmup,
        "iters": args.iters,
        "mean_ms": sum(times_ms) / len(times_ms),
        "p50_ms": _percentile(times_ms, 0.50),
        "p90_ms": _percentile(times_ms, 0.90),
        "peak_cuda_mem_mb": torch.cuda.max_memory_allocated(device) / 2**20 if cuda else None,
    }


def format_report(r):
    """The one-screen report. A baseline's says what the timed call covered, so a
    latency next to the demo's is not read as the same measurement."""
    if r.get("baseline"):
        lines = [
            f"baseline {r['baseline']}, input {r['input_size']}, device {r['device']}, "
            f"backend {r['backend']}, dtype {r['dtype']}, batch {r['batch_size']}",
            f"input size from {r['input_size_source']}",
            f"params {r['params']:,}",
            f"{r['iters']} iters after {r['warmup']} warmup: "
            f"mean {r['mean_ms']:.2f} ms, p50 {r['p50_ms']:.2f} ms, p90 {r['p90_ms']:.2f} ms",
            f"timed: {r['timed']}",
        ]
        if r.get("caveat"):
            lines.append(f"caveat {r['caveat']}: {r['caveat_note']}")
        if r["peak_cuda_mem_mb"] is not None:
            lines.append(f"peak CUDA memory {r['peak_cuda_mem_mb']:.1f} MB")
        return "\n".join(lines)
    lines = [
        f"arch {r['arch']}, input {r['input_size']}, device {r['device']}, "
        f"backend {r['backend']}, dtype {r['dtype']}",
        f"params {r['params']:,}",
        f"{r['iters']} iters after {r['warmup']} warmup: "
        f"mean {r['mean_ms']:.2f} ms, p50 {r['p50_ms']:.2f} ms, p90 {r['p90_ms']:.2f} ms",
    ]
    if r["peak_cuda_mem_mb"] is not None:
        lines.append(f"peak CUDA memory {r['peak_cuda_mem_mb']:.1f} MB")
    return "\n".join(lines)


def main(argv=None):
    args = _parse_args(argv)
    try:
        report = run(args)
    except BenchError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    print(json.dumps(report) if args.json else format_report(report))
    return 0


if __name__ == "__main__":
    sys.exit(main())
