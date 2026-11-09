"""Camera round 0 measurement (design D8): cross-model agreement on saved frames.

Two models disagree a little on any frame, so an absolute number on the uncertain
frames proves nothing; what the gate reads is the gap between the uncertain set
and the camera control sample (D9). This module produces those numbers and
nothing else: it is read-only with respect to MinIO and it never trains.

Run it in the bounded Job (``openspec/changes/camera-uncertain-frames-round0``
section 5), or locally with ``--frames-dir`` for a smoke test on a few frames.
"""

from __future__ import annotations

import csv
import datetime
import hashlib
import json
import os
import random
import statistics
import sys
from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_BUCKET = "camera-uncertain-frames"
DEFAULT_PREFIX = "round0"

# D8: fixed before the first frame is scored. A stronger 2-class checkpoint is
# named by the lead (--checkpoint) and its sha256 recorded in the task; choosing
# the comparison after seeing the gap is what these defaults exist to prevent.
DEFAULT_CHECKPOINTS = (
    ("live", "finetuned-models", "hcm_cocoa_seg_best/20261022_144436"),
    ("strong", "finetuned-models", "hcm_cocoa_seg_best/20261024_104051"),
)

BDD_CONTROL_N = 300
BDD_CONTROL_SEED = 0
GATE_MARGIN = 0.10
MIN_CONTROLS_PER_NIGHT = 25
# D9's night structure, pre-registered with the margin: the round is five counted
# nights, a counted night holds at least 100 uncertain frames, and the gate needs
# 500 frames in total.
COUNTED_NIGHTS = 5
MIN_UNCERTAIN_PER_NIGHT = 100
MIN_FRAMES_ROUND = 500
# A "exactly 0.10 below" boundary has to pass, and 0.75 - 0.10 is 0.6499999999999999
# in binary floating point, so the comparison carries an absolute tolerance. It is
# absolute on purpose: a relative one would move with the means it is judging.
GATE_EPS = 1e-9
# The comparison grid is pre-registered (design D8): the live model's input size on
# the day the gate was written. It is not "whatever the live checkpoint happens to
# be" -- re-pointing the live model at a different size would silently change every
# number the gate compares, so the run refuses instead.
COMMON_GRID = 352


class MeasurementError(Exception):
    """A mismatch or a missing pin: refuse to score rather than report a number."""


def common_grid(sizes):
    """The pre-registered scoring grid (D8), or a refusal if live has moved.

    Both arms are scored on the live model's grid, and the gate's numbers were
    registered on 352. Reading the grid off whichever live checkpoint is configured
    would let a re-pointed live model change the comparison silently.
    """
    grid = sizes["live"]
    if grid != COMMON_GRID:
        raise MeasurementError(
            f"the live model's input size is {grid}; the round-0 common grid is "
            f"pre-registered as {COMMON_GRID} (design D8)")
    return grid


@dataclass
class Checkpoint:
    name: str
    bucket: str
    prefix: str
    path: Path
    sha256: str
    meta: dict = field(default_factory=dict)

    @property
    def run_id(self):
        return self.prefix.rsplit("/", 1)[-1]


@dataclass
class Frame:
    name: str
    night: str
    kind: str
    path: Path
    score: float | None = None
    max_p_fg: float | None = None


# ---------------------------------------------------------------------------
# Arguments: every checkpoint needs its recorded sha256 before anything runs
# ---------------------------------------------------------------------------

def parse_checkpoints(checkpoint_args=None, sha_args=None):
    """``name -> (bucket, prefix, sha256)`` from ``--checkpoint``/``--sha256`` pairs.

    A default checkpoint keeps its identity unless ``--checkpoint`` overrides it,
    but it never keeps a default sha256: no pin, no scoring (5.4).
    """
    specs = {name: [bucket, prefix, None] for name, bucket, prefix in DEFAULT_CHECKPOINTS}
    for spec in checkpoint_args or ():
        name, sep, target = spec.partition("=")
        if not sep or not name or "/" not in target:
            raise MeasurementError(f"--checkpoint expects NAME=BUCKET/PREFIX, got {spec!r}")
        bucket, _slash, prefix = target.partition("/")
        specs.setdefault(name, [bucket, prefix, None])
        specs[name][0], specs[name][1] = bucket, prefix
    for spec in sha_args or ():
        name, sep, sha = spec.partition("=")
        if not sep or name not in specs:
            raise MeasurementError(f"--sha256 expects NAME=SHA256 for a named checkpoint, got {spec!r}")
        sha = sha.strip().lower()
        if len(sha) != 64 or any(c not in "0123456789abcdef" for c in sha):
            raise MeasurementError(f"{name}: {sha!r} is not a sha256 hex digest")
        specs[name][2] = sha
    missing = [name for name, (_b, _p, sha) in specs.items() if sha is None]
    if missing:
        raise MeasurementError(
            "no recorded sha256 for " + ", ".join(missing) +
            ": record the artefact's sha256 before scoring and pass --sha256 NAME=SHA256"
        )
    return {name: (b, p, sha) for name, (b, p, sha) in specs.items()}


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Fetch
# ---------------------------------------------------------------------------

def fetch_checkpoints(client, specs, dest_root, log=print):
    """Download ``model.pth`` + ``metadata.json`` per spec and verify each pin.

    The hash is taken from the bytes on disk after the download, so what is
    verified is what will be loaded. Nothing is scored if either mismatches.
    """
    dest_root = Path(dest_root)
    out = {}
    for name, (bucket, prefix, want) in specs.items():
        dest = dest_root / name
        dest.mkdir(parents=True, exist_ok=True)
        hashes = {}
        for fn in ("model.pth", "metadata.json"):
            path = dest / fn
            client.fget_object(bucket, f"{prefix}/{fn}", str(path))
            hashes[fn] = sha256_file(path)
            log(f"CKPT {name} {bucket}/{prefix}/{fn} bytes={path.stat().st_size} "
                f"sha256={hashes[fn]}")
        if hashes["model.pth"] != want:
            raise MeasurementError(
                f"{name}: sha256 mismatch for {bucket}/{prefix}/model.pth: recorded {want}, "
                f"fetched {hashes['model.pth']}. Refusing to score."
            )
        meta = json.loads((dest / "metadata.json").read_text())
        out[name] = Checkpoint(name, bucket, prefix, dest / "model.pth", want, meta)
    return out


def fetch_frames(client, dest_root, *, bucket=DEFAULT_BUCKET, prefix=DEFAULT_PREFIX, log=print):
    """Download every saved frame and its sidecar under ``prefix``; return the frames.

    The sidecar carries the kind and the score the demo recorded at save time, so
    a frame's selection history survives without re-deriving it. A frame whose
    sidecar is missing or unparseable is dropped rather than guessed at.
    """
    dest_root = Path(dest_root)
    root = prefix.strip("/")
    names = []
    for obj in client.list_objects(bucket, prefix=root + "/", recursive=True):
        name = obj.object_name.rsplit("/", 1)[-1]
        if not name.endswith(".jpg"):
            continue
        local = dest_root / obj.object_name
        local.parent.mkdir(parents=True, exist_ok=True)
        client.fget_object(bucket, obj.object_name, str(local))
        names.append(obj.object_name)

    frames, dropped = {}, 0
    for object_name in names:
        local = dest_root / object_name
        side = local.with_suffix(".json")
        try:
            client.fget_object(bucket, str(Path(object_name).with_suffix(".json")), str(side))
            meta = json.loads(side.read_text())
        except Exception:  # noqa: BLE001 - absent or unparseable sidecar: drop the frame
            dropped += 1
            continue
        night = object_name[len(root) + 1:].split("/", 1)[0]
        frames[local.name] = Frame(local.name, night, meta.get("kind", "uncertain"), local,
                                   meta.get("score"), meta.get("max_p_fg"))
    log(f"FRAMES {len(frames)} under s3://{bucket}/{root}/ (dropped {dropped} without a sidecar)")
    return frames


def frames_from_dir(root, *, prefix=DEFAULT_PREFIX):
    """The local-tree form of :func:`fetch_frames`, for a smoke test."""
    root = Path(root) / prefix.strip("/")
    frames = {}
    for side in sorted(root.glob("*/*.json")):
        name = side.with_suffix("").name
        image = side.with_suffix(".jpg")
        if not image.is_file():
            continue
        meta = json.loads(side.read_text())
        frames[name] = Frame(name, side.parent.name, meta.get("kind", "uncertain"), image,
                             meta.get("score"), meta.get("max_p_fg"))
    return frames


# ---------------------------------------------------------------------------
# The recorded stop: nothing captured after it enters the measurement
# ---------------------------------------------------------------------------

def parse_last_night(value):
    """The round's recorded last night, validated as ``YYYYmmdd``.

    A night is the bucket's ``round0/<YYYYmmdd>/`` directory, so the value is
    compared as a string; parsing it first rejects a typo that would otherwise
    silently exclude every night or none.
    """
    text = str(value).strip()
    try:
        night = datetime.datetime.strptime(text, "%Y%m%d").strftime("%Y%m%d")
    except ValueError:
        raise MeasurementError(
            "last night %r is not a YYYYmmdd date" % (value,)) from None
    # strptime is lenient about a short day ("2026111" parses as 1 November), and a
    # value that round-trips differently from what it parses to is a typo either way.
    if night != text:
        raise MeasurementError(
            "last night %r is not a YYYYmmdd date" % (value,))
    return night


def exclude_after(frames, last_night):
    """Drop the frames on a night later than the round's recorded last night.

    Returns the kept frames and how many were dropped per night. The capture can
    run on after the stop (it did in round 0: night 20261102 was written after the
    stop, 10:11:36Z), and the gate refuses six counted nights, so without this the
    round would stop for the wrong reason.
    """
    if last_night is None:
        return dict(frames), {}
    kept, dropped = {}, {}
    for name, frame in frames.items():
        if frame.night > last_night:
            dropped[frame.night] = dropped.get(frame.night, 0) + 1
        else:
            kept[name] = frame
    return kept, dict(sorted(dropped.items()))


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def _default_preprocess():
    """The demo's own preprocessing, so the Job scores frames as the camera does."""
    from jetson.realtime_demo import preprocess_bgr  # noqa: PLC0415 - needs cv2
    return preprocess_bgr


def predict_fg(model, frame_bgr, input_size, device, grid_size, fp16=False, preprocess=None):
    """Foreground mask for one frame on the common ``grid_size`` grid.

    Each model takes the frame at its own recorded size; the mask comes off the
    logits' grid by argmax and is brought to the common grid with the demo's
    nearest-neighbour upsample, so a size handicap applies to both arms alike
    (the differential is what is read).
    """
    import torch  # noqa: PLC0415

    pre = _default_preprocess() if preprocess is None else preprocess
    x = pre(frame_bgr, input_size).to(device=device, dtype=torch.float32)
    model.eval()
    with torch.no_grad():
        if fp16:
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                logits = model(x)
        else:
            logits = model(x)
    if not torch.is_tensor(logits) and isinstance(logits, (tuple, list)):
        logits = logits[0]
    if logits.ndim != 4 or logits.shape[1] < 2:
        raise MeasurementError(f"expected logits [1,2,h,w], got {tuple(logits.shape)}")
    mask = (logits.argmax(dim=1) == 1).float()
    if tuple(mask.shape[-2:]) != (grid_size, grid_size):
        # The demo's display upsample is nearest-neighbour; the common grid is
        # built the same way so a size handicap applies to both arms alike.
        import torch.nn.functional as F  # noqa: PLC0415
        mask = F.interpolate(mask.unsqueeze(1), size=(grid_size, grid_size), mode="nearest")
        mask = mask.squeeze(1)
    return mask[0].cpu().numpy().astype(bool)


def pair_iou_dice(mask_a, mask_b):
    """``(iou, dice)`` for two boolean masks, through the metric every val_iou uses."""
    import torch  # noqa: PLC0415
    from ecs_hcm_model.core.trainer import iou_dice_fg_from_pred  # noqa: PLC0415

    a = torch.as_tensor(mask_a, dtype=torch.bool).unsqueeze(0)
    b = torch.as_tensor(mask_b, dtype=torch.bool).unsqueeze(0)
    iou, dice = iou_dice_fg_from_pred(a, b)
    return float(iou), float(dice)


def _read_bgr(path):
    """The frame as a BGR uint8 array, the order ``preprocess_bgr`` expects."""
    from PIL import Image  # noqa: PLC0415
    import numpy as np  # noqa: PLC0415

    try:
        with Image.open(path) as im:
            return np.asarray(im.convert("RGB"))[:, :, ::-1].copy()
    except (OSError, ValueError):
        return None


def score_frames(models, input_sizes, frames, device, grid_size, *, fp16=False, log=print):
    """Score every frame with both models; return one row per frame."""
    from PIL import Image  # noqa: PLC0415

    rows = []
    for i, frame in enumerate(frames, 1):
        img = _read_bgr(frame.path)
        if img is None:
            log(f"SKIP {frame.name}: unreadable")
            continue
        preds = {name: predict_fg(model, img, input_sizes[name], device, grid_size, fp16)
                 for name, model in models.items()}
        names = list(preds)
        iou, dice = pair_iou_dice(preds[names[0]], preds[names[1]])
        rows.append({
            "name": frame.name, "night": frame.night, "kind": frame.kind,
            "iou": iou, "dice": dice,
            "score": frame.score, "max_p_fg": frame.max_p_fg,
        })
        if i % 100 == 0:
            log(f"SCORED {i}/{len(frames)}")
    return rows


# ---------------------------------------------------------------------------
# Aggregates
# ---------------------------------------------------------------------------

def aggregate(values):
    """Mean, median, count and the fraction below the 0.50 disagreement mark."""
    vals = [float(v) for v in values]
    if not vals:
        return {"n": 0, "mean": None, "median": None, "frac_lt_050": None, "std": None}
    return {
        "n": len(vals),
        "mean": statistics.fmean(vals),
        "median": statistics.median(vals),
        "frac_lt_050": sum(1 for v in vals if v < 0.50) / len(vals),
        # Sample sd when there is more than one frame: the per-night control
        # standard deviation the report has to carry (D9).
        "std": statistics.stdev(vals) if len(vals) > 1 else 0.0,
    }


def by_kind(rows, kind):
    return [r for r in rows if r["kind"] == kind]


def per_night(rows):
    """``night -> aggregate`` over the rows given, nights in date order."""
    groups = {}
    for row in rows:
        groups.setdefault(row["night"], []).append(row["iou"])
    return {night: aggregate(groups[night]) for night in sorted(groups)}


def nights_touched(rows):
    """Every round-0 night in the rows, control frames or not (D9).

    Deliberately not built from the control rows: a night whose controls all
    failed to upload still produced uncertain frames, and it has to be named as
    short rather than vanish from the report.
    """
    return sorted({r["night"] for r in rows if r["kind"] != "bdd"})


def summarize(rows, *, last_night=None, excluded=None):
    """The three pooled sets plus the per-night control aggregates (D8 step 6).

    ``last_night`` and ``excluded`` are the round's recorded stop and what it kept
    out of the numbers; they ride along in the summary so the run records why the
    gate saw five nights when the bucket held six.
    """
    uncertain = by_kind(rows, "uncertain")
    control = by_kind(rows, "control")
    bdd = by_kind(rows, "bdd")
    nights = per_night(control)
    return {
        "uncertain": aggregate(r["iou"] for r in uncertain),
        "control": aggregate(r["iou"] for r in control),
        "bdd": aggregate(r["iou"] for r in bdd),
        "control_by_night": nights,
        "uncertain_by_night": per_night(uncertain),
        "short_nights": sorted(
            n for n in nights_touched(rows)
            if nights.get(n, {"n": 0})["n"] < MIN_CONTROLS_PER_NIGHT
        ),
        "last_night": last_night,
        "excluded_after_stop": dict(excluded or {}),
    }


def _name_nights(nights):
    return "; short nights: " + ", ".join(nights) if nights else ""


def gate(rows):
    """D9's pre-registered rule over one round's scored frames: (verdict, reason).

    ``verdict`` is ``"recommend"`` or ``"stop"``; ``reason`` is a sentence the
    report can carry as it stands, naming the night and the numbers that decided
    it. The thresholds are constants here rather than arguments: the rule was
    fixed before the data arrived, and a gate whose cut-off can be passed in is a
    gate that can be moved after the fact.
    """
    uncertain = by_kind(rows, "uncertain")
    control = by_kind(rows, "control")

    counts = {}
    for row in uncertain:
        counts[row["night"]] = counts.get(row["night"], 0) + 1
    counted = sorted(n for n, c in counts.items() if c >= MIN_UNCERTAIN_PER_NIGHT)
    short = sorted(n for n, c in counts.items() if c < MIN_UNCERTAIN_PER_NIGHT)

    if len(uncertain) < MIN_FRAMES_ROUND:
        return "stop", ("%d saved uncertain frames; the gate needs %d%s"
                        % (len(uncertain), MIN_FRAMES_ROUND, _name_nights(short)))
    if len(counted) < COUNTED_NIGHTS:
        return "stop", ("only %d counted nights, %d needed%s"
                        % (len(counted), COUNTED_NIGHTS, _name_nights(short)))
    if len(counted) > COUNTED_NIGHTS:
        return "stop", ("%d nights hold %d or more uncertain frames, but the round is "
                        "exactly %d: nothing captured after the stop enters the gate"
                        % (len(counted), MIN_UNCERTAIN_PER_NIGHT, COUNTED_NIGHTS))

    # A counted night's exposure is its own controls' count; a short night's is
    # reported, never fatal (D9).
    night_control = per_night(control)
    thin = [(n, night_control.get(n, {"n": 0})["n"]) for n in counted
            if night_control.get(n, {"n": 0})["n"] < MIN_CONTROLS_PER_NIGHT]
    if thin:
        return "stop", ("insufficient exposure: %s; a counted night needs %d control "
                        "frames" % ("; ".join("%s has %d" % (n, c) for n, c in thin),
                                    MIN_CONTROLS_PER_NIGHT))

    night_uncertain = per_night(uncertain)
    pooled_uncertain = aggregate(r["iou"] for r in uncertain)["mean"]
    pooled_control = aggregate(r["iou"] for r in control)["mean"]
    if pooled_uncertain > pooled_control - GATE_MARGIN + GATE_EPS:
        return "stop", ("pooled uncertain %.4f is not %.2f below the pooled control %.4f"
                        % (pooled_uncertain, GATE_MARGIN, pooled_control))

    gaps = {n: night_uncertain[n]["mean"] - night_control[n]["mean"] for n in counted}
    failing = [n for n in counted if gaps[n] > -GATE_MARGIN + GATE_EPS]
    if failing:
        worst = max(failing, key=gaps.get)
        others = [n for n in failing if n != worst]
        return "stop", ("night %s: uncertain %.4f against that night's own controls "
                        "%.4f, a gap of %.4f against the %.2f required%s"
                        % (worst, night_uncertain[worst]["mean"],
                           night_control[worst]["mean"], gaps[worst], GATE_MARGIN,
                           ("; also " + ", ".join(others)) if others else ""))

    closest = max(gaps, key=gaps.get)
    return "recommend", ("%d uncertain frames over %d counted nights; pooled gap %.4f; "
                         "the closest night is %s at %.4f against its own controls"
                         % (len(uncertain), COUNTED_NIGHTS,
                            pooled_uncertain - pooled_control, closest, gaps[closest]))


def write_frames_csv(rows, path):
    """Per-frame artefact: one row per scored frame, ``kind`` included."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["name", "night", "kind", "iou", "dice", "score", "max_p_fg"]
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in fields})
    return path


# ---------------------------------------------------------------------------
# MLflow
# ---------------------------------------------------------------------------

def log_run(checkpoints, summary, csv_path, *, experiment="camera-uncertain-frames",
            mlflow=None, log=print):
    """Log one run: both identities and pins as params, every aggregate as a metric.

    ``mlflow`` is injectable so the test asserts what is logged without a server.
    """
    if mlflow is None:
        import mlflow  # noqa: PLC0415
    mlflow.set_experiment(experiment)
    with mlflow.start_run(run_name="camera-round0-measure") as run:
        for name in sorted(checkpoints):
            ckpt = checkpoints[name]
            mlflow.log_param(f"{name}_run_id", ckpt.run_id)
            mlflow.log_param(f"{name}_sha256", ckpt.sha256)
            mlflow.log_param(f"{name}_bucket", ckpt.bucket)
            mlflow.log_param(f"{name}_prefix", ckpt.prefix)
        for name in ("uncertain", "control", "bdd"):
            agg = summary[name]
            if agg["n"] == 0:
                continue
            for key, value in agg.items():
                if key != "n" and value is not None:
                    mlflow.log_metric(f"{name}_{key}", float(value))
            mlflow.log_metric(f"{name}_n", float(agg["n"]))
        mlflow.log_param("last_night", summary.get("last_night") or "none")
        for night, count in (summary.get("excluded_after_stop") or {}).items():
            mlflow.log_metric(f"excluded_after_stop_{night}", float(count))
        for night, agg in summary["control_by_night"].items():
            for key in ("mean", "std"):
                mlflow.log_metric(f"control_{key}_{night}", float(agg[key] or 0.0))
            mlflow.log_metric(f"control_n_{night}", float(agg["n"]))
        mlflow.log_metric("short_nights", float(len(summary["short_nights"])))
        if csv_path is not None:
            mlflow.log_artifact(str(csv_path))
        run_id = getattr(run, "info", None)
        run_id = getattr(run_id, "run_id", None) or getattr(run, "run_id", None)
        log(f"MLFLOW run {run_id} experiment={experiment} artifact={csv_path}")
        return run_id


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def bdd_control_paths(root, n=BDD_CONTROL_N, seed=BDD_CONTROL_SEED, split="val"):
    """``n`` random BDD validation image paths, fixed seed, sorted by stem.

    A secondary reference only (D8): it measures in-distribution disagreement and
    never gates the round.
    """
    from ecs_hcm_model.core.bdd import BDD100kSegDataset  # noqa: PLC0415

    ds = BDD100kSegDataset(root, split=split)
    rng = random.Random(seed)
    picked = sorted(rng.sample(range(len(ds)), min(n, len(ds))),
                    key=lambda i: ds.stems[i])
    return [Frame(name=ds.stems[i], night="bdd", kind="bdd", path=ds.image_path(i))
            for i in picked]


def _minio_client():
    from minio import Minio  # noqa: PLC0415

    endpoint = os.environ.get("MINIO_ENDPOINT")
    user = os.environ.get("MINIO_USER")
    password = os.environ.get("MINIO_PASSWORD")
    if not endpoint or not user or not password:
        raise MeasurementError("MINIO_ENDPOINT, MINIO_USER and MINIO_PASSWORD must be set")
    secure = os.environ.get("MINIO_SECURE", "").strip().lower() in ("1", "true", "yes")
    region = os.environ.get("MINIO_REGION", "").strip() or "us-east-1"
    return Minio(endpoint, user, password, secure=secure, region=region)


def parse_args(argv=None):
    import argparse  # noqa: PLC0415

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", action="append", metavar="NAME=BUCKET/PREFIX",
                    default=None, help="override a default checkpoint's identity")
    ap.add_argument("--sha256", action="append", metavar="NAME=SHA256", default=None,
                    help="the recorded sha256 of that checkpoint's model.pth (required)")
    ap.add_argument("--frames-dir", type=Path, default=None,
                    help="read frames from a local tree instead of MinIO")
    ap.add_argument("--last-night", default=None, metavar="YYYYMMDD",
                    help="the round's recorded last night; frames on a later night "
                         "are dropped before scoring and gating")
    ap.add_argument("--bdd-root", type=Path, default=None,
                    help="prepared BDD tree for the secondary reference sample")
    ap.add_argument("--out-dir", type=Path, default=Path("/work/out"))
    ap.add_argument("--experiment", default="camera-uncertain-frames")
    ap.add_argument("--device", default="cuda" if _cuda_available() else "cpu")
    ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--no-mlflow", action="store_true", help="skip logging (smoke test)")
    return ap.parse_args(argv)


def _cuda_available():
    try:
        import torch  # noqa: PLC0415
        return torch.cuda.is_available()
    except Exception:  # noqa: BLE001 - no torch is "no cuda"
        return False


def _load_models(checkpoints, device, log):
    from ecs_hcm_model.eval import seg  # noqa: PLC0415

    models, sizes = {}, {}
    for name in sorted(checkpoints):
        ckpt = checkpoints[name]
        size, _source = seg.resolve_size(ckpt.path, explicit_meta=ckpt.path.parent / "metadata.json")
        sizes[name] = int(size)
        models[name] = seg.load_checkpoint(ckpt.path, device, meta=ckpt.meta, log=log)
        log(f"MODEL {name} run={ckpt.run_id} input_size={sizes[name]} sha256={ckpt.sha256}")
    return models, sizes


def main(argv=None):
    args = parse_args(argv)
    log = lambda *a: print(*a, flush=True)  # noqa: E731
    try:
        specs = parse_checkpoints(args.checkpoint, args.sha256)
    except MeasurementError as e:
        log(f"REFUSE {e}")
        return 2
    try:
        last_night = (parse_last_night(args.last_night)
                      if args.last_night is not None else None)
    except MeasurementError as e:
        log(f"REFUSE {e}")
        return 2

    args.out_dir.mkdir(parents=True, exist_ok=True)
    if args.frames_dir is not None:
        frames = frames_from_dir(args.frames_dir)
    else:
        client = _minio_client()
        frames = fetch_frames(client, args.out_dir / "frames")

    # Even a smoke test on a local tree fetches the weights the same way, so the
    # pins it verifies are the ones the Job verifies.
    ckpt_client = _minio_client() if args.frames_dir is not None else client
    checkpoints = fetch_checkpoints(ckpt_client, specs, args.out_dir / "ckpt", log=log)

    # The capture ran past the stop, so the bucket can hold a night the round does
    # not count. Drop it before scoring and gating, and say what was dropped.
    frames, excluded = exclude_after(frames, last_night)
    for night, count in excluded.items():
        log(f"EXCLUDED {count} frames on night {night}: after the recorded last "
            f"night {last_night}")

    if not frames:
        log("REFUSE no frames to score"
            + (f" on or before {last_night}" if last_night else ""))
        return 2

    models, sizes = _load_models(checkpoints, args.device, log)
    try:
        grid_size = common_grid(sizes)
    except MeasurementError as e:
        log(f"REFUSE {e}")
        return 2
    log(f"GRID common grid {grid_size} (pre-registered)")

    rows = score_frames(models, sizes, sorted(frames.values(), key=lambda f: f.name),
                        args.device, grid_size, fp16=args.fp16, log=log)
    if args.bdd_root is not None:
        rows += score_frames(models, sizes, bdd_control_paths(args.bdd_root),
                            args.device, grid_size, fp16=args.fp16, log=log)

    summary = summarize(rows, last_night=last_night, excluded=excluded)
    for name in ("uncertain", "control", "bdd"):
        agg = summary[name]
        log(f"RESULT {name} n={agg['n']} mean={agg['mean']} median={agg['median']} "
            f"frac_lt_050={agg['frac_lt_050']}")
    for night, agg in summary["control_by_night"].items():
        log(f"RESULT control night={night} n={agg['n']} mean={agg['mean']} std={agg['std']}")
    if summary["short_nights"]:
        log("SHORT nights with fewer than %d control frames: %s"
            % (MIN_CONTROLS_PER_NIGHT, ", ".join(summary["short_nights"])))

    csv_path = write_frames_csv(rows, args.out_dir / "camera_frames_round0.csv")
    if not args.no_mlflow:
        log_run(checkpoints, summary, csv_path, experiment=args.experiment, log=log)
    return 0


if __name__ == "__main__":
    sys.exit(main())
