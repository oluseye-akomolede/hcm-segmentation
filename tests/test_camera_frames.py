"""Section 5 of camera-uncertain-frames-round0: the measurement refuses, then measures.

The pins are the point: a run that scores with an unverified or unmatched
checkpoint produces a number that looks like evidence. These tests exercise that
refusal, the aggregate arithmetic the gate reads, and what reaches MLflow.
"""

from __future__ import annotations

import json
import subprocess
import sys
import types
from pathlib import Path

import yaml

import numpy as np
import pytest

from ecs_hcm_model.eval import camera_frames as cf


SHA_A = "a" * 64
SHA_B = "b" * 64


class FakeMinio:
    """Just the two calls fetch_* use, over an in-memory object dict."""

    def __init__(self, objects):
        self.objects = objects     # "bucket/name" -> bytes
        self.gets = []

    class _Obj:
        def __init__(self, name):
            self.object_name = name

    def list_objects(self, bucket, prefix="", recursive=False):
        for key in sorted(self.objects):
            b, name = key.split("/", 1)
            if b == bucket and name.startswith(prefix):
                yield self._Obj(name)

    def fget_object(self, bucket, name, dest):
        key = f"{bucket}/{name}"
        self.gets.append(key)
        if key not in self.objects:
            raise FileNotFoundError(key)
        with open(dest, "wb") as fh:
            fh.write(self.objects[key])


def _checkpoint_objects(sha_for):
    """A bucket holding both default checkpoints, keyed by the files they hold."""
    objs = {}
    for name, bucket, prefix in cf.DEFAULT_CHECKPOINTS:
        body = f"weights-{name}".encode()
        import hashlib
        if sha_for.get(name) == "actual":
            sha_for[name] = hashlib.sha256(body).hexdigest()
        objs[f"{bucket}/{prefix}/model.pth"] = body
        objs[f"{bucket}/{prefix}/metadata.json"] = json.dumps({"input_size": 352}).encode()
    return objs


# -- the pins ----------------------------------------------------------------

def test_parse_checkpoints_requires_a_recorded_sha():
    with pytest.raises(cf.MeasurementError, match="no recorded sha256"):
        cf.parse_checkpoints()


def test_parse_checkpoints_refuses_a_short_sha():
    with pytest.raises(cf.MeasurementError, match="not a sha256"):
        cf.parse_checkpoints(sha_args=["live=abc", "strong=" + SHA_B])


def test_parse_checkpoints_takes_the_defaults_with_both_pins():
    specs = cf.parse_checkpoints(sha_args=["live=" + SHA_A, "strong=" + SHA_B])
    assert specs["live"] == ("finetuned-models", "hcm_cocoa_seg_best/20261022_144436", SHA_A)
    assert specs["strong"] == ("finetuned-models", "hcm_cocoa_seg_best/20261024_104051", SHA_B)


def test_parse_checkpoints_lets_the_lead_name_a_stronger_checkpoint():
    specs = cf.parse_checkpoints(
        checkpoint_args=["strong=finetuned-models/hcm_cocoa_seg_best/20261101_000000"],
        sha_args=["live=" + SHA_A, "strong=" + SHA_B],
    )
    assert specs["strong"][1].endswith("20261101_000000")


def test_a_sha_mismatch_refuses_and_names_both_hashes(tmp_path):
    sha_for = {"live": "actual", "strong": "actual"}
    objs = _checkpoint_objects(sha_for)
    # Record the wrong pin for one arm: what the artefact's hash is not.
    specs = {"live": ("finetuned-models", cf.DEFAULT_CHECKPOINTS[0][2], "0" * 64),
             "strong": ("finetuned-models", cf.DEFAULT_CHECKPOINTS[1][2], sha_for["strong"])}
    with pytest.raises(cf.MeasurementError) as excinfo:
        cf.fetch_checkpoints(FakeMinio(objs), specs, tmp_path, log=lambda *a: None)
    msg = str(excinfo.value)
    assert "live" in msg and "0" * 64 in msg and sha_for["live"] in msg
    assert "Refusing to score" in msg


def test_matching_pins_load_and_keep_the_recorded_sha(tmp_path):
    sha_for = {"live": "actual", "strong": "actual"}
    objs = _checkpoint_objects(sha_for)
    specs = {name: (bucket, prefix, sha_for[name]) for name, bucket, prefix in cf.DEFAULT_CHECKPOINTS}
    ckpts = cf.fetch_checkpoints(FakeMinio(objs), specs, tmp_path, log=lambda *a: None)
    assert ckpts["live"].sha256 == sha_for["live"]
    assert ckpts["live"].run_id == "20261022_144436"
    assert ckpts["strong"].meta["input_size"] == 352


def test_main_refuses_before_touching_minio(monkeypatch, tmp_path):
    """> an unpinned run must not download weights and must not score."""
    built = []
    monkeypatch.setattr(cf, "frames_from_dir", lambda *a, **k: built.append("frames") or {})
    assert cf.main(["--frames-dir", str(tmp_path), "--out-dir", str(tmp_path)]) == 2
    assert built == []


# -- aggregates the gate reads ----------------------------------------------

def test_aggregate_mean_median_and_fraction_below_half():
    agg = cf.aggregate([1.0, 0.9, 0.4, 0.45])
    assert agg["n"] == 4
    assert agg["mean"] == pytest.approx(0.6875)
    assert agg["median"] == pytest.approx(0.675)
    assert agg["frac_lt_050"] == pytest.approx(0.5)
    assert agg["std"] > 0


def test_aggregate_of_nothing_is_explicitly_empty():
    assert cf.aggregate([])["n"] == 0
    assert cf.aggregate([])["mean"] is None


def _row(kind, night, iou, name=None):
    return {"name": name or f"{kind}-{night}-{iou}", "night": night, "kind": kind,
            "iou": iou, "dice": iou, "score": 0.1, "max_p_fg": 0.5}


def test_controls_are_grouped_per_night_with_their_own_mean_and_std():
    rows = ([_row("control", "20261102", v) for v in (0.8, 0.9)] +
            [_row("control", "20261103", v) for v in (0.5, 0.6, 0.7)] +
            [_row("uncertain", "20261102", 0.2)])
    summary = cf.summarize(rows)
    assert set(summary["control_by_night"]) == {"20261102", "20261103"}
    n2 = summary["control_by_night"]["20261102"]
    assert n2["mean"] == pytest.approx(0.85) and n2["n"] == 2 and n2["std"] > 0
    assert summary["control_by_night"]["20261103"]["mean"] == pytest.approx(0.6)
    # The pooled uncertain mean is over uncertain frames only.
    assert summary["uncertain"]["n"] == 1 and summary["uncertain"]["mean"] == pytest.approx(0.2)


def test_a_night_with_too_few_controls_is_named_not_silently_accepted():
    rows = ([_row("control", "20261102", 0.8) for _ in range(25)] +
            [_row("control", "20261103", 0.8) for _ in range(24)])
    summary = cf.summarize(rows)
    assert summary["short_nights"] == ["20261103"]


def test_the_bdd_sample_never_counts_as_a_control():
    rows = [_row("bdd", "bdd", 0.95), _row("control", "20261102", 0.8)]
    summary = cf.summarize(rows)
    assert summary["bdd"]["n"] == 1
    assert set(summary["control_by_night"]) == {"20261102"}


def test_a_night_whose_controls_all_failed_is_named_not_dropped():
    """D9 compares each night against its own controls; a night with none is short.

    Building the night set from the control rows alone would drop it entirely, so
    the report would look complete while one night had no baseline at all.
    """
    rows = ([_row("control", "20261102", 0.8) for _ in range(25)] +
            [_row("uncertain", "20261103", 0.2) for _ in range(3)])
    summary = cf.summarize(rows)
    assert summary["short_nights"] == ["20261103"]
    assert "20261103" not in summary["control_by_night"]


# -- the client, the grid ----------------------------------------------------


def _fake_sdk(monkeypatch, captured):
    class _Minio:
        def __init__(self, *args, **kwargs):
            captured["args"] = args
            captured["kwargs"] = kwargs

    monkeypatch.setitem(sys.modules, "minio", types.SimpleNamespace(Minio=_Minio))
    monkeypatch.setenv("MINIO_ENDPOINT", "minio.example:9000")
    monkeypatch.setenv("MINIO_USER", "u")
    monkeypatch.setenv("MINIO_PASSWORD", "p")
    monkeypatch.setenv("MINIO_SECURE", "false")


def test_the_measurement_client_is_built_with_a_region(monkeypatch):
    """minio-py asks the bucket for its location unless told, and a scoped key may
    not be allowed to answer. The default region is us-east-1 (D6's rule)."""
    captured = {}
    _fake_sdk(monkeypatch, captured)
    monkeypatch.delenv("MINIO_REGION", raising=False)
    cf._minio_client()
    assert captured["kwargs"]["region"] == "us-east-1"


def test_the_measurement_client_honours_minio_region(monkeypatch):
    captured = {}
    _fake_sdk(monkeypatch, captured)
    monkeypatch.setenv("MINIO_REGION", "eu-west-1")
    cf._minio_client()
    assert captured["kwargs"]["region"] == "eu-west-1"
    # Whitespace-only is unset, not a region named " ".
    monkeypatch.setenv("MINIO_REGION", "   ")
    cf._minio_client()
    assert captured["kwargs"]["region"] == "us-east-1"


def _live_checkpoint(tmp_path, input_size):
    """A Checkpoint as fetch_checkpoints builds one: weights plus the parsed sidecar.

    The sidecar is the shape the live artefact carries (20261022_144436): the size
    sits under ``custom_metadata``, not at the top level, so a loader that falls
    back to the 192 default instead of reading it is visible here.
    """
    run = tmp_path / "ckpt" / "live"
    run.mkdir(parents=True)
    (run / "model.pth").write_bytes(b"weights")
    meta = {
        "name": "hcm_cocoa_seg_best",
        "architecture": "hcm",
        "custom_metadata": {"input_size": input_size, "out_stride": 16,
                            "backbone": "convnextv2_tiny.fcmae_ft_in22k_in1k"},
    }
    (run / "metadata.json").write_text(json.dumps(meta))
    return cf.Checkpoint("live", "finetuned-models", "hcm_cocoa_seg_best/20261022_144436",
                         run / "model.pth", SHA_A, meta)


def test_load_models_reads_the_size_from_the_sidecar_next_to_the_weights(monkeypatch, tmp_path):
    """The size comes from the metadata file the fetch wrote, not from the dict.

    Passing the parsed dict where a path is read raised TypeError on the box run
    (10-29 10:15Z), and the tuple resolve_size returns failed the int() after it.
    Only load_checkpoint is replaced: no torch model is built to check a size.
    """
    from ecs_hcm_model.eval import seg

    ckpt = _live_checkpoint(tmp_path, 352)
    monkeypatch.setattr(seg, "load_checkpoint",
                        lambda path, device, **kw: "model:" + ckpt.name)
    logs = []
    models, sizes = cf._load_models({"live": ckpt}, "cpu", logs.append)
    assert sizes["live"] == 352            # the sidecar's size, not the 192 default
    assert models["live"] == "model:live"
    assert "input_size=352" in logs[0]


def test_the_common_grid_is_the_pre_registered_352():
    assert cf.common_grid({"live": 352, "strong": 352}) == 352


def test_a_live_model_off_the_common_grid_refuses_rather_than_rescaling():
    """D8: the grid is pre-registered, not read off the live checkpoint. Re-pointing
    live at a 448 model would otherwise change every number the gate compares."""
    with pytest.raises(cf.MeasurementError) as exc:
        cf.common_grid({"live": 448, "strong": 352})
    assert "352" in str(exc.value) and "448" in str(exc.value)


def test_the_per_frame_csv_carries_the_kind(tmp_path):
    rows = [_row("control", "20261102", 0.8), _row("uncertain", "20261102", 0.3)]
    path = cf.write_frames_csv(rows, tmp_path / "frames.csv")
    lines = path.read_text().strip().splitlines()
    assert lines[0] == "name,night,kind,iou,dice,score,max_p_fg"
    assert [line.split(",")[2] for line in lines[1:]] == ["control", "uncertain"]


# -- the mask path -----------------------------------------------------------

def test_two_identical_masks_agree_completely_and_disjoint_ones_not_at_all():
    a = np.zeros((8, 8), bool)
    a[:4] = True
    assert cf.pair_iou_dice(a, a) == (pytest.approx(1.0), pytest.approx(1.0))
    b = np.zeros((8, 8), bool)
    b[4:] = True
    iou, dice = cf.pair_iou_dice(a, b)
    # The metric's eps=1e-6 denominator leaves a residue of ~1e-8, not a hard 0.
    assert iou == pytest.approx(0.0, abs=1e-6) and dice == pytest.approx(0.0, abs=1e-6)


class _FixedMaskModel:
    """Returns logits whose argmax is a known mask, on the grid it is given."""

    def __init__(self, mask, size):
        self.mask = mask
        self.size = size

    def eval(self):
        return self

    def __call__(self, x):
        import torch
        logits = torch.zeros((1, 2, self.size, self.size))
        # Foreground is confident wherever the mask is set and not where it is
        # not, so argmax picks the mask exactly.
        logits[:, 1] = torch.where(torch.as_tensor(self.mask, dtype=torch.bool), 1.0, -1.0)
        return logits


def test_a_mask_off_the_common_grid_is_brought_to_it_nearest_neighbour():
    """D8: both arms score on the live grid, whichever grid the logits came on."""
    import torch
    import torch.nn.functional as F

    img = np.zeros((48, 64, 3), np.uint8)
    live = np.zeros((88, 88), bool)
    live[20:70, 20:70] = True
    model = _FixedMaskModel(live, 88)
    grid = 88 * 4
    # The preprocessing is the demo's, which needs cv2; this test is about the
    # grid rule, so it supplies a stand-in that returns any correctly shaped input.
    fake_pre = lambda frame, size: torch.zeros(1, 3, size, size)  # noqa: E731
    got = cf.predict_fg(model, img, 352, "cpu", grid, preprocess=fake_pre)
    assert got.shape == (grid, grid)
    assert got.dtype == np.bool_
    # Same content, nearest-neighbour upsample: the block boundaries scale by 4.
    expect = F.interpolate(torch.as_tensor(live).float()[None, None],
                           size=(grid, grid), mode="nearest")
    assert (got == expect[0, 0].numpy().astype(bool)).all()


# -- MLflow ------------------------------------------------------------------

class _FakeMlflow:
    def __init__(self):
        self.params, self.metrics, self.artifacts, self.experiments = {}, {}, [], []

    def set_experiment(self, name):
        self.experiments.append(name)

    def log_param(self, k, v):
        self.params[k] = v

    def log_metric(self, k, v):
        self.metrics[k] = v

    def log_artifact(self, path):
        self.artifacts.append(str(path))

    def start_run(self, run_name=None):
        outer = self

        class _Run:
            info = type("I", (), {"run_id": "run-123"})()

            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

        return _Run()


def test_the_run_carries_both_pins_every_set_and_the_per_frame_csv(tmp_path):
    ckpts = {
        "live": cf.Checkpoint("live", "finetuned-models", "finetuned-models/x/20261022_144436",
                              tmp_path / "a.pth", SHA_A, {}),
        "strong": cf.Checkpoint("strong", "finetuned-models", "finetuned-models/x/20261024_104051",
                                tmp_path / "b.pth", SHA_B, {}),
    }
    rows = ([_row("uncertain", "20261102", 0.2) for _ in range(3)] +
            [_row("control", "20261102", 0.8) for _ in range(30)] +
            [_row("bdd", "bdd", 0.95)])
    csv_path = cf.write_frames_csv(rows, tmp_path / "f.csv")
    fake = _FakeMlflow()
    run_id = cf.log_run(ckpts, cf.summarize(rows), csv_path, mlflow=fake, log=lambda *a: None)

    assert run_id == "run-123"
    assert fake.experiments == ["camera-uncertain-frames"]
    assert fake.params["live_sha256"] == SHA_A and fake.params["strong_sha256"] == SHA_B
    assert fake.params["live_run_id"] == "20261022_144436"
    for name in ("uncertain", "control", "bdd"):
        assert f"{name}_mean" in fake.metrics
        assert f"{name}_n" in fake.metrics
    assert fake.metrics["control_mean_20261102"] == pytest.approx(0.8)
    assert fake.metrics["control_n_20261102"] == 30
    assert fake.artifacts == [str(csv_path)]
    assert fake.metrics["short_nights"] == 0


# ---------------------------------------------------------------------------
# D9: the gate. Every boundary in task 6.2, one test each.
# ---------------------------------------------------------------------------

NIGHTS = ["2026-11-02", "2026-11-03", "2026-11-04", "2026-11-05", "2026-11-06"]


def _frames(night, kind, n, iou):
    return [{"name": f"{night}-{kind}-{i}", "night": night, "kind": kind, "iou": iou}
            for i in range(n)]


def _round(uncertain=("2026-11-02", "2026-11-03", "2026-11-04", "2026-11-05", "2026-11-06"),
           per_night=100, controls=30, u_iou=0.60, c_iou=0.80, extra=None):
    """Five ordinary nights: ``per_night`` uncertain frames and ``controls`` controls each."""
    rows = []
    for night in uncertain:
        rows.extend(_frames(night, "uncertain", per_night, u_iou))
        if controls:
            rows.extend(_frames(night, "control", controls, c_iou))
    return rows + list(extra or [])


def test_a_round_that_clears_every_condition_recommends():
    verdict, reason = cf.gate(_round())
    assert verdict == "recommend"
    assert "500 uncertain frames" in reason and "5 counted nights" in reason


def test_499_saved_frames_stop_and_500_pass():
    """Five counted nights need 500 frames, so 499 is always short somewhere: the
    frame count is what gets reported, because it is the first condition D9 lists."""
    rows = _round()
    last_uncertain = max(i for i, r in enumerate(rows) if r["kind"] == "uncertain")
    verdict, reason = cf.gate(rows[:last_uncertain] + rows[last_uncertain + 1:])
    assert verdict == "stop"
    assert "499 saved uncertain frames" in reason and "needs 500" in reason

    assert cf.gate(rows)[0] == "recommend"


def test_100_frames_count_a_night_and_99_do_not():
    """A night at 100 counts; the same night at 99 is short, reported, and not counted."""
    rows = _round(per_night=101)
    assert cf.gate(rows)[0] == "recommend"

    thin = [r for r in rows if not (r["night"] == NIGHTS[-1] and r["kind"] == "uncertain")]
    thin += _frames(NIGHTS[-1], "uncertain", 99, 0.60)     # 503 frames, 4 counted nights
    verdict, reason = cf.gate(thin)
    assert verdict == "stop"
    assert "only 4 counted nights" in reason and NIGHTS[-1] in reason


def test_24_controls_in_a_counted_night_stop_and_name_that_night():
    rows = _round(controls=30)
    rows = [r for r in rows if not (r["night"] == NIGHTS[2] and r["kind"] == "control")]
    rows += _frames(NIGHTS[2], "control", 24, 0.80)
    verdict, reason = cf.gate(rows)
    assert verdict == "stop"
    assert "insufficient exposure" in reason
    assert f"{NIGHTS[2]} has 24" in reason

    rows += _frames(NIGHTS[2], "control", 1, 0.80)
    assert cf.gate(rows)[0] == "recommend"


def test_24_controls_in_a_short_night_do_not_stop_the_round():
    """A night with fewer than 100 uncertain frames is reported, never counted, never fatal."""
    rows = _round()
    rows += _frames("2026-11-07", "uncertain", 40, 0.60)
    rows += _frames("2026-11-07", "control", 24, 0.90)
    verdict, reason = cf.gate(rows)
    assert verdict == "recommend"
    assert "2026-11-07" not in reason


def test_a_night_exactly_at_the_margin_passes():
    """0.75 - 0.10 is 0.65 in intent and 0.6499999999999999 in binary; the boundary passes."""
    rows = _round(u_iou=0.65, c_iou=0.75)
    verdict, reason = cf.gate(rows)
    assert verdict == "recommend", reason

    rows = _round(u_iou=0.651, c_iou=0.75)
    assert cf.gate(rows)[0] == "stop"


def test_pass_overall_fail_one_night():
    """Pooled passes easily; one night's own controls do not (spec: one bad night fails)."""
    rows = _round()
    rows = [r for r in rows if r["night"] != NIGHTS[4]]
    rows += _frames(NIGHTS[4], "uncertain", 100, 0.60)
    rows += _frames(NIGHTS[4], "control", 30, 0.68)
    verdict, reason = cf.gate(rows)
    assert verdict == "stop"
    assert NIGHTS[4] in reason and "that night's own controls" in reason


def test_the_night_gap_is_not_taken_against_the_pooled_control_mean():
    """Four bright nights lift the pooled mean over the margin; the dim one still fails."""
    rows = _round()
    rows = [r for r in rows if r["night"] != NIGHTS[4]]
    rows += _frames(NIGHTS[4], "uncertain", 100, 0.60)
    rows += _frames(NIGHTS[4], "control", 30, 0.66)

    pooled_uncertain = cf.aggregate(r["iou"] for r in rows if r["kind"] == "uncertain")["mean"]
    pooled_control = cf.aggregate(r["iou"] for r in rows if r["kind"] == "control")["mean"]
    # The flip: against the pooled mean this round would pass, which is why D9 forbids it.
    assert pooled_uncertain <= pooled_control - cf.GATE_MARGIN

    verdict, reason = cf.gate(rows)
    assert verdict == "stop"
    assert NIGHTS[4] in reason


def test_a_short_round_stops_on_the_night_count_with_the_numbers_it_has():
    """Enough frames, not enough nights: the stop names the nights it has."""
    rows = _round(uncertain=NIGHTS[:3], per_night=200)      # 600 frames over 3 nights
    verdict, reason = cf.gate(rows)
    assert verdict == "stop"
    assert "only 3 counted nights, 5 needed" in reason


def test_a_sixth_counted_night_stops_rather_than_being_dropped():
    """The round is exactly five nights; a sixth means capture ran past the stop."""
    rows = _round()
    rows += _frames("2026-11-07", "uncertain", 100, 0.60)
    rows += _frames("2026-11-07", "control", 30, 0.80)
    verdict, reason = cf.gate(rows)
    assert verdict == "stop"
    assert "exactly 5" in reason


def test_validation_frames_never_gate():
    extra = [{"name": "bdd-1", "night": "bdd", "kind": "bdd", "iou": 0.10}]
    assert cf.gate(_round(extra=extra))[0] == "recommend"


# ---------------------------------------------------------------------------
# Task 6.1: the measurement honours the round's recorded stop
#
# The capture ran past the stop, so the bucket holds a sixth night. The gate is
# right to refuse six counted nights; the tool has to be told which night is the
# last one the round counts, or it stops the round for the wrong reason.
# ---------------------------------------------------------------------------

ROUND_NIGHTS = ["20261028", "20261029", "20261030", "20261031", "20261101"]
STOP_NIGHT = "20261101"
POST_STOP_NIGHT = "20261102"


def _write_round_tree(root, nights, *, per_night=100, controls=30):
    """A local tree shaped like the bucket: round0/<night>/<stem>.json + .jpg."""
    base = Path(root) / "round0"
    for night in nights:
        day = base / night
        day.mkdir(parents=True)
        for kind, count in (("uncertain", per_night), ("control", controls)):
            for i in range(count):
                stem = f"{night}-{kind}-{i:03d}"
                (day / f"{stem}.json").write_text(json.dumps({"kind": kind}))
                (day / f"{stem}.jpg").write_bytes(b"")
    return root


def _rows_from(frames, *, u_iou=0.60, c_iou=0.80):
    """The rows scoring would produce for these frames (the gate's input)."""
    return [{"name": f.name, "night": f.night, "kind": f.kind,
             "iou": u_iou if f.kind == "uncertain" else c_iou}
            for f in frames.values()]


def test_frames_after_the_recorded_last_night_are_dropped_before_the_gate(tmp_path):
    """Six nights in the bucket; with the stop recorded the gate counts five."""
    frames = cf.frames_from_dir(_write_round_tree(tmp_path, ROUND_NIGHTS + [POST_STOP_NIGHT]))
    # a44c669's behaviour: every night under round0/ counts, so the run stops.
    verdict, reason = cf.gate(_rows_from(frames))
    assert verdict == "stop" and "exactly 5" in reason

    kept, dropped = cf.exclude_after(frames, STOP_NIGHT)
    assert dropped == {POST_STOP_NIGHT: 130}       # 100 uncertain + 30 controls
    verdict, reason = cf.gate(_rows_from(kept))
    assert verdict == "recommend"
    assert "5 counted nights" in reason


def test_the_summary_records_the_stop_and_what_it_excluded(tmp_path):
    frames = cf.frames_from_dir(_write_round_tree(tmp_path, ROUND_NIGHTS + [POST_STOP_NIGHT]))
    kept, dropped = cf.exclude_after(frames, STOP_NIGHT)
    summary = cf.summarize(_rows_from(kept), last_night=STOP_NIGHT, excluded=dropped)
    assert summary["last_night"] == STOP_NIGHT
    assert summary["excluded_after_stop"] == {POST_STOP_NIGHT: 130}
    assert POST_STOP_NIGHT not in summary["control_by_night"]


def test_without_the_flag_every_night_is_still_counted(tmp_path):
    """No --last-night is the old behaviour, unchanged, and nothing is dropped."""
    frames = cf.frames_from_dir(_write_round_tree(tmp_path, ROUND_NIGHTS + [POST_STOP_NIGHT]))
    kept, dropped = cf.exclude_after(frames, None)
    assert dropped == {} and len(kept) == len(frames)
    assert cf.gate(_rows_from(kept))[0] == "stop"


@pytest.mark.parametrize("bad", ["2026-11-01", "2026111", "202611", "20261301", "", "  "])
def test_a_malformed_last_night_refuses(bad):
    with pytest.raises(cf.MeasurementError):
        cf.parse_last_night(bad)


def test_main_refuses_a_malformed_last_night_before_reading_frames(monkeypatch, tmp_path):
    read = []
    monkeypatch.setattr(cf, "frames_from_dir", lambda *a, **k: read.append(1) or {})
    rc = cf.main(["--frames-dir", str(tmp_path), "--out-dir", str(tmp_path),
                  "--sha256", f"live={SHA_A}", "--sha256", f"strong={SHA_B}",
                  "--last-night", "2026-11-01"])
    assert rc == 2 and read == []


def test_main_drops_the_night_after_the_stop_before_scoring(monkeypatch, tmp_path, capsys):
    """The wiring, not just the helper: main excludes before it scores."""
    frames = cf.frames_from_dir(
        _write_round_tree(tmp_path / "tree", ROUND_NIGHTS + [POST_STOP_NIGHT]))
    sha = {"live": "actual", "strong": "actual"}
    objs = _checkpoint_objects(sha)

    monkeypatch.setattr(cf, "frames_from_dir", lambda *a, **k: frames)
    monkeypatch.setattr(cf, "_minio_client", lambda: FakeMinio(objs))
    monkeypatch.setattr(cf, "_load_models",
                        lambda ckpts, dev, log: ({}, {"live": 352, "strong": 352}))
    monkeypatch.setattr(cf, "common_grid", lambda sizes: 352)
    scored = {}

    def fake_score(models, sizes, frame_list, device, grid, *, fp16=False, log=print):
        scored["nights"] = sorted({f.night for f in frame_list})
        return _rows_from({f.name: f for f in frame_list})

    monkeypatch.setattr(cf, "score_frames", fake_score)
    rc = cf.main(["--frames-dir", str(tmp_path / "tree"), "--out-dir", str(tmp_path / "out"),
                  "--sha256", f"live={sha['live']}", "--sha256", f"strong={sha['strong']}",
                  "--last-night", STOP_NIGHT, "--no-mlflow"])
    assert rc == 0
    assert scored["nights"] == ROUND_NIGHTS
    out = capsys.readouterr().out
    assert f"EXCLUDED 130 frames on night {POST_STOP_NIGHT}" in out


# -- the pin's wiring: launcher -> ConfigMap -> Job env -> run.sh -------------

REPO = Path(__file__).resolve().parents[1]
LAUNCHER = REPO / "deploy" / "jobs" / "launch_camera_frames_measure.sh"
RUN_SH = REPO / "deploy" / "jobs" / "camera_frames_measure.sh"
JOB_YAML = REPO / "deploy" / "jobs" / "camera_frames_measure_job.yaml"


def test_the_job_reads_the_last_night_from_the_pins_configmap():
    doc = yaml.safe_load(JOB_YAML.read_text())
    containers = doc["spec"]["template"]["spec"]["containers"]
    env = {e["name"]: e for e in containers[0]["env"]}
    assert env["CAMERA_ROUND0_LAST_NIGHT"]["valueFrom"]["configMapKeyRef"] == {
        "name": "camera-round0-pins", "key": "last_night"}


def test_run_sh_requires_the_pin_and_passes_it_to_the_tool():
    text = RUN_SH.read_text()
    assert subprocess.run(["bash", "-n", str(RUN_SH)]).returncode == 0
    assert "${CAMERA_ROUND0_LAST_NIGHT:?" in text          # required, like the shas
    assert '--last-night "${CAMERA_ROUND0_LAST_NIGHT}"' in text


def test_the_launcher_refuses_a_last_night_that_is_not_a_date():
    # No REPO= is passed on purpose: a bad argument must refuse before the launcher
    # needs a checkout, because CI runs this copy from an export at /tmp/src.
    r = subprocess.run(
        ["bash", str(LAUNCHER), "--live-sha256", SHA_A, "--strong-sha256", SHA_B,
         "--last-night", "20260231", "--dry-run"],
        capture_output=True, text=True)
    assert r.returncode == 2
    assert "--last-night" in r.stderr and "REFUSE" in r.stderr


def test_the_launcher_writes_the_last_night_to_the_pins_configmap():
    text = LAUNCHER.read_text()
    assert "--from-literal=last_night=$LAST_NIGHT" in text
    assert '--last-night)    LAST_NIGHT=$2' in text
