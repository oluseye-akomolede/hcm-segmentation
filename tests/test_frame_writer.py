"""Writer queue, naming, index and the disk cap (jetson/frame_writer.py)."""

import json
import time
from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np
import pytest

from jetson.frame_writer import (
    INDEX_NAME,
    QUEUE_MAX,
    UPLOADED_SUFFIX,
    FrameWriter,
    frame_name,
)
from jetson.uncertain_frames import CONTROL, UNCERTAIN, Counters, SaveDecision

LOCAL = ZoneInfo("America/Chicago")


def _ts(y, mo, d, h=0, mi=0, s=0):
    return datetime(y, mo, d, h, mi, s, tzinfo=LOCAL).timestamp()


def _frame():
    return np.zeros((480, 640, 3), dtype=np.uint8)


def _writer(tmp_path, **kw):
    kw.setdefault("encode", lambda frame: b"\xff\xd8" + b"x" * 998)
    kw.setdefault("counts", Counters())
    return FrameWriter(tmp_path, **kw)


def _write(writer, decision, **kw):
    name = writer.offer(decision, _frame(), kw.get("score", 0.5), kw.get("p_max", 0.9),
                        fps=kw.get("fps", 17.3))
    assert name is not None
    return name


def _await(path, timeout=10.0):
    """Wait for the writer thread to finish this file (``flush`` also stops it)."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if path.exists():
            return
        time.sleep(0.01)
    raise AssertionError(f"{path} was never written")


class TestNaming:
    def test_the_name_is_the_utc_epoch_in_ms(self):
        assert frame_name(1_700_000_000.123) == "1700000000123"

    def test_a_control_is_prefixed(self):
        assert frame_name(1_700_000_000.123, CONTROL) == "control-1700000000123"

    def test_the_name_sorts_by_time(self):
        later = _ts(2026, 11, 2, 12, 0, 30)
        earlier = _ts(2026, 11, 2, 12, 0, 0)
        assert sorted([frame_name(later), frame_name(earlier)]) == [
            frame_name(earlier), frame_name(later)]

    def test_the_name_matches_its_record(self):
        decision = SaveDecision(UNCERTAIN, _ts(2026, 11, 2, 12, 0, 0))
        assert frame_name(decision.ts, decision.kind) == f"{int(round(decision.ts * 1000))}"


class TestQueue:
    def test_the_caller_is_never_blocked_when_writing_is_stuck(self, tmp_path):
        # No writer thread: the queue is exactly as blocked as a stalled disk,
        # so every offer after the queue's depth must return immediately.
        writer = FrameWriter(tmp_path, encode=lambda f: b"x", counts=Counters())
        offered = [writer.offer(SaveDecision(UNCERTAIN, 1.0 + i), _frame(), 0.5, 0.9)
                   for i in range(QUEUE_MAX + 1)]
        assert offered[:QUEUE_MAX] != [None] * QUEUE_MAX
        assert offered[-1] is None
        assert writer.counts.get("capture_queue_full") == 1
        assert writer.counts.get("written", 0) == 0


class TestFiles:
    def test_a_frame_and_its_record_and_its_index_line_are_written(self, tmp_path):
        writer = _writer(tmp_path).start()
        ts = _ts(2026, 11, 2, 12, 0, 0)
        name = _write(writer, SaveDecision(UNCERTAIN, ts), score=0.42, p_max=0.88)
        writer.flush()
        day = "20261102"
        assert (tmp_path / day / f"{name}.jpg").exists()
        side = json.loads((tmp_path / day / f"{name}.json").read_text())
        index = json.loads((tmp_path / INDEX_NAME).read_text().splitlines()[0])
        assert side == index
        assert side["kind"] == UNCERTAIN
        assert side["file"] == f"{name}.jpg"
        assert side["score"] == pytest.approx(0.42)
        assert side["max_p_fg"] == pytest.approx(0.88)
        assert side["fps"] == pytest.approx(17.3)

    def test_the_record_carries_what_the_spec_names(self, tmp_path):
        writer = _writer(tmp_path, context={"model": "model.pth", "input_size": 352}).start()
        name = _write(writer, SaveDecision(UNCERTAIN, _ts(2026, 11, 2, 12, 0, 0)))
        writer.flush()
        doc = json.loads((tmp_path / "20261102" / f"{name}.json").read_text())
        for field in ("file", "kind", "ts_utc", "score", "max_p_fg", "fps",
                      "model", "input_size"):
            assert field in doc

    def test_a_control_keeps_its_kind_in_the_index(self, tmp_path):
        writer = _writer(tmp_path).start()
        _write(writer, SaveDecision(CONTROL, _ts(2026, 11, 2, 12, 0, 0)))
        writer.flush()
        index = json.loads((tmp_path / INDEX_NAME).read_text().splitlines()[0])
        assert index["kind"] == CONTROL
        assert index["file"].startswith("control-")

    def test_frames_straddling_local_midnight_land_in_different_directories(self, tmp_path):
        writer = _writer(tmp_path).start()
        midnight = _ts(2026, 11, 2, 0, 0, 0)
        _write(writer, SaveDecision(UNCERTAIN, midnight - 1), score=0.5)
        _write(writer, SaveDecision(UNCERTAIN, midnight), score=0.5)
        writer.flush()
        dirs = sorted(p.name for p in tmp_path.iterdir() if p.is_dir())
        assert dirs == ["20261101", "20261102"]

    def test_the_saved_image_is_the_frame_bytes_the_encoder_returned(self, tmp_path):
        marker = b"RAWFRAME"
        writer = FrameWriter(tmp_path, encode=lambda frame: marker).start()
        name = _write(writer, SaveDecision(UNCERTAIN, _ts(2026, 11, 2, 12, 0, 0)))
        writer.flush()
        assert (tmp_path / "20261102" / f"{name}.jpg").read_bytes() == marker


class TestDiskCap:
    def _cap(self, tmp_path, frames):
        writer = FrameWriter(tmp_path, encode=lambda f: b"x" * 60_000,
                             disk_cap=frames * 70_000, counts=Counters()).start()
        return writer

    def test_the_cap_holds_and_the_drop_is_counted(self, tmp_path):
        writer = self._cap(tmp_path, frames=2)
        for i in range(4):
            writer.offer(SaveDecision(UNCERTAIN, _ts(2026, 11, 2, 12, 0, 0) + i),
                         _frame(), 0.5, 0.9)
        writer.flush()
        caps = writer.counts.get("disk_cap")
        assert caps >= 1
        assert writer.counts.get("written") + caps == 4
        total = sum(p.stat().st_size for p in tmp_path.rglob("*") if p.is_file())
        assert total <= 2 * 70_000

    def test_an_uploaded_frame_is_pruned_before_a_capture_is_skipped(self, tmp_path):
        writer = self._cap(tmp_path, frames=2)
        ts = _ts(2026, 11, 2, 12, 0, 0)
        first = writer.offer(SaveDecision(UNCERTAIN, ts), _frame(), 0.5, 0.9)
        second = writer.offer(SaveDecision(UNCERTAIN, ts + 1), _frame(), 0.5, 0.9)
        # Confirm the first upload, as the uploader would; the writer takes jobs in
        # order, so this marker exists before the third frame is looked at.
        (tmp_path / "20261102").mkdir(exist_ok=True)
        (tmp_path / "20261102" / f"{first}.jpg{UPLOADED_SUFFIX}").write_text("")
        third = writer.offer(SaveDecision(UNCERTAIN, ts + 2), _frame(), 0.5, 0.9)
        writer.flush()
        assert third is not None
        assert writer.counts.get("disk_pruned") == 1
        assert not (tmp_path / "20261102" / f"{first}.jpg").exists()
        assert (tmp_path / "20261102" / f"{second}.jpg").exists()

    def test_a_frame_the_uploader_removed_does_not_hold_the_cap(self, tmp_path):
        """The uploader deletes a confirmed frame and its sidecar, then leaves the marker.

        Nothing charges those bytes back, so the writer's counter overstates the
        directory until it re-measures. Without the re-measure this refuses a frame
        while the disk is nearly empty, turning the cap into a bound on bytes ever
        written.
        """
        writer = self._cap(tmp_path, frames=2)
        day = tmp_path / "20261102"
        ts = _ts(2026, 11, 2, 12, 0, 0)
        first = _write(writer, SaveDecision(UNCERTAIN, ts))
        _await(day / f"{first}.jpg")
        # Exactly what jetson/frame_uploader.py leaves behind after a confirmed put.
        (day / f"{first}.jpg{UPLOADED_SUFFIX}").write_text("")
        (day / f"{first}.jpg").unlink()
        (day / f"{first}.json").unlink()

        second = _write(writer, SaveDecision(UNCERTAIN, ts + 1))
        _await(day / f"{second}.jpg")
        third = _write(writer, SaveDecision(UNCERTAIN, ts + 2))
        _await(day / f"{third}.jpg")
        writer.flush()

        assert writer.counts.get("disk_cap") == 0
        assert writer.counts.get("disk_pruned") == 0
        assert (day / f"{second}.jpg").exists()
        assert writer.disk_bytes() <= 2 * 70_000

    def test_an_unconfirmed_frame_is_never_pruned(self, tmp_path):
        writer = self._cap(tmp_path, frames=2)
        ts = _ts(2026, 11, 2, 12, 0, 0)
        names = [writer.offer(SaveDecision(UNCERTAIN, ts + i), _frame(), 0.5, 0.9)
                 for i in range(2)]
        writer.flush()
        assert writer.prune_uploaded() == 0
        for name in names:
            assert (tmp_path / "20261102" / f"{name}.jpg").exists()

    def test_a_resumed_directory_is_measured_before_the_cap_is_applied(self, tmp_path):
        writer = _writer(tmp_path).start()
        _write(writer, SaveDecision(UNCERTAIN, _ts(2026, 11, 2, 12, 0, 0)))
        writer.flush()
        before = writer.disk_bytes()
        assert before > 0
        again = _writer(tmp_path)
        assert again.disk_bytes() == before
