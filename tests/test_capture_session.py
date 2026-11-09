"""The capture session the demo builds, and the demo's wiring of it."""

import ast
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from jetson.capture_session import CaptureConfigError, CaptureSession, build_capture
from jetson.frame_writer import FrameWriter
from jetson.uncertain_frames import CapturePolicy, Counters, local_day

REPO = Path(__file__).resolve().parents[1]
REALTIME = (REPO / "jetson" / "realtime_demo.py").read_text()

ENV = {"MINIO_ENDPOINT": "minio.minio.svc.cluster.local:9000",
       "MINIO_USER": "hcm-camera-upload",
       "MINIO_PASSWORD": "s3cret"}


def _frame():
    return np.zeros((480, 640, 3), dtype=np.uint8)


def _refusing_put(key, path, content_type):
    raise ConnectionError("bucket unreachable")


class TestOptIn:
    def test_without_the_flag_nothing_is_constructed(self, monkeypatch):
        def boom(*a, **k):
            raise AssertionError("capture must not be constructed without the flag")
        monkeypatch.setattr(CaptureSession, "from_env", boom)
        assert build_capture(None) is None

    def test_the_flag_defaults_to_off_and_takes_a_directory(self):
        # Read at the source: importing the demo needs a camera's OpenCV build.
        tree = ast.parse(REALTIME)
        args = [n for n in ast.walk(tree) if isinstance(n, ast.Call)
                and getattr(n.func, "attr", "") == "add_argument"]
        flag = [n for n in args if n.args and ast.unparse(n.args[0]) == "'--capture-uncertain'"]
        assert len(flag) == 1
        kwargs = {k.arg: ast.unparse(k.value) for k in flag[0].keywords}
        assert kwargs["default"] == "None"


class TestStartupRefusal:
    def test_a_missing_variable_stops_the_capture_before_it_starts(self, tmp_path):
        with pytest.raises(CaptureConfigError) as err:
            build_capture(tmp_path, {"MINIO_ENDPOINT": ENV["MINIO_ENDPOINT"]})
        assert "MINIO_USER" in str(err.value)

    def test_the_chosen_scheme_is_reported(self, tmp_path, capsys):
        session = build_capture(tmp_path, ENV, put=_refusing_put, scan_period=3600)
        try:
            assert "http://minio" in capsys.readouterr().out
        finally:
            session.stop()
        secure = dict(ENV, MINIO_SECURE="yes")
        session = build_capture(tmp_path, secure, put=_refusing_put, scan_period=3600)
        try:
            assert "https://minio" in capsys.readouterr().out
        finally:
            session.stop()


class TestDiskCapOverride:
    """7.4 needs a cap it can reach in minutes, not the 512 MiB of a real round."""

    def test_unset_leaves_the_writer_default(self, tmp_path):
        from jetson.capture_session import _disk_cap_from_env
        from jetson.frame_writer import DISK_CAP_BYTES
        assert _disk_cap_from_env({}) == DISK_CAP_BYTES
        assert _disk_cap_from_env({"CAPTURE_DISK_CAP_BYTES": "  "}) == DISK_CAP_BYTES

    def test_a_value_is_taken_as_bytes(self):
        from jetson.capture_session import _disk_cap_from_env
        assert _disk_cap_from_env({"CAPTURE_DISK_CAP_BYTES": "131072"}) == 131072

    @pytest.mark.parametrize("raw", ["lots", "0", "-5"])
    def test_a_cap_that_cannot_hold_a_frame_stops_the_capture(self, tmp_path, raw):
        env = dict(ENV, CAPTURE_DISK_CAP_BYTES=raw)
        with pytest.raises(CaptureConfigError) as err:
            build_capture(tmp_path, env)
        assert "CAPTURE_DISK_CAP_BYTES" in str(err.value)

    def test_the_override_reaches_the_writer(self, tmp_path):
        env = dict(ENV, CAPTURE_DISK_CAP_BYTES="131072")
        session = build_capture(tmp_path, env, put=_refusing_put, scan_period=3600)
        try:
            assert session.writer.disk_cap == 131072
        finally:
            session.stop()


class TestShutdownCounters:
    def _line(self, capsys):
        lines = [ln for ln in capsys.readouterr().out.splitlines()
                 if ln.startswith("capture stats: ")]
        assert len(lines) == 1, lines
        return json.loads(lines[0][len("capture stats: "):])

    def test_stop_prints_the_counters_in_one_parseable_line(self, tmp_path, capsys):
        session = build_capture(tmp_path, ENV, put=_refusing_put, scan_period=3600)
        session.stop()
        stats = self._line(capsys)
        # 7.4 reads the drop counters out of this line; a counter only appears
        # once it has been bumped, so the always-present keys are the writer's.
        assert "disk_bytes" in stats and "upload_pending" in stats

    def test_a_counted_drop_is_in_the_line(self, tmp_path, capsys):
        ts = 1_800_000_000.0
        counts = Counters()
        # No control sample, so the one offer is the uncertain frame being tested.
        policy = CapturePolicy(clock=lambda: ts, counts=counts, control_period=1e9)
        writer = FrameWriter(tmp_path, counts=counts, encode=lambda f: b"JPEG" * 100,
                             disk_cap=1)
        session = CaptureSession(tmp_path, policy=policy, writer=writer,
                                 counts=counts, clock=lambda: ts).start()
        p_fg = torch.zeros(4, 4)
        p_fg[::2] = 0.5                          # half the pixels undecided
        try:
            # offer() returns at enqueue; the cap is decided on the writer thread.
            session.offer(_frame(), p_fg)
        finally:
            session.stop()
        assert self._line(capsys)["disk_cap"] == 1


class TestOffer:
    def _session(self, tmp_path, ts):
        counts = Counters()
        return CaptureSession(
            tmp_path,
            policy=CapturePolicy(clock=lambda: ts, counts=counts, control_period=1e9),
            writer=FrameWriter(tmp_path, counts=counts, encode=lambda f: b"JPEG"),
            counts=counts,
            clock=lambda: ts,
        ).start()

    def test_a_frame_is_saved_when_the_score_is_high(self, tmp_path):
        session = self._session(tmp_path, 1_800_000_000.0)
        try:
            p_fg = torch.zeros(352, 352)
            p_fg[::2] = 0.5                     # half the pixels undecided
            name = session.offer(_frame(), p_fg, fps=15.0)
            assert name is not None
            session.writer.flush()
            day = local_day(1_800_000_000.0)
            doc = json.loads((tmp_path / day / f"{name}.json").read_text())
            assert doc["score"] == pytest.approx(0.5, abs=1e-2)
            assert doc["fps"] == pytest.approx(15.0)
        finally:
            session.stop()

    def test_the_score_used_is_the_one_the_policy_sees(self, tmp_path, monkeypatch):
        ts = 1_800_000_000.0
        seen = []
        session = self._session(tmp_path, ts)
        monkeypatch.setattr(session.policy, "offer",
                            lambda score, now=None: seen.append(score) or None)
        p_fg = torch.full((4, 4), 0.5)
        session.offer(_frame(), p_fg)
        assert seen == [pytest.approx(1.0)]


class TestDemoWiring:
    """The demo's call path, pinned at the source: the score is the overlay's tensor."""

    def _loop(self):
        tree = ast.parse(REALTIME)
        for node in ast.walk(tree):
            if isinstance(node, ast.While):
                return node
        raise AssertionError("no loop in realtime_demo.py")

    def test_the_capture_is_offered_the_frame_and_the_overlay_tensor(self):
        calls = [n for n in ast.walk(self._loop())
                 if isinstance(n, ast.Call) and getattr(n.func, "attr", "") == "offer"]
        assert len(calls) == 1
        args = [ast.unparse(a) for a in calls[0].args]
        assert args[:2] == ["frame", "p_fg"]

    def test_the_score_comes_from_the_only_forward_in_the_loop(self):
        loop = self._loop()
        forwards = [n for n in ast.walk(loop)
                    if isinstance(n, ast.Call)
                    and ast.unparse(n.func) in {"engine.infer", "demo_forward"}]
        assert len(forwards) == 2      # exactly one branch each, no extra pass
        names = {ast.unparse(t) for n in ast.walk(loop) if isinstance(n, ast.Assign)
                 for t in n.targets}
        assert "p_fg" in names

    def test_the_frame_is_saved_before_the_overlay_is_drawn(self):
        loop = self._loop()
        lines = sorted(n.lineno for n in ast.walk(loop)
                       if isinstance(n, ast.Call) and getattr(n.func, "attr", "") == "offer")
        overlay = [n.lineno for n in ast.walk(loop)
                   if isinstance(n, ast.Name) and n.id == "out"]
        assert lines[0] < min(overlay)
