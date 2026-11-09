"""The Jetson window harness: what it reads off the wire, and the three verdicts.

Nothing here starts a demo or opens a socket. The harness exists to be run by
hand on the box, so these tests pin the parts it would otherwise get wrong
silently: a warm-up that leaks into the window, a part split across two reads
counted twice, a 95% boundary decided by binary floating point, a cap verdict
that passes because nothing was ever tried.
"""

import ast
from pathlib import Path

import pytest

from jetson import verify_capture_windows as vcw

REPO = Path(__file__).resolve().parents[1]


# --------------------------------------------------------------------------
# Synthetic wire bytes
# --------------------------------------------------------------------------

def _jpeg(width=640, height=480, payload=200):
    app0 = (b"\xff\xe0" + (16).to_bytes(2, "big")
            + b"JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00")
    sof = (b"\xff\xc0" + (17).to_bytes(2, "big") + b"\x08"
           + height.to_bytes(2, "big") + width.to_bytes(2, "big")
           + b"\x03" + b"\x01\x22\x00\x02\x11\x01\x03\x11\x01")
    sos = (b"\xff\xda" + (12).to_bytes(2, "big")
           + b"\x03\x01\x00\x02\x11\x03\x11\x00\x3f\x00")
    return b"\xff\xd8" + app0 + sof + sos + b"\x00" * payload + b"\xff\xd9"


def _part(frame):
    return vcw.BOUNDARY + b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"


class _Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


def _reader(clock, chunks, step=0.5):
    """Each read advances the clock by ``step``, then returns the next chunk."""
    pending = list(chunks)

    def read():
        clock.t += step
        return pending.pop(0) if pending else b""

    return read


# --------------------------------------------------------------------------
# jpeg_size
# --------------------------------------------------------------------------

class TestJpegSize:
    def test_a_start_of_frame_marker_carries_the_resolution(self):
        assert vcw.jpeg_size(_jpeg(640, 480)) == (640, 480)
        assert vcw.jpeg_size(_jpeg(1280, 720)) == (1280, 720)

    def test_bytes_that_are_not_a_jpeg_are_refused_not_guessed(self):
        with pytest.raises(ValueError):
            vcw.jpeg_size(b"not a jpeg at all")

    def test_a_scan_before_any_frame_header_reports_unknown(self):
        assert vcw.jpeg_size(b"\xff\xd8" + b"\xff\xda\x00\x0c" + b"\x00" * 32) == (-1, -1)


# --------------------------------------------------------------------------
# measure
# --------------------------------------------------------------------------

class TestMeasure:
    def test_the_warm_up_is_not_counted(self):
        clock = _Clock()
        read = _reader(clock, [_part(_jpeg())] * 12, step=0.5)
        stats = vcw.measure(read, clock=clock, warmup=2.0, window=4.0)
        # Reads land at 0.5 s .. 6.0 s; the window is [2.0, 6.0), so 8 of the 12.
        assert stats["parts"] == 8
        assert stats["fps"] == pytest.approx(2.0)
        assert stats["dims"] == [(640, 480)]
        assert stats["median_bytes"] == len(_jpeg())

    def test_a_part_split_across_two_reads_counts_once(self):
        clock = _Clock()
        whole = _part(_jpeg())
        cut = len(vcw.BOUNDARY) + 20
        read = _reader(clock, [whole[:cut], whole[cut:]] + [_part(_jpeg())] * 60)
        stats = vcw.measure(read, clock=clock, warmup=1.0, window=6.0)
        # A part is complete when the next boundary arrives, so the split one is
        # counted on read 3 (1.5 s) and one more per read after it; the window
        # closes after read 13 (6.5 s), which leaves 11.
        assert stats["parts"] == 11

    def test_a_stream_that_closes_inside_the_window_is_an_error(self):
        clock = _Clock()
        read = _reader(clock, [_part(_jpeg())])
        with pytest.raises(EOFError):
            vcw.measure(read, clock=clock, warmup=1.0, window=10.0)

    def test_bytes_with_no_part_boundary_yield_an_empty_window(self):
        clock = _Clock()
        read = _reader(clock, [b"x" * 16] * 100)
        stats = vcw.measure(read, clock=clock, warmup=1.0, window=3.0)
        assert stats["parts"] == 0 and stats["fps"] == 0.0
        assert stats["median_bytes"] == 0 and stats["dims"] == []


# --------------------------------------------------------------------------
# The running demo
# --------------------------------------------------------------------------

def _proc_entry(root, pid, argv, env=None):
    entry = Path(root) / str(pid)
    entry.mkdir(parents=True)
    (entry / "cmdline").write_bytes(b"\0".join(a.encode() for a in argv) + b"\0")
    if env is not None:
        (entry / "environ").write_bytes(
            b"\0".join(("%s=%s" % kv).encode() for kv in env.items()) + b"\0")
    return entry


class TestFindDemo:
    def test_the_python_running_the_script_is_found(self, tmp_path):
        _proc_entry(tmp_path, 4101, ["/usr/bin/python3", "-B", "jetson/realtime_demo.py"])
        assert vcw.find_demo(tmp_path) == 4101

    def test_a_shell_whose_text_mentions_the_script_is_not_the_demo(self, tmp_path):
        _proc_entry(tmp_path, 4102, ["/bin/sh", "-c", "python3 jetson/realtime_demo.py"])
        assert vcw.find_demo(tmp_path) is None

    def test_the_searcher_can_exclude_itself(self, tmp_path):
        _proc_entry(tmp_path, 4103, ["/usr/bin/python3", "jetson/verify_capture_windows.py"])
        _proc_entry(tmp_path, 4104, ["/usr/bin/python3", "jetson/realtime_demo.py"])
        assert vcw.find_demo(tmp_path) == 4104
        assert vcw.find_demo(tmp_path, exclude=(4104,)) is None

    def test_the_discovered_demo_can_be_rebuilt_exactly(self, tmp_path):
        _proc_entry(tmp_path, 4105,
                    ["/usr/bin/python3", "-B", "jetson/realtime_demo.py", "--engine", "e"],
                    env={"PYTHONPATH": "/home/user/repos/ecs_hcm_model",
                         "MINIO_ENDPOINT": "10.0.1.93:30591"})
        demo = vcw.Demo.discover(root=tmp_path)
        assert demo.argv[2] == "jetson/realtime_demo.py"
        assert demo.env["MINIO_ENDPOINT"] == "10.0.1.93:30591"

    def test_a_process_that_is_gone_has_no_argv_and_no_environment(self, tmp_path):
        assert vcw.read_argv(9999, root=tmp_path) is None
        assert vcw.read_environ(9999, root=tmp_path) == {}


class TestWithCapture:
    ARGV = ["/usr/bin/python3", "jetson/realtime_demo.py", "--engine", "e"]

    def _demo(self):
        return vcw.Demo(1, self.ARGV, {"PYTHONPATH": "/x"}, None)

    def test_the_on_arm_appends_the_flag_and_keeps_the_rest(self, monkeypatch):
        monkeypatch.setattr(vcw, "minio_env", lambda path=vcw.MINIO_ENV_FILE:
                            {"MINIO_ENDPOINT": "10.0.1.93:30591",
                             "MINIO_USER": "hcm-camera-upload"})
        argv, env = self._demo().with_capture("/tmp/w02-on")
        assert argv[-2:] == [vcw.CAPTURE_FLAG, "/tmp/w02-on"]
        assert argv[2:4] == ["--engine", "e"]
        assert env["MINIO_USER"] == "hcm-camera-upload"
        assert env["PYTHONPATH"] == "/x"

    def test_an_explicit_override_wins_over_the_env_file(self, monkeypatch):
        monkeypatch.setattr(vcw, "minio_env", lambda path=vcw.MINIO_ENV_FILE:
                            {"MINIO_ENDPOINT": "10.0.1.93:30591"})
        _argv, env = self._demo().with_capture("/tmp/x",
                                               {"MINIO_ENDPOINT": vcw.UNREACHABLE})
        assert env["MINIO_ENDPOINT"] == vcw.UNREACHABLE

    def test_a_demo_that_already_captures_cannot_be_the_off_arm(self):
        demo = vcw.Demo(1, self.ARGV + [vcw.CAPTURE_FLAG, "/tmp/already"], {}, None)
        with pytest.raises(vcw.CaptureRefused) as err:
            demo.with_capture("/tmp/new")
        # The same sentence main() prints, so the two cannot drift apart.
        assert str(err.value) == vcw.CAPTURE_ALREADY_ON

    def test_the_env_file_is_parsed_as_key_value(self, tmp_path):
        # The exact format on the box: bare KEY=VALUE, the file is not sourced.
        path = tmp_path / "minio.env"
        path.write_text("# camera upload user\n"
                        "MINIO_ENDPOINT=10.0.1.93:30591\n"
                        "MINIO_USER=hcm-camera-upload\n"
                        "MINIO_PASSWORD_FILE=/home/user/.config/hcm-camera/pw\n"
                        "MINIO_SECURE\n")
        env = vcw.minio_env(path)
        assert env["MINIO_ENDPOINT"] == "10.0.1.93:30591"
        assert env["MINIO_PASSWORD_FILE"].endswith("/pw")
        assert "MINIO_SECURE" not in env          # a bare word is not a variable

    def test_an_absent_env_file_is_empty_not_an_error(self, tmp_path):
        assert vcw.minio_env(tmp_path / "nope") == {}


class TestCaptureVarsPresent:
    def test_the_demo_environment_alone_is_enough(self):
        assert vcw.capture_vars_present({"MINIO_ENDPOINT": "e", "MINIO_USER": "u"},
                                        path="/nonexistent")

    def test_a_password_file_stands_in_for_the_password(self):
        assert vcw.capture_vars_present({"MINIO_ENDPOINT": "e",
                                         "MINIO_PASSWORD_FILE": "/f"},
                                        path="/nonexistent")

    def test_a_missing_endpoint_is_reported_missing(self):
        assert not vcw.capture_vars_present({"MINIO_USER": "u"}, path="/nonexistent")

    def test_a_missing_user_is_reported_missing(self):
        assert not vcw.capture_vars_present({"MINIO_ENDPOINT": "e"}, path="/nonexistent")


# --------------------------------------------------------------------------
# The verdicts
# --------------------------------------------------------------------------

class TestSlowdownVerdict:
    def test_a_parity_run_passes(self):
        passed, text = vcw.slowdown_verdict([12.0, 12.2, 12.1], [12.0, 12.2, 12.1])
        assert passed and text.startswith("PASS")

    def test_exactly_95_percent_passes_despite_the_float_boundary(self):
        # 0.95 * 20.0 is 19.000000000000004 in binary floating point.
        passed, text = vcw.slowdown_verdict([20.0, 20.0, 20.0], [19.0, 19.0, 19.0])
        assert passed, text

    def test_below_95_percent_fails_and_reports_both_arms(self):
        passed, text = vcw.slowdown_verdict([20.0, 20.0, 20.0], [18.5, 18.6, 18.7])
        assert not passed and text.startswith("FAIL")
        assert "spread" in text and "off windows" in text and "on windows" in text

    def test_the_medians_are_medians_not_means(self):
        # A mean would be dragged down by the one slow off window and pass this.
        passed, _ = vcw.slowdown_verdict([20.0, 20.0, 4.0], [19.0, 19.0, 19.0])
        assert passed


class TestFeedVerdict:
    def _stats(self, dims=((640, 480),), median=52000):
        return {"dims": list(dims), "median_bytes": median}

    def test_the_same_feed_passes(self):
        ok, text = vcw.feed_verdict(self._stats(), self._stats(median=54000))
        assert ok and "unchanged" in text

    def test_a_changed_resolution_fails(self):
        ok, text = vcw.feed_verdict(self._stats(), self._stats(dims=((1280, 720),)))
        assert ok is False and "resolution" in text

    def test_a_frame_size_outside_the_band_fails(self):
        ok, text = vcw.feed_verdict(self._stats(), self._stats(median=40000))
        assert ok is False and "median frame" in text

    def test_no_baseline_is_a_skip_not_a_pass(self):
        ok, text = vcw.feed_verdict(None, self._stats())
        assert ok is None and text.startswith("SKIPPED")


class TestCapVerdict:
    def test_under_the_cap_with_refusals_counted_passes(self):
        ok, text = vcw.cap_verdict(1000, {"bytes": 900, "frames": 3, "markers": 0},
                                   {"disk_cap": 2})
        assert ok and text.startswith("PASS")

    def test_over_the_cap_fails_even_with_refusals(self):
        ok, text = vcw.cap_verdict(1000, {"bytes": 1200, "frames": 4, "markers": 0},
                                   {"disk_cap": 2})
        assert ok is False and text.startswith("FAIL")

    def test_never_reaching_the_cap_is_inconclusive_not_a_pass(self):
        ok, text = vcw.cap_verdict(1000, {"bytes": 500, "frames": 1, "markers": 0}, {})
        assert ok is None and text.startswith("INCONCLUSIVE")

    def test_the_directory_is_measured_from_what_is_on_disk(self, tmp_path):
        day = tmp_path / "20261102"
        day.mkdir()
        (day / "a.jpg").write_bytes(b"x" * 100)
        (day / "a.json").write_bytes(b"y" * 50)
        (day / "a.jpg.uploaded").write_bytes(b"")
        assert vcw.dir_usage(tmp_path) == {"files": 3, "bytes": 150,
                                           "frames": 1, "markers": 1}

    def test_an_absent_directory_is_empty_not_an_exception(self, tmp_path):
        assert vcw.dir_usage(tmp_path / "nope")["bytes"] == 0


class TestStatsLine:
    def test_the_last_stats_line_wins(self, tmp_path):
        log = tmp_path / "demo.log"
        log.write_text("device: cuda\n"
                       "capture stats: {\"offered\": 1}\n"
                       "device: cuda\n"
                       "capture stats: {\"disk_cap\": 3, \"offered\": 9}\n")
        assert vcw.last_capture_stats(log) == {"disk_cap": 3, "offered": 9}

    def test_a_log_without_counters_reports_none(self, tmp_path):
        log = tmp_path / "demo.log"
        log.write_text("device: cuda\n")
        assert vcw.last_capture_stats(log) is None
        assert vcw.last_capture_stats(tmp_path / "absent.log") is None

    def test_a_truncated_line_does_not_hide_the_last_good_one(self, tmp_path):
        log = tmp_path / "demo.log"
        log.write_text("capture stats: {\"disk_cap\": 2}\ncapture stats: {\"disk\n")
        assert vcw.last_capture_stats(log) == {"disk_cap": 2}


# --------------------------------------------------------------------------
# The plan, and the dry run
# --------------------------------------------------------------------------

class TestPlan:
    def test_the_windows_alternate_starting_with_capture_off(self):
        assert vcw.window_plan(3) == ["off", "on", "off", "on", "off", "on"]
        assert vcw.window_plan(1) == ["off", "on"]

    def test_a_dry_run_prints_every_command_and_starts_nothing(self, monkeypatch, capsys):
        def boom(*a, **k):
            raise AssertionError("a dry run must not start a process")

        monkeypatch.setattr(vcw.subprocess, "Popen", boom)
        rc = vcw.main(["--dry-run",
                       "--argv", "python3 jetson/realtime_demo.py --engine /models/e",
                       "--pairs", "1", "--capture-dir", "/tmp/r0-dry"])
        out = capsys.readouterr().out
        assert rc == 0
        assert out.count("[dry-run] start:") == 3       # two 7.2 windows and 7.4
        assert out.count(vcw.CAPTURE_FLAG) == 2         # the on arm and 7.4
        assert "MINIO_ENDPOINT=%s" % vcw.UNREACHABLE in out
        assert "CAPTURE_DISK_CAP_BYTES=131072" in out

    def test_a_capturing_demo_is_refused_before_any_window(self, monkeypatch, capsys):
        def boom(*a, **k):
            raise AssertionError("nothing may start once the demo is refused")

        monkeypatch.setattr(vcw.subprocess, "Popen", boom)
        argv = ("python3 jetson/realtime_demo.py --engine /models/e %s /tmp/already"
                % vcw.CAPTURE_FLAG)
        common = ["--dry-run", "--argv", argv, "--capture-dir", "/tmp/r0-ref"]
        assert vcw.main(common) == 2
        assert vcw.CAPTURE_ALREADY_ON in capsys.readouterr().err
        # 7.1 needs no on-arm, so the same demo is usable there.
        assert vcw.main(common + ["--only", "7.1"]) == 0

    def test_no_demo_and_no_argv_stops_before_running_anything(self, monkeypatch, capsys):
        monkeypatch.setattr(vcw, "find_demo", lambda *a, **k: None)
        assert vcw.main(["--dry-run"]) == 2
        assert "pass --argv" in capsys.readouterr().err


# --------------------------------------------------------------------------
# Stopping, waiting for memory, and proving the restore
# --------------------------------------------------------------------------

class _FakeClock:
    """A clock that only moves when the code sleeps."""

    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now

    def sleep(self, seconds):
        self.now += seconds


class _Resp:
    def __init__(self, chunks=()):
        self._chunks = list(chunks)

    def read(self, n=-1):
        return self._chunks.pop(0) if self._chunks else b""

    def close(self):
        pass


class _Box:
    """A box whose starts either serve a frame or do not, in a scripted order.

    ``serves`` has one entry per start: a start that serves answers every read with
    a frame, one that does not refuses every connection.
    """

    def __init__(self, serves=(), dry=False, baseline=0, **kw):
        self.dry = dry
        self.script = list(serves)
        self.serving = False
        self.baseline = baseline        # reads of the feed that was already running
        self.starts = []
        self.stopped = []
        self.stop_pids = []
        self.stop_pid_result = vcw.StopResult(True, True)

    def start(self, argv, env, cwd, log_path, env_delta=None):
        self.starts.append(str(log_path))
        if self.script:
            self.serving = bool(self.script.pop(0))
        return object()

    def read_stream(self, port, timeout):
        if self.baseline:
            self.baseline -= 1
            return _Resp([_jpeg()])
        if self.serving:
            return _Resp([_jpeg()])
        raise OSError("connection refused")

    def stop(self, proc, timeout=40.0):
        self.stopped.append(proc)
        return vcw.StopResult(True, True)

    def stop_pid(self, pid, timeout=40.0):
        self.stop_pids.append(pid)
        self.stopped.append(pid)
        return self.stop_pid_result

    def close(self, resp):
        pass


class TestMemoryWait:
    def test_a_reading_back_at_the_old_level_is_released(self, monkeypatch):
        clock = _FakeClock()
        monkeypatch.setattr(vcw, "mem_available", lambda *a, **k: 1000)
        assert vcw.wait_for_memory(1000, clock=clock, sleep=clock.sleep) == (True, 1000)

    def test_a_reading_below_the_old_level_is_waited_for(self, monkeypatch):
        monkeypatch.setattr(vcw, "MEM_SLACK_BYTES", 0)
        clock = _FakeClock()
        readings = iter([100, 300, 700, 1000])
        monkeypatch.setattr(vcw, "mem_available", lambda *a, **k: next(readings))
        released, seen = vcw.wait_for_memory(1000, poll=1.0, clock=clock,
                                             sleep=clock.sleep)
        assert (released, seen) == (True, 1000)
        assert clock.now == 3.0            # it actually waited

    def test_a_box_that_never_recovers_is_reported_not_waited_on_forever(
            self, monkeypatch):
        monkeypatch.setattr(vcw, "MEM_SLACK_BYTES", 0)
        clock = _FakeClock()
        monkeypatch.setattr(vcw, "mem_available", lambda *a, **k: 10)
        released, seen = vcw.wait_for_memory(1000, timeout=5.0, poll=1.0,
                                             clock=clock, sleep=clock.sleep)
        assert released is False and seen == 10
        assert clock.now == 5.0            # bounded

    def test_the_slack_allows_a_reading_a_little_below(self, monkeypatch):
        monkeypatch.setattr(vcw, "mem_available", lambda *a, **k: 1000)
        slack = vcw.MEM_SLACK_BYTES
        assert vcw.wait_for_memory(1000 + slack // 2)[0] is True

    def test_an_unreadable_meminfo_does_not_block_the_run(self, monkeypatch):
        monkeypatch.setattr(vcw, "mem_available", lambda *a, **k: None)
        assert vcw.wait_for_memory(1000) == (True, None)

    def test_meminfo_is_parsed_in_bytes(self, tmp_path):
        path = tmp_path / "meminfo"
        path.write_text("MemTotal: 8000000 kB\nMemAvailable:   4096 kB\n")
        assert vcw.mem_available(path) == 4096 * 1024
        path.write_text("MemTotal: 8000000 kB\n")
        assert vcw.mem_available(path) is None
        assert vcw.mem_available(tmp_path / "absent") is None


class TestBoxStop:
    def test_a_clean_stop_reports_its_memory(self, monkeypatch):
        monkeypatch.setattr(vcw, "mem_available", lambda *a, **k: 1000)
        monkeypatch.setattr(vcw, "stop", lambda proc, timeout: True)
        box = vcw.Box(dry=False)
        result = box.stop(object())
        assert (result.clean, result.released) == (True, True)
        assert result.describe() == "stopped cleanly"

    def test_a_killed_process_is_not_a_clean_stop(self, monkeypatch):
        monkeypatch.setattr(vcw, "mem_available", lambda *a, **k: 1000)
        monkeypatch.setattr(vcw, "stop", lambda proc, timeout: False)
        result = vcw.Box(dry=False).stop(object())
        assert result.clean is False
        assert "killed" in result.describe()

    def test_a_stop_waits_on_the_reading_taken_before_it(self, monkeypatch):
        """The comparison is against the memory in use *while the demo ran*."""
        monkeypatch.setattr(vcw, "mem_available", lambda *a, **k: 777)
        seen = []

        def _wait(before, **kw):
            seen.append(before)
            return True, 777

        monkeypatch.setattr(vcw, "wait_for_memory", _wait)
        monkeypatch.setattr(vcw, "stop", lambda proc, timeout: True)
        vcw.Box(dry=False).stop(object())
        assert seen == [777]

    def test_a_dry_run_stop_starts_nothing(self):
        box = vcw.Box(dry=True, echo=lambda m: None)
        assert box.stop(None).clean is True
        assert box.stop_pid(123).clean is True


class TestFeedServes:
    def test_a_serving_feed_is_true_at_once(self):
        box = _Box()
        box.serving = True
        assert vcw.feed_serves(box, 5000, timeout=5.0) is True

    def test_a_refused_connection_is_not_serving(self):
        box = _Box(serves=[])
        assert vcw.feed_serves(box, 5000, timeout=0.0) is False

    def test_the_wait_is_bounded(self):
        clock = _FakeClock()
        box = _Box(serves=[])
        assert vcw.feed_serves(box, 5000, timeout=6.0, poll=2.0,
                               clock=clock, sleep=clock.sleep) is False
        assert clock.now <= 6.0

    def test_a_port_that_is_not_http_is_not_an_exception(self):
        """`BadStatusLine` is an HTTPException, not an OSError: it must not escape.

        This runs inside restore_demo's "never raises" promise, in main's finally --
        an escape there replaces the report's "feed is DOWN" with a traceback.
        """
        import http.client

        class _NotHttp(_Box):
            def read_stream(self, port, timeout):
                raise http.client.BadStatusLine("garbage not http")

        box = _NotHttp()
        clock = _FakeClock()
        assert vcw.feed_serves(box, 5000, timeout=2.0, poll=1.0,
                               clock=clock, sleep=clock.sleep) is False
        ok, text = vcw.restore_demo(box, vcw.Demo(1, ["python3", "d.py"], {}, "/r"),
                                    "/tmp/r0", timeout=0.0, settle=0.0)
        assert ok is False and "FAILED restore" in text

    def test_a_read_error_mid_frame_is_not_serving(self):
        class _Broken(_Box):
            def read_stream(self, port, timeout):
                class _Resp:
                    def read(self, n=-1):
                        raise ConnectionResetError("reset mid-frame")
                    def close(self):
                        pass
                return _Resp()

        clock = _FakeClock()
        assert vcw.feed_serves(_Broken(), 5000, timeout=2.0, poll=1.0,
                               clock=clock, sleep=clock.sleep) is False


class _Serving:
    """A response whose reads advance the clock, like a stream at 2 fps."""

    def __init__(self, read):
        self._read = read
        self.closed = False

    def read1(self, n=-1):
        return self._read()

    def close(self):
        self.closed = True


class TestConnectRetry:
    """The connect happens while the window's demo is still importing its stack.

    Box run 2 (10-28 19:34Z) failed window 1 on the first `ConnectionResetError`:
    the demo had not reached `app.run` yet. That is the normal state of the first
    seconds of every window, not an error, and it cost a whole run.
    """

    def test_a_demo_still_loading_is_retried_until_it_serves(self):
        clock = _FakeClock()
        parts = [_part(_jpeg())] * 40

        def read():
            clock.now += 0.5
            return parts.pop(0) if parts else b""

        served = _Serving(read)
        answers = [ConnectionRefusedError(111, "Connection refused"),
                   ConnectionResetError(104, "Connection reset by peer"),
                   served]

        def attempt(port, timeout):
            answer = answers.pop(0)
            if isinstance(answer, Exception):
                raise answer
            return answer

        resp = vcw.connect_stream(5000, timeout=180.0, clock=clock,
                                  sleep=clock.sleep, attempt=attempt)
        assert resp is served
        stats = vcw.measure(vcw.stream_reader(resp), clock=clock, warmup=0.0, window=2.0)
        assert stats["parts"] > 0          # the window measured instead of ending the run

    def test_a_port_that_never_serves_stops_at_the_deadline(self):
        clock = _FakeClock()
        tries = []

        def attempt(port, timeout):
            tries.append(timeout)
            raise ConnectionRefusedError(111, "Connection refused")

        with pytest.raises(ConnectionRefusedError):
            vcw.connect_stream(5000, timeout=10.0, clock=clock, sleep=clock.sleep,
                               attempt=attempt)
        assert clock.now >= 10.0
        assert 5 <= len(tries) <= 12       # ~1.5 s apart, and it does stop

    def test_a_non_200_is_retried_and_then_names_the_status(self, monkeypatch):
        """A listener that answers but is not the feed yet is also a warm-up state."""
        clock = _FakeClock()
        made = []

        class _Answer:
            def __init__(self, status):
                self.status = status
                self.closed = False

            def close(self):
                self.closed = True

        class _Conn:
            def __init__(self, host, port, timeout=None):
                self.answer = None

            def request(self, method, path):
                assert (method, path) == ("GET", "/video")
                self.answer = _Answer(503 if len(made) < 2 else 200)
                made.append(self.answer)

            def getresponse(self):
                return self.answer

            def close(self):
                pass

        monkeypatch.setattr(vcw.http.client, "HTTPConnection", _Conn)
        resp = vcw.connect_stream(5000, timeout=60.0, clock=clock, sleep=clock.sleep)
        assert resp.status == 200
        assert [a.status for a in made] == [503, 503, 200]
        assert all(a.closed for a in made[:2])     # a refused answer is not leaked

    def test_a_non_200_that_never_clears_says_which_status(self, monkeypatch):
        clock = _FakeClock()

        class _Conn:
            def __init__(self, host, port, timeout=None):
                pass

            def request(self, method, path):
                pass

            def getresponse(self):
                class _Answer:
                    status = 503

                    def close(self):
                        pass
                return _Answer()

            def close(self):
                pass

        monkeypatch.setattr(vcw.http.client, "HTTPConnection", _Conn)
        with pytest.raises(vcw.StreamNotReady) as caught:
            vcw.connect_stream(5000, timeout=5.0, clock=clock, sleep=clock.sleep)
        assert "returned 503" in str(caught.value)


class TestRestoreDemo:
    def _demo(self):
        return vcw.Demo(4242, ["python3", "jetson/realtime_demo.py"],
                        {"A": "1"}, "/repo")

    def test_the_demo_returning_before_is_restored(self):
        box = _Box(serves=[True])
        clock = _FakeClock()
        ok, text = vcw.restore_demo(box, self._demo(), "/tmp/r0", timeout=5.0,
                                    clock=clock, sleep=clock.sleep)
        assert ok is True
        assert "restored and serving" in text
        assert box.starts[-1].endswith("restored.log")

    def test_a_start_that_does_not_serve_is_retried_and_reported(self):
        box = _Box(serves=[False, False])
        ok, text = vcw.restore_demo(box, self._demo(), "/tmp/r0", timeout=5.0,
                                    settle=0.0)
        assert ok is False
        assert text.startswith("FAILED restore")
        assert "feed is DOWN" in text
        assert len(box.starts) == 2
        assert box.starts[0].endswith("restored.log")
        assert box.starts[1].endswith("restored-2.log")
        # A start that never served is taken down again, not left holding the GPU.
        assert len(box.stopped) == 2

    def test_a_start_that_serves_on_the_second_attempt_is_restored(self):
        box = _Box(serves=[False, True])
        clock = _FakeClock()
        ok, text = vcw.restore_demo(box, self._demo(), "/tmp/r0", timeout=5.0,
                                    settle=0.0, clock=clock, sleep=clock.sleep)
        assert ok is True and "restored and serving" in text
        assert box.starts[-1].endswith("restored-2.log")

    def test_a_start_that_cannot_run_is_reported_not_raised(self):
        class _Broken(_Box):
            def start(self, *a, **k):
                raise OSError("no such file")

        ok, text = vcw.restore_demo(_Broken(), self._demo(), "/tmp/r0",
                                    timeout=0.0, settle=0.0)
        assert ok is False and "start 1 failed: no such file" in text


class TestMainRestore:
    """The restarted demo must serve; the run says so, and says so loudly when not."""

    def _run(self, monkeypatch, tmp_path, serves, only="7.4", raises=None,
             baseline=1):
        monkeypatch.setattr(vcw, "measure", lambda *a, **k: {
            "parts": 3, "fps": 1.0, "median_bytes": 100, "min_bytes": 90,
            "max_bytes": 110, "dims": [(640, 480)], "window": 1.0, "warmup": 0.0})
        monkeypatch.setattr(vcw, "FEED_TIMEOUT", 0.05)
        # The settle between restore attempts is 20 s of real sleep; the tests that
        # do not measure the wait itself should not spend it.
        monkeypatch.setattr(vcw, "RESTORE_SETTLE", 0.0)
        box = _Box(serves=serves, baseline=baseline)
        monkeypatch.setattr(vcw, "Box", lambda **kw: box)
        demo = vcw.Demo(4242, ["python3", "jetson/realtime_demo.py"],
                        {"MINIO_ENDPOINT": "x", "MINIO_USER": "u"}, "/repo")
        monkeypatch.setattr(vcw.Demo, "discover",
                            classmethod(lambda cls, pid=None, root=None: demo))
        out = tmp_path / "report.txt"
        argv = ["--only", only, "--capture-dir", str(tmp_path / "cap"),
                "--out", str(out), "--baseline-window", "1",
                "--cap-window", "1"]
        if raises is not None:
            with pytest.raises(raises):
                vcw.main(argv)
            return None, out.read_text(), box
        return vcw.main(argv), out.read_text(), box

    def test_a_demo_that_comes_back_serving_is_reported_restored(
            self, monkeypatch, tmp_path):
        # 7.4 starts one demo of its own, then the restore starts another.
        rc, report, box = self._run(monkeypatch, tmp_path, [True, True])
        assert rc == 0
        assert "restored and serving" in report
        assert box.starts[-1].endswith("restored.log")

    def test_a_demo_that_never_serves_fails_the_run(self, monkeypatch, tmp_path):
        rc, report, box = self._run(monkeypatch, tmp_path, [True, False, False])
        assert rc == 4
        assert "FAILED restore" in report
        assert "feed is DOWN" in report
        assert len(box.starts) == 3          # the window, then one retry, named
        assert box.starts[-1].endswith("restored-2.log")

    def test_a_window_whose_demo_never_serves_ends_the_run_nonzero(
            self, monkeypatch, tmp_path):
        """The observed box path: the probe raises, the run stops, the feed comes back.

        Exit 1 (the traceback's status) is the behaviour the box showed on 10-28,
        and the point of pinning it is that the error is not swallowed: a run that
        turned it into a 0 would report a verification it never measured.
        """
        # No baseline with --only 7.4, so every read belongs to the window.
        _, report, box = self._run(monkeypatch, tmp_path, [False, False],
                                   raises=OSError, baseline=0)
        assert "/restored" in box.starts[-1]                # the feed was put back
        assert "feed" in report.lower()

    def test_a_run_that_never_stopped_the_demo_does_not_claim_a_restore(
            self, monkeypatch, tmp_path):
        """--only 7.1 starts nothing, so the old process is still on the port.

        Starting a second demo there and calling it "restored" would be the port
        talking, not the process this run started.
        """
        rc, report, box = self._run(monkeypatch, tmp_path, [], only="7.1")
        assert rc == 0
        assert "restored" not in report
        assert "never stopped" in report
        assert box.starts == []
        assert box.stop_pids == []


class TestNoHeavyImports:
    def test_the_harness_pulls_in_no_torch_or_cv2(self):
        tree = ast.parse((REPO / "jetson" / "verify_capture_windows.py").read_text())
        imported = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module.split(".")[0])
        assert not imported & {"torch", "cv2", "numpy", "flask", "jetson"}
