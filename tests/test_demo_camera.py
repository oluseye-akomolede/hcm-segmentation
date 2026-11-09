"""Camera-read retries and the rate-limited encode warning (jetson/capture.py)."""

from pathlib import Path

import pytest

from jetson.capture import (
    READ_RETRIES,
    CameraReadError,
    RateLimitedWarning,
    read_with_retry,
)

REPO = Path(__file__).resolve().parents[1]


def _reader(results):
    calls = iter(results)
    return lambda: next(calls)


class TestReadWithRetry:
    def test_a_good_first_read_returns_the_frame(self):
        assert read_with_retry(_reader([(True, "f")]), sleep=lambda _s: None) == "f"

    def test_a_brief_hiccup_is_ridden_out(self):
        sleeps = []
        frame = read_with_retry(
            _reader([(False, None), (True, None), (True, "f")]), sleep=sleeps.append
        )
        assert frame == "f"
        assert len(sleeps) == 2

    def test_a_camera_that_stays_gone_raises_naming_the_tries(self):
        sleeps = []
        with pytest.raises(CameraReadError, match=f"{READ_RETRIES} times"):
            read_with_retry(_reader([(False, None)] * READ_RETRIES), sleep=sleeps.append)
        # No sleep after the last try.
        assert len(sleeps) == READ_RETRIES - 1

    def test_it_never_reads_past_the_budget(self):
        reads = []

        def read():
            reads.append(1)
            return False, None

        with pytest.raises(CameraReadError):
            read_with_retry(read, retries=3, sleep=lambda _s: None)
        assert len(reads) == 3


class TestRateLimitedWarning:
    def _warning(self):
        now = [0.0]
        out = []
        w = RateLimitedWarning("encode failed", interval=10.0, clock=lambda: now[0], emit=out.append)
        return w, now, out

    def test_the_first_failure_warns(self):
        w, _now, out = self._warning()
        w()
        assert out == ["WARNING: encode failed"]

    def test_failures_inside_the_interval_are_counted_not_printed(self):
        w, now, out = self._warning()
        w()
        for t in (1.0, 2.0, 9.9):
            now[0] = t
            w()
        assert len(out) == 1
        now[0] = 10.0
        w()
        assert out[1] == "WARNING: encode failed (3 more since the last warning)"

    def test_the_count_resets_after_a_warning(self):
        w, now, out = self._warning()
        w()
        now[0] = 1.0
        w()
        now[0] = 10.0
        w()
        now[0] = 25.0
        w()
        assert out[-1] == "WARNING: encode failed"


class TestTheDemoUsesThem:
    SRC = (REPO / "jetson" / "realtime_demo.py").read_text()

    def test_the_loop_reads_through_the_retry(self):
        assert "read_with_retry(cap.read)" in self.SRC

    def test_a_lost_camera_exits_nonzero(self):
        assert "except CameraReadError" in self.SRC
        assert "raise SystemExit(f\"Stopping: {e}\")" in self.SRC
        assert 'print("camera read failed")' not in self.SRC

    def test_an_encode_failure_warns(self):
        assert "encode_failed()" in self.SRC
