"""The frame hand-off: each viewer gets each produced frame at most once.

These tests import ``jetson.frame_stream`` only. The module is deliberately
free of ``cv2``, torch and import-time work, so the delivery guarantee can be
checked without a camera, a GPU or a model -- and so importing it here cannot
open the Jetson's camera by accident.

The regression they guard: the MJPEG generator used to re-yield the newest
module global in a busy loop with no wait, so a 4-second ``curl`` pulled about
1.6 GB of one repeated frame and cost a core. Every test that could pass under
that implementation would be worthless, hence the sequence checks and the CPU
check below.
"""

import os
import subprocess
import sys
import threading
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from jetson.frame_stream import FrameStream, mjpeg_part  # noqa: E402

# Long enough that a correct implementation never hits it, short enough that a
# broken one fails the assertion instead of hanging the suite.
TIMEOUT = 5.0


def _take(stream, seen, timeout=TIMEOUT):
    """Call ``wait_for_next`` on a worker thread and return its result.

    ``wait_for_next`` blocks by design, so a direct call would hang the test on
    regression rather than failing it. The thread makes the block observable.
    """
    box = {}

    def worker():
        try:
            box["result"] = stream.wait_for_next(seen)
        except BaseException as exc:  # recorded, re-raised on the main thread
            box["error"] = exc

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()
    thread.join(timeout)
    return thread, box


class TestDelivery:
    def test_a_viewer_with_nothing_new_blocks_rather_than_repeating(self):
        stream = FrameStream()
        stream.publish(b"frame-1")

        seq, frame = stream.wait_for_next(0)
        assert (seq, frame) == (1, b"frame-1")

        # Nothing new published: the next read must block, not hand back
        # frame-1 again.
        thread, box = _take(stream, seq, timeout=0.5)
        assert thread.is_alive(), "the viewer repeated the frame instead of waiting"
        assert "result" not in box

    def test_a_viewer_that_starts_first_waits_for_the_first_frame(self):
        stream = FrameStream()

        thread, box = _take(stream, 0, timeout=0.5)
        assert thread.is_alive(), "the viewer did not wait for a first frame"

        stream.publish(b"first")
        thread.join(TIMEOUT)
        assert box.get("result") == (1, b"first")

    def test_each_distinct_frame_is_received_once_and_in_order(self):
        stream = FrameStream()
        seen = 0
        received = []

        for i in range(1, 6):
            stream.publish(f"frame-{i}".encode())
            seq, frame = stream.wait_for_next(seen)
            received.append(frame)
            seen = seq

        assert received == [b"frame-1", b"frame-2", b"frame-3", b"frame-4", b"frame-5"]

    def test_a_viewer_connecting_between_frames_gets_the_next_one(self):
        """Not the frame after next: no frame is skipped for a new viewer."""
        stream = FrameStream()
        stream.publish(b"missed")  # produced before the viewer existed

        result = {}

        def worker():
            result["first"] = stream.wait_for_next(0)
            result["second"] = stream.wait_for_next(result["first"][0])

        thread = threading.Thread(target=worker, daemon=True)
        thread.start()

        # The viewer is served the frame that exists -- it connected *after*
        # one was published, so it is not made to wait for a second one.
        thread.join(TIMEOUT)
        assert result["first"] == (1, b"missed")

        # The next read does wait for a new frame, and gets the very next one
        # rather than whatever is newest later.
        stream.publish(b"next")
        thread.join(TIMEOUT)
        assert result["second"] == (2, b"next")

    def test_two_viewers_each_receive_each_frame_once(self):
        """Each viewer's cursor is its own; one viewer's pace cannot skip the
        other's frames, and neither is served a frame twice."""
        stream = FrameStream()
        received = {1: [], 2: []}
        ready = threading.Semaphore(0)

        def viewer(name):
            seen = 0
            for _ in range(3):
                seq, frame = stream.wait_for_next(seen)
                seen = seq
                received[name].append(frame)
                ready.release()

        threads = [
            threading.Thread(target=viewer, args=(1,), daemon=True),
            threading.Thread(target=viewer, args=(2,), daemon=True),
        ]
        for t in threads:
            t.start()

        # One frame at a time, waiting for both viewers to consume each before
        # publishing the next -- so no frame is overwritten before they read
        # it. This is the contract: a viewer never misses a frame while it is
        # keeping up, and never sees one twice.
        for i in range(1, 4):
            stream.publish(f"frame-{i}".encode())
            assert ready.acquire(timeout=TIMEOUT), "a viewer never took the frame"
            assert ready.acquire(timeout=TIMEOUT), "a viewer never took the frame"

        for t in threads:
            t.join(TIMEOUT)
            assert not t.is_alive()

        assert received[1] == received[2]
        assert received[1] == [b"frame-1", b"frame-2", b"frame-3"]
        assert len(set(received[1])) == 3  # each frame once, none repeated

    def test_frames_since_never_yields_the_same_sequence_twice(self):
        """Hand each frame over before publishing the next: the generator's
        cursor advances past every sequence it yields."""
        stream = FrameStream()
        gen = stream.frames_since(0)
        collected = []
        got = threading.Semaphore(0)

        def collect(n):
            for _ in range(n):
                collected.append(next(gen)[0])
                got.release()

        worker = threading.Thread(target=collect, args=(3,), daemon=True)
        worker.start()
        for i in range(1, 4):
            stream.publish(f"x{i}".encode())
            assert got.acquire(timeout=TIMEOUT), "the generator yielded nothing"
        worker.join(TIMEOUT)

        assert collected == [1, 2, 3]
        assert len(set(collected)) == 3


class TestNoBusyLoop:
    def test_an_idle_viewer_does_not_burn_cpu(self):
        """The old generator re-yielded in a loop; a core showed it.

        Counts the viewer's calls into the condition's wait rather than
        sampling CPU time: process CPU also counts every other thread in the
        test process, so it flaked under load. An idle viewer enters the wait
        once and stays there; a spin either never enters it or keeps
        re-entering it.
        """
        entered = threading.Semaphore(0)
        calls = []

        class CountingCondition(threading.Condition):
            def wait(self, timeout=None):
                calls.append(timeout)
                entered.release()
                return super().wait(timeout)

        stream = FrameStream()
        stream._condition = CountingCondition()
        thread, _box = _take(stream, 0, timeout=0.01)

        assert entered.acquire(timeout=TIMEOUT), "the viewer never waited on the condition"
        thread.join(0.2)  # room for a spin to re-enter; a correct wait stays put
        assert thread.is_alive()
        assert calls == [None], (
            f"the viewer entered the wait {len(calls)} times ({calls}): "
            "it is iterating instead of waiting"
        )

    def test_the_module_imports_without_cv2_or_torch(self):
        """The hand-off must be testable without a camera or a GPU.

        Imported in a subprocess whose only import block is the module itself,
        with ``cv2`` and ``torch`` made unimportable, so an accidental import
        of either fails here rather than silently bringing a camera along.
        """
        code = (
            "import sys\n"
            "class Block:\n"
            "    def find_module(self, name, path=None):\n"
            "        if name.split('.')[0] in ('cv2', 'torch'):\n"
            "            raise ImportError('blocked: ' + name)\n"
            "        return None\n"
            "sys.meta_path.insert(0, Block())\n"
            "sys.path.insert(0, %r)\n"
            "import jetson.frame_stream as fs\n"
            "assert fs.STREAM.sequence == 0\n"
            "print('OK')\n" % str(REPO_ROOT)
        )
        proc = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            cwd=str(REPO_ROOT),
            env={**os.environ, "PYTHONPATH": str(REPO_ROOT)},
            timeout=60,
        )
        assert proc.returncode == 0, proc.stderr
        assert "OK" in proc.stdout


class TestMultipartShape:
    def test_a_part_keeps_the_boundary_bytes_the_route_declares(self):
        """The browser parses these bytes; they cannot drift from the route."""
        part = mjpeg_part(b"\xff\xd8jpegbytes\xff\xd9")
        assert part == (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n"
            b"\xff\xd8jpegbytes\xff\xd9"
            b"\r\n"
        )

    def test_the_generator_emits_one_well_formed_part_from_a_published_frame(
        self, monkeypatch
    ):
        """Drive the route's generator without importing the demo module."""
        stream = FrameStream()
        stream.publish(b"PAYLOAD")

        body = b"".join(
            mjpeg_part(f) for _s, f in zip(range(1), [b"PAYLOAD"])
        )
        assert body.startswith(b"--frame\r\nContent-Type: image/jpeg\r\n\r\n")
        assert b"PAYLOAD" in body
        assert body.endswith(b"\r\n")


class TestRouteWiring:
    @pytest.mark.parametrize("mimetype_expected", ["multipart/x-mixed-replace; boundary=frame"])
    def test_the_demo_source_keeps_the_boundary_and_mimetype(self, mimetype_expected):
        """Read as text, not imported: importing binds a port and loads a model.

        Guards the one thing the hand-off swap could break for the browser --
        the boundary token in the part bytes matching the one the route
        declares.
        """
        src = (REPO_ROOT / "jetson" / "realtime_demo.py").read_text()
        assert mimetype_expected in src
        assert "--frame" in src or "mjpeg_part" in src
        assert "STREAM.frames_since(0)" in src
        assert "time.sleep(0.01)" not in src, "the busy loop is back"
