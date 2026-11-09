"""Tasks 7.1, 7.2 and 7.4 on the Jetson: is the feed unchanged, is the loop as
fast with capture on, and does the disk cap hold when the bucket is gone?

Run by hand on the box:

    python3 -B jetson/verify_capture_windows.py --capture-dir ~/round0-verify/w1

It imports nothing from torch, cv2 or the demo. Its job is to start the demo
exactly as it is already running, count MJPEG parts off the wire, and write one
text report. Each window is a fresh demo process, because capture is chosen at
startup and cannot be switched on inside a running one.

The command line, environment and working directory are read from the running
demo's ``/proc`` entry before the first window and restored at the end -- on an
exception too -- so the box is left as it was found. "Restored" means the restarted
demo *serves a frame*: a start is not a restore, and one retry is allowed because a
TensorRT engine dies at warm-up when it asks for its workspace while the previous
demo's memory is still being released. Each stop waits for MemAvailable to return
to its pre-stop reading, bounded, and a memory or restore failure is reported in
the report rather than hidden. When no demo is running, ``--argv`` names the
command instead; 7.1 then has no pre-change feed to compare against and says so
rather than passing itself.

``--dry-run`` prints every command it would run and executes none of them.

Exit codes: 0 ran, 2 refused (no demo, capture already on, no credentials), 3 the
running demo would not stop, 4 the demo could not be brought back serving. 4 is the
completed run's code: one that raised on the way still says in its report that the
feed is DOWN, but exits with the traceback's own status.

7.1 uses the first capture-off window of 7.2 (a window is the expensive part),
so it needs 7.2 in ``--only``.
"""

from __future__ import annotations

import argparse
import http.client
import json
import os
import signal
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

BOUNDARY = b"--frame\r\n"
PART_HEADER_END = b"\r\n\r\n"
# The demo binds this port unconditionally; a check that compares ports can only
# confirm that the same one is served.
PORT = 5000
DEMO_SCRIPT = "realtime_demo.py"
CAPTURE_FLAG = "--capture-uncertain"
# One sentence, used by the pre-flight check in main() and by with_capture(), so a
# demo that is already capturing is refused the same way wherever it is noticed.
CAPTURE_ALREADY_ON = ("the demo already runs with %s, so it cannot be the 'capture "
                      "off' arm: restart it without the flag, or run --only 7.1"
                      % CAPTURE_FLAG)
STATS_PREFIX = "capture stats: "
SLOWDOWN_RATIO = 0.95
# "At least 95%" has a boundary that has to pass, and 0.95 * 20 is
# 19.000000000000004 in binary floating point. Absolute, like the gate's.
EPS = 1e-9
# Frame byte size moves with what the camera is looking at, so the size check is
# a band; resolution and port are exact.
SIZE_TOLERANCE = 0.10
SOF_MARKERS = frozenset((0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7,
                         0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF))
# Port 9 (discard) is closed on the Jetson: the connect fails at once instead of
# hanging on an unroutable address.
UNREACHABLE = "127.0.0.1:9"
MINIO_ENV_FILE = "~/.config/hcm-camera/minio.env"
# A stopped demo does not hand its memory back at once. TensorRT asks for its
# workspace in one large block, so a demo started on a box whose previous demo is
# still being torn down dies at warm-up -- "OutOfMemory (Requested size was
# 62213632 bytes)" at 18:33Z on 10-28 -- and the feed stayed down. Wait for
# MemAvailable to return to what it was before the stop, bounded, and report it
# when it does not.
MEMINFO = "/proc/meminfo"
MEM_TIMEOUT = 120.0
MEM_POLL = 1.0
MEM_SLACK_BYTES = 32 * 1024 * 1024
# A restarted demo is not "restored" until it serves a frame: the same run wrote
# "restored:" about a process that had already died, and nobody noticed the feed
# was down for minutes.
FEED_TIMEOUT = 180.0
RESTORE_ATTEMPTS = 2
RESTORE_SETTLE = 20.0


# --------------------------------------------------------------------------
# Reading the wire
# --------------------------------------------------------------------------

def jpeg_size(frame):
    """``(width, height)`` from a JPEG's start-of-frame marker, or ``(-1, -1)``.

    Read out of the encoded bytes so the harness needs no image library: the
    check is that capture-off serves the resolution it always did.
    """
    if not frame.startswith(b"\xff\xd8"):
        raise ValueError("not a JPEG (no start-of-image marker)")
    n = len(frame)
    i = 2
    while i + 3 < n:
        if frame[i] != 0xFF:
            i += 1
            continue
        marker = frame[i + 1]
        if marker == 0xFF:                  # fill byte
            i += 1
            continue
        if marker == 0x01 or 0xD0 <= marker <= 0xD9:
            i += 2                          # standalone markers carry no length
            continue
        if marker == 0xDA:                  # a scan began before any SOF
            break
        length = int.from_bytes(frame[i + 2:i + 4], "big")
        if marker in SOF_MARKERS and i + 9 <= n:
            height = int.from_bytes(frame[i + 5:i + 7], "big")
            width = int.from_bytes(frame[i + 7:i + 9], "big")
            return width, height
        if length < 2:
            break
        i += 2 + length
    return -1, -1


def measure(read, clock=time.monotonic, warmup=30.0, window=60.0):
    """Count the MJPEG parts that arrive in ``window`` seconds, after ``warmup``.

    Parts are counted as they complete, so the number is frames the demo
    actually published -- not frames a viewer asked for, and not a rate the
    loop reports about itself.
    """
    begin = clock() + warmup
    end = begin + window
    pending = b""
    times = []
    sizes = []
    dims = set()
    while True:
        if clock() >= end:
            break
        chunk = read()
        if not chunk:
            raise EOFError("the stream closed before the window ended")
        pending += chunk
        parts = pending.split(BOUNDARY)
        pending = parts.pop()               # trailing, still incomplete
        for part in parts:
            _, sep, payload = part.partition(PART_HEADER_END)
            if not sep:
                continue
            payload = payload.rstrip(b"\r\n")
            if len(payload) < 2:
                continue
            arrived = clock()
            if arrived < begin or arrived >= end:
                continue
            times.append(arrived)
            sizes.append(len(payload))
            try:
                dims.add(jpeg_size(payload))
            except ValueError:
                dims.add((-1, -1))
    return {
        "parts": len(times),
        "fps": len(times) / float(window),
        "median_bytes": int(statistics.median(sizes)) if sizes else 0,
        "min_bytes": min(sizes) if sizes else 0,
        "max_bytes": max(sizes) if sizes else 0,
        "dims": sorted(dims),
        "window": window,
        "warmup": warmup,
    }


def stream_reader(resp, chunk=65536):
    """A ``read`` that returns what has arrived, not the whole (endless) body."""
    read1 = getattr(resp, "read1", None)
    if read1 is not None:
        return lambda: read1(chunk)
    return lambda: resp.read(chunk)


CONNECT_RETRY_S = 1.5


class StreamNotReady(RuntimeError):
    """``/video`` answered, but not with a stream."""


# What a connect may raise while the demo behind the port is still starting.
RETRYABLE = (ConnectionError, http.client.HTTPException, StreamNotReady)


def _open_stream(port, timeout):
    """One attempt at ``GET /video``, closing its connection if it does not serve."""
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=timeout)
    try:
        conn.request("GET", "/video")
        resp = conn.getresponse()
    except BaseException:
        conn.close()
        raise
    if resp.status != 200:
        resp.close()
        conn.close()
        raise StreamNotReady("GET /video on port %d returned %d" % (port, resp.status))
    return resp


def connect_stream(port, timeout=180.0, *, clock=time.monotonic, sleep=time.sleep,
                   interval=CONNECT_RETRY_S, attempt=_open_stream):
    """An open ``/video`` response: the read half of the MJPEG connection.

    The first answer is usually not a stream. A window starts the demo and
    connects to it at once, while that demo is still importing numpy, so a
    refused connection, a reset, a ``RemoteDisconnected`` or a non-200 is the
    ordinary state of the first seconds of *every* window. On 10-28 the box's
    second run met one of them once and lost the whole run before window 1
    measured a frame. Attempts therefore repeat until ``timeout`` runs out --
    a demo that never serves must still end the run, with the last error raised.
    """
    deadline = clock() + timeout
    while True:
        left = deadline - clock()
        try:
            return attempt(port, max(1.0, min(15.0, left)))
        except RETRYABLE:
            if left <= 0:
                raise
            sleep(min(interval, left))


# --------------------------------------------------------------------------
# The running demo: what it is, and how to put it back
# --------------------------------------------------------------------------

def read_argv(pid, root=Path("/proc")):
    """The exact argv of ``pid``, or ``None`` when it is gone."""
    try:
        raw = (root / str(pid) / "cmdline").read_bytes()
    except OSError:
        return None
    argv = [a.decode("utf-8", "surrogateescape") for a in raw.split(b"\0") if a]
    return argv or None


def read_environ(pid, root=Path("/proc")):
    """The environment of ``pid`` as a dict; unreadable entries are dropped."""
    try:
        raw = (root / str(pid) / "environ").read_bytes()
    except OSError:
        return {}
    env = {}
    for item in raw.split(b"\0"):
        if not item:
            continue
        key, sep, value = item.decode("utf-8", "surrogateescape").partition("=")
        if sep:
            env[key] = value
    return env


def find_demo(root=Path("/proc"), script=DEMO_SCRIPT, exclude=(os.getpid(),)):
    """``pid`` of the process running the demo, or ``None``.

    Scanned directly rather than through ``pgrep -f``: the pattern would also
    match the shell that launched this harness, and the wrong match here means
    stopping the wrong process.
    """
    for entry in sorted(root.iterdir()):
        if not entry.name.isdigit():
            continue
        pid = int(entry.name)
        if pid in exclude:
            continue
        argv = read_argv(pid, root)
        if not argv:
            continue
        # The interpreter has to be python: a shell wrapper whose text mentions
        # the script is not the demo.
        if not os.path.basename(argv[0]).startswith("python"):
            continue
        if any(a == script or a.endswith("/" + script) for a in argv[1:]):
            return pid
    return None


def minio_env(path=MINIO_ENV_FILE):
    """MINIO_* from the Jetson's camera env file (``KEY=VALUE`` lines)."""
    out = {}
    p = Path(path).expanduser()
    if not p.exists():
        return out
    for line in p.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        out[key.strip()] = value.strip().strip('"').strip("'")
    return out


class CaptureRefused(Exception):
    """The demo cannot be used as the capture-off arm."""


class Demo:
    """What the running demo was, so it can be started again exactly."""

    def __init__(self, pid, argv, env, cwd):
        self.pid = pid
        self.argv = list(argv)
        self.env = dict(env)
        self.cwd = cwd

    @classmethod
    def discover(cls, pid=None, root=Path("/proc")):
        pid = pid or find_demo(root)
        if pid is None:
            return None
        argv = read_argv(pid, root)
        if not argv:
            return None
        try:
            cwd = str((root / str(pid) / "cwd").resolve())
        except OSError:
            cwd = None
        return cls(pid, argv, read_environ(pid, root), cwd)

    def describe(self):
        return "pid %s: %s (cwd %s)" % (self.pid, " ".join(self.argv), self.cwd)

    def with_capture(self, capture_dir, extra_env=None):
        """The same command plus capture, and the environment it needs."""
        if CAPTURE_FLAG in self.argv:
            raise CaptureRefused(CAPTURE_ALREADY_ON)
        argv = list(self.argv) + [CAPTURE_FLAG, str(capture_dir)]
        env = dict(self.env)
        env.update(minio_env())
        env.update(extra_env or {})
        return argv, env


def stop_pid(pid, timeout=40.0, root=Path("/proc")):
    """Stop a process this harness did not start. True when it exited on SIGINT."""
    try:
        os.kill(pid, signal.SIGINT)
    except ProcessLookupError:
        return True
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not (root / str(pid)).exists() or _is_zombie(pid, root):
            return True
        time.sleep(0.2)
    try:
        os.kill(pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    return False


def _is_zombie(pid, root=Path("/proc")):
    try:
        stat = (root / str(pid) / "stat").read_text()
    except OSError:
        return True
    # The state is the field after the parenthesised command name.
    return stat.rpartition(")")[2].split()[:1] == ["Z"]


def stop(proc, timeout=40.0):
    """SIGINT, because atexit is what flushes the writer and stops the uploader.

    Returns True when the process went down on the signal. A process that had to
    be killed lost whatever was still queued, so the caller reports it.
    """
    if proc is None or proc.poll() is not None:
        return True
    proc.send_signal(signal.SIGINT)
    try:
        proc.wait(timeout=timeout)
        return True
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=10.0)
        return False


def mem_available(path=MEMINFO):
    """MemAvailable in bytes, or None when it cannot be read."""
    try:
        for line in Path(path).read_text().splitlines():
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) * 1024
    except (OSError, ValueError, IndexError):
        return None
    return None


def wait_for_memory(before, *, path=MEMINFO, timeout=None, poll=None,
                    clock=time.monotonic, sleep=time.sleep):
    """Wait until MemAvailable is back to what it was while the demo still ran.

    ``before`` is read before the stop, so returning to it means the memory that
    process held has been handed back. Bounded: a box that never recovers is
    reported, not waited on forever. Returns ``(released, last reading)``.
    """
    timeout = MEM_TIMEOUT if timeout is None else timeout
    poll = MEM_POLL if poll is None else poll
    if before is None:
        return True, None
    floor = before - MEM_SLACK_BYTES
    deadline = clock() + timeout
    seen = mem_available(path)
    while seen is not None and seen < floor and clock() < deadline:
        sleep(poll)
        seen = mem_available(path)
    if seen is None:
        return True, None               # nothing to judge by
    return seen >= floor, seen


class StopResult:
    """What a stop did: did the process go down, and did its memory come back.

    Truthy only when the process exited on the signal. ``released`` is separate
    because a clean stop can still leave the box unrecovered for a while.
    """

    __slots__ = ("clean", "released", "before", "after")

    def __init__(self, clean, released, before=None, after=None):
        self.clean = clean
        self.released = released
        self.before = before
        self.after = after

    def __bool__(self):
        return bool(self.clean)

    def describe(self):
        if self.clean and self.released:
            return "stopped cleanly"
        if not self.clean:
            return "did not stop on SIGINT (killed)"
        return ("stopped, but memory had not come back (%s of %s bytes free)"
                % (self.after, self.before))


class Box:
    """The machine: one demo process at a time, and the stream it serves.

    A dry run uses the same object with ``dry=True``: every method prints what it
    would have run and returns, so the plan is exercised without a camera, a GPU
    or a port.
    """

    def __init__(self, dry=False, popen=None, connect=connect_stream, echo=print):
        self.dry = dry
        self._popen = popen
        self._connect = connect
        self.echo = echo

    def start(self, argv, env, cwd, log_path, env_delta=None):
        if self.dry:
            self.echo(_describe_command(argv, env_delta or {}, cwd))
            return None
        log = open(log_path, "ab")
        try:
            return (self._popen or subprocess.Popen)(
                argv, env=env, cwd=cwd, stdin=subprocess.DEVNULL,
                stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        finally:
            log.close()

    def read_stream(self, port, timeout):
        if self.dry:
            self.echo("[dry-run] read http://127.0.0.1:%d/video (warm-up, then window)"
                      % port)
            return None
        return self._connect(port, timeout)

    def stop(self, proc, timeout=40.0):
        if self.dry:
            self.echo("[dry-run] stop: SIGINT, then wait for its memory")
            return StopResult(True, True)
        return self._stop_then_wait(lambda: stop(proc, timeout))

    def stop_pid(self, pid, timeout=40.0):
        """Stop a process this harness did not start, and wait for its memory."""
        if self.dry:
            self.echo("[dry-run] stop pid %s: SIGINT, then wait for its memory" % pid)
            return StopResult(True, True)
        return self._stop_then_wait(lambda: stop_pid(pid, timeout=timeout))

    def _stop_then_wait(self, stopper):
        before = mem_available()
        clean = stopper()
        released, after = wait_for_memory(before)
        if not released:
            self.echo("memory had not come back after the stop: %s of %s bytes free"
                      % (after, before))
        return StopResult(clean, released, before, after)

    def close(self, resp):
        if resp is None:
            return
        try:
            resp.close()
        except Exception:                    # noqa: BLE001 -- closing must not raise
            pass


def feed_serves(box, port, *, timeout=None, poll=2.0, clock=time.monotonic, sleep=time.sleep):
    """True once ``/video`` has delivered part of a frame, within a bounded wait.

    Connecting is not serving: the demo binds the port late in start-up, and a
    process that has died still looks running until it is reaped.

    Never raises. A port held by something that is not an HTTP server answers with
    ``http.client.BadStatusLine``, which is an ``HTTPException`` and not an
    ``OSError``; letting that through would abort ``restore_demo``'s "never raises"
    promise and with it main's ``finally``, replacing the report's "feed is DOWN"
    with a traceback.
    """
    timeout = FEED_TIMEOUT if timeout is None else timeout
    deadline = clock() + timeout
    while True:
        left = deadline - clock()
        resp, error = None, None
        try:
            resp = box.read_stream(port, timeout=max(1.0, min(15.0, left)))
        except Exception as err:        # noqa: BLE001 -- refused, reset, not-HTTP
            error = err
        if resp is not None:
            read = stream_reader(resp)
            seen = b""
            try:
                while clock() < deadline:
                    chunk = read()
                    if not chunk:
                        break
                    seen += chunk
                    if len(seen) > len(BOUNDARY) + 128:
                        return True
            except Exception as err:    # noqa: BLE001 -- a broken stream is no
                error = err            # frame, not an exception to hand upwards
            finally:
                box.close(resp)
        if clock() >= deadline:
            return False
        sleep(poll)


def restore_demo(box, demo, logs, port=PORT, *, attempts=None, settle=None,
                 timeout=None, echo=None, clock=time.monotonic, sleep=time.sleep):
    """Start the demo again from its recorded argv, environment and cwd, and prove it serves.

    Never raises: it runs in a ``finally``, and the report has to survive whatever
    the box does. Returns ``(ok, text)``; the text is what the report records.
    """
    attempts = RESTORE_ATTEMPTS if attempts is None else attempts
    settle = RESTORE_SETTLE if settle is None else settle
    timeout = FEED_TIMEOUT if timeout is None else timeout
    notes = []
    for attempt in range(1, attempts + 1):
        name = "restored.log" if attempt == 1 else "restored-%d.log" % attempt
        try:
            proc = box.start(demo.argv, demo.env, demo.cwd, Path(logs) / name)
        except OSError as err:
            notes.append("start %d failed: %s" % (attempt, err))
        else:
            if feed_serves(box, port, timeout=timeout, clock=clock, sleep=sleep):
                return True, ("restored and serving: %s (log %s)"
                              % (" ".join(demo.argv), name))
            # Not serving: take it down before the next attempt, which also waits
            # for the memory it held -- a demo left in warm-up would block the retry.
            box.stop(proc)
            notes.append("start %d did not serve a frame within %.0f s (log %s)"
                         % (attempt, timeout, name))
        if attempt < attempts and settle:
            if echo:
                echo("restore attempt %d did not serve; retrying in %.0f s"
                     % (attempt, settle))
            sleep(settle)
    return False, ("FAILED restore: %s. The feed is DOWN -- start the demo by hand "
                   "(%s, log in %s)." % ("; ".join(notes), " ".join(demo.argv), logs))


def _describe_command(argv, env_delta, cwd):
    env = " ".join("%s=%s" % (k, v) for k, v in sorted(env_delta.items()))
    where = (" (cd %s)" % cwd) if cwd else ""
    return "[dry-run] start: %s%s%s" % (env + " " if env else "",
                                        " ".join(argv), where)


def run_window(box, argv, env, cwd, log_path, *, port=PORT, warmup=30.0, window=60.0,
               env_delta=None):
    """Start the demo, measure one window off the wire, stop it, read its stats."""
    proc = box.start(argv, env, cwd, log_path, env_delta=env_delta)
    if box.dry:
        box.read_stream(port, timeout=0)
        box.stop(proc)
        return None
    try:
        resp = box.read_stream(port, timeout=warmup + window + 120.0)
        try:
            stats = measure(stream_reader(resp), warmup=warmup, window=window)
        finally:
            box.close(resp)
    finally:
        stopped = box.stop(proc)
    stats["stopped_cleanly"] = stopped.clean
    stats["released"] = stopped.released
    stats["counters"] = last_capture_stats(log_path)
    return stats


def last_capture_stats(path):
    """The last ``capture stats:`` JSON the demo printed, or ``None``."""
    try:
        text = Path(path).read_text(errors="replace")
    except OSError:
        return None
    found = None
    for line in text.splitlines():
        if line.startswith(STATS_PREFIX):
            try:
                found = json.loads(line[len(STATS_PREFIX):])
            except ValueError:
                continue
    return found


# --------------------------------------------------------------------------
# The three checks
# --------------------------------------------------------------------------

def summarize(values):
    values = list(values)
    return {"n": len(values), "median": statistics.median(values),
            "min": min(values), "max": max(values),
            "spread": max(values) - min(values)}


def slowdown_verdict(off_fps, on_fps):
    """7.2: the capture-on median is at least 95% of the capture-off median."""
    off = summarize(off_fps)
    on = summarize(on_fps)
    need = SLOWDOWN_RATIO * off["median"]
    passed = on["median"] + EPS >= need
    text = ("%s: capture-on median %.2f fps against capture-off median %.2f fps; "
            "the bar is %.2f (95%%)\n"
            "  off windows: %s\n  on windows:  %s"
            % ("PASS" if passed else "FAIL", on["median"], off["median"], need,
               _fmt_runs(off), _fmt_runs(on)))
    return passed, text


def _fmt_runs(summary):
    return ("%d windows, median %.2f fps, %.2f-%.2f, spread %.2f"
            % (summary["n"], summary["median"], summary["min"], summary["max"],
               summary["spread"]))


def feed_verdict(baseline, off, port=PORT):
    """7.1: the capture-off feed is the one the box already served."""
    if baseline is None:
        return None, ("SKIPPED: no demo was running when the harness started, so "
                      "there is no pre-change feed to compare against")
    problems = []
    if baseline["dims"] != off["dims"]:
        problems.append("resolution %s -> %s" % (baseline["dims"], off["dims"]))
    ratio = (off["median_bytes"] / float(baseline["median_bytes"])
             if baseline["median_bytes"] else 0.0)
    if not (1.0 - SIZE_TOLERANCE <= ratio <= 1.0 + SIZE_TOLERANCE):
        problems.append("median frame %d -> %d bytes (%.2fx, tolerance %.0f%%)"
                        % (baseline["median_bytes"], off["median_bytes"], ratio,
                           100 * SIZE_TOLERANCE))
    if problems:
        return False, "FAIL: " + "; ".join(problems)
    dims = off["dims"][0] if off["dims"] else (-1, -1)
    return True, ("PASS: feed unchanged -- %dx%d at port %d, median frame %d bytes "
                  "(baseline %d, %.2fx)"
                  % (dims[0], dims[1], port, off["median_bytes"],
                     baseline["median_bytes"], ratio))


def dir_usage(path):
    """Every file the writer owns under ``path``: count, bytes, kinds."""
    root = Path(path)
    files = [p for p in root.rglob("*") if p.is_file()] if root.exists() else []
    return {"files": len(files),
            "bytes": sum(p.stat().st_size for p in files),
            "frames": sum(1 for p in files if p.suffix == ".jpg"),
            "markers": sum(1 for p in files if p.name.endswith(".uploaded"))}


def cap_verdict(cap, usage, counters):
    """7.4: nothing past the cap, and the refusal was counted when it happened."""
    refused = 0 if not counters else int(counters.get("disk_cap", 0))
    lines = ["%d bytes on disk against a cap of %d (%d frames, %d upload markers)"
             % (usage["bytes"], cap, usage["frames"], usage["markers"]),
             "refused by the cap: %d" % refused]
    if usage["bytes"] > cap:
        return False, "FAIL: the directory is over the cap. " + "; ".join(lines)
    if refused == 0:
        return None, ("INCONCLUSIVE: the cap was never reached in the window, so "
                      "nothing was refused. " + "; ".join(lines))
    return True, "PASS: " + "; ".join(lines)


# --------------------------------------------------------------------------
# The run
# --------------------------------------------------------------------------

ARMS = ("off", "on")


def window_plan(pairs=3):
    """7.2's windows: off, on, off, on, off, on."""
    return [ARMS[i % 2] for i in range(2 * pairs)]


def _stamp():
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _log_lines(report, line=""):
    report.write(line + "\n")
    report.flush()
    if report is not sys.stdout:
        print(line, flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true",
                    help="print every command and run nothing")
    ap.add_argument("--out", type=Path, default=None,
                    help="report file (default: ./camera-round0-windows-<utc>.txt)")
    ap.add_argument("--pid", type=int, default=None,
                    help="the running demo (default: found by scanning /proc)")
    ap.add_argument("--argv", default=None,
                    help="the demo command, when none is running (split on spaces)")
    ap.add_argument("--port", type=int, default=PORT)
    ap.add_argument("--pairs", type=int, default=3, help="7.2 off/on pairs")
    ap.add_argument("--warmup", type=float, default=30.0)
    ap.add_argument("--window", type=float, default=60.0)
    ap.add_argument("--baseline-window", type=float, default=20.0,
                    help="how long 7.1 samples the feed that is already running")
    ap.add_argument("--capture-dir", type=Path, default=None,
                    help="where the on arms write (default: ~/round0-verify/<utc>)")
    ap.add_argument("--cap-bytes", type=int, default=128 * 1024,
                    help="CAPTURE_DISK_CAP_BYTES for 7.4")
    ap.add_argument("--cap-window", type=float, default=300.0,
                    help="how long 7.4 observes the cap")
    ap.add_argument("--unreachable", default=UNREACHABLE,
                    help="MINIO_ENDPOINT for 7.4 (a closed port)")
    ap.add_argument("--only", default="7.1,7.2,7.4",
                    help="comma-separated checks to run (7.1 needs 7.2's first "
                         "off window)")
    args = ap.parse_args(argv)

    only = {s.strip() for s in args.only.split(",") if s.strip()}
    stamp = _stamp()
    base = Path(args.capture_dir).expanduser() if args.capture_dir else \
        Path("~/round0-verify").expanduser() / stamp
    out_path = args.out or Path("camera-round0-windows-%s.txt" % stamp)
    logs = base / "logs"

    box = Box(dry=args.dry_run)
    demo = Demo.discover(args.pid) if args.argv is None else \
        Demo(None, args.argv.split(), dict(os.environ), None)
    report = sys.stdout if args.dry_run else None

    if demo is None:
        print("no demo running: pass --argv to name the command", file=sys.stderr)
        return 2
    needed = {"7.2", "7.4"} & only
    # Refused here, not when the first on-arm is built: the alternative is a
    # traceback after 7.1's baseline and the first off window, with the demo already
    # restarted. --only 7.1 needs no on-arm, so it is unaffected.
    if needed and CAPTURE_FLAG in demo.argv:
        print(CAPTURE_ALREADY_ON, file=sys.stderr)
        return 2
    if not args.dry_run:
        if needed and not capture_vars_present(demo.env):
            print("capture needs MINIO_ENDPOINT and MINIO_USER (or "
                  "MINIO_PASSWORD_FILE): put them in %s or in the running demo's "
                  "environment" % MINIO_ENV_FILE, file=sys.stderr)
            return 2
        base.mkdir(parents=True, exist_ok=True)
        logs.mkdir(parents=True, exist_ok=True)
        report = open(out_path, "w")

    status = 0
    restore_failed = False
    # Whether this run took the demo it found down. Read in the finally: a restore
    # is only meaningful for a demo this harness actually stopped.
    stopped_running = False
    try:
        _log_lines(report, "camera round 0 -- window verification")
        _log_lines(report, "started %s (UTC)" % stamp)
        _log_lines(report, "demo: " + demo.describe())
        _log_lines(report, "capture dir: %s" % base)
        _log_lines(report)

        baseline = None
        if "7.1" in only and demo is not None and demo.pid is not None and not args.dry_run:
            resp = box.read_stream(args.port, timeout=args.baseline_window + 120.0)
            try:
                baseline = measure(stream_reader(resp), warmup=0.0,
                                   window=args.baseline_window)
            finally:
                box.close(resp)
            baseline["port"] = args.port
            _log_lines(report, "baseline: %d frames in %.0f s, %s, median %d bytes"
                       % (baseline["parts"], args.baseline_window,
                          baseline["dims"], baseline["median_bytes"]))
            _log_lines(report)

        off_stats = None
        # Any check that starts a demo needs the box to itself: the running demo owns
        # the port and the camera, so a second one cannot serve, and a measurement
        # taken then belongs to the old process while looking like the new one's.
        if ({"7.2", "7.4"} & only) and demo.pid is not None and not args.dry_run:
            stopped = box.stop_pid(demo.pid)
            stopped_running = True
            _log_lines(report, "stopped the running demo: " + stopped.describe())
            # Its memory has to be back before the first window starts, or that
            # window's demo dies at warm-up asking for its TensorRT workspace.
            if not stopped:
                print("the running demo (%d) would not stop" % demo.pid,
                      file=sys.stderr)
                status = 3
            elif not stopped.released:
                _log_lines(report, "  starting the windows anyway: a start may "
                                   "fail at warm-up and be retried")
            _log_lines(report)
        if "7.2" in only:
            _log_lines(report, "## 7.2 no measurable slowdown (off, on, off, on, off, on)")
            fps = {"off": [], "on": []}
            if status == 0:
                for index, arm in enumerate(window_plan(args.pairs), start=1):
                    capture_dir = base / ("w%02d-%s" % (index, arm))
                    if arm == "on":
                        cmd, env = demo.with_capture(capture_dir)
                        delta = dict(minio_env())
                    else:
                        cmd, env, delta = list(demo.argv), dict(demo.env), {}
                    stats = run_window(box, cmd, env, demo.cwd,
                                       logs / ("w%02d-%s.log" % (index, arm)),
                                       port=args.port, warmup=args.warmup,
                                       window=args.window, env_delta=delta)
                    if box.dry:
                        continue
                    fps[arm].append(stats["fps"])
                    if arm == "off" and off_stats is None:
                        off_stats = stats
                    dims = stats["dims"][0] if stats["dims"] else (-1, -1)
                    _log_lines(report, "  w%d %-3s %6.2f fps  %dx%d  %5d B median%s%s"
                               % (index, arm, stats["fps"], dims[0], dims[1],
                                  stats["median_bytes"],
                                  "" if stats["stopped_cleanly"]
                                  else "  (killed: queued frames lost)",
                                  "" if stats["released"]
                                  else "  (memory not yet back)"))
                if not box.dry:
                    _passed, text = slowdown_verdict(fps["off"], fps["on"])
                    _log_lines(report, "  " + text.replace("\n", "\n  "))
            _log_lines(report)

        if "7.1" in only:
            _log_lines(report, "## 7.1 feed unchanged with capture off")
            if box.dry or off_stats is None:
                _log_lines(report, "  (7.1 is judged from 7.2's first capture-off "
                                   "window; nothing ran)")
            else:
                _ok, text = feed_verdict(baseline, off_stats, port=args.port)
                _log_lines(report, "  " + text)
            _log_lines(report)

        if "7.4" in only:
            _log_lines(report, "## 7.4 disk cap holds with the bucket unreachable")
            capture_dir = base / "cap"
            cmd, env = demo.with_capture(
                capture_dir, {"MINIO_ENDPOINT": args.unreachable,
                              "CAPTURE_DISK_CAP_BYTES": str(args.cap_bytes)})
            env["MINIO_USER"] = env.get("MINIO_USER") or "unreachable"
            env["MINIO_PASSWORD"] = env.get("MINIO_PASSWORD") or "unreachable"
            delta = {"MINIO_ENDPOINT": args.unreachable,
                     "CAPTURE_DISK_CAP_BYTES": str(args.cap_bytes)}
            stats = run_window(box, cmd, env, demo.cwd, logs / "cap.log",
                               port=args.port, warmup=args.warmup,
                               window=args.cap_window, env_delta=delta)
            if not box.dry:
                counters = stats["counters"] or {}
                ok, text = cap_verdict(args.cap_bytes, dir_usage(capture_dir),
                                       counters)
                _log_lines(report, "  " + text)
                _log_lines(report, "  counters: " + json.dumps(counters, sort_keys=True))
            _log_lines(report)
    finally:
        # The demo that was running comes back whatever happened above, and its
        # serving is checked: a start is not a restore. This runs here, not at the
        # end of the run, so a traceback in the middle cannot leave the feed down.
        if demo is not None and demo.pid is not None and not args.dry_run:
            if not stopped_running:
                # Nothing here stopped it (--only 7.1, or a run that failed before
                # the stop), so it is still serving. A start would prove nothing it
                # did not already prove, and "restored and serving" would be the
                # port talking, not the process this run started.
                _log_lines(report, "the demo that was running (%d) was never stopped "
                                   "by this run; its feed was left as it is"
                           % demo.pid)
            else:
                ok, text = restore_demo(box, demo, logs, args.port,
                                        echo=lambda m: print(m, file=sys.stderr))
                _log_lines(report, text)
                if not ok:
                    restore_failed = True
                    print(text, file=sys.stderr)
        if not args.dry_run:
            report.close()
            print("report: %s" % out_path)
    return 4 if restore_failed else status


def capture_vars_present(demo_env, path=MINIO_ENV_FILE):
    """Does capture have what ``MinioConfig.from_env`` insists on?

    The on arm is a fresh process, so a variable missing from both the running
    demo and the camera env file means the window cannot start at all.
    """
    env = dict(demo_env)
    env.update(minio_env(path))
    if not env.get("MINIO_ENDPOINT"):
        return False
    return bool(env.get("MINIO_USER") or env.get("MINIO_PASSWORD_FILE"))


if __name__ == "__main__":
    sys.exit(main())
