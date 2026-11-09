"""Raw-frame writer for camera round 0: JPEG + sidecar + index line per frame.

The capture path must not slow the demo (spec: saving never blocks the inference
loop), so ``offer`` copies the frame, names it, and puts it on a bounded queue; a
writer thread does the JPEG encode, the sidecar write and the index append. When
the queue is full the frame is dropped and counted -- a slow disk degrades the
sample, never the live loop.

Layout under ``<out_dir>``:

    index.jsonl                 one record per saved frame (name, kind, score, ...)
    <America/Chicago day>/<name>.jpg      the raw camera frame
    <America/Chicago day>/<name>.json     the same record as the index line
    <America/Chicago day>/<name>.uploaded marker written only after a confirmed upload

The directory is byte-capped. On reaching the cap the writer prunes frames the
uploader has already confirmed (oldest first); if that frees nothing it skips the
capture and counts it, so the cap holds while the bucket is unreachable.
"""

from __future__ import annotations

import json
import os
import queue
import threading
from datetime import datetime, timezone
from pathlib import Path

from jetson.uncertain_frames import CONTROL, LOCAL_TZ, Counters, local_day

QUEUE_MAX = 2
DISK_CAP_BYTES = 512 * 1024 * 1024
INDEX_NAME = "index.jsonl"
UPLOADED_SUFFIX = ".uploaded"
JPEG_QUALITY = 95


def _jpeg_bytes(frame):
    """JPEG at quality 95. Imported here so the writer loads without OpenCV."""
    import cv2  # pylint: disable=import-outside-toplevel

    ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
    if not ok:
        raise RuntimeError("cv2.imencode failed")
    return buf.tobytes()


def frame_name(ts, kind="uncertain"):
    """``[control-]<UTC epoch ms>``: the time is in the name and sorts by it.

    The name is fixed when the frame is offered, not when it is written, so the
    sidecar's timestamps and the file name agree to the millisecond. Controls are
    distinguishable from the name alone (spec), so a lost index still sorts.
    """
    ms = int(round(float(ts) * 1000.0))
    return f"{'control-' if kind == CONTROL else ''}{ms}"


def record(name, decision, score, p_fg_max, band, fps=None, model=None,
           input_size=None, tz=LOCAL_TZ, context=None):
    """The metadata for one frame: the index line and the sidecar are the same record."""
    ts = decision.ts
    doc = {
        "file": f"{name}.jpg",
        "kind": decision.kind,
        "ts_utc": datetime.fromtimestamp(ts, timezone.utc).isoformat(),
        "ts_local": datetime.fromtimestamp(ts, tz).isoformat(),
        "tz": str(tz),
        "day": local_day(ts, tz),
        "score": float(score),
        "max_p_fg": float(p_fg_max),
        "band_low": float(band[0]),
        "band_high": float(band[1]),
        "fps": fps,
        "model": model,
        "input_size": input_size,
    }
    if context:
        doc.update(context)
    return doc


class FrameWriter:
    """Bounded queue + writer thread. ``offer`` is safe to call from the loop."""

    def __init__(self, out_dir, *, context=None, queue_max=QUEUE_MAX,
                 disk_cap=DISK_CAP_BYTES, counts=None, band=(0.35, 0.65),
                 tz=LOCAL_TZ, encode=None):
        self.out_dir = Path(out_dir)
        self.context = dict(context or {})
        self.band = tuple(band)
        self.tz = tz
        self.counts = counts if counts is not None else Counters()
        self.disk_cap = int(disk_cap)
        self._encode = encode if encode is not None else _jpeg_bytes
        self._queue = queue.Queue(maxsize=int(queue_max))
        self._lock = threading.Lock()
        self._bytes_on_disk = self._measure()
        self._thread = None
        self._closed = False
        self._stop = object()
        self._last_error = None

    # -- producer side -----------------------------------------------------

    def offer(self, decision, frame, score, p_fg_max, fps=None):
        """Copy the frame, name it, enqueue it. Returns the name, or None if dropped."""
        name = frame_name(decision.ts, decision.kind)
        job = (name, decision, frame.copy(), float(score), float(p_fg_max),
               None if fps is None else float(fps))
        try:
            self._queue.put_nowait(job)
        except queue.Full:
            self.counts.bump("capture_queue_full")
            return None
        return name

    def start(self):
        if self._thread is None or not self._thread.is_alive():
            self._thread = threading.Thread(target=self._run, name="frame-writer", daemon=True)
            self._thread.start()
        return self

    def flush(self, timeout=20.0):
        """Wait for the queue to drain, then stop the thread. Idempotent."""
        if self._closed or self._thread is None:
            return self
        self._closed = True
        # The sentinel goes in last, from the caller: the writer takes jobs in
        # order, so everything offered before this call is written before it stops.
        try:
            self._queue.put(self._stop, timeout=timeout)
        except queue.Full:
            pass
        self._thread.join(timeout)
        return self

    # -- consumer side -----------------------------------------------------

    def _run(self):
        while True:
            job = self._queue.get()
            try:
                if job is self._stop:
                    return
                self._write(*job)
            except Exception as exc:  # never let a write bug kill the demo
                self._fail(repr(exc))
            finally:
                self._queue.task_done()

    def _write(self, name, decision, frame, score, p_fg_max, fps):
        day_dir = self.out_dir / local_day(decision.ts, self.tz)
        frame_path = day_dir / f"{name}.jpg"
        side_path = day_dir / f"{name}.json"
        doc = record(name, decision, score, p_fg_max, self.band, fps=fps,
                     tz=self.tz, context=self.context)
        payload = json.dumps(doc).encode("utf-8")

        # The frame's JPEG, its sidecar and its index line (which keeps its
        # newline) all cost disk.
        record_bytes = len(payload) + 1
        reserve = _FRAME_ENVELOPE + len(payload) + record_bytes
        if not self._reserve(reserve):
            self.counts.bump("disk_cap")
            return
        try:
            payload_jpg = self._encode(frame)
        except Exception as exc:
            self._refund(reserve)
            self._fail(repr(exc))
            return
        try:
            day_dir.mkdir(parents=True, exist_ok=True)
            written = frame_path.write_bytes(payload_jpg)
            side_path.write_bytes(payload)
            with open(self.out_dir / INDEX_NAME, "ab") as idx:
                idx.write(payload + b"\n")
        except OSError as exc:
            self._refund(reserve)
            self._fail(repr(exc))
            return
        # Reconcile the reservation with the true frame size plus both records.
        with self._lock:
            self._bytes_on_disk += written + len(payload) + record_bytes - reserve
        self.counts.bump("written")

    def _reserve(self, reserve):
        """Take space for a frame, pruning confirmed uploads if the cap is hit."""
        # A confirmed upload leaves a marker but no frame: the uploader deletes the
        # bytes itself, so prune_uploaded cannot charge them back and the counter
        # overstates the directory. A marker is the only evidence that can have
        # happened, so note it before pruning consumes it -- and re-measure rather
        # than refuse, or the cap becomes a bound on bytes written since startup.
        markers_possible = self._has_markers()
        with self._lock:
            if self._bytes_on_disk + reserve <= self.disk_cap:
                self._bytes_on_disk += reserve
                return True
        freed = self.prune_uploaded()
        if freed:
            self.counts.bump("disk_pruned", freed)
        with self._lock:
            if self._bytes_on_disk + reserve <= self.disk_cap:
                self._bytes_on_disk += reserve
                return True
        if markers_possible:
            measured = self._measure()
            with self._lock:
                self._bytes_on_disk = measured
                if self._bytes_on_disk + reserve <= self.disk_cap:
                    self._bytes_on_disk += reserve
                    return True
        return False

    def _has_markers(self):
        """Has any upload been confirmed? Only then can the counter be overstated."""
        return any(self.out_dir.glob(f"*/*{UPLOADED_SUFFIX}"))

    def _refund(self, reserve):
        with self._lock:
            self._bytes_on_disk -= reserve

    def prune_uploaded(self):
        """Delete frames already confirmed uploaded, oldest first. Returns count.

        Only ``.uploaded`` files are touched: a frame whose upload has not been
        confirmed must stay locally, so the cap can never destroy the only copy.
        """
        freed = 0
        if not self.out_dir.exists():
            return 0
        markers = []
        for marker in self.out_dir.glob(f"*/*{UPLOADED_SUFFIX}"):
            try:
                markers.append((marker.stat().st_mtime, marker))
            except OSError:
                continue
        for _mtime, marker in sorted(markers):
            frame = Path(str(marker)[: -len(UPLOADED_SUFFIX)])
            lost = 0
            for path in (marker, frame, frame.with_suffix(".json")):
                try:
                    size = path.stat().st_size
                    path.unlink()
                    lost += size
                except OSError:
                    continue
            if lost:
                freed += 1
                with self._lock:
                    self._bytes_on_disk -= lost
        return freed

    def _fail(self, reason):
        self._last_error = reason
        self.counts.bump("write_error")
        print(f"[frame-writer] write failed: {reason}", flush=True)

    def _measure(self):
        """Bytes already in ``out_dir`` (a resumed session must not overfill)."""
        total = 0
        if not self.out_dir.exists():
            return 0
        for root, _dirs, files in os.walk(self.out_dir):
            for name in files:
                try:
                    total += os.path.getsize(os.path.join(root, name))
                except OSError:
                    pass
        return total

    def disk_bytes(self):
        with self._lock:
            return self._bytes_on_disk

    def stats(self):
        out = self.counts.as_dict()
        out["disk_bytes"] = self.disk_bytes()
        out["write_error_last"] = self._last_error
        return out


# Conservative per-frame charge used to hold the disk cap; a 640x480 JPEG of a
# busy street scene is well under this.
_FRAME_ENVELOPE = 64 * 1024
