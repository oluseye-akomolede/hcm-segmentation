"""The one object the demo touches when ``--capture-uncertain DIR`` is given.

It bundles the three pieces the spec separates -- the selection policy, the
writer and the uploader -- behind ``offer``, and owns their lifecycle. The demo
calls it once per frame with the frame as read and the ``p_fg`` it already
computed; with no flag the session is never built, so the frame path is
unchanged (spec: capture is opt-in).
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

from jetson.frame_uploader import CaptureConfigError, FrameUploader, MinioConfig
from jetson.frame_writer import DISK_CAP_BYTES, FrameWriter
from jetson.uncertain_frames import (
    BAND_HIGH,
    BAND_LOW,
    Counters,
    CapturePolicy,
    max_p_fg,
    uncertain_fraction,
)


class CaptureSession:
    """Policy + writer + uploader, started together and stopped together."""

    def __init__(self, out_dir, *, policy=None, writer=None, uploader=None,
                 counts=None, clock=time.time):
        self.out_dir = Path(out_dir)
        self.counts = counts if counts is not None else Counters()
        self.clock = clock
        self.policy = policy if policy is not None else CapturePolicy(
            clock=clock, counts=self.counts)
        self.writer = writer if writer is not None else FrameWriter(
            self.out_dir, counts=self.counts, band=(BAND_LOW, BAND_HIGH))
        self.uploader = uploader

    @classmethod
    def from_env(cls, out_dir, env=None, *, context=None, clock=time.time, put=None,
                 encode=None, scan_period=None):
        """Build a session, refusing -- before the camera opens -- if MinIO is unset."""
        env = os.environ if env is None else env
        config = MinioConfig.from_env(env)   # raises CaptureConfigError, naming variables
        counts = Counters()
        session = cls(
            out_dir,
            policy=CapturePolicy(clock=clock, counts=counts),
            writer=FrameWriter(out_dir, counts=counts, band=(BAND_LOW, BAND_HIGH),
                               context=context, encode=encode,
                               disk_cap=_disk_cap_from_env(env)),
            uploader=FrameUploader(out_dir, config, counts=counts, put=put,
                                   **({} if scan_period is None
                                      else {"scan_period": scan_period})),
            counts=counts,
            clock=clock,
        )
        print(session.uploader.describe(), flush=True)
        return session

    def start(self):
        self.writer.start()
        if self.uploader is not None:
            self.uploader.start()
        return self

    def set_context(self, **fields):
        """Record fields the demo only knows later (the input size is resolved after
        the camera opens); they land in every subsequent frame's record."""
        self.writer.context.update(fields)
        return self

    def offer(self, frame, p_fg, fps=None):
        """Score one frame and keep it if the policy says so. Returns the name or None.

        ``frame`` must be the camera frame as read: the saved file is the raw
        image, before any overlay (spec).
        """
        score = uncertain_fraction(p_fg)
        decision = self.policy.offer(score, now=self.clock())
        if decision is None:
            return None
        name = self.writer.offer(decision, frame, score, max_p_fg(p_fg), fps=fps)
        if name is not None:
            self.counts.bump("offered")
        return name

    def stop(self, timeout=20.0):
        """Flush what is queued, then publish the counters this run collected.

        Safe when never started (the no-flag path). The counters are printed
        rather than only held in ``stats()``: the drops a round needs to explain
        -- a full disk, a closed bucket, a refused upload -- happen on a box
        nobody is watching, and a run that dropped frames should say so in its
        own log instead of requiring a debugger.
        """
        self.writer.flush(timeout)
        if self.uploader is not None:
            self.uploader.stop(timeout)
        print("capture stats: " + json.dumps(self.stats(), sort_keys=True), flush=True)
        return self

    def stats(self):
        out = dict(self.counts.as_dict())
        out.update(self.writer.stats())
        if self.uploader is not None:
            out.update(self.uploader.stats())
        return out


def _disk_cap_from_env(env):
    """``CAPTURE_DISK_CAP_BYTES``, or the writer's own default.

    The cap is configurable so 7.4 can be run at all: filling the real 512 MiB
    with in-band frames would take hours at the policy's 20 s spacing, while a
    cap of a few frames is reached in the same minutes a check can afford.
    """
    raw = (env.get("CAPTURE_DISK_CAP_BYTES") or "").strip()
    if not raw:
        return DISK_CAP_BYTES
    try:
        cap = int(raw)
    except ValueError:
        cap = 0
    if cap <= 0:
        raise CaptureConfigError(
            "CAPTURE_DISK_CAP_BYTES must be a positive number of bytes, not %r" % raw)
    return cap


def build_capture(capture_dir, env=None, *, context=None, clock=time.time, **kw):
    """``None`` when capture is off; a started session when ``capture_dir`` is given.

    Kept as a function so a test can assert the no-flag path constructs nothing.
    """
    if capture_dir is None:
        return None
    return CaptureSession.from_env(capture_dir, env, context=context, clock=clock,
                                   **kw).start()


__all__ = ["CaptureSession", "build_capture", "CaptureConfigError"]
