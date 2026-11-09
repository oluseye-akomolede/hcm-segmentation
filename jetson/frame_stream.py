"""Hand-off of encoded frames from the capture loop to HTTP viewers.

The demo's MJPEG response generator used to re-yield a module global in a busy
loop. That loop had no wait in it, so a viewer with no new frame to read still
iterated as fast as the CPU allowed: a 4-second ``curl`` pulled about 1.6 GB of
the *same* JPEG repeated. The stream looked fine and cost a core per viewer.

This module holds the rule the generator needs: every viewer gets each produced
frame at most once, and a viewer with nothing new to read blocks instead of
spinning. It deliberately has no ``cv2``, no torch, and no work at import, so
the coupling between "what the stream guarantees" and "a camera and a model
must exist" is broken and the guarantee can be tested on its own.

Only the newest frame is kept. A viewer slower than the capture loop therefore
skips frames rather than accumulating a backlog -- for a live camera feed the
current frame is the useful one, and a backlog is just latency.
"""

import threading


class FrameStream:
    """A one-slot latest-frame buffer that each viewer drains independently."""

    def __init__(self):
        self._condition = threading.Condition()
        self._seq = 0
        self._frame = None

    @property
    def sequence(self) -> int:
        """How many frames have been published. Starts at 0 (none yet)."""
        with self._condition:
            return self._seq

    def publish(self, frame: bytes) -> int:
        """Make ``frame`` the newest frame and wake every waiting viewer.

        Returns the sequence number assigned to it.
        """
        with self._condition:
            self._seq += 1
            self._frame = frame
            self._condition.notify_all()
            return self._seq

    def wait_for_next(self, last_seen: int) -> tuple[int, bytes]:
        """Block until a frame newer than ``last_seen`` exists, then return it.

        Returns ``(sequence, frame)``. The wait is on the condition's own
        monitor, so an idle viewer costs no CPU. Two viewers call this with
        their own ``last_seen`` and do not interfere: each blocks only until
        *some* frame newer than its own was published. The frame returned is the
        newest at the moment the caller wakes, which is why a viewer that
        connects between two frames is served the next one produced rather than
        a stale copy.
        """
        with self._condition:
            while self._seq <= last_seen:
                self._condition.wait()
            return self._seq, self._frame

    def frames_since(self, last_seen: int):
        """Yield ``(sequence, frame)`` forever, one per newly produced frame.

        Never yields the same sequence twice, so this is the per-viewer
        guarantee the MJPEG generator needs.
        """
        seen = last_seen
        while True:
            seq, frame = self.wait_for_next(seen)
            seen = seq
            yield seq, frame


# The stream the demo's capture loop publishes to and its viewers read from.
STREAM = FrameStream()


def mjpeg_part(frame: bytes) -> bytes:
    """One ``multipart/x-mixed-replace`` part carrying a JPEG.

    The boundary bytes and the ``Content-Type`` header are part of the wire
    format the browser parses; the MJPEG boundary token must match the one the
    ``/video`` route passes to Flask, so neither is parameterised here.
    """
    return (b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")
