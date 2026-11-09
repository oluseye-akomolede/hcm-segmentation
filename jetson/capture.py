"""Camera-read retries and rate-limited warnings for the realtime demo.

The demo used to print "camera read failed" and break out of its loop on the
first bad read, then exit 0: a supervisor saw a clean exit and an unplugged
camera looked like a normal stop. A JPEG encode failure dropped the frame with
no word at all. Pure (no cv2, no torch) so the rules are testable off-device.
"""

import time

# A USB camera can return one or two bad reads on a bandwidth hiccup; five
# tries 0.2 s apart rides out about a second of that without hiding a camera
# that is really gone.
READ_RETRIES = 5
READ_RETRY_DELAY_S = 0.2


class CameraReadError(RuntimeError):
    """The camera returned no frame on every try."""


def read_with_retry(read, retries=READ_RETRIES, delay=READ_RETRY_DELAY_S, sleep=time.sleep):
    """Call ``read()`` (``cap.read``) until it yields a frame, up to ``retries`` times.

    Returns the frame. Raises :class:`CameraReadError` naming the try count when
    every read fails.
    """
    for attempt in range(retries):
        ok, frame = read()
        if ok and frame is not None:
            return frame
        if attempt + 1 < retries:
            sleep(delay)
    raise CameraReadError(f"camera read failed {retries} times in a row")


class RateLimitedWarning:
    """Print a warning at most once per ``interval`` seconds, with a count of the rest."""

    def __init__(self, message, interval=10.0, clock=time.monotonic, emit=print):
        self.message = message
        self.interval = interval
        self.clock = clock
        self.emit = emit
        self._last = None
        self._suppressed = 0

    def __call__(self):
        now = self.clock()
        if self._last is not None and now - self._last < self.interval:
            self._suppressed += 1
            return
        extra = f" ({self._suppressed} more since the last warning)" if self._suppressed else ""
        self.emit(f"WARNING: {self.message}{extra}")
        self._last = now
        self._suppressed = 0
