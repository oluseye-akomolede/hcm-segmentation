"""Stop requests for the realtime demo: remembered, then acted on once.

A demo started as a background job of a non-interactive shell inherits SIGINT set
to SIG_IGN, and CPython installs its own handler only when the inherited
disposition is *not* SIG_IGN. Such a process ignores Ctrl-C and a supervisor's
SIGINT outright: the 10-28 round-0 window run waited its whole 40 s on a demo that
would never look at the signal, killed it, and left the box without a feed.

Installing the handler explicitly makes the behaviour independent of how the demo
was started. SIGTERM is handled the same way, which also matters for
``jetson/promote_model.py``: its SIGTERM used to end the process before the
capture queue's ``atexit`` hook could flush it.

Pure: no cv2, no torch, no signal delivery, so the rule is testable off-device.
"""

import signal

# Both are requests to stop. SIGTERM is what a supervisor sends; SIGINT is what a
# terminal sends.
STOP_SIGNALS = (signal.SIGINT, signal.SIGTERM)


class StopRequest:
    """A stop signal, remembered until the loop can leave on its own terms.

    The first request sets ``signum`` and returns, so the loop finishes the frame
    it is on, releases the camera and lets ``atexit`` flush the capture session.
    A second request -- the operator asking again because the first appears to have
    done nothing -- raises ``KeyboardInterrupt`` instead, so a demo wedged inside a
    driver call can still be stopped.
    """

    def __init__(self):
        self.signum = None

    @property
    def requested(self):
        return self.signum is not None

    @property
    def name(self):
        try:
            return signal.Signals(self.signum).name
        except ValueError:            # pragma: no cover -- not a real signal number
            return str(self.signum)

    def __call__(self, signum, frame):
        if self.signum is not None:
            raise KeyboardInterrupt
        self.signum = signum

    def install(self, signaller=signal.signal, signals=STOP_SIGNALS):
        """Take over ``signals`` in this thread (only the main thread may)."""
        for sig in signals:
            signaller(sig, self)
        return self


__all__ = ["STOP_SIGNALS", "StopRequest"]
