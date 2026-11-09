"""Stop requests: remembered once, acted on once, and handled however the demo started.

The failure these pin is silent and expensive: a demo launched as a background job
inherits SIGINT ignored, so Ctrl-C and a supervisor's SIGINT do nothing at all. The
10-28 window run waited 40 s on such a process, killed it, and left the box without
a feed.
"""

import ast
import signal
from pathlib import Path

import pytest

from jetson.demo_stop import STOP_SIGNALS, StopRequest

REPO = Path(__file__).resolve().parents[1]


class _Signaller:
    def __init__(self):
        self.installed = {}

    def __call__(self, signum, handler):
        self.installed[signum] = handler


class TestStopRequest:
    def test_both_stop_signals_get_a_handler(self):
        signaller = _Signaller()
        stop = StopRequest().install(signaller=signaller)
        assert set(signaller.installed) == {signal.SIGINT, signal.SIGTERM}
        assert STOP_SIGNALS == (signal.SIGINT, signal.SIGTERM)
        assert signaller.installed[signal.SIGINT] is stop

    def test_the_first_request_is_remembered_and_does_not_raise(self):
        stop = StopRequest()
        assert stop.requested is False
        stop(signal.SIGTERM, None)            # must return: the loop leaves on its own
        assert stop.requested is True
        assert stop.signum == signal.SIGTERM
        assert stop.name == "SIGTERM"

    def test_a_second_request_stops_waiting(self):
        stop = StopRequest()
        stop(signal.SIGINT, None)
        with pytest.raises(KeyboardInterrupt):
            stop(signal.SIGINT, None)

    def test_the_handler_is_reachable_through_the_os(self):
        """Installed, a real signal sets the flag -- that is the whole fix."""
        stop = StopRequest()
        previous = signal.signal(signal.SIGUSR1, stop)
        try:
            signal.raise_signal(signal.SIGUSR1)
        finally:
            signal.signal(signal.SIGUSR1, previous)
        assert stop.requested is True


class TestTheDemoUsesIt:
    """A correct handler that the demo never installs fixes nothing."""

    def test_the_demo_installs_the_handlers_and_leaves_on_its_own(self):
        source = (REPO / "jetson" / "realtime_demo.py").read_text()
        assert "StopRequest().install()" in source
        assert "while not stop.requested:" in source
        # Leaving normally is what lets the capture session's atexit hook flush.
        assert 'print(f"stopping on {stop.name}", flush=True)' in source

    def test_the_loop_is_not_still_unconditional(self):
        source = (REPO / "jetson" / "realtime_demo.py").read_text()
        assert "    while True:" not in source


def _main_body():
    """The source of ``realtime_demo.main`` alone, so the order is the real one."""
    source = (REPO / "jetson" / "realtime_demo.py").read_text()
    tree = ast.parse(source)
    main = next(node for node in tree.body
                if isinstance(node, ast.FunctionDef) and node.name == "main")
    following = [node.lineno for node in tree.body if node.lineno > main.lineno]
    lines = source.splitlines()
    return "\n".join(lines[main.lineno - 1:(following[0] - 1) if following else None])


class TestTheHandlerIsInstalledBeforeTheLoad:
    """A stop during start-up is where the demo was killed and the feed was lost.

    The handler has to be in place before the model, the engine and the camera are
    touched, or a request that arrives during that stretch meets the inherited
    SIG_IGN and is discarded at delivery rather than held by the interpreter.
    """

    def test_install_comes_before_every_expensive_step(self):
        body = _main_body()
        install = body.index("StopRequest().install()")
        for expensive in ("load_demo_model(", "TrtEngine.load", "cv2.VideoCapture"):
            assert install < body.index(expensive), expensive

    def test_the_handler_is_installed_exactly_once(self):
        body = _main_body()
        assert body.count("StopRequest().install()") == 1

    def test_it_runs_before_the_first_thing_that_can_block(self):
        """Nothing that can wait on a device or a file may precede it."""
        body = _main_body()
        install = body.index("StopRequest().install()")
        for earlier in ("build_capture(", "sha256_file(", "start_server"):
            assert body.index(earlier) > install, earlier
