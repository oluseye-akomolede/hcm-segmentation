"""Uncertainty scoring and frame selection for camera round 0 (design D1, D2).

Pure logic: no I/O, no threads, no model. The demo computes ``p_fg`` for the
overlay it is about to draw and hands that same tensor to ``uncertain_fraction``,
so the score costs no second forward and cannot disagree with what it drew.

A *night* is one America/Chicago calendar day -- the day the daily cap counts
and the day the capture directory is named (D2) -- so the per-night gate in D9
is computable from the directory alone.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from math import floor
from zoneinfo import ZoneInfo

import torch

# D1: the band is the same distance below the demo's 0.65 display gate, so
# "undecided" means "flipping the display rule would change this pixel".
BAND_LOW = 0.35
BAND_HIGH = 0.65

# D2: a candidate needs about 2,480 undecided pixels on the 352 grid.
SCORE_FLOOR = 0.02
MIN_SPACING_S = 20.0
HOURLY_CAP = 40
DAILY_CAP = 300

# D2: the clock-chosen control sample. At 5 minutes a counted night at the caps'
# floor yields ~30 controls; at 20 minutes it would yield 7, which is noise.
CONTROL_PERIOD_S = 300.0
CONTROL_DAILY_CAP = 288

LOCAL_TZ = ZoneInfo("America/Chicago")

UNCERTAIN = "uncertain"
CONTROL = "control"

# One counter key per refusal, so the report can say why a frame was not saved
# without re-reading the frames.
REFUSALS = ("below_floor", "spacing", "hourly_cap", "daily_cap", "control_cap")


class Counters:
    """Named counts, one key per reason. Shared by the policy, writer and uploader."""

    def __init__(self):
        self._counts = {}

    def bump(self, key, n=1):
        self._counts[key] = self._counts.get(key, 0) + n

    def get(self, key, default=0):
        return self._counts.get(key, default)

    def as_dict(self):
        return dict(sorted(self._counts.items()))


def local_day(ts, tz=LOCAL_TZ):
    """``YYYYmmdd`` for the America/Chicago day ``ts`` (UTC epoch seconds) falls in.

    Not UTC: the owner counts nights, and the daily cap, the capture directory
    and the gate's per-night grouping are all that same day (D2).
    """
    return datetime.fromtimestamp(float(ts), tz).strftime("%Y%m%d")


def uncertain_fraction(p_fg, lo=BAND_LOW, hi=BAND_HIGH):
    """Fraction of pixels whose foreground probability is strictly inside ``(lo, hi)``.

    ``p_fg`` is the ``[H,W]`` tensor the demo's overlay is drawn from; the score
    is the share of pixels that would change side if the display gate moved.
    """
    band = (p_fg > lo) & (p_fg < hi)
    total = band.numel()
    if not total:
        raise ValueError("p_fg is empty: there is no frame to score")
    return float(band.sum().item()) / float(total)


def max_p_fg(p_fg):
    """The frame's most confident foreground pixel. Recorded beside the score."""
    return float(p_fg.max())


@dataclass(frozen=True)
class SaveDecision:
    """One frame the policy chose to keep, and why."""

    kind: str
    ts: float


class CapturePolicy:
    """Which frames to save: sparse, capped in two dimensions, controls by the clock.

    ``clock`` is injected as UTC epoch seconds so a test can walk over an hour,
    a local midnight and a 5-minute boundary without sleeping.
    """

    def __init__(self, *, clock=time.time, tz=LOCAL_TZ,
                 min_spacing=MIN_SPACING_S, hourly_cap=HOURLY_CAP, daily_cap=DAILY_CAP,
                 score_floor=SCORE_FLOOR, control_period=CONTROL_PERIOD_S,
                 control_daily_cap=CONTROL_DAILY_CAP, counts=None):
        self.clock = clock
        self.tz = tz
        self.min_spacing = float(min_spacing)
        self.hourly_cap = int(hourly_cap)
        self.daily_cap = int(daily_cap)
        self.score_floor = float(score_floor)
        self.control_period = float(control_period)
        self.control_daily_cap = int(control_daily_cap)
        self.counts = counts if counts is not None else Counters()

        self.saved = {UNCERTAIN: 0, CONTROL: 0}
        self._saved_times = deque()   # uncertain saves inside the rolling hour
        self._last_saved = None       # any save, either kind
        self._day = None
        self._uncertain_today = 0
        self._controls_today = 0
        self._next_control = None     # the 5-minute boundary not yet sampled

    def offer(self, score, now=None):
        """Offer one frame's score; return the ``SaveDecision`` or None.

        The control sample is taken first and independently: at a boundary the
        frame is kept whatever its score, and even when the uncertain caps are
        full, because the control has to be ordinary content (D2).
        """
        now = float(self.clock() if now is None else now)
        self._roll(now)

        decision = self._control_at_boundary(now)
        if decision is not None:
            return decision

        if score < self.score_floor:
            self.counts.bump("below_floor")
            return None
        if self._last_saved is not None and now - self._last_saved < self.min_spacing:
            self.counts.bump("spacing")
            return None

        cutoff = now - 3600.0
        while self._saved_times and self._saved_times[0] <= cutoff:
            self._saved_times.popleft()
        if len(self._saved_times) >= self.hourly_cap:
            self.counts.bump("hourly_cap")
            return None
        if self._uncertain_today >= self.daily_cap:
            self.counts.bump("daily_cap")
            return None

        self._saved_times.append(now)
        self._uncertain_today += 1
        self._record_save(now, UNCERTAIN)
        return SaveDecision(UNCERTAIN, now)

    # -- internals ---------------------------------------------------------

    def _record_save(self, now, kind):
        self._last_saved = now
        self.saved[kind] += 1

    def _roll(self, now):
        """Reset the day's counters when the America/Chicago date changes."""
        day = local_day(now, self.tz)
        if day != self._day:
            self._day = day
            self._uncertain_today = 0
            self._controls_today = 0

    def _control_at_boundary(self, now):
        """Save the first frame after a 5-minute boundary; None otherwise."""
        if self._next_control is None:
            # The first frame of a session is the control for the slot it starts
            # in, so a short session still contributes a control.
            self._next_control = floor(now / self.control_period) * self.control_period
        if now < self._next_control:
            return None
        # Move past this boundary whatever the outcome: one frame per boundary,
        # so a full control cap is counted once per boundary, not per frame.
        self._next_control = (floor(now / self.control_period) + 1) * self.control_period
        if self._controls_today >= self.control_daily_cap:
            self.counts.bump("control_cap")
            return None
        self._controls_today += 1
        self._record_save(now, CONTROL)
        return SaveDecision(CONTROL, now)
