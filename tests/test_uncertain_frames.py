"""Score, selection caps and the per-night clock (jetson/uncertain_frames.py)."""

from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import pytest
import torch

from jetson.uncertain_frames import (
    BAND_HIGH,
    BAND_LOW,
    CONTROL,
    CONTROL_DAILY_CAP,
    CONTROL_PERIOD_S,
    DAILY_CAP,
    HOURLY_CAP,
    MIN_SPACING_S,
    SCORE_FLOOR,
    UNCERTAIN,
    CapturePolicy,
    Counters,
    local_day,
    max_p_fg,
    uncertain_fraction,
)

LOCAL = ZoneInfo("America/Chicago")


def _ts(y, mo, d, h=0, mi=0, s=0):
    """Epoch seconds for a local (America/Chicago) wall-clock time."""
    return datetime(y, mo, d, h, mi, s, tzinfo=LOCAL).timestamp()


def _opened(t, **kw):
    """A policy whose opening control is already spent and whose spacing is clear.

    Every session starts by saving one control for the slot it opens in, so a
    test about *uncertain* frames has to get that frame out of the way first.
    """
    policy = CapturePolicy(clock=lambda: t[0], **kw)
    assert policy.offer(0.0).kind == CONTROL
    t[0] += MIN_SPACING_S
    return policy


class TestUncertainFraction:
    def test_all_undecided_is_one(self):
        assert uncertain_fraction(torch.full((8, 8), 0.5)) == pytest.approx(1.0)

    def test_all_confident_is_zero(self):
        assert uncertain_fraction(torch.full((8, 8), 0.9)) == pytest.approx(0.0)

    def test_a_pixel_on_a_bound_is_decided(self):
        # Exactly on a bound is decided: moving the display rule across it would
        # not flip the pixel.
        p = torch.tensor([[BAND_LOW, BAND_HIGH, 0.5, 0.51]])
        assert uncertain_fraction(p) == pytest.approx(0.5)

    def test_the_fraction_does_not_depend_on_the_grid_size(self):
        p = torch.zeros(352, 352)
        p[:176] = 0.5
        assert uncertain_fraction(p) == pytest.approx(0.5, abs=1e-3)

    def test_max_p_fg_is_the_peak(self):
        assert max_p_fg(torch.tensor([[0.1, 0.7], [0.2, 0.3]])) == pytest.approx(0.7)

    def test_an_empty_grid_is_refused(self):
        with pytest.raises(ValueError):
            uncertain_fraction(torch.zeros(0, 0))


class TestCaps:
    def test_spacing_thins_consecutive_frames(self):
        t = [_ts(2026, 11, 2, 12, 0, 0)]
        policy = _opened(t)
        assert policy.offer(0.9) is not None
        t[0] += MIN_SPACING_S - 1
        assert policy.offer(0.9) is None
        assert policy.counts.get("spacing") == 1
        t[0] += 1
        assert policy.offer(0.9) is not None

    def test_the_score_floor_drops_and_counts(self):
        t = [_ts(2026, 11, 2, 12, 0, 0)]
        policy = _opened(t)
        assert policy.offer(SCORE_FLOOR - 0.001) is None
        assert policy.counts.get("below_floor") == 1

    def test_a_frame_on_the_floor_is_kept(self):
        t = [_ts(2026, 11, 2, 12, 0, 0)]
        policy = _opened(t)
        assert policy.offer(SCORE_FLOOR) is not None

    def test_floor_and_spacing_are_counted_separately(self):
        t = [_ts(2026, 11, 2, 12, 0, 0)]
        policy = _opened(t)
        assert policy.offer(0.001) is None          # floor; no save, so no spacing
        assert policy.offer(0.9) is not None        # saved
        assert policy.offer(0.9) is None            # now spacing
        assert policy.counts.get("below_floor") == 1
        assert policy.counts.get("spacing") == 1

    def test_the_hourly_cap_stops_saves(self):
        t = [_ts(2026, 11, 2, 12, 0, 0)]
        policy = _opened(t, control_period=1e9)     # no controls mid-test
        for _ in range(HOURLY_CAP + 5):
            policy.offer(0.9)
            t[0] += MIN_SPACING_S
        assert policy.saved[UNCERTAIN] == HOURLY_CAP
        assert policy.counts.get("hourly_cap") >= 5

    def test_the_daily_cap_stops_saves(self):
        t = [_ts(2026, 11, 2, 0, 0, 0)]
        policy = _opened(t, min_spacing=0, hourly_cap=10_000, control_period=1e9)
        for _ in range(DAILY_CAP + 3):
            t[0] += 60
            policy.offer(0.9)
        assert policy.saved[UNCERTAIN] == DAILY_CAP
        assert policy.counts.get("daily_cap") >= 1


class TestLocalDay:
    def test_the_day_rolls_at_local_midnight_not_utc(self):
        midnight = _ts(2026, 11, 2, 0, 0, 0)
        before, after = midnight - 1, midnight
        assert local_day(before) == "20261101"
        assert local_day(after) == "20261102"
        # Both instants are the *same* UTC date, so a UTC day would not have
        # split them: this test fails unless the day is local.
        utc_before = datetime.fromtimestamp(before, timezone.utc).date()
        utc_after = datetime.fromtimestamp(after, timezone.utc).date()
        assert utc_before == utc_after

    def test_the_daily_counter_resets_on_the_local_day(self):
        t = [_ts(2026, 11, 2, 23, 59, 0)]
        policy = _opened(t, min_spacing=0, daily_cap=1, control_period=1e9)
        assert policy.offer(0.9) is not None         # the day's one uncertain frame
        assert policy.offer(0.9) is None             # cap reached
        assert policy.counts.get("daily_cap") == 1
        t[0] = _ts(2026, 11, 3, 0, 0, 1)
        assert policy.offer(0.9) is not None         # a new local day


class TestControls:
    def test_the_opening_frame_is_a_control_regardless_of_score(self):
        policy = CapturePolicy(clock=lambda: _ts(2026, 11, 2, 12, 0, 0))
        decision = policy.offer(0.0)
        assert decision is not None and decision.kind == CONTROL

    def test_one_control_per_boundary(self):
        t = [_ts(2026, 11, 2, 12, 0, 0)]
        policy = CapturePolicy(clock=lambda: t[0])
        assert policy.offer(0.9).kind == CONTROL
        t[0] += 1
        assert policy.offer(0.9) is None             # spacing, not a second control
        t[0] = _ts(2026, 11, 2, 12, 5, 1)
        assert policy.offer(0.9).kind == CONTROL

    def test_a_control_is_saved_after_the_uncertain_caps_are_full(self):
        t = [_ts(2026, 11, 2, 12, 0, 0)]
        policy = CapturePolicy(clock=lambda: t[0], min_spacing=0, daily_cap=1,
                               hourly_cap=1)
        assert policy.offer(0.9).kind == CONTROL     # opening control
        assert policy.offer(0.9).kind == UNCERTAIN   # the one uncertain frame
        assert policy.offer(0.9) is None             # caps are full
        t[0] = _ts(2026, 11, 2, 12, 5, 1)
        decision = policy.offer(0.0)
        assert decision is not None and decision.kind == CONTROL
        assert policy.saved[UNCERTAIN] == 1          # controls do not count

    def test_the_control_cap_stops_controls_and_counts(self):
        t = [_ts(2026, 11, 2, 0, 0, 0)]
        policy = CapturePolicy(clock=lambda: t[0], min_spacing=0, control_daily_cap=2)
        controls = 0
        for i in range(5):
            t[0] = _ts(2026, 11, 2, 0, 0, 0) + i * CONTROL_PERIOD_S
            decision = policy.offer(0.0)
            if decision is not None and decision.kind == CONTROL:
                controls += 1
        assert controls == 2
        assert policy.counts.get("control_cap") >= 1

    def test_the_cadence_and_cap_are_the_ones_the_spec_names(self):
        assert CONTROL_PERIOD_S == 300.0
        assert CONTROL_DAILY_CAP == 288


class TestCounters:
    def test_a_fresh_policy_reports_nothing(self):
        assert Counters().as_dict() == {}

    def test_counters_are_named_per_reason(self):
        counts = Counters()
        t = [_ts(2026, 11, 2, 12, 0, 0)]
        policy = CapturePolicy(clock=lambda: t[0], counts=counts, min_spacing=0,
                               daily_cap=1, control_period=1e9)
        policy.offer(0.001)     # opening control, then below floor
        policy.offer(0.001)
        assert counts.get("below_floor") == 1
        policy.offer(0.9)
        policy.offer(0.9)
        assert counts.get("daily_cap") == 1
