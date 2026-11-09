"""Resume selection: which stored checkpoint a new run continues from.

Every save path writes architecture "hcm" into the same bucket -- per-epoch,
``_best``, ``_last`` and smoke runs alike -- so "latest wins" would resume a
real run from a 32-batch smoke checkpoint. Found in review of the checkpointing
work, not by the tests written alongside it.
"""

import pytest

from ecs_hcm_model.core.model_upload_download import (
    is_resume_candidate,
    select_resume_checkpoint,
)


def _ckpt(timestamp, is_best=None, is_smoke=None, name="hcm_imagenet1k"):
    """A metadata.json document as it sits in the bucket."""
    custom = {}
    if is_best is not None:
        custom["is_best"] = is_best
    if is_smoke is not None:
        custom["is_smoke"] = is_smoke
    return {
        "name": name,
        "architecture": "hcm",
        "upload_timestamp": timestamp,
        "custom_metadata": custom,
    }


class TestResumeCandidate:
    def test_smoke_is_never_a_candidate(self):
        assert is_resume_candidate(_ckpt("2026-01-05T00:00:00", is_best=True, is_smoke=True)) is False

    def test_best_is_a_candidate(self):
        assert is_resume_candidate(_ckpt("2026-01-05T00:00:00", is_best=True, is_smoke=False)) is True

    def test_non_best_epoch_is_not_a_candidate(self):
        assert is_resume_candidate(_ckpt("2026-01-05T00:00:00", is_best=False, is_smoke=False)) is False

    def test_legacy_checkpoint_stays_loadable(self):
        """No is_best key at all: predates the scheme, must remain eligible."""
        assert is_resume_candidate(_ckpt("2020-01-01T00:00:00")) is True

    def test_missing_custom_metadata_is_legacy(self):
        assert is_resume_candidate({"architecture": "hcm", "upload_timestamp": "2020-01-01T00:00:00"}) is True

    def test_smoke_legacy_without_is_best_is_still_rejected(self):
        """is_smoke alone, with no is_best: the smoke flag must still win."""
        assert is_resume_candidate(_ckpt("2026-01-05T00:00:00", is_smoke=True)) is False


class TestSelectResumeCheckpoint:
    """The four cases the lead asked for, in the owner's real scenario."""

    def test_legacy_only_bucket_picks_latest_legacy(self):
        older = _ckpt("2025-12-01T00:00:00")
        newer = _ckpt("2026-01-05T00:00:00")
        assert select_resume_checkpoint([older, newer]) is newer

    def test_legacy_plus_newer_best_picks_best(self):
        legacy = _ckpt("2025-12-01T00:00:00")
        best = _ckpt("2026-01-05T00:00:00", is_best=True, is_smoke=False)
        assert select_resume_checkpoint([legacy, best]) is best

    def test_smoke_newer_than_best_picks_best(self):
        """The exact failure: a smoke run after the real one must not win."""
        best = _ckpt("2026-01-05T00:00:00", is_best=True, is_smoke=False)
        smoke = _ckpt("2026-01-06T00:00:00", is_best=True, is_smoke=True)
        assert select_resume_checkpoint([best, smoke]) is best

    def test_epoch_or_last_newer_than_best_picks_best(self):
        best = _ckpt("2026-01-05T00:00:00", is_best=True, is_smoke=False)
        epoch = _ckpt("2026-01-06T00:00:00", is_best=False, is_smoke=False)
        last = _ckpt("2026-01-07T00:00:00", is_best=False, is_smoke=False)
        assert select_resume_checkpoint([best, epoch, last]) is best

    def test_no_candidates_returns_none(self):
        smoke = _ckpt("2026-01-06T00:00:00", is_best=True, is_smoke=True)
        epoch = _ckpt("2026-01-07T00:00:00", is_best=False, is_smoke=False)
        assert select_resume_checkpoint([smoke, epoch]) is None

    def test_empty_list_returns_none(self):
        assert select_resume_checkpoint([]) is None

    def test_unparseable_timestamp_raises(self):
        # Changed by resume hardening (#6): a timestamp we read but can't parse
        # is corruption, not age, and used to quietly sort oldest.
        good = _ckpt("2026-01-05T00:00:00")
        bad = _ckpt("not-a-timestamp")
        with pytest.raises(ValueError, match="not-a-timestamp"):
            select_resume_checkpoint([bad, good])

    def test_missing_timestamp_still_sorts_oldest(self):
        good = _ckpt("2026-01-05T00:00:00")
        legacy = _ckpt(None)
        assert select_resume_checkpoint([legacy, good]) is good

    def test_order_does_not_matter(self):
        best = _ckpt("2026-01-05T00:00:00", is_best=True, is_smoke=False)
        smoke = _ckpt("2026-01-06T00:00:00", is_best=True, is_smoke=True)
        assert select_resume_checkpoint([smoke, best]) is best
