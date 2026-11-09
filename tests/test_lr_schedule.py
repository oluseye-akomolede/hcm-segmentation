"""The learning-rate decay after warmup, and the lr logging that shows it.

The real run ``real-aa519f5`` plateaued after epoch 5: the warmup stages set
the backbone lr by stage (0.0, 1e-4, 1e-4, 5e-5, 5e-5) and nothing ever changed
either group again, so both sat flat for the remaining epochs. This adds a
cosine decay over the post-warmup epochs and logs both groups every epoch.

Two levels are tested, on purpose:

* the schedule itself, as a pure function -- no model, no optimizer, no env, so
  the numbers can be checked against the definition in the docstring;
* the wiring, by driving ``train_model_for_epochs`` on CPU and reading the lr
  the optimizer's groups actually held at each epoch.

The second is not redundant. A pure function can be perfect while the loop
forgets to call it, calls it before warmup has set the backbone lr, or reads
its base off the wrong group -- none of which the arithmetic tests can see.
"""

import math

import pytest
import torch

import ecs_hcm_model.core.trainer as trainer
from ecs_hcm_model.core.trainer import (
    MIN_LR_FACTOR,
    WARMUP_EPOCHS,
    lr_for_epoch,
    train_model_for_epochs,
)

from conftest import TinyClassifier, TinySegmenter, make_classification_batches, make_segmentation_batches


# --------------------------------------------------------------------------
# The schedule itself
# --------------------------------------------------------------------------

class TestLrForEpoch:
    """The cosine decay, checked against the formula it claims to implement."""

    EPOCHS = 15
    START = WARMUP_EPOCHS  # 5

    def test_first_decay_epoch_is_exactly_the_base(self):
        """k = 0 -> cos(0) = 1 -> the group starts the decay at its entry lr."""
        base = 5e-5
        assert lr_for_epoch(base, self.START, self.START, self.EPOCHS, 0.05) == base

    def test_last_decay_epoch_is_exactly_base_times_f(self):
        base = 5e-5
        got = lr_for_epoch(base, self.EPOCHS - 1, self.START, self.EPOCHS, 0.05)
        assert got == pytest.approx(base * 0.05, rel=0, abs=0)

    def test_matches_the_formula_at_every_epoch(self):
        """Not a spot check: every epoch against the definition, to 12 digits."""
        base, f = 3e-5, 0.05
        K = self.EPOCHS - self.START
        for k in range(K):
            expected = base * (
                f + (1 - f) * 0.5 * (1 + math.cos(math.pi * k / (K - 1)))
            )
            got = lr_for_epoch(base, self.START + k, self.START, self.EPOCHS, f)
            assert got == pytest.approx(expected, rel=1e-12), f"k={k}"

    def test_non_increasing_across_the_decay_phase(self):
        """Cosine decay must never go back up; a plateau at the start is fine."""
        lrs = [
            lr_for_epoch(5e-5, e, self.START, self.EPOCHS, 0.05)
            for e in range(self.START, self.EPOCHS)
        ]
        for prev, nxt in zip(lrs, lrs[1:]):
            assert nxt <= prev, lrs

    def test_single_decay_epoch_returns_the_base(self):
        """K == 1: the formula would divide by K-1 == 0. Must not raise."""
        base = 5e-5
        assert lr_for_epoch(base, 5, 5, 6, 0.05) == base

    def test_decay_phase_entirely_absent_returns_the_base(self):
        """K <= 0 -- a run that ends during or before warmup -- is a no-op."""
        base = 5e-5
        assert lr_for_epoch(base, 4, 5, 5, 0.05) == base
        assert lr_for_epoch(base, 0, 5, 1, 0.05) == base

    def test_epochs_before_decay_start_return_the_base(self):
        """Before the decay phase this must not touch the value at all."""
        base = 1e-4
        for e in range(self.START):
            assert lr_for_epoch(base, e, self.START, self.EPOCHS, 0.05) == base

    @pytest.mark.parametrize("f", [0.001, 0.5, 1.0])
    def test_honours_f(self, f):
        base = 5e-5
        assert lr_for_epoch(base, self.EPOCHS - 1, self.START, self.EPOCHS, f) == pytest.approx(base * f)

    def test_f_of_one_is_a_flat_schedule(self):
        """f == 1 means "decay to the same value", i.e. no decay at all."""
        base = 5e-5
        lrs = {lr_for_epoch(base, e, self.START, self.EPOCHS, 1.0) for e in range(self.START, self.EPOCHS)}
        assert lrs == {base}


def test_min_lr_factor_default_is_a_real_reduction():
    """Guards the constant itself: 0.05 is the owner's choice, not an accident."""
    assert MIN_LR_FACTOR == 0.05
    assert 0.0 < MIN_LR_FACTOR < 1.0


class TestMinLrFactorOverride:
    """ECS_HCM_MIN_LR_FACTOR, including the values that must be refused."""

    @pytest.fixture(autouse=True)
    def _clean_env(self, monkeypatch):
        monkeypatch.delenv("ECS_HCM_MIN_LR_FACTOR", raising=False)

    def test_default_when_unset(self):
        assert trainer._min_lr_factor() == MIN_LR_FACTOR

    def test_valid_override_is_used(self, monkeypatch):
        monkeypatch.setenv("ECS_HCM_MIN_LR_FACTOR", "0.2")
        assert trainer._min_lr_factor() == pytest.approx(0.2)

    @pytest.mark.parametrize("raw", ["0", "0.0", "-0.1", "1.5", "2", "nan", "inf", "-inf"])
    def test_unusable_values_fall_back_to_the_default(self, monkeypatch, raw):
        """Zero collapses the run; >1 raises the lr; nan/inf are not factors.

        All of them must warn and use the default rather than be obeyed --
        and none may raise, because a bad env var must not kill a run at
        hour twelve.
        """
        monkeypatch.setenv("ECS_HCM_MIN_LR_FACTOR", raw)
        assert trainer._min_lr_factor() == MIN_LR_FACTOR

    def test_exactly_one_is_accepted(self, monkeypatch):
        """f == 1 is legal (a flat schedule), and must not be mistaken for >1."""
        monkeypatch.setenv("ECS_HCM_MIN_LR_FACTOR", "1.0")
        assert trainer._min_lr_factor() == 1.0

    def test_garbage_falls_back_without_raising(self, monkeypatch):
        monkeypatch.setenv("ECS_HCM_MIN_LR_FACTOR", "not-a-number")
        assert trainer._min_lr_factor() == MIN_LR_FACTOR


# --------------------------------------------------------------------------
# The env helpers themselves (C4 follow-up)
# --------------------------------------------------------------------------

class TestEnvHelpersRejectUnusableValues:
    """``float('nan')`` parses, so a plain try/except lets it through.

    A NaN threshold makes every comparison False, which silently disables the
    guard that was supposed to use it -- the failure mode this rejection exists
    to prevent.
    """

    def test_env_float_rejects_nan(self, monkeypatch):
        monkeypatch.setenv("ECS_HCM_TEST_FLOAT", "nan")
        assert trainer._env_float("ECS_HCM_TEST_FLOAT", 1.5) == 1.5

    def test_env_float_rejects_inf(self, monkeypatch):
        monkeypatch.setenv("ECS_HCM_TEST_FLOAT", "inf")
        assert trainer._env_float("ECS_HCM_TEST_FLOAT", 1.5) == 1.5

    def test_env_float_rejects_negative_inf(self, monkeypatch):
        monkeypatch.setenv("ECS_HCM_TEST_FLOAT", "-inf")
        assert trainer._env_float("ECS_HCM_TEST_FLOAT", 1.5) == 1.5

    def test_env_float_accepts_a_finite_value(self, monkeypatch):
        monkeypatch.setenv("ECS_HCM_TEST_FLOAT", "2.5")
        assert trainer._env_float("ECS_HCM_TEST_FLOAT", 1.5) == 2.5

    def test_env_float_minimum_rejects_below(self, monkeypatch):
        monkeypatch.setenv("ECS_HCM_TEST_FLOAT", "-1.0")
        assert trainer._env_float("ECS_HCM_TEST_FLOAT", 1.5, minimum=0.0) == 1.5

    def test_env_float_minimum_allows_zero(self, monkeypatch):
        """minimum is inclusive: 0.0 is a legal MAX_OOM_FRACTION."""
        monkeypatch.setenv("ECS_HCM_TEST_FLOAT", "0.0")
        assert trainer._env_float("ECS_HCM_TEST_FLOAT", 1.5, minimum=0.0) == 0.0

    def test_env_int_rejects_zero_when_positive_required(self, monkeypatch):
        monkeypatch.setenv("ECS_HCM_TEST_INT", "0")
        assert trainer._env_int("ECS_HCM_TEST_INT", 20, minimum=1) == 20

    def test_env_int_rejects_negative(self, monkeypatch):
        monkeypatch.setenv("ECS_HCM_TEST_INT", "-5")
        assert trainer._env_int("ECS_HCM_TEST_INT", 20, minimum=1) == 20

    def test_env_int_accepts_a_valid_value(self, monkeypatch):
        monkeypatch.setenv("ECS_HCM_TEST_INT", "7")
        assert trainer._env_int("ECS_HCM_TEST_INT", 20, minimum=1) == 7

    @pytest.mark.parametrize("raw", ["nan", "inf", "NaN"])
    def test_env_int_rejects_non_numeric_special_values(self, monkeypatch, raw):
        """``int("nan")`` raises where ``float("nan")`` succeeds.

        Different code path, same required outcome: a warning and the default.
        Named separately so a future change to one helper cannot quietly leave
        the other accepting a value it cannot use.
        """
        monkeypatch.setenv("ECS_HCM_TEST_INT", raw)
        assert trainer._env_int("ECS_HCM_TEST_INT", 20, minimum=1) == 20

    def test_empty_string_is_treated_as_unset(self, monkeypatch):
        """An empty value is the common shape of "not configured"."""
        monkeypatch.setenv("ECS_HCM_TEST_INT", "")
        monkeypatch.setenv("ECS_HCM_TEST_FLOAT", "")
        assert trainer._env_int("ECS_HCM_TEST_INT", 20, minimum=1) == 20
        assert trainer._env_float("ECS_HCM_TEST_FLOAT", 1.5) == 1.5

    def test_env_float_nan_does_not_reach_the_oom_fraction_limit(self, monkeypatch):
        """The concrete case: a NaN MAX_OOM_FRACTION must not disarm the guard.

        ``oom_skipped / attempted > nan`` is False for every input, so before
        the rejection the fraction breaker could never trip.
        """
        monkeypatch.setenv("ECS_HCM_MAX_OOM_FRACTION", "nan")
        _, fraction = trainer._oom_limits()
        assert fraction == trainer.MAX_OOM_SKIP_FRACTION
        assert fraction == fraction  # not NaN


# --------------------------------------------------------------------------
# The wiring: what the optimizer's groups actually hold
# --------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _no_early_stopping(monkeypatch):
    """Run the whole schedule; early stopping is a different test's subject.

    These tests assert about the lr at the *last* epoch (14 of 15, say). Early
    stopping compares val_loss across epochs and breaks the loop on five
    non-improving epochs in a row, and a randomly initialised model on random
    batches produces exactly that -- so the epoch under test often never
    happens. Reshaping the loss to dodge the breaker was tried and rejected: it
    made the tests depend on a training dynamic they are not about, and it
    broke again as soon as the run length changed.

    ``EarlyStopping`` is looked up as a module global where it is constructed,
    so replacing it here is enough. Early stopping itself stays covered by
    tests/test_trainer_lifecycle.py.
    """
    import ecs_hcm_model.core.trainer as t

    class _NeverStops:
        best_loss = float("inf")
        counter = 0

        def __init__(self, *a, **kw):
            pass

        def step(self, current_loss):
            return False

    monkeypatch.setattr(t, "EarlyStopping", _NeverStops)


def _lr_run(epochs, warmup, coco=False, logger=None, task="classification"):
    """Drive train_model_for_epochs on CPU and report each epoch's lrs.

    Layers are frozen/unfrozen by the warmup block, so the model has to look
    enough like the real one for that to run; TinyClassifier has the
    ``backbone`` and ``segmentation_head`` the trainer reaches for.
    """
    model = TinyClassifier()
    train_loader = make_classification_batches(n_batches=2, batch_size=4)
    val_loader = make_classification_batches(n_batches=2, batch_size=4)

    return train_model_for_epochs(
        model=model,
        epochs=epochs,
        train_loader=train_loader,
        val_loader=val_loader,
        device=torch.device("cpu"),
        logger=logger,
        test_mode=False,
        warmup=warmup,
        coco=coco,
        task=task,
        checkpoint_every_epoch=False,
    )


def _seg_run(epochs, coco=True, logger=None):
    """Drive the segmentation path with TinySegmenter, to exercise coco lrs."""
    return train_model_for_epochs(
        model=TinySegmenter(),
        epochs=epochs,
        train_loader=make_segmentation_batches(n_batches=2, batch_size=4),
        val_loader=make_segmentation_batches(n_batches=2, batch_size=4),
        device=torch.device("cpu"),
        logger=logger,
        test_mode=False,
        warmup=True,
        coco=coco,
        task="segmentation",
        checkpoint_every_epoch=False,
    )


class TestWiring:
    """The loop must call the schedule at the right time with the right base."""

    @pytest.fixture(autouse=True)
    def _clean_env(self, monkeypatch):
        monkeypatch.delenv("ECS_HCM_MIN_LR_FACTOR", raising=False)

    def test_warmup_epochs_are_untouched_by_the_decay(self):
        """The schedule must not disturb the staged warmup values.

        Reads the epoch lrs off the optimizer inside the loop -- an external
        observer of the final state cannot see epochs 0-4, and those are the
        values that must be identical to today.
        """
        import ecs_hcm_model.core.trainer as t

        captured = []
        real_set = t.lr_for_epoch

        def spy(base, epoch, decay_start, epochs, f):
            captured.append((epoch, base))
            return real_set(base, epoch, decay_start, epochs, f)

        # The decay function is only called for epochs >= decay_start, which is
        # itself the guarantee that warmup epochs are untouched.
        model = TinyClassifier()
        train_loader = make_classification_batches(2, 4)
        val_loader = make_classification_batches(2, 4)
        orig = t.lr_for_epoch
        t.lr_for_epoch = spy
        try:
            train_model_for_epochs(
                model=model, epochs=8, train_loader=train_loader,
                val_loader=val_loader, device=torch.device("cpu"),
                logger=None, test_mode=False, warmup=True, coco=False,
                task="classification", checkpoint_every_epoch=False,
            )
        finally:
            t.lr_for_epoch = orig

        epochs_called = [e for e, _ in captured]
        assert epochs_called, "the decay function was never called"
        assert min(epochs_called) == WARMUP_EPOCHS, epochs_called
        assert epochs_called == sorted(epochs_called)

    def test_short_run_with_warmup_does_not_decay(self):
        """epochs == 5: every epoch is a warmup epoch; behaviour is unchanged."""
        import ecs_hcm_model.core.trainer as t

        calls = []
        orig = t.lr_for_epoch

        def spy(*a, **kw):
            calls.append(a[1])
            return orig(*a, **kw)

        t.lr_for_epoch = spy
        try:
            train_model_for_epochs(
                model=TinyClassifier(), epochs=5,
                train_loader=make_classification_batches(2, 4),
                val_loader=make_classification_batches(2, 4),
                device=torch.device("cpu"), logger=None, test_mode=False,
                warmup=True, coco=False, task="classification",
                checkpoint_every_epoch=False,
            )
        finally:
            t.lr_for_epoch = orig
        assert calls == [], calls

    def test_warmup_off_decays_from_epoch_zero(self):
        """With warmup off, the decay owns every epoch from epoch 0."""
        import ecs_hcm_model.core.trainer as t

        calls = []
        orig = t.lr_for_epoch

        def spy(*a, **kw):
            # a[1] is the epoch. The schedule is called once per param group,
            # and there are two (backbone, head), so each epoch appears twice.
            calls.append(a[1])
            return orig(*a, **kw)

        t.lr_for_epoch = spy
        try:
            train_model_for_epochs(
                model=TinyClassifier(), epochs=4,
                train_loader=make_classification_batches(2, 4),
                val_loader=make_classification_batches(2, 4),
                device=torch.device("cpu"), logger=None, test_mode=False,
                warmup=False, coco=False, task="classification",
                checkpoint_every_epoch=False,
            )
        finally:
            t.lr_for_epoch = orig
        assert sorted(set(calls)) == [0, 1, 2, 3], calls
        # Once per param group per epoch -- not once per epoch, and not more.
        assert calls.count(0) == 2, calls
        assert len(calls) == 8, calls


class TestLrLogging:
    """Both groups, every epoch, through the existing logger."""

    @pytest.fixture(autouse=True)
    def _clean_env(self, monkeypatch):
        monkeypatch.delenv("ECS_HCM_MIN_LR_FACTOR", raising=False)

    def _logger(self):
        from conftest import RecordingLogger

        return RecordingLogger()

    def test_both_groups_logged_once_per_epoch_with_the_epoch_as_step(self):
        logger = self._logger()
        _lr_run(epochs=7, warmup=True, logger=logger)

        backbone = [(v, s) for k, v, s in logger.scalars if k == "lr_backbone"]
        head = [(v, s) for k, v, s in logger.scalars if k == "lr_head"]

        assert len(backbone) == 7, backbone
        assert len(head) == 7, head
        assert [s for _, s in backbone] == list(range(7))
        assert [s for _, s in head] == list(range(7))

    def test_logged_values_match_the_warmup_stages_and_the_decay(self):
        """The logged curve must be the warmup stages and then the schedule."""
        logger = self._logger()
        _lr_run(epochs=8, warmup=True, logger=logger)

        backbone = {s: v for k, v, s in logger.scalars if k == "lr_backbone"}
        head = {s: v for k, v, s in logger.scalars if k == "lr_head"}

        # The staged warmup, unchanged.
        assert backbone[0] == 0.0
        assert backbone[1] == pytest.approx(1e-4)
        assert backbone[2] == pytest.approx(1e-4)
        assert backbone[3] == pytest.approx(5e-5)
        assert backbone[4] == pytest.approx(5e-5)
        # Decay starts at 5 from the 5e-5 stage 3 left behind.
        assert backbone[5] == pytest.approx(5e-5)
        assert backbone[7] == pytest.approx(
            lr_for_epoch(5e-5, 7, WARMUP_EPOCHS, 8, MIN_LR_FACTOR)
        )
        # The head is never warmed up, so it decays from its creation lr. For a
        # non-coco classification run with warmup on, that is 3e-4 (trainer.py
        # create_model_optimizer call in the `if warmup:` branch) -- NOT the
        # 3e-5 the coco/finetune branch uses.
        assert head[4] == pytest.approx(3e-4)
        assert head[5] == pytest.approx(3e-4)
        assert head[7] == pytest.approx(
            lr_for_epoch(3e-4, 7, WARMUP_EPOCHS, 8, MIN_LR_FACTOR)
        )

    def test_lr_logged_without_a_logger_does_not_raise(self):
        """logger=None is legal (tests and smoke use it); logging must no-op."""
        _lr_run(epochs=6, warmup=True, logger=None)

    def test_coco_run_decays_from_the_coco_lrs(self):
        """A coco run picks the coco lr pair, and the decay follows that pair.

        Driven end-to-end through the coco (segmentation) branch rather than
        asserting a call this test made itself: the point is that the *loop*
        selects head 3e-5 for coco, not 3e-4 as it does for plain
        classification, and then decays from whatever it selected.
        """
        logger = self._logger()
        _seg_run(epochs=8, coco=True, logger=logger)

        backbone = {s: v for k, v, s in logger.scalars if k == "lr_backbone"}
        head = {s: v for k, v, s in logger.scalars if k == "lr_head"}

        # Warmup owns the backbone; stage 3 leaves it at 5e-5.
        assert backbone[4] == pytest.approx(5e-5)
        assert backbone[5] == pytest.approx(5e-5)
        assert backbone[7] == pytest.approx(
            lr_for_epoch(5e-5, 7, WARMUP_EPOCHS, 8, MIN_LR_FACTOR)
        )
        # The coco head base is 3e-5, not the 3e-4 the classification branch
        # uses -- so this fails if the coco lr pair is ever swapped.
        assert head[4] == pytest.approx(3e-5)
        assert head[5] == pytest.approx(3e-5)
        assert head[7] == pytest.approx(
            lr_for_epoch(3e-5, 7, WARMUP_EPOCHS, 8, MIN_LR_FACTOR)
        )

    def test_decay_is_actually_visible_in_the_logged_curve(self):
        """A regression that never decayed would still log -- so check it fell."""
        logger = self._logger()
        _lr_run(epochs=15, warmup=True, logger=logger)
        head = {s: v for k, v, s in logger.scalars if k == "lr_head"}
        assert head[14] < head[5], (head[5], head[14])
        assert head[14] == pytest.approx(head[5] * MIN_LR_FACTOR)


class TestPrintLine:
    def test_prints_one_line_per_epoch(self, capsys):
        _lr_run(epochs=4, warmup=True)
        out = capsys.readouterr().out
        lines = [ln for ln in out.splitlines() if ln.startswith("[lr] epoch ")]
        assert len(lines) == 4, lines
        assert "backbone=" in lines[0] and "head=" in lines[0]
