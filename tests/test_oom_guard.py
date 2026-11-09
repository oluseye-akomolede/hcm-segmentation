"""The OOM circuit breakers, and the accounting that makes them visible.

Regression target: run ``real-aa519f5`` was double-booked onto a V100 another
workload held (~1.6 GiB free). Every one of its 14,664 training batches raised
CUDA OOM. The per-batch handlers printed and ``continue``d, so the epoch was
reported as *completed* having trained nothing, ``batch_count`` stayed at 0 --
which meant the OOM evidence inside ``if batch_count > 0:`` was never reached --
and the run raised nothing until ``evaluate()`` blew up uncaught.

These tests are CPU-only. CUDA OOM is injected as a ``RuntimeError`` exactly as
PyTorch raises it, so the handlers are exercised on the string they actually
match, and ``torch.cuda.mem_get_info`` is monkeypatched for the preflight.
"""

import pytest
import torch

import ecs_hcm_model.core.trainer as trainer
from ecs_hcm_model.core.trainer import (
    MAX_CONSECUTIVE_OOM_SKIPS,
    MIN_BATCHES_FOR_OOM_FRACTION,
    train_model_for_epochs,
    evaluate,
)

from conftest import (
    RecordingLogger,
    TinyClassifier,
    make_classification_batches,
)


def _oom(msg="CUDA out of memory. Tried to allocate 2.00 GiB"):
    return RuntimeError(msg)


class FlakyModel(TinyClassifier):
    """A classifier whose forwards OOM according to a script.

    ``plan`` is a list of batch indices (within the *training* loop, resetting
    each epoch) that raise CUDA OOM. Everything not listed succeeds.

    Two properties matter and are deliberately explicit:

    * the index is the training batch index, so an epoch's script restarts with
      it -- otherwise a plan would have to know how many validation forwards
      happened in between;
    * it raises only while ``enabled``, so the validation path can be run on the
      same instance without inheriting the training script. A shared counter
      across train and val was exactly the bug in an earlier version of this
      file: the OOMs leaked into ``evaluate()`` and the run failed with
      "validation OOM" instead of the epoch-level reason under test.
    """

    def __init__(self, oom_indices=(), always_oom=False, **kw):
        super().__init__(**kw)
        # 1-based, matching `train_calls`: the first training forward is 1, not
        # 0. Keeping the two on the same footing is what stops an off-by-one
        # plan from silently skipping an index (an earlier version of this file
        # used 0-based plans against a 1-based counter, which made every count
        # come out exactly one short).
        self.oom_indices = set(oom_indices)
        self.always_oom = always_oom
        self.enabled = False
        self.train_calls = 0
        self.val_calls = 0

    def forward(self, x):
        if self.enabled:
            self.train_calls += 1
            if self.always_oom or self.train_calls in self.oom_indices:
                raise _oom()
        else:
            self.val_calls += 1
        return super().forward(x)


def _run(epochs=1, plan=(), n_batches=3, val_batches=2, batch_size=4,
         test_mode=False, logger=None, always_oom=False, **kw):
    """Drive train_model_for_epochs on CPU with a scripted OOM plan.

    ``plan`` lists the *training* batch indices that raise OOM, counted within
    each epoch (0-based). ``always_oom=True`` makes every training batch fail,
    which is the ``real-aa519f5`` shape. The model is only armed while training
    so validation always runs cleanly.
    """
    model = FlakyModel(oom_indices=plan, always_oom=always_oom)
    train_loader = make_classification_batches(n_batches, batch_size)
    val_loader = make_classification_batches(val_batches, batch_size)

    # Arm the model for the training loop and disarm it for evaluate(). The
    # trainer calls the same instance on both paths, so the switch has to be
    # driven from outside; wrapping the loaders would not help because the
    # model, not the data, decides to OOM.
    return train_model_for_epochs(
        model=model,
        epochs=epochs,
        train_loader=_Armed(train_loader, model, armed=True),
        val_loader=_Armed(val_loader, model, armed=False),
        device=torch.device("cpu"),
        logger=logger,
        test_mode=test_mode,
        finetune=False,
        coco=False,
        task="classification",
        checkpoint_every_epoch=False,
        **kw,
    )


class _Armed:
    """Wraps a batch list, arming/disarming the OOM model as it is consumed."""

    def __init__(self, batches, model, armed):
        self._batches = batches
        self._model = model
        self._armed = armed

    def __iter__(self):
        self._model.enabled = self._armed
        self._model.train_calls = 0
        yield from self._batches
        self._model.enabled = False

    def __len__(self):
        return len(self._batches)


@pytest.fixture(autouse=True)
def _default_thresholds(monkeypatch):
    """Pin the thresholds to their module defaults for every test here.

    The suite runs on a machine that may carry these in its environment; a
    stray ``ECS_HCM_*`` would silently retune the limits under test. The
    override tests below set them explicitly, after this fixture.
    """
    for name in ("ECS_HCM_MAX_CONSECUTIVE_OOM", "ECS_HCM_MAX_OOM_FRACTION",
                 "ECS_HCM_MIN_FREE_GPU_GIB"):
        monkeypatch.delenv(name, raising=False)


class TestConsecutiveOomLimit:
    """Requirement 1a: a run that skips too many batches in a row is dead."""

    def test_consecutive_skips_trip_the_limit(self):
        limit = MAX_CONSECUTIVE_OOM_SKIPS
        # One more OOM than the limit allows: the batch at index `limit` is the
        # (limit+1)-th consecutive skip, and trips.
        plan = list(range(1, limit + 2))

        with pytest.raises(RuntimeError) as exc:
            _run(plan=plan, n_batches=limit + 1)

        message = str(exc.value)
        assert str(limit + 1) in message
        assert str(limit) in message
        assert "out of GPU memory" in message

    def test_one_below_the_limit_does_not_trip(self):
        """The limit is a real bound, not an off-by-one that fires early."""
        limit = MAX_CONSECUTIVE_OOM_SKIPS
        # Batches 0..limit-1 OOM; the last batch succeeds, so the run never
        # sees limit+1 skips in a row.
        plan = list(range(1, limit + 1))

        result = _run(plan=plan, n_batches=limit + 1)
        assert result["oom_skipped_total"] == limit

    def test_consecutive_counter_resets_on_a_successful_batch(self, monkeypatch):
        """Interleaved skips under the limit must not accumulate into a trip."""
        limit = MAX_CONSECUTIVE_OOM_SKIPS
        rounds = 50
        # The point under test is the *consecutive* counter, but 20 skips per
        # 21 batches is a 95% skip rate, which the fraction limit would rightly
        # kill. Lift the fraction limit so the consecutive reset is the only
        # thing that can decide this run.
        monkeypatch.setenv("ECS_HCM_MAX_OOM_FRACTION", "1.0")
        # Each round is `limit` skips (1-based: the first forward is 1) then one
        # success, so the run never sees `limit + 1` in a row.
        plan = [
            r * (limit + 1) + k
            for r in range(rounds)
            for k in range(1, limit + 1)
        ]

        result = _run(plan=plan, n_batches=rounds * (limit + 1))
        assert result["oom_skipped_total"] == limit * rounds


class TestOomFractionLimit:
    """Requirement 1b: a run that mostly skips is failing, however it interleaves."""

    def test_fraction_trips_above_the_limit(self):
        # 1 in 10 skipped over 1000 batches is 10%, well above the 1% default.
        plan = list(range(1, 1001, 10))

        with pytest.raises(RuntimeError) as exc:
            _run(plan=plan, n_batches=1000)

        assert "out of GPU memory" in str(exc.value)

    def test_fraction_is_not_judged_below_the_min_batch_floor(self):
        """1 skip out of 21 is ~5%, but it is far too early to call it dead."""
        assert MIN_BATCHES_FOR_OOM_FRACTION == 200
        plan = [1]

        result = _run(plan=plan, n_batches=21)
        assert result["oom_skipped_total"] == 1

    def test_scattered_single_ooms_under_both_limits_do_not_fail(self):
        """Occasional OOMs are survivable and must be counted, not fatal."""
        # 2 skipped in 600 (0.33%) with the rest succeeding, so neither the
        # consecutive limit nor the fraction limit is reached.
        plan = [1, 301]

        result = _run(plan=plan, n_batches=600)
        assert result["oom_skipped_total"] == 2

    def test_fraction_judgement_waits_for_the_floor(self):
        """The same 1-in-4 rate is fatal once enough batches are attempted.

        25% over 8 batches is not judged (below the floor); 25% over 800 is.
        """
        # Below the floor: no trip.
        _run(plan=list(range(1, 9, 4)), n_batches=8)

        with pytest.raises(RuntimeError):
            _run(plan=list(range(1, 801, 4)), n_batches=800)


class TestZeroBatchEpoch:
    """Requirement 1c: an epoch that trains nothing is a failed run."""

    def test_all_oom_epoch_raises(self):
        """The exact incident: every batch OOMs, and the run must not 'finish'."""
        with pytest.raises(RuntimeError) as exc:
            _run(always_oom=True, n_batches=5)

        message = str(exc.value)
        assert "0 batches" in message
        assert "5 attempted" in message

    def test_all_oom_epoch_raises_in_test_mode_too(self):
        """test_mode is not an exemption -- it is how the incident reproduces
        cheaply, so it must fail there as loudly as on a real run."""
        with pytest.raises(RuntimeError) as exc:
            _run(always_oom=True, n_batches=5, test_mode=True)

        assert "0 batches" in str(exc.value)

    def test_failed_epoch_raises_even_when_a_previous_epoch_trained(self):
        """Epoch 2 of 3 trains nothing while epoch 1 was healthy."""
        # `_Armed` resets train_calls each epoch, so a per-epoch plan cannot
        # express "only epoch 2 fails". Count forwards across the whole run
        # instead: the first 3 (epoch 1) succeed, everything after fails.
        train_loader = make_classification_batches(3, 4)
        val_loader = make_classification_batches(2, 4)

        class EpochAware(FlakyModel):
            def __init__(self):
                super().__init__()
                self.total_train_forwards = 0

            def forward(self, x):
                if self.enabled:
                    self.total_train_forwards += 1
                    if self.total_train_forwards > 3:
                        raise _oom()
                return TinyClassifier.forward(self, x)

        model = EpochAware()
        with pytest.raises(RuntimeError) as exc:
            train_model_for_epochs(
                model=model,
                epochs=3,
                train_loader=_Armed(train_loader, model, armed=True),
                val_loader=_Armed(val_loader, model, armed=False),
                device=torch.device("cpu"),
                logger=None,
                test_mode=False,
                finetune=False,
                coco=False,
                task="classification",
                checkpoint_every_epoch=False,
            )

        assert "epoch 2" in str(exc.value)


class TestOomMetricsAreEmitted:
    """Requirement 2: the counts must survive a run that logged nothing else."""

    def test_metrics_are_logged_when_no_batch_trained(self):
        """The direct regression: batch_count == 0 and the evidence still lands.

        Before this change every OOM metric sat inside ``if batch_count > 0:``,
        the one block a zero-batch epoch never reaches -- so the run that
        skipped 14,664 batches reported nothing at all.
        """
        logger = RecordingLogger()

        with pytest.raises(RuntimeError):
            _run(always_oom=True, n_batches=5, logger=logger)

        assert "batch_oom_skipped" in logger.scalar_keys()
        assert "train_oom_skipped" in logger.scalar_keys()
        assert logger.scalar_values()["train_oom_skipped"] == 5.0

    def test_first_skip_is_logged_immediately(self):
        """A stuck run is visible from MLflow without waiting for the cadence."""
        logger = RecordingLogger()

        with pytest.raises(RuntimeError):
            _run(always_oom=True, n_batches=5, logger=logger)

        steps = [step for key, _, step in logger.scalars if key == "batch_oom_skipped"]
        assert steps[0] == 0, "the very first skip must be logged at once"

    def test_healthy_epoch_still_logs_a_zero(self):
        """A healthy epoch emits the same key a failed one does, valued 0."""
        logger = RecordingLogger()

        result = _run(plan=[], n_batches=3, logger=logger)

        assert result["oom_skipped_total"] == 0
        assert logger.scalar_values()["train_oom_skipped"] == 0.0

    def test_run_total_is_returned_for_the_caller_to_tag(self):
        result = _run(plan=[1], n_batches=3)
        assert result["oom_skipped_total"] == 1

    def test_metrics_are_logged_on_the_oom_trip_path(self):
        """A tripping run flushes its count before it raises.

        The consecutive limit trips at 21 skips, well short of the 200-batch
        cadence, so the only flush before the raise is the immediate first-skip
        one. That is exactly the guarantee worth pinning: a run that dies on
        OOM has already told MLflow it was skipping.
        """
        logger = RecordingLogger()
        plan = list(range(1, 101))

        with pytest.raises(RuntimeError):
            _run(plan=plan, n_batches=200, logger=logger)

        skips = [v for k, v, _ in logger.scalars if k == "batch_oom_skipped"]
        assert skips, "a tripping run must have logged its skips before raising"
        assert skips[0] == 1.0

    def test_oom_metrics_are_flushed_at_the_periodic_cadence(self, monkeypatch):
        """A slow-burning run flushes periodically, not only on the first skip.

        Both guards are lifted here so the run survives past 200 skips and the
        cadence -- not a trip -- is what the assertion sees. At the defaults a
        run with a 50% rate trips the fraction guard at skip #2, and even a 1%
        rate trips it at skip #2 of 200, so the cadence would never be reached
        and the flush could rot unnoticed.
        """
        logger = RecordingLogger()
        monkeypatch.setenv("ECS_HCM_MAX_CONSECUTIVE_OOM", "100000")
        monkeypatch.setenv("ECS_HCM_MAX_OOM_FRACTION", "1.0")
        # Odd forwards skip, even ones succeed: a 50% rate that now trips
        # nothing, so the run reaches 200 skips and flushes at the cadence.
        plan = list(range(1, 401, 2))

        _run(plan=plan, n_batches=400, epochs=1, logger=logger)

        skips = [v for k, v, _ in logger.scalars if k == "batch_oom_skipped"]
        assert skips[0] == 1.0, f"first-skip flush missing: {skips}"
        assert 200.0 in skips, f"no cadence flush in {skips}"


class TestEvaluateNeverSkips:
    """Requirement 3: validation reports, or it fails. It never averages a subset."""

    def test_validation_oom_raises_naming_the_batch(self):
        class ValOom(TinyClassifier):
            def forward(self, x):
                raise _oom()

        loader = make_classification_batches(3, 4)
        with pytest.raises(RuntimeError) as exc:
            evaluate(ValOom(), loader, torch.device("cpu"), is_segmentation=False)

        assert "validation OOM at batch 0" in str(exc.value)

    def test_validation_oom_is_raised_not_skipped_after_a_good_batch(self):
        """Batch 1 fails: the batch index must say 1, not be swallowed."""
        class ValOomAtOne(TinyClassifier):
            def __init__(self):
                super().__init__()
                self.calls = 0

            def forward(self, x):
                self.calls += 1
                if self.calls == 2:
                    raise _oom()
                return super().forward(x)

        loader = make_classification_batches(3, 4)
        with pytest.raises(RuntimeError) as exc:
            evaluate(ValOomAtOne(), loader, torch.device("cpu"), is_segmentation=False)

        assert "validation OOM at batch 1" in str(exc.value)

    def test_empty_validation_raises(self):
        """0 validated batches must not average to a 'perfect' 0.0."""
        with pytest.raises(RuntimeError) as exc:
            evaluate(TinyClassifier(), [], torch.device("cpu"), is_segmentation=False)

        assert "0 batches" in str(exc.value)

    def test_partial_validation_returns_normally(self):
        """A short validation is not an error -- only an empty one is."""
        loader = make_classification_batches(3, 4)
        metrics = evaluate_limited(TinyClassifier(), loader, torch.device("cpu"))

        assert "val_loss" in metrics
        assert metrics["val_loss"] > 0.0


def evaluate_limited(model, loader, device, max_batches=2):
    """evaluate() capped short of the whole loader, for the partial case."""
    return evaluate(model, loader, device, max_batches=max_batches,
                    is_segmentation=False)


class TestGpuPreflight:
    """Requirement 4: refuse an under-resourced GPU before the run starts."""

    def _prepare(self, monkeypatch, free_gib, total_gib=16.0):
        """Wire train_hcm_imagenet1k onto a fake CUDA box and record the call.

        This torch is built without CUDA, so a genuine ``torch.device("cuda:0")``
        cannot be used for anything real. The preflight runs on the *device
        check* before any tensor moves, so the test stops the run at the model
        load -- the first thing after the preflight -- and asserts on whether it
        got that far. Nothing CUDA is ever actually touched.
        """
        from tests.test_mlflow_lifecycle import _patch_trainer

        logger = RecordingLogger()
        _patch_trainer(monkeypatch, logger)
        monkeypatch.setattr(trainer.torch.cuda, "is_available", lambda: True)
        monkeypatch.setattr(trainer.torch.cuda, "set_device", lambda d: None)
        monkeypatch.setattr(trainer.torch.cuda, "empty_cache", lambda: None)
        monkeypatch.setattr(
            trainer.torch.cuda, "get_device_name", lambda d: "Fake V100"
        )

        seen = {}

        def _mem_get_info(index=None):
            seen["called"] = True
            return (int(free_gib * 1024 ** 3), int(total_gib * 1024 ** 3))

        monkeypatch.setattr(trainer.torch.cuda, "mem_get_info", _mem_get_info)

        def _reached_past_the_preflight(**kw):
            seen["trained"] = True
            raise AssertionError("reached training")

        monkeypatch.setattr(
            trainer, "load_hcm_model", _reached_past_the_preflight
        )
        return logger, seen

    def test_preflight_trips_below_the_threshold(self, monkeypatch):
        logger, seen = self._prepare(monkeypatch, free_gib=4.0)

        with pytest.raises(RuntimeError) as exc:
            trainer.train_hcm_imagenet1k(environment="development", max_epochs=1)

        message = str(exc.value)
        assert "4.0 GiB free" in message
        assert "16.0 GiB" in message
        assert "8.0 GiB is required" in message
        assert "another process" in message.lower()
        assert seen.get("called"), "the preflight must actually ask the driver"
        assert "trained" not in seen, "the run must not get past the preflight"
        assert "FAILED" in logger.finish_statuses()

    def test_preflight_passes_above_the_threshold(self, monkeypatch):
        logger, seen = self._prepare(monkeypatch, free_gib=12.0)

        # The stub raises, so the run reaching it is what "the preflight let it
        # through" means.
        with pytest.raises(AssertionError, match="reached training"):
            trainer.train_hcm_imagenet1k(environment="development", max_epochs=1)

        assert seen.get("trained"), "the preflight wrongly blocked a healthy GPU"

    def test_preflight_reads_free_first(self, monkeypatch):
        """mem_get_info returns (free, total) -- the order decides everything.

        A swap here would compare total against the threshold and pass a fully
        occupied GPU, so the direction is asserted rather than assumed.
        """
        logger, seen = self._prepare(monkeypatch, free_gib=1.0, total_gib=16.0)

        with pytest.raises(RuntimeError) as exc:
            trainer.train_hcm_imagenet1k(environment="development", max_epochs=1)

        assert "1.0 GiB free of 16.0 GiB" in str(exc.value)

    def test_cpu_box_is_not_touched_by_the_preflight(self, monkeypatch):
        """No CUDA means no mem_get_info call and no chance of a false refusal."""
        from tests.test_mlflow_lifecycle import _patch_trainer

        logger = RecordingLogger()
        _patch_trainer(monkeypatch, logger)
        monkeypatch.setattr(trainer.torch.cuda, "is_available", lambda: False)

        def _explode(index=None):
            raise AssertionError("mem_get_info must not be called on a CPU box")

        monkeypatch.setattr(trainer.torch.cuda, "mem_get_info", _explode)
        monkeypatch.setattr(
            trainer, "load_hcm_model",
            lambda **kw: (_ for _ in ()).throw(AssertionError("reached training")),
        )

        with pytest.raises(AssertionError, match="reached training"):
            trainer.train_hcm_imagenet1k(environment="development", max_epochs=1)


class TestThresholdOverrides:
    """The thresholds are operator-tunable from the environment."""

    def test_consecutive_limit_override(self, monkeypatch):
        monkeypatch.setenv("ECS_HCM_MAX_CONSECUTIVE_OOM", "2")

        # 3 in a row trips a limit of 2, where the default of 20 would not.
        with pytest.raises(RuntimeError):
            _run(plan=[1, 2, 3], n_batches=3)

    def test_fraction_override(self, monkeypatch):
        monkeypatch.setenv("ECS_HCM_MAX_OOM_FRACTION", "0.0")

        # Any skip after the floor now exceeds a 0% allowance.
        plan = [1]
        with pytest.raises(RuntimeError):
            _run(plan=plan, n_batches=MIN_BATCHES_FOR_OOM_FRACTION)

    def test_min_free_gpu_override(self, monkeypatch):
        from tests.test_mlflow_lifecycle import _patch_trainer

        logger = RecordingLogger()
        _patch_trainer(monkeypatch, logger)
        monkeypatch.setattr(trainer.torch.cuda, "is_available", lambda: True)
        monkeypatch.setattr(trainer.torch.cuda, "set_device", lambda d: None)
        monkeypatch.setattr(trainer.torch.cuda, "empty_cache", lambda: None)
        monkeypatch.setattr(trainer.torch.cuda, "get_device_name", lambda d: "Fake")
        monkeypatch.setattr(
            trainer.torch.cuda, "mem_get_info",
            lambda index=None: (6 * 1024 ** 3, 16 * 1024 ** 3),
        )
        monkeypatch.setattr(
            trainer, "load_hcm_model",
            lambda **kw: (_ for _ in ()).throw(AssertionError("reached training")),
        )
        # 6 GiB free fails the 8 GiB default but passes a lowered 4 GiB bar.
        monkeypatch.setenv("ECS_HCM_MIN_FREE_GPU_GIB", "4")

        with pytest.raises(AssertionError, match="reached training"):
            trainer.train_hcm_imagenet1k(environment="development", max_epochs=1)

    def test_unparseable_override_falls_back_to_the_default(self, monkeypatch):
        """A typo'd env var must not silently disable the guard."""
        monkeypatch.setenv("ECS_HCM_MAX_CONSECUTIVE_OOM", "not-a-number")

        assert trainer._env_int(
            "ECS_HCM_MAX_CONSECUTIVE_OOM", MAX_CONSECUTIVE_OOM_SKIPS
        ) == MAX_CONSECUTIVE_OOM_SKIPS

    def test_thresholds_are_read_per_call_not_at_import(self, monkeypatch):
        """The env is consulted on each run, so an operator can retune live."""
        monkeypatch.setenv("ECS_HCM_MAX_CONSECUTIVE_OOM", "2")
        assert trainer._oom_limits()[0] == 2

        monkeypatch.setenv("ECS_HCM_MAX_CONSECUTIVE_OOM", "7")
        assert trainer._oom_limits()[0] == 7
