"""The fp16 overflow that killed the real run, and the fix for it.

A run died at epoch 1, batch ~13.7k with `Non-finite detected in isc_out[0]`:
an fp16 tensor holding a `-inf`. The cause was `hcm_isc.forward` computing in
fp32 and then returning `output.to(dtype)` -- casting to the fp16 *input* dtype.
`output` is unbounded after its group_norm, so past |65504| the cast itself
produces inf, silently. `_check("output", output)` could not see it because it
ran on the fp32 value, one statement before the cast that killed it.

These tests are CPU-only, offline, and construct no pretrained weights.
"""

import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# The autocast context these tests use to drive real fp16 behaviour. Defined at
# module scope, before any class that references it at call time: the CUDA-API
# variant is inert on a CUDA-less box and would turn those assertions into
# no-ops, so every autocast-sensitive test here goes through this instead.
LIVE_AUTOCAST = ("cpu", torch.float16)


def _cpu_fp16_autocast_supported() -> bool:
    """Can this torch actually run CPU fp16 autocast?

    It cannot everywhere, and the difference is the interpreter, not the box.
    The dev venv is torch 2.14, where ``torch.autocast("cpu", dtype=float16)``
    works. The trainer image pins torch 2.1.0, where *entering* that context
    raises ``RuntimeError: AutocastCPU only support Bfloat16``. Nothing in this
    module is wrong on 2.1.0 -- the capability simply is not there, so the
    autocast-driven assertions cannot mean anything and the tests must skip
    rather than fail.

    Probed once, at import, by entering the context AND running an op inside it.
    A version check would be a guess about which release grew the feature, and
    the thing that matters is the behaviour. Running an op matters too: if the
    unsupported dtype were rejected at the first dispatch rather than at context
    entry, a probe with an empty body would report "supported" and the tests
    would fail exactly as before.

    The exception type is deliberately broad -- this can fail from unrelated
    machinery, and any such failure is an equally good reason not to run these
    tests.
    """
    try:
        with torch.autocast(device_type=LIVE_AUTOCAST[0], dtype=LIVE_AUTOCAST[1]):
            nn.Conv2d(2, 2, 1)(torch.zeros(1, 2, 2, 2))
    except Exception:
        return False
    return True


CPU_FP16_AUTOCAST = _cpu_fp16_autocast_supported()

requires_cpu_fp16_autocast = pytest.mark.skipif(
    not CPU_FP16_AUTOCAST,
    reason=(
        f"this torch ({torch.__version__}) cannot run CPU fp16 autocast; "
        f"entering torch.autocast('cpu', dtype=torch.float16) raises. The "
        f"trainer image pins torch 2.1.0, which behaves this way. These tests "
        f"exist to exercise live fp16 autocast, so without it they would be "
        f"comparing no-autocast paths and proving nothing."
    ),
)

from ecs_hcm_model.core import hcm_nn  # noqa: E402
from ecs_hcm_model.core.hcm_nn import (  # noqa: E402
    FP16_MAX_SAFE,
    RESNET_OUTPUT_CHANNEL_NUMBER,
    _guard_fp16_boundary,
    hcm_isc,
)


def _isc_pushed_past_fp16(scale=100.0, channels=RESNET_OUTPUT_CHANNEL_NUMBER):
    """An hcm_isc whose output deliberately exceeds the fp16 ceiling.

    The 1x1 branch (`conv1_1_1x1`) feeds `output` directly and carries no clamp,
    so scaling *its* weights past the fp16 ceiling while zeroing the rest gives a
    deterministic fp32 output far above 65504 -- with an input that is itself
    perfectly finite in fp16, which is the shape of the real failure (the input
    was fine; the *output* was what overflowed).
    """
    torch.manual_seed(0)
    isc = hcm_isc()
    with torch.no_grad():
        isc.conv1_1_1x1.weight.fill_(1.0)
        isc.conv1_1_1x1.bias.fill_(0.0)
        for module in (
            isc.conv1_2_1x1, isc.conv1_3_1x1, isc.conv2_1_3x3, isc.conv2_2_5x5,
            isc.conv3_1_3x3, isc.conv3_2_5x5, isc.conv4_1_3x3,
        ):
            module.weight.zero_()
            module.bias.zero_()
    x = torch.full((1, channels, 4, 4), scale).half()
    return isc, x


def _old_behaviour(isc, x):
    """The removed `return output.to(dtype)`, for the fail-before comparison."""
    out = isc(x)
    return out.to(torch.float16)


class TestIscNoLongerCastsToFp16:
    """(a) the overflow, and that the fix removes it."""

    def test_isc_output_is_fp32_and_finite_past_the_fp16_ceiling(self):
        """The input is fp16-safe, the fp32 output is huge, and nothing is inf."""
        isc, x = _isc_pushed_past_fp16()
        assert torch.isfinite(x.float()).all(), "the input itself must be fp16-safe"

        out = isc(x)

        assert out.dtype == torch.float32, (
            "hcm_isc must return fp32: casting back to the fp16 input dtype is "
            "what produced the -inf the run died on"
        )
        assert torch.isfinite(out).all(), (
            "the fp32 output must be finite even where an fp16 cast would not be"
        )
        assert out.abs().amax().item() > 65504, (
            "this fixture must actually exceed the fp16 ceiling, or it proves "
            "nothing about the overflow"
        )

    def test_the_removed_cast_would_have_produced_inf(self):
        """Fail-before evidence: the old trailing cast saturates to inf.

        This reproduces the original defect on the same tensor the new code
        returns finite, which is why the fix is a genuine behaviour change and
        not a no-op.
        """
        isc, x = _isc_pushed_past_fp16()

        old = _old_behaviour(isc, x)

        assert not torch.isfinite(old).all(), (
            "the old `.to(float16)` was expected to saturate and produce inf"
        )
        assert torch.isinf(old).any(), "the old cast should yield +/-inf entries"


class TestIscToCscStaysFp32:
    """(b) the half-fix: dropping the cast alone leaves the CSC stage exposed.

    CORRECTION (this originally claimed something false, and the claim is left
    visible rather than quietly deleted): it used to say that an fp32 tensor
    under autocast "comes back fp32 too" with the input "silently downcast".
    That observation came from `torch.cuda.amp.autocast`, which is INERT on a
    CUDA-less box -- it was comparing two identical no-autocast paths. Measured
    against device autocast that actually engages, autocast casts the conv's
    OUTPUT to fp16 (the input arrives fp32), so the returned dtype DOES
    discriminate. The fix is still correct; the reasoning below is what is
    corrected.
    """

    @requires_cpu_fp16_autocast
    def test_csc_conv_inside_no_autocast_keeps_fp32(self):
        """fp32 in, fp32 out -- with a live autocast block switched off."""
        conv = nn.Conv2d(4, 4, 1)
        x = torch.full((1, 4, 2, 2), 70000.0)

        with torch.autocast(device_type=LIVE_AUTOCAST[0], dtype=LIVE_AUTOCAST[1]):
            with torch.autocast(device_type=LIVE_AUTOCAST[0], enabled=False):
                out = conv.float()(x)
        assert out.dtype == torch.float32
        assert torch.isfinite(out).all()

        # And the same op without the inner block: fp16, non-finite. This is
        # what the CSC stage would have inherited had it merely been handed
        # fp32 tensors while still sitting under the caller's autocast.
        with torch.autocast(device_type=LIVE_AUTOCAST[0], dtype=LIVE_AUTOCAST[1]):
            out_ac = conv(x)
        assert out_ac.dtype == torch.float16, (
            "without the guard block the conv returns fp16; the block placement "
            "is what keeps the CSC stage in fp32"
        )
        assert not torch.isfinite(out_ac.float()).all()

    @requires_cpu_fp16_autocast
    def test_csc_forward_preserves_fp32_when_run_outside_autocast(self):
        """The real hcm_csc: fp32 in, fp32 out, finite, with autocast off."""
        from ecs_hcm_model.core.hcm_nn import hcm_csc

        torch.manual_seed(0)
        csc = hcm_csc()
        f_s = torch.randn(1, 32, 8, 8) * 1e3
        f_next = torch.randn(1, 32, 4, 4) * 1e3

        with torch.autocast(device_type=LIVE_AUTOCAST[0], dtype=LIVE_AUTOCAST[1]):
            with torch.autocast(device_type=LIVE_AUTOCAST[0], enabled=False):
                out = csc(f_s.float(), f_next.float())

        assert out.dtype == torch.float32, (
            "hcm_csc must stay fp32 when the surrounding block has autocast off"
        )
        assert torch.isfinite(out).all()


class TestFp16BoundaryGuard:
    """(c) one guarded boundary, raising with the tensor name and its max."""

    def test_passes_just_below_the_threshold(self):
        t = torch.full((1, 4, 2, 2), FP16_MAX_SAFE - 1.0)
        assert _guard_fp16_boundary("rrd_out[0]", t) is t

    def test_raises_just_above_the_threshold_naming_the_tensor_and_max(self):
        t = torch.full((1, 4, 2, 2), FP16_MAX_SAFE + 1.0)
        with pytest.raises(RuntimeError) as caught:
            _guard_fp16_boundary("rrd_out[0]", t)
        message = str(caught.value)
        assert "rrd_out[0]" in message, "the error must name the offending tensor"
        assert "60001" in message or "6.0001e+04" in message, (
            f"the error must report the observed max; got: {message}"
        )

    def test_raises_on_a_non_finite_value(self):
        t = torch.tensor([[[[float("inf")]]]])
        with pytest.raises(RuntimeError):
            _guard_fp16_boundary("rrd_out[0]", t)

    def test_messages_carry_an_int_tensor_without_raising(self):
        t = torch.zeros((1, 4, 2, 2), dtype=torch.int64)
        assert _guard_fp16_boundary("int_tensor", t) is t


class TestActivationMonitoring:
    """(3) the running absmax the trainer logs."""

    def test_running_max_accumulates_and_resets(self):
        """The running max folds in the top stage, and the trainer can read it.

        The state lives on the whole model, not on the individual submodule --
        the trainer holds one `hcm_nn` and needs one number per stage, so
        `_record_absmax` takes the owning module as its first argument. Here a
        bare submodule stands in for that owner; the real call sites pass
        `self` from `forward_internal`.
        """
        owner = hcm_isc()

        hcm_nn._record_absmax(owner, "isc", torch.full((1, 4, 2, 2), 12.0))
        assert owner.activation_absmax["isc"] == pytest.approx(12.0)

        # A smaller value must not lower the running max.
        hcm_nn._record_absmax(owner, "isc", torch.full((1, 4, 2, 2), 3.0))
        assert owner.activation_absmax["isc"] == pytest.approx(12.0)

        # A larger one must raise it.
        hcm_nn._record_absmax(owner, "isc", torch.full((1, 4, 2, 2), 40.0))
        assert owner.activation_absmax["isc"] == pytest.approx(40.0)

        # Stages stay independent.
        hcm_nn._record_absmax(owner, "csc", torch.full((1, 4, 2, 2), 7.0))
        assert owner.activation_absmax["csc"] == pytest.approx(7.0)
        assert owner.activation_absmax["isc"] == pytest.approx(40.0)

    def test_hcm_nn_exposes_reset(self):
        model = hcm_nn.hcm_nn.__new__(hcm_nn.hcm_nn)
        model.activation_absmax = {s: 5.0 for s in hcm_nn.ACTIVATION_STAGES}
        model.reset_activation_absmax()
        assert all(v == 0.0 for v in model.activation_absmax.values())

    def test_stage_magnitudes_are_not_percentage_guarded(self):
        """A 60000 magnitude must not trip the 0-100 guard on the way out.

        The activation metrics ride `log_scalar`, which skips the percentage
        check entirely; routing them through the data_obj path would have made
        a value of 60000 raise as an out-of-range percentage and lose the
        measurement.
        """
        from ecs_hcm_model.core.model_logger import (
            PERCENT_METRICS, _check_percentage,
        )

        for stage in hcm_nn.ACTIVATION_STAGES:
            for prefix in ("batch", "train"):
                key = f"{prefix}_{stage}_absmax"
                assert key not in PERCENT_METRICS

        # And the guard really would have rejected it, which is the hazard.
        with pytest.raises(Exception):
            _check_percentage("train_isc_absmax", 60000.0)


class TestCheckpointShapeUnchanged:
    """(d) the fix must not change the parameter shapes the run resumes into."""

    @staticmethod
    def _offline_hcm(**kwargs):
        """Build a real hcm_nn with the pretrained fetch defanged.

        The real constructor reaches torchvision for ResNet-50 ImageNet weights,
        which needs network. Stubbing `models.resnet50` to build an untrained
        model of the same shape keeps every real module in the tree -- which is
        the thing under test -- while making the test offline and fast.
        """
        from torchvision import models as tv_models

        real_resnet50 = tv_models.resnet50

        def offline_resnet50(weights=None, replace_stride_with_dilation=None):
            return real_resnet50(
                weights=None,
                replace_stride_with_dilation=replace_stride_with_dilation,
            )

        saved = tv_models.resnet50
        tv_models.resnet50 = offline_resnet50
        try:
            return hcm_nn.hcm_nn(**kwargs)
        finally:
            tv_models.resnet50 = saved

    def test_state_dict_loads_strict_true(self):
        """A model's own state_dict round-trips with strict=True.

        This is the property the resuming run depends on: the Jan-5 checkpoint
        must load with no missing and no unexpected keys. If the fix had changed
        a parameter shape -- which the fp32/CSC rework could plausibly have done
        -- this fails with the offending key named.
        """
        model = self._offline_hcm(enable_finetuning=True, num_segmentation_classes=2)
        state = model.state_dict()

        # The load the resuming run performs, in miniature.
        fresh = self._offline_hcm(enable_finetuning=True, num_segmentation_classes=2)
        missing, unexpected = fresh.load_state_dict(state, strict=True)

        assert missing == [] and unexpected == []

    def test_no_activation_state_leaks_into_the_checkpoint(self):
        """Monitoring state must stay out of the checkpoint.

        `activation_absmax` is a plain dict, not a buffer, so it cannot appear in
        a state_dict -- if it did, the Jan-5 checkpoint would fail to load
        strict=True for an unrelated reason and the run would refuse to resume.
        """
        model = self._offline_hcm(enable_finetuning=True, num_segmentation_classes=2)
        model.activation_absmax["isc"] = 12345.0

        state = model.state_dict()

        assert not any("activation_absmax" in key for key in state)
        fresh = self._offline_hcm(enable_finetuning=True, num_segmentation_classes=2)
        assert fresh.activation_absmax == {"isc": 0.0, "csc": 0.0, "isc3": 0.0}


class TestActivationMetricCadence:
    """(e) the trainer logs the right keys at the right cadence.

    Fake logger, fake model, no MLflow -- this pins the seam between the model's
    running maxima and what reaches the metric store.
    """

    @staticmethod
    def _patch(monkeypatch, logger, batches):
        import ecs_hcm_model.core.trainer as trainer

        monkeypatch.setattr(trainer, "set_env_vars", lambda *a, **k: None)
        monkeypatch.setattr(
            trainer, "get_training_dataloader", lambda **kw: batches
        )
        monkeypatch.setattr(
            trainer, "get_validation_dataloader",
            lambda **kw: [(torch.randn(4, 8), torch.randint(0, 10, (4,)))],
        )
        monkeypatch.setattr(
            trainer, "ModelUploaderAndDownloader",
            lambda *a, **k: FakeUploader(),
        )
        logger.set_logger = lambda config=None: None
        monkeypatch.setattr(trainer, "logger_config", object())
        monkeypatch.setattr(trainer, "MLFlowHCMLogger", lambda *a, **k: logger)
        return trainer

    def test_batch_keys_appear_at_the_nth_batch_and_epoch_keys_once(
        self, monkeypatch, logger
    ):
        from tests.conftest import TinyClassifier

        batches = [
            (torch.randn(4, 8), torch.randint(0, 10, (4,))) for _ in range(420)
        ]
        trainer = self._patch(monkeypatch, logger, batches)

        model = TinyClassifier()
        trainer.train_model_for_epochs(
            model=model,
            epochs=1,
            train_loader=batches,
            # A non-empty validation loader: evaluate() now raises on an
            # empty one ("ran 0 batches"), which is the point of the guard.
            val_loader=[(torch.randn(4, 8), torch.randint(0, 10, (4,)))],
            device=torch.device("cpu"),
            logger=logger,
            test_mode=True,
            max_batches_per_epoch=420,
            warmup=False,
            task="classification",
        )

        batch_keys = [k for k in logger.scalar_keys() if k.startswith("batch_")]
        assert "batch_isc_absmax" in batch_keys
        assert "batch_csc_absmax" in batch_keys

        # 420 batches at ACTIVATION_LOG_EVERY=200 -> two flushes (batch 200, 400).
        # Asserting the count pins the cadence, not merely that a key appeared.
        isc_batch_steps = [
            step for k, _, step in logger.scalars if k == "batch_isc_absmax"
        ]
        assert isc_batch_steps == [200, 400], (
            f"expected flushes at batches 200 and 400, got {isc_batch_steps}"
        )

    def test_magnitudes_above_one_hundred_are_recorded_not_rejected(
        self, monkeypatch, logger
    ):
        """A 60000 magnitude must reach the log, not raise as a percentage."""
        from tests.conftest import TinyClassifier

        batches = [
            (torch.randn(4, 8), torch.randint(0, 10, (4,))) for _ in range(250)
        ]
        trainer = self._patch(monkeypatch, logger, batches)

        model = TinyClassifier()
        # Make the model report a huge activation: this is the value the guard
        # would have choked on had it gone through the percentage path.
        original_forward = model.forward

        def big_forward(x):
            out = original_forward(x)
            model.activation_absmax["isc"] = 60000.0
            model.activation_absmax["csc"] = 59000.0
            return out

        model.forward = big_forward

        trainer.train_model_for_epochs(
            model=model,
            epochs=1,
            train_loader=batches,
            # A non-empty validation loader: evaluate() now raises on an
            # empty one ("ran 0 batches"), which is the point of the guard.
            val_loader=[(torch.randn(4, 8), torch.randint(0, 10, (4,)))],
            device=torch.device("cpu"),
            logger=logger,
            test_mode=True,
            max_batches_per_epoch=250,
            warmup=False,
            task="classification",
        )

        values = logger.scalar_values()
        assert values.get("batch_isc_absmax") == pytest.approx(60000.0)
        assert values.get("train_csc_absmax") == pytest.approx(59000.0)


class FakeUploader:
    def __init__(self, *a, **k):
        pass

    def upload_model(self, *a, **k):
        return "models/fake/model.pth"


class TestMaxBatchesPlumbing:
    """(f) --max_batches_per_epoch reaches the loop cap from the CLI."""

    def test_flag_is_parsed_from_argv(self):
        from main import setup_subpparsers

        parser = setup_subpparsers()
        args = parser.parse_args(
            ["train_hcm_imagenet1k", "--max_batches_per_epoch", "50"]
        )
        assert args.max_batches_per_epoch == 50

    def test_default_is_zero_meaning_uncapped(self):
        from main import setup_subpparsers

        parser = setup_subpparsers()
        args = parser.parse_args(["train_hcm_imagenet1k"])
        assert args.max_batches_per_epoch == 0

    def test_flag_reaches_the_service_call(self, monkeypatch):
        import main

        captured = {}

        def fake_run(**kwargs):
            captured.update(kwargs)

        monkeypatch.setattr(main, "run_train_hcm_imagenet1k", fake_run)
        parser = main.setup_subpparsers()
        args = parser.parse_args(
            ["train_hcm_imagenet1k", "--max_batches_per_epoch", "7"]
        )
        args.func(args)

        assert captured["max_batches_per_epoch"] == 7

    def test_a_cap_without_test_mode_still_truncates_the_epoch(
        self, monkeypatch
    ):
        from tests.conftest import RecordingLogger
        """The bug the plumbing exposed: the break used to require test_mode.

        A cap passed *without* test_mode was silently ignored, so a run asked to
        stop after N batches ran the whole epoch.
        """
        logger = RecordingLogger()
        batches = [
            (torch.randn(4, 8), torch.randint(0, 10, (4,))) for _ in range(30)
        ]
        trainer = TestActivationMetricCadence._patch(monkeypatch, logger, batches)
        from tests.conftest import TinyClassifier

        seen = []
        seen_val = []
        original_forward = TinyClassifier.forward

        def counting_forward(self, x):
            # The cap bounds *training* batches; validation still runs, and its
            # forwards must not be counted against the cap (a shared counter
            # would have read as 6 here).
            (seen if self.training else seen_val).append(1)
            return original_forward(self, x)

        monkeypatch.setattr(TinyClassifier, "forward", counting_forward)

        trainer.train_model_for_epochs(
            model=TinyClassifier(),
            epochs=1,
            train_loader=batches,
            # A non-empty validation loader: evaluate() now raises on an
            # empty one ("ran 0 batches"), which is the point of the guard.
            val_loader=[(torch.randn(4, 8), torch.randint(0, 10, (4,)))],
            device=torch.device("cpu"),
            logger=logger,
            test_mode=False,
            max_batches_per_epoch=5,
            warmup=False,
            task="classification",
        )

        assert len(seen_val) > 0, "the validation loader must still be evaluated"
        assert len(seen) == 5, (
            f"a cap of 5 without test_mode must stop at 5 batches, ran {len(seen)}"
        )


# ---------------------------------------------------------------------------
# The rrd_list[3] exposure: the last raw ISC tensor that reached fp16 code.
#
# Autocast semantics on this box, established empirically (see the queue item
# for the full trace), because the naive reading is wrong in a way that makes
# tests look meaningful when they are not:
#
#   * torch.cuda.amp.autocast() SILENTLY DISABLES ITSELF with no CUDA. A test
#     that wraps code in it compares two identical no-autocast paths and proves
#     nothing. These tests therefore drive torch.autocast(device_type="cpu",
#     dtype=torch.float16), which does engage here, and check that with
#     torch.is_autocast_cpu_enabled() so a future box cannot quietly turn them
#     back into no-ops.
#   * Autocast casts the conv's OUTPUT, not its input. An instrumented
#     F.conv2d under live autocast receives fp32 (absmax 100000.0) and returns
#     fp16 -- non-finite. So the range loss happens inside the op.
#   * Consequently the observable difference between guarded and unguarded is
#     the OUTPUT DTYPE. Finiteness alone does not discriminate, and
#     _guard_fp16_boundary cannot detect this class of failure at all: it runs
#     on the returned tensor, after the overflow has already happened inside.
# ---------------------------------------------------------------------------


class TestAutocastIsActuallyEngaged:
    """Guards the tests below from silently becoming no-ops."""

    @requires_cpu_fp16_autocast
    def test_device_autocast_engages_on_this_box(self):
        """If this fails the rrd tests below prove nothing; do not delete it."""
        with torch.autocast(device_type=LIVE_AUTOCAST[0], dtype=LIVE_AUTOCAST[1]):
            assert torch.is_autocast_cpu_enabled(), (
                "device-type autocast did not engage, so the rrd dtype/finiteness "
                "assertions are not exercising autocast at all"
            )

    def test_the_cuda_amp_api_does_not_engage_here(self):
        """Documents the trap: cuda.amp autocast is inert without CUDA.

        Pinned because cc8ff5f's tests were written against this API and
        therefore compared two identical no-autocast paths.
        """
        from torch.cuda.amp import autocast as cuda_autocast

        with cuda_autocast():
            assert not torch.is_autocast_enabled(), (
                "cuda.amp autocast unexpectedly engaged without CUDA; the "
                "reasoning in the cc8ff5f test comments would need revisiting"
            )


class TestAutocastOverflowsTheConv:
    """The op-level mechanism, on the exact conv shape rrd_list[3] uses."""

    @staticmethod
    def _conv():
        from ecs_hcm_model.core.hcm_nn import ISC_OUTPUT_CHANNEL_NUM

        conv = nn.Conv2d(
            ISC_OUTPUT_CHANNEL_NUM, ISC_OUTPUT_CHANNEL_NUM,
            kernel_size=3, padding=1, stride=(1, 2),
        )
        # Identity-ish gain so the magnitude is not attenuated by init.
        with torch.no_grad():
            conv.weight.fill_(0.0)
            c = ISC_OUTPUT_CHANNEL_NUM
            for k in range(c):
                conv.weight[k, k, 1, 1] = 1.0
            conv.bias.zero_()
        return conv

    @requires_cpu_fp16_autocast
    def test_above_fp16_goes_non_finite_under_live_autocast(self):
        conv = self._conv()
        x = torch.full((1, 32, 8, 8), 1.0e5)   # > 65504

        with torch.autocast(device_type=LIVE_AUTOCAST[0], dtype=LIVE_AUTOCAST[1]):
            out = conv(x)

        assert out.dtype == torch.float16
        assert not torch.isfinite(out.float()).all(), (
            "a magnitude above the fp16 ceiling must not come back finite from "
            "a conv running under live autocast"
        )

    @requires_cpu_fp16_autocast
    def test_same_op_inside_the_guard_block_stays_fp32_and_finite(self):
        conv = self._conv()
        x = torch.full((1, 32, 8, 8), 1.0e5)

        with torch.autocast(device_type=LIVE_AUTOCAST[0], dtype=LIVE_AUTOCAST[1]):
            with torch.autocast(device_type=LIVE_AUTOCAST[0], enabled=False):
                out = conv(x)

        assert out.dtype == torch.float32
        assert torch.isfinite(out).all()


class TestRrdListThreeRunsInFp32:
    """The rrd_list[3] guard, to the extent this box can actually verify it.

    This class originally held two dtype assertions that looked like the
    fail-before/pass-after pair for the fix. They were worthless, and the reason
    is worth keeping:

        with autocast(enabled=False):
            out = rrd(f_s, p1)          # <-- the TEST supplies the guard

    A test that wraps the call in its own ``enabled=False`` re-implements the
    very edit under test, so it passed with and without the production change.
    It proved nothing about ``forward_internal``. What follows asserts only
    claims this environment can actually support.

    The verification limit, measured three ways on this CUDA-less box:

        outer cuda.amp + inner cuda.amp(enabled=False)       -> fp32 (outer inert; proves nothing)
        outer DEVICE autocast + inner cuda.amp(enabled=False) -> fp16, non-finite (guard is a NO-OP)
        outer DEVICE autocast + inner DEVICE enabled=False    -> fp32, finite

    ``forward_internal`` and the new guard are both the cuda.amp family, which
    only engages with CUDA. So through the production call path here,
    ``rrd_list[3]`` returns fp16 both before and after the change -- the guard
    cannot suppress the device autocast the test needs to engage. On the cluster
    the caller is cuda.amp (trainer.py) and the guard does suppress it. The end
    to end property is therefore not verifiable on this box, and the last test
    below records that as an explicit skip rather than a green tick.
    """

    @staticmethod
    def _rrd():
        from ecs_hcm_model.core.hcm_nn import hcm_rrd

        torch.manual_seed(0)
        return hcm_rrd()

    @requires_cpu_fp16_autocast
    def test_conv_overflow_mechanism_is_real_and_the_block_fixes_it(self):
        """The op-level fact the fix rests on, at rrd_list[3]'s conv shape."""
        from ecs_hcm_model.core.hcm_nn import ISC_OUTPUT_CHANNEL_NUM

        c = ISC_OUTPUT_CHANNEL_NUM
        conv = nn.Conv2d(c, c, kernel_size=3, padding=1, stride=(1, 2))
        with torch.no_grad():
            conv.weight.zero_()
            for k in range(c):
                conv.weight[k, k, 1, 1] = 1.0
            conv.bias.zero_()

        x = torch.full((1, c, 8, 8), 1.0e5)      # > 65504

        with torch.autocast(device_type=LIVE_AUTOCAST[0], dtype=LIVE_AUTOCAST[1]):
            unguarded = conv(x)
        assert unguarded.dtype == torch.float16
        assert not torch.isfinite(unguarded.float()).all(), (
            "a magnitude above the fp16 ceiling must not survive an unguarded "
            "conv under live autocast"
        )

        with torch.autocast(device_type=LIVE_AUTOCAST[0], dtype=LIVE_AUTOCAST[1]):
            with torch.autocast(device_type=LIVE_AUTOCAST[0], enabled=False):
                guarded = conv(x)
        assert guarded.dtype == torch.float32
        assert torch.isfinite(guarded).all(), (
            "the guard block must keep the overflow from happening at all"
        )

    def test_hcm_nn_wraps_only_the_rung_that_takes_isc_output(self):
        """Pin the shape of the fix in the source, since dtype cannot fully.

        rrd_list[0..2] consume CSC outputs (already small) and stay as they
        were; only the i == 3 branch runs outside autocast. This is the assertion
        that actually moves between the pre- and post-fix trees: it fails on
        cc8ff5f, where the rung is called bare.
        """
        import inspect

        from ecs_hcm_model.core import hcm_nn as module

        source = inspect.getsource(module.hcm_nn.forward_internal)
        assert "isc_output[i].float(), aspp_output.float()" in source, (
            "rrd_list[3] must be fed explicitly-cast fp32 inputs"
        )
        # The rrd block must contain exactly one autocast guard: the one on the
        # i == 3 branch. More than one would mean the guard leaked onto the
        # CSC-consuming rungs, which the item explicitly said to leave alone.
        rrd_start = source.index("rrd_output = [None, None, None, None]")
        assert source.count("autocast(enabled=False)", rrd_start) == 1, (
            "only the i == 3 branch may be wrapped; rrd_list[0..2] take CSC "
            "outputs and were explicitly left as they were"
        )

    @pytest.mark.skip(
        reason=(
            "UNVERIFIABLE ON THIS BOX: the end-to-end claim (rrd_list[3] no "
            "longer hands fp16 to the decoder) cannot be shown without CUDA. "
            "forward_internal and the guard are both torch.cuda.amp, which is "
            "inert without CUDA, while the only autocast that engages here is "
            "the device-type API, which cuda.amp(enabled=False) does not "
            "suppress -- so through the production path rrd_list[3] returns "
            "fp16 before AND after the change. Measured: outer device autocast "
            "+ inner cuda.amp(enabled=False) -> fp16/non-finite; outer device + "
            "inner device enabled=False -> fp32/finite. See the queue item."
        )
    )
    def test_rrd_list_three_returns_fp32_through_the_production_path(self):
        """The claim that needs a CUDA host to verify."""
        raise AssertionError("unreachable while skipped")


class TestIsc3Monitoring:
    """The optional extra: stage 3 tracked on its own."""

    def test_isc3_is_a_tracked_stage(self):
        from ecs_hcm_model.core.hcm_nn import ACTIVATION_STAGES

        assert "isc3" in ACTIVATION_STAGES
        assert "isc" in ACTIVATION_STAGES and "csc" in ACTIVATION_STAGES

    def test_isc3_records_independently_of_the_isc_aggregate(self):
        from ecs_hcm_model.core.hcm_nn import _record_absmax, hcm_isc

        owner = hcm_isc()
        _record_absmax(owner, "isc", torch.full((1, 4, 2, 2), 9.0))
        _record_absmax(owner, "isc3", torch.full((1, 4, 2, 2), 70000.0))

        assert owner.activation_absmax["isc"] == pytest.approx(9.0)
        assert owner.activation_absmax["isc3"] == pytest.approx(70000.0), (
            "isc3 must record its own magnitude; it is the stage that reaches "
            "the decoder, so an aggregate that hides it is not enough"
        )

    def test_trainer_logs_the_isc3_keys(self, monkeypatch, logger):
        from tests.conftest import TinyClassifier

        batches = [
            (torch.randn(4, 8), torch.randint(0, 10, (4,))) for _ in range(250)
        ]
        trainer = TestActivationMetricCadence._patch(monkeypatch, logger, batches)

        model = TinyClassifier()
        original_forward = model.forward

        def big_forward(x):
            out = original_forward(x)
            model.activation_absmax["isc3"] = 66000.0
            return out

        model.forward = big_forward

        trainer.train_model_for_epochs(
            model=model,
            epochs=1,
            train_loader=batches,
            # A non-empty validation loader: evaluate() now raises on an
            # empty one ("ran 0 batches"), which is the point of the guard.
            val_loader=[(torch.randn(4, 8), torch.randint(0, 10, (4,)))],
            device=torch.device("cpu"),
            logger=logger,
            test_mode=True,
            max_batches_per_epoch=250,
            warmup=False,
            task="classification",
        )

        values = logger.scalar_values()
        assert values.get("batch_isc3_absmax") == pytest.approx(66000.0)
        assert values.get("train_isc3_absmax") == pytest.approx(66000.0)
