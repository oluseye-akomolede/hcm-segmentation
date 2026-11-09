"""The demos' opt-in --fp16: one forward (demo_model.demo_forward), fp32 by default.

The demos import cv2, which this VM lacks, so their wiring is checked on the
source (as in test_demo_paths); demo_forward and the flag parsing run for real.
"""
import ast
import argparse
from pathlib import Path

import pytest
import torch

from ecs_hcm_model.core.hcm_nn import DECODER_V1, DECODER_V2, hcm_nn
from jetson.demo_model import demo_forward, require_fp16_device
from tests.test_decoder_arch import _offline
# The same probe and skip reason the trainer's fp16 tests use: torch 2.1 (the
# CI image) raises on entering CPU float16 autocast.
from tests.test_fp16_overflow import requires_cpu_fp16_autocast

REPO = Path(__file__).resolve().parents[1]
DEMOS = [REPO / "jetson" / "realtime_demo.py", REPO / "video" / "video_demo.py",
         REPO / "video" / "image_demo.py"]


def _cpu_autocast_state():
    """(enabled, dtype) of CPU autocast. torch 2.1 (the CI image) has only the
    per-device functions; 2.4+ takes the device as an argument."""
    try:
        on = torch.is_autocast_enabled("cpu")
        return on, (torch.get_autocast_dtype("cpu") if on else None)
    except TypeError:
        on = torch.is_autocast_cpu_enabled()
        return on, (torch.get_autocast_cpu_dtype() if on else None)


class SpyAutocast:
    """Stands in for torch.autocast: records each context's arguments and
    whether a forward ran inside it. Needs no CPU fp16 support, so the wiring
    is proven on torch 2.1 too."""

    def __init__(self):
        self.calls = []
        self.active = False

    def __call__(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        spy = self

        class _Ctx:
            def __enter__(self):
                spy.active = True

            def __exit__(self, *exc):
                spy.active = False

        return _Ctx()


class Probe(torch.nn.Module):
    """Records what it ran under. Under the spy it returns fp16, as a real
    float16 autocast would, so a forward that forgets the fp32 cast shows."""

    def __init__(self, spy=None):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 2, 1)
        self.spy = spy
        self.seen = []

    def forward(self, x):
        under_spy = bool(self.spy and self.spy.active)
        self.seen.append((under_spy, _cpu_autocast_state()))
        y = self.conv(x)
        return y.half() if under_spy else y


@pytest.fixture
def spy(monkeypatch):
    s = SpyAutocast()
    monkeypatch.setattr(torch, "autocast", s)
    return s


def test_without_the_flag_no_autocast_context_is_entered(spy):
    probe, x = Probe(spy), torch.randn(1, 3, 4, 4)
    y = demo_forward(probe, x)
    assert spy.calls == []
    assert probe.seen == [(False, (False, None))]
    assert y.dtype == torch.float32 and not y.requires_grad


def test_with_the_flag_the_forward_runs_inside_float16_autocast_and_returns_fp32(spy):
    probe, x = Probe(spy), torch.randn(1, 3, 4, 4)
    y = demo_forward(probe, x, fp16=True)
    assert spy.calls == [((), {"device_type": "cpu", "dtype": torch.float16})]
    assert [under for under, _ in probe.seen] == [True]
    assert y.dtype == torch.float32 and not y.requires_grad


@requires_cpu_fp16_autocast
def test_with_the_flag_cpu_autocast_is_live_at_float16():
    # The real context, where this torch supports it: the spy above proves the
    # call, this proves torch honours it.
    probe, x = Probe(), torch.randn(1, 3, 4, 4)
    y = demo_forward(probe, x, fp16=True)
    assert probe.seen == [(False, (True, torch.float16))]
    assert y.dtype == torch.float32 and not y.requires_grad


@requires_cpu_fp16_autocast
@pytest.mark.parametrize("arch", [DECODER_V1, DECODER_V2])
def test_fp16_and_fp32_agree_coarsely_on_a_seeded_tiny_model(arch):
    # A sanity check, not the accuracy evidence (that is the eval's 70.88 on
    # the Jetson): hcm_nn keeps some regions fp32 with torch.cuda.amp's
    # autocast(enabled=False), which does not switch off CPU autocast, so on
    # CPU those regions run fp16 too. Measured on seed 0: rel 0.024 (v1),
    # 0.015 (v2); argmax agreement 0.996 for both.
    torch.manual_seed(0)
    model = _offline(lambda: hcm_nn(enable_finetuning=True, num_segmentation_classes=2,
                                    out_stride=16, decoder_arch=arch)).eval()
    x = torch.randn(1, 3, 64, 64, generator=torch.Generator().manual_seed(0))
    fp32, fp16 = demo_forward(model, x), demo_forward(model, x, fp16=True)
    assert not torch.equal(fp32, fp16)  # autocast really changed the arithmetic
    assert (fp32 - fp16).abs().max() / fp32.abs().max() < 0.05
    assert (fp32.argmax(1) == fp16.argmax(1)).float().mean() >= 0.98


def test_fp16_is_refused_off_cuda_and_allowed_on_it():
    require_fp16_device(False, "cpu")
    require_fp16_device(True, "cuda")
    require_fp16_device(True, torch.device("cuda", 0))
    with pytest.raises(ValueError, match=r"^--fp16 needs a CUDA device \(this run is on cpu\)$"):
        require_fp16_device(True, "cpu")


def _function(src_file, name):
    tree = ast.parse(src_file.read_text())
    (fn,) = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == name]
    return fn


@pytest.mark.parametrize("demo", DEMOS[:2], ids=lambda p: p.name)
def test_the_flag_parses_off_by_default(demo):
    # The module-level names _parse_args closes over: argparse, and Path for --engine.
    ns = {"argparse": argparse, "__doc__": "", "Path": Path}
    exec(compile(ast.Module([_function(demo, "_parse_args")], []), str(demo), "exec"), ns)
    assert ns["_parse_args"]([]).fp16 is False
    assert ns["_parse_args"](["--fp16"]).fp16 is True


@pytest.mark.parametrize("demo", DEMOS, ids=lambda p: p.name)
class TestEveryDemo:
    def test_it_has_the_flag(self, demo):
        assert 'ap.add_argument("--fp16", action="store_true",' in demo.read_text()

    def test_every_forward_is_demo_forward_carrying_the_flag(self, demo):
        tree = ast.parse(demo.read_text())
        calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)]
        forwards = [c for c in calls if isinstance(c.func, ast.Name) and c.func.id == "demo_forward"]
        assert len(forwards) == 2  # warmup and inference
        for c in forwards:
            (kw,) = [k for k in c.keywords if k.arg == "fp16"]
            # args.fp16 / ARGS.fp16, or image_demo's load_model parameter
            assert ast.unparse(kw.value) in ("args.fp16", "ARGS.fp16", "fp16")
        assert not [c for c in calls if isinstance(c.func, ast.Name) and c.func.id == "model"]

    def test_fp16_is_checked_before_the_model_loads(self, demo):
        calls = [n for n in ast.walk(ast.parse(demo.read_text()))
                 if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)]
        lines = lambda name: [c.lineno for c in calls if c.func.id == name]
        # image_demo loads through its own load_model(), defined above main().
        loads = lines("load_model") or lines("load_demo_model")
        (check,) = lines("require_fp16_device")
        assert loads and check < min(loads)


def test_image_demo_passes_the_flag_to_its_warmup():
    src = (REPO / "video" / "image_demo.py").read_text()
    assert "input_size_source=input_size_source,\n                          fp16=args.fp16)" in src


def test_run_demo_passes_its_arguments_through():
    assert '"$PYTHON" -u "$DEMO" "$@"' in (REPO / "jetson" / "run_demo.sh").read_text()
