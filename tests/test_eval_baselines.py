"""A public baseline is scored through the eval tool's own path, or refused.

The adapters in ``ecs_hcm_model/eval/baselines.py`` are exercised here with a
known-answer adapter registered under a test name, so the whole path -- the
trainer's transform, its collate, ``evaluate()``, the metric and ``--by-size`` --
runs without any baseline package installed. A real package is only imported by
``build()``, for a run that names one.
"""
import json
import os
import subprocess
import sys
import types
from pathlib import Path

import pytest
import torch

from ecs_hcm_model.eval import baselines, seg
from jetson import bench_model
from tests.test_eval_seg import PixelOracle, coco_root  # noqa: F401  (fixture)

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def _pinned_datasets(monkeypatch):
    """Report the trainer's pinned datasets, as the eval tests do."""
    monkeypatch.setattr(seg, "_datasets_version", lambda: "2.14.0")


class OracleForeground(baselines.ForegroundModel):
    """A known-answer adapter: foreground exactly where the input is bright.

    The same rule as ``PixelOracle``, in the adapter's currency -- a probability
    rather than logits -- so the metric's foreground channel has to round-trip it.
    """

    def __init__(self, spec, module, device, size=None, invert=False):
        super().__init__(spec, module, device, size)
        self.invert = invert

    def foreground(self, images):
        # [B,H,W], one probability per pixel -- what an adapter's contract is.
        fg = (images.mean(dim=1) > 0).float()
        return (1.0 - fg) if self.invert else fg


class InvertedForeground(OracleForeground):
    def __init__(self, spec, module, device, size=None):
        super().__init__(spec, module, device, size, invert=True)


def _spec(name="oracle", native_size=192, caveat=None, caveat_note=None,
          wrapper=OracleForeground, load=None, timed_note="the adapter's own forward"):
    return baselines.BaselineSpec(
        name=name, native_size=native_size, size_source=f"adapter {name} (native {native_size})",
        wrapper=wrapper, load=load or (lambda device: torch.nn.Linear(1, 1)),
        package="torch", install="pip install torch",
        caveat=caveat, caveat_note=caveat_note, timed_note=timed_note,
    )


@pytest.fixture
def oracle(monkeypatch):
    """A known-answer adapter registered under a test name."""
    monkeypatch.setitem(baselines.BASELINES, "oracle", _spec())
    return "oracle"


def _cli(capsys, root, *extra):
    code = seg.main(["--dataset", "coco", "--data-root", str(root), "--device", "cpu", *extra])
    return code, capsys.readouterr()


# ---------------------------------------------------------------------------
# The registry: what is declared, and what is refused
# ---------------------------------------------------------------------------

class TestRegistry:
    def test_the_public_baselines_are_declared(self):
        assert baselines.names() == ["torchvision-lraspp", "yolo11n-seg"]

    def test_each_declares_a_size_and_an_install(self):
        for name in baselines.names():
            spec = baselines.get(name)
            assert spec.native_size > 0 and spec.size_source and spec.package and spec.install

    def test_an_unknown_name_names_the_known_ones(self):
        with pytest.raises(baselines.BaselineError) as e:
            baselines.resolve("yolo")
        assert "unknown baseline 'yolo'" in str(e.value)
        assert "torchvision-lraspp" in str(e.value) and "yolo11n-seg" in str(e.value)

    def test_the_declared_size_is_the_size_used_when_none_is_asked_for(self):
        spec, size, source = baselines.resolve("yolo11n-seg")
        assert (size, spec.native_size) == (640, 640)
        assert "adapter" in source

    def test_the_declared_size_may_be_repeated(self):
        assert baselines.resolve("yolo11n-seg", 640)[1] == 640

    def test_another_size_is_used_and_the_row_says_which_it_was(self):
        """The row wants each model "as published" and at our own 352, so another
        size is a deliberate run, not a refusal -- but the row must not read as
        having run at 640."""
        spec, size, source = baselines.resolve("yolo11n-seg", 352)
        assert size == 352
        assert "640" in source and "native" in source and "--input-size 352" in source

    def test_a_size_nothing_can_be_built_at_is_refused_as_a_checkpoints_is(self):
        """``--input-size`` reaches two resolvers; a non-positive one must not be
        refused by only one of them, so the refusal is the same sentence."""
        for bad in (0, -1):
            with pytest.raises(baselines.BaselineError) as e:
                baselines.resolve("yolo11n-seg", bad)
            with pytest.raises(seg.EvalError) as ckpt:
                seg.resolve_size("a-checkpoint.pth", bad)
            assert str(e.value) == str(ckpt.value) == f"--input-size must be positive, got {bad}"

    def test_a_missing_package_is_refused_with_the_install_command(self, monkeypatch):
        def load(device):
            raise ImportError("No module named 'nope'")
        monkeypatch.setitem(baselines.BASELINES, "needs-nope", _spec(name="needs-nope", load=load))
        with pytest.raises(baselines.BaselineError) as e:
            baselines.build("needs-nope", "cpu")
        assert "needs the torch package" in str(e.value)
        assert "pip install torch" in str(e.value)


def test_importing_the_adapters_imports_no_baseline_package():
    """The tool loads on a box where no baseline is installed. A package is only
    imported by build(), so a module-scope import fails here."""
    code = ("import sys; import ecs_hcm_model.eval.baselines as b; "
            "print(sorted(s.package for s in b.BASELINES.values() if s.package in sys.modules))")
    p = subprocess.run([sys.executable, "-W", "ignore", "-c", code],
                       env=dict(os.environ, PYTHONPATH=str(REPO)),
                       capture_output=True, text=True, timeout=300)
    assert p.returncode == 0, p.stderr[-2000:]
    assert p.stdout.strip() == "[]"


# ---------------------------------------------------------------------------
# The metric's own interface: probability in, the same probability out
# ---------------------------------------------------------------------------

class TestTheMetricInterface:
    def test_the_probability_round_trips(self):
        p = torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0]).view(1, 1, 1, 5)
        p_fg = torch.softmax(baselines.foreground_logits(p), dim=1)[:, 1].flatten()
        assert torch.allclose(p_fg, p.flatten(), atol=1e-6)
        # The metric's 0.5 is the model's own 0.5: 0.5 itself is background.
        assert [bool(v) for v in (p_fg > 0.5)] == [False, False, False, True, True]

    def test_a_probability_that_is_not_per_pixel_is_refused(self, oracle):
        """The [B,2,H,W] logits are built from a [B,H,W] probability: a [B,1,H,W]
        one is a shape the metric would take and score wrongly."""

        class Keepdim(baselines.ForegroundModel):
            def foreground(self, images):
                return (images.mean(dim=1, keepdim=True) > 0).float()

        model = Keepdim(baselines.get(oracle), torch.nn.Linear(1, 1), "cpu")
        with pytest.raises(baselines.BaselineError) as e:
            model(torch.zeros(2, 3, 8, 8))
        assert "[B,H,W]" in str(e.value) and "(2, 1, 8, 8)" in str(e.value)

    def test_denormalize_undoes_the_transform_the_tool_uses(self, coco_root):
        # Exactly the transform's mean: zeros in normalized space are the mean, so
        # this pins the affine step itself, which a clamp of the input also passes.
        zeros = torch.zeros(1, 3, 2, 2)
        back = baselines.denormalize(zeros)
        assert torch.allclose(back[0, :, 0, 0], torch.tensor(baselines.IMAGENET_MEAN), atol=1e-6)
        row = seg.TransformedRows(seg.CocoRows(coco_root), 192)[0]
        pixels = row["pixel_values"].unsqueeze(0)
        back = baselines.denormalize(pixels)
        assert back.shape == pixels.shape
        # The fixture paints DARK (20) and BRIGHT (235), so the extremes must come
        # back: a wrong mean or std in the adapter would clamp them short.
        assert back.min() <= 20 / 255 + 0.02
        assert back.max() >= 235 / 255 - 0.02


# ---------------------------------------------------------------------------
# Refusals, all before any image is read
# ---------------------------------------------------------------------------

class TestRefusalsBeforeAnyImageIsRead:
    @pytest.fixture(autouse=True)
    def _no_data(self, monkeypatch):
        monkeypatch.setattr(seg, "build_rows", lambda *a, **k: pytest.fail("data was read"))

    def test_an_unknown_baseline(self, coco_root, capsys):
        code, out = _cli(capsys, coco_root, "--baseline", "yolo")
        assert code == 2
        assert "unknown baseline" in out.err and "yolo11n-seg" in out.err

    def test_a_baseline_with_a_checkpoint_path(self, coco_root, tmp_path, capsys):
        code, out = _cli(capsys, coco_root, "--baseline", "yolo11n-seg", str(tmp_path / "a.pth"))
        assert code == 2
        assert "checkpoint path" in out.err and "--baseline yolo11n-seg" in out.err

    def test_a_baseline_with_a_checkpoints_options(self, coco_root, tmp_path, capsys):
        for flag, value in (("--metadata", str(tmp_path / "m.json")), ("--out-stride", "8")):
            code, out = _cli(capsys, coco_root, "--baseline", "yolo11n-seg", flag, value)
            assert code == 2 and flag in out.err, flag

    def test_no_checkpoint_and_no_baseline(self, coco_root, capsys):
        code, out = _cli(capsys, coco_root)
        assert code == 2
        assert "no checkpoint given" in out.err and "--baseline" in out.err

    @pytest.mark.parametrize("size", ["0", "-1"])
    def test_a_non_positive_size_fails_the_run_before_any_image(self, coco_root, capsys, size):
        """A baseline is built at the size it is resolved to, and the size is
        resolved before the val set is read (the ``_no_data`` fixture above is what
        says so), so a bad size costs seconds, not a pass over 4952 images."""
        code, out = _cli(capsys, coco_root, "--baseline", "yolo11n-seg", "--input-size", size)
        assert code == 2
        assert f"--input-size must be positive, got {size}" in out.err


# ---------------------------------------------------------------------------
# End to end: the same transform, collate, metric and breakdown
# ---------------------------------------------------------------------------

class TestScoring:
    def test_a_known_answer_adapter_scores_what_a_checkpoint_scores(self, coco_root, oracle):
        rows = seg.CocoRows(coco_root)
        got = seg.score(baselines.build(oracle, "cpu"), rows, 192, "cpu")
        want = seg.score(PixelOracle(), rows, 192, "cpu")
        assert got["iou"] > 95
        assert got["iou"] == pytest.approx(want["iou"], abs=0.05)
        assert got["dice"] == pytest.approx(want["dice"], abs=0.05)
        assert got["n_images"] == want["n_images"]

    def test_an_inverted_adapter_scores_near_zero(self, coco_root, monkeypatch):
        monkeypatch.setitem(baselines.BASELINES, "inverted", _spec(name="inverted",
                                                                   wrapper=InvertedForeground))
        rows = seg.CocoRows(coco_root)
        got = seg.score(baselines.build("inverted", "cpu"), rows, 192, "cpu")
        assert got["iou"] < 5

    def test_the_breakdown_is_the_one_a_checkpoint_gets(self, coco_root, oracle):
        rows = seg.CocoRows(coco_root)
        got = seg.score(baselines.build(oracle, "cpu"), rows, 192, "cpu", by_size=True)
        want = seg.score(PixelOracle(), rows, 192, "cpu", by_size=True)
        assert got["by_size"] == want["by_size"]

    def test_the_cli_names_the_baseline_and_where_its_size_came_from(self, coco_root, oracle,
                                                                    capsys):
        code, out = _cli(capsys, coco_root, "--baseline", oracle)
        assert code == 0
        [line] = out.out.strip().splitlines()
        assert line.startswith("RESULT baseline:oracle: ")
        assert "input_size=192 (adapter oracle (native 192))" in line
        assert "n_images=4" in line and "iou=" in line

    def test_the_cli_runs_the_breakdown_for_a_baseline(self, coco_root, oracle, capsys):
        code, out = _cli(capsys, coco_root, "--baseline", oracle, "--by-size")
        assert code == 0
        assert "by_size=small:" in out.out and "rule=origpx" in out.out

    def test_a_baseline_runs_at_the_size_its_result_reports(self, coco_root, monkeypatch):
        """The row's two rows per model: the size the model is handed is the size
        the result reports, whether that is the adapter's native one or ours."""
        class Recorder(OracleForeground):
            def __init__(self, spec, module, device, size=None):
                super().__init__(spec, module, device, size)
                self.grids = []

            def foreground(self, images):
                self.grids.append(tuple(images.shape[-2:]))
                return super().foreground(images)

        monkeypatch.setitem(baselines.BASELINES, "recorder", _spec(
            name="recorder", native_size=192, wrapper=Recorder))
        rows = seg.CocoRows(coco_root)
        for size, want in ((None, 192), (96, 96)):
            model = baselines.build("recorder", "cpu", size)
            r = seg.score(model, rows, size or 192, "cpu")
            assert r["input_size"] == want
            assert model.grids and set(model.grids) == {(want, want)}

    def test_the_cli_scores_a_baseline_at_our_own_size(self, coco_root, oracle, capsys):
        code, out = _cli(capsys, coco_root, "--baseline", oracle, "--input-size", "96")
        assert code == 0
        [line] = out.out.strip().splitlines()
        assert "input_size=96 (--input-size 96 (adapter oracle's native size is 192))" in line

    def test_a_baseline_that_raises_its_own_error_still_reports_a_result_status(self, coco_root,
                                                                               monkeypatch, capsys):
        def load(device):
            raise baselines.BaselineError("the weights could not be downloaded")
        monkeypatch.setitem(baselines.BASELINES, "broken", _spec(name="broken", load=load))
        code, out = _cli(capsys, coco_root, "--baseline", "broken")
        assert code == 1
        assert "error: baseline broken:" in out.err and "could not be downloaded" in out.err
        assert out.out == ""


# ---------------------------------------------------------------------------
# The caveat travels with the number
# ---------------------------------------------------------------------------

class TestTheCaveat:
    def test_a_non_comparable_baseline_carries_its_caveat(self, coco_root, monkeypatch, capsys):
        monkeypatch.setitem(baselines.BASELINES, "voc", _spec(
            name="voc", caveat="voc21-argmax", caveat_note="VOC's 21 classes, not COCO's 80"))
        code, out = _cli(capsys, coco_root, "--baseline", "voc", "--json")
        r = json.loads(out.out)
        assert code == 0
        assert r["caveat"] == "voc21-argmax" and "21 classes" in r["caveat_note"]
        assert " " not in r["caveat"]

    def test_the_result_line_carries_the_id_and_stderr_the_sentence(self, coco_root, monkeypatch,
                                                                   capsys):
        monkeypatch.setitem(baselines.BASELINES, "voc", _spec(
            name="voc", caveat="voc21-argmax", caveat_note="VOC's 21 classes, not COCO's 80"))
        code, out = _cli(capsys, coco_root, "--baseline", "voc")
        assert code == 0
        [line] = out.out.strip().splitlines()
        assert line.endswith("caveat=voc21-argmax")
        assert "note: baseline voc: voc21-argmax: VOC's 21 classes, not COCO's 80" in out.err

    def test_a_comparable_baseline_reports_no_caveat_at_all(self, coco_root, oracle, capsys):
        code, out = _cli(capsys, coco_root, "--baseline", oracle, "--json")
        r = json.loads(out.out)
        assert code == 0
        assert "caveat" not in r and "caveat_note" not in r
        assert "caveat" not in out.err

    def test_the_declared_baselines_carry_the_caveat_they_should(self):
        # yolo11n-seg's 80 classes are ours; the torchvision weights are not a
        # COCO instance model at all, and that has to be on the row.
        assert baselines.get("yolo11n-seg").caveat is None
        assert baselines.get("torchvision-lraspp").caveat == "voc21-argmax"


# ---------------------------------------------------------------------------
# The package-facing halves: what the adapter hands over, and what it reads back
# ---------------------------------------------------------------------------

class _StubResult:
    """One ultralytics result, as far as the adapter reads it."""

    def __init__(self, data):
        self.masks = None if data is None else types.SimpleNamespace(data=data)


class _StubYolo:
    """The package's YOLO object as the adapter uses it: ``.model`` and ``.predict``."""

    def __init__(self, results, params=1):
        self.model = torch.nn.Linear(params, 1)
        self.results = results
        self.calls = []

    def predict(self, **kwargs):
        self.calls.append(kwargs)
        return self.results


def _yolo_adapter(results, native=640, size=None):
    spec = _spec(name="stub-yolo", native_size=native, wrapper=baselines.YoloForeground)
    return baselines.YoloForeground(spec, _StubYolo(results), "cpu", size)


class TestThePackageAdapters:
    def test_the_package_is_called_at_the_scored_size_on_denormalized_pixels(self):
        adapter = _yolo_adapter([_StubResult(None)])
        # -2 is outside 0-1: only undoing the transform's own normalization can
        # bring it back inside, so this also catches an adapter that skipped it.
        images = torch.full((1, 3, 8, 8), -2.0)
        adapter(images)
        [call] = adapter.module.calls
        assert call["imgsz"] == 640 == adapter.spec.native_size
        assert call["conf"] == baselines.YOLO_CONF
        assert 0.0 <= float(call["source"].min()) and float(call["source"].max()) <= 1.0

    def test_another_size_is_the_size_the_package_is_called_at(self):
        """The row's second row per model: 352, not the model's own 640."""
        adapter = _yolo_adapter([_StubResult(None)], size=352)
        adapter(torch.zeros(1, 3, 8, 8))
        [call] = adapter.module.calls
        assert (call["imgsz"], adapter.spec.native_size) == (352, 640)

    def test_a_result_count_that_does_not_match_the_batch_is_refused(self):
        adapter = _yolo_adapter([_StubResult(None)])
        with pytest.raises(baselines.BaselineError) as e:
            adapter(torch.zeros(2, 3, 8, 8))
        assert "1 results for a batch of 2" in str(e.value)

    def test_no_masks_is_background_everywhere(self):
        fg = baselines.image_foreground(_StubResult(None), torch.zeros(1, 3, 4, 4))
        assert fg.shape == (1, 4, 4) and not fg.any()

    def test_the_union_is_the_largest_instance_per_pixel(self):
        # Two instances over one pixel: a pixel is foreground if either covers it,
        # so the union is the larger probability. The larger one is FIRST, so
        # "the last instance written" does not produce the same number.
        data = torch.tensor([[[0.9]], [[0.1]]])
        fg = baselines.image_foreground(_StubResult(data), torch.zeros(1, 3, 2, 2))
        assert fg.shape == (1, 2, 2)
        assert torch.allclose(fg, torch.full((1, 2, 2), 0.9), atol=1e-6)

    def test_the_torchvision_foreground_is_the_models_own_argmax(self):
        class Stub(torch.nn.Module):
            def forward(self, x):
                out = torch.zeros(x.shape[0], 21, *x.shape[-2:])
                half = out.shape[-1] // 2
                out[:, 0, :, :half] = 4.0     # background wins the left half
                out[:, 7, :, half:] = 6.0     # a class wins the right half
                return {"out": out}

        spec = _spec(name="stub-tv", native_size=8, wrapper=baselines.TorchvisionForeground)
        adapter = baselines.TorchvisionForeground(spec, Stub(), "cpu")
        images = torch.zeros(1, 3, 8, 8)
        p_fg = adapter.foreground(images)
        half = 4
        assert p_fg.shape == (1, 8, 8)
        # A hard 0/1: where the model's argmax is background there is no foreground
        # probability at all, and where it is a class there is all of it.
        assert torch.equal(p_fg[:, :, :half], torch.zeros(1, 8, half))
        assert torch.equal(p_fg[:, :, half:], torch.ones(1, 8, half))
        # The metric's 0.5 then reproduces the argmax rule, so our threshold is not
        # a second decision on top of the model's.
        through_metric = torch.softmax(adapter(images), dim=1)[:, 1]
        assert bool((through_metric[:, :, :half] <= 0.5).all())
        assert bool((through_metric[:, :, half:] > 0.5).all())

    def test_the_argmax_rule_is_not_the_probability_one(self):
        """A pixel where background is the argmax but only just (p_bg < 0.5): the
        argmax rule says background, a `1 - p_background` rule would say
        foreground. This is the rule the row asks for, so pin it."""
        class Stub(torch.nn.Module):
            def forward(self, x):
                # Background leads by a hair, but the 20 classes behind it hold
                # most of the mass: p_bg = 1/(1 + 20 e^-0.05) = 0.05.
                out = torch.full((x.shape[0], 21, *x.shape[-2:]), -0.05)
                out[:, 0] = 0.0
                return {"out": out}

        spec = _spec(name="stub-tv", native_size=8, wrapper=baselines.TorchvisionForeground)
        adapter = baselines.TorchvisionForeground(spec, Stub(), "cpu")
        p_bg = torch.softmax(adapter.module(torch.zeros(1, 3, 8, 8))["out"], dim=1)[:, 0]
        assert float(p_bg.min()) < 0.5, "the fixture must be the ambiguous case"
        assert not adapter.foreground(torch.zeros(1, 3, 8, 8)).any()


# ---------------------------------------------------------------------------
# The Jetson latency leg
# ---------------------------------------------------------------------------

class _StubAdapter:
    """What the bench needs of an adapter: eval(), parameters(), and a call."""

    def __init__(self):
        self.calls = 0

    def eval(self):
        return self

    def parameters(self):
        return [torch.nn.Parameter(torch.zeros(1))]

    def __call__(self, images):
        self.calls += 1
        return torch.zeros(images.shape[0], 2, *images.shape[-2:])


class TestBench:
    @pytest.fixture
    def bench(self, monkeypatch):
        """A registered baseline and a stubbed build: the timing code is what runs."""
        monkeypatch.setitem(baselines.BASELINES, "oracle", _spec(native_size=64))
        monkeypatch.setitem(baselines.BASELINES, "voc", _spec(
            name="voc", native_size=64, caveat="voc21-argmax", caveat_note="other classes"))
        built = []

        def build(name, device, size=None):
            model = _StubAdapter()
            built.append((name, model))
            return model

        monkeypatch.setattr(baselines, "build", build)
        return built

    def _run(self, capsys, *extra):
        code = bench_model.main(["--device", "cpu", "--warmup", "2", "--iters", "3", *extra])
        out, err = capsys.readouterr()
        return code, out, err

    def test_the_report_names_the_baseline_its_size_and_what_was_timed(self, bench, capsys):
        code, out, _ = self._run(capsys, "--baseline", "oracle", "--json")
        r = json.loads(out)
        assert code == 0
        assert (r["baseline"], r["input_size"], r["batch_size"]) == ("oracle", 64, 1)
        assert r["arch"] is None and r["backend"] == "torch" and r["dtype"] == "float32"
        assert r["timed"] == "the adapter's own forward"
        assert 0 < r["p50_ms"] <= r["p90_ms"]
        assert "caveat" not in r

    def test_every_timed_call_is_the_adapters(self, bench, capsys):
        assert self._run(capsys, "--baseline", "oracle")[0] == 0
        [(name, model)] = bench
        assert (name, model.calls) == ("oracle", 2 + 3)

    def test_the_text_report_prints_what_was_timed_and_the_caveat(self, bench, capsys):
        code, out, _ = self._run(capsys, "--baseline", "voc")
        assert code == 0
        assert out.splitlines()[0].startswith("baseline voc, input 64, device cpu")
        assert "timed: the adapter's own forward" in out
        assert "caveat voc21-argmax: other classes" in out

    @pytest.mark.parametrize("flag,value", [("--arch", "v2"), ("--checkpoint", "x"),
                                           ("--out-stride", "8"), ("--engine", "e")])
    def test_the_demos_options_are_refused_with_a_baseline(self, bench, capsys, flag, value):
        code, _, err = self._run(capsys, "--baseline", "oracle", flag, value)
        assert code == 2 and flag in err

    def test_the_latency_leg_runs_at_the_size_it_is_asked_for(self, bench, capsys):
        """The table has two rows per model ("as published" and our 352), so the
        requested size is what is timed -- and the row says where it came from."""
        code, out, _ = self._run(capsys, "--baseline", "oracle", "--input-size", "352", "--json")
        r = json.loads(out)
        assert code == 0
        assert r["input_size"] == 352
        assert "native size is 64" in r["input_size_source"]

    def test_fp16_does_not_apply_to_a_baseline(self, bench, capsys):
        code, _, err = self._run(capsys, "--baseline", "oracle", "--fp16")
        assert code == 2 and "--fp16" in err
