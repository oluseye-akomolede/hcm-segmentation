"""jetson/bench_model.py times the demo's own build and forward (CPU, tiny size)."""
import json
from pathlib import Path

import pytest

from ecs_hcm_model.core.hcm_nn import DECODER_V1, DECODER_V2, hcm_nn
from jetson import bench_model, demo_model
from tests.test_decoder_arch import _offline
from tests.test_v2_consumers import SIZE, uploaded  # noqa: F401  (fixture)

FAST = ["--input-size", str(SIZE), "--warmup", "2", "--iters", "3", "--device", "cpu"]


def _bench(argv, capsys):
    code = _offline(lambda: bench_model.main(argv))
    out, err = capsys.readouterr()
    return code, out, err


@pytest.mark.parametrize("arch", [DECODER_V1, DECODER_V2])
def test_both_decoders_build_and_report_json(arch, capsys):
    code, out, _ = _bench(["--arch", arch, "--json", *FAST], capsys)
    assert code == 0
    r = json.loads(out)
    expected_params = sum(p.numel() for p in _offline(lambda: hcm_nn(
        enable_finetuning=True, num_segmentation_classes=2, out_stride=16, decoder_arch=arch)).parameters())
    assert (r["arch"], r["input_size"], r["device"], r["dtype"]) == (arch, SIZE, "cpu", "float32")
    assert (r["params"], r["warmup"], r["iters"], r["peak_cuda_mem_mb"]) == (expected_params, 2, 3, None)
    assert 0 < r["p50_ms"] <= r["p90_ms"] and r["mean_ms"] > 0


def test_the_text_report(capsys):
    code, out, _ = _bench(["--arch", DECODER_V2, *FAST], capsys)
    assert code == 0
    assert out.splitlines()[0] == f"arch v2, input {SIZE}, device cpu, backend torch, dtype float32"
    assert out.splitlines()[1].startswith("params ")
    assert "3 iters after 2 warmup: mean " in out and "p50 " in out and "p90 " in out
    assert "CUDA" not in out


def test_every_timed_call_is_the_demos_forward_on_the_demos_build(monkeypatch, capsys):
    calls = {"forward": 0, "build": []}
    real_forward, real_build = demo_model.demo_forward, demo_model.build_demo_model

    def forward(model, x, fp16=False):
        calls["forward"] += 1
        return real_forward(model, x, fp16=fp16)

    def build(*a, **k):
        calls["build"].append(a)
        return real_build(*a, **k)

    monkeypatch.setattr(demo_model, "demo_forward", forward)
    monkeypatch.setattr(demo_model, "build_demo_model", build)
    assert _bench(["--arch", DECODER_V2, *FAST], capsys)[0] == 0
    assert calls["forward"] == 2 + 3
    assert calls["build"] == [(2, 16, DECODER_V2, bench_model.torch.device("cpu"))]


def test_a_checkpoint_goes_through_the_demo_loader(uploaded, monkeypatch, capsys):  # noqa: F811
    arch, art, _, _ = uploaded
    seen = []
    real = demo_model.load_demo_model

    def load(weights, meta, **k):
        seen.append((Path(weights), meta["custom_metadata"]["decoder_arch"], k["input_size"]))
        return real(weights, meta, **k)

    monkeypatch.setattr(demo_model, "load_demo_model", load)
    code, out, _ = _bench(["--checkpoint", str(art), "--json", *FAST], capsys)
    assert code == 0 and json.loads(out.splitlines()[-1])["arch"] == arch
    assert seen == [(art / "model.pth", arch, SIZE)]


def test_an_arch_the_checkpoint_contradicts_is_refused(uploaded, capsys):  # noqa: F811
    arch, art, _, _ = uploaded
    other = DECODER_V1 if arch == DECODER_V2 else DECODER_V2
    code, _, err = _bench(["--checkpoint", str(art), "--arch", other, *FAST], capsys)
    assert code == 2 and err.strip() == f"error: --arch {other} but {art} is a {arch} model"


@pytest.mark.parametrize("argv,msg", [
    (["--arch", "v2", "--fp16"], "--fp16 needs a CUDA device (this run is on cpu)"),
    ([], "--arch is required without --checkpoint"),
    (["--arch", "v2", "--iters", "0"], "--warmup must be >= 0 and --iters >= 1"),
])
def test_refusals_are_one_line_and_exit_2(argv, msg, capsys):
    code, out, err = _bench([*FAST, *argv], capsys)
    assert (code, out, err.strip()) == (2, "", f"error: {msg}")


def test_a_missing_checkpoint_is_refused(tmp_path, capsys):
    code, _, err = _bench(["--checkpoint", str(tmp_path), *FAST], capsys)
    assert code == 2 and err.strip() == f"error: no model.pth in {tmp_path}"


def test_bench_and_demo_import_no_cv2_and_the_demo_uses_the_shared_forward():
    root = Path(__file__).resolve().parents[1] / "jetson"
    for name in ("bench_model.py", "demo_model.py"):
        assert "import cv2" not in (root / name).read_text()
    demo = (root / "realtime_demo.py").read_text()
    assert demo.count("demo_forward(model, ") == 2  # warmup and every frame
    assert "model(x)" not in demo and "model(dummy)" not in demo


# --- bench-followups: the timed loop, the input size, the loader's refusals ---

def _edit_meta(art, **custom):
    path = art / "metadata.json"
    meta = json.loads(path.read_text())
    meta["custom_metadata"].update(custom)
    path.write_text(json.dumps(meta))


class FakeClock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


def _fake_timing(monkeypatch, durations_ms, sync_ms=0.0):
    """Timed forwards take ``durations_ms`` in turn and each sync ``sync_ms``,
    on a fake clock; the model is still built through the demo's path."""
    clock, durations, syncs = FakeClock(), iter(durations_ms), []

    def forward(model, x, fp16=False):
        clock.t += next(durations) / 1000

    def device_sync(device):
        def sync():
            syncs.append(clock.t)
            clock.t += sync_ms / 1000
        return sync

    monkeypatch.setattr(bench_model, "_clock", clock)
    monkeypatch.setattr(demo_model, "demo_forward", forward)
    monkeypatch.setattr(bench_model, "_device_sync", device_sync)
    return syncs


def test_percentiles_are_by_rank_on_a_known_distribution(monkeypatch, capsys):
    _fake_timing(monkeypatch, [7, 2, 10, 4, 1, 9, 3, 8, 6, 5])
    code, out, _ = _bench(["--arch", DECODER_V2, "--json", *FAST, "--warmup", "0", "--iters", "10"], capsys)
    r = json.loads(out)
    assert code == 0
    assert (r["p50_ms"], r["p90_ms"], r["mean_ms"]) == pytest.approx((5.0, 9.0, 5.5))


def test_each_timed_iteration_ends_with_the_device_sync(monkeypatch, capsys):
    # A sync that takes 100 ms must land inside every timed iteration.
    syncs = _fake_timing(monkeypatch, [1, 2, 3], sync_ms=100.0)
    code, out, _ = _bench(["--arch", DECODER_V2, "--json", *FAST, "--warmup", "0", "--iters", "3"], capsys)
    r = json.loads(out)
    # One fence before timing in each of the two calls (warmup, timed), then one per timed iteration.
    assert code == 0 and len(syncs) == 2 + 3
    assert (r["p50_ms"], r["p90_ms"]) == pytest.approx((102.0, 103.0))


def test_the_device_sync_synchronises_cuda_and_nothing_else(monkeypatch):
    seen = []
    monkeypatch.setattr(bench_model.torch.cuda, "synchronize", lambda device=None: seen.append(device))
    bench_model._device_sync(bench_model.torch.device("cpu"))()
    assert seen == []
    cuda = bench_model.torch.device("cuda", 0)
    bench_model._device_sync(cuda)()
    assert seen == [cuda]


def test_every_timed_call_runs_in_eval_mode_with_grad_disabled(monkeypatch, capsys):
    seen = []
    real_build = demo_model.build_demo_model

    def build(*a, **k):
        model = real_build(*a, **k)
        model.register_forward_pre_hook(lambda m, inp: seen.append((m.training, bench_model.torch.is_grad_enabled())))
        return model

    monkeypatch.setattr(demo_model, "build_demo_model", build)
    assert _bench(["--arch", DECODER_V2, *FAST], capsys)[0] == 0
    assert seen == [(False, False)] * (2 + 3)


def _ckpt_args(art, *extra):
    return ["--checkpoint", str(art), "--json", "--warmup", "1", "--iters", "1", "--device", "cpu", *extra]


def test_a_checkpoint_without_a_size_flag_runs_at_its_metadata_size(uploaded, monkeypatch, capsys):  # noqa: F811
    _, art, _, _ = uploaded
    _edit_meta(art, input_size=48)
    seen = []
    real = bench_model.resolve_input_size
    monkeypatch.setattr(bench_model, "resolve_input_size", lambda meta, env: seen.append(env) or real(meta, env))
    monkeypatch.delenv("ECS_HCM_DEMO_INPUT_SIZE", raising=False)
    code, out, _ = _bench(_ckpt_args(art), capsys)
    assert code == 0 and json.loads(out.splitlines()[-1])["input_size"] == 48
    assert seen == [bench_model.os.environ]  # the demo's resolver, with the demo's environment


def test_the_environment_wins_over_the_metadata_as_in_the_demo(uploaded, monkeypatch, capsys):  # noqa: F811
    _, art, _, _ = uploaded
    monkeypatch.setenv("ECS_HCM_DEMO_INPUT_SIZE", "48")
    code, out, _ = _bench(_ckpt_args(art), capsys)
    assert code == 0 and json.loads(out.splitlines()[-1])["input_size"] == 48


def test_a_size_flag_that_contradicts_the_metadata_is_refused(uploaded, capsys):  # noqa: F811
    _, art, _, _ = uploaded
    code, out, err = _bench(_ckpt_args(art, "--input-size", "48"), capsys)
    assert code == 2 and err.strip().splitlines() == [
        f"error: --input-size 48 but the checkpoint's metadata.custom_metadata.input_size is {SIZE}"]


def test_a_size_flag_that_agrees_with_the_metadata_runs(uploaded, capsys):  # noqa: F811
    _, art, _, _ = uploaded
    code, out, _ = _bench(_ckpt_args(art, "--input-size", str(SIZE)), capsys)
    assert code == 0 and json.loads(out.splitlines()[-1])["input_size"] == SIZE


def test_without_a_checkpoint_the_default_size_is_352(monkeypatch, capsys):
    shapes = []
    monkeypatch.setattr(demo_model, "demo_forward", lambda model, x, fp16=False: shapes.append(tuple(x.shape)))
    code, out, _ = _bench(["--arch", DECODER_V2, "--json", "--device", "cpu", "--warmup", "0", "--iters", "1"], capsys)
    assert code == 0 and json.loads(out)["input_size"] == 352 and shapes == [(1, 3, 352, 352)]


def test_a_checkpoint_the_loader_refuses_exits_2_with_its_message(uploaded, capsys):  # noqa: F811
    arch, art, _, _ = uploaded
    _edit_meta(art, decoder_arch=DECODER_V1 if arch == DECODER_V2 else DECODER_V2)
    code, out, err = _bench(_ckpt_args(art), capsys)
    lines = err.strip().splitlines()
    assert code == 2 and len(lines) == 1 and lines[0].startswith("error: Cannot use this checkpoint: ")


# --- bench-size-flag-precedence: the flag with no stated size, and over the env ---

def _record_shapes(monkeypatch):
    shapes = []
    monkeypatch.setattr(demo_model, "demo_forward", lambda model, x, fp16=False: shapes.append(tuple(x.shape)))
    return shapes


def test_a_size_flag_is_used_when_the_metadata_states_no_size(uploaded, monkeypatch, capsys):  # noqa: F811
    _, art, _, _ = uploaded
    path = art / "metadata.json"
    meta = json.loads(path.read_text())
    meta["custom_metadata"].pop("input_size")
    meta.pop("input_size", None)
    path.write_text(json.dumps(meta))
    monkeypatch.delenv("ECS_HCM_DEMO_INPUT_SIZE", raising=False)
    shapes = _record_shapes(monkeypatch)
    code, out, err = _bench(_ckpt_args(art, "--input-size", "512"), capsys)
    assert code == 0, err
    assert json.loads(out.splitlines()[-1])["input_size"] == 512
    assert set(shapes) == {(1, 3, 512, 512)}


def test_a_size_flag_wins_over_the_environment(uploaded, monkeypatch, capsys):  # noqa: F811
    _, art, _, _ = uploaded
    monkeypatch.setenv("ECS_HCM_DEMO_INPUT_SIZE", "48")
    shapes = _record_shapes(monkeypatch)
    code, out, err = _bench(_ckpt_args(art, "--input-size", str(SIZE)), capsys)
    assert code == 0, err
    assert json.loads(out.splitlines()[-1])["input_size"] == SIZE
    assert set(shapes) == {(1, 3, SIZE, SIZE)}
