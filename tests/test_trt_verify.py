"""What ``--verify`` gates on (openspec change: trt-export-fused-groupnorm).

The old gate was mask IoU >= 0.99, and it was unreachable for a *correct* engine: the
smallest mask this change measures is 6 foreground pixels, which score 0.8571 IoU
(6/7 union pixels, the seeded ``random[3]``) however right the engine is. What separates
the good engine from the mis-compiled one (max abs logit diff 0.06 against 6.9) is the
*logits*, not the mask. These tests pin the new gates
individually -- including a case that the old gate would have failed and the new ones pass,
and a case where only the logit ceiling catches a wrong engine -- and pin the two edges the
incident exposed: non-finite output is a failure, and verifying on random input alone is
reported rather than passed off as a real verification.
"""
import contextlib
import io
import os
from pathlib import Path

import pytest
import torch
import torch.nn as nn

from jetson import export_trt

_HEIGHT = _WIDTH = 32
# A 10x10 blob: 100 foreground pixels of 1024, the "small mask" the night stills produce.
_FOREGROUND = [(y, x) for y in range(10) for x in range(10)]
# Half the gap to the other class, so a pixel can change class for 0.2 of logit -- under
# the 0.25 ceiling. A pixel that flips class across a wide gap could only ever fail the
# ceiling too, which would prove nothing about the ceiling being independent.
_MARGIN = 0.1


def _has_cv2() -> bool:
    try:
        import cv2  # noqa: F401  pylint: disable=import-outside-toplevel,unused-import
    except ImportError:
        return False
    return True


def _mask_logits(foreground, *, margin=_MARGIN, height=_HEIGHT, width=_WIDTH):
    """Logits whose argmax mask is ``foreground``: class 1 inside the blob, class 0 outside."""
    logits = torch.zeros(1, 2, height, width)
    logits[:, 1] = -margin
    for y, x in foreground:
        logits[0, 1, y, x] = margin
    return logits


def _pairs(ref, got, *, kind="random", label="random[0]"):
    return [(kind, label, ref, got)]


def test_a_good_engine_passes_and_the_report_names_every_floor():
    torch.manual_seed(0)
    ref = torch.randn(1, 2, 8, 8)
    report = export_trt._verdict(_pairs(ref, ref.clone()), verify_dir=None)

    assert report["passed"] is True
    assert report["inputs"] == 1 and report["random_inputs"] == 1
    assert report["nonfinite_outputs"] == 0
    assert report["max_abs_logit_diff"] == 0.0
    assert report["min_pixel_agreement"] == 1.0
    # Every floor is in the report, so a reader can see what was applied.
    assert report["max_logit_diff_allowed"] == export_trt.MAX_LOGIT_DIFF_CEILING
    assert report["pixel_agreement_required"] == export_trt.PIXEL_AGREEMENT_FLOOR
    assert report["mask_iou_reference"] == export_trt.REFERENCE_MASK_IOU


def test_a_mask_that_is_slightly_off_still_passes(monkeypatch):
    """The false negative the old IoU gate produced, in miniature.

    Three pixels of the blob change class: IoU drops to 0.97, under the 0.99 the old gate
    demanded, and the engine is still correct -- every pixel moved by 0.2 of logit, which is
    what a near-tie looks like. The new gates pass it and still report the IoU.
    """
    ref = _mask_logits(_FOREGROUND)
    got = ref.clone()
    for y, x in _FOREGROUND[:3]:
        got[0, 1, y, x] = -_MARGIN

    report = export_trt._verdict(_pairs(ref, got), verify_dir=None)

    assert report["passed"] is True
    assert report["min_mask_iou"] == pytest.approx(97 / 100)
    assert report["min_mask_iou"] < export_trt.REFERENCE_MASK_IOU  # the old gate's floor
    assert report["max_abs_logit_diff"] == pytest.approx(2 * _MARGIN)


def test_the_logit_ceiling_catches_an_engine_the_mask_cannot_see():
    """A common-mode offset: same argmax everywhere, wrong logits by more than the ceiling."""
    torch.manual_seed(0)
    ref = torch.randn(1, 2, 8, 8)
    got = ref + 0.3

    with pytest.raises(export_trt.EngineError, match="max abs logit diff"):
        export_trt._verdict(_pairs(ref, got), verify_dir=None)


def test_the_logit_ceiling_is_reported_even_when_the_masks_agree():
    """The same case, read off the attached report: the mask gates measured nothing wrong."""
    torch.manual_seed(0)
    ref = torch.randn(1, 2, 8, 8)

    with pytest.raises(export_trt.EngineError) as raised:
        export_trt._verdict(_pairs(ref, ref + 0.3), verify_dir=None)

    report = raised.value.report
    assert report["min_pixel_agreement"] == 1.0
    assert report["min_mask_iou"] == 1.0
    assert report["max_abs_logit_diff"] == pytest.approx(0.3)


def test_a_nonfinite_output_fails_and_the_count_is_in_the_report():
    """The NaN failure: whatever the numbers say, this engine cannot be shipped."""
    ref = torch.zeros(1, 2, 4, 4)
    got = torch.zeros(1, 2, 4, 4)
    got[0, 0, 0, 0] = float("inf")
    got[0, 1, 1, 1] = float("nan")

    with pytest.raises(export_trt.EngineError, match="2 non-finite") as raised:
        export_trt._verdict(_pairs(ref, got), verify_dir=None)

    assert raised.value.report["nonfinite_outputs"] == 2


def test_a_nonfinite_output_fails_even_when_every_other_number_is_perfect():
    """Non-finiteness is not folded into the diff: the diffs here are all zero."""
    ref = torch.zeros(1, 2, 4, 4)
    got = torch.zeros(1, 2, 4, 4)
    got[0, 0, 2, 2] = float("-inf")

    with pytest.raises(export_trt.EngineError) as raised:
        export_trt._verdict(_pairs(ref, got), verify_dir=None)

    report = raised.value.report
    assert report["nonfinite_outputs"] == 1
    assert report["max_abs_logit_diff"] == 0.0
    assert report["min_pixel_agreement"] == 1.0


def test_verifying_without_a_directory_says_so():
    torch.manual_seed(0)
    ref = torch.randn(1, 2, 8, 8)
    report = export_trt._verdict(_pairs(ref, ref.clone()), verify_dir=None)

    assert any("--verify-dir" in note for note in report["notes"])
    assert report["verify_dir"] is None and report["image_inputs"] == 0


def test_verifying_with_a_directory_does_not_add_the_note():
    torch.manual_seed(0)
    ref = torch.randn(1, 2, 8, 8)
    pairs = _pairs(ref, ref.clone()) + _pairs(ref, ref.clone(), kind="image", label="a.jpg")
    report = export_trt._verdict(pairs, verify_dir=Path("/tmp/images"))

    assert report["notes"] == []
    assert report["image_inputs"] == 1 and report["random_inputs"] == 1
    assert report["verify_dir"] == "/tmp/images"


def test_verify_refuses_a_non_cuda_device_before_touching_anything():
    """The engine cannot run on the CPU, so the refusal must come before the model moves."""
    with pytest.raises(export_trt.EngineError, match="needs a CUDA device"):
        export_trt.verify(None, None, input_size=32, images=1, device="cpu")


def test_only_jpg_and_png_files_are_image_candidates(tmp_path):
    (tmp_path / "sub").mkdir()
    for name in ("b.PNG", "a.jpg", "notes.txt", "c.jpeg"):
        (tmp_path / name).write_bytes(b"")
    (tmp_path / "sub" / "d.png").write_bytes(b"")  # a file inside a subdirectory only

    names = [p.name for p in export_trt._verify_image_paths(tmp_path, limit=10)]
    assert names == ["a.jpg", "b.PNG", "c.jpeg"]  # sorted, suffix-matched, no recursion
    assert [p.name for p in export_trt._verify_image_paths(tmp_path, limit=2)] == [
        "a.jpg", "b.PNG"]


def test_the_random_inputs_come_first_and_the_images_follow(monkeypatch, tmp_path):
    """The plumbing of a real directory, with the image decode stubbed out.

    ``_load_verify_image`` is the only part that needs OpenCV, and it is not what this
    test is about: the label a failure names, and the fact that the random inputs are kept.
    """
    for name in ("b.png", "a.jpg", "c.jpg", "notes.txt"):
        (tmp_path / name).write_bytes(b"")
    seen = []

    def fake_load(path, *, input_size, device):
        seen.append((path.name, input_size, device))
        return torch.zeros(1, 3, 4, 4)

    monkeypatch.setattr(export_trt, "_load_verify_image", fake_load)
    inputs = export_trt._verify_inputs(
        input_size=4, images=2, verify_dir=tmp_path, device="cpu")

    assert [(kind, label) for kind, label, _ in inputs] == [
        ("random", "random[0]"), ("random", "random[1]"),
        ("image", "a.jpg"), ("image", "b.png")]
    assert seen == [("a.jpg", 4, "cpu"), ("b.png", 4, "cpu")]
    assert inputs[0][2].shape == (1, 3, 4, 4)


def test_a_directory_with_no_images_is_refused(tmp_path):
    (tmp_path / "notes.txt").write_bytes(b"")
    with pytest.raises(export_trt.EngineError, match="holds no"):
        export_trt._verify_inputs(
            input_size=4, images=2, verify_dir=tmp_path, device="cpu")


def test_the_random_inputs_are_deterministic():
    first = export_trt._verify_inputs(
        input_size=8, images=2, verify_dir=None, device="cpu")
    second = export_trt._verify_inputs(
        input_size=8, images=2, verify_dir=None, device="cpu")
    assert all(torch.equal(a[2], b[2]) for a, b in zip(first, second))


def test_a_missing_verify_dir_is_refused_before_anything_is_written(monkeypatch, tmp_path):
    """The refusal has to leave no trace, or the next run is refused for the wreckage.

    That is why this check sits with the other argument checks, ahead of ``--out`` being
    created. The export dependencies are stubbed out: on a host without the ONNX package
    the real check would refuse first, and the ordering is what is under test.
    """
    monkeypatch.setattr(export_trt, "_require_export_deps", lambda: None)
    weights = tmp_path / "weights.pt"
    weights.write_bytes(b"weights")
    monkeypatch.setattr(export_trt, "_load_checkpoint", lambda path: (nn.Identity(), weights, 32))
    # Everything past the argument checks is stubbed to fail loudly: if the check ever moves
    # below ``--out``'s creation, the run reaches this and leaves a directory behind.
    monkeypatch.setattr(
        export_trt, "export_onnx",
        lambda *a, **k: pytest.fail("--out was created before the --verify-dir check ran"))
    out = tmp_path / "out"

    class _Args:
        checkpoint = tmp_path / "ckpt"
        out = None
        fp16 = False
        workspace_mb = 1024
        opset = 17
        verify = True
        verify_images = 4
        verify_dir = tmp_path / "missing"
        device = "cuda"

    args = _Args()
    args.out = out
    with pytest.raises(export_trt.EngineError, match="is not a directory"):
        export_trt.export(args)
    assert not out.exists()


def test_the_summary_prints_the_floors_and_the_note():
    record = {
        "source_checkpoint": "/tmp/ckpt",
        "source_sha256": "a" * 64,
        "input_size": 352,
        "precision": "fp32",
        "opset": 17,
        "tensorrt_version": "10.3",
        "input_name": "input",
        "input_dtype": "float32",
        "outputs": [{"name": "logits", "shape": [1, 2, 352, 352], "dtype": "float32"}],
        "verify": {
            "random_inputs": 4, "image_inputs": 0, "nonfinite_outputs": 0,
            "max_abs_logit_diff": 0.0604, "max_logit_diff_allowed": 0.25,
            "min_pixel_agreement": 0.999, "pixel_agreement_required": 0.995,
            "min_mask_iou": 0.8571, "mask_iou_reference": 0.99,
            "notes": ["no --verify-dir: verified on seeded random input only, not on real images"],
        },
    }
    text = export_trt.format_report(record)

    assert "limit 0.25" in text and "need 0.995" in text
    assert "0.8571" in text and "not gated" in text
    assert "verify note: no --verify-dir" in text


@pytest.mark.skipif(_has_cv2(), reason="cv2 is installed, so this host can decode images")
def test_without_opencv_the_directory_mode_says_why(tmp_path):
    path = tmp_path / "frame.png"
    path.write_bytes(b"not an image")
    with pytest.raises(export_trt.EngineError, match="OpenCV"):
        export_trt._load_verify_image(path, input_size=32, device="cpu")


@pytest.mark.skipif(not _has_cv2(), reason="cv2 is not installed on the VM or CI")
def test_a_frame_goes_through_the_demos_own_preprocessing(tmp_path, capsys):
    """No second copy of the resize/normalise: the demo's function is what runs."""
    cv2 = pytest.importorskip("cv2")
    np = pytest.importorskip("numpy")
    frame = np.zeros((64, 48, 3), dtype=np.uint8)
    frame[..., 0] = 255  # blue, in the BGR order cv2 reads
    path = tmp_path / "frame.png"
    cv2.imwrite(str(path), frame)

    with contextlib.redirect_stdout(io.StringIO()):
        from jetson import realtime_demo

    got = export_trt._load_verify_image(path, input_size=32, device="cpu")
    want = realtime_demo.preprocess_bgr(frame, 32)
    assert got.dtype == torch.float32 and got.device.type == "cpu"
    assert torch.equal(got, want.to(dtype=torch.float32))
    # Importing the demo prints a flask banner; the exporter captures it, so --json stays
    # parseable.
    assert capsys.readouterr().out == ""


@pytest.mark.skipif(not _has_cv2(), reason="cv2 is not installed on the VM or CI")
def test_a_file_that_is_not_an_image_is_refused(tmp_path):
    path = tmp_path / "frame.png"
    path.write_bytes(b"not an image")
    with pytest.raises(export_trt.EngineError, match="could not be read"):
        export_trt._load_verify_image(path, input_size=32, device="cpu")


# --- the reference is computed in fp32 (openspec change: trt-fp32-reference) -----------
#
# The engine is built with BuilderFlag.TF32 cleared, but torch leaves TF32 on for cuDNN by
# default, so the model --verify compared against was itself a 10-bit-mantissa
# computation and the gate blamed the engine for a difference the reference contributed.
#
# These tests drive verify() itself rather than the context manager alone: the bug they
# guard against is a context manager that exists and is never entered, which a test of the
# helper cannot see.
#
# verify() refuses a non-CUDA device, but its gate is a check on torch.device(...).type,
# which is "cuda" without a GPU present; _verify_inputs is the only part that would build a
# CUDA tensor, so these tests replace it with a CPU one and run everything else for real.


def _flag_values():
    return (torch.backends.cudnn.allow_tf32, torch.backends.cuda.matmul.allow_tf32)


def _drive_verify(monkeypatch, reference):
    """Run the real ``verify`` body against a fake engine and a replaced demo forward."""
    logits = _mask_logits(_FOREGROUND)
    monkeypatch.setattr(
        export_trt, "_verify_inputs",
        lambda **_: iter([("random", "random[0]", logits)]))
    monkeypatch.setattr(export_trt.demo_model, "demo_forward", reference)

    class _Engine:
        def infer(self, x):
            return x.clone()

    class _Model:
        def to(self, device):
            return self

    return export_trt.verify(_Engine(), _Model(), input_size=32, images=1, device="cuda")


def test_the_reference_runs_with_tf32_refused(monkeypatch):
    seen = []

    def reference(model, x):
        seen.append(_flag_values())
        return x.clone()

    report = _drive_verify(monkeypatch, reference)

    assert seen == [(False, False)]
    assert report["passed"] is True


def test_the_flags_are_restored_after_verify(monkeypatch):
    before = _flag_values()

    _drive_verify(monkeypatch, lambda model, x: x.clone())

    assert _flag_values() == before


def test_the_flags_are_restored_when_the_reference_raises(monkeypatch):
    before = _flag_values()

    def reference(model, x):
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError, match="boom"):
        _drive_verify(monkeypatch, reference)

    assert _flag_values() == before


def test_a_callers_own_flag_values_survive(monkeypatch):
    # The two flags have different defaults (cuDNN's is true; matmul's is false on recent
    # torch), so a restore that hardcoded either one would pass the tests above by luck.
    # Inverting both makes a hardcoded restore fail here.
    before = _flag_values()
    torch.backends.cudnn.allow_tf32 = not before[0]
    torch.backends.cuda.matmul.allow_tf32 = not before[1]
    try:
        _drive_verify(monkeypatch, lambda model, x: x.clone())
        assert _flag_values() == (not before[0], not before[1])
    finally:
        torch.backends.cudnn.allow_tf32, torch.backends.cuda.matmul.allow_tf32 = before


def test_verify_does_not_set_the_tf32_override(monkeypatch):
    # NVIDIA_TF32_OVERRIDE would taint the engine rather than fix the comparison: TensorRT
    # records the variable in the engine and refuses to run one whose build-time and
    # run-time values differ, so an engine built under it is only servable by a process
    # that sets it too. The reference is made fp32 through torch's own flags instead.
    # Sampled from inside the reference forward, not after verify returns: a variable set
    # for the duration of the comparison and torn down afterwards is gone by then, which
    # is exactly what an implementation reaching for the override would look like.
    before = os.environ.get("NVIDIA_TF32_OVERRIDE")
    seen = []

    def reference(model, x):
        seen.append(os.environ.get("NVIDIA_TF32_OVERRIDE"))
        return x.clone()

    _drive_verify(monkeypatch, reference)

    assert seen == [before]
