"""The export-time static pooling swap (openspec change: trt-export-static-pool).

The ONNX legacy tracer loses the spatial extent of a tensor that has passed through
GroupNorm, so ``adaptive_avg_pool2d`` has no input size to derive its bins from and the
export of the real model fails with ``SymbolicValueError``. The export swaps in a
static-shape equivalent for the duration of the export call and always restores the
original. These tests pin the equivalent against torch's own operator, pin that the swap
is confined to the export, and -- where the ONNX package is installed -- pin the export
itself against the failure it exists to avoid.
"""
import contextlib
import inspect

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from jetson import export_trt

# The real operator, captured at import time. Every restore assertion compares against
# this and re-installs it first, so a swap that fails to restore cannot be mistaken for
# the original by the next test in the file.
_ORIGINAL_POOL = torch.nn.functional.adaptive_avg_pool2d


def _has_onnx() -> bool:
    try:
        import onnx  # noqa: F401  pylint: disable=import-outside-toplevel,unused-import
    except ImportError:
        return False
    return True


def _export_takes_dynamo() -> bool:
    """Whether this torch's ``torch.onnx.export`` accepts the ``dynamo`` keyword.

    The CI trainer image runs torch 2.1, whose exporter has an explicit parameter list
    without ``dynamo``, so ``export_onnx`` raises
    ``TypeError: export() got an unexpected keyword argument 'dynamo'`` there before any
    tracing happens. The VM (2.14) and the Jetson (2.8) take it. Read off the signature
    rather than a version number; none of these versions wraps the export in ``**kwargs``,
    so the parameter list is the whole story.
    """
    try:
        return "dynamo" in inspect.signature(torch.onnx.export).parameters
    except (TypeError, ValueError):  # not introspectable: assume the call would fail
        return False


# (input shape, output size) -- a divisible extent, an uneven one, an extent smaller
# than the output (overlapping bins), an int output size, and a non-square extent whose
# two axes need different windows (a transposed pair of matrices would pass a square one).
POOL_CASES = [
    ((2, 3, 88, 88), (16, 16)),
    ((2, 3, 22, 22), (16, 16)),
    ((2, 3, 5, 7), (8, 9)),
    ((1, 3, 16, 16), 16),
    ((2, 3, 88, 22), (16, 3)),
]


@pytest.mark.parametrize("shape,output_size", POOL_CASES)
def test_the_static_pool_agrees_with_torch(shape, output_size):
    torch.manual_seed(0)
    x = torch.randn(*shape)
    got = export_trt._static_adaptive_avg_pool2d(x, output_size)
    want = F.adaptive_avg_pool2d(x, output_size)
    assert got.shape == want.shape
    # rtol=0 on purpose: allclose defaults to rtol=1e-5, which for the ~0.5-sized values
    # here would allow a slack of ~5e-6 and make the tight atol inert.
    assert torch.allclose(got, want, rtol=0, atol=1e-6), (got - want).abs().max().item()


def test_the_static_pool_follows_the_input_dtype():
    """float64 in, float64 out, agreeing to machine epsilon.

    The atol is the float64 floor rather than a formula tolerance: the bins are built in
    the input's dtype, so the only difference from the operator is the order the per-bin
    means are summed in (measured worst case 2.2e-16 over the cases above). Building the
    bins in float32 and casting would round twice and miss this by orders of magnitude --
    measured 3.3e-08 for the mutation, which only this atol catches because rtol is 0.

    88 -> 16 is deliberate: its bin widths are 5 and 6, so the weights are not exactly
    representable in float32. A divisible extent (32 -> 16, weights of exactly 1/2) would
    agree even after the double rounding and cannot see the difference.
    """
    x = torch.randn(2, 3, 88, 88, dtype=torch.float64)
    got = export_trt._static_adaptive_avg_pool2d(x, (16, 16))
    assert got.dtype == torch.float64
    assert torch.allclose(
        got, F.adaptive_avg_pool2d(x, (16, 16)), rtol=0, atol=1e-14
    ), (got - F.adaptive_avg_pool2d(x, (16, 16))).abs().max().item()


def test_the_context_manager_swaps_and_restores(monkeypatch):
    monkeypatch.setattr(torch.nn.functional, "adaptive_avg_pool2d", _ORIGINAL_POOL)
    with export_trt._static_pool_for_export():
        assert torch.nn.functional.adaptive_avg_pool2d is export_trt._static_adaptive_avg_pool2d
    assert torch.nn.functional.adaptive_avg_pool2d is _ORIGINAL_POOL


def test_the_original_is_restored_when_the_body_raises(monkeypatch):
    monkeypatch.setattr(torch.nn.functional, "adaptive_avg_pool2d", _ORIGINAL_POOL)
    with pytest.raises(RuntimeError, match="tracer exploded"):
        with export_trt._static_pool_for_export():
            assert torch.nn.functional.adaptive_avg_pool2d is export_trt._static_adaptive_avg_pool2d
            raise RuntimeError("tracer exploded")
    assert torch.nn.functional.adaptive_avg_pool2d is _ORIGINAL_POOL


def test_the_swap_is_installed_for_the_export_call_and_gone_after(monkeypatch, tmp_path):
    """The patch reaches torch.onnx.export and nothing outside it.

    Cheaper and more direct than the real export: the ONNX package is not needed to
    observe when the swap is in place.
    """
    monkeypatch.setattr(torch.nn.functional, "adaptive_avg_pool2d", _ORIGINAL_POOL)
    seen = []

    def spy(*args, **kwargs):
        seen.append(torch.nn.functional.adaptive_avg_pool2d)

    monkeypatch.setattr(torch.onnx, "export", spy)
    export_trt.export_onnx(nn.Identity(), 32, tmp_path / "model.onnx", 17)

    assert seen == [export_trt._static_adaptive_avg_pool2d]
    assert torch.nn.functional.adaptive_avg_pool2d is _ORIGINAL_POOL


def test_the_original_is_restored_when_the_export_raises(monkeypatch, tmp_path):
    monkeypatch.setattr(torch.nn.functional, "adaptive_avg_pool2d", _ORIGINAL_POOL)

    def boom(*args, **kwargs):
        raise RuntimeError("SymbolicValueError: tracer exploded")

    monkeypatch.setattr(torch.onnx, "export", boom)
    with pytest.raises(RuntimeError, match="tracer exploded"):
        export_trt.export_onnx(nn.Identity(), 32, tmp_path / "model.onnx", 17)
    assert torch.nn.functional.adaptive_avg_pool2d is _ORIGINAL_POOL


class _ToySegHead(nn.Module):
    """The failing path in miniature: GroupNorm, interpolate to a symbolic size, add, pool.

    One GroupNorm group over the three channels ``export_onnx`` feeds it, so the trace
    reaches the pooling operator rather than stopping at a channel mismatch.
    """

    def __init__(self):
        super().__init__()
        self.norm = nn.GroupNorm(num_groups=1, num_channels=3)

    def forward(self, x):
        f_s = self.norm(x)
        f_splus1 = self.norm(x)
        upsampled = F.interpolate(
            f_splus1, size=f_s.shape[-2:], mode="bilinear", align_corners=False)
        added_input = torch.add(f_s, upsampled)
        return F.adaptive_avg_pool2d(added_input, (16, 16))


@pytest.mark.skipif(
    not _export_takes_dynamo(),
    reason="this torch's torch.onnx.export takes no dynamo= keyword (CI runs torch 2.1)",
)
def test_a_model_that_loses_its_extent_fails_without_the_swap(monkeypatch, tmp_path):
    """The failure this change exists to avoid, reproduced with the swap disabled.

    The symbolic pass runs before the ONNX package is needed, so this holds on a host
    without it -- which is where the regression would otherwise go unnoticed. It does
    need an exporter that accepts ``dynamo=False``, which is how ``export_onnx`` calls it.
    """
    without = tmp_path / "without.onnx"
    with monkeypatch.context() as patch:
        patch.setattr(export_trt, "_static_pool_for_export", contextlib.nullcontext)
        with pytest.raises(Exception, match="adaptive_avg_pool2d"):
            export_trt.export_onnx(_ToySegHead().eval(), 88, without, 17)
    assert not without.exists()


@pytest.mark.skipif(
    not _export_takes_dynamo(),
    reason="this torch's torch.onnx.export takes no dynamo= keyword (CI runs torch 2.1)",
)
@pytest.mark.skipif(not _has_onnx(), reason="the ONNX package is not installed")
def test_a_model_that_loses_its_extent_exports_with_the_swap(tmp_path):
    """The same model, the same call, with the swap in place: the graph is written."""
    path = tmp_path / "with.onnx"
    export_trt.export_onnx(_ToySegHead().eval(), 88, path, 17)
    assert path.is_file() and path.stat().st_size > 0

    import onnx  # pylint: disable=import-outside-toplevel

    onnx.checker.check_model(str(path))
