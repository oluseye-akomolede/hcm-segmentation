"""The export-time fusion of the v2 rrd's GroupNorm pair (change: trt-export-fused-groupnorm).

TensorRT builds a *wrong* fp32 engine for the v2 decoder: the second GroupNorm chain is
mis-compiled where it lands in the concat buffer, so the engine's logits differ from the
torch model's by ~6.9 (max abs) while eager torch and ONNX Runtime agree to 5e-5. The
export therefore traces the same arithmetic written as a single normalisation over the
channel-concatenated tensor. These tests pin that rewrite against the original forward
bit-for-bit, pin that it is confined to the export call (and restored when the export
raises), and -- where the ONNX package is installed -- pin the exported graph itself.
"""
import contextlib
import inspect

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from ecs_hcm_model.core.hcm_nn import DECODER_V1, DECODER_V2, hcm_rrd
from jetson import export_trt

# Captured at import. The restore assertions compare against this and re-install it first,
# so a swap that fails to restore cannot be mistaken for the original by the next test in
# the file. The import-time test below deliberately does *not* re-install.
_ORIGINAL_RRD_FORWARD = hcm_rrd.forward
_ORIGINAL_POOL = torch.nn.functional.adaptive_avg_pool2d

# (f_s spatial size, p_splus1 spatial size): a 2x upsample, no upsample at all, and a
# non-square pair whose axes change differently.
_SIZES = [((16, 16), (8, 8)), ((12, 12), (12, 12)), ((8, 16), (4, 4))]


def _has_onnx() -> bool:
    try:
        import onnx  # noqa: F401  pylint: disable=import-outside-toplevel,unused-import
    except ImportError:
        return False
    return True


def _export_takes_dynamo() -> bool:
    """Whether this torch's ``torch.onnx.export`` accepts the ``dynamo`` keyword.

    The CI trainer image runs torch 2.1, whose exporter has an explicit parameter list
    without ``dynamo``, so a real export raises ``TypeError`` there. The VM (2.14) and the
    Jetson (2.8) take it. Read off the signature rather than a version number.
    """
    try:
        return "dynamo" in inspect.signature(torch.onnx.export).parameters
    except (TypeError, ValueError):  # not introspectable: assume the call would fail
        return False


def _gated_copies(f_s, p_splus1):
    """v1's two gated copies of ``f_s``, at the size the v2 path would use."""
    repeated_p_s = F.interpolate(
        p_splus1, size=f_s.shape[-2:], mode="bilinear", align_corners=False)
    sig_p_s = F.sigmoid(repeated_p_s)
    return torch.mul(f_s, sig_p_s), torch.mul(f_s, torch.sub(1, sig_p_s))


def _spy_group_norm(monkeypatch, seen, *, expect=None):
    """Record the group count of every ``F.group_norm`` call, in order.

    ``expect`` is asserted *inside* the call rather than from the recorded list, so a
    decoder routed down the wrong branch fails on the group count it was not supposed to
    use -- before that branch can raise an ``AttributeError`` for a layer it does not have.
    """
    original = F.group_norm

    def spy(input, num_groups, *args, **kwargs):
        if expect is not None:
            assert num_groups == expect, f"expected {expect} groups, saw {num_groups}"
        seen.append(num_groups)
        return original(input, num_groups, *args, **kwargs)

    monkeypatch.setattr(F, "group_norm", spy)


@pytest.mark.parametrize("f_size,p_size", _SIZES)
def test_the_fused_forward_is_exact_for_the_v2_decoder(f_size, p_size, monkeypatch):
    """The rewrite is the same function, not a close one: bit-for-bit, not within a tolerance.

    ``torch.equal`` rather than ``allclose``: a tolerance would hide exactly the failure
    this rewrite has to avoid, and ``allclose``'s default ``rtol`` would swamp any tight
    ``atol`` anyway.
    """
    monkeypatch.setattr(hcm_rrd, "forward", _ORIGINAL_RRD_FORWARD)
    torch.manual_seed(0)
    rrd = hcm_rrd(DECODER_V2).eval()
    f_s = torch.randn(1, 32, *f_size)
    p_splus1 = torch.randn(1, 32, *p_size)

    with torch.no_grad():
        want = _ORIGINAL_RRD_FORWARD(rrd, f_s, p_splus1)
        with export_trt._fused_rrd_groupnorm_for_export():
            got = rrd(f_s, p_splus1)

    assert got.shape == want.shape
    assert torch.equal(got, want), (got - want).abs().max().item()


@pytest.mark.parametrize("f_size,p_size", _SIZES)
def test_the_swap_installs_the_fused_forward_for_the_call(f_size, p_size, monkeypatch):
    """Inside the block the class attribute really is the fused call, not a no-op."""
    monkeypatch.setattr(hcm_rrd, "forward", _ORIGINAL_RRD_FORWARD)
    with export_trt._fused_rrd_groupnorm_for_export():
        assert hcm_rrd.forward is not _ORIGINAL_RRD_FORWARD
        assert hcm_rrd.forward.__name__ == "fused_forward"
    assert hcm_rrd.forward is _ORIGINAL_RRD_FORWARD


def test_the_original_forward_is_restored_when_the_body_raises(monkeypatch):
    monkeypatch.setattr(hcm_rrd, "forward", _ORIGINAL_RRD_FORWARD)
    with pytest.raises(RuntimeError, match="tracer exploded"):
        with export_trt._fused_rrd_groupnorm_for_export():
            assert hcm_rrd.forward is not _ORIGINAL_RRD_FORWARD
            raise RuntimeError("tracer exploded")
    assert hcm_rrd.forward is _ORIGINAL_RRD_FORWARD


def test_importing_the_exporter_installs_nothing():
    """A module-level patch would leak into every import of the exporter, training included.

    Checked by provenance, not identity: ``_ORIGINAL_RRD_FORWARD`` is captured after
    ``export_trt`` is imported, so under an import-time leak it *is* the leaked function and
    an identity check would compare the leak with itself. A leak only persists if something
    holds the context manager (a bare ``...().__enter__()`` is collected and its ``finally``
    restores the original), so the leak this guards against is a held reference or a direct
    class-attribute assignment. The identity check stays for a swap left by an earlier test.
    """
    assert hcm_rrd.forward.__module__ == "ecs_hcm_model.core.hcm_nn"
    assert hcm_rrd.forward.__qualname__ == "hcm_rrd.forward"
    assert hcm_rrd.forward is _ORIGINAL_RRD_FORWARD


@pytest.mark.parametrize("f_size,p_size", _SIZES)
def test_the_v1_decoder_is_untouched_by_the_swap(f_size, p_size, monkeypatch):
    monkeypatch.setattr(hcm_rrd, "forward", _ORIGINAL_RRD_FORWARD)
    torch.manual_seed(0)
    rrd = hcm_rrd(DECODER_V1).eval()
    # v1 concatenates the two gated copies on the *width*, then halves it with a strided
    # conv, so the tensor it hands to the second normalisation is a different one.
    f_s = torch.randn(1, 32, *f_size)
    p_splus1 = torch.randn(1, 32, *p_size)

    with torch.no_grad():
        want = _ORIGINAL_RRD_FORWARD(rrd, f_s, p_splus1)
        with export_trt._fused_rrd_groupnorm_for_export():
            got = rrd(f_s, p_splus1)

    assert torch.equal(got, want), (got - want).abs().max().item()


def test_the_swapped_v2_forward_normalises_once_over_all_sixteen_groups(monkeypatch):
    """The swap has to be *reached*, and an unchanged output cannot show that it was.

    A fused body reduced to ``return original(self, f_s, p_splus1)`` returns exactly the
    tensor the rewrite does, so the bit-for-bit test above passes it -- as does every other
    case in this file. What separates the two is the call the swap actually makes, which is
    also the shape TensorRT's lowering depends on: one normalisation over the 64-channel
    concat rather than two over its halves.
    """
    monkeypatch.setattr(hcm_rrd, "forward", _ORIGINAL_RRD_FORWARD)
    seen = []
    _spy_group_norm(monkeypatch, seen)
    torch.manual_seed(0)
    rrd = hcm_rrd(DECODER_V2).eval()
    f_s = torch.randn(1, 32, 8, 8)
    p_splus1 = torch.randn(1, 32, 4, 4)

    with torch.no_grad(), export_trt._fused_rrd_groupnorm_for_export():
        rrd(f_s, p_splus1)

    assert seen == [16], seen  # not the [8, 8] the original forward spells
    assert 16 == 2 * export_trt.RRD_GROUPS


def test_the_swapped_v1_forward_still_normalises_twice_over_eight_groups(monkeypatch):
    """The guard itself, observed the same way: v1 keeps the pair the export must not fuse.

    The group count is asserted inside the spy, so a guard inverted to send v1 down the
    fused branch fails on the count -- an assertion -- rather than on the ``fuse`` layer
    that a v1 ``hcm_rrd`` has not got.
    """
    monkeypatch.setattr(hcm_rrd, "forward", _ORIGINAL_RRD_FORWARD)
    seen = []
    _spy_group_norm(monkeypatch, seen, expect=export_trt.RRD_GROUPS)
    torch.manual_seed(0)
    rrd = hcm_rrd(DECODER_V1).eval()
    f_s = torch.randn(1, 32, 8, 8)
    p_splus1 = torch.randn(1, 32, 4, 4)

    with torch.no_grad(), export_trt._fused_rrd_groupnorm_for_export():
        rrd(f_s, p_splus1)

    assert seen == [export_trt.RRD_GROUPS, export_trt.RRD_GROUPS]
    assert export_trt.RRD_GROUPS == 8


def test_fusing_the_v1_decoder_would_be_a_different_function(monkeypatch):
    """Why the ``decoder_arch`` guard is load-bearing rather than a courtesy.

    v1 concatenates the two gated copies on the *width*, and a normalisation over that
    concat pools its statistics across both copies -- so the joint spelling is a different
    function from the two separate normalisations, not a rearrangement of them. On the
    channel axis (the v2 path) the group boundaries fall between the halves and the joint
    spelling is the same function; here they do not.
    """
    monkeypatch.setattr(hcm_rrd, "forward", _ORIGINAL_RRD_FORWARD)
    torch.manual_seed(0)
    f_s = torch.randn(1, 32, 8, 8)
    p_splus1 = torch.randn(1, 32, 4, 4)

    with torch.no_grad():
        a, b = _gated_copies(f_s, p_splus1)
        separate = torch.cat(
            (F.group_norm(a, export_trt.RRD_GROUPS, eps=1e-5),
             F.group_norm(b, export_trt.RRD_GROUPS, eps=1e-5)), dim=-1)
        joint = F.group_norm(
            torch.cat((a, b), dim=-1), export_trt.RRD_GROUPS, eps=1e-5)

    assert separate.shape == joint.shape
    assert not torch.allclose(joint, separate, rtol=0, atol=1e-6)


def test_both_swaps_are_installed_for_the_export_call_and_gone_after(monkeypatch, tmp_path):
    """The patch reaches ``torch.onnx.export`` and nothing outside it.

    Cheaper and more direct than a real export: neither the ONNX package nor a torch whose
    exporter takes ``dynamo`` is needed to observe when the swaps are in place.
    """
    monkeypatch.setattr(hcm_rrd, "forward", _ORIGINAL_RRD_FORWARD)
    monkeypatch.setattr(torch.nn.functional, "adaptive_avg_pool2d", _ORIGINAL_POOL)
    seen = []

    def spy(*args, **kwargs):
        seen.append((hcm_rrd.forward, torch.nn.functional.adaptive_avg_pool2d))

    monkeypatch.setattr(torch.onnx, "export", spy)
    export_trt.export_onnx(nn.Identity(), 32, tmp_path / "model.onnx", 17)

    assert len(seen) == 1
    assert seen[0][0].__name__ == "fused_forward"
    assert seen[0][1] is export_trt._static_adaptive_avg_pool2d
    assert hcm_rrd.forward is _ORIGINAL_RRD_FORWARD
    assert torch.nn.functional.adaptive_avg_pool2d is _ORIGINAL_POOL


def test_the_original_forward_is_restored_when_the_export_raises(monkeypatch, tmp_path):
    monkeypatch.setattr(hcm_rrd, "forward", _ORIGINAL_RRD_FORWARD)

    def boom(*args, **kwargs):
        raise RuntimeError("trt mis-compiled the graph")

    monkeypatch.setattr(torch.onnx, "export", boom)
    with pytest.raises(RuntimeError, match="mis-compiled"):
        export_trt.export_onnx(nn.Identity(), 32, tmp_path / "model.onnx", 17)
    assert hcm_rrd.forward is _ORIGINAL_RRD_FORWARD


class _ToyV2Head(nn.Module):
    """A 3-channel input into a v2 rrd -- what ``export_onnx`` builds a dummy for.

    ``p_splus1`` is derived from the input rather than held as a buffer so the trace really
    walks the interpolate-concat-normalise path instead of folding it away.
    """

    def __init__(self):
        super().__init__()
        self.entry = nn.Conv2d(3, 32, kernel_size=1)
        self.prior = nn.Conv2d(32, 32, kernel_size=1)
        self.rrd = hcm_rrd(DECODER_V2)

    def forward(self, x):
        f_s = self.entry(x)
        return self.rrd(f_s, self.prior(f_s))


@pytest.mark.skipif(
    not _export_takes_dynamo(),
    reason="this torch's torch.onnx.export takes no dynamo= keyword (CI runs torch 2.1)",
)
@pytest.mark.skipif(not _has_onnx(), reason="the ONNX package is not installed")
def test_the_exported_v2_graph_still_computes_the_model(tmp_path):
    """The export succeeds under the swap and the graph it writes is the model's function."""
    model = _ToyV2Head().eval()
    path = tmp_path / "v2.onnx"
    export_trt.export_onnx(model, 16, path, 17)
    assert path.is_file() and path.stat().st_size > 0

    import onnx  # pylint: disable=import-outside-toplevel

    onnx.checker.check_model(str(path))

    ort = pytest.importorskip("onnxruntime")
    torch.manual_seed(0)
    x = torch.randn(1, 3, 16, 16)
    with torch.no_grad():
        want = model(x)
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    got = torch.from_numpy(session.run(None, {"input": x.numpy()})[0])

    assert got.shape == want.shape
    assert torch.allclose(got, want, rtol=0, atol=1e-4), (got - want).abs().max().item()


@pytest.mark.skipif(
    not _export_takes_dynamo(),
    reason="this torch's torch.onnx.export takes no dynamo= keyword (CI runs torch 2.1)",
)
@pytest.mark.skipif(not _has_onnx(), reason="the ONNX package is not installed")
def test_the_v2_forward_is_untouched_after_a_real_export(tmp_path, monkeypatch):
    """The swap is bounded by the export call in the path that actually runs it."""
    monkeypatch.setattr(hcm_rrd, "forward", _ORIGINAL_RRD_FORWARD)
    export_trt.export_onnx(_ToyV2Head().eval(), 16, tmp_path / "v2.onnx", 17)
    assert hcm_rrd.forward is _ORIGINAL_RRD_FORWARD
