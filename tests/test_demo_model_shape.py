"""The demos refuse a non-binary head and build at the model's own out_stride.

Pure-module tests plus source checks: the demos import cv2 and load weights at
import or in main(), and this VM has no cv2.
"""
import ast
import re
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from jetson.model_shape import (  # noqa: E402
    DEFAULT_OUT_STRIDE,
    ModelShapeError,
    OUT_STRIDE_ENV_VAR,
    require_two_classes,
    resolve_demo_out_stride,
    resolve_out_stride,
)

DEMOS = [REPO_ROOT / "jetson" / "realtime_demo.py",
         REPO_ROOT / "video" / "video_demo.py",
         REPO_ROOT / "video" / "image_demo.py"]


class TestTheHeadMustBeBinary:
    def test_two_classes_load(self):
        require_two_classes(2)

    @pytest.mark.parametrize("n", [1, 3, 21, 81])
    def test_any_other_head_is_refused(self, n):
        with pytest.raises(ModelShapeError, match=f"{n} classes"):
            require_two_classes(n)


class TestTheStrideComesFromTheModel:
    def test_custom_metadata_wins(self):
        meta = {"custom_metadata": {"out_stride": 8}, "out_stride": 32}
        assert resolve_out_stride(meta, {"out_stride": 16}) == (8, "metadata.custom_metadata.out_stride")

    def test_then_the_top_level(self):
        assert resolve_out_stride({"out_stride": 32}, {}) == (32, "metadata.out_stride")

    def test_then_the_checkpoint(self):
        assert resolve_out_stride({}, {"out_stride": 8, "state_dict": {}}) == (8, "checkpoint")
        assert resolve_out_stride(None, {"model_config": {"out_stride": 32}}) == (32, "checkpoint")

    def test_then_16(self):
        assert resolve_out_stride(None, {"a.weight": object()}) == (DEFAULT_OUT_STRIDE, "default")
        assert resolve_out_stride({"custom_metadata": {"input_size": 352}}, None) == (16, "default")

    def test_a_null_is_unstated(self):
        meta = {"custom_metadata": {"out_stride": None}}
        assert resolve_out_stride(meta, {"model_config": {"out_stride": None}}) == (16, "default")

    @pytest.mark.parametrize("bad", [4, 12, 0, -16, "abc", 16.5])
    def test_a_stated_stride_the_backbone_cannot_build_raises(self, bad):
        with pytest.raises(ModelShapeError, match="out_stride"):
            resolve_out_stride({"custom_metadata": {"out_stride": bad}}, None)


LOADER = REPO_ROOT / "jetson" / "demo_model.py"


@pytest.mark.parametrize("demo", DEMOS, ids=lambda p: p.name)
class TestEveryDemoUsesIt:
    def test_it_loads_through_the_shared_loader(self, demo):
        src = demo.read_text()
        assert re.search(r"^from jetson\.demo_model import .*\bload_demo_model\b", src, re.M)
        assert "load_demo_model(" in src
        assert "env=os.environ" in src
        # The model is built in one place only.
        assert "hcm_nn(" not in src
        assert "load_state_dict(" not in src

    def test_it_takes_an_out_stride_flag(self, demo):
        src = demo.read_text()
        assert '"--out-stride"' in src
        assert "cli_stride=" in src


class TestTheSharedLoader:
    def test_the_hardcoded_stride_is_gone(self):
        src = LOADER.read_text()
        assert "out_stride=16" not in src
        assert "out_stride=out_stride" in src

    def test_the_checks_run_before_the_model_is_built(self):
        src = LOADER.read_text()
        # The names, not the literal line: the import is parenthesised and its
        # order is the linter's business, but every check the loader makes must
        # come from the torch-free resolver.
        imported = set()
        for node in ast.walk(ast.parse(src)):
            if isinstance(node, ast.ImportFrom) and node.module == "jetson.model_shape":
                imported |= {alias.name for alias in node.names}
        required = {"ModelShapeError", "resolve_decoder_arch", "require_two_classes",
                    "resolve_demo_out_stride", "resolve_backbone", "resolve_neck_channels",
                    "resolve_mask_stride"}
        assert required <= imported, sorted(required - imported)
        checks = src.index("require_two_classes(num_classes)")
        assert checks < src.index("resolve_demo_out_stride(") < src.index("out_stride=out_stride")
        try_block = src[:checks].rsplit("try:", 1)[1] + src[checks:].split("except ModelShapeError")[0]
        assert "cli_stride, env, meta, ckpt)" in try_block
        assert "Cannot use this checkpoint" in src

    def test_one_line_reports_both_size_and_stride(self):
        src = LOADER.read_text()
        assert 'out_stride: {out_stride} (from {out_stride_source})' in src
        assert 'print(f"input size: {input_size} (from {input_size_source})")' not in src

    def test_it_imports_no_cv2(self):
        # Importability on this VM (no cv2) is the real check: test_v2_consumers imports it.
        assert "import cv2" not in LOADER.read_text()


SIDE8 = {"custom_metadata": {"out_stride": 8}}


class TestTheDemoOverrides:
    """4abc8a4a: --out-stride > env > sidecar > checkpoint > 16."""

    def test_the_flag_wins_over_everything(self):
        stride, source, _ = resolve_demo_out_stride(32, {OUT_STRIDE_ENV_VAR: "16"}, SIDE8, {"out_stride": 16})
        assert (stride, source) == (32, "--out-stride")

    def test_then_the_environment(self):
        stride, source, _ = resolve_demo_out_stride(None, {OUT_STRIDE_ENV_VAR: "32"}, SIDE8, None)
        assert (stride, source) == (32, f"env:{OUT_STRIDE_ENV_VAR}")

    def test_then_the_sidecar(self):
        assert resolve_demo_out_stride(None, {}, SIDE8, {"out_stride": 32}) == (
            8, "metadata.custom_metadata.out_stride", None)

    def test_then_the_checkpoint(self):
        assert resolve_demo_out_stride(None, {}, {}, {"out_stride": 32}) == (32, "checkpoint", None)

    def test_a_null_sidecar_falls_through_to_16(self):
        meta = {"custom_metadata": {"out_stride": None}, "out_stride": None}
        assert resolve_demo_out_stride(None, {}, meta, None) == (16, "default", None)

    def test_an_empty_environment_value_is_unset(self):
        assert resolve_demo_out_stride(None, {OUT_STRIDE_ENV_VAR: ""}, SIDE8, None)[:2] == (
            8, "metadata.custom_metadata.out_stride")

    @pytest.mark.parametrize("cli,env", [(16, {}), (None, {OUT_STRIDE_ENV_VAR: "32"})])
    def test_an_override_that_disagrees_with_the_sidecar_warns(self, cli, env):
        stride, _, warning = resolve_demo_out_stride(cli, env, SIDE8, None)
        assert stride != 8 and "overrides 8" in warning

    def test_an_agreeing_override_does_not_warn(self):
        assert resolve_demo_out_stride(8, {}, SIDE8, None)[2] is None

    def test_an_override_against_the_default_does_not_warn(self):
        assert resolve_demo_out_stride(32, {}, None, None) == (32, "--out-stride", None)

    def test_an_unusable_flag_raises(self):
        with pytest.raises(ModelShapeError, match="--out-stride"):
            resolve_demo_out_stride(12, {}, None, None)


def test_the_launcher_forwards_its_arguments():
    src = (REPO_ROOT / "jetson" / "run_demo.sh").read_text()
    assert '"$DEMO" "$@"' in src
