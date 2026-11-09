"""Environment overrides for the demos' directories, and no mkdir at load.

The demos import cv2 and torch, and video_demo runs at module level, so these
check the path expressions by evaluating them as written in the source.
"""

import ast
import os
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
REALTIME = REPO / "jetson" / "realtime_demo.py"
VIDEO = REPO / "video" / "video_demo.py"


def _path_expr(src_file, name):
    """The right-hand side of ``name = ...`` at module level, as source."""
    tree = ast.parse(src_file.read_text())
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == name for t in node.targets
        ):
            return ast.get_source_segment(src_file.read_text(), node.value)
    raise AssertionError(f"{name} not assigned at module level in {src_file.name}")


def _eval_path(src_file, name, env):
    old = dict(os.environ)
    try:
        for k in ("ECS_HCM_DEMO_ART_DIR", "ECS_HCM_DEMO_FRAMES_DIR", "ECS_HCM_DEMO_OUT_DIR"):
            os.environ.pop(k, None)
        os.environ.update(env)
        return eval(_path_expr(src_file, name), {"Path": Path, "os": os})
    finally:
        os.environ.clear()
        os.environ.update(old)


CASES = [
    (REALTIME, "ART_DIR", "ECS_HCM_DEMO_ART_DIR", Path("/home/user/models")),
    (VIDEO, "ART_DIR", "ECS_HCM_DEMO_ART_DIR", Path("/home/user/models")),
    (VIDEO, "FRAMES_DIR", "ECS_HCM_DEMO_FRAMES_DIR", Path("~/clustering/videos/frames").expanduser()),
    (VIDEO, "OUT_DIR", "ECS_HCM_DEMO_OUT_DIR", Path("~/clustering/videos/frames_out").expanduser()),
]


@pytest.mark.parametrize("src,name,var,default", CASES, ids=lambda c: getattr(c, "name", str(c)))
def test_unset_keeps_the_current_path(src, name, var, default):
    assert _eval_path(src, name, {}) == default


@pytest.mark.parametrize("src,name,var,default", CASES, ids=lambda c: getattr(c, "name", str(c)))
def test_the_variable_moves_the_path(src, name, var, default, tmp_path):
    assert _eval_path(src, name, {var: str(tmp_path)}) == tmp_path


@pytest.mark.parametrize("src", [REALTIME, VIDEO], ids=["realtime", "video"])
def test_weights_and_metadata_follow_art_dir(src):
    text = src.read_text()
    assert "WEIGHTS = ART_DIR /" in text
    assert "META    = ART_DIR /" in text


def test_the_video_demo_makes_its_output_dir_only_after_finding_frames():
    text = VIDEO.read_text()
    mkdir = text.index("OUT_DIR.mkdir(")
    assert text.index("frames = list_frames(FRAMES_DIR)") < mkdir
    assert text.index('raise RuntimeError(f"No frames found') < mkdir
    assert text.count("OUT_DIR.mkdir(") == 1


def test_the_realtime_demo_makes_no_directory_at_load():
    tree = ast.parse(REALTIME.read_text())
    top_level = [n for n in tree.body if not isinstance(n, (ast.FunctionDef, ast.ClassDef, ast.If))]
    src = REALTIME.read_text()
    assert not any("mkdir" in (ast.get_source_segment(src, n) or "") for n in top_level)
