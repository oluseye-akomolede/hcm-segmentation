"""Item 6 of the demo audit: dead code gone, skip count reported, P_MIN unchanged."""

import ast
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
REALTIME = (REPO / "jetson" / "realtime_demo.py").read_text()
VIDEO = (REPO / "video" / "video_demo.py").read_text()


def _top_level_names(src):
    names = set()
    for node in ast.parse(src).body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, ast.Assign):
            names.update(t.id for t in node.targets if isinstance(t, ast.Name))
    return names


def test_the_realtime_demo_has_no_dead_helpers():
    gone = {"make_palette", "logits_to_mask", "overlay_mask", "THRESH", "ALPHA"}
    assert not gone & _top_level_names(REALTIME)
    assert "palette = np.array(" not in REALTIME


def test_the_video_demo_reports_its_skips_at_done():
    assert "skipped_read += 1" in VIDEO
    assert "skipped_write += 1" in VIDEO
    done = VIDEO[VIDEO.index('print(f"Done.'):]
    assert "skipped_read" in done and "skipped_write" in done


def test_p_min_is_unchanged_in_both_demos():
    assert "P_MIN = 0.5" in REALTIME
    assert "P_MIN = 0.2" in VIDEO


def test_the_video_comment_no_longer_claims_to_match_realtime():
    assert "same as your realtime code" not in VIDEO
