"""The source closure the measurement Job ships in a ConfigMap (tools/src_closure.py).

A closure that misses a module is an ImportError in a pod two hours into a run, so
these pin the two ways a naive scan gets it wrong: ``from pkg import submodule``
imports the module, and this repo defers imports into functions.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

_spec = importlib.util.spec_from_file_location(
    "src_closure", ROOT / "tools" / "src_closure.py")
src_closure = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(src_closure)


def _tree(tmp_path, files):
    for name, body in files.items():
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(body)
    return tmp_path


def _closure(tmp_path, entry):
    got = src_closure.closure(tmp_path / entry, root=tmp_path)
    return {p.relative_to(tmp_path).as_posix() for p in got}


def test_a_from_import_of_a_submodule_is_followed(tmp_path):
    """``from pkg import sub`` imports pkg.sub; resolving only pkg loses the module."""
    _tree(tmp_path, {
        "pkg/__init__.py": "",
        "pkg/entry.py": "from pkg import sub\n",
        "pkg/sub.py": "",
    })
    assert _closure(tmp_path, "pkg/entry.py") == {
        "pkg/__init__.py", "pkg/entry.py", "pkg/sub.py"}


def test_an_import_deferred_into_a_function_is_followed(tmp_path):
    """The measurement defers minio into a function; a body-only scan misses it."""
    _tree(tmp_path, {
        "pkg/__init__.py": "",
        "pkg/entry.py": (
            "def build():\n"
            "    from pkg.late import thing\n"
            "    return thing\n"),
        "pkg/late.py": "",
    })
    assert "pkg/late.py" in _closure(tmp_path, "pkg/entry.py")


def test_an_import_inside_a_class_body_is_followed(tmp_path):
    """A class body runs at import time too, so it is not a deferred import."""
    _tree(tmp_path, {
        "pkg/__init__.py": "",
        "pkg/entry.py": (
            "class K:\n"
            "    from pkg.body import thing\n"),
        "pkg/body.py": "",
    })
    assert "pkg/body.py" in _closure(tmp_path, "pkg/entry.py")


def test_third_party_imports_are_not_shipped(tmp_path):
    """torch, numpy and minio are the image's job; site-packages is not in the pod."""
    _tree(tmp_path, {
        "pkg/__init__.py": "",
        "pkg/entry.py": "import torch\nfrom numpy import array\nfrom pkg import sub\n",
        "pkg/sub.py": "",
    })
    assert _closure(tmp_path, "pkg/entry.py") == {
        "pkg/__init__.py", "pkg/entry.py", "pkg/sub.py"}


def test_the_real_entry_reaches_the_measurement_modules():
    """Guards the tree the Job actually ships, not a fixture."""
    got = src_closure.closure(ROOT / "tools" / "measure_camera_frames.py")
    names = {p.relative_to(ROOT).as_posix() for p in got}
    assert {"tools/measure_camera_frames.py",
            "ecs_hcm_model/eval/camera_frames.py",
            "ecs_hcm_model/eval/seg.py",
            "ecs_hcm_model/core/bdd.py"} <= names
