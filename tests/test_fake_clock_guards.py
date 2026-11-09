"""The fake clocks only work if the code reads the clock the way they patch it.

test_epoch_timing replaces ``trainer.time`` with a namespace holding a fake
``perf_counter``; test_eval_seg does the same to ``seg.time`` for
``CountingLoader``. Either fake is silently bypassed if the code under it
imports ``perf_counter`` directly, binds it to a name at import time, or reads
another clock (``time.time``, ``time.monotonic``): the tests would then pass on
the real clock and flake under load again. These checks read the source so
that cannot happen unnoticed.
"""

import ast
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
ALLOWED = {"perf_counter"}


def _time_problems(tree, scope):
    """Every way ``scope`` reads the clock other than calling time.perf_counter()."""
    problems = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module == "time":
            problems.append(f"line {node.lineno}: from time import ...")
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "time" and alias.asname not in (None, "time"):
                    problems.append(f"line {node.lineno}: import time as {alias.asname}")
    called = {id(n.func) for n in ast.walk(scope) if isinstance(n, ast.Call)}
    for node in ast.walk(scope):
        if (isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name)
                and node.value.id == "time"):
            if node.attr not in ALLOWED:
                problems.append(f"line {node.lineno}: time.{node.attr}")
            elif id(node) not in called:
                problems.append(f"line {node.lineno}: time.{node.attr} bound, not called")
    return problems


def _class(tree, name):
    return next(n for n in ast.walk(tree) if isinstance(n, ast.ClassDef) and n.name == name)


@pytest.mark.parametrize("path,scope", [
    ("ecs_hcm_model/core/trainer.py", None),             # the whole module
    ("ecs_hcm_model/eval/seg.py", "CountingLoader"),     # the faked class only
])
def test_faked_code_reads_only_time_perf_counter(path, scope):
    tree = ast.parse((REPO / path).read_text())
    node = tree if scope is None else _class(tree, scope)
    assert _time_problems(tree, node) == [], (
        f"{path}: the tests' fake clock patches only time.perf_counter")


@pytest.mark.parametrize("src,expected", [
    ("from time import perf_counter\n", "from time import"),
    ("import time as t\n", "import time as t"),
    ("import time\nx = time.monotonic()\n", "time.monotonic"),
    ("import time\nx = time.time()\n", "time.time"),
    ("import time\npc = time.perf_counter\n", "bound, not called"),
])
def test_the_guard_catches_each_bypass(src, expected):
    tree = ast.parse(src)
    problems = _time_problems(tree, tree)
    assert any(expected in p for p in problems), problems


def test_the_guard_accepts_the_allowed_form():
    tree = ast.parse("import time\nt0 = time.perf_counter()\n")
    assert _time_problems(tree, tree) == []
