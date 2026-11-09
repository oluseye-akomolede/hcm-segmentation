"""The repo's own modules a measurement pod needs, in import order (design D8).

The Job mounts its source from a ConfigMap, and a ConfigMap holds at most 1 MiB
of key data (measured 2026-10-27: 1,000,000 bytes accepted). Shipping the whole
package would spend that budget on modules the measurement never imports, so the
launcher ships the closure from the entry script instead -- and this is the
closure.

Imports anywhere in the file count, not just at module level: the measurement
defers ``minio`` and ``torch`` into functions (``from minio import Minio  # noqa:
PLC0415``), so a body-only scan would miss them.

Only names that resolve inside this repo are returned. Third-party imports
(``torch``, ``numpy``, ``minio``) are the image's job, and site-packages is not
shipped.

It lives in ``tools/`` rather than ``deploy/jobs/`` so the CI job ships it (the
suite archives ``tools/`` but not ``deploy/``); its test therefore runs there too.

Usage: python tools/src_closure.py tools/measure_camera_frames.py
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _module_path(dotted, root=ROOT):
    """The file a dotted name resolves to inside ``root``, or None."""
    base = root.joinpath(*dotted.split("."))
    for cand in (base.with_suffix(".py"), base / "__init__.py"):
        if cand.is_file():
            return cand
    return None


def _package_of(path, root=ROOT):
    """The dotted name of the package containing ``path`` ('' at the top)."""
    rel = path.relative_to(root)
    parts = list(rel.parts[:-1])
    if not parts:
        return ""
    return ".".join(parts)


def imports_of(path):
    """Every module name ``path`` imports, wherever the import sits."""
    tree = ast.parse(path.read_text(), filename=str(path))
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                names.add((alias.name, 0))
        elif isinstance(node, ast.ImportFrom):
            names.add((node.module or "", node.level))
            # ``from ecs_hcm_model.eval import seg`` imports the *module* seg;
            # resolving only the package would drop it.
            for alias in node.names:
                if node.module and alias.name != "*":
                    names.add((f"{node.module}.{alias.name}", node.level))
    return names


def closure(entry, root=ROOT):
    """Every repo file ``entry`` needs, parent packages included, sorted."""
    entry = Path(entry).resolve()
    seen, out = set(), {entry}
    queue = [entry]
    while queue:
        path = queue.pop()
        if path in seen:
            continue
        seen.add(path)
        # A package's __init__ runs before any module inside it.
        pkg = _package_of(path, root)
        for i in range(1, len(pkg.split(".")) + 1):
            init = _module_path(".".join(pkg.split(".")[:i]), root)
            if init is not None and init not in out:
                out.add(init)
                queue.append(init)
        for dotted, level in imports_of(path):
            if level:
                # ``from . import x`` / ``from ..pkg.mod import y``.
                base = _package_of(path, root).split(".")
                base = base[: len(base) - (level - 1)] if level > 1 else base
                dotted = ".".join([p for p in base if p] + ([dotted] if dotted else []))
            if not dotted:
                continue
            found = _module_path(dotted, root)
            if found is None:
                continue
            if found not in out:
                out.add(found)
                queue.append(found)
    return sorted(out)


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv:
        print(__doc__.strip().splitlines()[-1], file=sys.stderr)
        return 2
    for path in closure(argv[0]):
        print(path.relative_to(ROOT))
    return 0


if __name__ == "__main__":
    sys.exit(main())
