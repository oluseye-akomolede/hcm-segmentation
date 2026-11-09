"""Split a directory tree into a fixed number of parts, each under a byte cap.

The trainer image used to COPY /opt/venv as one 5 GB layer, and Harbor's
verification of a blob that size outlasts docker's ~47 s PUT timeout, so pushes
needed several tries. The builder stage runs this to move the venv into
``<dest>/part0 .. part<N-1>``; the runtime stage COPYs each part onto the same
path, so the tree is reassembled but no single layer exceeds the cap.

Files are assigned in sorted path order, filling one part before starting the
next, so the split is deterministic and keeps neighbouring files together.
Symlinks move as links (their own size, not their target's). Directories are
recreated in whichever part needs them; empty ones go to part0 so they survive.
Every part directory is created even if it ends up empty, because the
Dockerfile COPYs a fixed list of them.

Fails, rather than shipping an oversized layer, if a single file exceeds the
cap or the tree needs more parts than allowed.

Usage: split_venv.py SRC DEST --parts N --cap-bytes B
"""

import argparse
import os
import stat
import sys


def _entries(src):
    """Every file, symlink and empty directory under ``src``, as sorted relative paths."""
    files, empty_dirs = [], []
    for root, dirs, names in os.walk(src, followlinks=False):
        # os.walk lists symlinks to directories under ``dirs``; they move as links.
        links = [d for d in dirs if os.path.islink(os.path.join(root, d))]
        dirs[:] = sorted(d for d in dirs if d not in links)
        rel_root = os.path.relpath(root, src)
        for name in names + links:
            files.append(os.path.normpath(os.path.join(rel_root, name)))
        if not dirs and not names and not links and rel_root != ".":
            empty_dirs.append(rel_root)
    return sorted(files), sorted(empty_dirs)


def plan(sizes, parts, cap):
    """Assign ``(path, size)`` pairs, in order, to part indices under ``cap`` bytes each."""
    assignment, totals = {}, [0] * parts
    current = 0
    for path, size in sizes:
        if size > cap:
            raise ValueError(f"{path} is {size} bytes, over the {cap}-byte cap on its own")
        if totals[current] + size > cap:
            current += 1
            if current >= parts:
                total = sum(s for _p, s in sizes)
                raise ValueError(
                    f"{total} bytes do not fit in {parts} parts of {cap} bytes"
                )
        assignment[path] = current
        totals[current] += size
    return assignment, totals


def split(src, dest, parts, cap):
    files, empty_dirs = _entries(src)
    sizes = [(p, os.lstat(os.path.join(src, p)).st_size) for p in files]
    assignment, totals = plan(sizes, parts, cap)
    for i in range(parts):
        os.makedirs(os.path.join(dest, f"part{i}"), exist_ok=True)
    for path in files:
        target = os.path.join(dest, f"part{assignment[path]}", path)
        os.makedirs(os.path.dirname(target), exist_ok=True)
        os.rename(os.path.join(src, path), target)
    for path in empty_dirs:
        os.makedirs(os.path.join(dest, "part0", path), exist_ok=True)
    _copy_dir_modes(src, dest, parts)
    return totals


def _copy_dir_modes(src, dest, parts):
    """Give every part's copy of a directory the source directory's mode.

    makedirs applies the build umask, so a 0700 directory would come back 0755
    once the parts are copied onto one path. The source directories are still
    there (only their contents moved), so read the mode from each. Deepest
    first, so a restrictive parent cannot block reaching its children.
    """
    dirs = [d for d, _sub, _files in os.walk(src)]
    for d in sorted(dirs, key=lambda p: p.count(os.sep), reverse=True):
        mode = stat.S_IMODE(os.lstat(d).st_mode)
        rel = os.path.relpath(d, src)
        for i in range(parts):
            target = os.path.normpath(os.path.join(dest, f"part{i}", rel))
            if os.path.isdir(target) and not os.path.islink(target):
                os.chmod(target, mode)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("src")
    parser.add_argument("dest")
    parser.add_argument("--parts", type=int, required=True)
    parser.add_argument("--cap-bytes", type=int, required=True)
    args = parser.parse_args(argv)
    try:
        totals = split(args.src, args.dest, args.parts, args.cap_bytes)
    except ValueError as exc:
        print(f"split_venv: {exc}", file=sys.stderr)
        return 1
    for i, total in enumerate(totals):
        print(f"part{i}: {total} bytes")
    return 0


if __name__ == "__main__":
    sys.exit(main())
