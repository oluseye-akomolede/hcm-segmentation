"""The trainer image's venv is split into capped layers, and reassembles intact."""

import ast
import importlib.util
import os
import re
import subprocess
import shutil
import stat
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
DOCKER_DIR = ROOT / "dockerfiles" / "ecs_hcm_trainer"
DOCKERFILE = (DOCKER_DIR / "Dockerfile").read_text()
DOCKERIGNORE = (DOCKER_DIR / "Dockerfile.dockerignore").read_text()

# The entry points the repo documents for the eval-only DAG. The seg-eval spec is
# where they are documented; the test below keeps this list honest against it.
ENTRY_POINTS = ("tools/eval_seg.py", "tools/fetch_seg_val.py")

_spec = importlib.util.spec_from_file_location("split_venv", DOCKER_DIR / "split_venv.py")
split_venv = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(split_venv)


def _write(path, size):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(bytes([len(path.name) % 251]) * size)


def _venv(tmp_path):
    """A small tree shaped like a venv: big libs, small modules, a link, an empty dir."""
    src = tmp_path / "venv"
    _write(src / "lib/site-packages/torch/lib/libtorch_cuda.so", 900)
    _write(src / "lib/site-packages/torch/lib/libcublas.so.12", 600)
    _write(src / "lib/site-packages/torch/__init__.py", 50)
    _write(src / "lib/site-packages/numpy/core.py", 400)
    _write(src / "pyvenv.cfg", 10)
    (src / "bin").mkdir()
    os.symlink("/usr/bin/python3", src / "bin/python")
    (src / "include").mkdir()
    return src


def _snapshot(root):
    """Relative path -> file bytes, link target, or a directory marker."""
    out = {}
    for dirpath, dirs, files in os.walk(root, followlinks=False):
        for name in dirs + files:
            p = Path(dirpath) / name
            rel = str(p.relative_to(root))
            if p.is_symlink():
                out[rel] = ("link", os.readlink(p))
            elif p.is_dir():
                out[rel] = ("dir",)
            else:
                out[rel] = ("file", p.read_bytes())
    return out


def _reassemble(dest, parts, into):
    """What the runtime stage's COPY lines do: lay every part onto one path."""
    into.mkdir()
    for i in range(parts):
        shutil.copytree(dest / f"part{i}", into, symlinks=True, dirs_exist_ok=True)


def _part_bytes(part):
    return sum(
        os.lstat(Path(d) / f).st_size
        for d, _dirs, files in os.walk(part)
        for f in files
    ) + sum(
        os.lstat(Path(d) / x).st_size
        for d, dirs, _files in os.walk(part)
        for x in dirs
        if (Path(d) / x).is_symlink()
    )


class TestSplit:
    def test_the_parts_reassemble_into_the_original_tree(self, tmp_path):
        src = _venv(tmp_path)
        before = _snapshot(src)
        split_venv.split(str(src), str(tmp_path / "parts"), 3, 1000)
        _reassemble(tmp_path / "parts", 3, tmp_path / "back")
        assert _snapshot(tmp_path / "back") == before

    def test_no_part_exceeds_the_cap(self, tmp_path):
        src = _venv(tmp_path)
        totals = split_venv.split(str(src), str(tmp_path / "parts"), 3, 1000)
        assert all(t <= 1000 for t in totals)
        for i in range(3):
            assert _part_bytes(tmp_path / "parts" / f"part{i}") <= 1000
        # The fixture's 1960 bytes cannot fit one part: the split really happened.
        assert sum(1 for t in totals if t) >= 2

    def test_symlinks_move_as_links_and_empty_dirs_survive(self, tmp_path):
        src = _venv(tmp_path)
        split_venv.split(str(src), str(tmp_path / "parts"), 3, 1000)
        _reassemble(tmp_path / "parts", 3, tmp_path / "back")
        assert os.readlink(tmp_path / "back/bin/python") == "/usr/bin/python3"
        assert (tmp_path / "back/include").is_dir()

    def test_directory_modes_survive_the_split(self, tmp_path):
        src = _venv(tmp_path)
        _write(src / "private/secret.py", 10)
        os.chmod(src / "private", 0o700)
        os.chmod(src / "lib/site-packages", 0o750)
        os.chmod(src, 0o711)
        split_venv.split(str(src), str(tmp_path / "parts"), 3, 1000)
        for i in range(3):
            part = tmp_path / "parts" / f"part{i}"
            assert stat.S_IMODE(part.stat().st_mode) == 0o711
            for rel, mode in (("private", 0o700), ("lib/site-packages", 0o750)):
                if (part / rel).is_dir():
                    assert stat.S_IMODE((part / rel).stat().st_mode) == mode, (i, rel)
        _reassemble(tmp_path / "parts", 3, tmp_path / "back")
        assert stat.S_IMODE((tmp_path / "back/private").stat().st_mode) == 0o700
        assert stat.S_IMODE((tmp_path / "back/lib/site-packages").stat().st_mode) == 0o750

    def test_every_part_directory_exists_even_when_empty(self, tmp_path):
        src = tmp_path / "venv"
        _write(src / "a.py", 10)
        totals = split_venv.split(str(src), str(tmp_path / "parts"), 3, 1000)
        assert totals == [10, 0, 0]
        assert all((tmp_path / "parts" / f"part{i}").is_dir() for i in range(3))

    def test_the_plan_is_deterministic(self):
        sizes = [("a", 400), ("b", 400), ("c", 400), ("d", 100)]
        assert split_venv.plan(sizes, 3, 900) == split_venv.plan(sizes, 3, 900)
        assignment, totals = split_venv.plan(sizes, 3, 900)
        assert assignment == {"a": 0, "b": 0, "c": 1, "d": 1}
        assert totals == [800, 500, 0]

    def test_a_single_file_over_the_cap_fails(self, tmp_path):
        with pytest.raises(ValueError, match="over the 100-byte cap"):
            split_venv.plan([("huge.so", 101)], 3, 100)

    def test_a_tree_needing_more_parts_than_allowed_fails(self):
        with pytest.raises(ValueError, match="do not fit in 2 parts"):
            split_venv.plan([("a", 90), ("b", 90), ("c", 90)], 2, 100)

    def test_the_cli_fails_without_moving_anything(self, tmp_path, capsys):
        src = tmp_path / "venv"
        _write(src / "huge.so", 500)
        rc = split_venv.main([str(src), str(tmp_path / "parts"), "--parts", "3", "--cap-bytes", "100"])
        assert rc == 1
        assert "over the 100-byte cap" in capsys.readouterr().err
        assert (src / "huge.so").exists()


class TestDockerfile:
    def _split_args(self):
        m = re.search(r"split_venv\.py /opt/venv /opt/venv-parts --parts (\d+) --cap-bytes (\d+)", DOCKERFILE)
        assert m, "the builder stage must run split_venv.py on /opt/venv"
        return int(m.group(1)), int(m.group(2))

    def test_the_cap_is_under_three_gb(self):
        _parts, cap = self._split_args()
        assert cap <= 3_000_000_000

    def test_the_runtime_stage_copies_every_part_and_only_the_parts(self):
        parts, _cap = self._split_args()
        copies = re.findall(r"^COPY --from=builder --link (\S+) (\S+)$", DOCKERFILE, re.M)
        assert copies == [(f"/opt/venv-parts/part{i}/", "/opt/venv/") for i in range(parts)]

    def test_the_split_script_reaches_the_builder(self):
        assert "COPY dockerfiles/ecs_hcm_trainer/split_venv.py /tmp/split_venv.py" in DOCKERFILE
        ignore = (DOCKER_DIR / "Dockerfile.dockerignore").read_text()
        assert "!dockerfiles/ecs_hcm_trainer/" in ignore

    def test_the_split_runs_after_the_last_venv_install(self):
        assert DOCKERFILE.index("split_venv.py /opt/venv") > DOCKERFILE.rindex("pip install")
        assert DOCKERFILE.index("split_venv.py /opt/venv") < DOCKERFILE.index("as runtime")

    @staticmethod
    def _builder_runs():
        """The builder stage's RUN instructions, continuation lines joined."""
        stage = DOCKERFILE[DOCKERFILE.index("as builder"):DOCKERFILE.index("as runtime")]
        # Docker drops comment lines before it joins continuations.
        stage = "\n".join(l for l in stage.splitlines() if not l.lstrip().startswith("#"))
        joined = re.sub(r"\\\n", " ", stage)
        return [line for line in joined.splitlines() if line.startswith("RUN ")]

    def test_every_venv_install_and_the_split_share_one_run(self):
        # A rename across RUNs copies the file up first: that was the 960 s.
        runs = self._builder_runs()
        split_runs = [r for r in runs if "split_venv.py /opt/venv" in r]
        assert len(split_runs) == 1
        installs = [r for r in runs if "pip install" in r and "--upgrade pip" not in r]
        assert installs == split_runs, "a venv install outside the split's RUN"
        assert "torch==" in split_runs[0] and "mlflow==" in split_runs[0]
        assert split_runs[0].index("mlflow==") < split_runs[0].index("split_venv.py")

    def test_the_pip_cache_is_a_mount_not_a_layer(self):
        (run,) = [r for r in self._builder_runs() if "split_venv.py /opt/venv" in r]
        assert run.startswith("RUN --mount=type=cache,target=/root/.cache/pip ")

    def test_every_tolerated_failure_is_grouped(self):
        # 'a && b || true && c' is '(a && b) || true' then c: a failed install ran on.
        (run,) = [r for r in self._builder_runs() if "split_venv.py /opt/venv" in r]
        assert run.count("|| true") == len(re.findall(r"\{ .*?\|\| true; \}", run))

    @pytest.mark.parametrize("failing", ["torch==", "mlflow=="])
    def test_a_failed_install_fails_the_run_before_the_split(self, failing):
        (run,) = [r for r in self._builder_runs() if "split_venv.py /opt/venv" in r]
        body = run.split(" ", 2)[2]  # drop "RUN --mount=..."
        stubs = (f'pip() {{ case "$*" in *{failing}*) return 1;; esac; }}; '
                 "find() { case \"$*\" in *-exec*) return 1;; esac; }; rm() { :; }; python() { echo SPLIT; }; ")
        r = subprocess.run(["sh", "-c", stubs + body], capture_output=True, text=True)
        assert r.returncode != 0 and "SPLIT" not in r.stdout

    @pytest.mark.parametrize("pattern", ["*.pyc", "*.pyo"])
    def test_a_failed_delete_find_fails_the_run_before_the_split(self, pattern):
        # The -delete finds are not tolerated: a failure there must stop the build too.
        (run,) = [r for r in self._builder_runs() if "split_venv.py /opt/venv" in r]
        body = run.split(" ", 2)[2]
        stubs = ("pip() { :; }; "
                 f'find() {{ case "$*" in *-exec*) return 1;; *"{pattern}"*-delete*) return 1;; esac; }}; '
                 "rm() { :; }; python() { echo SPLIT; }; ")
        r = subprocess.run(["sh", "-c", stubs + body], capture_output=True, text=True)
        assert r.returncode != 0 and "SPLIT" not in r.stdout

    def test_a_clean_install_reaches_the_split(self):
        (run,) = [r for r in self._builder_runs() if "split_venv.py /opt/venv" in r]
        body = run.split(" ", 2)[2]
        stubs = "pip() { :; }; find() { case \"$*\" in *-exec*) return 1;; esac; }; rm() { :; }; python() { echo SPLIT; }; "
        r = subprocess.run(["sh", "-c", stubs + body], capture_output=True, text=True)
        assert r.returncode == 0 and "SPLIT" in r.stdout

    def test_the_torch_lib_first_path_and_libcublas_check_are_intact(self):
        assert "ENV LD_LIBRARY_PATH=/opt/venv/lib/torch-cuda-libs:${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}" in DOCKERFILE
        assert "test -e /opt/venv/lib/torch-cuda-libs/libcublas.so.12" in DOCKERFILE
        # The symlink and check run after the venv is reassembled, not before.
        assert DOCKERFILE.index("torch-cuda-libs/libcublas.so.12") > DOCKERFILE.rindex("/opt/venv-parts/part")


# ---------------------------------------------------------------------------
# The image carries what its documented entry points run from
# ---------------------------------------------------------------------------
#
# Whether a tree ships is one decision made in two places: a COPY line in the
# final stage has to take it, and the ignore file has to leave it in the build
# context. Both halves are read here, the ignore half through a matcher, because
# an entry point that ships without the tree it imports is a run-time failure in
# the image and invisible everywhere else.


def _ignore_patterns(text):
    """``(pattern, is_negation)`` for the live lines, in the order Docker reads them."""
    out = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        negated = line.startswith("!")
        if negated:
            line = line[1:].strip()
        # Docker cleans each pattern and drops a leading slash before matching.
        line = os.path.normpath(line).replace(os.sep, "/")
        if line not in ("", "."):
            out.append((line, negated))
    return out


def _glob_regex(pattern):
    """Go's filepath.Match as a regex, plus Docker's ``**`` (any number of dirs).

    Anchored at the context root, an ignore line and a COPY source alike: ``*.md`` is a
    root-level ``.md`` and ``docs/`` is the root ``docs/``, so depth has to be asked for
    with ``**/``. The gitignore rule for a slash-less pattern -- a match at every depth
    -- does not apply here, which is what this matcher used to assume (and what the
    archived `eval-followups-20261007` design.md still says). Probed on the engine that
    builds this image, Docker 29.8.1 with BuildKit; pinned by
    ``test_ignore_patterns_are_anchored_at_the_context_root``.
    """
    out, i = [], 0
    while i < len(pattern):
        if pattern.startswith("**", i):
            out.append(".*")
            i += 2
            if i < len(pattern) and pattern[i] == "/":
                i += 1
        elif pattern[i] == "*":
            out.append("[^/]*")
            i += 1
        elif pattern[i] == "?":
            out.append("[^/]")
            i += 1
        elif pattern[i] == "[" and "]" in pattern[i + 1:]:
            end = pattern.index("]", i + 1)
            body = pattern[i + 1:end]
            out.append("[" + ("^" + body[1:] if body.startswith("!") else body) + "]")
            i = end + 1
        else:
            out.append(re.escape(pattern[i]))
            i += 1
    return "".join(out)


def _pattern_matches(pattern, path):
    return re.fullmatch(_glob_regex(pattern), path) is not None


def _ignored(path, text=DOCKERIGNORE):
    """Would Docker leave ``path`` in the context?

    Anchored patterns, last matching line wins, and a line matches either the path or
    one of its parent directories -- which is how a directory line takes its whole
    subtree with it. A negation may also name a file *inside* an excluded directory:
    Docker keeps that file. The ``.dockerignore`` docs say the opposite ("it is not
    possible to re-include a file if a parent directory of that file is excluded"),
    and the docs are wrong for the engine that builds this image -- probed on Docker
    29.8.1 (BuildKit) with a scratch ``FROM scratch`` + ``COPY . /ctx/``, ignoring
    ``dir/`` then ``!dir/keep2.txt``: the file came through. So a parent-exclusion
    veto here would be a matcher that disagrees with the build it is checking.
    """
    prefixes = [path]
    parent = os.path.dirname(path)
    while parent:
        prefixes.append(parent)
        parent = os.path.dirname(parent)
    ignored = False
    for pattern, negated in _ignore_patterns(text):
        if any(_pattern_matches(pattern, p) for p in prefixes):
            ignored = not negated
    return ignored


def _runtime_copies():
    """``(source, destination)`` for every COPY the final stage makes from the context."""
    stage = DOCKERFILE[DOCKERFILE.index("as runtime"):]
    copies = []
    for line in stage.splitlines():
        tokens = line.split()
        if not tokens or tokens[0] != "COPY" or any(t.startswith("--from=") for t in tokens):
            continue
        copies.append((tokens[-2], tokens[-1]))
    return copies


def _image_path(path):
    """Where the final stage puts a context path, or None if no COPY takes it.

    The three source forms a COPY uses here: a directory (``tools/``), a single
    file, and a glob (``*.py``).
    """
    for src, dst in _runtime_copies():
        base = dst.rstrip("/")
        if src.endswith("/"):
            if path == src.rstrip("/"):
                return base + "/"
            if path.startswith(src):
                return f"{base}/{path[len(src):]}"
        elif path == src or _pattern_matches(src, path):
            return f"{base}/{path.rsplit('/', 1)[-1]}"
    return None


def _repo_module(dotted):
    """The file in this repo for a dotted module name, or None if it is not ours."""
    path = ROOT.joinpath(*dotted.split("."))
    for candidate in (path.with_suffix(".py"), path / "__init__.py"):
        if candidate.is_file():
            return candidate
    return None


def _imports_in(node):
    if isinstance(node, ast.Import):
        return [alias.name for alias in node.names]
    if isinstance(node, ast.ImportFrom) and not node.level and node.module:
        # `from a.b import c` is either the module a.b.c or the module a.b.
        return [f"{node.module}.{alias.name}" for alias in node.names] + [node.module]
    return []


def _imported_trees(entry_points):
    """``{top-level tree: the module that imports it}`` reachable from ``entry_points``.

    Every import statement in a reachable module counts, so this over-approximates:
    an import only a branch the image never takes would run still counts. That is
    the safe direction, since the failure looked for is a needed tree that is
    absent, not a shipped tree that is unneeded.
    """
    queue = [ROOT / entry for entry in entry_points]
    seen, trees = set(), {}
    while queue:
        path = queue.pop()
        if path in seen:
            continue
        seen.add(path)
        for node in ast.walk(ast.parse(path.read_text(), str(path))):
            for dotted in _imports_in(node):
                target = _repo_module(dotted)
                if target is not None:
                    trees.setdefault(target.relative_to(ROOT).parts[0], path)
                    queue.append(target)
    return trees


def _reached_repo_files(entry_points):
    """Every file of this repository the walk from ``entry_points`` reads, the entry
    points included.

    ``_imported_trees`` collapses each reachable module to its top-level tree, which is
    the unit a COPY takes; this is the same walk without the collapse. The difference
    matters when a *sub*directory is what the context drops: ``**/lib/`` would take a
    reached module out while the tree above it still ships, and the tree check alone
    cannot see that.
    """
    queue, seen = [ROOT / entry for entry in entry_points], set()
    while queue:
        path = queue.pop()
        if path in seen:
            continue
        seen.add(path)
        for node in ast.walk(ast.parse(path.read_text(), str(path))):
            for dotted in _imports_in(node):
                target = _repo_module(dotted)
                if target is not None:
                    queue.append(target)
    return seen


# The install side of the same question: a tree can ship and still not run if a module in
# it imports a package the venv never installed. An import name mostly equals its
# distribution name once PEP 503 normalisation is applied (`pytorch_lightning` and
# `huggingface_hub` do); the ones that differ are named here.
_MODULE_DISTRIBUTIONS = {
    "cv2": "opencv-python-headless",
    "PIL": "pillow",
    "yaml": "pyyaml",
    "sklearn": "scikit-learn",
}


def _distribution_for(module):
    return _MODULE_DISTRIBUTIONS.get(module, re.sub(r"[-_.]+", "-", module).lower())


def _installed_distributions():
    """The distributions the image's venv installs, read from the builder stage.

    The runtime stage installs nothing -- it reassembles the parts the builder split --
    so the builder's pip lines are the whole list. Comment lines are dropped before the
    ``\\`` continuations are joined, because one of them sits inside the install block
    between two package names; that is the order `TestDockerfile._builder_runs` uses.
    """
    stage = DOCKERFILE[DOCKERFILE.index("as builder"):DOCKERFILE.index("as runtime")]
    stage = "\n".join(l for l in stage.splitlines() if not l.lstrip().startswith("#"))
    names = set()
    for line in re.sub(r"\\\n", " ", stage).splitlines():
        for command in line.split("&&"):
            tokens = command.split()
            if "install" not in tokens:
                continue
            at = tokens.index("install")
            if "pip" not in tokens[max(0, at - 2):at]:
                continue  # an apt-get install: an OS package, not a distribution
            for token in tokens[at + 1:]:
                if token.startswith("-") or "://" in token or token.endswith(".txt"):
                    continue  # a flag, an index URL, or a requirements file
                names.add(_distribution_for(re.split(r"[=<>!\[]", token, 1)[0]))
    return names


def _import_sites(path):
    """``[(top-level module, deferred)]`` for one file's import statements.

    ``deferred`` marks an import inside a function body. It does not run when the module
    is imported, so its package being absent is not an import failure of the entry point
    -- which is the difference the requirement turns on. A class body is *not* deferred:
    it runs when the module is imported, so an import there has to be satisfiable inside
    the image like a module-scope one.
    """
    sites = []

    def visit(node, deferred):
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.Import, ast.ImportFrom)):
                for dotted in _imports_in(child):
                    sites.append((dotted.split(".")[0], deferred))
            elif isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                # A function body defers; a class body does not, because it runs when
                # the module is imported. A class node therefore falls through to the
                # ``else`` with the level it is itself at, and only the bodies of its
                # methods defer.
                visit(child, True)
            else:
                visit(child, deferred)

    visit(ast.parse(path.read_text(), str(path)), False)
    return sites


def _declared_packages(path):
    """The ``package=`` names a file declares: what its loader reports when that
    package is not installed, so the tool can name the install instead of crashing."""
    return {
        node.value.value
        for node in ast.walk(ast.parse(path.read_text(), str(path)))
        if isinstance(node, ast.keyword)
        and node.arg == "package"
        and isinstance(node.value, ast.Constant)
    }


def _third_party_import_sites(entry_points):
    """``[(module, file, deferred)]`` for the third-party imports the entry points reach.

    The walk is the one `_imported_trees` does, so the reachable set is the same one the
    tree test uses; the repo's own trees and the standard library are the two things
    that are not third party.
    """
    repo_trees = set(_imported_trees(entry_points))
    repo_trees.update(entry.split("/")[0] for entry in entry_points)
    queue = [ROOT / entry for entry in entry_points]
    seen, sites = set(), []
    while queue:
        path = queue.pop()
        if path in seen:
            continue
        seen.add(path)
        for module, deferred in _import_sites(path):
            if module in sys.stdlib_module_names or module in repo_trees:
                continue
            sites.append((module, path, deferred))
        for node in ast.walk(ast.parse(path.read_text(), str(path))):
            for dotted in _imports_in(node):
                target = _repo_module(dotted)
                if target is not None:
                    queue.append(target)
    return sites


class TestTheImageShipsItsEntryPoints:
    def test_the_checked_entry_points_are_the_documented_ones(self):
        """The list above is checked against the spec that documents the tools, so
        a third documented entry point cannot be added without being covered."""
        spec = (ROOT / "openspec" / "specs" / "seg-eval" / "spec.md").read_text()
        assert sorted(set(re.findall(r"tools/[A-Za-z0-9_]+\.py", spec))) == sorted(ENTRY_POINTS)

    @pytest.mark.parametrize("entry", ENTRY_POINTS)
    def test_a_documented_entry_point_ships_where_its_launcher_looks_for_it(self, entry):
        wanted = f"/app/{entry}"
        where = _image_path(entry)
        assert where == wanted, f"{entry} is copied to {where}; its launcher needs {wanted}"
        assert not _ignored(entry), f"the build context drops {entry}"
        assert (ROOT / entry).is_file()

    def test_the_trees_a_documented_entry_point_imports_ship_too(self):
        """The entry point is a launcher: what it needs is the trees beside it. An
        import of a tree the image does not carry fails inside the container.

        The trees are what a COPY names, so they are checked first; then the walk's own
        files, because a tree can ship and still not run when what the context drops is a
        subdirectory between the entry point and the module inside it.
        """
        trees = _imported_trees(ENTRY_POINTS)
        complaints = []
        for tree, importer in sorted(trees.items()):
            source = tree + "/" if (ROOT / tree).is_dir() else tree
            who = f"{tree} (imported by {importer.relative_to(ROOT)})"
            where = _image_path(source)
            if where != f"/app/{source}":
                complaints.append(f"{who} lands at {where}, not /app/{source}")
            elif _ignored(tree):
                complaints.append(f"{who} is dropped by the build context")
        reached = {path.relative_to(ROOT).as_posix() for path in _reached_repo_files(ENTRY_POINTS)}
        assert len(reached) > len(ENTRY_POINTS), "the walk reached nothing beyond the launchers"
        for source in sorted(reached):
            if _ignored(source):
                complaints.append(
                    f"{source} is reached from a documented entry point but the build context drops it"
                )
        assert not complaints, "a tree a documented entry point imports is not in the image: " + "; ".join(complaints)
        # The walk reached something, so the loop above is not vacuous: the eval
        # entry point imports these two. (tools/ is where the entry points live,
        # not something they import, so it is covered by the test above.)
        assert set(trees) >= {"ecs_hcm_model", "jetson"}

    def test_an_import_in_a_class_body_is_not_deferred(self, tmp_path):
        """A class body runs when the module is imported, so an import there has to be
        satisfiable inside the image; only a function body defers.

        The flag is what excuses the one uninstalled package the entry points reach (the
        test below), so reading a class body as deferred would excuse an import the image
        cannot satisfy.
        """
        source = tmp_path / "synthetic_levels.py"
        source.write_text(
            "import at_module_scope\n"
            "class C:\n"
            "    import in_class_body\n"
            "    class Inner:\n"
            "        import in_nested_class_body\n"
            "    def m(self):\n"
            "        import in_method_body\n"
            "def f():\n"
            "    import in_function_body\n"
        )
        assert dict(_import_sites(source)) == {
            "at_module_scope": False,
            "in_class_body": False,
            "in_nested_class_body": False,
            "in_method_body": True,
            "in_function_body": True,
        }

    def test_every_package_the_entry_points_reach_is_installed_or_declared(self):
        """The requirement's last clause -- the entry points, and everything reachable
        from them, must not import a package the image never installs -- checked against
        the distributions the image's own pip lines install.

        Two ways to meet it, and both are asserted. Either the module maps to an
        installed distribution (`_distribution_for` turns an import name into the
        distribution name), or, for a package the image deliberately does not install,
        the module defers the import into a function *and* declares the package, so the
        tool reports the install command instead of dying on the import. Read literally
        as "the reachable modules contain no such import statement", the sentence was
        false of `ecs_hcm_model/eval/baselines.py` -- it imports `ultralytics` on demand,
        the AGPL-3.0 baseline package the image deliberately does not ship -- so this
        change's spec delta says which half is the observable property (an import at
        module scope of an uninstalled package is the failure; a deferred one has to be
        declared) instead of leaving a MUST the code contradicts. See design.md, "The
        reachable-import clause".
        """
        installed = _installed_distributions()
        sites = _third_party_import_sites(ENTRY_POINTS)
        # Both halves have to be real: the list was read off the Dockerfile, and the walk
        # reached the packages the entry points actually use.
        assert installed >= {
            "torch", "pillow", "pytorch-lightning", "opencv-python-headless",
        }
        assert {module for module, _, _ in sites} >= {"torch", "PIL", "datasets"}
        complaints = []
        for module, path, deferred in sites:
            if _distribution_for(module) in installed:
                continue
            where = path.relative_to(ROOT)
            if not deferred:
                complaints.append(
                    f"{where} imports {module} at module scope, and the image installs "
                    f"no {_distribution_for(module)}"
                )
            elif module not in _declared_packages(path):
                complaints.append(
                    f"{where} imports {module}, which the image does not install, "
                    f"without declaring it"
                )
        assert not complaints, "; ".join(complaints)
        # Non-vacuity for the second half: with no uninstalled package in the closure,
        # the deferral and declaration checks would pass over nothing at all.
        assert {module for module, _, deferred in sites if deferred} >= {"ultralytics"}

    def test_a_root_glob_does_not_claim_a_nested_file(self):
        """A COPY source stops at a slash: a nested .py ships because the COPY of its
        directory takes it, not because of ``*.py``."""
        assert _pattern_matches("*.py", "main.py")
        assert not _pattern_matches("*.py", "tools/eval_seg.py")

    def test_ignore_patterns_are_anchored_at_the_context_root(self):
        """A slash-less ignore line is a root-level name, not a match at every depth.

        Docker 29.8.1, scratch build, ignore file ``__pycache__/`` + ``*.pyc`` +
        ``*.md`` + ``docs/``: root ``x.pyc`` and ``__pycache__/y.pyc`` were dropped,
        while ``a/__pycache__/x.pyc``, ``a/b/notes.md`` and ``a/docs/readme.md`` all
        shipped. So the first two lines reach nothing below the root, and a tree
        needing depth has to say ``**/``.
        """
        text = "__pycache__/\n*.pyc\n*.md\ndocs/\n"
        assert _ignored("x.pyc", text)
        assert _ignored("__pycache__/y.pyc", text)
        assert _ignored("docs/RUNBOOK.md", text)
        assert not _ignored("a/__pycache__/x.pyc", text)
        assert not _ignored("a/b/notes.md", text)
        assert not _ignored("a/docs/readme.md", text)
        # ``**/`` is the form that asks for depth, and the repository's own file now
        # carries it for the cache lines: the same nested paths are dropped, while the
        # lines that did not get the form (here ``*.md`` and ``docs/``) still reach
        # nothing below the root.
        deep = "**/__pycache__/\n**/*.pyc\n*.md\ndocs/\n"
        assert _ignored("a/__pycache__/x.pyc", deep)
        assert _ignored("a/b/x.pyc", deep)
        assert not _ignored("a/b/notes.md", deep)
        assert not _ignored("a/docs/readme.md", deep)

    def test_a_negation_reaches_inside_an_excluded_directory(self):
        """``!dir/keep2.txt`` keeps that file even though ``dir/`` excluded its parent.

        This is the case that separates a matcher which models Docker from one which
        models the docs' parent-exclusion rule (see ``_ignored``); both agree on the
        two negations this repo actually ships, since both name directories, so the
        disagreement needs a synthetic ignore file to be visible at all.
        """
        text = "dir/\n!dir/keep2.txt\n!dir/sub/\n"
        assert _ignored("dir/child.txt", text)
        assert not _ignored("dir/keep2.txt", text)
        assert not _ignored("dir/sub/deep.txt", text)

    def test_the_ignore_matcher_is_docker_s_own_rules(self):
        """The matcher decides the assertions above, so it is pinned: it drops what
        the context drops, keeps what it keeps, and honours a negation."""
        for path, ignored in (
            ("tests/test_split_venv.py", True),          # a bare directory pattern
            ("docs/RUNBOOK.md", True),                   # docs/ takes its subtree
            # The depth forms are what reach these: ``**/__pycache__/``,
            # ``**/build/`` and ``**/*.py[cod]`` / ``**/*.pth``. With only the bare
            # lines they shipped -- the anchoring test above shows why, and change
            # trainer-image-ignore-depth (tasks.md 3.1) is where the forms were added.
            ("ecs_hcm_model/__pycache__/x.pyc", True),
            ("ecs_hcm_model/nested/build/leftover.so", True),
            ("ecs_hcm_model/model.pth", True),
            ("tools/__pycache__/z.pyc", True),
            # The directory entries themselves. These are what the directory forms
            # are for: an empty __pycache__/ or build/ still rides the COPY, and the
            # extension forms alone would leave both.
            ("ecs_hcm_model/__pycache__", True),
            ("ecs_hcm_model/nested/build", True),
            # One case per remaining depth form, each in a subtree no *other* line
            # drops. That is what a form needs to be pinned: a .pyc under
            # __pycache__/ or a .so under build/ is dropped by the directory line
            # whatever the extension line says, so those cases cannot tell whether
            # ``**/*.py[cod]`` / ``**/*.so`` is there at all.
            ("ecs_hcm_model/nested/loose.pyc", True),         # **/*.py[cod]
            ("ecs_hcm_model/nested/loose2.so", True),         # **/*.so
            ("ecs_hcm_model/weights/z.pt", True),             # **/*.pt
            ("ecs_hcm_model/weights/w.onnx", True),           # **/*.onnx
            ("ecs_hcm_model/weights/v.torchscript", True),    # **/*.torchscript
            ("tmpdir/t.tmp", True),                           # **/*.tmp
            ("tmpdir/t.temp", True),                          # **/*.temp
            # The two directory entries that only the depth form takes.
            ("ecs_hcm_model/nested/lib", True),               # **/lib/
            ("ecs_hcm_model/cache/.cache", True),             # **/.cache/
            ("K8s_Training/train.yaml", True),
            ("tools/eval_seg.py", False),
            ("jetson/input_size.py", False),
            ("dockerfiles/ecs_hcm_trainer/split_venv.py", False),  # ! re-includes it
            ("dockerfiles/base_image_hf_minio/Dockerfile", True),
        ):
            assert _ignored(path) is ignored, path

