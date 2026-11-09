"""Promote a checkpoint to the Jetson demo in three steps: stage, accept, apply.

    stage  <bucket>/<prefix>
        Copy model.pth + metadata.json from MinIO into
        ~/models_staging/<prefix basename>, with a sha256 manifest. MinIO is
        named by MINIO_ENDPOINT, MINIO_USER and MINIO_PASSWORD (MINIO_SECURE=1
        for https) from the environment; none of them is ever printed, and a
        failed fetch is reported without the endpoint or the user.

    accept <staged dir> --data-root <coco val root> [--models-dir ~/models]
        Stop the demo, score the candidate and the baseline with
        ecs_hcm_model.eval.seg on COCO val (each at its own recorded input
        size), restart the demo exactly as it was running, and print PASS only
        if the candidate's IoU is strictly higher. Exit 0 PASS, 1 FAIL, 2 if it
        cannot decide (no score, or the two failed on different image counts).

    apply  <staged dir> --code-dir <repo dir> --yes [--models-dir ~/models]
        Stop the demo, move ~/models to ~/models_prev_<UTC stamp> (never
        deleted), put the staged files in place, start jetson/run_demo.sh from
        --code-dir, and require HTTP 200 plus a first stream part on
        :5000/video. Prints a rollback command.

Run it on the Jetson from a v2-aware checkout:
    PYTHONPATH=$PWD .venv/bin/python jetson/promote_model.py <step> ...

Everything that touches the machine goes through ``Host``, so the steps are
tested on the VM with fakes.
"""

import argparse
import datetime
import hashlib
import json
import os
import re
import shlex
import shutil
import signal
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

# The bracket keeps the pattern from matching any command line that contains
# the pattern itself (a shell running `pkill -f ...` over ssh would otherwise
# match and kill itself).
DEMO_PATTERN = r"[r]ealtime_demo\.py"
DEMO_URL = "http://127.0.0.1:5000/video"
CHECKPOINT_FILES = ("model.pth", "metadata.json")
MANIFEST = "STAGED.json"
# Environment carried into the printed rollback command. Only what the demo
# needs; the full environment of the old process may hold secrets.
RELAUNCH_ENV = ("PYTHONPATH", "PYTORCH_NO_CUDA_MEMORY_CACHING")
RELAUNCH_ENV_PREFIX = "ECS_HCM_"
REPO_ROOT = Path(__file__).resolve().parents[1]


class PromoteError(Exception):
    """A refusal or failure, reported as one line and a non-zero exit."""


def _default_minio(endpoint, access_key, secret_key, secure):
    from minio import Minio
    return Minio(endpoint, access_key=access_key, secret_key=secret_key, secure=secure)


def _default_http_get(url, timeout):
    """``(status, first body bytes)``. The demo sends its status line only
    with the first frame, so a non-empty first chunk means a forward ran."""
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        return resp.status, resp.read1(64)


class Host:
    """The machine: processes, /proc, HTTP, MinIO, clock. Tests replace parts."""

    def __init__(self, home=None, proc_root="/proc", env=None, run=subprocess.run,
                 popen=subprocess.Popen, kill=os.kill, http_get=_default_http_get,
                 minio=_default_minio, sleep=time.sleep, clock=time.monotonic,
                 utcnow=None, out=None):
        self.home = Path(home) if home is not None else Path.home()
        self.proc_root = Path(proc_root)
        self.env = os.environ if env is None else env
        self.run = run
        self.popen = popen
        self.kill = kill
        self.http_get = http_get
        self.minio = minio
        self.sleep = sleep
        self.clock = clock
        self.utcnow = utcnow or (lambda: datetime.datetime.now(datetime.timezone.utc))
        self.out = out or (lambda *a: print(*a, flush=True))

    def stamp(self):
        return self.utcnow().strftime("%Y%m%dT%H%M%SZ")

    def alive(self, pid):
        return (self.proc_root / str(pid)).exists()


# ---------------------------------------------------------------- demo process

class DemoProcess:
    """A running demo, captured well enough to start it again as it was."""

    def __init__(self, pid, argv, cwd, env):
        self.pid, self.argv, self.cwd, self.env = pid, argv, cwd, env

    def relaunch_env(self):
        return {k: v for k, v in self.env.items()
                if k in RELAUNCH_ENV or k.startswith(RELAUNCH_ENV_PREFIX)}

    def shell_command(self, log):
        """The command line that restarts this demo, for a human to paste."""
        env = " ".join(f"{k}={shlex.quote(v)}" for k, v in sorted(self.relaunch_env().items()))
        argv = " ".join(shlex.quote(a) for a in self.argv)
        return f"cd {shlex.quote(self.cwd)} && env {env} nohup {argv} >> {shlex.quote(str(log))} 2>&1 &"


def _demo_pids(host):
    r = host.run(["pgrep", "-f", DEMO_PATTERN], capture_output=True, text=True)
    return [int(p) for p in r.stdout.split()]


def _read_demo(host, pid):
    d = host.proc_root / str(pid)
    try:
        argv = [a for a in (d / "cmdline").read_bytes().decode().split("\0") if a]
        cwd = os.readlink(d / "cwd")
        raw = (d / "environ").read_bytes().decode(errors="replace").split("\0")
    except OSError as e:
        if not d.exists():
            return None  # exited between pgrep and the read
        # Still there but unreadable (another user's process): treating it as
        # "no demo" would let accept/apply run beside it.
        host.out(f"WARNING: pgrep matched pid {pid} but /proc/{pid} is unreadable "
                 f"({type(e).__name__})")
        raise PromoteError(f"cannot read pid {pid}, which matches the demo pattern; "
                           f"stop it by hand (as its owner) and rerun") from e
    env = dict(kv.split("=", 1) for kv in raw if "=" in kv)
    return DemoProcess(pid, argv, cwd, env)


def find_demo(host):
    """The running demo's python process, or None. More than one is refused:
    which of them would be restarted is a guess."""
    found = []
    for pid in _demo_pids(host):
        p = _read_demo(host, pid)
        # pgrep also matches a wrapping shell; the demo is the python process.
        if p and p.argv and re.match(r"python", Path(p.argv[0]).name):
            found.append(p)
    if len(found) > 1:
        raise PromoteError(f"{len(found)} demo processes are running "
                           f"(pids {[p.pid for p in found]}); stop the extras first")
    return found[0] if found else None


def stop_demo(host, demo, timeout=30.0):
    if demo is None:
        return
    host.out(f"stopping demo pid {demo.pid}")
    host.kill(demo.pid, signal.SIGTERM)
    deadline = host.clock() + timeout
    while host.alive(demo.pid):
        if host.clock() >= deadline:
            host.out(f"demo pid {demo.pid} ignored SIGTERM for {timeout:.0f}s; killing")
            host.kill(demo.pid, signal.SIGKILL)
            break
        host.sleep(0.5)


def wait_for_stream(host, pid, url=DEMO_URL, timeout=180.0):
    """True once ``url`` answers 200 with a first stream part. False if the
    demo process exits or the timeout passes first."""
    deadline = host.clock() + timeout
    while host.clock() < deadline:
        if pid is not None and not host.alive(pid):
            host.out(f"demo pid {pid} exited before serving a frame")
            return False
        try:
            status, chunk = host.http_get(url, 5)
            if status == 200 and chunk:
                host.out(f"stream OK: HTTP {status} with a first frame from {url}")
                return True
            host.out(f"stream not ready: HTTP {status}")
        except OSError:
            pass  # not listening yet, or no frame yet
        host.sleep(2.0)
    host.out(f"no HTTP 200 with a frame from {url} within {timeout:.0f}s")
    return False


def restart_as_before(host, demo, check_timeout):
    """Start ``demo`` again with its own argv, working dir and environment."""
    log = host.home / f"realtime_demo_restart_{host.stamp()}.log"
    with open(log, "ab") as fh:
        proc = host.popen(demo.argv, cwd=demo.cwd, env=demo.env, stdout=fh,
                          stderr=subprocess.STDOUT, start_new_session=True)
    host.out(f"demo restarted from {demo.cwd}: pid {proc.pid}, log {log}")
    return wait_for_stream(host, proc.pid, timeout=check_timeout)


# ---------------------------------------------------------------- staged files

def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def verify_staged(staged):
    """The manifest, after checking every staged file against it."""
    staged = Path(staged)
    try:
        manifest = json.loads((staged / MANIFEST).read_text())
    except (OSError, ValueError) as e:
        raise PromoteError(f"{staged} is not a staged checkpoint ({MANIFEST}: {e})") from e
    for name in CHECKPOINT_FILES:
        want = manifest["files"][name]["sha256"]
        if not (staged / name).is_file() or _sha256(staged / name) != want:
            raise PromoteError(f"{staged / name} does not match its staged sha256")
    return manifest


# ----------------------------------------------------------------------- steps

def cmd_stage(host, spec):
    from ecs_hcm_model.core.model_utils import parse_checkpoint_spec
    try:
        bucket, prefix = parse_checkpoint_spec(spec)
    except ValueError as e:
        raise PromoteError(str(e)) from e
    missing = [k for k in ("MINIO_ENDPOINT", "MINIO_USER", "MINIO_PASSWORD") if not host.env.get(k)]
    if missing:
        raise PromoteError(f"set {', '.join(missing)} in the environment")
    dest = host.home / "models_staging" / Path(prefix).name
    if dest.exists():
        raise PromoteError(f"{dest} already exists; remove it or stage elsewhere")
    try:
        client = host.minio(host.env["MINIO_ENDPOINT"], host.env["MINIO_USER"],
                            host.env["MINIO_PASSWORD"],
                            host.env.get("MINIO_SECURE", "").lower() in ("1", "true", "yes"))
    except ValueError as e:  # minio rejects a malformed endpoint ("invalid port", ...)
        raise PromoteError(f"MINIO_ENDPOINT is not usable: {_redact(str(e), host.env)}") from None
    partial = dest.with_name(dest.name + ".partial")
    shutil.rmtree(partial, ignore_errors=True)
    partial.mkdir(parents=True)
    try:
        files = {}
        for name in CHECKPOINT_FILES:
            try:
                client.fget_object(bucket, f"{prefix}/{name}", str(partial / name))
            except Exception as e:  # minio's S3Error, or the connection failing
                raise PromoteError(f"could not fetch {bucket}/{prefix}/{name}: "
                                   f"{_fetch_error(e, host.env)}") from None
            files[name] = {"sha256": _sha256(partial / name),
                           "bytes": (partial / name).stat().st_size}
        try:
            json.loads((partial / "metadata.json").read_text())
        except ValueError as e:
            raise PromoteError(f"{bucket}/{prefix}/metadata.json does not parse: {e}") from e
        (partial / MANIFEST).write_text(json.dumps({
            "source": f"{bucket}/{prefix}",
            "staged_at": host.utcnow().isoformat(),
            "files": files,
        }, indent=2) + "\n")
        partial.rename(dest)
    except BaseException:
        shutil.rmtree(partial, ignore_errors=True)
        raise
    for name, f in files.items():
        host.out(f"staged {dest / name}  sha256 {f['sha256']}  {f['bytes']} bytes")
    host.out(f"next: accept {dest} --data-root <coco val root>")
    return 0


def _fetch_error(e, env):
    """A fetch failure, described without the endpoint or the user.

    S3Error's own fields (code, message) name the object, not the host. Any
    other error, e.g. urllib3's MaxRetryError, puts host:port in its text, so
    only its type is reported; the redaction is a second guard on both."""
    if type(e).__name__ == "S3Error":
        text = f"S3Error {getattr(e, 'code', '?')}: {getattr(e, 'message', '')}"
    else:
        text = type(e).__name__
    return _redact(text, env)


def _redact(text, env):
    """``text`` with the endpoint, its host, the user and the password removed."""
    endpoint = env.get("MINIO_ENDPOINT", "")
    for secret in (endpoint, endpoint.rsplit(":", 1)[0], env.get("MINIO_USER", ""),
                   env.get("MINIO_PASSWORD", "")):
        if secret:
            text = text.replace(secret, "<redacted>")
    return text


def _score(host, data_root, checkpoints, python):
    """``{checkpoint path: eval result}`` from one eval.seg run."""
    env = dict(host.env)
    env["PYTHONPATH"] = os.pathsep.join(p for p in (str(REPO_ROOT), env.get("PYTHONPATH")) if p)
    env["PYTORCH_NO_CUDA_MEMORY_CACHING"] = "1"  # NVML assert on this device otherwise
    cmd = [python, "-m", "ecs_hcm_model.eval.seg", "--dataset", "coco",
           "--data-root", str(data_root), "--json", *map(str, checkpoints)]
    host.out("scoring: " + " ".join(shlex.quote(c) for c in cmd))
    r = host.run(cmd, stdout=subprocess.PIPE, text=True, env=env)
    results = {}
    for line in (r.stdout or "").splitlines():
        try:
            row = json.loads(line)
        except ValueError:
            continue
        if isinstance(row, dict) and "checkpoint" in row:
            results[row["checkpoint"]] = row
    return results


def cmd_accept(host, staged, data_root, models_dir, python, check_timeout):
    staged = Path(staged)
    verify_staged(staged)
    candidate, baseline = staged / "model.pth", Path(models_dir) / "model.pth"
    if not baseline.is_file():
        raise PromoteError(f"no baseline at {baseline}")
    demo = find_demo(host)
    stop_demo(host, demo)
    try:
        results = _score(host, data_root, (candidate, baseline), python)
    finally:
        if demo is None:
            host.out("the demo was not running before; not starting it")
        elif not restart_as_before(host, demo, check_timeout):
            host.out("WARNING: the demo did not come back; restart it by hand: "
                     + demo.shell_command(host.home / "realtime_demo.log"))
    cand, base = results.get(str(candidate)), results.get(str(baseline))
    if cand is None or base is None:
        missing = [str(p) for p, r in ((candidate, cand), (baseline, base)) if r is None]
        raise PromoteError(f"eval produced no score for {', '.join(missing)}")
    for label, path, r in (("candidate", candidate, cand), ("baseline ", baseline, base)):
        host.out(f"{label} {path}: iou={r['iou']:.2f} at input_size={r['input_size']} "
                 f"n_images={r.get('n_images')} n_failed={r.get('n_failed')}")
    if "n_failed" not in cand or "n_failed" not in base:
        raise PromoteError("eval did not report n_failed; cannot tell the IoUs cover the same images")
    if cand["n_failed"] != base["n_failed"]:
        # The means are then over different image sets; neither PASS nor FAIL is sound.
        raise PromoteError(f"the checkpoints failed on different numbers of images "
                           f"(candidate {cand['n_failed']}, baseline {base['n_failed']}); "
                           f"their IoUs are not comparable")
    if cand["iou"] > base["iou"]:
        host.out(f"PASS: candidate beats the baseline by {cand['iou'] - base['iou']:.2f} IoU")
        return 0
    host.out("FAIL: candidate does not beat the baseline")
    return 1


def _check_code_dir(code_dir):
    code_dir = Path(code_dir)
    needed = {
        "jetson/run_demo.sh": "the launcher",
        "jetson/demo_model.py": "the v2-aware shared loader (older checkouts cannot load v2)",
        ".venv/bin/python": "run_demo.sh's interpreter (a fresh code dir has none; symlink a venv)",
    }
    missing = [f"{rel} ({why})" for rel, why in needed.items() if not (code_dir / rel).exists()]
    if missing:
        raise PromoteError(f"--code-dir {code_dir} is missing: " + "; ".join(missing))


def cmd_apply(host, staged, code_dir, models_dir, yes, check_timeout):
    if not yes:
        raise PromoteError("apply replaces the demo's weights and restarts it; pass --yes")
    staged, code_dir, models_dir = Path(staged), Path(code_dir).resolve(), Path(models_dir)
    _check_code_dir(code_dir)
    manifest = verify_staged(staged)
    stamp = host.stamp()
    prev = models_dir.with_name(f"{models_dir.name}_prev_{stamp}")
    failed = models_dir.with_name(f"{models_dir.name}_failed_{stamp}")
    if prev.exists():
        raise PromoteError(f"{prev} already exists")

    demo = find_demo(host)
    stop_demo(host, demo)
    if models_dir.exists():
        models_dir.rename(prev)
        host.out(f"kept the old weights at {prev}")

    # Printed right after the rename, before anything else can fail (a copy
    # running out of disk, the restart), so it is on screen whatever follows.
    # The new dir may not exist yet if the failure came first, hence the test.
    md, fd = shlex.quote(str(models_dir)), shlex.quote(str(failed))
    restore = f"{{ [ ! -e {md} ] || mv {md} {fd}; }}"
    if prev.exists():
        restore += f" && mv {shlex.quote(str(prev))} {md}"
    relaunch = (demo.shell_command(Path(demo.cwd) / "realtime_demo_rollback.log") if demo
                else "# no demo was running before apply; nothing to relaunch")
    host.out("rollback:\n"
             f"  pkill -f '{DEMO_PATTERN}'; {restore}\n"
             f"  {relaunch}")

    models_dir.mkdir(parents=True)
    for name in CHECKPOINT_FILES:
        shutil.copy2(staged / name, models_dir / name)
    (models_dir / "PROMOTED.json").write_text(json.dumps(
        {**manifest, "promoted_at": host.utcnow().isoformat(), "code_dir": str(code_dir)},
        indent=2) + "\n")
    host.out(f"installed {manifest['source']} into {models_dir}")

    r = host.run([str(code_dir / "jetson" / "run_demo.sh")], cwd=str(code_dir),
                 capture_output=True, text=True)
    host.out((r.stdout or "").rstrip())
    m = re.search(r"pid (\d+)", r.stdout or "")
    if r.returncode != 0 or m is None:
        host.out(f"run_demo.sh failed (exit {r.returncode}): {(r.stderr or '').strip()}")
        return 1
    if not wait_for_stream(host, int(m.group(1)), timeout=check_timeout):
        host.out(f"demo is not serving; see {code_dir / 'run_demo.log'} and roll back")
        return 1
    return 0


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = p.add_subparsers(dest="step", required=True)
    s = sub.add_parser("stage", help="copy a checkpoint from MinIO into ~/models_staging")
    s.add_argument("spec", help="<bucket>/<prefix>, e.g. coco-models/hcm_coco_seg_best/<stamp>")
    for name, helptext in (("accept", "score candidate vs baseline; PASS only if better"),
                           ("apply", "install a staged checkpoint and restart the demo")):
        a = sub.add_parser(name, help=helptext)
        a.add_argument("staged", help="a directory made by `stage`")
        a.add_argument("--models-dir", default=str(Path.home() / "models"),
                       help="the demo's artifact dir (default ~/models)")
        a.add_argument("--check-timeout", type=float, default=180.0,
                       help="seconds to wait for the restarted demo's stream")
    sub.choices["accept"].add_argument("--data-root", required=True,
                                       help="COCO val root as tools/fetch_seg_val.py lays it out")
    sub.choices["accept"].add_argument("--python", default=sys.executable,
                                       help="interpreter for the eval (default: this one)")
    sub.choices["apply"].add_argument("--code-dir", required=True,
                                      help="checkout to run jetson/run_demo.sh from")
    sub.choices["apply"].add_argument("--yes", action="store_true",
                                      help="confirm replacing the demo's weights")
    return p.parse_args(argv)


def main(argv=None, host=None):
    args = parse_args(argv)
    host = host or Host()
    try:
        if args.step == "stage":
            return cmd_stage(host, args.spec)
        if args.step == "accept":
            return cmd_accept(host, args.staged, args.data_root, args.models_dir,
                              args.python, args.check_timeout)
        return cmd_apply(host, args.staged, args.code_dir, args.models_dir,
                         args.yes, args.check_timeout)
    except PromoteError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
