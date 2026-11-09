"""jetson/promote_model.py: stage, accept and apply, against fakes for MinIO,
subprocess, /proc, HTTP and the clock. Nothing here touches a real process,
port or bucket."""

import json
import re
import signal
from pathlib import Path

import pytest

from jetson import promote_model as pm

SECRET = "s3cr3t-password-value"
DEMO_ARGV = [".venv/bin/python", "-u", "jetson/realtime_demo.py"]
DEMO_ENV = {"PYTHONPATH": "/home/u/repos/old", "PYTORCH_NO_CUDA_MEMORY_CACHING": "1",
            "ECS_HCM_DEMO_ART_DIR": "/home/u/models", "MINIO_PASSWORD": SECRET}


class FakeMinio:
    def __init__(self, objects):
        self.objects = objects
        self.opened_with = None

    def __call__(self, endpoint, access_key, secret_key, secure):
        self.opened_with = (endpoint, access_key, secret_key, secure)
        return self

    def fget_object(self, bucket, key, path):
        if (bucket, key) not in self.objects:
            raise RuntimeError(f"NoSuchKey {key}")
        Path(path).write_bytes(self.objects[bucket, key])


class FakeHost(pm.Host):
    """A Host whose processes are directories under a fake /proc."""

    def __init__(self, tmp, eval_scores=None, stream=True, sigterm_works=True,
                 run_demo_pid=4242, eval_raises=False, demo_spawns=True):
        self.events = []
        self.lines = []
        self.now = 0.0
        self.eval_scores = eval_scores or {}
        self.stream = stream
        self.sigterm_works = sigterm_works
        self.run_demo_pid = run_demo_pid
        self.eval_raises = eval_raises
        self.demo_spawns = demo_spawns
        self.popened = []
        self.eval_call = None
        super().__init__(home=tmp / "home", proc_root=tmp / "proc",
                         env={"MINIO_ENDPOINT": "minio.example.internal:9000", "MINIO_USER": "promoter",
                              "MINIO_PASSWORD": SECRET},
                         run=self._run, popen=self._popen, kill=self._kill,
                         http_get=self._http_get, minio=FakeMinio({}),
                         sleep=self._sleep, clock=lambda: self.now,
                         utcnow=lambda: __import__("datetime").datetime(2026, 10, 2, 15, 0, 0),
                         out=lambda *a: self.lines.append(" ".join(map(str, a))))
        self.home.mkdir(parents=True)
        self.proc_root.mkdir(parents=True)

    @property
    def output(self):
        return "\n".join(self.lines)

    def add_process(self, pid, argv, cwd, env):
        d = self.proc_root / str(pid)
        d.mkdir()
        (d / "cmdline").write_bytes("\0".join(argv).encode() + b"\0")
        (d / "environ").write_bytes(b"\0".join(f"{k}={v}".encode() for k, v in env.items()))
        Path(cwd).mkdir(parents=True, exist_ok=True)
        (d / "cwd").symlink_to(cwd)

    def _sleep(self, s):
        self.now += s

    def _kill(self, pid, sig):
        self.events.append(("kill", pid, sig))
        if sig == signal.SIGKILL or self.sigterm_works:
            for f in (self.proc_root / str(pid)).iterdir():
                f.unlink()
            (self.proc_root / str(pid)).rmdir()

    def _run(self, cmd, **kw):
        res = type("R", (), {"returncode": 0, "stdout": "", "stderr": ""})()
        if cmd[0] == "pgrep":
            self.events.append(("pgrep", cmd[2]))
            res.stdout = "\n".join(p.name for p in self.proc_root.iterdir()) + "\n"
        elif "ecs_hcm_model.eval.seg" in cmd:
            self.events.append(("eval",))
            self.eval_call = (cmd, kw)
            if self.eval_raises:
                raise OSError("eval interpreter missing")
            # A score is an IoU, or (IoU, n_failed) to model failed transforms.
            res.stdout = "loading...\n" + "".join(
                json.dumps({"checkpoint": c, "iou": s[0], "input_size": 352,
                            "n_images": 5000, "n_failed": s[1]}) + "\n"
                for c, s in ((c, v if isinstance(v, tuple) else (v, 0))
                             for c, v in self.eval_scores.items()))
        elif cmd[0].endswith("run_demo.sh"):
            self.events.append(("run_demo", kw.get("cwd")))
            if self.demo_spawns:
                self.add_process(self.run_demo_pid, DEMO_ARGV, kw["cwd"], DEMO_ENV)
            res.stdout = f"demo started: pid {self.run_demo_pid}\nlog: x\n"
        else:
            raise AssertionError(f"unexpected command {cmd}")
        return res

    def _popen(self, argv, **kw):
        self.events.append(("popen",))
        self.popened.append((argv, kw))
        self.add_process(777, argv, kw["cwd"], kw["env"])
        return type("P", (), {"pid": 777})()

    def _http_get(self, url, timeout):
        if not self.stream:
            raise ConnectionRefusedError("nothing on :5000")
        return 200, b"--frame\r\n"


def _staged(host, name="20261003_010203", model=b"new-weights", meta=b'{"decoder_arch": "v2"}'):
    host.minio.objects = {("coco-models", f"hcm_coco_seg_best/{name}/model.pth"): model,
                          ("coco-models", f"hcm_coco_seg_best/{name}/metadata.json"): meta}
    assert pm.cmd_stage(host, f"coco-models/hcm_coco_seg_best/{name}") == 0
    return host.home / "models_staging" / name


def _baseline(host):
    models = host.home / "models"
    models.mkdir()
    (models / "model.pth").write_bytes(b"old-weights")
    (models / "metadata.json").write_text('{"input_size": 352}')
    return models


def _code_dir(tmp, demo_model=True):
    code = tmp / "repos" / "new"
    (code / "jetson").mkdir(parents=True)
    (code / ".venv" / "bin").mkdir(parents=True)
    (code / ".venv" / "bin" / "python").write_text("")
    (code / "jetson" / "run_demo.sh").write_text("")
    if demo_model:
        (code / "jetson" / "demo_model.py").write_text("")
    return code


def _running_demo(host, tmp):
    host.add_process(1234, DEMO_ARGV, str(tmp / "repos" / "old"), DEMO_ENV)


# ------------------------------------------------------------------- the pattern

def test_the_demo_pattern_never_matches_its_own_command_line():
    assert re.search(pm.DEMO_PATTERN, "/x/.venv/bin/python -u /x/jetson/realtime_demo.py")
    assert not re.search(pm.DEMO_PATTERN, f"bash -c \"pkill -f '{pm.DEMO_PATTERN}'\"")
    # The bracket form specifically, as the ssh/pkill rule requires.
    assert pm.DEMO_PATTERN.startswith("[r]")


def test_find_demo_skips_a_wrapping_shell_and_refuses_two_demos(tmp_path):
    host = FakeHost(tmp_path)
    host.add_process(10, ["bash", "-c", "nohup python jetson/realtime_demo.py"], str(tmp_path), {})
    host.add_process(11, DEMO_ARGV, str(tmp_path), DEMO_ENV)
    demo = pm.find_demo(host)
    assert (demo.pid, demo.argv, demo.cwd) == (11, DEMO_ARGV, str(tmp_path))
    assert ("pgrep", pm.DEMO_PATTERN) in host.events
    host.add_process(12, ["/usr/bin/python3", "realtime_demo.py"], str(tmp_path), {})
    with pytest.raises(pm.PromoteError, match="2 demo processes"):
        pm.find_demo(host)


def test_stop_demo_escalates_to_sigkill_after_the_timeout(tmp_path):
    host = FakeHost(tmp_path, sigterm_works=False)
    _running_demo(host, tmp_path)
    pm.stop_demo(host, pm.find_demo(host), timeout=5)
    assert [e for e in host.events if e[0] == "kill"] == [
        ("kill", 1234, signal.SIGTERM), ("kill", 1234, signal.SIGKILL)]
    assert not host.alive(1234)


# ------------------------------------------------------------------------- stage

def test_stage_copies_both_files_with_a_manifest_and_never_prints_the_secret(tmp_path):
    host = FakeHost(tmp_path)
    staged = _staged(host)
    assert (staged / "model.pth").read_bytes() == b"new-weights"
    manifest = json.loads((staged / pm.MANIFEST).read_text())
    assert manifest["source"] == "coco-models/hcm_coco_seg_best/20261003_010203"
    assert manifest["files"]["model.pth"]["sha256"] == pm._sha256(staged / "model.pth")
    assert host.minio.opened_with == ("minio.example.internal:9000", "promoter", SECRET, False)
    assert SECRET not in host.output
    assert pm.verify_staged(staged) == manifest


@pytest.mark.parametrize("unset", ["MINIO_ENDPOINT", "MINIO_USER", "MINIO_PASSWORD"])
def test_stage_refuses_without_each_credential(tmp_path, unset):
    host = FakeHost(tmp_path)
    del host.env[unset]
    with pytest.raises(pm.PromoteError, match=unset):
        pm.cmd_stage(host, "coco-models/hcm_coco_seg_best/x")
    assert not (host.home / "models_staging").exists()


def test_stage_leaves_nothing_behind_when_an_object_is_missing(tmp_path):
    host = FakeHost(tmp_path)
    host.minio.objects = {("coco-models", "p/x/model.pth"): b"w"}  # no metadata.json
    with pytest.raises(pm.PromoteError, match="metadata.json"):
        pm.cmd_stage(host, "coco-models/p/x")
    assert list((host.home / "models_staging").iterdir()) == []


class _RaisingMinio(FakeMinio):
    def __init__(self, error):
        super().__init__({})
        self.error = error

    def fget_object(self, bucket, key, path):
        raise self.error


def _connection_error(host):
    # urllib3's MaxRetryError text names host and port; the user is added so a
    # message that echoes it is caught too.
    endpoint, user = host.env["MINIO_ENDPOINT"], host.env["MINIO_USER"]
    host_part, port = endpoint.rsplit(":", 1)
    return type("MaxRetryError", (Exception,), {})(
        f"HTTPConnectionPool(host='{host_part}', port={port}): Max retries exceeded "
        f"with url: /coco-models?user={user} (Caused by NameResolutionError)")


def test_a_failed_fetch_never_reports_the_endpoint_or_the_user(tmp_path, capsys):
    host = FakeHost(tmp_path)
    host.env["MINIO_ENDPOINT"] = "secret-endpoint-xyz.invalid:9000"
    host.minio = _RaisingMinio(_connection_error(host))
    with pytest.raises(pm.PromoteError) as err:
        pm.cmd_stage(host, "coco-models/hcm_coco_seg_best/x")
    assert "MaxRetryError" in str(err.value)
    assert err.value.__cause__ is None and err.value.__suppress_context__
    assert pm.main(["stage", "coco-models/hcm_coco_seg_best/x"], host=host) == 2
    seen = str(err.value) + host.output + "".join(capsys.readouterr())
    for leaked in ("secret-endpoint-xyz", "9000", "promoter", SECRET):
        assert leaked not in seen
    assert list((host.home / "models_staging").iterdir()) == []


@pytest.mark.parametrize("endpoint", ["secret-endpoint-xyz:notaport", "secret-endpoint-xyz:9000/p"])
def test_a_malformed_endpoint_is_a_refusal_not_a_traceback(tmp_path, capsys, endpoint):
    host = FakeHost(tmp_path)
    host.env["MINIO_ENDPOINT"] = endpoint
    host.minio = pm._default_minio  # the real constructor raises the ValueError
    with pytest.raises(pm.PromoteError, match="MINIO_ENDPOINT is not usable") as err:
        pm.cmd_stage(host, "coco-models/x")
    assert pm.main(["stage", "coco-models/x"], host=host) == 2
    seen = str(err.value) + host.output + "".join(capsys.readouterr())
    for leaked in ("secret-endpoint-xyz", "promoter", SECRET):
        assert leaked not in seen
    staging = host.home / "models_staging"
    assert not staging.exists() or list(staging.iterdir()) == []


def test_a_missing_object_keeps_the_s3_error_code(tmp_path):
    from minio.error import S3Error
    host = FakeHost(tmp_path)
    # Keyword args: minio 7.2 moved `response` to the front, so positional args
    # land in different fields on 7.1.17 (the Jetson's and CI's version).
    host.minio = _RaisingMinio(S3Error(
        code="NoSuchKey", message="The specified key does not exist.",
        resource="/coco-models/x/model.pth", request_id="req1", host_id="hostid1",
        response=None, bucket_name="coco-models", object_name="x/model.pth"))
    with pytest.raises(pm.PromoteError) as err:
        pm.cmd_stage(host, "coco-models/x")
    assert "coco-models/x/model.pth: S3Error NoSuchKey" in str(err.value)
    assert "minio.example.internal" not in str(err.value) and "promoter" not in str(err.value)


def test_stage_refuses_a_metadata_json_that_does_not_parse(tmp_path):
    host = FakeHost(tmp_path)
    host.minio.objects = {("coco-models", "p/x/model.pth"): b"w",
                          ("coco-models", "p/x/metadata.json"): b"{not json"}
    with pytest.raises(pm.PromoteError, match="metadata.json does not parse"):
        pm.cmd_stage(host, "coco-models/p/x")
    assert list((host.home / "models_staging").iterdir()) == []


def test_stage_refuses_an_existing_staging_dir_and_a_bad_spec(tmp_path):
    host = FakeHost(tmp_path)
    _staged(host)
    with pytest.raises(pm.PromoteError, match="already exists"):
        _staged(host)
    with pytest.raises(pm.PromoteError, match="<bucket>/<prefix>"):
        pm.cmd_stage(host, "just-a-bucket")


# ------------------------------------------------------------------------ accept

def _accept(host, staged, models):
    return pm.cmd_accept(host, staged, "/data/coco", models, "/venv/python", check_timeout=30)


@pytest.mark.parametrize("cand_iou,base_iou,code", [(71.2, 70.86, 0), (70.86, 70.86, 1), (69.0, 70.86, 1)])
def test_accept_passes_only_a_strictly_better_candidate(tmp_path, cand_iou, base_iou, code):
    host = FakeHost(tmp_path)
    staged, models = _staged(host), _baseline(host)
    host.eval_scores = {str(staged / "model.pth"): cand_iou, str(models / "model.pth"): base_iou}
    assert _accept(host, staged, models) == code
    assert ("PASS" if code == 0 else "FAIL") in host.output
    assert f"iou={cand_iou:.2f}" in host.output and f"iou={base_iou:.2f}" in host.output


def test_accept_stops_the_demo_scores_on_coco_and_restarts_it_as_it_was(tmp_path):
    host = FakeHost(tmp_path)
    staged, models = _staged(host), _baseline(host)
    _running_demo(host, tmp_path)
    host.eval_scores = {str(staged / "model.pth"): 71.0, str(models / "model.pth"): 70.86}
    assert _accept(host, staged, models) == 0
    kinds = [e[0] for e in host.events if e[0] in ("kill", "eval", "popen")]
    assert kinds == ["kill", "eval", "popen"]
    cmd, kw = host.eval_call
    assert cmd[:3] == ["/venv/python", "-m", "ecs_hcm_model.eval.seg"]
    assert cmd[cmd.index("--dataset") + 1] == "coco" and "--json" in cmd
    assert cmd[-2:] == [str(staged / "model.pth"), str(models / "model.pth")]
    assert "--input-size" not in cmd  # each checkpoint at its own recorded size
    assert kw["env"]["PYTORCH_NO_CUDA_MEMORY_CACHING"] == "1"
    assert str(pm.REPO_ROOT) in kw["env"]["PYTHONPATH"]
    argv, popen_kw = host.popened[0]
    assert argv == DEMO_ARGV and popen_kw["cwd"] == str(tmp_path / "repos" / "old")
    assert popen_kw["env"] == DEMO_ENV
    assert "stream OK" in host.output


@pytest.mark.parametrize("failure", ["raises", "no_scores"])
def test_accept_restarts_the_demo_even_when_scoring_fails(tmp_path, failure):
    host = FakeHost(tmp_path, eval_raises=(failure == "raises"))
    staged, models = _staged(host), _baseline(host)
    _running_demo(host, tmp_path)
    with pytest.raises((OSError, pm.PromoteError)):
        _accept(host, staged, models)
    assert len(host.popened) == 1


def test_accept_refuses_scores_over_different_image_sets(tmp_path, capsys):
    host = FakeHost(tmp_path)
    staged, models = _staged(host), _baseline(host)
    _running_demo(host, tmp_path)
    # The candidate "wins" only because it skipped images the baseline scored.
    host.eval_scores = {str(staged / "model.pth"): (75.0, 12), str(models / "model.pth"): (70.86, 0)}
    with pytest.raises(pm.PromoteError, match="candidate 12, baseline 0"):
        _accept(host, staged, models)
    assert "PASS" not in host.output and "n_failed=12" in host.output and "n_failed=0" in host.output
    assert len(host.popened) == 1  # the demo still comes back
    assert pm.main(["accept", str(staged), "--data-root", "/d", "--models-dir", str(models)],
                   host=host) == 2
    assert "not comparable" in capsys.readouterr().err


def test_accept_refuses_an_eval_that_does_not_report_n_failed(tmp_path):
    host = FakeHost(tmp_path)
    staged, models = _staged(host), _baseline(host)
    _running_demo(host, tmp_path)
    host.eval_scores = {str(staged / "model.pth"): 75.0, str(models / "model.pth"): 70.86}
    real_run = host.run

    def run_without_n_failed(cmd, **kw):
        res = real_run(cmd, **kw)
        if "ecs_hcm_model.eval.seg" in cmd:
            res.stdout = "".join(
                (json.dumps({k: v for k, v in json.loads(l).items() if k != "n_failed"}) + "\n")
                if l.startswith("{") else l + "\n" for l in res.stdout.splitlines())
        return res

    host.run = run_without_n_failed
    with pytest.raises(pm.PromoteError, match="did not report n_failed"):
        _accept(host, staged, models)
    assert "PASS" not in host.output
    assert len(host.popened) == 1  # the demo still comes back


def test_accept_leaves_a_stopped_demo_stopped(tmp_path):
    host = FakeHost(tmp_path)
    staged, models = _staged(host), _baseline(host)
    host.eval_scores = {str(staged / "model.pth"): 71.0, str(models / "model.pth"): 70.86}
    assert _accept(host, staged, models) == 0
    assert host.popened == [] and "not running before" in host.output


def test_accept_refuses_a_staged_file_that_changed(tmp_path):
    host = FakeHost(tmp_path)
    staged, models = _staged(host), _baseline(host)
    (staged / "model.pth").write_bytes(b"tampered")
    with pytest.raises(pm.PromoteError, match="sha256"):
        _accept(host, staged, models)
    assert ("eval",) not in host.events


# ------------------------------------------------------------------------- apply

def _apply(host, staged, code, models, yes=True):
    return pm.cmd_apply(host, staged, code, models, yes, check_timeout=30)


def test_apply_refuses_without_yes_before_touching_anything(tmp_path):
    host = FakeHost(tmp_path)
    staged, models, code = _staged(host), _baseline(host), _code_dir(tmp_path)
    _running_demo(host, tmp_path)
    with pytest.raises(pm.PromoteError, match="--yes"):
        _apply(host, staged, code, models, yes=False)
    assert host.alive(1234) and (models / "model.pth").read_bytes() == b"old-weights"


@pytest.mark.parametrize("missing", ["jetson/demo_model.py", "jetson/run_demo.sh", ".venv/bin/python"])
def test_apply_refuses_a_code_dir_that_cannot_run_v2(tmp_path, missing):
    host = FakeHost(tmp_path)
    staged, models, code = _staged(host), _baseline(host), _code_dir(tmp_path)
    (code / missing).unlink()
    _running_demo(host, tmp_path)
    with pytest.raises(pm.PromoteError, match=re.escape(missing)):
        _apply(host, staged, code, models)
    assert host.alive(1234) and (models / "model.pth").read_bytes() == b"old-weights"


def test_apply_keeps_the_old_weights_installs_the_new_and_checks_the_stream(tmp_path):
    host = FakeHost(tmp_path)
    staged, models, code = _staged(host), _baseline(host), _code_dir(tmp_path)
    _running_demo(host, tmp_path)
    assert _apply(host, staged, code, models) == 0
    prev = host.home / "models_prev_20261002T150000Z"
    assert (prev / "model.pth").read_bytes() == b"old-weights"
    assert (models / "model.pth").read_bytes() == b"new-weights"
    assert json.loads((models / "PROMOTED.json").read_text())["source"].endswith("20261003_010203")
    assert not host.alive(1234)
    assert ("run_demo", str(code.resolve())) in host.events
    assert "stream OK" in host.output
    rollback = host.output.split("rollback:")[1]
    assert f"pkill -f '{pm.DEMO_PATTERN}'" in rollback
    assert f"mv {prev} {models}" in rollback
    assert "cd " + str(tmp_path / "repos" / "old") in rollback
    assert "PYTORCH_NO_CUDA_MEMORY_CACHING=1" in rollback
    assert SECRET not in host.output  # the old process's env had it


def test_apply_prints_the_rollback_before_the_copy_can_fail(tmp_path, monkeypatch):
    host = FakeHost(tmp_path)
    staged, models, code = _staged(host), _baseline(host), _code_dir(tmp_path)
    _running_demo(host, tmp_path)

    def disk_full(src, dst):
        raise OSError(28, "No space left on device")
    monkeypatch.setattr(pm.shutil, "copy2", disk_full)
    with pytest.raises(OSError):
        _apply(host, staged, code, models)
    prev = host.home / "models_prev_20261002T150000Z"
    rollback = host.output.split("rollback:")[1]
    assert f"mv {prev} {models}" in rollback and "cd " + str(tmp_path / "repos" / "old") in rollback
    assert (prev / "model.pth").read_bytes() == b"old-weights"


def test_an_unreadable_demo_process_is_refused_not_ignored(tmp_path):
    host = FakeHost(tmp_path)
    staged, models, code = _staged(host), _baseline(host), _code_dir(tmp_path)
    _running_demo(host, tmp_path)
    # A directory in place of the file: read_bytes raises OSError even as root,
    # standing in for another user's /proc/<pid>/environ.
    environ = host.proc_root / "1234" / "environ"
    environ.unlink()
    environ.mkdir()
    with pytest.raises(pm.PromoteError, match="cannot read pid 1234"):
        _apply(host, staged, code, models)
    assert "WARNING: pgrep matched pid 1234" in host.output
    assert ("kill", 1234, signal.SIGTERM) not in host.events
    assert (models / "model.pth").read_bytes() == b"old-weights"


@pytest.mark.parametrize("failure", ["no_stream", "demo_exits"])
def test_apply_reports_a_demo_that_does_not_serve(tmp_path, failure):
    host = FakeHost(tmp_path, stream=False, demo_spawns=(failure == "no_stream"))
    staged, models, code = _staged(host), _baseline(host), _code_dir(tmp_path)
    assert _apply(host, staged, code, models) == 1
    assert (host.home / "models_prev_20261002T150000Z" / "model.pth").exists()
    assert "rollback:" in host.output
    expected = "exited before serving" if failure == "demo_exits" else "no HTTP 200"
    assert expected in host.output


def test_a_200_without_a_frame_is_not_healthy(tmp_path):
    host = FakeHost(tmp_path)
    host.http_get = lambda url, timeout: (200, b"")  # server up, model never loaded
    assert pm.wait_for_stream(host, None, timeout=10) is False
    assert "stream OK" not in host.output


def test_main_reports_a_refusal_as_exit_2(tmp_path, capsys):
    host = FakeHost(tmp_path)
    assert pm.main(["apply", str(tmp_path), "--code-dir", str(tmp_path)], host=host) == 2
    assert "--yes" in capsys.readouterr().err
