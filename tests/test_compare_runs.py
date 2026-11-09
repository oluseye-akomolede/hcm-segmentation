"""tools/compare_runs.py against a fake MLflow HTTP layer. No request leaves
the process, and no MLflow logger is built (its ctor resets the URI and can
create real runs)."""

import importlib.util
import json
import re
import urllib.error
import urllib.parse
from pathlib import Path

import pytest

TOOL = Path(__file__).resolve().parents[1] / "tools" / "compare_runs.py"
_spec = importlib.util.spec_from_file_location("compare_runs", TOOL)
cr = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cr)

H = 3600 * 1000  # one hour in ms
BASE_ID, RUN_ID = "a" * 32, "b" * 32


def _run(run_id, name, status, start, keys, max_epochs=None):
    params = [{"key": "max epochs", "value": str(max_epochs)}] if max_epochs else []
    return {"info": {"run_id": run_id, "run_name": name, "status": status, "start_time": start},
            "data": {"metrics": [{"key": k, "value": 0.0, "step": 0} for k in keys],
                     "params": params}}


class FakeMlflow:
    """runs/search, runs/get and metrics/get-history over in-memory runs."""

    def __init__(self, runs, histories, page_size=1000):
        self.runs = runs                # {run_id: run}
        self.histories = histories      # {(run_id, key): [(step, value, ts)]}
        self.page_size = page_size
        self.calls = []

    def __call__(self, method, url, body=None):
        parsed = urllib.parse.urlparse(url)
        endpoint = parsed.path.split("/api/2.0/mlflow/")[1]
        q = dict(urllib.parse.parse_qsl(parsed.query))
        self.calls.append((method, endpoint, q, body))
        if endpoint == "runs/search":
            name = re.search(r"run_name = '(.*)'", body["filter"]).group(1)
            return {"runs": [r for r in self.runs.values() if r["info"]["run_name"] == name]}
        if endpoint == "runs/get":
            return {"run": self.runs[q["run_id"]]}
        if endpoint == "metrics/get-history":
            pts = self.histories.get((q["run_id"], q["metric_key"]), [])
            start = int(q.get("page_token", 0))
            page = pts[start:start + self.page_size]
            out = {"metrics": [{"key": q["metric_key"], "step": s, "value": v, "timestamp": t}
                               for s, v, t in page]}
            if start + self.page_size < len(pts):
                out["next_page_token"] = str(start + self.page_size)
            return out
        raise AssertionError(f"unexpected endpoint {endpoint}")


def _hist(values, start, hours, step0=0):
    """[(epoch, value, ts)] with epoch i ending at start + hours[0] + ... + hours[i]."""
    out, ts = [], start
    for i, (v, h) in enumerate(zip(values, hours)):
        ts += int(h * H)
        out.append((step0 + i, v, ts))
    return out


def _world(run_status="RUNNING", run_max=None, base_keys=("val_iou", "val_dice", "val_loss")):
    base = _run(BASE_ID, "carefree-rook-316", "FINISHED", 0, list(base_keys) + ["train_loss"], 15)
    run = _run(RUN_ID, "v2-retrain", run_status, 100 * H,
               ["val_iou", "val_dice", "val_loss", "val_extra", "train_loss"], run_max)
    hist = {}
    for k, vals in {"val_iou": [60, 66, 70], "val_dice": [70, 76, 80],
                    "val_loss": [0.45, 0.39, 0.35], "train_loss": [0.5, 0.4, 0.35]}.items():
        hist[BASE_ID, k] = _hist(vals, 0, [3, 3, 3])
    for k, vals in {"val_iou": [61, 65], "val_dice": [71, 75],
                    "val_loss": [0.44, 0.40], "val_extra": [1, 2], "train_loss": [0.49, 0.41]}.items():
        hist[RUN_ID, k] = _hist(vals, 100 * H, [2.5, 2.0])
    return FakeMlflow({BASE_ID: base, RUN_ID: run}, hist)


def _rows(lines):
    return {int(l.split()[0]): l.split() for l in lines if re.match(r"\s*\d+\s", l)}


def test_each_epoch_is_compared_with_the_baselines_same_epoch():
    lines = cr.compare(cr.MlflowReader("http://mlflow", _world()), "v2-retrain", "carefree-rook-316")
    header = next(l for l in lines if l.strip().startswith("epoch"))
    # Keys come from the baseline, headline first; the run-only val_extra is not shown.
    assert header.split() == ["epoch", "val_iou", "Δ", "val_dice", "Δ", "val_loss", "Δ",
                              "train_loss", "Δ", "epoch_h"]
    rows = _rows(lines)
    assert sorted(rows) == [0, 1]
    assert rows[0][1:3] == ["61.000", "+1.000"] and rows[1][1:3] == ["65.000", "-1.000"]
    assert rows[1][5:7] == ["0.400", "+0.010"]
    assert rows[0][-1] == "2.50" and rows[1][-1] == "2.00"  # epoch 0 from the run start
    assert "baseline best val_iou 70.000 at epoch 2" in lines
    assert "run best      val_iou 65.000 at epoch 1" in lines


def test_an_epoch_the_baseline_never_reached_has_no_delta():
    world = _world()
    world.histories[RUN_ID, "val_iou"].append((3, 72.0, 110 * H))
    rows = _rows(cr.compare(cr.MlflowReader("http://m", world), "v2-retrain", "carefree-rook-316"))
    assert rows[3][1:3] == ["72.000", "-"]


def test_the_projection_uses_the_mean_epoch_time_and_the_planned_epochs():
    world = _world(run_max=10)
    lines = cr.compare(cr.MlflowReader("http://m", world), "v2-retrain", "carefree-rook-316")
    # 2 epochs done, mean 2.25 h, last val at 104.5 h: 8 to go -> 104.5 + 18 = 122.5 h.
    want = cr._fmt_time(int(122.5 * H))
    assert f"projected finish of 10 epochs (8 to go): {want}" in lines
    lines = cr.compare(cr.MlflowReader("http://m", world), "v2-retrain", "carefree-rook-316", epochs=4)
    assert f"projected finish of 4 epochs (2 to go): {cr._fmt_time(int(109 * H))}" in lines


def test_no_projection_without_a_plan_or_for_a_finished_run():
    lines = cr.compare(cr.MlflowReader("http://m", _world()), "v2-retrain", "carefree-rook-316")
    assert any("planned epochs unknown" in l for l in lines)
    lines = cr.compare(cr.MlflowReader("http://m", _world("FINISHED", 10)), "v2-retrain",
                       "carefree-rook-316")
    assert "run is FINISHED; no projection" in lines


def test_only_read_endpoints_are_called_and_an_id_uses_runs_get():
    world = _world()
    cr.compare(cr.MlflowReader("http://m", world), RUN_ID, "carefree-rook-316")
    assert {(m, e) for m, e, _, _ in world.calls} == {
        ("POST", "runs/search"), ("GET", "runs/get"), ("GET", "metrics/get-history")}
    search = next(b for _, e, _, b in world.calls if e == "runs/search")
    assert search["experiment_ids"] == ["7"]


def test_history_pages_are_followed_and_the_last_log_of_an_epoch_wins():
    world = _world()
    world.page_size = 1
    world.histories[RUN_ID, "val_iou"].insert(1, (0, 99.0, 103 * H))  # a re-log of epoch 0
    hist = cr.MlflowReader("http://m", world).history(RUN_ID, "val_iou")
    assert hist == {0: (99.0, 103 * H), 1: (65.0, int(104.5 * H))}
    assert sum(1 for _, e, _, _ in world.calls if e == "metrics/get-history") == 3


def test_a_baseline_without_val_metrics_is_refused(capsys):
    world = _world(base_keys=())
    assert cr.main(["v2-retrain"], http=world) == 2
    assert "logged no val_* metric" in capsys.readouterr().err


@pytest.mark.parametrize("dupes,expected", [(0, "no run named"), (2, "pass a run id")])
def test_an_unknown_or_ambiguous_name_is_refused(dupes, expected):
    world = _world()
    for i in range(dupes):
        world.runs[f"{i}" * 32] = _run(f"{i}" * 32, "twin", "FINISHED", 0, ["val_iou"])
    with pytest.raises(cr.CompareError, match=expected):
        cr.compare(cr.MlflowReader("http://m", world), "twin", "carefree-rook-316")


def test_the_uri_defaults_to_the_dev_server_and_follows_the_environment(monkeypatch, capsys):
    from ecs_hcm_model.core.model_logger import MLFLOW_DEV
    assert cr.MLFLOW_DEV == MLFLOW_DEV
    seen = []
    world = _world()

    def http(method, url, body=None):
        seen.append(url)
        return world(method, url, body)

    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    assert cr.main(["v2-retrain"], http=http) == 0
    assert all(u.startswith(MLFLOW_DEV + "/api/2.0/mlflow/") for u in seen)
    seen.clear()
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://other:5000")
    assert cr.main(["v2-retrain"], http=http) == 0
    assert all(u.startswith("http://other:5000/api/2.0/mlflow/") for u in seen)


def test_the_tool_never_imports_mlflow_or_the_trainer_logger():
    src = TOOL.read_text()
    assert not re.search(r"^\s*(import|from)\s+(mlflow|ecs_hcm_model)", src, re.M)


def _stale_lines(lines):
    return [l for l in lines if l.startswith("stale?")]


def test_a_running_run_that_stopped_logging_is_flagged_stale():
    world = _world(run_max=10)
    reader = cr.MlflowReader("http://m", world)
    # Last val log at 104.5 h, mean epoch 2.25 h, finish projected at 122.5 h.
    fresh = cr.compare(reader, "v2-retrain", "carefree-rook-316", now_ms=int(106 * H))
    assert _stale_lines(fresh) == []
    quiet = cr.compare(reader, "v2-retrain", "carefree-rook-316", now_ms=int(110 * H))
    assert len(_stale_lines(quiet)) == 1 and "5.5 h ago, over 2x the mean epoch" in quiet[-1]
    assert "has passed" not in quiet[-1]
    late = cr.compare(reader, "v2-retrain", "carefree-rook-316", now_ms=int(123 * H))
    assert "has passed" in _stale_lines(late)[0]


def test_a_quiet_run_with_no_plan_is_still_flagged_and_a_finished_one_never():
    reader = cr.MlflowReader("http://m", _world())  # RUNNING, no 'max epochs'
    assert _stale_lines(cr.compare(reader, "v2-retrain", "carefree-rook-316", now_ms=int(110 * H)))
    done = cr.MlflowReader("http://m", _world("FINISHED", 10))
    assert _stale_lines(cr.compare(done, "v2-retrain", "carefree-rook-316",
                                   now_ms=int(1000 * H))) == []


@pytest.mark.parametrize("err", [
    urllib.error.URLError("[Errno 111] Connection refused"),
    urllib.error.HTTPError("http://u:pw@mlflow:5000/api", 503, "Service Unavailable", {}, None)])
def test_an_unreachable_or_failing_server_is_an_error_line_and_exit_2(monkeypatch, capsys, err):
    def http(method, url, body=None):
        raise err

    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://u:pw@mlflow:5000")
    assert cr.main(["v2-retrain"], http=http) == 2
    out, errtext = capsys.readouterr()
    assert errtext.startswith("error: MLflow at http://mlflow:5000: ")
    assert "Traceback" not in errtext and "u:pw" not in out + errtext


@pytest.mark.parametrize("uri,shown", [
    ("http://alice:s3cret@mlflow:5000", "http://mlflow:5000"),
    ("https://tok@mlflow.example/path", "https://mlflow.example/path"),
    ("http://10.0.1.102:9090", "http://10.0.1.102:9090")])
def test_the_printed_tracking_uri_never_carries_credentials(monkeypatch, capsys, uri, shown):
    monkeypatch.setenv("MLFLOW_TRACKING_URI", uri)
    assert cr.main(["v2-retrain"], http=_world()) == 0
    out = capsys.readouterr().out
    assert f"tracking {shown}\n" in out
    assert "s3cret" not in out and "alice" not in out and "tok@" not in out


def test_a_write_endpoint_is_refused_by_a_raise_not_an_assert():
    reader = cr.MlflowReader("http://m", _world())
    with pytest.raises(ValueError, match="not one of the read endpoints"):
        reader._call("POST", "runs/delete", {"run_id": RUN_ID})
    # python -O strips asserts; the guard must not be one.
    src = TOOL.read_text()
    assert not re.search(r"^\s*assert\b", src, re.M)


def test_a_run_with_no_epochs_left_is_judged_by_age_only():
    # 2 of 2 epochs logged: finalising, so its projected finish (104.5 h) is
    # already "past" and must not by itself make it stale.
    reader = cr.MlflowReader("http://m", _world(run_max=2))
    finishing = cr.compare(reader, "v2-retrain", "carefree-rook-316", now_ms=int(106 * H))
    assert "projected finish of 2 epochs (0 to go)" in " ".join(finishing)
    assert _stale_lines(finishing) == []
    quiet = _stale_lines(cr.compare(reader, "v2-retrain", "carefree-rook-316", now_ms=int(110 * H)))
    assert len(quiet) == 1 and "over 2x the mean epoch" in quiet[0] and "has passed" not in quiet[0]


def _non_json_200(method, url, body=None):
    return json.loads(b"<html>proxy login</html>")  # what the default http does with the body


def _timeout(method, url, body=None):
    raise TimeoutError("timed out")


@pytest.mark.parametrize("http,kind", [(_timeout, "TimeoutError"), (_non_json_200, "JSONDecodeError")])
def test_a_timeout_or_a_non_json_reply_is_an_error_line_and_exit_2(monkeypatch, capsys, http, kind):
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://u:pw@mlflow:5000")
    assert cr.main(["v2-retrain"], http=http) == 2
    out, errtext = capsys.readouterr()
    assert errtext.startswith(f"error: MLflow at http://mlflow:5000: {kind}: ")
    assert "Traceback" not in errtext and "u:pw" not in out + errtext


# --------------------------------------------------------------------------
# The trainer's timing split (openspec epoch-timing-split)
# --------------------------------------------------------------------------

def _with_split(world, epochs=(0,)):
    """Give the run the five timing keys, logged for ``epochs`` only."""
    run = world.runs[RUN_ID]
    for key, _, _ in cr.TIMING_COLS:
        run["data"]["metrics"].append({"key": key, "value": 0.0, "step": 0})
    vals = {"epoch_train_s": 7200.0, "epoch_data_wait_s": 5400.0, "epoch_step_s": 1440.0,
            "epoch_val_s": 900.0, "train_samples_per_s": 3.25}
    for key, v in vals.items():
        world.histories[RUN_ID, key] = [(e, v, 100 * H + (e + 1) * H) for e in epochs]
    return world


def test_a_run_with_the_split_gets_its_columns_in_hours():
    lines = cr.compare(cr.MlflowReader("http://mlflow", _with_split(_world())),
                       "v2-retrain", "carefree-rook-316")
    header = next(l for l in lines if l.strip().startswith("epoch"))
    assert header.split()[-7:] == ["epoch_h", "train_h", "wait_h", "step_h", "val_h", "smp/s", "other_h"]
    rows = _rows(lines)
    # other_h = 7200 - 5400 - 1440 s = 0.10 h
    assert rows[0][-6:] == ["2.00", "1.50", "0.40", "0.25", "3.25", "0.10"]
    # Epoch 1 logged no split (say, a resumed run): blanks, not zeros.
    assert rows[1][-6:] == ["-"] * 6


def test_an_older_run_gets_exactly_the_report_it_had():
    world = _world()
    lines = cr.compare(cr.MlflowReader("http://mlflow", world), "v2-retrain", "carefree-rook-316")
    header = next(l for l in lines if l.strip().startswith("epoch"))
    assert header.split()[-1] == "epoch_h"
    assert all(_rows(lines)[e][-1] in ("2.50", "2.00") for e in (0, 1))
    # No history is fetched for keys the run never logged.
    fetched = {q["metric_key"] for _, ep, q, _ in world.calls if ep == "metrics/get-history"}
    assert not fetched & {k for k, _, _ in cr.TIMING_COLS}


def test_the_columns_are_the_trainers_keys():
    import ecs_hcm_model.core.trainer as trainer
    assert tuple(k for k, _, _ in cr.TIMING_COLS) == trainer.EPOCH_TIMING_KEYS


def test_other_is_shown_only_when_train_wait_and_step_were_all_logged():
    world = _with_split(_world())
    run = world.runs[RUN_ID]
    run["data"]["metrics"] = [m for m in run["data"]["metrics"] if m["key"] != "epoch_step_s"]
    lines = cr.compare(cr.MlflowReader("http://mlflow", world), "v2-retrain", "carefree-rook-316")
    header = next(l for l in lines if l.strip().startswith("epoch"))
    assert "other_h" not in header.split() and header.split()[-1] == "smp/s"


def test_other_is_blank_for_an_epoch_missing_one_of_its_parts():
    world = _with_split(_world(), epochs=(0, 1))
    world.histories[RUN_ID, "epoch_data_wait_s"] = world.histories[RUN_ID, "epoch_data_wait_s"][:1]
    rows = _rows(cr.compare(cr.MlflowReader("http://mlflow", world), "v2-retrain", "carefree-rook-316"))
    assert rows[0][-1] == "0.10" and rows[1][-1] == "-"


# --brief: one line for an owner report. Epoch ms 0 is 1970-01-01 00:00Z, so
# 100 h in is Mon 01-05 04:00Z = Sun 01-04 10:00 PM CST in Chicago.

def _brief(world, now_h, **kw):
    return cr.brief(cr.MlflowReader("http://m", world), "v2-retrain", "carefree-rook-316",
                    now_ms=int(now_h * H), **kw)


def test_brief_is_one_line_with_the_baseline_at_the_same_epoch_and_a_chicago_finish():
    line = _brief(_world(run_max=10), 105)
    # Now 105 h = Mon 3:00 AM CST. Last val log 104.5 h, mean 2.25 h, 8 epochs left
    # -> 122.5 h = Mon 8:30 PM CST: the same Chicago day, so no weekday.
    assert line == ("Epoch 2/10: COCO val 65.0 (baseline at epoch 2: 66.0, best 70.00)."
                    " ~2.2 h/epoch, done ~8:30 PM CST.")
    assert "\n" not in line and len(line) <= 160


def test_brief_before_the_first_epoch_projects_from_the_baselines_epoch_time():
    world = _world(run_max=10)
    for k in [k for k in world.histories if k[0] == RUN_ID]:
        del world.histories[k]
    # Now 101 h = Sun 11:00 PM CST; 100 h + 10 x 3 h = 130 h = Tue 4:00 AM CST.
    assert _brief(world, 101) == ("No epochs logged yet (1.0 h in). At the baseline's"
                                  " ~3.0 h/epoch, 10 epochs done ~4:00 AM CST Tue.")


def test_brief_flags_a_run_that_stopped_logging_and_never_projects_a_finished_one():
    assert "No val log" not in _brief(_world(run_max=10), 106)
    assert _brief(_world(run_max=10), 110).endswith("No val log for 5.5 h; check it is alive.")
    done = _brief(_world(run_status="FINISHED", run_max=10), 110)
    assert done.endswith("Run is FINISHED.") and "done ~" not in done


def test_brief_without_a_plan_says_so_and_with_no_epochs_left_says_finishing():
    assert _brief(_world(), 105).endswith(", planned epochs unknown.")
    assert _brief(_world(run_max=2), 105).endswith(", all epochs logged, finishing up.")
    assert _brief(_world(), 105, epochs=10).endswith("done ~8:30 PM CST.")


def test_brief_on_the_cli_prints_one_line_and_only_reads(monkeypatch, capsys):
    world = _world(run_max=10)
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://m")
    assert cr.main(["v2-retrain", "--brief"], http=world) == 0
    out = capsys.readouterr().out
    assert out.count("\n") == 1 and out.startswith("Epoch 2/10: COCO val 65.0")
    assert {c[1] for c in world.calls} <= set(cr.READ_ENDPOINTS)


def test_brief_after_one_epoch_projects_from_that_epoch():
    world = _world(run_max=10)
    for k in [k for k in world.histories if k[0] == RUN_ID]:
        world.histories[k] = world.histories[k][:1]
    # One epoch of 2.5 h ending at 102.5 h; 9 left -> 125 h = Mon 11:00 PM CST.
    assert _brief(world, 103) == ("Epoch 1/10: COCO val 61.0 (baseline at epoch 1: 60.0,"
                                  " best 70.00). ~2.5 h/epoch, done ~11:00 PM CST.")


def test_chicago_times_follow_dst_and_name_the_weekday_off_today():
    import datetime as dt
    ms = lambda *a: int(dt.datetime(*a, tzinfo=dt.timezone.utc).timestamp() * 1000)
    now = ms(2026, 10, 31, 17)                                      # Sat 12:00 PM CDT
    assert cr._fmt_chicago(ms(2026, 11, 1, 0, 10), now) == "7:10 PM CDT"
    assert cr._fmt_chicago(ms(2026, 11, 1, 18), now) == "12:00 PM CST Sun"  # after fall-back


def _no_epochs(world):
    for k in [k for k in world.histories if k[0] == RUN_ID]:
        del world.histories[k]
    return world


def test_brief_labels_by_mlflow_step_so_a_resumed_run_compares_the_same_epoch():
    # A resume logging into a new MLflow run: its first val_iou is at step 7.
    world = _world(run_max=10)
    world.histories[BASE_ID, "val_iou"] = _hist([60, 61, 62, 63, 64, 65, 66, 67, 68, 69], 0, [3] * 10)
    world.histories[RUN_ID, "val_iou"] = _hist([71], 100 * H, [2.5], step0=7)
    # Step 7 = epoch 8; 2 left x 2.5 h after 102.5 h -> 107.5 h = Mon 5:30 AM CST.
    assert _brief(world, 103) == ("Epoch 8/10: COCO val 71.0 (baseline at epoch 8: 67.0,"
                                  " best 69.00). ~2.5 h/epoch, done ~5:30 AM CST.")


def test_brief_a_failed_run_with_no_epochs_is_named_and_never_projected():
    world = _no_epochs(_world(run_status="FAILED", run_max=10))
    assert _brief(world, 101) == "No epochs logged yet (1.0 h in). Run is FAILED."


@pytest.mark.parametrize("run_max", [10, None])
def test_brief_with_no_epochs_flags_a_run_quiet_past_2x_the_baselines_epoch(run_max):
    # Baseline mean epoch 3 h; the run started at 100 h.
    world = _no_epochs(_world(run_max=run_max))
    assert "check it is alive" not in _brief(world, 105.9)
    assert "check it is alive" not in _brief(world, 106)  # exactly 2x: not yet
    assert _brief(world, 106.1).endswith(
        " Over 2x the baseline's epoch with none logged; check it is alive.")
