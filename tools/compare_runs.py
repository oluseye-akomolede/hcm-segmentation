"""Compare an MLflow run with a baseline run, epoch by epoch. Read-only.

    python tools/compare_runs.py <run name or id> [<baseline>] [--epochs N]

The baseline defaults to carefree-rook-316, the 352 v1 run in experiment 7.
The val metric keys are whatever ``val_*`` keys the baseline logged; each
metric's step is the epoch. An epoch's wall time is the gap between its val
log and the previous one (epoch 0: from the run start, so it includes setup).
A run that logged the trainer's timing split (epoch_train_s and the rest) also
gets those columns, the run's own values only; older runs do not.

Only MLflow's read endpoints are called, over plain HTTP. This deliberately
does not import the trainer's logger: its constructor resets the tracking URI
to the dev server, and a stray write there creates a real run.
"""

import argparse
import datetime
import json
import os
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from zoneinfo import ZoneInfo

MLFLOW_DEV = "http://10.0.1.102:9090"  # ecs_hcm_model.core.model_logger.MLFLOW_DEV
DEFAULT_BASELINE = "carefree-rook-316"
DEFAULT_EXPERIMENT = "7"  # COCO seg
TRAIN_LOSS = "train_loss"
PRIMARY = "val_iou"
# The trainer's per-epoch timing split (ecs_hcm_model.core.trainer
# EPOCH_TIMING_KEYS), as (key, column, divisor): seconds shown in hours.
TIMING_COLS = (
    ("epoch_train_s", "train_h", 3600.0),
    ("epoch_data_wait_s", "wait_h", 3600.0),
    ("epoch_step_s", "step_h", 3600.0),
    ("epoch_val_s", "val_h", 3600.0),
    ("train_samples_per_s", "smp/s", 1.0),
)
# other_h = train - data wait - step, per epoch.
OTHER_PARTS = ("epoch_train_s", "epoch_data_wait_s", "epoch_step_s")
READ_ENDPOINTS = ("runs/search", "runs/get", "metrics/get-history")
CHICAGO = ZoneInfo("America/Chicago")


class CompareError(Exception):
    pass


def _default_http(method, url, body=None):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, method=method,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read() or b"{}")


class MlflowReader:
    """The three read endpoints this tool needs, and nothing else."""

    def __init__(self, uri, http=_default_http):
        self.base = uri.rstrip("/") + "/api/2.0/mlflow/"
        self.http = http

    def _call(self, method, endpoint, body=None, query=None):
        # A raise, not an assert: python -O strips asserts, and this is the
        # guard that keeps the tool read-only.
        if endpoint not in READ_ENDPOINTS:
            raise ValueError(f"{endpoint!r} is not one of the read endpoints {READ_ENDPOINTS}")
        url = self.base + endpoint
        if query:
            url += "?" + urllib.parse.urlencode(query)
        return self.http(method, url, body)

    def run(self, name_or_id, experiment):
        if re.fullmatch(r"[0-9a-f]{32}", name_or_id):
            return self._call("GET", "runs/get", query={"run_id": name_or_id})["run"]
        quoted = name_or_id.replace("'", "\\'")
        runs = self._call("POST", "runs/search", body={
            "experiment_ids": [experiment],
            "filter": f"attributes.run_name = '{quoted}'",
            "max_results": 10,
        }).get("runs", [])
        if not runs:
            raise CompareError(f"no run named {name_or_id!r} in experiment {experiment}")
        if len(runs) > 1:
            ids = ", ".join(r["info"]["run_id"] for r in runs)
            raise CompareError(f"{len(runs)} runs are named {name_or_id!r} ({ids}); pass a run id")
        return runs[0]

    def history(self, run_id, key):
        """``{epoch: (value, timestamp_ms)}``; the last log of an epoch wins."""
        points, token = [], None
        while True:
            query = {"run_id": run_id, "metric_key": key, "max_results": 25000}
            if token:
                query["page_token"] = token
            page = self._call("GET", "metrics/get-history", query=query)
            points += page.get("metrics", [])
            token = page.get("next_page_token")
            if not token:
                break
        out = {}
        for m in sorted(points, key=lambda m: (int(m["step"]), int(m["timestamp"]))):
            out[int(m["step"])] = (float(m["value"]), int(m["timestamp"]))
        return out


def val_keys_of(run):
    keys = sorted(m["key"] for m in run["data"].get("metrics", []) if m["key"].startswith("val_"))
    if not keys:
        raise CompareError(f"baseline {run['info'].get('run_name')} logged no val_* metric")
    # The headline metric first, the rest alphabetically.
    return sorted(keys, key=lambda k: (k != PRIMARY, k))


def planned_epochs(run):
    for p in run["data"].get("params", []):
        if p["key"] == "max epochs":
            return int(p["value"])
    return None


def epoch_seconds(run, timing):
    """``{epoch: seconds}`` from consecutive val timestamps."""
    out, prev = {}, int(run["info"]["start_time"])
    for epoch in sorted(timing):
        ts = timing[epoch][1]
        out[epoch] = (ts - prev) / 1000.0
        prev = ts
    return out


def display_uri(uri):
    """``uri`` without any ``user:pass@``, for printing."""
    parts = urllib.parse.urlsplit(uri)
    if "@" not in parts.netloc:
        return uri
    return urllib.parse.urlunsplit(parts._replace(netloc=parts.netloc.rsplit("@", 1)[1]))


def _fmt_time(ms):
    t = datetime.datetime.fromtimestamp(ms / 1000.0, datetime.timezone.utc)
    return f"{t:%m-%d %H:%MZ} ({t.astimezone(CHICAGO):%I:%M %p %Z})"


def compare(reader, run_ref, baseline_ref, experiment=DEFAULT_EXPERIMENT, epochs=None,
            now_ms=None):
    """The report as a list of lines. ``now_ms`` (default: now) dates the stale check."""
    base = reader.run(baseline_ref, experiment)
    run = reader.run(run_ref, experiment)
    keys = val_keys_of(base)
    cols = keys + [TRAIN_LOSS]
    hist = {side: {k: reader.history(r["info"]["run_id"], k) for k in cols}
            for side, r in (("run", run), ("base", base))}
    timing = hist["run"][keys[0]]
    secs = epoch_seconds(run, timing)
    # The run's own timing split, only if it logged one: runs from before the
    # trainer did keep exactly the report they had.
    logged = {m["key"] for m in run["data"].get("metrics", [])}
    split = [c for c in TIMING_COLS if c[0] in logged]
    split_hist = {key: reader.history(run["info"]["run_id"], key) for key, _, _ in split}
    # What the train clock saw outside data wait and step: the host-to-device
    # copy and loop overhead. Named only when all three parts were logged.
    other = all(k in logged for k in OTHER_PARTS)

    def name(r):
        i = r["info"]
        return f"{i.get('run_name')} ({i['run_id'][:8]}, {i['status']})"

    lines = [f"run      {name(run)}", f"baseline {name(base)}", ""]
    header = f"{'epoch':>5}" + "".join(f"  {k:>11} {'Δ':>7}" for k in cols) + f"  {'epoch_h':>7}"
    header += "".join(f"  {col:>7}" for _, col, _ in split)
    header += f"  {'other_h':>7}" if other else ""
    lines.append(header)
    for epoch in sorted(timing):
        row = f"{epoch:>5}"
        for k in cols:
            v = hist["run"][k].get(epoch)
            b = hist["base"][k].get(epoch)
            row += f"  {v[0]:>11.3f}" if v else f"  {'-':>11}"
            row += f" {v[0] - b[0]:>+7.3f}" if (v and b) else f" {'-':>7}"
        row += f"  {secs[epoch] / 3600:>7.2f}"
        for key, _, div in split:
            v = split_hist[key].get(epoch)
            row += f"  {v[0] / div:>7.2f}" if v else f"  {'-':>7}"
        if other:
            parts = [split_hist[k].get(epoch) for k in OTHER_PARTS]
            row += (f"  {(parts[0][0] - parts[1][0] - parts[2][0]) / 3600:>7.2f}"
                    if all(parts) else f"  {'-':>7}")
        lines.append(row)
    if not timing:
        lines.append("  (no epochs logged yet)")

    lines.append("")
    base_best = max(hist["base"][PRIMARY].items(), key=lambda e: e[1][0]) if hist["base"].get(PRIMARY) else None
    run_best = max(hist["run"][PRIMARY].items(), key=lambda e: e[1][0]) if hist["run"].get(PRIMARY) else None
    if base_best:
        lines.append(f"baseline best {PRIMARY} {base_best[1][0]:.3f} at epoch {base_best[0]}")
    if run_best:
        lines.append(f"run best      {PRIMARY} {run_best[1][0]:.3f} at epoch {run_best[0]}")

    planned = epochs if epochs is not None else planned_epochs(run)
    done = len(timing)
    if timing:
        mean = sum(secs.values()) / len(secs)
        lines.append(f"mean epoch {mean / 3600:.2f} h over {done} epoch(s)")
        last_ts = timing[max(timing)][1]
        finish = None
        remaining = None
        if run["info"]["status"] != "RUNNING":
            lines.append(f"run is {run['info']['status']}; no projection")
        elif planned is None:
            lines.append("planned epochs unknown (no 'max epochs' param); pass --epochs")
        else:
            remaining = max(planned - done, 0)
            finish = last_ts + remaining * mean * 1000
            lines.append(f"projected finish of {planned} epochs ({remaining} to go): {_fmt_time(finish)}")
        if run["info"]["status"] == "RUNNING":
            # MLflow leaves a run RUNNING when its pod dies, so a RUNNING run
            # that has stopped logging is likely dead rather than slow.
            now = time.time() * 1000 if now_ms is None else now_ms
            why = []
            # With no epochs left the run is finalising (test pass, upload):
            # "finish has passed" would fire at once, so only age counts.
            if finish is not None and finish < now and remaining > 0:
                why.append(f"the projected finish {_fmt_time(finish)} has passed")
            if now - last_ts > 2 * mean * 1000:
                why.append(f"the last val log was {(now - last_ts) / 3.6e6:.1f} h ago, "
                           f"over 2x the mean epoch")
            if why:
                lines.append("stale? " + "; ".join(why) + "; check the run is still alive")
    return lines


DATASETS = {"7": "COCO"}  # experiment id -> what its val set is called


def _fmt_chicago(ms, now_ms):
    """``7:10 PM CDT``, plus the weekday when it is not today in Chicago."""
    t = datetime.datetime.fromtimestamp(ms / 1000.0, CHICAGO)
    today = datetime.datetime.fromtimestamp(now_ms / 1000.0, CHICAGO).date()
    out = f"{t:%I:%M %p}".lstrip("0") + f" {t:%Z}"
    return out if t.date() == today else f"{out} {t:%a}"


def brief(reader, run_ref, baseline_ref, experiment=DEFAULT_EXPERIMENT, epochs=None,
          now_ms=None):
    """One phone-sized line for an owner report on a running run."""
    now = time.time() * 1000 if now_ms is None else now_ms
    base = reader.run(baseline_ref, experiment)
    run = reader.run(run_ref, experiment)
    if PRIMARY not in val_keys_of(base):
        raise CompareError(f"baseline {base['info'].get('run_name')} logged no {PRIMARY}")
    timing = reader.history(run["info"]["run_id"], PRIMARY)
    base_timing = reader.history(base["info"]["run_id"], PRIMARY)
    label = f"{DATASETS.get(experiment, 'exp ' + experiment)} val"
    planned = epochs if epochs is not None else planned_epochs(run)
    status = run["info"]["status"]
    best = max(v for v, _ in base_timing.values()) if base_timing else None

    if not timing:
        start = int(run["info"]["start_time"])
        line = f"No epochs logged yet ({(now - start) / 3.6e6:.1f} h in)."
        base_secs = epoch_seconds(base, base_timing)
        if status != "RUNNING":
            return f"{line} Run is {status}."
        if not base_secs:
            return line
        mean = sum(base_secs.values()) / len(base_secs)
        if planned is not None:
            finish = start + planned * mean * 1000
            line += (f" At the baseline's ~{mean / 3600:.1f} h/epoch, "
                     f"{planned} epochs done ~{_fmt_chicago(finish, now)}.")
        # No first epoch after 2x the baseline's epoch: likely a dead pod.
        if now - start > 2 * mean * 1000:
            line += " Over 2x the baseline's epoch with none logged; check it is alive."
        return line

    # Label by MLflow step (epoch = step + 1), not by how many epochs this run
    # logged: a resume into a new run starts past step 0.
    last = max(timing)
    n = last + 1
    value, last_ts = timing[last]
    at = base_timing.get(last)
    vs = (f"baseline at epoch {n}: {at[0]:.1f}" if at else f"baseline had no epoch {n}")
    vs += f", best {best:.2f}" if best is not None else ""
    secs = epoch_seconds(run, timing)
    mean = sum(secs.values()) / len(secs)
    line = f"Epoch {n}/{planned if planned is not None else '?'}: {label} {value:.1f} ({vs})."
    line += f" ~{mean / 3600:.1f} h/epoch"
    if status != "RUNNING":
        return f"{line}. Run is {status}."
    if planned is None:
        line += ", planned epochs unknown."
    elif planned <= n:
        line += ", all epochs logged, finishing up."
    else:
        line += f", done ~{_fmt_chicago(last_ts + (planned - n) * mean * 1000, now)}."
    # The full report's stale rule: MLflow leaves a dead pod's run RUNNING.
    if now - last_ts > 2 * mean * 1000:
        line += f" No val log for {(now - last_ts) / 3.6e6:.1f} h; check it is alive."
    return line


def main(argv=None, http=_default_http):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("run", help="run name (in --experiment) or 32-hex run id")
    p.add_argument("baseline", nargs="?", default=DEFAULT_BASELINE)
    p.add_argument("--experiment", default=DEFAULT_EXPERIMENT)
    p.add_argument("--epochs", type=int, default=None,
                   help="planned epochs for the projection (default: the run's 'max epochs' param)")
    p.add_argument("--brief", action="store_true",
                   help="one line for an owner report: latest epoch, baseline, finish in Chicago time")
    args = p.parse_args(argv)
    uri = os.environ.get("MLFLOW_TRACKING_URI") or MLFLOW_DEV
    shown = display_uri(uri)
    try:
        reader = MlflowReader(uri, http)
        if args.brief:
            print(brief(reader, args.run, args.baseline, args.experiment, args.epochs))
            return 0
        lines = compare(reader, args.run, args.baseline, args.experiment, args.epochs)
    except CompareError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    # HTTPError is a URLError; a timeout, or a 200 whose body is not JSON
    # (a proxy page), is the same "MLflow is not answering" failure.
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as e:
        print(f"error: MLflow at {shown}: {type(e).__name__}: {e}".replace(uri, shown),
              file=sys.stderr)
        return 2
    print(f"tracking {shown}")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
