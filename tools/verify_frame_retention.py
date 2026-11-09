#!/usr/bin/env python3
"""Verify that captured camera frames really expire after 30 days (design D7).

The bucket carries an ILM expiry rule for ``round0/``. A config rule can be true
while the effect is broken, so this check asserts both: the rule is still the one
we asked for, and nothing under the prefix has outlived it. It reports what it
found and exits non-zero when either half fails, so the daily CronJob's failure
is the signal.

Exit codes:
  0  the rule is in place and every object is younger than 31 days
  1  retention is broken (rule absent/wrong, or an object is too old)
  2  the check could not run (missing environment)
  3  the bucket refused the read (the verifier credential lacks its grant)

Usage: MINIO_ENDPOINT=... MINIO_USER=... MINIO_PASSWORD=... \
       python tools/verify_frame_retention.py [--bucket B] [--prefix round0/]
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime, timedelta, timezone

BUCKET = "camera-uncertain-frames"
PREFIX = "round0/"
REQUIRED_DAYS = 30
MAX_AGE_DAYS = 31
DEFAULT_REGION = "us-east-1"

EXIT_OK, EXIT_BAD, EXIT_ENV, EXIT_REFUSED = 0, 1, 2, 3


def describe_rules(config):
    """The fields we judge, from minio-py's parsed ``LifecycleConfig``.

    Kept as plain dicts so the judgement below is testable without an SDK object
    and so a change in the SDK's attribute names fails loudly here, once.
    """
    rules = []
    for rule in (getattr(config, "rules", None) or []):
        expiration = getattr(rule, "expiration", None)
        rule_filter = getattr(rule, "rule_filter", None)
        status = getattr(rule, "status", "") or ""
        status = str(getattr(status, "value", status)).lower()
        rules.append({
            "id": getattr(rule, "rule_id", None),
            "enabled": "enabled" in status,
            "prefix": (getattr(rule_filter, "prefix", None) or "") if rule_filter else "",
            "days": getattr(expiration, "days", None) if expiration else None,
            "delete_marker_only": bool(getattr(expiration, "expired_object_delete_marker", False))
            if expiration else False,
        })
    return rules


def rule_violations(rules, prefix=PREFIX, required_days=REQUIRED_DAYS):
    """Whether some enabled rule expires everything under ``prefix`` in time."""
    covering = [r for r in rules
                if r["enabled"] and not r["delete_marker_only"]
                and r["days"] is not None
                and prefix.startswith(r["prefix"] or "")]
    if not rules:
        return ["no lifecycle rule is set on the bucket"]
    if not covering:
        return ["no enabled expiry rule covers %r: %s"
                % (prefix, _summarize(rules))]
    soonest = min(r["days"] for r in covering)
    if soonest > required_days:
        return ["the expiry rule covering %r allows %d days, more than the %d agreed"
                % (prefix, soonest, required_days)]
    return []


def _summarize(rules):
    return "; ".join(
        "%s=%s%s%s" % (r["id"] or "(no id)",
                       "%dd" % r["days"] if r["days"] is not None else "no-days",
                       "" if r["enabled"] else " disabled",
                       "" if not r["prefix"] else " prefix=%r" % r["prefix"])
        for r in rules)


def age_violations(objects, now, max_age_days=MAX_AGE_DAYS):
    """Objects older than the tolerated age; ``objects`` is (key, last_modified)."""
    limit = timedelta(days=max_age_days)
    return ["%s is %.1f days old" % (key, (now - when).total_seconds() / 86400.0)
            for key, when in objects if (now - when) > limit]


def check(client, bucket=BUCKET, prefix=PREFIX, now=None):
    """Run both halves against a Minio-like client; return (lines, exit code)."""
    now = now or datetime.now(timezone.utc)
    lines = []
    failures = []

    try:
        config = client.get_bucket_lifecycle(bucket)
    except Exception as exc:  # noqa: BLE001 - classified below, never swallowed
        code = getattr(exc, "code", "")
        if code in ("AccessDenied", "SignatureDoesNotMatch", "InvalidAccessKeyId"):
            return (["the bucket refused the lifecycle read (%s); the verifier "
                     "credential needs ListBucket and the lifecycle read on %s"
                     % (code, bucket)], EXIT_REFUSED)
        raise

    rules = describe_rules(config)
    bad = rule_violations(rules, prefix=prefix)
    if bad:
        failures.extend(bad)
        lines.append("rule: FAIL -- " + "; ".join(bad))
    else:
        lines.append("rule: ok -- " + _summarize(rules) + " (required <= %d days on %r)"
                     % (REQUIRED_DAYS, prefix))

    try:
        objects = client.list_objects(bucket, prefix=prefix, recursive=True)
        oldest = None
        count = 0
        for obj in objects:
            count += 1
            if oldest is None or obj.last_modified < oldest[1]:
                oldest = (obj.object_name, obj.last_modified)
    except Exception as exc:  # noqa: BLE001
        code = getattr(exc, "code", "")
        if code in ("AccessDenied", "SignatureDoesNotMatch", "InvalidAccessKeyId"):
            return (lines + ["the bucket refused the object list (%s); the verifier "
                             "credential needs ListBucket on %s" % (code, bucket)],
                    EXIT_REFUSED)
        raise

    if oldest is None:
        lines.append("objects: none under %r yet" % prefix)
    else:
        age = (now - oldest[1]).total_seconds() / 86400.0
        lines.append("objects: %d under %r, oldest %s at %.1f days"
                     % (count, prefix, oldest[0], age))
        # The oldest object decides: if it is inside the limit, every other one is too.
        failures.extend(age_violations([oldest], now))

    if failures:
        for line in failures:
            lines.append("FAIL " + line)
        return lines, EXIT_BAD
    lines.append("PASS: the rule is in place and nothing under %r is older than %d days"
                 % (prefix, MAX_AGE_DAYS))
    return lines, EXIT_OK


def _client(env):
    from minio import Minio

    missing = [name for name in ("MINIO_ENDPOINT", "MINIO_USER", "MINIO_PASSWORD")
               if not (env.get(name) or "").strip()]
    if missing:
        raise SystemExit("cannot run: " + ", ".join(missing) + " is unset")
    return Minio(env["MINIO_ENDPOINT"].strip(),
                 access_key=env["MINIO_USER"].strip(),
                 secret_key=env["MINIO_PASSWORD"].strip(),
                 secure=(env.get("MINIO_SECURE") or "").strip().lower()
                 in ("1", "true", "yes", "on"),
                 region=(env.get("MINIO_REGION") or "").strip() or DEFAULT_REGION)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bucket", default=os.environ.get("CAMERA_FRAMES_BUCKET", BUCKET))
    parser.add_argument("--prefix", default=PREFIX)
    args = parser.parse_args(argv)

    try:
        client = _client(os.environ)
    except SystemExit as exc:
        print(str(exc), file=sys.stderr)
        return EXIT_ENV

    lines, code = check(client, bucket=args.bucket, prefix=args.prefix)
    for line in lines:
        print(line, flush=True)
    return code


if __name__ == "__main__":
    sys.exit(main())
