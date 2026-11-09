"""The retention verifier (tools/verify_frame_retention.py, design D7).

A lifecycle rule can be configured and still let frames live forever -- the rule
can be dropped, narrowed to another prefix, or stretched -- so the check judges
both the rule and the oldest object that actually exists. These pin the two
outcomes the daily CronJob turns on: a 32-day-old object fails the run, a young
one passes it.
"""

from __future__ import annotations

import importlib.util
import sys
import types
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

_spec = importlib.util.spec_from_file_location(
    "verify_frame_retention", ROOT / "tools" / "verify_frame_retention.py")
verify = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(verify)

NOW = datetime(2026, 10, 28, 12, 0, tzinfo=timezone.utc)


class _Status:
    def __init__(self, value):
        self.value = value


class _Expiration:
    def __init__(self, days):
        self.days = days
        self.expired_object_delete_marker = False


class _Filter:
    def __init__(self, prefix):
        self.prefix = prefix


class _Rule:
    def __init__(self, days=30, prefix="round0/", status="Enabled", rule_id="ilm-1"):
        self.rule_id = rule_id
        self.status = _Status(status)
        self.expiration = _Expiration(days)
        self.rule_filter = _Filter(prefix)


class _Config:
    def __init__(self, rules):
        self.rules = rules


class _Object:
    def __init__(self, name, age_days):
        self.object_name = name
        self.last_modified = NOW - timedelta(days=age_days)


class _Refused(Exception):
    def __init__(self, code="AccessDenied"):
        super().__init__(code)
        self.code = code


class _Bucket:
    """A Minio-like client standing in for the fixture bucket."""

    def __init__(self, rules=None, objects=(), refuse=None):
        self._rules = rules if rules is not None else [_Rule()]
        self._objects = list(objects)
        self._refuse = refuse or set()

    def get_bucket_lifecycle(self, bucket):
        if "lifecycle" in self._refuse:
            raise _Refused()
        return _Config(self._rules)

    def list_objects(self, bucket, prefix="", recursive=False):
        if "list" in self._refuse:
            raise _Refused()
        return iter(self._objects)


def _check(client):
    return verify.check(client, bucket="camera-uncertain-frames", now=NOW)


def test_an_object_past_31_days_fails_the_run():
    """The fixture the task asks for: the check must fail on a 32-day-old object."""
    lines, code = _check(_Bucket(objects=[_Object("round0/20260926/a.jpg", 32),
                                         _Object("round0/20261027/b.jpg", 1)]))
    assert code == verify.EXIT_BAD
    joined = "\n".join(lines)
    assert "round0/20260926/a.jpg" in joined
    assert "32.0 days old" in joined


def test_a_young_object_passes_and_the_rule_is_reported():
    lines, code = _check(_Bucket(objects=[_Object("round0/20261027/b.jpg", 1)]))
    assert code == verify.EXIT_OK
    joined = "\n".join(lines)
    assert "30d" in joined and "round0/" in joined
    assert "PASS" in joined


def test_an_empty_prefix_passes_but_says_so():
    """No frames yet is not evidence of a broken rule; it is also not evidence of one."""
    lines, code = _check(_Bucket(objects=[]))
    assert code == verify.EXIT_OK
    assert "none under 'round0/'" in "\n".join(lines)


def test_an_absent_rule_fails_even_when_the_objects_are_young():
    lines, code = _check(_Bucket(rules=[], objects=[_Object("round0/x.jpg", 1)]))
    assert code == verify.EXIT_BAD
    assert "no lifecycle rule" in "\n".join(lines)


def test_a_rule_on_another_prefix_does_not_cover_round0():
    lines, code = _check(_Bucket(rules=[_Rule(prefix="hold/")],
                                 objects=[_Object("round0/x.jpg", 1)]))
    assert code == verify.EXIT_BAD
    assert "no enabled expiry rule covers 'round0/'" in "\n".join(lines)


def test_a_rule_that_allows_longer_than_30_days_fails():
    lines, code = _check(_Bucket(rules=[_Rule(days=45)],
                                 objects=[_Object("round0/x.jpg", 1)]))
    assert code == verify.EXIT_BAD
    assert "45 days" in "\n".join(lines)


def test_a_disabled_rule_does_not_count():
    lines, code = _check(_Bucket(rules=[_Rule(status="Disabled")],
                                 objects=[_Object("round0/x.jpg", 1)]))
    assert code == verify.EXIT_BAD


def test_an_object_just_inside_the_tolerance_passes():
    """31 days is the tolerance's edge: expiry runs daily, so a day of slack is not a fault."""
    lines, code = _check(_Bucket(objects=[_Object("round0/x.jpg", 30.9)]))
    assert code == verify.EXIT_OK


def test_a_refused_lifecycle_read_is_reported_as_a_credential_problem():
    lines, code = _check(_Bucket(refuse={"lifecycle"}))
    assert code == verify.EXIT_REFUSED
    assert "refused the lifecycle read" in "\n".join(lines)


def test_a_refused_object_list_is_reported_as_a_credential_problem():
    lines, code = _check(_Bucket(refuse={"list"}))
    assert code == verify.EXIT_REFUSED
    assert "refused the object list" in "\n".join(lines)


def test_the_client_is_built_with_a_region(monkeypatch):
    """minio-py asks the bucket for its location before a write unless told the region,
    and that question is exactly what a scoped bucket policy refuses."""
    captured = {}
    module = types.ModuleType("minio")

    def _minio(endpoint, access_key=None, secret_key=None, secure=True, region=None):
        captured.update(endpoint=endpoint, secure=secure, region=region)
        return object()

    module.Minio = _minio
    monkeypatch.setitem(sys.modules, "minio", module)

    verify._client({"MINIO_ENDPOINT": "minio.example:9000", "MINIO_USER": "u",
                    "MINIO_PASSWORD": "p"})
    assert captured["region"] == "us-east-1"

    verify._client({"MINIO_ENDPOINT": "minio.example:9000", "MINIO_USER": "u",
                    "MINIO_PASSWORD": "p", "MINIO_REGION": "eu-west-1"})
    assert captured["region"] == "eu-west-1"


def test_missing_environment_exits_before_any_client_is_built(monkeypatch):
    module = types.ModuleType("minio")

    def _boom(*a, **k):
        raise AssertionError("a client was built without its environment")

    module.Minio = _boom
    monkeypatch.setitem(sys.modules, "minio", module)
    monkeypatch.setattr(verify.os, "environ", {"MINIO_ENDPOINT": "minio.example:9000"})
    assert verify.main([]) == verify.EXIT_ENV
