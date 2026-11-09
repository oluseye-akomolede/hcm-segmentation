"""Uploader retries, marking and credential configuration (jetson/frame_uploader.py)."""

import json
import sys
import types
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

from jetson.frame_uploader import (
    BACKOFF_MAX_S,
    CaptureConfigError,
    FrameUploader,
    MinioConfig,
    object_key,
)
from jetson.uncertain_frames import Counters

LOCAL = ZoneInfo("America/Chicago")


def _ts(y, mo, d, h=0, mi=0, s=0):
    return datetime(y, mo, d, h, mi, s, tzinfo=LOCAL).timestamp()


ENV = {"MINIO_ENDPOINT": "minio.minio.svc.cluster.local:9000",
       "MINIO_USER": "hcm-camera-upload",
       "MINIO_PASSWORD": "s3cret"}


class FakeBucket:
    """A put target that can be told to fail a few times."""

    def __init__(self, failures=0):
        self.failures = failures
        self.objects = {}

    def put(self, key, path, content_type):
        if self.failures:
            self.failures -= 1
            raise ConnectionError("bucket unreachable")
        self.objects[key] = (path.read_bytes(), content_type)


@pytest.fixture()
def capture_dir(tmp_path):
    day = tmp_path / "20261102"
    day.mkdir()
    (day / "1700000000000.jpg").write_bytes(b"JPEG")
    (day / "1700000000000.json").write_text(json.dumps({"file": "1700000000000.jpg"}))
    return tmp_path


def _uploader(capture_dir, bucket, **kw):
    sleeps = []
    kw.setdefault("sleep", sleeps.append)
    counts = Counters()
    up = FrameUploader(capture_dir, MinioConfig.from_env(ENV), put=bucket.put,
                       counts=counts, **kw)
    return up, sleeps


class TestConfig:
    def test_a_missing_variable_is_named_and_refused(self):
        with pytest.raises(CaptureConfigError) as err:
            MinioConfig.from_env({"MINIO_ENDPOINT": ENV["MINIO_ENDPOINT"]})
        assert "MINIO_USER" in str(err.value) and "MINIO_PASSWORD" in str(err.value)

    def test_the_scheme_follows_minio_secure(self):
        assert MinioConfig.from_env(ENV).scheme == "http"
        secure = dict(ENV, MINIO_SECURE="true")
        assert MinioConfig.from_env(secure).scheme == "https"

    def test_the_secure_flag_is_read_as_a_boolean(self):
        for value, expected in (("1", True), ("true", True), ("yes", True),
                                ("on", True), ("0", False), ("false", False),
                                ("", False)):
            env = dict(ENV, MINIO_SECURE=value)
            assert MinioConfig.from_env(env).secure is expected

    def test_a_credential_can_come_from_a_file(self, tmp_path):
        secret_file = tmp_path / "pw"
        secret_file.write_text("from-file\n")
        env = {"MINIO_ENDPOINT": ENV["MINIO_ENDPOINT"], "MINIO_USER": "u",
               "MINIO_PASSWORD_FILE": str(secret_file)}
        assert MinioConfig.from_env(env).password == "from-file"

    def test_the_bucket_defaults_and_the_prefix_ends_in_a_slash(self):
        config = MinioConfig.from_env(ENV)
        assert config.bucket == "camera-uncertain-frames"
        assert config.prefix == "round0/"

    def test_the_object_layout_is_round_zero_by_day(self):
        assert object_key("round0/", "20261102", "x.jpg") == "round0/20261102/x.jpg"

    def test_the_region_defaults_and_can_be_overridden(self):
        assert MinioConfig.from_env(ENV).region == "us-east-1"
        assert MinioConfig.from_env(dict(ENV, MINIO_REGION="eu-west-1")).region == "eu-west-1"
        assert MinioConfig.from_env(dict(ENV, MINIO_REGION="   ")).region == "us-east-1"


class TestClientConstruction:
    def test_the_client_is_built_with_a_region(self, monkeypatch):
        """Without a region minio-py asks GetBucketLocation, which the policy denies."""
        seen = {}

        class FakeMinio:
            def __init__(self, endpoint, **kwargs):
                seen["endpoint"] = endpoint
                seen.update(kwargs)

            def fput_object(self, bucket, key, path, content_type=None):
                seen["put"] = (bucket, key, path, content_type)

        monkeypatch.setitem(sys.modules, "minio",
                            types.SimpleNamespace(Minio=FakeMinio))
        config = MinioConfig.from_env(dict(ENV, MINIO_SECURE="true"))
        put = FrameUploader(Path("/nonexistent"), config)._minio_put()

        assert seen["endpoint"] == ENV["MINIO_ENDPOINT"]
        assert seen["region"] == "us-east-1"
        assert seen["access_key"] == ENV["MINIO_USER"]
        assert seen["secret_key"] == ENV["MINIO_PASSWORD"]
        assert seen["secure"] is True

        key = object_key(config.prefix, "20261102", "1700000000000.jpg")
        put(key, Path("/tmp/1700000000000.jpg"), "image/jpeg")
        assert seen["put"] == ("camera-uncertain-frames", key,
                               "/tmp/1700000000000.jpg", "image/jpeg")

    def test_the_override_reaches_the_client(self, monkeypatch):
        seen = {}

        class FakeMinio:
            def __init__(self, endpoint, **kwargs):
                seen.update(kwargs)

        monkeypatch.setitem(sys.modules, "minio",
                            types.SimpleNamespace(Minio=FakeMinio))
        config = MinioConfig.from_env(dict(ENV, MINIO_REGION="eu-west-1"))
        FrameUploader(Path("/nonexistent"), config)
        assert seen["region"] == "eu-west-1"


class TestUpload:
    def test_a_confirmed_upload_marks_then_deletes(self, capture_dir):
        bucket = FakeBucket()
        up, _ = _uploader(capture_dir, bucket)
        assert up.upload(capture_dir / "20261102" / "1700000000000.jpg") is True
        assert set(bucket.objects) == {"round0/20261102/1700000000000.jpg",
                                       "round0/20261102/1700000000000.json"}
        assert not (capture_dir / "20261102" / "1700000000000.jpg").exists()
        marker = capture_dir / "20261102" / "1700000000000.jpg.uploaded"
        assert marker.exists()
        assert up.counts.get("uploaded") == 1

    def test_a_failed_upload_keeps_the_file_and_is_retried(self, capture_dir):
        bucket = FakeBucket(failures=1)
        up, sleeps = _uploader(capture_dir, bucket)
        path = capture_dir / "20261102" / "1700000000000.jpg"
        assert up.upload(path) is False
        assert path.exists()
        assert not (capture_dir / "20261102" / "1700000000000.jpg.uploaded").exists()
        assert up.counts.get("upload_failed") == 1
        assert sleeps and sleeps[0] > 0
        assert up.upload(path) is True          # the retry succeeds
        assert up.counts.get("uploaded") == 1

    def test_a_retry_reuses_the_key_so_there_is_one_object_per_frame(self, capture_dir):
        bucket = FakeBucket(failures=2)
        up, sleeps = _uploader(capture_dir, bucket)
        path = capture_dir / "20261102" / "1700000000000.jpg"
        up.upload(path)          # fails
        up.upload(path)          # fails
        up.upload(path)          # succeeds
        assert len(bucket.objects) == 2          # the jpeg and its record
        assert list(bucket.objects) == ["round0/20261102/1700000000000.jpg",
                                        "round0/20261102/1700000000000.json"]

    def test_backoff_grows_and_is_capped(self, capture_dir):
        bucket = FakeBucket(failures=10)
        up, sleeps = _uploader(capture_dir, bucket)
        for _ in range(8):
            up.upload(capture_dir / "20261102" / "1700000000000.jpg")
        assert sleeps[0] < sleeps[-1]
        assert max(sleeps) <= BACKOFF_MAX_S

    def test_a_failure_never_raises_into_the_caller(self, capture_dir):
        bucket = FakeBucket(failures=99)
        up, _ = _uploader(capture_dir, bucket)
        assert up.sweep() == 0           # no exception escapes

    def test_a_marked_frame_is_not_uploaded_again(self, capture_dir):
        bucket = FakeBucket()
        up, _ = _uploader(capture_dir, bucket)
        path = capture_dir / "20261102" / "1700000000000.jpg"
        (capture_dir / "20261102" / "1700000000000.jpg.uploaded").write_text("")
        assert up.sweep() == 0
        assert bucket.objects == {}

    def test_the_uploaded_record_is_the_index_record(self, capture_dir):
        bucket = FakeBucket()
        up, _ = _uploader(capture_dir, bucket)
        up.upload(capture_dir / "20261102" / "1700000000000.jpg")
        doc = json.loads(bucket.objects["round0/20261102/1700000000000.json"][0])
        assert doc == {"file": "1700000000000.jpg"}
        assert bucket.objects["round0/20261102/1700000000000.jpg"][1] == "image/jpeg"
