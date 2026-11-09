"""Upload captured frames to the home-cluster MinIO, off the inference loop.

The rules the spec fixes (design D10):

- Credentials come from the environment, never from the repo. The credential is
  upload-only (``PutObject`` + ``AbortMultipartUpload`` on ``round0/*``); this
  module cannot list, read or delete objects, so a stolen key yields no data.
- Uploads run on their own thread and retry with backoff while the bucket is
  unreachable. Nothing here raises into the demo: a dead network degrades the
  upload, it does not stop the capture or the camera.
- A frame is marked (``<name>.uploaded``) only after a confirmed upload, and
  only then deleted locally. Until the marker exists the local file is the only
  copy, so pruning never removes it.
"""

from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path

from jetson.uncertain_frames import Counters
from jetson.frame_writer import UPLOADED_SUFFIX

SCAN_PERIOD_S = 30.0
BACKOFF_BASE_S = 2.0
BACKOFF_MAX_S = 120.0
RETRY_ATTEMPTS = 6
DEFAULT_BUCKET = "camera-uncertain-frames"
DEFAULT_PREFIX = "round0/"
DEFAULT_REGION = "us-east-1"

REQUIRED_ENV = ("MINIO_ENDPOINT", "MINIO_USER", "MINIO_PASSWORD")
_CRED_FILES = {"MINIO_USER": "MINIO_USER_FILE",
               "MINIO_PASSWORD": "MINIO_PASSWORD_FILE"}
TRUE = {"1", "true", "yes", "on"}


class CaptureConfigError(RuntimeError):
    """Capture was asked for but the environment cannot support it.

    Raised before the camera is opened, naming the missing variable, so a
    capture never starts in a state where it would silently fail to upload.
    """


def _secret(env, name):
    """A credential from ``ENV`` or from the file named by ``ENV_FILE``."""
    value = (env.get(name) or "").strip()
    if value:
        return value
    path = (env.get(_CRED_FILES[name]) or "").strip()
    if path:
        try:
            return Path(path).read_text(encoding="utf-8").strip()
        except OSError as exc:
            raise CaptureConfigError(f"{_CRED_FILES[name]} points at {path}, which cannot be read: {exc}")
    return ""


@dataclass(frozen=True)
class MinioConfig:
    endpoint: str
    user: str
    password: str
    bucket: str
    prefix: str
    secure: bool
    region: str = DEFAULT_REGION

    @property
    def scheme(self):
        return "https" if self.secure else "http"

    @property
    def url(self):
        return f"{self.scheme}://{self.endpoint}"

    @classmethod
    def from_env(cls, env=None):
        """Read and validate ``MINIO_*``; raise naming every missing variable."""
        env = os.environ if env is None else env
        missing = []
        for name in REQUIRED_ENV:
            if name in _CRED_FILES:
                if not _secret(env, name):
                    missing.append(name)
            elif not (env.get(name) or "").strip():
                missing.append(name)
        if missing:
            raise CaptureConfigError(
                "capture needs " + ", ".join(missing)
                + " (a credential may be given inline or by the file named in "
                  "MINIO_USER_FILE / MINIO_PASSWORD_FILE)"
            )
        prefix = (env.get("MINIO_PREFIX") or DEFAULT_PREFIX).strip()
        if not prefix.endswith("/"):
            prefix += "/"
        return cls(
            endpoint=env["MINIO_ENDPOINT"].strip(),
            user=_secret(env, "MINIO_USER"),
            password=_secret(env, "MINIO_PASSWORD"),
            bucket=(env.get("MINIO_BUCKET") or DEFAULT_BUCKET).strip(),
            prefix=prefix,
            secure=(env.get("MINIO_SECURE") or "").strip().lower() in TRUE,
            region=(env.get("MINIO_REGION") or "").strip() or DEFAULT_REGION,
        )


def object_key(prefix, day, filename):
    """``round0/<day>/<file>`` -- the layout the expiry rule and the verifier use."""
    return f"{prefix}{day}/{filename}"


class FrameUploader:
    """Thread that keeps pushing unconfirmed frames to the bucket."""

    def __init__(self, out_dir, config, *, put=None, counts=None,
                 scan_period=SCAN_PERIOD_S, attempts=RETRY_ATTEMPTS,
                 backoff_base=BACKOFF_BASE_S, backoff_max=BACKOFF_MAX_S,
                 sleep=time.sleep):
        self.out_dir = Path(out_dir)
        self.config = config
        self.counts = counts if counts is not None else Counters()
        self.scan_period = float(scan_period)
        self.attempts = int(attempts)
        self.backoff_base = float(backoff_base)
        self.backoff_max = float(backoff_max)
        self._sleep = sleep
        self._put = put if put is not None else self._minio_put()
        self._thread = None
        self._closed = threading.Event()
        self._last_error = None
        self._pending = {}  # path -> attempts already spent on it

    # -- construction ------------------------------------------------------

    def _minio_put(self):
        """The real put: a Minio client built from the validated config.

        ``region`` is required, not cosmetic: without it minio-py calls
        GetBucketLocation before every object write, and the upload-only
        policy grants PutObject + AbortMultipartUpload on ``round0/*`` and
        nothing else, so every upload is denied. Supplying the region means
        the client never asks, and the policy stays as narrow as it is.
        """
        from minio import Minio

        client = Minio(self.config.endpoint, access_key=self.config.user,
                       secret_key=self.config.password, secure=self.config.secure,
                       region=self.config.region)

        def put(key, path, content_type):
            client.fput_object(self.config.bucket, key, str(path),
                               content_type=content_type)
        return put

    def describe(self):
        """One line naming the scheme, so an http capture cannot go unnoticed."""
        return (f"[uploader] {self.config.scheme}://{self.config.endpoint} "
                f"bucket={self.config.bucket} prefix={self.config.prefix}")

    # -- lifecycle ---------------------------------------------------------

    def start(self):
        if self._thread is None or not self._thread.is_alive():
            self._closed.clear()
            self._thread = threading.Thread(target=self._run, name="frame-uploader", daemon=True)
            self._thread.start()
        return self

    def stop(self, timeout=10.0):
        self._closed.set()
        if self._thread is not None:
            self._thread.join(timeout)
        return self

    def _run(self):
        while not self._closed.is_set():
            try:
                self.sweep()
            except Exception as exc:  # the loop must survive any upload bug
                self._last_error = repr(exc)
                self.counts.bump("upload_error")
            self._closed.wait(self.scan_period)

    # -- work --------------------------------------------------------------

    def sweep(self):
        """One pass: upload every unconfirmed frame, oldest first. Returns count."""
        done = 0
        for frame in self._pending_frames():
            if self._closed.is_set():
                break
            if self.upload(frame):
                done += 1
            else:
                # One failure means the bucket is unreachable; the next scan
                # retries from the oldest frame rather than hammering the rest.
                break
        return done

    def _pending_frames(self):
        frames = []
        if not self.out_dir.exists():
            return frames
        for path in self.out_dir.glob("*/[!.]*.jpg"):
            if Path(str(path) + UPLOADED_SUFFIX).exists():
                continue
            try:
                frames.append((path.stat().st_mtime, path))
            except OSError:
                continue
        return [path for _mtime, path in sorted(frames)]

    def upload(self, frame_path):
        """Upload one frame and its sidecar, then mark and delete. True on success."""
        frame_path = Path(frame_path)
        day = frame_path.parent.name
        side = frame_path.with_suffix(".json")
        marker = Path(str(frame_path) + UPLOADED_SUFFIX)
        try:
            self._put(object_key(self.config.prefix, day, frame_path.name),
                      frame_path, "image/jpeg")
            if side.exists():
                self._put(object_key(self.config.prefix, day, side.name),
                          side, "application/json")
        except Exception as exc:
            self._on_failure(frame_path, exc)
            return False
        try:
            marker.write_text("", encoding="utf-8")
        except OSError as exc:
            # Without the marker the frame stays and is uploaded again: a
            # duplicate object is harmless, a lost marker is not.
            self._on_failure(frame_path, exc)
            return False
        for path in (frame_path, side):
            try:
                path.unlink()
            except OSError:
                pass
        self._pending.pop(frame_path, None)
        self.counts.bump("uploaded")
        return True

    def _on_failure(self, frame_path, exc):
        tries = self._pending.get(frame_path, 0) + 1
        self._pending[frame_path] = tries
        self._last_error = repr(exc)
        self.counts.bump("upload_failed")
        if tries < self.attempts:
            self._sleep(self._backoff(tries))

    def _backoff(self, tries):
        """Exponential backoff, capped: 2, 4, 8, ... <= 120 s."""
        return min(self.backoff_max, self.backoff_base * (2 ** (tries - 1)))

    def stats(self):
        out = self.counts.as_dict()
        out["upload_pending"] = len(self._pending)
        out["upload_error_last"] = self._last_error
        return out
