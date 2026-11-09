"""Fetch the COCO val2017 or COCOA val data that ``eval_seg`` reads.

    python tools/fetch_seg_val.py coco  --data-root ~/eval/coco
    HF_TOKEN=... python tools/fetch_seg_val.py cocoa --data-root ~/eval/cocoa

Files already present are skipped, so an interrupted fetch resumes. Images
are fetched one by one from images.cocodataset.org: only the images the
annotation file references, never the full zips. Every request is bounded by
``FETCH_TIMEOUT_S``, and a failed image is retried ``DOWNLOAD_TRIES`` times
before the fetch stops: a transient stall costs a retry, not the whole run.

COCOA's amodal annotations come from the tarball the builder itself names,
in the gated HF repo shunk031/COCOA-annotation: HF_TOKEN must be set to a token
that has accepted its terms. ``--annotations-tar FILE`` uses a local copy
instead. The step is skipped when the JSON is already in place.

Standard library only, so it runs before any model dependency is installed.
"""

import argparse
import hashlib
import http.client
import json
import os
import shutil
import sys
import tarfile
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path

COCO_ANNOTATIONS_ZIP = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
COCO_ANNOTATIONS_MEMBER = "annotations/instances_val2017.json"
COCO_IMAGE_URL = "http://images.cocodataset.org/val2017/{}"

COCOA_ANNOTATIONS_JSON = "COCO_amodal_val2014.json"
# Where the builder itself fetches them (its _URLS["annotations"]). The repo is
# gated: anonymous requests get 401, so HF_TOKEN is sent when it is set.
COCOA_ANNOTATIONS_URL = (
    "https://huggingface.co/datasets/shunk031/COCOA-annotation/resolve/main/annotations.tar.gz"
)
COCOA_IMAGE_URL = "http://images.cocodataset.org/val2014/{}"
# The builder script, pinned to the revision the evaluations so far have used.
COCOA_BUILDER_REVISION = "492c23f50ca928f08a61614b1810eb0d3dc80c44"
COCOA_BUILDER_URL = (
    "https://huggingface.co/datasets/shunk031/COCOA/resolve/"
    f"{COCOA_BUILDER_REVISION}/COCOA.py"
)
COCOA_BUILDER_SHA256 = "f4233104204a1f615e5a30f7ad81f1577ceb6eb057317a9fb637d00e27ac4929"

# A request that stops answering must not outlive this. urllib hands it to
# http.client, which puts it on the socket: it bounds the connect and every
# read on it. Without one, a stalled read sits in getresponse until the kernel's
# own TCP timeout -- one such GET killed a 5000-image fetch after ~12 minutes.
FETCH_TIMEOUT_S = 60.0
# Tries per file, and the wait before the next one (doubling each time).
DOWNLOAD_TRIES = 3
DOWNLOAD_BACKOFF_S = 1.0


def log(*a):
    print(*a, file=sys.stderr, flush=True)


class _DropAuthOnHostChange(urllib.request.HTTPRedirectHandler):
    """Follow redirects, but never carry Authorization off its host or off TLS.

    urllib copies every request header onto the redirected request, so a
    Bearer token sent to huggingface.co would reach the CDN host it redirects
    to. This drops it whenever the host changes, or the redirect downgrades
    https to http on the same host, as requests does.
    """

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        new = super().redirect_request(req, fp, code, msg, headers, newurl)
        old, dest = urllib.parse.urlsplit(req.full_url), urllib.parse.urlsplit(newurl)
        if new is not None and (dest.hostname != old.hostname
                                or (old.scheme == "https" and dest.scheme == "http")):
            for name in list(new.headers):
                if name.lower() == "authorization":
                    del new.headers[name]
        return new


class _TimedOpener:
    """The guarded opener with a request timeout bound into it.

    The timeout belongs on the opener rather than at each call site, so that no
    function in this module can issue a request without one. ``handlers`` is
    the real ``OpenerDirector``'s, so the redirect guard stays inspectable.
    """

    def __init__(self, timeout):
        self.director = urllib.request.build_opener(_DropAuthOnHostChange)
        self.timeout = timeout

    @property
    def handlers(self):
        return self.director.handlers

    def open(self, req):
        return self.director.open(req, timeout=self.timeout)


def make_opener(timeout=None):
    """The opener every fetch uses: urllib's, with the redirect handler above.

    ``timeout`` defaults to ``FETCH_TIMEOUT_S``, read when the opener is made.
    The returned ``open`` is bound to an opener holding it, so every request
    through it is bounded -- see ``FETCH_TIMEOUT_S``.
    """
    return _TimedOpener(FETCH_TIMEOUT_S if timeout is None else timeout).open


_open = make_opener()


class FetchError(OSError):
    """A download that did not complete after its tries."""


def _retryable(exc):
    """Is another try worth it? A 4xx is the server's answer, not a hiccup."""
    if isinstance(exc, urllib.error.HTTPError):
        return exc.code >= 500
    return True


def download(url, dest, opener=_open, headers=None, sleep=time.sleep):
    """Download ``url`` to ``dest`` through a temporary file; skip if ``dest`` exists.

    A failed or timed-out try is retried up to ``DOWNLOAD_TRIES`` times, waiting
    longer each time. Every try starts from an empty ``.part``, so a retry never
    continues a half-written file, and ``dest`` appears only as the rename of a
    complete one.
    """
    dest = Path(dest)
    if dest.is_file() and dest.stat().st_size > 0:
        return False
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".part")
    req = urllib.request.Request(url, headers=headers) if headers else url
    for attempt in range(1, DOWNLOAD_TRIES + 1):
        try:
            with opener(req) as resp, open(tmp, "wb") as out:
                shutil.copyfileobj(resp, out)
        except (OSError, http.client.HTTPException) as exc:
            tmp.unlink(missing_ok=True)
            if attempt == DOWNLOAD_TRIES or not _retryable(exc):
                if isinstance(exc, urllib.error.HTTPError):
                    # The server's answer: it carries its own url and code, and
                    # fetch_cocoa's message for the gated repo is built from it.
                    raise
                raise FetchError(f"{url}: {exc}") from exc
            log(f"  {url}: {exc} -- retrying ({attempt}/{DOWNLOAD_TRIES})")
            sleep(DOWNLOAD_BACKOFF_S * 2 ** (attempt - 1))
        else:
            tmp.replace(dest)
            return True


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch_images(ann_json, image_dir, url_template, opener=_open):
    """Fetch every image ``ann_json`` lists into ``image_dir``. Returns (fetched, present)."""
    with open(ann_json) as f:
        images = json.load(f)["images"]
    fetched = present = 0
    for n, info in enumerate(images, 1):
        if download(url_template.format(info["file_name"]),
                    Path(image_dir) / info["file_name"], opener=opener):
            fetched += 1
        else:
            present += 1
        if n % 500 == 0:
            log(f"  images {n}/{len(images)}")
    return fetched, present


def extract_member(archive, member_name, dest):
    """Copy the member of a zip or tar archive whose basename is ``member_name``."""
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    base = Path(member_name).name
    if zipfile.is_zipfile(archive):
        with zipfile.ZipFile(archive) as z:
            for name in z.namelist():
                if Path(name).name == base:
                    with z.open(name) as src, open(dest, "wb") as out:
                        shutil.copyfileobj(src, out)
                    return
    else:
        with tarfile.open(archive) as t:
            for m in t.getmembers():
                if m.isfile() and Path(m.name).name == base:
                    with t.extractfile(m) as src, open(dest, "wb") as out:
                        shutil.copyfileobj(src, out)
                    return
    raise FileNotFoundError(f"{base} not found in {archive}")


def fetch_coco(root, opener=_open):
    root = Path(root)
    ann = root / "annotations" / "instances_val2017.json"
    if ann.is_file():
        log(f"present: {ann}")
    else:
        zip_path = root / "annotations_trainval2017.zip"
        log(f"fetching {COCO_ANNOTATIONS_ZIP} (about 240 MB)")
        download(COCO_ANNOTATIONS_ZIP, zip_path, opener=opener)
        extract_member(zip_path, COCO_ANNOTATIONS_MEMBER, ann)
        zip_path.unlink()
        log(f"wrote {ann}")
    fetched, present = fetch_images(ann, root / "val2017", COCO_IMAGE_URL, opener=opener)
    log(f"val2017 images: {fetched} fetched, {present} already present")


def fetch_cocoa(root, annotations_tar=None, opener=_open):
    root = Path(root)
    builder = root / "COCOA.py"
    if download(COCOA_BUILDER_URL, builder, opener=opener):
        log(f"fetched {builder} @ {COCOA_BUILDER_REVISION[:7]}")
    digest = sha256(builder)
    if digest != COCOA_BUILDER_SHA256:
        raise SystemExit(
            f"{builder} sha256 {digest} is not the pinned {COCOA_BUILDER_SHA256}; "
            f"delete it to refetch"
        )

    ann = root / "annotations" / COCOA_ANNOTATIONS_JSON
    if ann.is_file():
        log(f"present: {ann}")
    else:
        if not annotations_tar:
            annotations_tar = root / "annotations.tar.gz"
            token = os.environ.get("HF_TOKEN")
            headers = {"Authorization": f"Bearer {token}"} if token else None
            log(f"fetching {COCOA_ANNOTATIONS_URL}")
            try:
                download(COCOA_ANNOTATIONS_URL, annotations_tar, opener=opener, headers=headers)
            except urllib.error.HTTPError as e:
                if e.code in (401, 403):
                    raise SystemExit(
                        f"{COCOA_ANNOTATIONS_URL} answered {e.code}: the repo is gated. Set "
                        f"HF_TOKEN to a token that has accepted its terms, or pass "
                        f"--annotations-tar FILE."
                    ) from None
                # Anything else, including a 400 from the CDN host a redirect
                # led to, is reported as the HTTP failure it is, with the way
                # around it, rather than as a traceback. It names the
                # configured URL, never the error's own: after a redirect that
                # is the presigned CDN URL, signature included.
                raise SystemExit(
                    f"fetching the COCOA annotations failed: {COCOA_ANNOTATIONS_URL} "
                    f"answered {e.code} {e.reason}. Retry, or download "
                    f"{COCOA_ANNOTATIONS_URL} with your HF token yourself and pass "
                    f"--annotations-tar FILE."
                ) from None
        extract_member(annotations_tar, COCOA_ANNOTATIONS_JSON, ann)
        log(f"wrote {ann}")
    fetched, present = fetch_images(ann, root / "val2014", COCOA_IMAGE_URL, opener=opener)
    log(f"val2014 images: {fetched} fetched, {present} already present")


def main(argv=None):
    p = argparse.ArgumentParser(description="Fetch COCO val2017 or COCOA val for eval_seg.")
    p.add_argument("dataset", choices=("coco", "cocoa"))
    p.add_argument("--data-root", required=True)
    p.add_argument("--annotations-tar", default=None,
                   help="cocoa: a local copy of the amodal annotation tarball "
                        "(default: download it; needs HF_TOKEN)")
    args = p.parse_args(argv)
    if args.dataset == "coco":
        fetch_coco(args.data_root)
    else:
        fetch_cocoa(args.data_root, args.annotations_tar)
    return 0


if __name__ == "__main__":
    sys.exit(main())
