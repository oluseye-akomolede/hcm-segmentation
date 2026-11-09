"""Turn the BDD100K zips into the prepared tree the trainer and the eval read.

Input: the two published archives --
``bdd100k_images_10k.zip`` (JPEG images) and ``bdd100k_seg_maps.zip`` (one
class-index PNG per image, 255 = unknown) -- and ``SHA256SUMS`` beside them.

Output, under ``--out``::

    images/<split>/<stem>.jpg
    masks/<split>/<stem>.png       # 8-bit, 0 or 255: foreground object or not
    <split>.txt                    # one stem per line, sorted
    manifest.json                  # counts, the class map, the sources' sizes

The mask is the object/background target from ``ecs_hcm_model.core.bdd``: the
eight object classes plus traffic lights are foreground and every other class,
including unknown, is background. The mapping is not repeated here.

Counts are checked, not assumed: the published set is 7,000 train and 1,000 val
seg maps, and a run that prepared fewer than it can see in the archives fails
rather than shipping a silently truncated tree. Pairs whose image and label do
not share a frame (six in the published set) are excluded and listed in the
manifest, and they count towards the total. ``--allow-subset`` is for the tiny
fixtures in the tests, which prepare a handful of rows end to end.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import re
import sys
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image

from ecs_hcm_model.core.bdd import (
    BDD100K_CLASSES,
    BDD_FOREGROUND_CLASSES,
    BDD_UNKNOWN_INDEX,
    PREPARED_MANIFEST_FORMAT,
    foreground_class_ids,
    object_mask_from_class_index,
)

EXPECTED_COUNTS = {"train": 7000, "val": 1000}
SPLITS = ("train", "val")
RAW_FILES = ("bdd100k_images_10k.zip", "bdd100k_seg_maps.zip")
# The class-id map's filename suffix: ``<id>_<token>_id.png``. The token is the
# *dataset group*, not the split — every entry uses ``_train_``, including the val
# split's (``labels/val/<id>_train_id.png``). Only the folder says which split a
# map belongs to.
LABEL_NAME = re.compile(r"(?P<stem>.+)_(?P<token>[A-Za-z]+)_id\.png")
FOREGROUND = 255  # the value written into the prepared mask


class PrepareError(RuntimeError):
    """The archives and the expected layout disagree."""


def _members(zf: zipfile.ZipFile, split: str, kind: str) -> dict[str, str]:
    """``{stem: member name}`` for one split of one archive.

    ``kind`` is ``"images"`` (JPEGs) or ``"labels"`` (seg PNGs). The published
    archives nest differently from each other (``10k/train/<id>.jpg`` against
    ``labels/train/<id>_train_id.png``), so the match is on the path's components
    rather than a fixed prefix.

    The label archive holds **two** maps per image: ``labels/<split>/
    <id>_train_id.png`` (class indices, the one that carries the target) and
    ``color_labels/<split>/<id>_train_color.png`` (a colourised preview). Only
    entries under ``labels/`` ending ``_id.png`` are labels; the colour maps are
    previews and would otherwise double the label count.

    The split is taken from the **folder** (``labels/val`` is val), never from
    the filename: the published zip names every map with a ``_train_`` token,
    val included, so keying off the token would drop all 1000 val rows.
    """
    wanted = {"images": ".jpg", "labels": "_id.png"}[kind]
    found = {}
    for name in zf.namelist():
        if name.endswith("/") or not name.lower().endswith(wanted):
            continue
        parts = name.split("/")
        if split not in parts:
            continue
        if kind == "labels" and parts[0] != "labels":
            continue
        base = parts[-1]
        if kind == "labels":
            match = LABEL_NAME.fullmatch(base)
            if match is None:
                raise PrepareError(f"unrecognised label name: {name}")
            stem = match.group("stem")
        else:
            stem = Path(name).stem
        if stem in found:
            raise PrepareError(f"two {kind} entries for {split}/{stem}: {found[stem]} and {name}")
        found[stem] = name
    return found


def prepare_split(images_zip: Path, labels_zip: Path, out: Path, split: str,
                  *, limit: int | None = None) -> dict:
    """Write one split. Returns the counts; a size-mismatched pair is excluded, not failed."""
    out_images = out / "images" / split
    out_masks = out / "masks" / split
    out_images.mkdir(parents=True, exist_ok=True)
    out_masks.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(images_zip) as zimg, zipfile.ZipFile(labels_zip) as zlab:
        images = _members(zimg, split, "images")
        labels = _members(zlab, split, "labels")
        only_images = sorted(set(images) - set(labels))
        only_labels = sorted(set(labels) - set(images))
        if only_images or only_labels:
            raise PrepareError(
                f"{split}: {len(only_images)} images have no label and "
                f"{len(only_labels)} labels have no image (first: "
                f"{(only_images or only_labels)[0]})"
            )
        stems = sorted(images)
        if limit is not None:
            stems = stems[:limit]

        written, excluded = [], []
        for i, stem in enumerate(stems):
            with zlab.open(labels[stem]) as f:
                index_mask = np.array(Image.open(f))
            with zimg.open(images[stem]) as f:
                image_bytes = f.read()
            image_size = Image.open(io.BytesIO(image_bytes)).size   # (width, height)
            label_size = (index_mask.shape[1], index_mask.shape[0])
            if image_size != label_size:
                # The published archives pair six frames whose image and label
                # cover different framings of the same scene: the label is a
                # landscape 1280x720 map while the image is a portrait 720x1280
                # photo, and only part of the two overlaps. No rotation or crop
                # recovers a sample, so the pair is excluded and recorded rather
                # than written with silently mismatched pixels.
                excluded.append({
                    "stem": stem,
                    "reason": "image and label cover different framings",
                    "image_size": list(image_size),
                    "label_size": list(label_size),
                })
                continue
            mask = object_mask_from_class_index(index_mask)
            Image.fromarray(mask * FOREGROUND, mode="L").save(out_masks / f"{stem}.png")
            (out_images / f"{stem}.jpg").write_bytes(image_bytes)
            written.append(stem)
            if (i + 1) % 1000 == 0:
                print(f"[prepare] {split}: {i + 1}/{len(stems)}", file=sys.stderr)

    (out / f"{split}.txt").write_text("\n".join(written) + "\n")
    return {"split": split, "pairs": len(stems), "written": len(written),
            "excluded": excluded}


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def prepare_tree(raw: Path, out: Path, *, allow_subset: bool = False,
                 limit: int | None = None) -> dict:
    images_zip = raw / "bdd100k_images_10k.zip"
    labels_zip = raw / "bdd100k_seg_maps.zip"
    for p in (images_zip, labels_zip):
        if not p.is_file():
            raise PrepareError(f"missing {p}")

    results = {split: prepare_split(images_zip, labels_zip, out, split, limit=limit)
               for split in SPLITS}

    if not allow_subset:
        for split, expected in EXPECTED_COUNTS.items():
            got = results[split]["written"]
            # Excluded pairs are accounted for, not silently missing: a truncated
            # archive still fails, while the published frames whose image and
            # label disagree are written down in the manifest instead.
            accounted = got + len(results[split]["excluded"])
            if accounted != expected:
                raise PrepareError(
                    f"{split}: prepared {got} rows and excluded "
                    f"{len(results[split]['excluded'])}, the published set has "
                    f"{expected}. Re-download the archives, or pass --allow-subset "
                    f"for a fixture."
                )

    manifest = {
        "format": PREPARED_MANIFEST_FORMAT,
        "foreground_classes": list(BDD_FOREGROUND_CLASSES),
        "foreground_ids": list(foreground_class_ids()),
        "class_names": list(BDD100K_CLASSES),
        "unknown_index": BDD_UNKNOWN_INDEX,
        "mask_values": {"background": 0, "foreground": FOREGROUND},
        "counts": {s: r["written"] for s, r in results.items()},
        "excluded": [{"split": s, **row} for s, r in results.items()
                     for row in r["excluded"]],
        "sources": {
            p.name: {"bytes": p.stat().st_size, "sha256": _sha256(p)}
            for p in (images_zip, labels_zip)
        },
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def _s3_client(endpoint: str):
    """The project's MinIO client: credentials from the environment."""
    from minio import Minio

    user = os.getenv("MINIO_USER")
    password = os.getenv("MINIO_PASSWORD")
    if not user or not password:
        raise PrepareError(
            "MINIO_USER and MINIO_PASSWORD must be set for the object-store mode"
        )
    return Minio(endpoint, access_key=user, secret_key=password, secure=False)


def download_raw(client, bucket: str, prefix: str, raw: Path) -> list[str]:
    """Fetch the two archives and their checksums into ``raw``.

    A file already there at the right size is kept, so a Job retried after a
    restart does not re-copy a gigabyte.
    """
    raw.mkdir(parents=True, exist_ok=True)
    wanted = set(RAW_FILES) | {"SHA256SUMS"}
    got = []
    for obj in client.list_objects(bucket, prefix=prefix + "/", recursive=True):
        name = obj.object_name.rsplit("/", 1)[-1]
        if name not in wanted:
            continue
        dest = raw / name
        if dest.is_file() and dest.stat().st_size == obj.size:
            got.append(name)
            continue
        client.fget_object(bucket, obj.object_name, str(dest))
        got.append(name)
    missing = wanted - set(got)
    if missing:
        raise PrepareError(
            f"s3://{bucket}/{prefix}/ is missing {sorted(missing)}"
        )
    return sorted(got)


def upload_tree(client, bucket: str, prefix: str, tree: Path) -> int:
    """Upload a prepared tree, manifest last, and return the file count.

    The manifest goes last for the same reason the reader treats it as the
    completeness marker: a listing that sees it has seen every other object.
    """
    files = sorted(p for p in tree.rglob("*") if p.is_file())
    files.sort(key=lambda p: p.name == "manifest.json")
    for path in files:
        rel = path.relative_to(tree).as_posix()
        client.fput_object(bucket, f"{prefix}/{rel}", str(path))
    return len(files)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--raw", type=Path, default=None,
                    help="directory holding bdd100k_images_10k.zip and bdd100k_seg_maps.zip")
    ap.add_argument("--out", type=Path, default=None, help="prepared tree to write")
    ap.add_argument("--allow-subset", action="store_true",
                    help="accept fewer rows than the published counts (fixtures only)")
    ap.add_argument("--limit", type=int, default=None,
                    help="stop after this many rows per split (fixtures only)")
    ap.add_argument("--from-s3", metavar="BUCKET/PREFIX", default=None,
                    help="fetch the archives from the object store instead of --raw")
    ap.add_argument("--upload-to-s3", metavar="BUCKET/PREFIX", default=None,
                    help="upload the prepared tree after writing it")
    ap.add_argument("--s3-endpoint", default="minio.minio.svc.cluster.local:9000",
                    help="object store endpoint for --from-s3 / --upload-to-s3")
    args = ap.parse_args(argv)

    if args.from_s3 and args.raw is None:
        args.raw = Path("/tmp/bdd100k/raw")
    if args.upload_to_s3 and args.out is None:
        args.out = Path("/tmp/bdd100k/prepared")
    if args.raw is None or args.out is None:
        ap.error("--raw and --out are required unless --from-s3 / --upload-to-s3 supply them")

    try:
        client = None
        if args.from_s3 or args.upload_to_s3:
            client = _s3_client(args.s3_endpoint)

        fetched = []
        if args.from_s3:
            bucket, _, prefix = args.from_s3.partition("/")
            fetched = download_raw(client, bucket, prefix, args.raw)
            print(f"[prepare] fetched {fetched} from s3://{bucket}/{prefix}/")

        manifest = prepare_tree(args.raw, args.out, allow_subset=args.allow_subset,
                                limit=args.limit)

        uploaded = 0
        if args.upload_to_s3:
            bucket, _, prefix = args.upload_to_s3.partition("/")
            uploaded = upload_tree(client, bucket, prefix, args.out)
            print(f"[prepare] uploaded {uploaded} files to s3://{bucket}/{prefix}/")
    except PrepareError as exc:
        print(f"[prepare] ERROR: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({"counts": manifest["counts"], "out": str(args.out),
                      "uploaded": uploaded}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
