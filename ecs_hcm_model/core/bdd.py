"""BDD100K street-scene segmentation: the object/background target and its data.

The fine-tune plan fixed the foreground once, for every consumer (data prep, the
trainer, the eval): the *objects* in a street scene are the target and everything
else -- road, sidewalk, building, wall, fence, pole, traffic sign, vegetation,
terrain, sky -- is background. That mapping is data, not code spread across three
call sites, so it lives here as :data:`BDD_FOREGROUND_CLASSES` and nothing else
names a class id.

BDD100K's semantic labels are a single-channel PNG whose pixel *value* is the
class index (0-18) and 255 for "unknown". Prepared data has already turned each
label into a binary object mask (`object_mask_from_class_index`), so the trainer
and the eval read the same 0/1 target the COCO path builds from polygons.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

# The dataset name the loaders and the CLI select BDD with.
BDD_DATASET_NAME = "bdd100k"

# Every class index BDD100K's semantic labels use, in index order. The list is
# the only place an index becomes a name; the foreground set below is expressed
# in names and resolved through it, so a reordered or renumbered label file is a
# test failure rather than a silently different target.
BDD100K_CLASSES = (
    "road",
    "sidewalk",
    "building",
    "wall",
    "fence",
    "pole",
    "traffic light",
    "traffic sign",
    "vegetation",
    "terrain",
    "sky",
    "person",
    "rider",
    "car",
    "truck",
    "bus",
    "train",
    "motorcycle",
    "bicycle",
)

# The foreground the plan fixed: objects, as named in the approved plan. Every
# other class -- and the unknown pixels (255) -- is background.
BDD_FOREGROUND_CLASSES = (
    "person",
    "rider",
    "car",
    "truck",
    "bus",
    "train",
    "motorcycle",
    "bicycle",
    "traffic light",
)

# The label value BDD uses where the annotator gave no class.
BDD_UNKNOWN_INDEX = 255


def foreground_class_ids() -> tuple[int, ...]:
    """The class indices ``BDD_FOREGROUND_CLASSES`` maps to, in class order."""
    index = {name: i for i, name in enumerate(BDD100K_CLASSES)}
    missing = [name for name in BDD_FOREGROUND_CLASSES if name not in index]
    if missing:
        raise KeyError(f"foreground classes not in BDD100K_CLASSES: {missing}")
    return tuple(index[name] for name in BDD_FOREGROUND_CLASSES)


def object_mask_from_class_index(index_mask: np.ndarray, *,
                                 foreground_ids: tuple[int, ...] | None = None) -> np.ndarray:
    """A binary object mask (``uint8`` 0/1) from a BDD class-index label map.

    Unknown pixels (255) and every background class are 0, so an unlabelled region
    is background rather than a target the model is asked to predict.
    """
    arr = np.asarray(index_mask)
    if arr.ndim != 2:
        raise ValueError(f"a BDD label map is 2-D (H, W); got shape {arr.shape}")
    ids = foreground_class_ids() if foreground_ids is None else tuple(foreground_ids)
    return np.isin(arr, ids).astype(np.uint8)


def _read_stems(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


class BDD100kSegDataset:
    """Prepared BDD100K rows: an image and the binary object mask beside it.

    The prepared tree (written by ``tools/prepare_bdd100k.py``) is::

        <root>/images/<split>/<stem>.jpg
        <root>/masks/<split>/<stem>.png     # 8-bit, 0 or 255
        <root>/<split>.txt                  # one stem per line, sorted

    Map-style, so the trainer's samplers (shuffle, DistributedSampler) work on it
    exactly as they do on the COCO map-style dataset. ``image_id`` is the stem's
    position in the split file, which is stable because the file is sorted; the
    stem itself stays available under ``"stem"`` for diagnostics.
    """

    def __init__(self, root, split: str = "train"):
        self.root = Path(root)
        self.split = split
        list_path = self.root / f"{split}.txt"
        if not list_path.is_file():
            raise FileNotFoundError(
                f"BDD split list not found: {list_path}. Prepare the data with "
                f"tools/prepare_bdd100k.py (or point --bdd-data-root at the prepared tree)."
            )
        self.stems = _read_stems(list_path)
        if not self.stems:
            raise ValueError(f"{list_path} is empty")
        self.image_dir = self.root / "images" / split
        self.mask_dir = self.root / "masks" / split
        self._transform = None

    def set_transform(self, transform) -> None:
        """Apply ``transform`` to each row, as the HuggingFace datasets do.

        The trainer's loaders hand the COCOA dataset a transform this way; a BDD
        row goes through the same function so the batch it returns has the same
        shape whichever dataset it came from.
        """
        self._transform = transform

    def __len__(self) -> int:
        return len(self.stems)

    def image_path(self, i: int) -> Path:
        return self.image_dir / f"{self.stems[i]}.jpg"

    def mask_path(self, i: int) -> Path:
        return self.mask_dir / f"{self.stems[i]}.png"

    def __getitem__(self, i: int) -> dict:
        from PIL import Image  # local: keeps this module importable without PIL

        stem = self.stems[i]
        img = Image.open(self.image_dir / f"{stem}.jpg").convert("RGB")
        mask = Image.open(self.mask_dir / f"{stem}.png")
        arr = np.array(mask)
        if arr.shape[:2] != (img.height, img.width):
            raise ValueError(
                f"mask {self.mask_dir / f'{stem}.png'} is {arr.shape[:2]} but the image "
                f"is {(img.height, img.width)}"
            )
        row = {
            "image": img,
            "binary_mask": arr > 0,
            "image_id": i,
            "stem": stem,
        }
        if self._transform is not None:
            return self._transform(row)
        return row


def prepared_root(explicit: str | os.PathLike | None = None) -> Path:
    """The prepared BDD tree: the argument, else ``$BDD100K_DATA_ROOT``."""
    if explicit:
        return Path(explicit)
    env = os.getenv("BDD100K_DATA_ROOT")
    if env:
        return Path(env)
    raise ValueError(
        "no prepared BDD100K tree: pass --bdd-data-root or set BDD100K_DATA_ROOT"
    )


# The object store the rest of the project uses, and the prefix the prepare
# script writes. A training pod that starts with an empty local tree fetches
# from here; nothing else needs to know the layout.
PREPARED_S3_PREFIX = "datasets/bdd100k/prepared/v1"
MINIO_ENDPOINT = "minio.minio.svc.cluster.local:9000"
MINIO_BUCKET = "models"
_MANIFEST = "manifest.json"
# The manifest's format string: written by the prepare script as its last object
# and read back here, so an older tree cannot be mistaken for a current one.
PREPARED_MANIFEST_FORMAT = "bdd100k-prepared/v1"


def _tree_is_complete(manifest_path: Path) -> bool:
    """Whether ``manifest_path`` is a readable prepared-tree manifest.

    Presence alone is not enough: the manifest is the last object written, so a
    download that died mid-file leaves a *truncated* manifest behind, and that
    tree would then be treated as complete forever. Parse it, and require it to
    be this format, before believing the tree.
    """
    try:
        data = json.loads(manifest_path.read_text())
    except (OSError, ValueError):
        return False
    return isinstance(data, dict) and data.get("format") == PREPARED_MANIFEST_FORMAT


def fetch_prepared_tree(root: str | os.PathLike, *, bucket: str = MINIO_BUCKET,
                        prefix: str = PREPARED_S3_PREFIX, client=None) -> Path:
    """Download the prepared tree into ``root`` unless it is already there.

    Idempotent and resumable: a file of the right size that is already on disk
    is kept, so a pod that restarts midway does not re-copy a gigabyte. The
    manifest is written last, and a tree whose manifest *parses* is complete --
    a half-downloaded tree has no manifest, or a truncated one, and is fetched
    again.

    Returns the local root.
    """
    root = Path(root)
    manifest_path = root / _MANIFEST
    if manifest_path.is_file() and _tree_is_complete(manifest_path):
        return root
    if client is None:
        client = _minio_client()

    objects = list(client.list_objects(bucket, prefix=prefix + "/", recursive=True))
    if not objects:
        raise FileNotFoundError(
            f"no prepared BDD100K tree at s3://{bucket}/{prefix}/"
        )
    root.mkdir(parents=True, exist_ok=True)
    # The manifest goes last, so its presence means every other file landed.
    objects.sort(key=lambda o: o.object_name.endswith("/" + _MANIFEST))
    for obj in objects:
        name = obj.object_name[len(prefix) + 1:]
        if not name or name.endswith("/"):
            continue
        dest = root / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.is_file() and dest.stat().st_size == obj.size:
            continue
        client.fget_object(bucket, obj.object_name, str(dest))
    return root


def _minio_client():
    """The project's MinIO client: cluster endpoint, credentials from the env."""
    from minio import Minio

    user = os.getenv("MINIO_USER")
    password = os.getenv("MINIO_PASSWORD")
    if not user or not password:
        raise ValueError(
            "MINIO_USER and MINIO_PASSWORD must be set to fetch the prepared tree"
        )
    return Minio(MINIO_ENDPOINT, access_key=user, secret_key=password, secure=False)
