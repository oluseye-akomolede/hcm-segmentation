"""BDD100K: the object/background mapping, the prepared data, and how the
trainer selects its best epoch.

Fixtures are tiny synthetic trees -- one image and one label per split -- so the
whole path (label -> prepared mask -> dataset row -> transform output) runs in
milliseconds and the assertions are about semantics, not about dataset size.
"""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from ecs_hcm_model.core.bdd import (
    BDD_DATASET_NAME,
    BDD100K_CLASSES,
    BDD_FOREGROUND_CLASSES,
    BDD100kSegDataset,
    BDD_UNKNOWN_INDEX,
    PREPARED_MANIFEST_FORMAT,
    foreground_class_ids,
    object_mask_from_class_index,
)
from ecs_hcm_model.core.model_utils import (
    binary_mask_for_sample,
    coco_val_transform_example,
)
from ecs_hcm_model.core.trainer import best_value_from, mix_ratio_for_share

PLAN_FOREGROUND = (
    "person", "rider", "car", "truck", "bus", "train", "motorcycle", "bicycle",
    "traffic light",
)
BACKGROUND_EXAMPLES = ("road", "sidewalk", "building", "sky", "vegetation", "traffic sign")


# --- the constant -----------------------------------------------------------

def test_foreground_constant_is_the_plan_s_object_list():
    """The mapping lives in one constant, and it is the plan's list."""
    assert BDD_FOREGROUND_CLASSES == PLAN_FOREGROUND


def test_every_foreground_name_resolves_to_its_bdd_index():
    index = {name: i for i, name in enumerate(BDD100K_CLASSES)}
    ids = foreground_class_ids()
    assert len(ids) == len(PLAN_FOREGROUND)
    assert ids == tuple(index[name] for name in PLAN_FOREGROUND)
    # Pinned, so a reordered class list is a failure here rather than a
    # silently different training target. The order is the plan's enumeration
    # order (traffic light last), not the label file's index order.
    assert ids == (11, 12, 13, 14, 15, 16, 17, 18, 6)


def test_object_mask_foreground_is_one_and_everything_else_is_zero():
    index = {name: i for i, name in enumerate(BDD100K_CLASSES)}
    label = np.zeros((6, 6), dtype=np.uint8)
    label[0, :] = index["road"]
    label[1, :] = index["person"]
    label[2, :] = index["car"]
    label[3, :] = index["traffic light"]
    label[4, :] = BDD_UNKNOWN_INDEX
    for name in BACKGROUND_EXAMPLES:
        label[5, index[name] % 6] = index[name]

    mask = object_mask_from_class_index(label)
    assert mask.shape == label.shape
    assert mask.dtype == np.uint8
    assert mask[0].sum() == 0            # road
    assert mask[1].sum() == 6            # person
    assert mask[2].sum() == 6            # car
    assert mask[3].sum() == 6            # traffic light: a listed object
    assert mask[4].sum() == 0            # unknown is background, not target
    assert mask[5].sum() == 0            # every other class is background
    assert set(np.unique(mask)) <= {0, 1}


def test_object_mask_refuses_a_non_2d_label():
    with pytest.raises(ValueError, match="2-D"):
        object_mask_from_class_index(np.zeros((4, 4, 3), dtype=np.uint8))


# --- the supplied-mask path through the transform ---------------------------

def test_transform_uses_the_prepared_mask_and_needs_no_annotations():
    img = Image.new("RGB", (64, 64), (10, 20, 30))
    full = np.ones((64, 64), dtype=bool)
    empty = np.zeros((64, 64), dtype=bool)
    block = np.zeros((64, 64), dtype=bool)
    block[16:48, 16:48] = True

    def score(mask):
        out = coco_val_transform_example(
            {"image": img, "binary_mask": mask, "image_id": 3}, input_size=64
        )
        assert out["valid"] is True
        return out["segmentation_mask"].sum().item()

    # The transform resizes the short side and centre-crops, so only the
    # degenerate masks have an arithmetic count: all stays all, none stays none.
    # A row with no `annotations` at all is still a valid sample -- which is the
    # point: the mask came from the row, not from polygons that are not there.
    assert score(full) == 64 * 64
    assert score(empty) == 0
    assert 0 < score(block) < 64 * 64


def test_transform_thresholds_a_0_255_mask_to_the_same_target():
    img = Image.new("RGB", (64, 64))
    as_bool = np.zeros((64, 64), dtype=bool)
    as_bool[8:56, :] = True
    as_bytes = (as_bool * 255).astype(np.uint8)

    a = coco_val_transform_example({"image": img, "binary_mask": as_bool, "image_id": 0},
                                   input_size=64)
    b = coco_val_transform_example({"image": img, "binary_mask": as_bytes, "image_id": 0},
                                   input_size=64)
    assert a["valid"] and b["valid"]
    assert torch.equal(a["segmentation_mask"], b["segmentation_mask"])


def test_a_mask_for_another_image_is_refused():
    img = Image.new("RGB", (64, 64))
    with pytest.raises(ValueError, match="binary mask"):
        binary_mask_for_sample(img, None, np.zeros((32, 32), dtype=bool))

    # The transform turns any per-row error into valid=False rather than a
    # silently misaligned target; the sample is then dropped from the mean.
    out = coco_val_transform_example(
        {"image": img, "binary_mask": np.zeros((32, 32), dtype=bool), "image_id": 0},
        input_size=64,
    )
    assert out["valid"] is False


def test_a_row_without_a_mask_still_builds_one_from_polygons():
    """The COCO path: no binary_mask, so the polygons are the target."""
    from ecs_hcm_model.core.model_utils import build_binary_mask_coco_all

    img = Image.new("RGB", (64, 64))
    row = {
        "image": img,
        "image_id": 7,
        "annotations": [{
            "segmentation": [[10.0, 10.0, 40.0, 10.0, 40.0, 40.0, 10.0, 40.0]],
            "iscrowd": 0,
            "bbox": [10.0, 10.0, 30.0, 30.0],
        }],
    }
    out = coco_val_transform_example(row, input_size=64)
    reference = build_binary_mask_coco_all(
        {"image": img, "annotations": row["annotations"]}, debug=False, stats={}
    )
    assert out["valid"] is True
    assert reference.sum() > 0
    # Same target as the polygon builder, before the resize.
    assert out["segmentation_mask"].sum().item() > 0


def test_binary_mask_for_sample_prefers_the_supplied_mask():
    img = Image.new("RGB", (8, 8))
    prepared = np.ones((8, 8), dtype=np.uint8)
    got = binary_mask_for_sample(img, None, prepared)
    assert got.sum() == 64
    assert got.dtype == np.uint8
    assert set(np.unique(got)) <= {0, 1}


# --- the prepared tree ------------------------------------------------------

def _write_split(out: Path, split: str, stems, *, image_color=(1, 2, 3)):
    (out / "images" / split).mkdir(parents=True, exist_ok=True)
    (out / "masks" / split).mkdir(parents=True, exist_ok=True)
    for i, stem in enumerate(stems):
        Image.new("RGB", (32, 32), image_color).save(out / "images" / split / f"{stem}.jpg")
        mask = np.zeros((32, 32), dtype=np.uint8)
        mask[: 8 + i, :] = 255
        Image.fromarray(mask, mode="L").save(out / "masks" / split / f"{stem}.png")
    (out / f"{split}.txt").write_text("\n".join(stems) + "\n")


@pytest.fixture
def prepared_tree(tmp_path):
    out = tmp_path / "prepared"
    _write_split(out, "train", ["b_stem", "a_stem"])
    _write_split(out, "val", ["v_stem"])
    return out


def test_dataset_reads_prepared_rows(prepared_tree):
    ds = BDD100kSegDataset(prepared_tree, split="val")
    assert len(ds) == 1
    row = ds[0]
    assert row["stem"] == "v_stem"
    assert row["image"].size == (32, 32)
    assert row["binary_mask"].dtype == np.bool_
    assert row["binary_mask"].sum() == 8 * 32
    assert isinstance(row["image_id"], int)


def test_dataset_ids_follow_the_sorted_split_file(prepared_tree):
    ds = BDD100kSegDataset(prepared_tree, split="train")
    assert [ds[i]["stem"] for i in range(len(ds))] == ["b_stem", "a_stem"]
    assert [ds[i]["image_id"] for i in range(len(ds))] == [0, 1]


def test_the_eval_tool_scores_the_prepared_val_split(prepared_tree):
    # The delta's eval scenario: --dataset bdd must route the eval at the
    # prepared tree's *val* split -- the rows the trainer validated on -- not at
    # COCO, and not at the train split.
    from ecs_hcm_model.eval import seg

    rows = seg.build_rows("bdd", prepared_tree)

    assert isinstance(rows, BDD100kSegDataset)
    assert rows.split == "val"
    assert len(rows) == 1
    assert rows[0]["stem"] == "v_stem"


def test_dataset_refuses_a_missing_split(prepared_tree):
    with pytest.raises(FileNotFoundError, match="test.txt"):
        BDD100kSegDataset(prepared_tree, split="test")


def test_dataset_refuses_a_mask_whose_size_disagrees(prepared_tree):
    Image.fromarray(np.zeros((16, 16), dtype=np.uint8), mode="L").save(
        prepared_tree / "masks" / "val" / "v_stem.png"
    )
    with pytest.raises(ValueError, match="but the image"):
        BDD100kSegDataset(prepared_tree, split="val")[0]


def test_the_trainer_loader_branch_returns_segmentation_batches(prepared_tree):
    """The dataset_name branch the trainer takes for a BDD refine."""
    from ecs_hcm_model.core.model_utils import (
        get_training_dataloader,
        get_validation_dataloader,
    )

    train = get_training_dataloader(
        dataset_name="bdd100k", bdd_data_root=str(prepared_tree), batch_size=2,
        workers=0, global_rank=0, world_size=1, shuffle=False, input_size=32,
    )
    batch = next(iter(train))
    assert batch["pixel_values"].shape == (2, 3, 32, 32)
    assert batch["segmentation_mask"].shape == (2, 32, 32)
    assert batch["segmentation_mask"].dtype == torch.int64
    assert batch["image_id"].shape == (2,)

    val = get_validation_dataloader(
        dataset_name="bdd100k", bdd_data_root=str(prepared_tree), batch_size=1,
        workers=0, global_rank=0, world_size=1, input_size=32,
    )
    vbatch = next(iter(val))
    assert vbatch["segmentation_mask"].shape == (1, 32, 32)
    assert vbatch["valid"].all()


# --- the prepare script -----------------------------------------------------

def _make_raw(tmp_path, *, train_stems, val_stems, drop_label=None,
              mismatched_stems=()):
    raw = tmp_path / "raw"
    raw.mkdir()
    index = {name: i for i, name in enumerate(BDD100K_CLASSES)}

    def label_png(stem: str) -> bytes:
        import io
        arr = np.full((16, 16), index["road"], dtype=np.uint8)
        arr[: len(stem) % 16 + 1, :] = index["person"]
        arr[-1, :] = BDD_UNKNOWN_INDEX
        buf = io.BytesIO()
        Image.fromarray(arr, mode="L").save(buf, format="PNG")
        return buf.getvalue()

    images = raw / "images.zip"
    labels = raw / "labels.zip"
    with zipfile.ZipFile(images, "w") as z:
        for split, stems in (("train", train_stems), ("val", val_stems)):
            for stem in stems:
                import io
                buf = io.BytesIO()
                # A mismatched stem is a portrait image under a landscape label,
                # the shape the published archives really contain six of.
                size = (16, 32) if stem in mismatched_stems else (16, 16)
                Image.new("RGB", size, (9, 9, 9)).save(buf, format="JPEG")
                # The published archive's real layout: no bdd100k/ prefix above 10k/.
                z.writestr(f"10k/{split}/{stem}.jpg", buf.getvalue())
    labels = raw / "labels.zip"
    with zipfile.ZipFile(labels, "w") as z:
        for split, stems in (("train", train_stems), ("val", val_stems)):
            for stem in stems:
                if drop_label and stem == drop_label:
                    continue
                # Two maps per image in the published zip: the class-id map we want,
                # and a colourised preview that must be ignored. Both carry a
                # ``_train_`` token even under labels/val — that is the real naming,
                # and it is why the split must come from the folder.
                z.writestr(f"labels/{split}/{stem}_train_id.png", label_png(stem))
                z.writestr(f"color_labels/{split}/{stem}_train_color.png", label_png(stem))
    return raw


def _stage_raw(raw: Path, tmp_path: Path) -> Path:
    raw.mkdir(parents=True, exist_ok=True)
    (raw / "bdd100k_images_10k.zip").write_bytes((raw / "images.zip").read_bytes())
    (raw / "bdd100k_seg_maps.zip").write_bytes((raw / "labels.zip").read_bytes())
    return raw


def test_prepare_writes_pairs_binary_masks_and_a_manifest(tmp_path):
    from tools.prepare_bdd100k import prepare_tree

    raw = _stage_raw(_make_raw(tmp_path, train_stems=["t2", "t1"], val_stems=["v1"]), tmp_path)
    out = tmp_path / "out"
    manifest = prepare_tree(raw, out, allow_subset=True)

    assert manifest["counts"] == {"train": 2, "val": 1}
    assert (out / "train.txt").read_text().split() == ["t1", "t2"]   # sorted
    assert (out / "val.txt").read_text().split() == ["v1"]
    assert (out / "images" / "train" / "t1.jpg").is_file()
    mask = np.array(Image.open(out / "masks" / "train" / "t1.png"))
    assert set(np.unique(mask)) <= {0, 255}
    assert mask[-1].sum() == 0          # unknown -> background
    assert manifest["foreground_classes"] == list(PLAN_FOREGROUND)
    assert manifest["mask_values"] == {"background": 0, "foreground": 255}
    assert set(manifest["sources"]) == {"bdd100k_images_10k.zip", "bdd100k_seg_maps.zip"}
    assert json.loads((out / "manifest.json").read_text())["counts"] == manifest["counts"]


def test_prepare_pairs_val_labels_named_with_the_train_token(tmp_path):
    """The published zip names the val maps ``<id>_train_id.png``.

    The split is the folder (``labels/val``), so keying off the filename token
    drops every val row — the 1018f failure. The token must be irrelevant.
    """
    from tools.prepare_bdd100k import prepare_tree

    raw = _stage_raw(_make_raw(tmp_path, train_stems=["t1"], val_stems=["v1", "v2"]),
                     tmp_path)
    with zipfile.ZipFile(raw / "bdd100k_seg_maps.zip") as z:
        assert "labels/val/v2_train_id.png" in z.namelist()

    manifest = prepare_tree(raw, tmp_path / "out", allow_subset=True)

    assert manifest["counts"] == {"train": 1, "val": 2}
    assert (tmp_path / "out" / "val.txt").read_text().split() == ["v1", "v2"]


def test_prepare_refuses_an_image_without_a_label(tmp_path):
    from tools.prepare_bdd100k import PrepareError, prepare_tree

    raw = _stage_raw(
        _make_raw(tmp_path, train_stems=["t1", "t2"], val_stems=["v1"], drop_label="t2"),
        tmp_path,
    )
    with pytest.raises(PrepareError, match="no label"):
        prepare_tree(raw, tmp_path / "out", allow_subset=True)


def test_prepare_refuses_a_truncated_tree_unless_a_subset_is_allowed(tmp_path):
    from tools.prepare_bdd100k import PrepareError, prepare_tree

    raw = _stage_raw(_make_raw(tmp_path, train_stems=["t1"], val_stems=["v1"]), tmp_path)
    with pytest.raises(PrepareError, match="published set has 7000"):
        prepare_tree(raw, tmp_path / "out", allow_subset=False)


def test_prepare_excludes_a_pair_whose_frames_disagree(tmp_path):
    """Six published pairs have a portrait image under a landscape label.

    Writing them would ship rows whose mask pixels belong to another framing,
    so prepare drops the pair, keeps it out of the split list, and records it
    in the manifest.
    """
    from tools.prepare_bdd100k import prepare_tree

    raw = _stage_raw(
        _make_raw(tmp_path, train_stems=["t1", "t2"], val_stems=["v1"],
                  mismatched_stems=["t2"]),
        tmp_path,
    )
    out = tmp_path / "out"
    manifest = prepare_tree(raw, out, allow_subset=True)

    assert manifest["counts"] == {"train": 1, "val": 1}
    assert (out / "train.txt").read_text().split() == ["t1"]
    assert not (out / "images" / "train" / "t2.jpg").exists()
    assert not (out / "masks" / "train" / "t2.png").exists()
    assert manifest["excluded"] == [{
        "split": "train", "stem": "t2",
        "reason": "image and label cover different framings",
        "image_size": [16, 32], "label_size": [16, 16],
    }]
    assert json.loads((out / "manifest.json").read_text())["excluded"] == manifest["excluded"]


def test_prepare_counts_excluded_pairs_towards_the_published_total(tmp_path, monkeypatch):
    """Exclusions are accounted for, so a truncated archive still fails."""
    from tools import prepare_bdd100k
    from tools.prepare_bdd100k import PrepareError, prepare_tree

    monkeypatch.setattr(prepare_bdd100k, "EXPECTED_COUNTS", {"train": 2, "val": 1})

    (tmp_path / "counted").mkdir()
    counted = _stage_raw(
        _make_raw(tmp_path / "counted", train_stems=["t1", "t2"], val_stems=["v1"],
                  mismatched_stems=["t2"]),
        tmp_path / "counted",
    )
    manifest = prepare_tree(counted, tmp_path / "out", allow_subset=False)
    assert manifest["counts"] == {"train": 1, "val": 1}
    assert len(manifest["excluded"]) == 1

    (tmp_path / "short").mkdir()
    truncated = _stage_raw(
        _make_raw(tmp_path / "short", train_stems=["t1"], val_stems=["v1"]),
        tmp_path / "short")
    with pytest.raises(PrepareError, match="excluded 0"):
        prepare_tree(truncated, tmp_path / "out-short", allow_subset=False)


# --- training selection rules ----------------------------------------------

def test_mix_ratio_for_share_converts_a_share_to_extra_batches():
    assert mix_ratio_for_share(0.0) == 0.0
    assert mix_ratio_for_share(0.5) == 1.0     # 50/50 by sample count
    assert mix_ratio_for_share(0.8) == pytest.approx(4.0)


def test_best_value_reads_the_primary_under_a_bdd_mix():
    metrics = {"val_iou": 70.0, "val_iou_bdd": 65.0, "val_iou_coco": 75.0}
    assert best_value_from(metrics, "val_iou", mixed=True, best_on_primary=True,
                           primary_suffix="bdd") == 65.0
    # A COCOA refine keeps the mean rule (the unsuffixed value).
    assert best_value_from(metrics, "val_iou", mixed=True, best_on_primary=False) == 70.0
    # A COCOA refine that does select on its own data reads the cocoa key.
    assert best_value_from({"val_iou": 1.0, "val_iou_cocoa": 61.0}, "val_iou",
                           mixed=True, best_on_primary=True) == 61.0
    # Unmixed: the unsuffixed value either way.
    assert best_value_from({"val_iou": 61.0}, "val_iou", mixed=False,
                           best_on_primary=True) == 61.0


def test_merge_names_the_primary_by_its_own_dataset():
    from ecs_hcm_model.core.trainer import merge_mixed_val_metrics

    def metrics(iou):
        return {"val_loss": 1.0, "val_iou": iou, "val_dice": iou}

    merged = merge_mixed_val_metrics(metrics(60.0), metrics(80.0), primary_suffix="bdd")
    assert merged["val_iou_bdd"] == 60.0
    assert merged["val_iou_coco"] == 80.0
    assert merged["val_iou"] == 70.0


def test_best_value_is_none_without_a_best_metric():
    assert best_value_from({"val_iou": 1.0}, None, mixed=True, best_on_primary=True) is None


def test_epoch_checkpoint_records_the_bdd_dataset(monkeypatch):
    """The checkpoint says BDD100K, not the refine task's COCOA label."""
    import ecs_hcm_model.core.trainer as trainer_module

    captured = {}

    class _Uploader:
        def __init__(self, bucket_name=None):
            pass

        def upload_model(self, **kwargs):
            captured.update(kwargs)
            return "s3://fake/bdd"

    monkeypatch.setattr(trainer_module, "ModelUploaderAndDownloader", _Uploader)
    trainer_module.save_epoch_checkpoint(
        model=torch.nn.Linear(2, 2),
        task="cocoa_seg",
        epoch=3,
        val_metrics={"val_iou": 50.0, "val_dice": 40.0, "val_loss": 1.0},
        avg_loss=1.0,
        val_loss=1.0,
        run_id="run-1",
        dataset_name="BDD100K+MSCOCO",
    )
    assert captured["metadata"]["dataset_name"] == "BDD100K+MSCOCO"
    # Without one, the task's own label is unchanged (the COCOA default).
    captured.clear()
    trainer_module.save_epoch_checkpoint(
        model=torch.nn.Linear(2, 2), task="cocoa_seg", epoch=3,
        val_metrics={"val_iou": 50.0, "val_dice": 40.0, "val_loss": 1.0},
        avg_loss=1.0, val_loss=1.0, run_id="run-1",
    )
    assert captured["metadata"]["dataset_name"] == "COCOA"


@pytest.mark.parametrize("kwargs, match", [
    (dict(bdd=True, finetune=False), "requires finetune=True"),
    (dict(bdd=True, finetune=True, coco=True), "a run trains one or the other"),
    (dict(bdd=True, finetune=True, bdd_coco_mix_share=1.0), r"\[0, 1\)"),
    (dict(bdd=True, finetune=True, bdd_coco_mix_share=-0.1), r"\[0, 1\)"),
    (dict(bdd=True, finetune=True, bdd_coco_mix_share=0.5, coco_mix_ratio=1.0),
     "pass only one"),
])
def test_bdd_arguments_are_refused_before_anything_runs(kwargs, match):
    """These are raised at the top of the entry point: no run, no download."""
    from ecs_hcm_model.core.trainer import train_hcm_imagenet1k

    with pytest.raises(ValueError, match=match):
        train_hcm_imagenet1k(test_mode=True, max_epochs=1, new_model=True, **kwargs)


# --- the entry point's wiring ------------------------------------------------

class _LoaderSpy:
    """Records the kwargs each loader factory is called with."""

    def __init__(self):
        self.train = []
        self.val = []

    def train_loader(self, **kwargs):
        self.train.append(kwargs)
        return [torch.randn(1)]

    def val_loader(self, **kwargs):
        self.val.append(kwargs)
        return [torch.randn(1)]


def _fake_train_result():
    from tests.conftest import TinyClassifier

    return {
        "best_epoch": 0,
        "best_score": 1.0,
        "best_state": TinyClassifier().state_dict(),
        "best_metrics": {"val_iou": 1.0},
        "best_train_loss": 0.1,
        "last_epoch": 0,
    }


def _drive_entry(monkeypatch, spy, **flags):
    """Run train_hcm_imagenet1k with stub storage, no downloads, no trainer."""
    from tests.conftest import RecordingLogger, RecordingUploader

    from ecs_hcm_model.core import trainer as T

    trainer_kwargs = []

    def _capture(**kwargs):
        trainer_kwargs.append(kwargs)
        return _fake_train_result()

    monkeypatch.setattr(T, "set_env_vars", lambda *a, **k: None)
    monkeypatch.setattr(T, "logger_config", lambda **kw: {"fake": True})
    monkeypatch.setattr(T, "MLFlowHCMLogger", lambda **kw: RecordingLogger())
    monkeypatch.setattr(T, "ModelUploaderAndDownloader",
                        lambda **kw: RecordingUploader())
    monkeypatch.setattr(T, "load_hcm_model",
                        lambda **kw: __import__(
                            "tests.conftest", fromlist=["TinyClassifier"]
                        ).TinyClassifier())
    monkeypatch.setattr(T, "train_model_for_epochs", _capture)
    # The prepared tree is fetched from the object store; these tests drive the
    # wiring, not the download (which has its own test below).
    monkeypatch.setattr(T, "fetch_prepared_tree", lambda p: Path(p))
    monkeypatch.setattr(T, "get_training_dataloader", spy.train_loader)
    monkeypatch.setattr(T, "get_validation_dataloader", spy.val_loader)

    T.train_hcm_imagenet1k(
        new_model=True, finetune=True, max_epochs=1, batch_size=2, **flags
    )
    return trainer_kwargs


def test_a_bdd_refine_wires_the_mix_val_and_best_epoch(monkeypatch, tmp_path):
    """The plan's default: half of every epoch from COCO, validated on both,
    best chosen on BDD val, logged under `bdd` keys.

    Each of those is one keyword at the boundary between the entry point and
    the loop; the pure functions are tested above, and this is the test that
    fails if the wiring drops one of them -- exactly the shape of a run that
    looks fine in its own dataset and quietly forgets the rest.
    """
    spy = _LoaderSpy()

    calls = _drive_entry(monkeypatch, spy, bdd=True,
                         bdd_data_root=str(tmp_path))

    assert len(calls) == 1
    kwargs = calls[0]
    # 50/50 by sample count is one mix batch per primary batch.
    assert kwargs["coco_mix_ratio"] == 1.0
    assert kwargs["mix_train_loader"] is not None
    assert kwargs["mix_val_loader"] is not None
    assert kwargs["best_on_primary"] is True
    assert kwargs["primary_suffix"] == "bdd"
    assert kwargs["dataset_label"] == "BDD100K+MSCOCO"

    primary = spy.train[0]
    assert primary["dataset_name"] == BDD_DATASET_NAME
    assert primary["bdd_data_root"] == str(tmp_path)
    # The COCO side of the mix is still COCO, with no BDD root.
    assert spy.train[1]["dataset_name"] == "shunk031/MSCOCO"
    assert spy.train[1].get("bdd_data_root") is None
    assert spy.val[0]["dataset_name"] == BDD_DATASET_NAME
    assert spy.val[1]["dataset_name"] == "shunk031/MSCOCO"


def test_a_bdd_refine_without_a_mix_validates_only_on_bdd(monkeypatch, tmp_path):
    """A share of 0 is BDD alone: no COCO loader, no COCO val, no mix suffix."""
    spy = _LoaderSpy()

    calls = _drive_entry(monkeypatch, spy, bdd=True,
                         bdd_data_root=str(tmp_path), bdd_coco_mix_share=0.0)

    kwargs = calls[0]
    assert kwargs["coco_mix_ratio"] == 0.0
    assert kwargs["mix_train_loader"] is None
    assert kwargs["mix_val_loader"] is None
    # Selection is still on BDD's own val; there is just nothing to average it with.
    assert kwargs["best_on_primary"] is True
    assert kwargs["dataset_label"] == "BDD100K"
    assert len(spy.train) == 1 and len(spy.val) == 1


# --- the prepared tree's object-store copy --------------------------------

class _Obj:
    def __init__(self, name, size):
        self.object_name = name
        self.size = size


class _FakeMinio:
    """Lists a fixed tree and writes each object's body on download."""

    def __init__(self, files):
        self.files = files          # {name: bytes}
        self.downloaded = []

    def list_objects(self, bucket, prefix="", recursive=False):
        return [_Obj(n, len(b)) for n, b in self.files.items()
                if n.startswith(prefix)]

    def fget_object(self, bucket, name, dest):
        self.downloaded.append(name)
        Path(dest).write_bytes(self.files[name])


def _complete_manifest() -> bytes:
    """The bytes of a manifest that marks a tree complete."""
    return json.dumps({"format": PREPARED_MANIFEST_FORMAT}).encode()


def test_fetch_prepared_tree_writes_the_manifest_last(tmp_path):
    # The manifest is the "tree is complete" marker, so it must not land first:
    # a pod killed mid-copy would otherwise leave a complete-looking tree.
    from ecs_hcm_model.core.bdd import fetch_prepared_tree

    files = {
        "datasets/bdd100k/prepared/v1/images/val/a.jpg": b"img",
        "datasets/bdd100k/prepared/v1/masks/val/a.png": b"mask",
        "datasets/bdd100k/prepared/v1/val.txt": b"a\n",
        "datasets/bdd100k/prepared/v1/manifest.json": _complete_manifest(),
    }
    client = _FakeMinio(files)

    root = fetch_prepared_tree(tmp_path / "prepared", client=client)

    assert client.downloaded[-1].endswith("manifest.json")
    assert (root / "images/val/a.jpg").read_bytes() == b"img"
    assert (root / "masks/val/a.png").read_bytes() == b"mask"
    # A second call with the manifest present is a no-op: no listing needed.
    assert fetch_prepared_tree(root, client=_FakeMinio({})) == root


def test_a_truncated_manifest_does_not_mark_the_tree_complete(tmp_path):
    # The manifest is the last object written, so it is the one a killed pod
    # leaves half-written. Presence alone must not be believed: a parse failure
    # re-fetches, and the truncated file is replaced by the real one.
    from ecs_hcm_model.core.bdd import fetch_prepared_tree

    root = tmp_path / "prepared"
    (root / "images/val").mkdir(parents=True)
    (root / "images/val/a.jpg").write_bytes(b"img")
    (root / "manifest.json").write_bytes(b'{"format": "bdd100k-prepa')  # truncated

    files = {
        "datasets/bdd100k/prepared/v1/images/val/a.jpg": b"img",
        "datasets/bdd100k/prepared/v1/masks/val/a.png": b"mask",
        "datasets/bdd100k/prepared/v1/manifest.json": _complete_manifest(),
    }
    client = _FakeMinio(files)

    fetch_prepared_tree(root, client=client)

    assert client.downloaded, "a truncated manifest must trigger a fetch"
    assert json.loads((root / "manifest.json").read_text())["format"] == (
        PREPARED_MANIFEST_FORMAT)
    assert (root / "masks/val/a.png").read_bytes() == b"mask"


def test_a_manifest_of_another_format_does_not_mark_the_tree_complete(tmp_path):
    # Well-formed JSON from an older or unrelated layout is not this tree.
    from ecs_hcm_model.core.bdd import fetch_prepared_tree

    root = tmp_path / "prepared"
    root.mkdir(parents=True)
    (root / "manifest.json").write_text(json.dumps({"format": "something/v0"}))

    client = _FakeMinio({
        "datasets/bdd100k/prepared/v1/val.txt": b"a\n",
        "datasets/bdd100k/prepared/v1/manifest.json": _complete_manifest(),
    })
    fetch_prepared_tree(root, client=client)

    assert (root / "val.txt").read_bytes() == b"a\n"


def test_fetch_prepared_tree_keeps_files_that_already_landed(tmp_path):
    # A restart mid-copy must not re-download a gigabyte: a file of the right
    # size stays, and the missing ones are fetched.
    from ecs_hcm_model.core.bdd import fetch_prepared_tree

    root = tmp_path / "prepared"
    (root / "images/val").mkdir(parents=True)
    (root / "images/val/a.jpg").write_bytes(b"img")
    files = {
        "datasets/bdd100k/prepared/v1/images/val/a.jpg": b"img",
        "datasets/bdd100k/prepared/v1/masks/val/a.png": b"mask",
        "datasets/bdd100k/prepared/v1/manifest.json": _complete_manifest(),
    }
    client = _FakeMinio(files)

    fetch_prepared_tree(root, client=client)

    assert "datasets/bdd100k/prepared/v1/images/val/a.jpg" not in client.downloaded
    assert (root / "masks/val/a.png").read_bytes() == b"mask"


def test_the_object_store_mode_reads_its_credentials_from_the_environment(monkeypatch):
    # _s3_client is the one place the CLI touches credentials; it must read the
    # environment (never a literal) and refuse loudly when they are absent, and
    # it must actually import the minio package it names.
    import sys
    import types

    from tools import prepare_bdd100k

    seen = {}

    class FakeMinio:
        def __init__(self, endpoint, access_key, secret_key, secure):
            seen.update(endpoint=endpoint, access_key=access_key,
                        secret_key=secret_key, secure=secure)

    monkeypatch.setitem(sys.modules, "minio", types.SimpleNamespace(Minio=FakeMinio))
    monkeypatch.setenv("MINIO_USER", "u")
    monkeypatch.setenv("MINIO_PASSWORD", "p")

    prepare_bdd100k._s3_client("example:9000")

    assert seen == {"endpoint": "example:9000", "access_key": "u",
                    "secret_key": "p", "secure": False}

    monkeypatch.delenv("MINIO_PASSWORD")
    with pytest.raises(prepare_bdd100k.PrepareError, match="MINIO_USER and MINIO_PASSWORD"):
        prepare_bdd100k._s3_client("example:9000")


def test_the_object_store_mode_fetches_only_the_archives(tmp_path):
    # The raw prefix also holds whatever else is there; the download must take
    # the two archives plus the checksum file and nothing else, and must refuse
    # a prefix that is missing one of them.
    from tools.prepare_bdd100k import PrepareError, download_raw

    class _Fake:
        def __init__(self, names):
            self.names = names
            self.downloaded = []

        def list_objects(self, bucket, prefix, recursive=True):
            import types
            return [types.SimpleNamespace(object_name=n, size=None) for n in self.names]

        def fget_object(self, bucket, name, dest):
            Path(dest).write_bytes(b"x")
            self.downloaded.append(name)

    raw = tmp_path / "raw"
    client = _Fake(["datasets/bdd100k/raw/bdd100k_images_10k.zip",
                    "datasets/bdd100k/raw/bdd100k_seg_maps.zip",
                    "datasets/bdd100k/raw/SHA256SUMS",
                    "datasets/bdd100k/raw/notes.txt"])
    got = download_raw(client, "models", "datasets/bdd100k/raw", raw)

    assert got == ["SHA256SUMS", "bdd100k_images_10k.zip", "bdd100k_seg_maps.zip"]
    assert not (raw / "notes.txt").exists()

    with pytest.raises(PrepareError, match="missing"):
        download_raw(_Fake(["datasets/bdd100k/raw/SHA256SUMS"]), "models",
                     "datasets/bdd100k/raw", tmp_path / "raw2")


def test_the_upload_writes_the_manifest_last(tmp_path):
    # Same contract as fetch_prepared_tree: a listing that sees the manifest has
    # seen the whole tree.
    from tools.prepare_bdd100k import upload_tree

    tree = tmp_path / "prepared"
    (tree / "images/val").mkdir(parents=True)
    (tree / "images/val/a.jpg").write_bytes(b"i")
    (tree / "manifest.json").write_bytes(b"{}")

    class _Fake:
        def __init__(self):
            self.put = []

        def fput_object(self, bucket, name, path):
            self.put.append(name)

    client = _Fake()
    assert upload_tree(client, "models", "p/v1", tree) == 2
    assert client.put[-1] == "p/v1/manifest.json"
