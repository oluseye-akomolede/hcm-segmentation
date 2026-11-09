"""Tests for the segmentation eval tool (ecs_hcm_model/eval).

The datasets are tiny synthetic COCO-format and COCOA-format sets on disk:
bright squares (the instances) on a dark background. They go through the real
loaders, the trainer's validation transform and collate, and the trainer's
``evaluate()``. The model is a pixel oracle that calls a pixel foreground when
it is bright, so it scores near 100 only if every instance reached the target
in the place the image shows it; its inverse scores near 0.

The COCOA set comes from a stand-in builder that declares the schema of
shunk031/COCOA's ``decode_rle=True`` configuration and yields its generator's
example shape. Set ``COCOA_BUILDER_SCRIPT`` to a copy of the real ``COCOA.py``
to also run the real builder.
"""

import http.client
import io
import json
import os
import shutil
import socket
import subprocess
import sys
import tarfile
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from ecs_hcm_model.eval import fetch
from ecs_hcm_model.eval import seg

REPO = Path(__file__).resolve().parents[1]
SIZE = 512
HALF = 32
DARK = (20, 20, 20)
BRIGHT = (235, 235, 235)
FOUR = [(160, 160), (352, 160), (160, 352), (352, 352)]


def _paint(path, centres):
    a = np.full((SIZE, SIZE, 3), DARK, dtype=np.uint8)
    for cx, cy in centres:
        a[cy - HALF:cy + HALF, cx - HALF:cx + HALF] = BRIGHT
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(a).save(path)


def _square_mask(centres):
    m = np.zeros((SIZE, SIZE), dtype=np.uint8)
    for cx, cy in centres:
        m[cy - HALF:cy + HALF, cx - HALF:cx + HALF] = 1
    return m


def _poly(cx, cy):
    return [cx - HALF, cy - HALF, cx + HALF, cy - HALF, cx + HALF, cy + HALF, cx - HALF, cy + HALF]


class PixelOracle(torch.nn.Module):
    """Foreground where the input is bright; ``invert`` predicts the opposite."""

    def __init__(self, invert=False):
        super().__init__()
        self.invert = invert

    def forward(self, x):
        fg = (x.mean(dim=1, keepdim=True) > 0).float() * 2 - 1
        if self.invert:
            fg = -fg
        return torch.cat([-10 * fg, 10 * fg], dim=1)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _pinned_datasets(monkeypatch):
    """Report the trainer's pinned datasets, whatever the test venv has installed."""
    monkeypatch.setattr(seg, "_datasets_version", lambda: "2.14.0")


# image id -> instance centres. Id 5 is the odd row the trainer's drop_last cuts.
COCO_IMAGES = {1: FOUR, 2: FOUR[:1], 3: [], 4: FOUR[1:3], 5: FOUR[:1]}


@pytest.fixture
def coco_root(tmp_path):
    root = tmp_path / "coco"
    images, anns = [], []
    for image_id, centres in COCO_IMAGES.items():
        name = f"{image_id:012d}.jpg"
        _paint(root / seg.COCO_IMAGES / name, centres)
        images.append({"id": image_id, "file_name": name, "height": SIZE, "width": SIZE})
        for cx, cy in centres:
            anns.append({
                "id": len(anns) + 1, "image_id": image_id, "category_id": 1, "iscrowd": 0,
                "segmentation": [_poly(cx, cy)], "area": float((2 * HALF) ** 2),
                "bbox": [cx - HALF, cy - HALF, 2 * HALF, 2 * HALF],
            })
    (root / "annotations").mkdir(parents=True)
    # Written out of id order: the loader must sort.
    with open(root / seg.COCO_ANNOTATIONS, "w") as f:
        json.dump({"images": images[::-1], "annotations": anns, "categories": [{"id": 1}]}, f)
    return root


def _cocoa_features():
    """shunk031/COCOA's features with decode_rle=True (COCOA.py ``_info``)."""
    import datasets as ds

    return ds.Features({
        "image_id": ds.Value("int64"),
        "license_id": ds.Value("int32"),
        "file_name": ds.Value("string"),
        "height": ds.Value("int32"),
        "width": ds.Value("int32"),
        "date_captured": ds.Value("string"),
        "flickr_url": ds.Value("string"),
        "image": ds.Image(),
        "coco_url": ds.Value("string"),
        "annotations": ds.Sequence({
            "author": ds.Value("string"),
            "url": ds.Value("string"),
            "regions": ds.Sequence({
                "segmentation": ds.Image(),
                "name": ds.Value("string"),
                "area": ds.Value("float32"),
                "is_stuff": ds.Value("bool"),
                "occlude_rate": ds.Value("float32"),
                "order": ds.Value("int32"),
                "visible_mask": ds.Image(),
                "invisible_mask": ds.Image(),
            }),
            "image_id": ds.Value("int64"),
            "depth_constraint": ds.Value("string"),
            "size": ds.Value("int32"),
        }),
    })


class StandInCocoaBuilder:
    """Yields what COCOA.py's ``_generate_examples`` yields with decode_rle=True."""

    class info:
        features = _cocoa_features()

    def __init__(self, images):
        self.images = images  # image id -> centres

    def _generate_examples(self, split, base_image_dir, amodal_annotation_path):
        assert split == "val" and Path(amodal_annotation_path).is_file()
        for idx, (image_id, centres) in enumerate(self.images.items()):
            name = f"COCO_val2014_{image_id:012d}.jpg"
            image = Image.open(Path(base_image_dir) / "val2014" / name).convert("RGB")
            regions = [{
                # The builder decodes each polygon to a 0/255 uint8 mask.
                "segmentation": _square_mask([c]) * 255, "name": f"r{i}",
                "area": float((2 * HALF) ** 2), "is_stuff": False, "occlude_rate": 0.0,
                "order": i + 1, "visible_mask": None, "invisible_mask": None,
            } for i, c in enumerate(centres)]
            yield idx, {
                "image_id": image_id, "license_id": 1, "file_name": name,
                "height": SIZE, "width": SIZE, "date_captured": "2013-01-01 00:00:00",
                "flickr_url": "", "coco_url": "", "image": image,
                "annotations": [{
                    "author": "a", "url": "", "regions": regions, "image_id": image_id,
                    "depth_constraint": "", "size": len(regions),
                }],
            }


# COCOA's builder skips images without amodal annotations, so every row has one.
COCOA_IMAGES = {11: FOUR, 12: FOUR[2:], 13: FOUR[:1], 14: FOUR[1:]}


def _cocoa_tree(root, images):
    for image_id, centres in images.items():
        _paint(root / "val2014" / f"COCO_val2014_{image_id:012d}.jpg", centres)
    (root / "annotations").mkdir(parents=True, exist_ok=True)


@pytest.fixture
def cocoa_rows(tmp_path):
    root = tmp_path / "cocoa"
    _cocoa_tree(root, COCOA_IMAGES)
    (root / seg.COCOA_ANNOTATIONS).write_text("{}")
    return seg.cocoa_rows(root, builder=StandInCocoaBuilder(COCOA_IMAGES))


def _score(rows, model, size=192, **kw):
    return seg.score(model, rows, size, "cpu", **kw)


# ---------------------------------------------------------------------------
# End to end: real loaders, real transform and collate, real evaluate()
# ---------------------------------------------------------------------------

class TestCoco:
    def test_rows_are_in_image_id_order(self, coco_root):
        rows = seg.CocoRows(coco_root)
        assert [rows[i]["image_id"] for i in range(len(rows))] == [1, 2, 4, 5]
        assert len(rows[0]["annotations"]) == 4

    def test_the_target_is_every_instance(self, coco_root):
        out = seg.TransformedRows(seg.CocoRows(coco_root), 192)[0]
        from scipy import ndimage
        assert ndimage.label(out["segmentation_mask"].numpy() > 0)[1] == 4

    def test_the_oracle_scores_near_perfect_and_its_inverse_near_zero(self, coco_root):
        rows = seg.CocoRows(coco_root)
        good = _score(rows, PixelOracle())
        bad = _score(rows, PixelOracle(invert=True))
        assert good["iou"] > 95 and good["dice"] > 95
        assert bad["iou"] < 5
        # Image 3 has no annotations and is skipped, as the HF split drops it.
        assert (good["n_rows"], good["n_images"], good["n_dropped_tail"], good["n_failed"]) == (4, 4, 0, 0)

    def test_scores_at_the_requested_input_size(self, coco_root):
        r = _score(seg.CocoRows(coco_root), PixelOracle(), size=352)
        assert r["input_size"] == 352 and r["iou"] > 95

    def test_max_batches_limits_the_images_scored(self, coco_root):
        r = _score(seg.CocoRows(coco_root), PixelOracle(), max_batches=1)
        assert r["n_images"] == 2 and r["n_failed"] == 0

    def test_images_without_annotations_are_skipped_and_reported(self, coco_root, capsys):
        rows = seg.CocoRows(coco_root)
        assert [rows[i]["image_id"] for i in range(len(rows))] == [1, 2, 4, 5]
        assert rows.skipped_empty == 1
        err = capsys.readouterr()
        assert "skipped 1 images with no annotations, scoring 4" in err.err
        assert err.out == ""   # stdout stays machine-readable

    def test_missing_annotations_is_an_eval_error(self, tmp_path):
        with pytest.raises(seg.EvalError, match="instances_val2017.json"):
            seg.CocoRows(tmp_path)


class TestCocoa:
    def test_rows_have_the_trainer_loader_shape(self, cocoa_rows):
        ann = cocoa_rows[0]["annotations"]
        # Sequence({...}) is stored as columns: a dict of lists, regions per annotation.
        assert isinstance(ann, dict) and isinstance(ann["regions"], list)
        assert len(ann["regions"][0]["segmentation"]) == 4

    def test_the_target_is_every_region(self, cocoa_rows):
        from scipy import ndimage
        out = seg.TransformedRows(cocoa_rows, 192)[0]
        assert out["valid"] is not False
        assert ndimage.label(out["segmentation_mask"].numpy() > 0)[1] == 4

    def test_the_oracle_scores_near_perfect_and_its_inverse_near_zero(self, cocoa_rows):
        good = _score(cocoa_rows, PixelOracle())
        bad = _score(cocoa_rows, PixelOracle(invert=True))
        assert good["iou"] > 95 and bad["iou"] < 5
        assert (good["n_images"], good["n_failed"]) == (4, 0)

    def test_missing_annotations_is_an_eval_error(self, tmp_path):
        with pytest.raises(seg.EvalError, match="COCO_amodal_val2014.json"):
            seg.cocoa_rows(tmp_path, builder=StandInCocoaBuilder({}))

    def test_missing_builder_script_is_an_eval_error(self, tmp_path):
        with pytest.raises(seg.EvalError, match="COCOA.py"):
            seg.load_cocoa_builder(tmp_path)


@pytest.mark.skipif(not os.environ.get("COCOA_BUILDER_SCRIPT"),
                    reason="set COCOA_BUILDER_SCRIPT to a copy of shunk031/COCOA's COCOA.py")
def test_the_real_cocoa_builder_end_to_end(tmp_path):
    pytest.importorskip("pycocotools")
    root = tmp_path / "cocoa"
    _cocoa_tree(root, {7: FOUR})
    shutil.copy(os.environ["COCOA_BUILDER_SCRIPT"], root / seg.COCOA_BUILDER)
    regions = [{"segmentation": _poly(cx, cy), "name": f"r{i}", "area": 4096.0,
                "isStuff": False, "occlude_rate": 0.0, "order": i + 1}
               for i, (cx, cy) in enumerate(FOUR)]
    name = "COCO_val2014_000000000007.jpg"
    (root / seg.COCOA_ANNOTATIONS).write_text(json.dumps({
        "images": [{"id": 7, "file_name": name, "height": SIZE, "width": SIZE,
                    "flickr_url": "", "coco_url": "", "license": 1,
                    "date_captured": "2013-01-01 00:00:00"}],
        "annotations": [{"image_id": 7, "author": "a", "url": "", "depth_constraint": "",
                         "size": 4, "regions": regions}],
    }))
    rows = seg.cocoa_rows(root)
    rows = [rows[0], rows[0]]  # one batch of the trainer's size
    good = _score(rows, PixelOracle())
    assert good["iou"] > 95 and good["n_images"] == 2


class TestPlaceholderBatches:
    def test_a_batch_that_failed_entirely_is_not_scored(self, coco_root):
        rows = list(seg.CocoRows(coco_root))[:2]
        broken = [{"image": object(), "annotations": [], "image_id": 99}] * 2
        r = _score(rows + broken, PixelOracle())
        assert (r["n_images"], r["n_failed"]) == (2, 2)
        assert r["iou"] > 95  # the blank placeholder did not drag the score

    def test_a_partly_failed_batch_counts_each_failure(self, coco_root):
        rows = list(seg.CocoRows(coco_root))[:3]
        broken = {"image": object(), "annotations": [], "image_id": 99}
        r = _score(rows + [broken], PixelOracle())
        assert (r["n_images"], r["n_failed"]) == (3, 1)


# ---------------------------------------------------------------------------
# Input size and checkpoints
# ---------------------------------------------------------------------------

class TestInputSize:
    def _ckpt(self, tmp_path, meta=None, sidecar="model.json"):
        ckpt = tmp_path / "model.pth"
        ckpt.write_bytes(b"")
        if meta is not None:
            (tmp_path / sidecar).write_text(meta if isinstance(meta, str) else json.dumps(meta))
        return ckpt

    def test_from_the_checkpoint_metadata(self, tmp_path):
        ckpt = self._ckpt(tmp_path, {"custom_metadata": {"input_size": 352}})
        size, source = seg.resolve_size(ckpt)
        assert size == 352 and "model.json" in source

    @pytest.mark.parametrize("sidecar", ["model.json", "model.meta.json", "metadata.json"])
    def test_every_sidecar_name_is_found(self, tmp_path, sidecar):
        ckpt = self._ckpt(tmp_path, {"input_size": 288}, sidecar=sidecar)
        assert seg.resolve_size(ckpt)[0] == 288

    def test_the_flag_overrides_the_metadata(self, tmp_path):
        ckpt = self._ckpt(tmp_path, {"input_size": 352})
        assert seg.resolve_size(ckpt, override=256) == (256, "--input-size")

    def test_no_metadata_is_the_default(self, tmp_path):
        assert seg.resolve_size(self._ckpt(tmp_path))[0] == 192

    def test_the_demo_environment_override_is_ignored(self, tmp_path, monkeypatch):
        monkeypatch.setenv("ECS_HCM_DEMO_INPUT_SIZE", "96")
        ckpt = self._ckpt(tmp_path, {"input_size": 352})
        assert seg.resolve_size(ckpt)[0] == 352

    def test_an_unreadable_sidecar_is_an_error_not_the_default(self, tmp_path):
        ckpt = self._ckpt(tmp_path, "{not json")
        with pytest.raises(seg.EvalError, match="could not be read"):
            seg.resolve_size(ckpt)

    def test_an_unusable_size_is_an_error(self, tmp_path):
        ckpt = self._ckpt(tmp_path, {"input_size": "big"})
        with pytest.raises(seg.EvalError):
            seg.resolve_size(ckpt)

    def test_an_explicit_metadata_file_that_is_missing_is_an_error(self, tmp_path):
        with pytest.raises(seg.EvalError, match="not found"):
            seg.resolve_size(self._ckpt(tmp_path), explicit_meta=tmp_path / "nope.json")

    @pytest.mark.parametrize("bad", [0, -192])
    def test_a_non_positive_flag_is_an_error(self, tmp_path, bad):
        with pytest.raises(seg.EvalError):
            seg.resolve_size(self._ckpt(tmp_path), override=bad)


class TestLoadCheckpoint:
    def test_a_state_dict_loads_strictly_into_hcm_nn(self, tmp_path):
        from ecs_hcm_model.core.hcm_nn import hcm_nn
        model = hcm_nn(enable_finetuning=True, num_segmentation_classes=2, out_stride=16)
        path = tmp_path / "m.pth"
        torch.save({"state_dict": model.state_dict()}, path)
        loaded = seg.load_checkpoint(path, "cpu")
        assert isinstance(loaded, hcm_nn)
        assert all(torch.equal(a, b) for a, b in zip(model.state_dict().values(),
                                                      loaded.state_dict().values()))

    def test_a_checkpoint_without_a_segmentation_head_is_refused(self, tmp_path):
        path = tmp_path / "m.pth"
        torch.save({"backbone.w": torch.zeros(1)}, path)
        with pytest.raises(seg.EvalError, match="segmentation"):
            seg.load_checkpoint(path, "cpu")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

class TestCli:
    @pytest.fixture
    def run(self, coco_root, tmp_path, monkeypatch, capsys):
        monkeypatch.setattr(seg, "load_checkpoint", lambda path, device, **_: PixelOracle())

        def _run(*extra, meta=None):
            ckpt = tmp_path / "best.pth"
            ckpt.write_bytes(b"")
            if meta is not None:
                (tmp_path / "best.json").write_text(json.dumps(meta))
            code = seg.main(["--dataset", "coco", "--data-root", str(coco_root),
                             "--device", "cpu", *extra, str(ckpt)])
            return code, capsys.readouterr()
        return _run

    def test_one_result_line_per_checkpoint(self, run):
        code, out = run(meta={"custom_metadata": {"input_size": 352}})
        assert code == 0
        [line] = out.out.strip().splitlines()
        assert line.startswith("RESULT ") and "dataset=coco" in line
        assert "n_images=4" in line and "input_size=352" in line and "iou=" in line

    def test_json_output(self, run):
        code, out = run("--json", "--input-size", "256")
        r = json.loads(out.out)
        assert code == 0 and r["input_size"] == 256 and r["input_size_source"] == "--input-size"
        assert {"iou", "dice", "n_images", "checkpoint"} <= set(r)

    def test_two_checkpoints_and_one_failure(self, coco_root, tmp_path, monkeypatch, capsys):
        def load(path, device, **_):
            if Path(path).name == "bad.pth":
                raise seg.EvalError("checkpoint has no segmentation_head.weight")
            return PixelOracle()
        monkeypatch.setattr(seg, "load_checkpoint", load)
        paths = []
        for name in ("a.pth", "bad.pth", "b.pth"):
            (tmp_path / name).write_bytes(b"")
            paths.append(str(tmp_path / name))
        code = seg.main(["--dataset", "coco", "--data-root", str(coco_root),
                         "--device", "cpu", *paths])
        out = capsys.readouterr()
        lines = out.out.strip().splitlines()
        assert code == 1 and len(lines) == 2
        assert "a.pth" in lines[0] and "b.pth" in lines[1]
        assert "bad.pth" in out.err

    def test_defaults_are_jetson_safe(self):
        args = seg.parse_args(["--dataset", "cocoa", "--data-root", "x", "a.pth"])
        assert (args.num_workers, args.batch_size, args.input_size) == (0, 2, None)


def test_a_bad_sidecar_fails_before_any_data_is_read(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(seg, "build_rows", lambda *a: pytest.fail("data was read"))
    ckpt = tmp_path / "best.pth"
    ckpt.write_bytes(b"")
    (tmp_path / "best.json").write_text("{broken")
    code = seg.main(["--dataset", "coco", "--data-root", str(tmp_path), str(ckpt)])
    assert code == 2 and "could not be read" in capsys.readouterr().err


def test_importing_the_tool_survives_a_broken_torchaudio(tmp_path):
    """The Jetson's torchaudio raises on import; the tool must never import it."""
    fake = tmp_path / "torchaudio"
    fake.mkdir()
    (fake / "__init__.py").write_text("raise OSError('broken native library')\n")
    env = dict(os.environ, PYTHONPATH=os.pathsep.join([str(tmp_path), str(REPO)]),
               MLFLOW_DISABLE_AGENT_HINT="1")
    code = ("import sys, ecs_hcm_model.eval.seg; "
            "print('torchaudio', sys.modules.get('torchaudio'))")
    p = subprocess.run([sys.executable, "-W", "ignore", "-c", code], env=env,
                       capture_output=True, text=True, timeout=300)
    assert p.returncode == 0, p.stderr[-2000:]
    assert "torchaudio None" in p.stdout


# ---------------------------------------------------------------------------
# Fetch helper
# ---------------------------------------------------------------------------

def _opener(payloads, calls, headers=None):
    def open_(req):
        url = getattr(req, "full_url", req)
        calls.append(url)
        if headers is not None:
            headers[url] = dict(getattr(req, "headers", {}))
        return io.BytesIO(payloads(url))
    return open_


def _gz_tar(path, member, body):
    with tarfile.open(path, "w:gz") as t:
        info = tarfile.TarInfo(member)
        info.size = len(body)
        t.addfile(info, io.BytesIO(body))


class TestFetch:
    def test_download_skips_a_file_that_is_present(self, tmp_path):
        calls = []
        dest = tmp_path / "a" / "x.jpg"
        assert fetch.download("u", dest, opener=_opener(lambda u: b"img", calls)) is True
        assert fetch.download("u", dest, opener=_opener(lambda u: b"img", calls)) is False
        assert calls == ["u"] and dest.read_bytes() == b"img"
        assert not dest.with_name("x.jpg.part").exists()

    def test_images_fetch_only_what_the_annotations_list(self, tmp_path):
        ann = tmp_path / "ann.json"
        ann.write_text(json.dumps({"images": [{"file_name": "a.jpg"}, {"file_name": "b.jpg"}]}))
        (tmp_path / "img").mkdir()
        (tmp_path / "img" / "a.jpg").write_bytes(b"old")
        calls = []
        got = fetch.fetch_images(ann, tmp_path / "img", "http://h/{}",
                                 opener=_opener(lambda u: b"new", calls))
        assert got == (1, 1) and calls == ["http://h/b.jpg"]
        assert (tmp_path / "img" / "a.jpg").read_bytes() == b"old"

    def test_cocoa_refuses_a_builder_that_is_not_the_pinned_one(self, tmp_path):
        (tmp_path / "COCOA.py").write_text("tampered")
        with pytest.raises(SystemExit, match="sha256"):
            fetch.fetch_cocoa(tmp_path, opener=_opener(lambda u: b"", []))

    def test_cocoa_downloads_the_gated_annotations_with_the_token(self, tmp_path, monkeypatch):
        (tmp_path / "COCOA.py").write_bytes(b"builder")
        monkeypatch.setattr(fetch, "COCOA_BUILDER_SHA256", fetch.sha256(tmp_path / "COCOA.py"))
        monkeypatch.setenv("HF_TOKEN", "tok")
        body = json.dumps({"images": [{"file_name": "COCO_val2014_1.jpg"}]}).encode()
        src = tmp_path / "src.tar.gz"
        _gz_tar(src, "annotations/" + fetch.COCOA_ANNOTATIONS_JSON, body)
        payloads = {fetch.COCOA_ANNOTATIONS_URL: src.read_bytes()}
        calls, headers = [], {}
        fetch.fetch_cocoa(tmp_path, opener=_opener(lambda u: payloads.get(u, b"img"), calls, headers))
        assert (tmp_path / "annotations" / fetch.COCOA_ANNOTATIONS_JSON).read_bytes() == body
        assert calls == [fetch.COCOA_ANNOTATIONS_URL, fetch.COCOA_IMAGE_URL.format("COCO_val2014_1.jpg")]
        assert headers[fetch.COCOA_ANNOTATIONS_URL] == {"Authorization": "Bearer tok"}
        # The token goes to Hugging Face only, never to the image host.
        assert headers[calls[1]] == {}

    def test_cocoa_refused_by_the_gate_says_what_to_do(self, tmp_path, monkeypatch):
        import urllib.error
        (tmp_path / "COCOA.py").write_bytes(b"builder")
        monkeypatch.setattr(fetch, "COCOA_BUILDER_SHA256", fetch.sha256(tmp_path / "COCOA.py"))
        monkeypatch.delenv("HF_TOKEN", raising=False)

        def gated(req):
            raise urllib.error.HTTPError(getattr(req, "full_url", req), 401, "Unauthorized", {}, None)
        with pytest.raises(SystemExit, match="HF_TOKEN.*--annotations-tar"):
            fetch.fetch_cocoa(tmp_path, opener=gated)

    def test_cocoa_annotations_come_out_of_the_tarball(self, tmp_path, monkeypatch):
        (tmp_path / "COCOA.py").write_bytes(b"builder")
        monkeypatch.setattr(fetch, "COCOA_BUILDER_SHA256", fetch.sha256(tmp_path / "COCOA.py"))
        body = json.dumps({"images": [{"file_name": "COCO_val2014_1.jpg"}]}).encode()
        tar_path = tmp_path / "cocoa_annotation.tar.gz"
        _gz_tar(tar_path, "annotations/" + fetch.COCOA_ANNOTATIONS_JSON, body)
        calls = []
        fetch.fetch_cocoa(tmp_path, annotations_tar=tar_path,
                          opener=_opener(lambda u: b"img", calls))
        assert (tmp_path / "annotations" / fetch.COCOA_ANNOTATIONS_JSON).read_bytes() == body
        assert calls == [fetch.COCOA_IMAGE_URL.format("COCO_val2014_1.jpg")]

    def test_coco_extracts_the_val_annotations_and_removes_the_zip(self, tmp_path):
        import zipfile
        body = json.dumps({"images": [{"file_name": "1.jpg"}]}).encode()
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as z:
            z.writestr(fetch.COCO_ANNOTATIONS_MEMBER, body)
        payloads = {fetch.COCO_ANNOTATIONS_ZIP: buf.getvalue()}
        calls = []
        fetch.fetch_coco(tmp_path, opener=_opener(lambda u: payloads.get(u, b"img"), calls))
        assert (tmp_path / "annotations" / "instances_val2017.json").read_bytes() == body
        assert not (tmp_path / "annotations_trainval2017.zip").exists()
        assert calls == [fetch.COCO_ANNOTATIONS_ZIP, fetch.COCO_IMAGE_URL.format("1.jpg")]


class _CutsOffMidResponse(io.BytesIO):
    """A response that hands over nothing and drops the connection."""

    def read(self, *a):
        raise http.client.IncompleteRead(self.getvalue())


class _FlakyOpener:
    """Fails the first ``fail_times`` tries partway through the body.

    Records at each try whether the destination's ``.part`` file was already
    there: a retry must always start from an empty one. The opener is called
    before the file is opened, so the record reflects the previous try's
    cleanup.
    """

    def __init__(self, fail_times, part, body=b"img"):
        self.fail_times = fail_times
        self.part = part
        self.body = body
        self.calls = 0
        self.part_at_try_start = []

    def __call__(self, req):
        self.calls += 1
        self.part_at_try_start.append(self.part.exists())
        if self.calls <= self.fail_times:
            return _CutsOffMidResponse(b"half-written")
        return io.BytesIO(self.body)


class _StalledServer:
    """Accepts requests and answers none for ``hold`` seconds.

    The incident: a GET to images.cocodataset.org that stopped answering, so
    ``http.client.getresponse`` sat on the socket until the kernel's own TCP
    timeout ended the whole fetch. After the hold the server also stops
    listening, so a client with no timeout fails on its next try instead of
    hanging the suite.
    """

    def __init__(self, hold=6.0):
        self.hold = hold
        self.sock = socket.socket()
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind(("127.0.0.1", 0))
        self.sock.listen(8)
        self.url = f"http://127.0.0.1:{self.sock.getsockname()[1]}/x.jpg"
        self.held, self.stop = [], threading.Event()
        self.thread = threading.Thread(target=self._serve, daemon=True)

    def _serve(self):
        self.sock.settimeout(0.1)
        end = time.monotonic() + self.hold
        while not self.stop.is_set() and time.monotonic() < end:
            try:
                conn, _ = self.sock.accept()
            except (TimeoutError, OSError):
                continue
            self.held.append(conn)  # accepted, never read, never answered
        # Give up on the hold: a client with no timeout must fail here rather
        # than block forever (the kernel's TCP timeout, in the incident).
        self._close()

    def _close(self):
        for conn in self.held:
            try:
                conn.close()
            except OSError:
                pass
        # Stop listening too, so a client that keeps trying is refused at once
        # rather than left waiting on a socket nobody will accept.
        try:
            self.sock.close()
        except OSError:
            pass

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *exc):
        self.stop.set()
        self.thread.join(timeout=10)
        self._close()
        return False


class TestFetchTimeoutAndRetry:
    """A stalled or dropped transfer is bounded, and costs one image, not the run."""

    def test_the_opener_hands_the_timeout_to_urllib_for_every_request(self, tmp_path, monkeypatch):
        seen = []

        class Director:
            handlers = []

            def open(self, req, timeout=None):
                seen.append(timeout)
                return io.BytesIO(b"img")

        monkeypatch.setattr(urllib.request, "build_opener", lambda *h: Director())
        opener = fetch.make_opener()
        assert fetch.download("http://h/a.jpg", tmp_path / "a", opener=opener) is True
        assert seen and seen[0] is not None, "the opener issued a request with no timeout"
        assert seen == [fetch.FETCH_TIMEOUT_S]

    def test_the_default_opener_carries_the_timeout_and_every_fetch_uses_it(self):
        import inspect
        timeout = getattr(fetch._open.__self__, "timeout", None)
        assert timeout is not None and timeout > 0, "the default opener is unbounded"
        assert timeout == fetch.FETCH_TIMEOUT_S
        # ... and it is the shared default, so no fetch function can opt out of it.
        for fn in (fetch.download, fetch.fetch_images, fetch.fetch_coco, fetch.fetch_cocoa):
            assert inspect.signature(fn).parameters["opener"].default is fetch._open

    def test_a_stalled_response_ends_at_the_timeout_with_the_url_in_the_error(self, tmp_path, monkeypatch):
        monkeypatch.setattr(fetch, "FETCH_TIMEOUT_S", 0.2, raising=False)
        monkeypatch.setattr(fetch, "DOWNLOAD_BACKOFF_S", 0.0, raising=False)
        dest = tmp_path / "x" / "img.jpg"
        with _StalledServer(hold=3.0) as srv:
            t0 = time.monotonic()
            with pytest.raises(OSError) as e:
                fetch.download(srv.url, dest, opener=fetch.make_opener())
            elapsed = time.monotonic() - t0
        # Three tries of 0.2 s is a 0.6 s floor, and a client with no timeout sits
        # on the held socket until the helper stops listening at 3 s, so the bound
        # has ~2.5x headroom over the floor and still fails fast.
        assert elapsed < 1.5, f"the read was not bounded ({elapsed:.1f}s)"
        assert str(e.value).startswith(srv.url), str(e.value)
        assert not dest.exists() and not dest.with_name("img.jpg.part").exists()

    def test_a_failed_try_is_retried_from_an_empty_part_file(self, tmp_path, monkeypatch):
        monkeypatch.setattr(fetch, "DOWNLOAD_BACKOFF_S", 0.0, raising=False)
        dest = tmp_path / "x" / "img.jpg"
        opener = _FlakyOpener(fail_times=1, part=dest.with_name("img.jpg.part"))
        assert fetch.download("http://h/img.jpg", dest, opener=opener) is True
        assert opener.calls == 2
        assert opener.part_at_try_start == [False, False]
        assert dest.read_bytes() == b"img"
        assert not dest.with_name("img.jpg.part").exists()

    def test_the_wait_grows_between_tries(self, tmp_path):
        waits = []
        dest = tmp_path / "img.jpg"
        opener = _FlakyOpener(fail_times=2, part=dest.with_name("img.jpg.part"))
        assert fetch.download("http://h/img.jpg", dest, opener=opener,
                              sleep=waits.append) is True
        assert waits == [fetch.DOWNLOAD_BACKOFF_S, 2 * fetch.DOWNLOAD_BACKOFF_S]
        assert opener.calls == 3

    def test_every_try_failing_names_the_url_and_leaves_nothing_behind(self, tmp_path, monkeypatch):
        monkeypatch.setattr(fetch, "DOWNLOAD_BACKOFF_S", 0.0, raising=False)
        dest = tmp_path / "x" / "img.jpg"
        url = "http://h/img.jpg"
        opener = _FlakyOpener(fail_times=3, part=dest.with_name("img.jpg.part"))  # DOWNLOAD_TRIES
        with pytest.raises(OSError) as e:
            fetch.download(url, dest, opener=opener)
        assert opener.calls == 3
        assert str(e.value).startswith(url), str(e.value)
        assert not dest.exists() and not dest.with_name("img.jpg.part").exists()

    def test_a_refusal_is_reported_at_once_and_never_retried(self, tmp_path):
        dest = tmp_path / "img.jpg"
        calls = []

        def refuse(req):
            calls.append(1)
            raise urllib.error.HTTPError("https://huggingface.co/x", 403, "Forbidden", {}, None)

        with pytest.raises(urllib.error.HTTPError) as e:
            fetch.download("https://huggingface.co/x", dest, opener=refuse)
        assert len(calls) == 1  # a refusal is the server's answer, not a hiccup
        assert type(e.value) is urllib.error.HTTPError and e.value.code == 403
        assert not dest.with_name("img.jpg.part").exists()

    def test_a_server_error_is_retried_and_its_answer_is_raised_unchanged(self, tmp_path, monkeypatch):
        monkeypatch.setattr(fetch, "DOWNLOAD_BACKOFF_S", 0.0, raising=False)
        url = "https://cdn.example/annotations.tar.gz"
        dest = tmp_path / "img.jpg"
        calls = []

        def unavailable(req):
            calls.append(1)
            raise urllib.error.HTTPError(url, 503, "Service Unavailable", {}, None)

        with pytest.raises(urllib.error.HTTPError) as e:
            fetch.download(url, dest, opener=unavailable)
        assert len(calls) == 3
        # Unwrapped: fetch_cocoa builds the gated-repo message from this object.
        assert type(e.value) is urllib.error.HTTPError
        assert (e.value.code, e.value.url) == (503, url)

    def test_an_empty_destination_is_still_fetched_not_skipped(self, tmp_path):
        dest = tmp_path / "img.jpg"
        dest.write_bytes(b"")
        opener = _FlakyOpener(fail_times=0, part=dest.with_name("img.jpg.part"))
        assert fetch.download("http://h/img.jpg", dest, opener=opener) is True
        assert opener.calls == 1 and dest.read_bytes() == b"img"


class _RedirectServer:
    """A local HTTP server that records the Authorization header per path.

    ``/cross`` redirects to the same port under the other loopback name, a
    different host to urllib; ``/same`` redirects to a path on this host.
    """

    def __init__(self):
        import http.server
        import threading
        seen = self.seen = {}

        class Handler(http.server.BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_GET(self):
                seen[self.path] = self.headers.get("Authorization")
                port = self.server.server_address[1]
                hops = {"/cross": f"http://localhost:{port}/landed-cross",
                        "/same": f"http://127.0.0.1:{port}/landed-same"}
                if self.path in hops:
                    self.send_response(302)
                    self.send_header("Location", hops[self.path])
                    self.send_header("Content-Length", "0")
                    self.end_headers()
                    return
                body = b"payload"
                self.send_response(200)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        self.httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.base = f"http://127.0.0.1:{self.httpd.server_address[1]}"
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    def close(self):
        self.httpd.shutdown()
        self.httpd.server_close()


@pytest.fixture
def redirect_server(monkeypatch):
    for var in ("http_proxy", "HTTP_PROXY", "all_proxy", "ALL_PROXY"):
        monkeypatch.delenv(var, raising=False)
    server = _RedirectServer()
    yield server
    server.close()


class TestTheTokenStaysOnItsHost:
    AUTH = {"Authorization": "Bearer tok"}

    def test_a_cross_host_redirect_drops_the_token(self, redirect_server, tmp_path):
        fetch.download(redirect_server.base + "/cross", tmp_path / "x",
                       opener=fetch.make_opener(), headers=self.AUTH)
        assert redirect_server.seen["/cross"] == "Bearer tok"
        assert "/landed-cross" in redirect_server.seen
        assert redirect_server.seen["/landed-cross"] is None
        assert (tmp_path / "x").read_bytes() == b"payload"

    def test_a_same_host_redirect_keeps_the_token(self, redirect_server, tmp_path):
        fetch.download(redirect_server.base + "/same", tmp_path / "x",
                       opener=fetch.make_opener(), headers=self.AUTH)
        assert redirect_server.seen["/landed-same"] == "Bearer tok"

    def test_plain_urllib_would_have_leaked_it(self, redirect_server, tmp_path):
        # The control: without the handler, the token follows the redirect.
        # If this ever fails, urllib changed and the handler's reason is gone.
        import urllib.request
        fetch.download(redirect_server.base + "/cross", tmp_path / "x",
                       opener=urllib.request.build_opener().open, headers=self.AUTH)
        assert redirect_server.seen["/landed-cross"] == "Bearer tok"

    def test_the_default_opener_is_the_guarded_one(self):
        import inspect
        for fn in (fetch.download, fetch.fetch_images, fetch.fetch_coco, fetch.fetch_cocoa):
            assert inspect.signature(fn).parameters["opener"].default is fetch._open
        assert any(isinstance(h, fetch._DropAuthOnHostChange)
                   for h in fetch._open.__self__.handlers)

    def test_a_cdn_refusal_after_the_redirect_is_a_clear_error(self, tmp_path, monkeypatch):
        import urllib.error
        (tmp_path / "COCOA.py").write_bytes(b"builder")
        monkeypatch.setattr(fetch, "COCOA_BUILDER_SHA256", fetch.sha256(tmp_path / "COCOA.py"))
        monkeypatch.setenv("HF_TOKEN", "tok")
        cdn = "https://cdn-lfs.example/annotations.tar.gz?X-Amz-Signature=abc"

        def presigned_400(req):
            raise urllib.error.HTTPError(cdn, 400, "Bad Request", {}, None)
        with pytest.raises(SystemExit, match=r"400 Bad Request.*--annotations-tar") as e:
            fetch.fetch_cocoa(tmp_path, opener=presigned_400)
        # The configured URL, never the presigned one the redirect led to:
        # its query string is a live signature.
        assert fetch.COCOA_ANNOTATIONS_URL in str(e.value)
        assert "cdn-lfs.example" not in str(e.value)
        assert "X-Amz-Signature" not in str(e.value)
        assert "tok" not in str(e.value).replace("token", "")

    @staticmethod
    def _hop(handler, old, new):
        import urllib.request
        req = urllib.request.Request(old, headers={"Authorization": "Bearer tok"})
        out = handler.redirect_request(req, None, 302, "Found", {}, new)
        return out.get_header("Authorization")

    def test_an_https_to_http_redirect_on_the_same_host_drops_the_token(self):
        h = fetch._DropAuthOnHostChange()
        assert self._hop(h, "https://huggingface.co/a", "http://huggingface.co/b") is None

    def test_https_to_https_on_the_same_host_keeps_the_token(self):
        h = fetch._DropAuthOnHostChange()
        assert self._hop(h, "https://huggingface.co/a", "https://huggingface.co/b") == "Bearer tok"

    def test_plain_urllib_would_have_carried_it_over_the_downgrade(self):
        # The control: the stock handler keeps the token on https -> http.
        import urllib.request
        h = urllib.request.HTTPRedirectHandler()
        assert self._hop(h, "https://huggingface.co/a", "http://huggingface.co/b") == "Bearer tok"


def test_the_result_line_reports_the_batch_size_and_the_dropped_tail():
    r = {"checkpoint": "a.pth", "dataset": "coco", "iou": 50.0, "dice": 60.0,
         "loss": 0.5, "n_images": 4, "input_size": 352, "input_size_source": "metadata",
         "batch_size": 2, "n_dropped_tail": 1, "seconds": 3.0, "n_failed": 0,
         "precision": "fp16", "precision_requested": "fp16", "score_seconds": 2.0, "data_wait_seconds": 1.5,
         "model_seconds": 0.5, "images_per_s": 2.0}
    line = seg.format_result(r)
    assert "batch_size=2" in line and "n_dropped_tail=1" in line


class TestEvalStride:
    """f4657a64 #3: a state dict is built at the recorded stride, not 16."""

    @staticmethod
    def _save(tmp_path):
        from ecs_hcm_model.core.hcm_nn import hcm_nn
        model = hcm_nn(enable_finetuning=True, num_segmentation_classes=2, out_stride=16)
        path = tmp_path / "best.pth"
        torch.save(model.state_dict(), path)
        return path

    def test_the_sidecar_stride_is_used(self, tmp_path):
        loaded = seg.load_checkpoint(self._save(tmp_path), "cpu",
                                     meta={"custom_metadata": {"out_stride": 8}})
        assert loaded.out_stride == 8

    def test_the_flag_beats_the_sidecar_and_warns(self, tmp_path):
        logged = []
        loaded = seg.load_checkpoint(self._save(tmp_path), "cpu", meta={"out_stride": 8},
                                     out_stride=32, log=logged.append)
        assert loaded.out_stride == 32
        assert any("overrides 8" in m for m in logged)

    def test_the_demo_environment_is_ignored(self, tmp_path, monkeypatch):
        monkeypatch.setenv("ECS_HCM_DEMO_OUT_STRIDE", "32")
        assert seg.load_checkpoint(self._save(tmp_path), "cpu").out_stride == 16

    def test_an_unusable_recorded_stride_is_an_eval_error(self, tmp_path):
        with pytest.raises(seg.EvalError, match="out_stride"):
            seg.load_checkpoint(self._save(tmp_path), "cpu", meta={"out_stride": 12})

    def test_the_cli_hands_each_checkpoint_its_sidecar(self, coco_root, tmp_path, monkeypatch, capsys):
        seen = {}
        def load(path, device, **kw):
            seen.update(kw)
            return PixelOracle()
        monkeypatch.setattr(seg, "load_checkpoint", load)
        ckpt = tmp_path / "best.pth"
        ckpt.write_bytes(b"")
        (tmp_path / "best.json").write_text(json.dumps({"custom_metadata": {"out_stride": 8}}))
        code = seg.main(["--dataset", "coco", "--data-root", str(coco_root), "--device", "cpu",
                         "--out-stride", "32", str(ckpt)])
        assert code == 0
        assert seen["meta"] == {"custom_metadata": {"out_stride": 8}} and seen["out_stride"] == 32


# ---------------------------------------------------------------------------
# datasets version guard
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("v", ["2.14.0", "2.14.7", "2.18.0", "2.18.9"])
def test_releases_before_the_exif_transpose_are_accepted(v):
    seg.check_datasets_version(v)


@pytest.mark.parametrize("v", ["2.19.0", "2.19.0.dev0", "2.21.0", "3.6.0", "5.0.1"])
def test_releases_that_exif_transpose_are_refused(v):
    with pytest.raises(seg.EvalError, match=r"datasets .* EXIF-transpose"):
        seg.check_datasets_version(v)


def test_no_datasets_installed_is_not_refused(monkeypatch):
    monkeypatch.setattr(seg, "_datasets_version", lambda: None)
    seg.check_datasets_version()


def test_the_cli_refuses_before_reading_any_data(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(seg, "_datasets_version", lambda: "2.19.0")

    def no_rows(*a, **k):
        raise AssertionError("rows were built despite a refused datasets release")

    monkeypatch.setattr(seg, "build_rows", no_rows)
    ckpt = tmp_path / "m.pt"
    ckpt.write_bytes(b"")
    code = seg.main(["--dataset", "coco", "--data-root", str(tmp_path), str(ckpt)])
    assert code == 2
    assert "datasets 2.19.0" in capsys.readouterr().err


def test_reading_the_version_does_not_import_datasets(monkeypatch):
    monkeypatch.undo()
    monkeypatch.delitem(sys.modules, "datasets", raising=False)
    seg._datasets_version()
    assert "datasets" not in sys.modules


# ---------------------------------------------------------------------------
# Precision and timing
# ---------------------------------------------------------------------------

class TestPrecisionAndTiming:
    @pytest.fixture
    def autocast_calls(self, monkeypatch):
        """Record ``enabled`` each time evaluate() opens the trainer's autocast.

        CUDA autocast is a no-op on this CPU-only VM in both modes, so the
        switch handed to it is what the test can observe.
        """
        import contextlib
        from ecs_hcm_model.core import trainer
        calls = []

        def recorder(*args, **kwargs):
            calls.append(kwargs.get("enabled", True))
            return contextlib.nullcontext()

        monkeypatch.setattr(trainer, "autocast", recorder)
        return calls

    def test_the_default_is_fp16_as_evaluated_so_far(self):
        args = seg.parse_args(["--dataset", "coco", "--data-root", "x", "a.pth"])
        assert args.precision == "fp16" == seg.DEFAULT_PRECISION

    def test_an_unknown_precision_is_refused(self, capsys):
        with pytest.raises(SystemExit):
            seg.parse_args(["--dataset", "coco", "--data-root", "x",
                            "--precision", "bf16", "a.pth"])

    def test_the_default_runs_every_forward_under_autocast(self, coco_root, autocast_calls):
        _score(seg.build_rows("coco", coco_root), PixelOracle())
        assert autocast_calls and all(autocast_calls)

    def test_fp16_runs_every_forward_under_autocast(self, coco_root, autocast_calls):
        _score(seg.build_rows("coco", coco_root), PixelOracle(), precision="fp16")
        assert autocast_calls and all(autocast_calls)

    def test_fp32_turns_autocast_off_for_every_forward(self, coco_root, autocast_calls):
        _score(seg.build_rows("coco", coco_root), PixelOracle(), precision="fp32")
        assert autocast_calls and not any(autocast_calls)

    def test_the_cli_flag_reaches_the_forward_and_the_output(
            self, coco_root, tmp_path, monkeypatch, capsys, autocast_calls):
        monkeypatch.setattr(seg, "load_checkpoint", lambda path, device, **_: PixelOracle())
        ckpt = tmp_path / "best.pth"
        ckpt.write_bytes(b"")
        code = seg.main(["--dataset", "coco", "--data-root", str(coco_root), "--device",
                         "cpu", "--precision", "fp32", "--json", str(ckpt)])
        r = json.loads(capsys.readouterr().out)
        assert code == 0 and r["precision"] == "fp32"
        assert autocast_calls and not any(autocast_calls)

    def test_the_result_line_records_precision_and_timing(self, coco_root, tmp_path,
                                                          monkeypatch, capsys):
        monkeypatch.setattr(seg, "load_checkpoint", lambda path, device, **_: PixelOracle())
        ckpt = tmp_path / "best.pth"
        ckpt.write_bytes(b"")
        seg.main(["--dataset", "coco", "--data-root", str(coco_root), "--device", "cpu",
                  str(ckpt)])
        [line] = capsys.readouterr().out.strip().splitlines()
        for field in ("precision=fp32", "images_per_s=", "data_wait_s=", "model_s="):
            assert field in line, field

    def test_fp16_off_cuda_is_recorded_as_the_fp32_it_ran_at(self, coco_root, tmp_path,
                                                             monkeypatch, capsys):
        monkeypatch.setattr(seg, "load_checkpoint", lambda path, device, **_: PixelOracle())
        ckpt = tmp_path / "best.pth"
        ckpt.write_bytes(b"")
        code = seg.main(["--dataset", "coco", "--data-root", str(coco_root), "--device",
                         "cpu", "--precision", "fp16", "--json", str(ckpt)])
        out = capsys.readouterr()
        r = json.loads(out.out)
        assert code == 0
        assert (r["precision"], r["precision_requested"]) == ("fp32", "fp16")
        assert "ran as fp32 on cpu" in out.err

    @pytest.mark.parametrize("requested,device,effective", [
        ("fp16", "cuda", "fp16"), ("fp16", "cuda:0", "fp16"), ("fp32", "cuda", "fp32"),
        ("fp16", "cpu", "fp32"), ("fp32", "cpu", "fp32"), ("fp16", "mps", "fp32"),
    ])
    def test_the_effective_precision(self, requested, device, effective):
        assert seg.effective_precision(requested, device) == effective

    def test_the_timing_adds_up(self, coco_root):
        r = _score(seg.build_rows("coco", coco_root), PixelOracle())
        assert r["score_seconds"] > 0
        assert r["data_wait_seconds"] >= 0 and r["model_seconds"] >= 0
        assert r["data_wait_seconds"] + r["model_seconds"] == pytest.approx(r["score_seconds"])
        assert r["images_per_s"] == pytest.approx(r["n_images"] / r["score_seconds"])

    def test_waiting_on_the_loader_is_counted_as_data_time(self, monkeypatch):
        # A fake clock, not sleeps: wall-clock bounds flake on a loaded VM.
        # The loader takes 0.05 s a batch and the consumer 1 s; only the
        # loader's share is data time.
        now = [0.0]
        monkeypatch.setattr(seg, "time", type("Clock", (), {
            "perf_counter": staticmethod(lambda: now[0])}))

        class SlowLoader:
            def __iter__(self):
                for _ in range(3):
                    now[0] += 0.05
                    yield {"pixel_values": torch.zeros(2, 3, 4, 4)}
                now[0] += 0.05  # finding the end is waiting on the loader too

        counted = seg.CountingLoader(SlowLoader(), batch_size=2)
        for _ in counted:
            now[0] += 1.0
        assert counted.images == 6
        assert counted.data_seconds == pytest.approx(0.20)


# ---------------------------------------------------------------------------
# --by-size: per-object-size foreground recall
# ---------------------------------------------------------------------------

# One 512x512 image, three disjoint instances, one per bucket, all inside the
# 192x192 centre crop (original x and y in [64, 448)). The bucket is the instance's
# pixel count in the *original* image -- 24^2 = 576 small, 64^2 = 4096 medium,
# 128^2 = 16384 large -- while SIZED_TARGET_PIXELS below is those same instances'
# footprint on the 192x192 target grid, after the resize: 12^2 = 144, 32^2 = 1024,
# 64^2 = 4096. Only the large one straddles the crop's vertical middle, so a model
# that calls the left half foreground gets exactly half of it and all of the two
# smaller ones -- or none of the medium one, which sits wholly right of the middle.
SIZED = [(90, 90, 24), (320, 140, 64), (256, 320, 128)]  # cx, cy, side
SIZED_TARGET_PIXELS = {"small": 144, "medium": 1024, "large": 4096}
SIZED_RECALL = {"small": 1.0, "medium": 0.0, "large": 0.5}


def _poly_box(x1, y1, x2, y2):
    return [x1, y1, x2, y1, x2, y2, x1, y2]


def _box_ann(x1, y1, x2, y2, index, image_id=1):
    return {
        "id": index, "image_id": image_id, "category_id": 1, "iscrowd": 0,
        "segmentation": [_poly_box(x1, y1, x2, y2)],
        "area": float((x2 - x1) * (y2 - y1)),
        "bbox": [x1, y1, x2 - x1, y2 - y1],
    }


def _square_anns(boxes):
    """One square annotation per ``(cx, cy, side)``, in the order given."""
    return [_box_ann(cx - s // 2, cy - s // 2, cx + s // 2, cy + s // 2, i + 1)
            for i, (cx, cy, s) in enumerate(boxes)]


def _paint_boxes(path, boxes):
    a = np.full((SIZE, SIZE, 3), DARK, dtype=np.uint8)
    for x1, y1, x2, y2 in boxes:
        a[y1:y2, x1:x2] = BRIGHT
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(a).save(path)


@pytest.fixture
def sized_root(tmp_path):
    """Two 512x512 images whose three instances cover all three buckets."""
    root = tmp_path / "sized"
    images, anns = [], []
    for image_id in (1, 2):
        name = f"{image_id:012d}.jpg"
        _paint_boxes(root / seg.COCO_IMAGES / name,
                     [(cx - s // 2, cy - s // 2, cx + s // 2, cy + s // 2)
                      for cx, cy, s in SIZED])
        images.append({"id": image_id, "file_name": name, "height": SIZE, "width": SIZE})
        for cx, cy, s in SIZED:
            anns.append(_box_ann(cx - s // 2, cy - s // 2, cx + s // 2, cy + s // 2,
                                 len(anns) + 1, image_id))
    (root / "annotations").mkdir(parents=True)
    with open(root / seg.COCO_ANNOTATIONS, "w") as f:
        json.dump({"images": images, "annotations": anns, "categories": [{"id": 1}]}, f)
    return root


class LeftHalf(torch.nn.Module):
    """Calls foreground left of the target's vertical middle, whatever the input."""

    def forward(self, x):
        w = x.shape[-1]
        left = torch.zeros(x.shape[0], 1, x.shape[-2], w, dtype=x.dtype, device=x.device)
        left[..., :w // 2] = 1.0
        fg = left * 2 - 1
        return torch.cat([-10 * fg, 10 * fg], dim=1)


class CountingModel(torch.nn.Module):
    """``LeftHalf`` that counts its forward passes."""

    def __init__(self):
        super().__init__()
        self.inner = LeftHalf()
        self.calls = 0

    def forward(self, x):
        self.calls += 1
        return self.inner(x)


class TestSizeBuckets:
    def test_the_three_buckets_and_their_pixel_counts(self):
        labels, instances = seg.bucket_map_for_row(
            _square_anns(SIZED), SIZE, SIZE, 192)
        assert instances == {"small": 1, "medium": 1, "large": 1}
        assert labels.shape == (192, 192)
        assert labels.dtype == np.uint8
        for name, expected in SIZED_TARGET_PIXELS.items():
            assert int((labels == seg.BUCKET_ID[name]).sum()) == expected, name

    def test_an_overlapping_pixel_belongs_to_the_smallest_instance(self):
        # A 24^2 instance inside a 128^2 one, both centred on the same point.
        labels, instances = seg.bucket_map_for_row(
            _square_anns([(256, 256, 24), (256, 256, 128)]), SIZE, SIZE, 192)
        assert instances == {"small": 1, "medium": 0, "large": 1}
        assert int((labels == seg.BUCKET_ID["small"]).sum()) == 144
        # The large instance owns everything of its own 4096 but the small one's 144.
        assert int((labels == seg.BUCKET_ID["large"]).sum()) == 4096 - 144

    def test_the_bucket_is_the_area_before_the_resize(self):
        # 50x40 = 2000 original pixels is medium; at input 192 the resize halves
        # it to 500, which is small. It must stay medium.
        ann = _box_ann(230, 240, 280, 280, 1)
        labels, instances = seg.bucket_map_for_row([ann], SIZE, SIZE, 192)
        assert instances == {"small": 0, "medium": 1, "large": 0}
        assert int((labels == seg.BUCKET_ID["medium"]).sum()) == 500

    def test_the_map_does_not_depend_on_the_annotation_order(self):
        forward, _ = seg.bucket_map_for_row(_square_anns(SIZED), SIZE, SIZE, 192)
        reverse, _ = seg.bucket_map_for_row(_square_anns(SIZED[::-1]), SIZE, SIZE, 192)
        assert np.array_equal(forward, reverse)

    def test_the_labels_cover_exactly_the_transform_target(self, coco_root):
        """The map's nonzero pixels are the target's foreground, on the target's grid.

        This is the claim the whole breakdown rests on: the label map goes
        through the transform's own resize and crop, so a bucket's pixels are a
        partition of the pixels the IoU is computed over.
        """
        rows = seg.CocoRows(coco_root)
        for i in range(len(rows)):
            row = rows[i]
            target = seg.coco_val_transform_example(row, input_size=192)["segmentation_mask"]
            labels, _instances = rows.bucket_map(row["image_id"], 192)
            assert labels.shape == tuple(target.shape)
            assert np.array_equal(labels > 0, target.numpy() > 0)

    def test_every_instance_lands_in_a_bucket(self, coco_root):
        rows = seg.CocoRows(coco_root)
        for i in range(len(rows)):
            row = rows[i]
            labels, instances = rows.bucket_map(row["image_id"], 192)
            # coco_root's instances are all 64x64, so all medium, and every
            # annotation is bucketed exactly once.
            assert sum(instances.values()) == len(row["annotations"]) == instances["medium"]
            assert int((labels > 0).sum()) > 0


class TestSizeRecall:
    def test_the_recall_is_reported_per_bucket(self, sized_root):
        r = _score(seg.CocoRows(sized_root), LeftHalf(), by_size=True)
        by = r["by_size"]
        for name, expected in SIZED_RECALL.items():
            assert by[name]["recall"] == pytest.approx(expected), name
            assert by[name]["gt_pixels"] == 2 * SIZED_TARGET_PIXELS[name], name
            assert by[name]["instances"] == 2, name
        assert by["images"] == 2
        assert by["rule"] == seg.SIZE_RULE

    def test_the_score_is_untouched_by_the_breakdown(self, sized_root):
        rows = seg.CocoRows(sized_root)
        plain = _score(rows, LeftHalf())
        sized = _score(rows, LeftHalf(), by_size=True)
        # Bit for bit, not approximately: the callback only reads the tensors.
        for key in ("iou", "dice", "loss", "n_images", "n_failed"):
            assert plain[key] == sized[key], key
        assert "by_size" not in plain
        assert "by_size" in sized

    def test_the_model_is_not_run_a_second_time(self, sized_root):
        rows = seg.CocoRows(sized_root)
        plain, sized = CountingModel(), CountingModel()
        _score(rows, plain)
        _score(rows, sized, by_size=True)
        # Two rows, batch size 2: one forward either way.
        assert plain.calls == sized.calls == 1

    def test_a_bucket_with_no_ground_truth_is_null_not_zero(self, coco_root):
        # coco_root's instances are all 64x64 -- medium -- so small and large
        # have no ground-truth pixels at all.
        r = _score(seg.CocoRows(coco_root), PixelOracle(), by_size=True)
        by = r["by_size"]
        assert by["small"]["recall"] is None and by["small"]["gt_pixels"] == 0
        assert by["large"]["recall"] is None and by["large"]["gt_pixels"] == 0
        assert by["medium"]["recall"] > 0.9 and by["medium"]["gt_pixels"] > 0
        assert "small:n/a(0px,0)" in seg.format_by_size(by)

    def test_instances_that_do_not_add_up_fail_loudly(self, sized_root, monkeypatch):
        rows = seg.CocoRows(sized_root)
        original = rows.bucket_map

        def missing_a_pixel(image_id, input_size):
            labels, instances = original(image_id, input_size)
            labels = labels.copy()
            ys, xs = np.nonzero(labels)
            labels[ys[0], xs[0]] = 0
            return labels, instances

        monkeypatch.setattr(rows, "bucket_map", missing_a_pixel)
        with pytest.raises(seg.EvalError, match=r"image 1: the instances"):
            _score(rows, LeftHalf(), by_size=True)


class TestBySizeCli:
    def test_the_result_line_and_the_json_carry_the_buckets_and_the_rule(
            self, sized_root, tmp_path, monkeypatch, capsys):
        monkeypatch.setattr(seg, "load_checkpoint", lambda path, device, **_: LeftHalf())
        ckpt = tmp_path / "best.pth"
        ckpt.write_bytes(b"")
        code = seg.main(["--dataset", "coco", "--data-root", str(sized_root), "--device",
                         "cpu", "--by-size", str(ckpt)])
        [line] = capsys.readouterr().out.strip().splitlines()
        assert code == 0
        assert "by_size=small:1.000(288px,2),medium:0.000(2048px,2),large:0.500(8192px,2)" in line
        assert f"rule={seg.SIZE_RULE}" in line

    def test_the_json_output_carries_the_breakdown(self, sized_root, tmp_path,
                                                   monkeypatch, capsys):
        monkeypatch.setattr(seg, "load_checkpoint", lambda path, device, **_: LeftHalf())
        ckpt = tmp_path / "best.pth"
        ckpt.write_bytes(b"")
        code = seg.main(["--dataset", "coco", "--data-root", str(sized_root), "--device",
                         "cpu", "--json", "--by-size", str(ckpt)])
        r = json.loads(capsys.readouterr().out)
        assert code == 0
        assert r["by_size"]["rule"] == seg.SIZE_RULE
        assert r["by_size"]["small"] == {"recall": 1.0, "gt_pixels": 288, "instances": 2}
        assert r["by_size"]["images"] == 2

    def test_a_breakdown_that_does_not_add_up_prints_no_score(
            self, sized_root, tmp_path, monkeypatch, capsys):
        monkeypatch.setattr(seg, "load_checkpoint", lambda path, device, **_: LeftHalf())

        def missing_a_pixel(self, image_id, input_size):
            labels, instances = seg.bucket_map_for_row(
                self.anns.get(image_id, []), SIZE, SIZE, input_size)
            labels = labels.copy()
            ys, xs = np.nonzero(labels)
            labels[ys[0], xs[0]] = 0
            return labels, instances

        monkeypatch.setattr(seg.CocoRows, "bucket_map", missing_a_pixel)
        ckpt = tmp_path / "best.pth"
        ckpt.write_bytes(b"")
        code = seg.main(["--dataset", "coco", "--data-root", str(sized_root), "--device",
                         "cpu", "--by-size", str(ckpt)])
        out = capsys.readouterr()
        assert code == 1
        assert out.out == ""
        assert "image 1" in out.err and "do not add up" in out.err

    def test_by_size_is_refused_for_cocoa_and_for_workers(self, coco_root, tmp_path,
                                                          monkeypatch, capsys):
        def read_data(*args, **kwargs):
            raise AssertionError("the refusal must come before any data is read")

        monkeypatch.setattr(seg, "build_rows", read_data)
        ckpt = tmp_path / "best.pth"
        ckpt.write_bytes(b"")
        code = seg.main(["--dataset", "cocoa", "--data-root", str(tmp_path), "--by-size",
                         str(ckpt)])
        err = capsys.readouterr().err
        assert code == 2 and "no such area" in err and "--dataset coco" in err

        code = seg.main(["--dataset", "coco", "--data-root", str(coco_root),
                         "--num-workers", "2", "--by-size", str(ckpt)])
        err = capsys.readouterr().err
        assert code == 2 and "--num-workers 0" in err

    def test_a_plain_run_is_refused_by_neither(self, coco_root, tmp_path, monkeypatch,
                                               capsys):
        # The refusals are the flag's, not the CLI's: without --by-size the same
        # cocoa/worker invocations still reach the data.
        monkeypatch.setattr(seg, "load_checkpoint", lambda path, device, **_: PixelOracle())
        ckpt = tmp_path / "best.pth"
        ckpt.write_bytes(b"")
        code = seg.main(["--dataset", "coco", "--data-root", str(coco_root),
                         "--num-workers", "2", "--device", "cpu", str(ckpt)])
        assert code == 0
