"""Resume hardening: the resume path fails loudly instead of quietly continuing
from the wrong place.

From the reviewer's resume audit (TICKET) and its
wrong-checkpoint addendum (TICKET). The store is driven
through an in-memory fake MinIO, so the real listing, selection, upload and
download code runs with no endpoint.
"""

import json

from ecs_hcm_model.core.model_upload_download import ModelUploaderAndDownloader


class TestOneSearchPath:
    def test_the_name_based_search_is_gone(self):
        assert hasattr(ModelUploaderAndDownloader, "search_and_download_model_by_architecture")
        assert not hasattr(ModelUploaderAndDownloader, "search_and_download_model")


# --- a fake MinIO and a store bound to it --------------------------------------

from pathlib import Path
import io

import pytest
import torch


class FakeObject:
    def __init__(self, name):
        self.object_name = name


class FakeMinio:
    """The calls the store makes, backed by a dict, in upload order.

    ``fail_put`` makes one file name's upload raise; ``fail_get`` makes reading
    an object raise, as a network or permissions error would.
    """

    def __init__(self, fail_put=None, fail_get=()):
        self.objects = {}
        self.puts = []
        self.fail_put = fail_put
        self.fail_get = set(fail_get)

    def fput_object(self, bucket, name, path):
        if self.fail_put and name.endswith("/" + self.fail_put):
            raise OSError(f"simulated upload failure for {name}")
        self.puts.append(name)
        self.objects[name] = Path(path).read_bytes()

    def fget_object(self, bucket, name, path):
        if name in self.fail_get:
            raise OSError(f"simulated read failure for {name}")
        if name not in self.objects:
            raise FileNotFoundError(name)
        Path(path).write_bytes(self.objects[name])

    def list_objects(self, bucket, prefix=None, recursive=False):
        return [FakeObject(n) for n in sorted(self.objects) if n.startswith(prefix or "")]


def _store(bucket="coco-models", client=None):
    # Bypass __init__: it needs MinIO credentials and a live endpoint.
    store = object.__new__(ModelUploaderAndDownloader)
    store.bucket_name = bucket
    store.client = client or FakeMinio()
    return store


class Tiny(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = torch.nn.Linear(2, 2)


class TestMetadataIsWrittenLast:
    def test_upload_order(self):
        store = _store()
        store.upload_model(Tiny(), "hcm_coco_seg_best", architecture="hcm", metadata={"is_best": True})
        files = [n.rsplit("/", 1)[1] for n in store.client.puts]
        assert files == ["model.pth", "model_info.pkl", "metadata.json"]

    @pytest.mark.parametrize("failing", ["model.pth", "model_info.pkl"])
    def test_a_failed_save_leaves_no_candidate(self, failing):
        """A save that dies before metadata.json raises and is never resumable."""
        store = _store(client=FakeMinio(fail_put=failing))
        with pytest.raises(OSError):
            store.upload_model(Tiny(), "hcm_coco_seg_best", architecture="hcm", metadata={"is_best": True})
        assert not any(n.endswith("metadata.json") for n in store.client.objects)
        # The candidate filter really does need metadata.json: the leftover
        # objects are not seen as a checkpoint at all.
        with pytest.raises(Exception, match="empty|no checkpoint|No model"):
            _search_or_raise(store)

    def test_the_candidate_filter_needs_metadata_json(self):
        """Weights and class info without metadata.json: an upload in flight."""
        store = _store()
        store.upload_model(Tiny(), "hcm_coco_seg_best", architecture="hcm", metadata={"is_best": True})
        for n in list(store.client.objects):
            if n.endswith("metadata.json"):
                del store.client.objects[n]
        with pytest.raises(Exception, match="empty|no checkpoint|No model"):
            _search_or_raise(store)


def _search_or_raise(store):
    """The search, made to raise however it reports "nothing to resume"."""
    result = store.search_and_download_model_by_architecture("hcm")
    if result is None:
        raise LookupError("No model")
    return result


# --- M3 / #6: unreadable metadata is counted, and never silently skipped past --

def _put(store, prefix, custom=None, *, raw=None, timestamp="2026-10-01T12:00:00", weights=None):
    """Write a checkpoint straight into the fake bucket.

    ``raw`` replaces metadata.json's bytes (to make it unreadable); otherwise it
    is a well-formed document with ``custom`` as its custom_metadata.
    """
    buf = io.BytesIO()
    torch.save((weights or Tiny()).state_dict(), buf)
    store.client.objects[f"{prefix}/model.pth"] = buf.getvalue()
    if raw is None:
        raw = json.dumps({
            "name": prefix.split("/")[0],
            "architecture": "hcm",
            "upload_timestamp": timestamp,
            "custom_metadata": {"is_best": True, **(custom or {})},
        }).encode()
    store.client.objects[f"{prefix}/metadata.json"] = raw


OLD = "hcm_coco_seg_best/20261001_100000"
NEW = "hcm_coco_seg_best/20261001_120000"


class TestUnreadableMetadata:
    def test_a_newer_unreadable_checkpoint_refuses_the_resume(self, capsys):
        store = _store()
        _put(store, OLD, timestamp="2026-10-01T10:00:00")
        _put(store, NEW, raw=b"{ not json")
        assert store.search_and_download_model_by_architecture("hcm") is None
        out = capsys.readouterr().out
        assert f"Refusing to resume from {OLD}" in out
        assert NEW in out

    def test_a_newer_checkpoint_that_cannot_be_fetched_also_refuses(self, capsys):
        store = _store()
        _put(store, OLD, timestamp="2026-10-01T10:00:00")
        _put(store, NEW, timestamp="2026-10-01T12:00:00")
        store.client.fail_get.add(f"{NEW}/metadata.json")
        assert store.search_and_download_model_by_architecture("hcm") is None
        assert "Refusing to resume" in capsys.readouterr().out

    def test_an_older_unreadable_checkpoint_is_counted_not_fatal(self, capsys):
        store = _store()
        _put(store, OLD, raw=b"{ not json")
        _put(store, NEW, timestamp="2026-10-01T12:00:00")
        assert store.search_and_download_model_by_architecture("hcm") is not None
        out = capsys.readouterr().out
        assert f"Resuming from {NEW}" in out
        assert "1 with unreadable metadata" in out

    def test_only_unreadable_checkpoints_are_named(self, capsys):
        store = _store()
        _put(store, NEW, raw=b"{ not json")
        assert store.search_and_download_model_by_architecture("hcm") is None
        out = capsys.readouterr().out
        assert "1 checkpoint(s) have unreadable metadata.json" in out

    def test_a_read_but_unparseable_timestamp_refuses(self, capsys):
        store = _store()
        _put(store, OLD, timestamp="2026-10-01T10:00:00")
        _put(store, NEW, timestamp="yesterday-ish")
        assert store.search_and_download_model_by_architecture("hcm") is None
        assert "unparseable upload_timestamp 'yesterday-ish'" in capsys.readouterr().out


# --- L2: the loader says why there is nothing to resume, naming the bucket -----

from ecs_hcm_model.core import model_utils


def _serve_store(monkeypatch, store):
    monkeypatch.setattr(model_utils, "ModelUploaderAndDownloader",
                        lambda bucket_name: store)


class TestTheRealReason:
    def _load(self, monkeypatch, store):
        _serve_store(monkeypatch, store)
        return model_utils.load_hcm_model(finetune=True, model_base="coco-models")

    def test_an_empty_bucket(self, monkeypatch):
        with pytest.raises(ValueError, match=r"bucket 'coco-models': the bucket has no checkpoint"):
            self._load(monkeypatch, _store())

    def test_none_eligible(self, monkeypatch):
        store = _store()
        _put(store, NEW, {"is_smoke": True})
        with pytest.raises(ValueError, match=r"bucket 'coco-models'.*unsuitable.*smoke checkpoint"):
            self._load(monkeypatch, store)

    def test_a_download_error(self, monkeypatch):
        store = _store()
        _put(store, NEW)
        store.client.fail_get.add(f"{NEW}/model.pth")
        with pytest.raises(ValueError, match=rf"bucket 'coco-models': downloading {NEW} failed"):
            self._load(monkeypatch, store)

    def test_the_old_message_is_gone(self, monkeypatch):
        with pytest.raises(ValueError) as e:
            self._load(monkeypatch, _store())
        assert "No HCM model found in MLflow" not in str(e.value)


# --- M1: the resume bucket comes from the task and --model-base ----------------

from ecs_hcm_model.core.model_upload_download import resume_bucket_for

BUCKET = {"classification": "models", "coco_seg": "coco-models", "cocoa_seg": "finetuned-models"}
ALLOWED = {
    ("classification", "classification"),
    ("coco_seg", "coco_seg"),
    ("cocoa_seg", "cocoa_seg"),
    ("classification", "coco_seg"),   # ImageNet -> COCO, head added
    ("coco_seg", "cocoa_seg"),        # COCO -> COCOA, the refine
}


class TestTheResumeBucket:
    @pytest.mark.parametrize("source", sorted(BUCKET))
    @pytest.mark.parametrize("task", sorted(BUCKET))
    def test_the_transition_table(self, source, task):
        if (source, task) in ALLOWED:
            assert resume_bucket_for(task, BUCKET[source]) == BUCKET[source]
        else:
            with pytest.raises(ValueError, match=rf"--model-base '{BUCKET[source]}'.*holds {source}"):
                resume_bucket_for(task, BUCKET[source])

    def test_the_error_names_the_run_flag_and_the_fix(self):
        with pytest.raises(ValueError, match=r"cocoa_seg run \(--finetune\).*use --model-base coco-models or finetuned-models"):
            resume_bucket_for("cocoa_seg", "models")

    def test_an_unknown_bucket_raises(self):
        with pytest.raises(ValueError, match=r"--model-base 'bogus' is not a checkpoint bucket"):
            resume_bucket_for("coco_seg", "bogus")

    def test_the_loader_no_longer_falls_back_to_models(self, monkeypatch):
        """finetuned-models without --finetune used to load from "models"."""
        _serve_store(monkeypatch, _store("models"))
        with pytest.raises(ValueError, match=r"classification run .* cannot resume from --model-base 'finetuned-models'"):
            model_utils.load_hcm_model(finetune=False, model_base="finetuned-models", task="classification")

    def test_a_caller_without_a_task_still_gets_the_bucket_checked(self, monkeypatch):
        _serve_store(monkeypatch, _store("models"))
        with pytest.raises(ValueError, match=r"'bogus' is not a checkpoint bucket"):
            model_utils.load_hcm_model(model_base="bogus")

    def test_a_fresh_model_ignores_the_bucket(self):
        model = model_utils.load_hcm_model(new_model=True, model_base="bogus", task="cocoa_seg",
                                           num_segmentation_classes=2)
        assert model is not None


# --- M2: the resume logs and checks the checkpoint's task and input size -------

class TestTheResumeChecksItsCheckpoint:
    def _resume(self, monkeypatch, bucket, custom, *, task, input_size=352):
        store = _store(bucket)
        _put(store, f"{bucket}_best/20261001_120000", custom)
        _serve_store(monkeypatch, store)
        return model_utils.load_hcm_model(
            finetune=True, model_base=bucket, num_segmentation_classes=2,
            task=task, input_size=input_size,
        )

    def test_the_refine_logs_what_it_resumes(self, monkeypatch, capsys):
        self._resume(monkeypatch, "coco-models",
                     {"task": "coco_seg", "input_size": 352, "coco_mix_ratio": 0.0},
                     task="cocoa_seg")
        out = capsys.readouterr().out
        assert "task=coco_seg, input_size=352, coco_mix_ratio=0.0" in out
        assert "WARNING" not in out

    def test_a_checkpoint_task_outside_the_table_raises(self, monkeypatch):
        # A cocoa_seg checkpoint sitting in the classification bucket.
        with pytest.raises(ValueError, match=r"trained for cocoa_seg, which a coco_seg run \(--coco\) may not resume from"):
            self._resume(monkeypatch, "models", {"task": "cocoa_seg"}, task="coco_seg")

    def test_a_new_input_size_warns_but_loads(self, monkeypatch, capsys):
        model = self._resume(monkeypatch, "coco-models",
                             {"task": "coco_seg", "input_size": 192}, task="coco_seg", input_size=352)
        assert model is not None
        out = capsys.readouterr().out
        assert "WARNING: resuming a checkpoint trained at input_size=192 into a run at input_size=352" in out

    def test_a_legacy_checkpoint_is_unknown_and_unchecked(self, monkeypatch, capsys):
        model = self._resume(monkeypatch, "models", {}, task="coco_seg")
        assert model is not None
        out = capsys.readouterr().out
        assert "task=unknown, input_size=unknown, coco_mix_ratio=unknown" in out
        assert "WARNING: resuming" not in out


# --- Addendum #1: newest still wins, but a better passed-over checkpoint is loud --

class TestAPassedOverBetterCheckpointIsLoud:
    def _probe(self, monkeypatch):
        """The reviewer's probe: a long-trained _best, then a dead run's epoch 0."""
        store = _store("coco-models")
        _put(store, "hcm_coco_seg_best/20260930_090000",
             {"task": "coco_seg", "epoch": 14, "val_metrics": {"val_iou": 0.61}},
             timestamp="2026-09-30T09:00:00")
        _put(store, "hcm_coco_seg_epoch0/20261001_100000",
             {"task": "coco_seg", "epoch": 0, "val_metrics": {"val_iou": 0.22}},
             timestamp="2026-10-01T10:00:00")
        _serve_store(monkeypatch, store)
        return model_utils.load_hcm_model(finetune=True, model_base="coco-models",
                                          num_segmentation_classes=2, task="coco_seg")

    def test_the_selection_is_unchanged_and_the_warning_names_both(self, monkeypatch, capsys):
        model = self._probe(monkeypatch)
        out = capsys.readouterr().out
        assert "Resuming from hcm_coco_seg_epoch0/20261001_100000" in out
        assert ("WARNING: resuming from the newest checkpoint 'hcm_coco_seg_epoch0' "
                "(epoch 0, val_iou 0.22), though 'hcm_coco_seg_best' (epoch 14, "
                "val_iou 0.61)") in out
        assert model.resumed_from == "coco-models/hcm_coco_seg_epoch0/20261001_100000"

    def test_no_warning_when_the_newest_is_also_the_best(self, monkeypatch, capsys):
        store = _store("coco-models")
        _put(store, "hcm_coco_seg_best/20260930_090000",
             {"task": "coco_seg", "epoch": 14, "val_metrics": {"val_iou": 0.40}},
             timestamp="2026-09-30T09:00:00")
        _put(store, "hcm_coco_seg_best/20261001_100000",
             {"task": "coco_seg", "epoch": 3, "val_metrics": {"val_iou": 0.55}},
             timestamp="2026-10-01T10:00:00")
        _serve_store(monkeypatch, store)
        model_utils.load_hcm_model(finetune=True, model_base="coco-models",
                                   num_segmentation_classes=2, task="coco_seg")
        assert "recorded a better val_iou" not in capsys.readouterr().out


from ecs_hcm_model.core import trainer  # noqa: E402


class TestResumedFromIsRecorded:
    def _capture(self, monkeypatch):
        captured = {}
        class Uploader:
            def __init__(self, bucket_name): pass
            def upload_model(self, model, model_name, architecture, metadata):
                captured.update(metadata); return "uri"
        monkeypatch.setattr(trainer, "ModelUploaderAndDownloader", Uploader)
        return captured

    def _save(self, model):
        trainer.save_epoch_checkpoint(model=model, task="coco_seg", epoch=0,
                                      val_metrics={"val_iou": 0.3}, avg_loss=1.0, val_loss=1.0)

    def test_a_resumed_model_records_its_parent(self, monkeypatch):
        captured = self._capture(monkeypatch)
        model = Tiny(); model.resumed_from = "coco-models/hcm_coco_seg_best/20260930_090000"
        self._save(model)
        assert captured["resumed_from"] == "coco-models/hcm_coco_seg_best/20260930_090000"

    def test_a_fresh_model_records_null(self, monkeypatch):
        captured = self._capture(monkeypatch)
        self._save(Tiny())
        assert "resumed_from" in captured and captured["resumed_from"] is None

    def test_a_wrapped_model_records_its_parent(self, monkeypatch):
        captured = self._capture(monkeypatch)
        inner = Tiny(); inner.resumed_from = "models/hcm_classification_best/x"
        wrapper = torch.nn.Module(); wrapper.module = inner
        self._save(wrapper)
        assert captured["resumed_from"] == "models/hcm_classification_best/x"
