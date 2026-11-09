"""The COCOA fine-tune starts from a v2 COCO checkpoint and stays v2.

After the v2 COCO retrain, the next run is ``--finetune --model-base
coco-models``: ``load_hcm_model`` with task ``cocoa_seg`` from the coco-models
bucket. Here a seeded v2 COCO model goes up through the trainer's own
``save_epoch_checkpoint``, the fine-tune's load takes it from that fake bucket,
and the fine-tune's own checkpoint goes up to finetuned-models and is resumed.

COCO and COCOA heads are both 2-class (the trainer builds both with
``num_segmentation_classes=2``), so the "swap" keeps the head: every tensor of
the COCO checkpoint, head and rrd fuse convs included, is loaded.
"""
import pickle

import pytest
import torch

from ecs_hcm_model.core import model_utils
from ecs_hcm_model.core import trainer as trainer_module
from ecs_hcm_model.core.hcm_nn import DECODER_V1, DECODER_V2, DecoderArchMismatchError, hcm_nn
from tests.test_decoder_arch import _offline
from tests.test_resume_hardening import FakeMinio, _store

SIZE = 64


class Buckets:
    """One FakeMinio per bucket, behind the store both the trainer and the loader build."""

    def __init__(self, monkeypatch):
        self.clients = {}
        factory = lambda bucket_name: _store(bucket_name, self.clients.setdefault(bucket_name, FakeMinio()))
        monkeypatch.setattr(trainer_module, "ModelUploaderAndDownloader", factory)
        monkeypatch.setattr(model_utils, "ModelUploaderAndDownloader", factory)

    def objects(self, bucket):
        return self.clients[bucket].objects


def _save(model, task, epoch=3):
    trainer_module.save_epoch_checkpoint(model=model, task=task, epoch=epoch,
                                         val_metrics={"val_iou": 0.71}, avg_loss=0.4,
                                         val_loss=0.5, is_best=True, input_size=SIZE)


def _finetune_load(model_base, caller_arch=DECODER_V2):
    # The arguments train_hcm_imagenet1k passes for --finetune (no --init-from).
    return _offline(lambda: model_utils.load_hcm_model(
        new_model=False, finetune=True, model_base=model_base, device=None,
        num_segmentation_classes=2, out_stride=16, task="cocoa_seg", input_size=SIZE,
        decoder_arch=caller_arch))


@pytest.fixture
def coco_v2(monkeypatch):
    buckets = Buckets(monkeypatch)
    torch.manual_seed(11)
    model = _offline(lambda: hcm_nn(enable_finetuning=True, num_segmentation_classes=2,
                                    out_stride=16, decoder_arch=DECODER_V2))
    _save(model, "coco_seg")
    return buckets, {k: v.clone() for k, v in model.state_dict().items()}


def _assert_same_tensors(model, source):
    state = model.state_dict()
    assert sorted(state) == sorted(source)
    assert [k for k in source if not torch.equal(state[k], source[k])] == []


@pytest.mark.parametrize("caller_arch", [DECODER_V2, DECODER_V1])
def test_the_finetune_start_is_v2_with_every_coco_tensor_copied(coco_v2, caller_arch):
    buckets, source = coco_v2
    # The caller's decoder_arch only shapes a new model; the checkpoint decides,
    # so even a v1 default cannot turn the start into v1.
    model = _finetune_load("coco-models", caller_arch)
    assert model.decoder_arch == DECODER_V2
    assert any(".fuse." in k for k in source)  # the v2-only tensors are in the comparison
    _assert_same_tensors(model, source)  # head and rrd_list.*.fuse included: nothing re-initialised
    assert model.segmentation_head.out_channels == 2 and model.enable_finetuning
    assert model.resumed_from.startswith("coco-models/")


def _record(objects, arch):
    """Rewrite the recorded decoder_arch in model_info.pkl and metadata.json."""
    import json
    for key in list(objects):
        if key.endswith("/model_info.pkl"):
            info = pickle.loads(objects[key])
            info["model_config"]["decoder_arch"] = arch
            objects[key] = pickle.dumps(info)
        elif key.endswith("/metadata.json"):
            meta = json.loads(objects[key])
            meta.setdefault("custom_metadata", {})["decoder_arch"] = arch
            objects[key] = json.dumps(meta).encode()


def test_a_coco_checkpoint_whose_record_contradicts_its_weights_is_refused(coco_v2):
    buckets, _ = coco_v2
    _record(buckets.objects("coco-models"), DECODER_V1)  # v1 recorded, v2 weights
    with pytest.raises(DecoderArchMismatchError, match="recorded decoder_arch='v1'"):
        _finetune_load("coco-models")


def test_a_resume_of_the_finetune_stays_v2(coco_v2):
    buckets, source = coco_v2
    start = _finetune_load("coco-models")
    with torch.no_grad():  # one "epoch" of change, so the resume is not just the COCO weights
        start.segmentation_head.bias.add_(1.0)
    tuned = {k: v.clone() for k, v in start.state_dict().items()}
    _save(start, "cocoa_seg", epoch=0)
    assert set(buckets.clients) == {"coco-models", "finetuned-models"}
    # The demo, eval and promotion tools read the version from metadata.json,
    # not model_info.pkl, so the fine-tune must state v2 there too.
    import json
    from jetson.model_shape import resolve_decoder_arch
    sidecars = [k for k in buckets.objects("finetuned-models") if k.endswith("/metadata.json")]
    assert sidecars
    for key in sidecars:
        arch, src = resolve_decoder_arch(json.loads(buckets.objects("finetuned-models")[key]), tuned)
        assert arch == DECODER_V2 and src != "weights", (key, arch, src)
    resumed =_finetune_load("finetuned-models", caller_arch=DECODER_V1)
    assert resumed.decoder_arch == DECODER_V2
    _assert_same_tensors(resumed, tuned)
    assert resumed.resumed_from.startswith("finetuned-models/")
