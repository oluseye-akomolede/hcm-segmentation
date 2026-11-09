"""Who wins in load_hcm_model: the caller for the mode, the checkpoint for the shape.

Since the loader rebuilds a model from its class info, enable_finetuning came
from the checkpoint and the caller could only force it on: a classification run
with --model-base coco-models stayed in segmentation mode and failed on the
loss shape. These tests drive load_hcm_model with the real rebuild
(build_model_from_info) behind a stand-in for the MinIO downloader.
"""
import pytest
import torch

from ecs_hcm_model.core import model_utils
from ecs_hcm_model.core.hcm_nn import hcm_nn
from ecs_hcm_model.core.model_upload_download import build_model_from_info


@pytest.fixture(scope="module")
def seg_sd():
    torch.manual_seed(0)
    return hcm_nn(num_segmentation_classes=2, out_stride=16).state_dict()


@pytest.fixture(scope="module")
def cls_sd():
    torch.manual_seed(1)
    return hcm_nn(num_segmentation_classes=None, out_stride=16).state_dict()


def _info(num_seg, out_stride, enable_finetuning):
    return {"model_class": hcm_nn,
            "model_config": {"num_segmentation_classes": num_seg, "out_stride": out_stride,
                             "enable_finetuning": enable_finetuning}}


LEGACY = dict(num_seg=None, out_stride=None)


def _serve(monkeypatch, checkpoint):
    """Make the downloader return ``checkpoint`` (a built model or a state dict)."""

    class FakeDownloader:
        def __init__(self, bucket_name="models"):
            self.bucket_name = bucket_name

        def search_and_download_model_by_architecture(self, architecture):
            return checkpoint

    monkeypatch.setattr(model_utils, "ModelUploaderAndDownloader", FakeDownloader)


def _load(monkeypatch, checkpoint, **kwargs):
    _serve(monkeypatch, checkpoint)
    kwargs.setdefault("model_base", "coco-models")
    return model_utils.load_hcm_model(**kwargs)


def _classifier(sd):
    return {k: v for k, v in sd.items() if k.startswith("roi_classifier.")}


class TestTheCallerDecidesTheMode:
    @pytest.mark.parametrize("recorded", [True, False, None])
    @pytest.mark.parametrize("finetune", [True, False])
    def test_new_checkpoint(self, monkeypatch, seg_sd, recorded, finetune):
        ckpt = build_model_from_info(_info(2, 16, recorded), seg_sd)
        model = _load(monkeypatch, ckpt, finetune=finetune)
        assert model.enable_finetuning is finetune

    @pytest.mark.parametrize("recorded", [True, False])
    @pytest.mark.parametrize("finetune", [True, False])
    def test_legacy_none_config(self, monkeypatch, seg_sd, recorded, finetune):
        ckpt = build_model_from_info(_info(**LEGACY, enable_finetuning=recorded), seg_sd)
        model = _load(monkeypatch, ckpt, finetune=finetune)
        assert model.enable_finetuning is finetune
        assert model.segmentation_head.out_channels == 2

    @pytest.mark.parametrize("finetune", [True, False])
    def test_bare_state_dict(self, monkeypatch, seg_sd, finetune):
        model = _load(monkeypatch, dict(seg_sd), finetune=finetune)
        assert model.enable_finetuning is finetune

    def test_classification_from_a_seg_checkpoint_returns_logits(self, monkeypatch, seg_sd):
        # The reported failure: a fine-tuned seg checkpoint, a classification run.
        ckpt = build_model_from_info(_info(2, 16, True), seg_sd)
        model = _load(monkeypatch, ckpt, finetune=False).eval()
        with torch.no_grad():
            out = model(torch.zeros(1, 3, 64, 64))
        assert out.shape == (1, 1000)
        for k, v in _classifier(seg_sd).items():
            assert torch.equal(model.state_dict()[k], v), "the checkpoint's classifier is used"

    def test_a_fine_tune_from_a_classification_checkpoint(self, monkeypatch, cls_sd):
        ckpt = build_model_from_info(_info(None, 16, False), cls_sd)
        model = _load(monkeypatch, ckpt, finetune=True, num_segmentation_classes=2)
        assert model.enable_finetuning is True
        assert model.segmentation_head.out_channels == 2


class TestClassificationNeedsAClassifier:
    @staticmethod
    def _headless(sd):
        return {k: v for k, v in sd.items() if not k.startswith("roi_classifier.")}

    def test_module_path_refuses(self, monkeypatch, seg_sd):
        ckpt = build_model_from_info(_info(2, 16, True), self._headless(seg_sd))
        with pytest.raises(ValueError, match=r"classification.*finetune=False.*no classification head"):
            _load(monkeypatch, ckpt, finetune=False)

    def test_state_dict_path_refuses(self, monkeypatch, seg_sd):
        with pytest.raises(ValueError, match=r"coco-models.*roi_classifier"):
            _load(monkeypatch, self._headless(seg_sd), finetune=False)

    def test_a_fine_tune_does_not_need_one(self, monkeypatch, seg_sd):
        ckpt = build_model_from_info(_info(2, 16, True), self._headless(seg_sd))
        assert _load(monkeypatch, ckpt, finetune=True).enable_finetuning is True


class TestTheCheckpointDecidesTheArchitecture:
    def test_module_path_keeps_the_checkpoints_class_count(self, monkeypatch, seg_sd):
        ckpt = build_model_from_info(_info(2, 16, True), seg_sd)
        with pytest.warns(UserWarning, match=r"num_segmentation_classes=5 was requested.*has num_segmentation_classes=2"):
            model = _load(monkeypatch, ckpt, finetune=True, num_segmentation_classes=5)
        assert model.segmentation_head.out_channels == 2
        assert torch.equal(model.segmentation_head.weight, seg_sd["segmentation_head.weight"])

    def test_legacy_module_path_keeps_the_inferred_class_count(self, monkeypatch, seg_sd):
        ckpt = build_model_from_info(_info(**LEGACY, enable_finetuning=True), seg_sd)
        with pytest.warns(UserWarning, match=r"=5 was requested.*=2"):
            model = _load(monkeypatch, ckpt, finetune=True, num_segmentation_classes=5)
        assert model.segmentation_head.out_channels == 2

    def test_state_dict_path_keeps_the_checkpoints_class_count(self, monkeypatch, seg_sd):
        # Before, the caller's 5 won and load_state_dict died on a size mismatch.
        with pytest.warns(UserWarning, match=r"=5 was requested.*=2"):
            model = _load(monkeypatch, dict(seg_sd), finetune=True, num_segmentation_classes=5)
        assert model.segmentation_head.out_channels == 2
        assert torch.equal(model.segmentation_head.weight, seg_sd["segmentation_head.weight"])

    def test_module_path_keeps_the_checkpoints_out_stride(self, monkeypatch, seg_sd):
        ckpt = build_model_from_info(_info(2, 16, True), seg_sd)
        with pytest.warns(UserWarning, match=r"out_stride=8 was requested.*has out_stride=16"):
            model = _load(monkeypatch, ckpt, finetune=True, out_stride=8)
        assert model.out_stride == 16

    def test_an_agreeing_caller_is_not_warned(self, monkeypatch, seg_sd, recwarn):
        ckpt = build_model_from_info(_info(2, 16, True), seg_sd)
        _load(monkeypatch, ckpt, finetune=True, num_segmentation_classes=2, out_stride=16)
        assert not [w for w in recwarn if "load_hcm_model" in str(w.message)]


class TestAnUnrecordedStrideIsTheCallers:
    """8643c428 / L1': a legacy class-info checkpoint recorded out_stride as None.

    The rebuild stands in 16, but that 16 is not the checkpoint's: the caller's
    stride applies, with no warning, and 16 only when the caller gives none.
    """

    def test_the_callers_stride_wins_without_a_warning(self, monkeypatch, seg_sd, recwarn):
        ckpt = build_model_from_info(_info(None, None, True), seg_sd)
        model = _load(monkeypatch, ckpt, finetune=True, out_stride=8)
        assert model.out_stride == 8
        assert not [w for w in recwarn if "load_hcm_model" in str(w.message)]
        # Same weights: the stride carries none.
        assert torch.equal(model.segmentation_head.weight, seg_sd["segmentation_head.weight"])
        assert model.segmentation_head.out_channels == 2

    def test_no_caller_stride_falls_back_to_16(self, monkeypatch, seg_sd, recwarn):
        ckpt = build_model_from_info(_info(None, None, True), seg_sd)
        model = _load(monkeypatch, ckpt, finetune=True, out_stride=None)
        assert model.out_stride == 16
        assert not [w for w in recwarn if "load_hcm_model" in str(w.message)]

    def test_a_recorded_stride_still_wins_and_warns(self, monkeypatch, seg_sd):
        ckpt = build_model_from_info(_info(2, 16, True), seg_sd)
        with pytest.warns(UserWarning, match=r"out_stride=8 was requested.*has out_stride=16"):
            model = _load(monkeypatch, ckpt, finetune=True, out_stride=8)
        assert model.out_stride == 16
