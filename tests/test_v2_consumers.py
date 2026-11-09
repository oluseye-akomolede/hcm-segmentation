"""A checkpoint as the trainer uploads it loads in the demos and the eval tool.

The retrain's v2 checkpoints go from the coco-models bucket into the Jetson's
artifact directory as ``model.pth`` + ``metadata.json``. Here a model goes up
through the trainer's own ``save_epoch_checkpoint`` into a fake bucket, those
two objects are copied out, and the demos' shared loader and the eval tool's
loader each rebuild it. Their outputs must equal the in-memory model's.
"""
import pytest
import torch

from ecs_hcm_model.core import trainer as trainer_module
from ecs_hcm_model.core.hcm_nn import DECODER_V1, DECODER_V2, hcm_nn
from ecs_hcm_model.eval.seg import find_metadata, load_checkpoint
from jetson.demo_model import load_demo_model
from jetson.input_size import load_metadata
from tests.test_decoder_arch import _offline
from tests.test_resume_hardening import FakeMinio, _store

SIZE = 64


def _upload(monkeypatch, model):
    """Save ``model`` as an epoch checkpoint; return the bucket's objects."""
    client = FakeMinio()
    monkeypatch.setattr(trainer_module, "ModelUploaderAndDownloader",
                        lambda bucket_name: _store(bucket_name, client))
    trainer_module.save_epoch_checkpoint(model=model, task="coco_seg", epoch=3,
                                         val_metrics={"val_iou": 0.71}, avg_loss=0.4,
                                         val_loss=0.5, is_best=True, input_size=SIZE)
    return client.objects


def _artifact_dir(objects, art):
    """Copy model.pth and metadata.json out of the bucket, as onto the Jetson."""
    art.mkdir()
    for name in ("model.pth", "metadata.json"):
        (key,) = [k for k in objects if k.endswith("/" + name)]
        (art / name).write_bytes(objects[key])
    return art


@pytest.fixture(params=[DECODER_V2, DECODER_V1])
def uploaded(request, monkeypatch, tmp_path):
    arch = request.param
    torch.manual_seed(7)
    model = _offline(lambda: hcm_nn(enable_finetuning=True, num_segmentation_classes=2,
                                    out_stride=16, decoder_arch=arch)).eval()
    art = _artifact_dir(_upload(monkeypatch, model), tmp_path / "models")
    x = torch.randn(1, 3, SIZE, SIZE, generator=torch.Generator().manual_seed(0))
    with torch.no_grad():
        expected = model(x)
    return arch, art, x, expected


def test_the_sidecar_records_the_version(uploaded):
    arch, art, _, _ = uploaded
    assert load_metadata(art / "metadata.json")["custom_metadata"]["decoder_arch"] == arch


def test_the_demo_loader_reproduces_the_uploaded_model(uploaded, capsys):
    arch, art, x, expected = uploaded
    meta = load_metadata(art / "metadata.json")
    model, num_classes = _offline(lambda: load_demo_model(
        art / "model.pth", meta, cli_stride=None, env={}, device="cpu",
        input_size=SIZE, input_size_source="test"))
    assert f"decoder: {arch} (from metadata.custom_metadata.decoder_arch)" in capsys.readouterr().out
    assert (model.decoder_arch, num_classes, model.training) == (arch, 2, False)
    with torch.no_grad():
        assert torch.equal(model(x), expected)


@pytest.mark.parametrize("edit", ["missing", "extra"])
def test_the_demo_loader_refuses_weights_that_do_not_match_exactly(uploaded, edit):
    _, art, _, _ = uploaded
    state = torch.load(art / "model.pth", map_location="cpu")
    if edit == "missing":
        del state[next(k for k in state if k.startswith("backbone."))]
    else:
        state["extra.weight"] = torch.zeros(1)
    torch.save(state, art / "model.pth")
    meta = load_metadata(art / "metadata.json")
    with pytest.raises(RuntimeError, match="Missing key" if edit == "missing" else "Unexpected key"):
        _offline(lambda: load_demo_model(art / "model.pth", meta, cli_stride=None, env={}, device="cpu",
                                         input_size=SIZE, input_size_source="test"))


def test_the_eval_loader_reproduces_the_uploaded_model(uploaded):
    arch, art, x, expected = uploaded
    meta, _ = find_metadata(art / "model.pth")
    logged = []
    model = _offline(lambda: load_checkpoint(art / "model.pth", "cpu", meta=meta, log=logged.append))
    assert any(f"decoder_arch {arch} from metadata.custom_metadata.decoder_arch" in m for m in logged)
    assert model.decoder_arch == arch
    with torch.no_grad():
        assert torch.equal(model.eval()(x), expected)
