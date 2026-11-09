"""Batch-norm mode during an epoch, for runs with and without warmup.

The trainer calls model.train() at the top of every epoch, and the staged
unfreezing chain re-pins the backbone's batch norm to eval mode for the warmup
epochs. These tests read the batch-norm mode while an epoch actually trains,
rather than a loss value, so an epoch at or past WARMUP_EPOCHS is the case
under test.
"""

import pytest
import torch
from torch import nn
import ecs_hcm_model.core.trainer as trainer
from conftest import RecordingLogger


class _BNBackbone(nn.Module):
    """A backbone with one batch-norm layer and layer2/3/4 linears.

    The batch norm is 2-D on purpose. ``backbone_bn_eval`` pins BatchNorm2d and
    SyncBatchNorm only, so a 1-D stand-in would be pinned by nothing, and the
    probe would read a mode the trainer never sets.
    """

    # A 4x4 side with 8 channels, so the flattened features are 8 * 4 * 4 = 128.
    def __init__(self, channels=8, in_features=128):
        super().__init__()
        self.bn = nn.BatchNorm2d(channels)
        self.layer2 = nn.Linear(in_features, in_features)
        self.layer3 = nn.Linear(in_features, in_features)
        self.layer4 = nn.Linear(in_features, in_features)

    def forward(self, x):
        return self.layer4(self.layer3(self.layer2(self.bn(x).flatten(1))))


class _BNClassifier(nn.Module):
    """Minimal stand-in for hcm_nn, recording the batch-norm mode per epoch."""

    def __init__(self, in_features=128, num_classes=10):
        super().__init__()
        self.backbone = _BNBackbone(in_features=in_features)
        self.head = nn.Linear(in_features, num_classes)
        # Unused on this path, but hcm_nn always has it and the trainer reads it.
        self.segmentation_head = nn.Linear(in_features, 1)
        self.seen = []

    def forward(self, x):
        # Record only a training pass. Validation runs under model.eval(), so
        # recording it would show eval mode with or without the pin and the
        # assertion would prove nothing.
        if self.training:
            self.seen.append({
                "bn_training": self.backbone.bn.training,
                "layer3_trainable": self.backbone.layer3.weight.requires_grad,
                "layer2_trainable": self.backbone.layer2.weight.requires_grad,
            })
        return self.head(self.backbone(x))


@pytest.fixture(autouse=True)
def _no_early_stopping(monkeypatch):
    class EarlyStopping:
        """Never stops, so an 8-epoch run actually reaches epoch 8."""

        def __init__(self, *args, **kwargs):
            pass

        def step(self, current_loss):
            return False

    monkeypatch.setattr(trainer, "EarlyStopping", EarlyStopping)


def _curve(logger, key):
    return [value for name, value, _ in logger.scalars if name == key]


def _run(epochs, warmup, lr_decay_start=None, init_from=None):
    model = _BNClassifier()
    if init_from is not None:
        model.init_from = init_from
    logger = RecordingLogger()
    # One batch per epoch, so record i is epoch i.
    batches = [(torch.randn(4, 8, 4, 4), torch.randint(0, 10, (4,)))]
    trainer.train_model_for_epochs(
        model=model,
        epochs=epochs,
        train_loader=batches,
        val_loader=batches,
        device=torch.device("cpu"),
        logger=logger,
        test_mode=False,
        warmup=warmup,
        task="classification",
        checkpoint_every_epoch=False,
        lr_decay_start=lr_decay_start,
    )
    return model, logger


def test_batch_norm_is_pinned_in_every_epoch_after_warmup():
    # Before the fix the last three records are True: model.train() re-enables
    # the backbone's batch norm from epoch 5 on, and nothing pins it back.
    model, _ = _run(epochs=8, warmup=True)
    assert len(model.seen) == 8
    assert [r["bn_training"] for r in model.seen] == [False] * 8


def test_the_post_warmup_epoch_leaves_the_trainable_stages_alone():
    # The pin must change the batch-norm mode only, not which stages train.
    model, _ = _run(epochs=8, warmup=True)
    # Index 5 onwards are the post-warmup epochs. Their count is pinned rather
    # than trusted, because an empty slice would make the loop below vacuous.
    post_warmup_records = model.seen[5:]
    assert len(post_warmup_records) == 3
    for record in post_warmup_records:
        assert record["layer3_trainable"] is True
        assert record["layer2_trainable"] is False


def test_the_post_warmup_backbone_learning_rate_is_held():
    # The first decay epoch is exactly the base, not merely near it.
    _, logger = _run(epochs=8, warmup=True, lr_decay_start=5)
    assert _curve(logger, "lr_backbone")[5] == pytest.approx(5e-5)


def test_without_warmup_the_batch_norm_trains():
    # The pin is tied to warmup: the no-warmup path is left alone.
    model, _ = _run(epochs=3, warmup=False)
    assert [r["bn_training"] for r in model.seen] == [True] * 3


def test_a_fresh_run_records_no_source_and_unknowns(monkeypatch):
    # The params block swallows its own exceptions, so only an explicit
    # assertion proves these three reached the logger at all.
    monkeypatch.delenv("GIT_SHA", raising=False)
    monkeypatch.delenv("IMAGE_TAG", raising=False)
    _, logger = _run(epochs=1, warmup=True)
    assert logger.params["init_from"] == "none"
    assert logger.params["git_commit"] == "unknown"
    assert logger.params["image_tag"] == "unknown"


def test_a_warm_started_run_names_its_source(monkeypatch):
    source = "coco-models/hcm_coco_seg_best/20261005_221855"
    _, logger = _run(epochs=1, warmup=True, init_from=source)
    assert logger.params["init_from"] == source


def test_the_run_records_the_revision_and_image_it_ran_under(monkeypatch):
    monkeypatch.setenv("GIT_SHA", "0787549")
    monkeypatch.setenv("IMAGE_TAG", "harbor.example/ecs_hcm_trainer:tag")
    _, logger = _run(epochs=1, warmup=True)
    assert logger.params["git_commit"] == "0787549"
    assert logger.params["image_tag"] == "harbor.example/ecs_hcm_trainer:tag"
