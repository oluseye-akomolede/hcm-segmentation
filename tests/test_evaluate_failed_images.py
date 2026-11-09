"""evaluate() does not score the blanks that failed images leave behind.

The collate drops samples that failed to load or transform. A batch whose
every sample failed arrives as a one-sample zero placeholder marked
``valid=False``; scored, it averaged a blank into ``val_iou`` as if it were
data. evaluate() now skips placeholders the way the eval tool does, counts the
failed images (a placeholder costs a whole batch; a short batch costs what it
is short of, since every val loader uses ``drop_last=True``), and the training
loop logs that count as ``val_failed_images`` every epoch.
"""

import pytest
import torch

from ecs_hcm_model.core import trainer as trainer_module
from ecs_hcm_model.core.trainer import evaluate, merge_mixed_val_metrics

from conftest import RecordingLogger, TinySegmenter, make_segmentation_batches
from test_coco_mix import _clean_seams, _script_evaluate  # noqa: F401  (fixture)

B, H, W = 2, 8, 8


class AlwaysForeground(torch.nn.Module):
    """Calls every pixel foreground: IoU 1 on an all-foreground target, 0 on a blank."""

    def forward(self, x):
        n = x.shape[0]
        fg = torch.full((n, 1, x.shape[2], x.shape[3]), 10.0)
        return torch.cat([-fg, fg], dim=1)


class Loader(list):
    """A list of batches with a DataLoader's ``batch_size``."""

    def __init__(self, batches, batch_size=B):
        super().__init__(batches)
        self.batch_size = batch_size


def good(n=B):
    return {
        "pixel_values": torch.ones(n, 3, H, W),
        "segmentation_mask": torch.ones(n, H, W, dtype=torch.int64),
        "valid": torch.ones(n, dtype=torch.bool),
    }


def placeholder():
    # What coco_collate_fn returns when every sample in the batch failed.
    return {
        "pixel_values": torch.zeros(1, 3, H, W),
        "segmentation_mask": torch.zeros(1, H, W, dtype=torch.int64),
        "valid": torch.tensor([False]),
    }


def run(loader):
    return evaluate(AlwaysForeground(), loader, torch.device("cpu"), is_segmentation=True)


class TestPlaceholdersAreNotScored:
    def test_the_score_is_unchanged_by_a_placeholder(self):
        clean = run(Loader([good(), good()]))
        with_blank = run(Loader([good(), placeholder(), good()]))
        assert clean["val_iou"] == pytest.approx(100.0, abs=0.01)
        assert with_blank["val_iou"] == pytest.approx(clean["val_iou"])
        assert with_blank["val_dice"] == pytest.approx(clean["val_dice"])

    def test_a_placeholder_counts_a_whole_batch_of_failed_images(self):
        assert run(Loader([good(), placeholder(), placeholder()]))["val_failed_images"] == 2 * B

    def test_a_short_batch_is_scored_and_counts_what_it_is_short_of(self):
        out = run(Loader([good(), good(1)], batch_size=B))
        assert out["val_failed_images"] == 1
        assert out["val_iou"] == pytest.approx(100.0, abs=0.01)

    def test_no_failures_counts_zero(self):
        assert run(Loader([good(), good()]))["val_failed_images"] == 0

    def test_only_placeholders_is_a_validation_with_nothing_to_report(self):
        with pytest.raises(RuntimeError, match="0 batches"):
            run(Loader([placeholder(), placeholder()]))

    def test_the_only_placeholders_error_names_the_failed_count(self):
        with pytest.raises(RuntimeError, match=rf"\({2 * B} images failed to load\)"):
            run(Loader([placeholder(), placeholder()]))

    def test_without_the_skip_the_blank_would_have_halved_the_score(self):
        # The control: the same two batches with the placeholder marked valid,
        # i.e. scored, is what the old evaluate() reported.
        blank = placeholder()
        blank["valid"] = torch.tensor([True])
        assert run(Loader([good(), blank]))["val_iou"] == pytest.approx(50.0, abs=0.01)

    def test_a_loader_without_a_batch_size_still_skips_placeholders(self):
        out = run([good(), placeholder(), good()])
        assert out["val_iou"] == pytest.approx(100.0, abs=0.01)
        assert out["val_failed_images"] == 1

    def test_classification_reports_no_failed_count(self):
        model = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(3 * H * W, 10))
        batch = {"pixel_values": torch.ones(B, 3, H, W), "label": torch.zeros(B, dtype=torch.int64)}
        out = evaluate(model, [batch], torch.device("cpu"), is_segmentation=False)
        assert "val_failed_images" not in out


class TestTheCountIsLogged:
    def test_merging_two_datasets_sums_the_count_and_keeps_each(self):
        a = {"val_loss": 1.0, "val_iou": 40.0, "val_dice": 50.0, "val_failed_images": 2}
        b = {"val_loss": 2.0, "val_iou": 60.0, "val_dice": 70.0, "val_failed_images": 3}
        merged = merge_mixed_val_metrics(a, b)
        assert merged["val_failed_images"] == 5
        assert merged["val_failed_images_cocoa"] == 2 and merged["val_failed_images_coco"] == 3
        assert merged["val_iou"] == pytest.approx(50.0)

    def test_the_loop_logs_it_every_epoch_zero_included(self, monkeypatch):
        _script_evaluate(monkeypatch, [
            {"val_loss": 1.0, "val_iou": 40.0, "val_dice": 50.0, "val_failed_images": 0},
            {"val_loss": 0.9, "val_iou": 41.0, "val_dice": 51.0, "val_failed_images": 4},
        ])
        logger = RecordingLogger()
        trainer_module.train_model_for_epochs(
            model=TinySegmenter(), epochs=2,
            train_loader=make_segmentation_batches(n_batches=2, batch_size=2),
            val_loader=make_segmentation_batches(n_batches=1, batch_size=2),
            device=torch.device("cpu"), logger=logger, test_mode=True,
            finetune=True, coco=False, task="cocoa_seg",
        )
        logged = [(v, step) for k, v, step in logger.scalars if k == "val_failed_images"]
        assert logged == [(0.0, 0), (4.0, 1)]


class TestEvaluateRunsUnderNoGrad:
    """evaluate() scores; it never trains, so its forward must build no graph.

    Without ``@torch.no_grad()`` a val pass keeps every activation alive for a
    backward that never comes -- the reviewer measured +172 MB peak on one pass --
    and the whole suite stayed green. The tensors ``on_metric_inputs`` receives are
    the model's own output (``calculate_segmentation_loss`` hands over
    ``pred_for_metrics`` with no detach), so they show the graph at its source.
    """

    def test_the_metric_receives_a_tensor_with_no_graph(self):
        seen = {}

        def spy(logits, target, batch):
            seen["requires_grad"] = logits.requires_grad
            seen["grad_fn"] = logits.grad_fn

        # A model with parameters: without them the output has no graph to find.
        model = TinySegmenter(in_features=3 * H * W, h=H, w=W)
        assert any(p.requires_grad for p in model.parameters())

        evaluate(model, Loader([good(), good()]), torch.device("cpu"),
                 is_segmentation=True, on_metric_inputs=spy)

        assert seen, "on_metric_inputs was never called"
        assert seen["requires_grad"] is False
        assert seen["grad_fn"] is None
