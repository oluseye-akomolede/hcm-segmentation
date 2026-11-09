"""Gradient accumulation over batches (openspec seg-aug-and-grad-accum), driven
on CPU through train_model_for_epochs.

The point of the feature is that the optimizer step applies the mean gradient of
the group, so --grad-accum-steps 2 over two batches of 2 must give the same
update as one batch of 4 -- that is what makes the ablation comparable with a
run that fits the whole batch in memory. The model here is the tiny MLP, which
has no BatchNorm: with BN the two runs would differ by the batch statistics of a
2-sample forward, which is inherent to splitting a batch and not something the
step can undo."""

import pytest
import torch

import ecs_hcm_model.core.trainer as trainer
from conftest import RecordingLogger, TinyClassifier
from test_oom_guard import FlakyModel, _Armed


@pytest.fixture(autouse=True)
def _no_early_stopping(monkeypatch):
    # As elsewhere: random batches can trip early stopping, which is not this
    # file's subject.
    class _NeverStops:
        best_loss = float("inf")
        counter = 0

        def __init__(self, *a, **kw):
            pass

        def step(self, current_loss):
            return False

    monkeypatch.setattr(trainer, "EarlyStopping", _NeverStops)


@pytest.fixture
def steps(monkeypatch):
    """Count optimizer steps, through the trainer's own step helper.

    Counting here rather than by wrapping torch.optim.AdamW keeps the patch
    local to the trainer module: the loop calls the bare global name, so the
    seam is exactly the call site under test.
    """
    count = {"n": 0}
    real_step = trainer._optimizer_step

    def _counting(model, optimizer, scaler):
        count["n"] += 1
        return real_step(model, optimizer, scaler)

    monkeypatch.setattr(trainer, "_optimizer_step", _counting)
    return count


def _batches(n_batches=2, batch_size=2, in_features=8, num_classes=10):
    return [
        (torch.randn(batch_size, in_features), torch.randint(0, num_classes, (batch_size,)))
        for _ in range(n_batches)
    ]


def _run(train_batches, grad_accum_steps=1, model=None, logger=None, epochs=1,
         val_batches=None, checkpoint_every_epoch=False):
    """One training run with a fresh, identically initialised tiny model.

    ``val_batches`` defaults to the same ``_Armed`` wrapper as the train loader,
    which is fine except where a test watches ``train_calls``: that wrapper
    resets the counter at the top of every iteration, and the val loader
    iterates after training.  A plain list keeps the count intact there.
    """
    torch.manual_seed(0)
    model = TinyClassifier() if model is None else model
    logger = RecordingLogger() if logger is None else logger
    if val_batches is None:
        val_loader = _Armed(_batches(2, 2), model, False)
    else:
        val_loader = val_batches
    trainer.train_model_for_epochs(
        model=model,
        epochs=epochs,
        train_loader=_Armed(train_batches, model, True),
        val_loader=val_loader,
        device=torch.device("cpu"),
        logger=logger,
        test_mode=False,
        warmup=True,
        task="classification",
        checkpoint_every_epoch=checkpoint_every_epoch,
        grad_accum_steps=grad_accum_steps,
    )
    return model, logger


def _train_loss(logger):
    """The epoch's reported train loss."""
    values = [value for key, value, _ in logger.metrics if key == "train_loss"]
    assert values, f"no train_loss logged: {logger.metrics!r}"
    return values[-1]


def test_a_group_of_two_batches_makes_one_step(steps):
    _run(_batches(2, 2), grad_accum_steps=2)
    assert steps["n"] == 1, "2 batches at --grad-accum-steps 2 must be one step"


def test_one_step_per_batch_is_the_default(steps):
    _run(_batches(4, 2))
    assert steps["n"] == 4, "without the flag every batch steps, as before"


def test_steps_one_is_identical_to_the_default(steps):
    """N=1 is the old path: same weights, one step per batch."""
    with_flag, _ = _run(_batches(3, 2), grad_accum_steps=1)
    default, _ = _run(_batches(3, 2))
    assert steps["n"] == 6
    for a, b in zip(with_flag.parameters(), default.parameters()):
        assert torch.equal(a, b)


def test_a_group_of_two_batches_matches_one_batch_of_four(steps):
    """The equivalence the feature exists for.

    Two batches of 2 with --grad-accum-steps 2, against one batch of 4 with the
    default: the same four samples, the same number of optimizer steps, so the
    same weights. Nothing here is random between the two runs (both seeded from
    0 inside _run), so the tolerance only has to absorb the order the gradients
    were summed in.
    """
    x = torch.randn(4, 8)
    y = torch.tensor([0, 1, 2, 3])
    grouped, _ = _run([(x[:2], y[:2]), (x[2:], y[2:])], grad_accum_steps=2)
    whole, _ = _run([(x, y)], grad_accum_steps=1)
    assert steps["n"] == 2, "one step for each run"
    for a, b in zip(grouped.parameters(), whole.parameters()):
        assert torch.allclose(a, b, atol=1e-6), (
            "a group must apply the mean gradient of its batches, not of one "
            "micro-batch or of the group twice"
        )


def test_a_group_applies_the_mean_gradient(monkeypatch):
    """The step must see the *mean* gradient of the group, not the sum.

    This is the test the weight comparison cannot be: Adam divides by the
    gradient's own scale, so its one-step weight update is the same whether a
    group summed its batches' gradients or averaged them -- the missing ``/ N``
    is invisible in the weights. The gradient the step is handed is not. Two
    batches of 2 with --grad-accum-steps 2 must present the same gradient as one
    batch of 4 with the default.
    """
    seen = []
    real = trainer._optimizer_step

    def _capture(model, optimizer, scaler):
        seen.append([p.grad.detach().clone() for p in model.parameters() if p.grad is not None])
        return real(model, optimizer, scaler)

    monkeypatch.setattr(trainer, "_optimizer_step", _capture)

    x = torch.randn(4, 8)
    y = torch.tensor([0, 1, 2, 3])
    _run([(x[:2], y[:2]), (x[2:], y[2:])], grad_accum_steps=2)
    _run([(x, y)], grad_accum_steps=1)
    assert len(seen) == 2, "one step for each run"
    grouped, whole = seen
    for a, b in zip(grouped, whole):
        assert torch.allclose(a, b, atol=1e-6), (
            "a group must accumulate the mean gradient of its batches -- a sum "
            "would be a factor of --grad-accum-steps too large"
        )


def test_the_reported_loss_is_the_mean_of_the_group(steps):
    """The reported loss must not be divided by the accumulation count.

    The group's mean over the two batches equals the mean over the same four
    samples in one batch, so the two runs must report the same train loss. A
    reported ``loss / N`` would come out at half.
    """
    x = torch.randn(4, 8)
    y = torch.tensor([0, 1, 2, 3])
    _, grouped = _run([(x[:2], y[:2]), (x[2:], y[2:])], grad_accum_steps=2)
    _, whole = _run([(x, y)], grad_accum_steps=1)
    assert _train_loss(grouped) == pytest.approx(_train_loss(whole), rel=1e-6)


def test_a_partial_final_group_still_steps(steps):
    """5 batches at N=2: two full groups and a last group of one."""
    _run(_batches(5, 2), grad_accum_steps=2)
    assert steps["n"] == 3, "the last batch of the epoch must not be dropped"


def test_a_single_batch_epoch_still_steps(steps):
    _run(_batches(1, 2), grad_accum_steps=4)
    assert steps["n"] == 1


def test_an_oom_batch_abandons_its_group(steps):
    """The batch that OOMs leaves no gradient behind for the next group.

    4 batches at N=2 where batch 2 OOMs. Batch 1's group is abandoned -- its
    gradient is dropped, not carried -- so batches 3 and 4 form one clean group
    and the epoch makes a single step. If the failed batch were simply skipped
    without abandoning the group, batch 3 would complete a phantom group of one
    real batch and the epoch would make two steps: the count is what separates
    "abandon" from "skip".

    (Three batches would not: either way the tail collapses to one step.)
    """
    model = FlakyModel([2])
    _run(_batches(4, 2), model=model, grad_accum_steps=2, val_batches=_batches(2, 2))
    assert model.train_calls == 4, "all four batches were attempted"
    assert steps["n"] == 1, "the abandoned group must not be completed by a later batch"


def test_a_bad_count_is_a_usage_error():
    with pytest.raises(trainer.GradAccumError):
        _run(_batches(2, 2), grad_accum_steps=0)
    with pytest.raises(trainer.GradAccumError):
        _run(_batches(2, 2), grad_accum_steps=-3)
    with pytest.raises(trainer.GradAccumError):
        _run(_batches(2, 2), grad_accum_steps="two")


def test_the_count_is_recorded_in_the_epoch_checkpoint(monkeypatch):
    """A checkpoint says which accumulation it trained with."""
    import json

    saved = {}

    def _capture(**kwargs):
        saved.update(kwargs)

    monkeypatch.setattr(trainer, "save_epoch_checkpoint", _capture)
    _run(_batches(2, 2), grad_accum_steps=3, checkpoint_every_epoch=True)
    assert saved["augmentation"]["grad_accum_steps"] == 3
    assert json.dumps(saved["augmentation"])  # JSON-serialisable as recorded
