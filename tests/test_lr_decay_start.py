"""Configurable lr decay start (openspec lr-decay-start).

The decay used to start at a fixed epoch -- 5 with warmup, 0 without. The flag
moves it. When it lands inside warmup the tricky part is *what* the factor
multiplies: warmup re-sets the backbone lr by stage every epoch but stops after
epoch 4, so multiplying the group's live lr would compound the taper across
epochs and then freeze the compound. The base has to be the stage's value for
the epoch, which is what staged_lr_for_epoch is given.

Driven on CPU through train_model_for_epochs, with the lr read back off the
scalars the loop already logs (one per group per epoch)."""

import sys

import pytest
import torch

import main as cli
import ecs_hcm_model.core.trainer as trainer
from conftest import RecordingLogger, TinyClassifier
from test_oom_guard import _Armed


def _batches(n_batches=1, batch_size=2, in_features=8, num_classes=10):
    return [
        (
            torch.randn(batch_size, in_features),
            torch.randint(0, num_classes, (batch_size,)),
        )
        for _ in range(n_batches)
    ]


@pytest.fixture(autouse=True)
def _no_early_stopping(monkeypatch):
    # As elsewhere: random batches can trip early stopping, which is not this
    # file's subject -- and these runs are multi-epoch on purpose.
    class _NeverStops:
        best_loss = float("inf")
        counter = 0

        def __init__(self, *a, **kw):
            pass

        def step(self, current_loss):
            return False

    monkeypatch.setattr(trainer, "EarlyStopping", _NeverStops)


@pytest.fixture(autouse=True)
def _shipped_min_lr_factor(monkeypatch):
    # The curves below are written against the shipped factor.
    monkeypatch.delenv("ECS_HCM_MIN_LR_FACTOR", raising=False)


def _run(*, epochs, lr_decay_start=None, warmup=True, logger=None,
         checkpoint_every_epoch=False):
    torch.manual_seed(0)
    model = TinyClassifier()
    logger = RecordingLogger() if logger is None else logger
    batches = _batches(1, 2)
    trainer.train_model_for_epochs(
        model=model,
        epochs=epochs,
        train_loader=_Armed(batches, model, True),
        val_loader=batches,
        device=torch.device("cpu"),
        logger=logger,
        test_mode=False,
        warmup=warmup,
        task="classification",
        checkpoint_every_epoch=checkpoint_every_epoch,
        lr_decay_start=lr_decay_start,
    )
    return model, logger


def _curve(logger, key):
    return [value for name, value, _ in logger.scalars if name == key]


# --- the pure functions ------------------------------------------------------


def test_the_factor_is_exactly_one_before_the_decay_starts():
    # Not merely close: before the decay the schedule must be a no-op, which is
    # what makes an omitted flag bit-for-bit unchanged.
    for epoch in range(5):
        assert trainer.lr_decay_factor(epoch, 5, 20, 0.05) == 1.0


def test_the_factor_reaches_the_minimum_at_the_last_epoch():
    assert trainer.lr_decay_factor(19, 5, 20, 0.05) == pytest.approx(0.05)


def test_the_decay_never_rises():
    values = [trainer.lr_decay_factor(e, 5, 20, 0.05) for e in range(5, 20)]
    assert values == sorted(values, reverse=True)


def test_a_phase_that_cannot_decay_has_factor_one():
    assert trainer.lr_decay_factor(0, 0, 0, 0.05) == 1.0  # run ends before it starts
    assert trainer.lr_decay_factor(0, 0, 1, 0.05) == 1.0  # one-epoch phase
    assert trainer.lr_decay_factor(3, 5, 10, 0.05) == 1.0  # before the start


def test_a_stage_lr_before_the_decay_passes_through_untouched():
    assert trainer.staged_lr_for_epoch(5e-5, 3, 5, 20, 0.05) == 5e-5


def test_a_decay_from_epoch_zero_lands_the_stage_on_the_minimum():
    stage = 3e-4
    assert trainer.staged_lr_for_epoch(stage, 19, 0, 20, 0.05) == pytest.approx(
        stage * 0.05
    )


def test_the_composition_matches_the_plain_schedule_once_warmup_is_over():
    for epoch in range(5, 20):
        assert trainer.staged_lr_for_epoch(
            5e-5, epoch, 5, 20, 0.05
        ) == trainer.lr_for_epoch(5e-5, epoch, 5, 20, 0.05)


def test_the_plain_schedule_still_reaches_the_minimum():
    # The refactor behind the flag must not have moved the existing curve.
    assert trainer.lr_for_epoch(5e-5, 19, 5, 20, 0.05) == pytest.approx(5e-5 * 0.05)


def test_an_omitted_start_resolves_to_the_warmup_mode_default():
    assert trainer.lr_decay_start_for(None, warmup=True) == 5
    assert trainer.lr_decay_start_for(None, warmup=False) == 0


def test_a_given_start_wins():
    assert trainer.lr_decay_start_for(0, warmup=True) == 0
    assert trainer.lr_decay_start_for(2, warmup=False) == 2


def test_a_negative_start_is_a_usage_error():
    with pytest.raises(trainer.LrDecayError, match=">= 0"):
        trainer.lr_decay_start_for(-1, warmup=True)


def test_a_non_integer_start_is_a_usage_error():
    with pytest.raises(trainer.LrDecayError):
        trainer.lr_decay_start_for(2.5, warmup=True)
    with pytest.raises(trainer.LrDecayError):
        trainer.lr_decay_start_for(True, warmup=True)


# --- the loop ---------------------------------------------------------------


def test_a_decay_start_inside_warmup_tapers_the_stage_values():
    """--lr-decay-start 0 over 5 epochs: the last epoch is stage * factor(4)."""
    _, logger = _run(epochs=5, lr_decay_start=0)

    backbone = _curve(logger, "lr_backbone")
    head = _curve(logger, "lr_head")

    assert len(backbone) == 5
    factor = trainer.lr_decay_factor(4, 0, 5, 0.05)
    # Warmup's last backbone stage is 5e-5, tapered by the epoch's factor.
    assert backbone[-1] == pytest.approx(5e-5 * factor)
    # Epoch 0's stage is 0.0, so tapering it stays 0.0.
    assert backbone[0] == 0.0
    # The head, which warmup never touches, is tapered off its creation lr.
    assert head[-1] == pytest.approx(3e-4 * factor)


def test_a_start_at_epoch_two_holds_the_stages_then_tapers_them():
    """--lr-decay-start 2 over 5 epochs, epoch by epoch.

    Epochs 0-2 are warmup's own stage values untouched (the decay has not
    started). Epochs 3 and 4 are *that epoch's* stage value times the factor:
    k = epoch - 2, K = 5 - 2 = 3, so factor = 0.05 + 0.95 * 0.5 *
    (1 + cos(pi * k / (K - 1))) -- the shipped cosine, which lands on
    MIN_LR_FACTOR at the final epoch. Warmup still sets the backbone's stage
    each epoch (5e-5 from epoch 3 on), which is exactly why the factor cannot
    multiply the live group lr.
    """
    _, logger = _run(epochs=5, lr_decay_start=2)

    assert _curve(logger, "lr_backbone") == pytest.approx([0.0, 1e-4, 1e-4, 2.625e-5, 2.5e-6])
    assert _curve(logger, "lr_head") == pytest.approx([3e-4, 3e-4, 3e-4, 1.575e-4, 1.5e-5])

    # Spelled out per group per epoch: unchanged stages before the start, stage
    # times the epoch's factor after it.
    for epoch, (stage, head_stage) in enumerate([(0.0, 3e-4), (1e-4, 3e-4), (1e-4, 3e-4),
                                                 (5e-5, 3e-4), (5e-5, 3e-4)]):
        factor = trainer.lr_decay_factor(epoch, 2, 5, 0.05)
        assert _curve(logger, "lr_backbone")[epoch] == pytest.approx(stage * factor)
        assert _curve(logger, "lr_head")[epoch] == pytest.approx(head_stage * factor)

    # The factor really is 1 before the start and the shipped minimum at the end.
    assert trainer.lr_decay_factor(2, 2, 5, 0.05) == 1.0
    assert trainer.lr_decay_factor(4, 2, 5, 0.05) == pytest.approx(0.05)


def test_the_taper_does_not_compound_epoch_on_epoch():
    """Epoch k's head lr is creation * factor(k), never the previous lr times a factor."""
    _, logger = _run(epochs=5, lr_decay_start=0)

    head = _curve(logger, "lr_head")

    for epoch in range(5):
        expected = trainer.staged_lr_for_epoch(3e-4, epoch, 0, 5, 0.05)
        assert head[epoch] == pytest.approx(expected), f"epoch {epoch} compounded"


def test_the_head_is_tapered_by_the_same_factor_as_the_backbone():
    _, logger = _run(epochs=4, lr_decay_start=0)

    backbone = _curve(logger, "lr_backbone")
    head = _curve(logger, "lr_head")

    # Both groups' last-epoch lr are their own stage value times the same factor.
    factor = trainer.lr_decay_factor(3, 0, 4, 0.05)
    assert backbone[-1] == pytest.approx(5e-5 * factor)
    assert head[-1] == pytest.approx(3e-4 * factor)


def test_without_the_flag_every_epoch_is_the_schedule_that_existed():
    _, logger = _run(epochs=7)

    backbone = _curve(logger, "lr_backbone")
    head = _curve(logger, "lr_head")

    # The backbone's warmup stages, untouched (factor 1.0) through epoch 4.
    assert backbone[:5] == [0.0, 1e-4, 1e-4, 5e-5, 5e-5]
    # From epoch 5 the default decay runs, exactly as lr_for_epoch does it.
    for epoch in range(5, 7):
        assert backbone[epoch] == trainer.lr_for_epoch(5e-5, epoch, 5, 7, 0.05)
    # The head is untouched during warmup and decays from its creation lr.
    assert head[:5] == [3e-4] * 5
    for epoch in range(7):
        assert head[epoch] == trainer.lr_for_epoch(3e-4, epoch, 5, 7, 0.05)


def test_an_explicit_start_at_the_default_is_identical_to_omitting_it():
    _, omitted = _run(epochs=7)
    _, explicit = _run(epochs=7, lr_decay_start=5)

    assert _curve(omitted, "lr_backbone") == _curve(explicit, "lr_backbone")
    assert _curve(omitted, "lr_head") == _curve(explicit, "lr_head")


def test_without_warmup_the_decay_still_starts_at_zero():
    _, logger = _run(epochs=4, warmup=False)

    backbone = _curve(logger, "lr_backbone")

    base = backbone[0]
    for epoch in range(4):
        assert backbone[epoch] == trainer.lr_for_epoch(base, epoch, 0, 4, 0.05)


def test_a_start_past_the_end_leaves_the_lr_flat():
    _, logger = _run(epochs=4, lr_decay_start=9)

    backbone = _curve(logger, "lr_backbone")

    assert backbone == [0.0, 1e-4, 1e-4, 5e-5]


def test_the_decay_start_is_logged_as_a_run_parameter():
    _, logger = _run(epochs=2, lr_decay_start=2)

    assert logger.params["lr_decay_start"] == 2


def test_an_omitted_start_logs_the_effective_default():
    _, logger = _run(epochs=2)

    assert logger.params["lr_decay_start"] == 5


def test_the_start_is_recorded_in_the_epoch_checkpoint(monkeypatch):
    saved = {}

    def _capture(**kwargs):
        saved.update(kwargs)

    monkeypatch.setattr(trainer, "save_epoch_checkpoint", _capture)
    _run(epochs=2, lr_decay_start=2, checkpoint_every_epoch=True)

    assert saved["lr_decay_start"] == 2


# --- the CLI ----------------------------------------------------------------


def _usage_error(monkeypatch, argv, capsys):
    monkeypatch.setattr(sys, "argv", ["main.py", *argv])
    with pytest.raises(SystemExit) as caught:
        cli.main()
    return caught.value.code, capsys.readouterr().err


def test_the_parser_defaults_the_start_to_none():
    parser = cli.setup_subpparsers()
    args = parser.parse_args(["train_hcm_imagenet1k"])

    assert args.lr_decay_start is None


def test_a_negative_decay_start_is_a_usage_error(monkeypatch, capsys):
    code, err = _usage_error(
        monkeypatch,
        ["train_hcm_imagenet1k", "--coco", "--lr-decay-start", "-1"],
        capsys,
    )

    assert code == 2
    assert "lr-decay-start" in err


def _parse(argv):
    return cli.setup_subpparsers().parse_args(argv)


def test_the_mapping_forwards_the_start(monkeypatch):
    seen = {}
    monkeypatch.setattr(cli, "run_train_hcm_imagenet1k",
                        lambda **kwargs: seen.update(kwargs))

    cli.train_hcm_1k(_parse(["train_hcm_imagenet1k", "--lr-decay-start", "2"]))

    assert seen["lr_decay_start"] == 2


def test_an_omitted_start_is_forwarded_as_none(monkeypatch):
    seen = {}
    monkeypatch.setattr(cli, "run_train_hcm_imagenet1k",
                        lambda **kwargs: seen.update(kwargs))

    cli.train_hcm_1k(_parse(["train_hcm_imagenet1k"]))

    assert seen["lr_decay_start"] is None
