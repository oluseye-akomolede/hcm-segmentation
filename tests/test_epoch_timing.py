"""The trainer's per-epoch timing split (openspec epoch-timing-split), driven
on CPU through train_model_for_epochs with loaders that "sleep" a known time.

The trainer reads only ``time.perf_counter``; here it reads a fake clock that
the loaders advance, so every figure is exact and a loaded machine cannot
move a bound (wall-clock bounds tripped under load on the VM)."""

import re
import types
from pathlib import Path

import pytest
import torch

import ecs_hcm_model.core.trainer as trainer
from conftest import (RecordingLogger, TinyClassifier, TinySegmenter,
                      make_classification_batches, make_segmentation_batches)

REPO = Path(__file__).resolve().parents[1]
SPEC_REQUIREMENT = "### Requirement: Each epoch logs its timing split"


@pytest.fixture(autouse=True)
def _no_early_stopping(monkeypatch):
    # As in test_lr_schedule: random batches can trip early stopping, which is
    # not this file's subject.
    class _NeverStops:
        best_loss = float("inf")
        counter = 0

        def __init__(self, *a, **kw):
            pass

        def step(self, current_loss):
            return False

    monkeypatch.setattr(trainer, "EarlyStopping", _NeverStops)


class FakeClock:
    """``perf_counter`` for the trainer: advances only when told to, plus
    1 us per read so no interval is exactly zero."""

    TICK = 1e-6

    def __init__(self):
        self.t = 0.0

    def __call__(self):
        self.t += self.TICK
        return self.t


@pytest.fixture(autouse=True)
def clock(monkeypatch):
    c = FakeClock()
    monkeypatch.setattr(trainer, "time", types.SimpleNamespace(perf_counter=c))
    return c


# Reads per batch are a handful; this bounds what ticks can add to a figure.
TICKS = 1e-3


class SleepyBatches(list):
    """A list of batches that advances the fake clock ``delay`` seconds
    before yielding each, as a loader that blocks would."""

    def __init__(self, batches, delay):
        super().__init__(batches)
        self.delay = delay

    def __iter__(self):
        clock = trainer.time.perf_counter
        for b in list.__iter__(self):
            clock.t += self.delay
            yield b


def _run(epochs=1, train_delay=0.0, val_delay=0.0, n_train=3, batch_size=4):
    logger = RecordingLogger()
    trainer.train_model_for_epochs(
        model=TinyClassifier(),
        epochs=epochs,
        train_loader=SleepyBatches(make_classification_batches(n_train, batch_size), train_delay),
        val_loader=SleepyBatches(make_classification_batches(2, batch_size), val_delay),
        device=torch.device("cpu"),
        logger=logger,
        test_mode=False,
        warmup=True,
        task="classification",
        checkpoint_every_epoch=False,
    )
    out = {}
    for key, value, step in logger.scalars:
        if key in trainer.EPOCH_TIMING_KEYS:
            out.setdefault(key, []).append((step, value))
    return out


def _at(split, key, epoch=0):
    return dict(split[key])[epoch]


def test_time_blocked_on_the_loader_is_data_wait_not_step():
    delay, n = 10.0, 3
    split = _run(train_delay=delay, n_train=n)
    wait, step, train = (_at(split, k) for k in
                         ("epoch_data_wait_s", "epoch_step_s", "epoch_train_s"))
    assert wait == pytest.approx(delay * n, abs=TICKS)
    assert step == pytest.approx(0, abs=TICKS)  # none of the wait leaks into it
    assert train == pytest.approx(wait + step, abs=TICKS)


def test_validation_is_timed_apart_from_training():
    # Training waits 3 x 10 s on its loader, validation 2 x 1 s on its own:
    # each figure must hold its own wait and not the other's.
    train_delay, val_delay = 10.0, 1.0
    split = _run(train_delay=train_delay, val_delay=val_delay, n_train=3)
    assert _at(split, "epoch_val_s") == pytest.approx(2 * val_delay, abs=TICKS)
    assert _at(split, "epoch_train_s") == pytest.approx(3 * train_delay, abs=TICKS)
    assert _at(split, "epoch_data_wait_s") == pytest.approx(3 * train_delay, abs=TICKS)


def test_a_mixed_runs_second_validation_is_inside_val_time():
    # A refine scores both datasets. The mixed val loader alone waits, so a
    # clock that stopped before the second evaluate would miss all of it.
    mix_val_delay = 5.0
    logger = RecordingLogger()
    trainer.train_model_for_epochs(
        model=TinySegmenter(),
        epochs=1,
        train_loader=make_segmentation_batches(n_batches=2, batch_size=2),
        val_loader=make_segmentation_batches(n_batches=1, batch_size=2),
        device=torch.device("cpu"),
        logger=logger,
        finetune=True,
        task="cocoa_seg",
        checkpoint_every_epoch=False,
        mix_val_loader=SleepyBatches(make_segmentation_batches(n_batches=2, batch_size=2), mix_val_delay),
    )
    val = [v for k, v, _ in logger.scalars if k == "epoch_val_s"]
    assert len(val) == 1 and val[0] == pytest.approx(2 * mix_val_delay, abs=TICKS)  # two batches


def test_samples_per_second_counts_samples_over_the_train_wall_time():
    split = _run(train_delay=2.0, n_train=3, batch_size=4)
    assert _at(split, "epoch_train_s") == pytest.approx(6.0, abs=TICKS)
    assert _at(split, "train_samples_per_s") == pytest.approx(12 / _at(split, "epoch_train_s"))


def test_each_key_is_logged_once_per_epoch_at_the_epoch_step():
    split = _run(epochs=3)
    assert sorted(split) == sorted(trainer.EPOCH_TIMING_KEYS)
    for key, points in split.items():
        assert [s for s, _ in points] == [0, 1, 2], key


def test_the_key_names_are_the_specs():
    """The five keys the trainer logs are the five the spec names, wherever
    the requirement lives (the change dir now, the main spec once archived)."""
    holders = [p for p in REPO.glob("openspec/**/experiment-logging/spec.md")
               if SPEC_REQUIREMENT in p.read_text()]
    assert holders, "no spec carries the timing-split requirement"
    text = holders[0].read_text().split(SPEC_REQUIREMENT, 1)[1].split("\n### ", 1)[0]
    named = re.findall(r"^- `([a-z_]+)`:", text, flags=re.M)
    assert named == list(trainer.EPOCH_TIMING_KEYS)


def test_cuda_is_synchronised_at_the_edges_only_not_per_batch(monkeypatch):
    calls = []
    monkeypatch.setattr(trainer, "_sync_if_cuda", lambda device: calls.append(device))
    trainer.train_model_for_epochs(
        model=TinyClassifier(), epochs=2,
        train_loader=make_classification_batches(5, 4),
        val_loader=make_classification_batches(2, 4),
        device=torch.device("cpu"), logger=RecordingLogger(), test_mode=False,
        warmup=True, task="classification", checkpoint_every_epoch=False,
    )
    # Epoch start, train end, val end: three per epoch whatever the batch count.
    assert len(calls) == 3 * 2


def test_sync_if_cuda_synchronises_only_a_cuda_device(monkeypatch):
    synced = []
    monkeypatch.setattr(torch.cuda, "synchronize", lambda device=None: synced.append(device))
    trainer._sync_if_cuda(torch.device("cpu"))
    assert synced == []
    trainer._sync_if_cuda("cuda:0")
    assert synced == ["cuda:0"]


def test_no_logger_does_not_raise():
    trainer.train_model_for_epochs(
        model=TinyClassifier(), epochs=1,
        train_loader=make_classification_batches(2, 4),
        val_loader=make_classification_batches(2, 4),
        device=torch.device("cpu"), logger=None, test_mode=False, warmup=True,
        task="classification", checkpoint_every_epoch=False,
    )
