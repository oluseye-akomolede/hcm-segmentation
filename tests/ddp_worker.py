"""One rank of a real two-process DDP run, driven by ``tests/test_ddp_training.py``.

Each scenario runs as an actual child process inside a gloo process group, so the
things these tests are about -- the parameter set the reducer was built with, the
all-reduces, the join at the end of an uneven epoch, and what becomes of the other
rank when one of them dies -- are the real ones rather than a mock's idea of them.

The parent sets the environment torchrun sets (WORLD_SIZE, RANK, LOCAL_RANK,
MASTER_ADDR, MASTER_PORT) and runs this module once per rank, so the rank wiring
under test is the production ``distributed_context_from_env``.

Usage::

    python tests/ddp_worker.py <scenario> <out_dir>

Each rank writes ``<out_dir>/rank-<rank>.json``. The file's ``ok`` is this rank's
own verdict on the scenario; the exit code is non-zero if it is false, or if the
scenario deliberately died (``rank_failure``). The parent asserts on both files
and both exit codes.
"""

import contextlib
import hashlib
import io
import json
import os
import sys
import traceback

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

import ecs_hcm_model.core.trainer as trainer


class _CountingLoader:
    """A loader that counts the batches the trainer actually pulled from it.

    Everything else is delegated, so the trainer still sees the wrapped
    loader's ``batch_size`` and ``__len__``.
    """

    def __init__(self, loader, counter):
        self._loader = loader
        self._counter = counter

    def __iter__(self):
        for batch in self._loader:
            self._counter[0] += 1
            yield batch

    def __len__(self):
        return len(self._loader)

    def __getattr__(self, name):
        if name == "_loader":
            raise AttributeError(name)
        return getattr(self._loader, name)


class _ListLoader:
    """The minimum the trainer asks of a loader: iteration, length, batch size.

    A list rather than a generator because the trainer takes a fresh iterator at
    the top of every epoch.
    """

    def __init__(self, batches, batch_size):
        self.batches = batches
        self.batch_size = batch_size

    def __iter__(self):
        return iter(self.batches)

    def __len__(self):
        return len(self.batches)


class _TwoStageBackbone(torch.nn.Module):
    """A backbone with two named stages.

    ``TinyClassifier``'s single Linear has no ``layer4`` attribute, so the
    trainer's staged unfreeze would be a no-op on it and the test would prove
    nothing. The widths match TinyClassifier's, so its head still fits.
    """

    def __init__(self, in_features=8, out_features=16):
        super().__init__()
        self.stem = torch.nn.Linear(in_features, out_features)
        # A layer with running statistics, so the ``staged_unfreeze`` scenario
        # can pin two things at once: DDP's per-forward broadcast of the owning
        # rank's buffers, and the eval-pinning walk keeping the statistics from
        # updating while the ranks would otherwise drift apart.
        self.bn = torch.nn.BatchNorm1d(out_features)
        self.layer4 = torch.nn.Linear(out_features, out_features)

    def forward(self, x):
        return self.layer4(self.bn(self.stem(x)))


def _model_class():
    """Defined per call so importing this module costs nothing (conftest is a
    test-support module, not something to import on behalf of a worker's own
    importers)."""
    from conftest import TinyClassifier

    class StagedClassifier(TinyClassifier):
        """TinyClassifier whose backbone can actually be frozen in stages."""

        def __init__(self):
            super().__init__()
            self.backbone = _TwoStageBackbone()

    return StagedClassifier


def _seeded(cls=None):
    """A model the ranks build identically, which is what a real run does by
    loading the same weights on every rank.

    DDP broadcasts rank 0's parameters and buffers to the other ranks when the
    wrapper is constructed (measured: a rank whose model was initialised from its
    own entropy ends the run with rank 0's weights even where it never received a
    gradient). Without this, "the frozen stage did not move" would be measuring
    DDP's construction-time sync rather than the freeze.
    """
    torch.manual_seed(4242)
    return (cls or _model_class())()


def _jsonable(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    return value


def _setup():
    """The context and process group every scenario runs inside."""
    ctx = trainer.distributed_context_from_env()
    device = torch.device("cpu")
    trainer.init_process_group_for(ctx, device)
    return ctx, device


def _loaders(ctx, n_batches, batch_size):
    """Per-rank distinct data: if the two ranks were training on the same
    examples, a test that looks for rank-to-rank drift would pass for the wrong
    reason."""
    from conftest import make_classification_batches

    torch.manual_seed(1000 + ctx.rank)
    batches = make_classification_batches(n_batches, batch_size)
    return (
        _ListLoader(batches, batch_size),
        _ListLoader(make_classification_batches(1, batch_size), batch_size),
    )


def _logger():
    from conftest import RecordingLogger

    return RecordingLogger()


def _checksum(loader):
    images, _ = next(iter(loader))
    return float(images.sum())


# --------------------------------------------------------------------------
# Scenarios
# --------------------------------------------------------------------------


def staged_unfreeze(ctx, device):
    """Two epochs over a staged unfreeze: one wrap, and the unfrozen stage is
    really synchronised.

    Epoch 0 freezes the whole backbone (so most of the model's parameters are
    unused in that iteration -- which is what has to stay legal), and epoch 1
    unfreezes layer4. The parameter set the reducer holds is fixed when the
    wrapper is built, so a wrap taken before that freeze is what makes layer4's
    gradient real; a fresh wrapper per epoch would show up as two wraps, and a
    wrap with find_unused_parameters off would raise on epoch 0.
    """
    model = _seeded()
    train_loader, val_loader = _loaders(ctx, n_batches=3, batch_size=3)

    grads = {"layer4": 0, "stem": 0}
    model.backbone.layer4.weight.register_hook(
        lambda g: grads.__setitem__("layer4", grads["layer4"] + 1)
    )
    model.backbone.stem.weight.register_hook(
        lambda g: grads.__setitem__("stem", grads["stem"] + 1)
    )

    initial = {
        k: v.detach().clone() for k, v in trainer.unwrap_or_self(model).state_dict().items()
    }
    wraps = []
    real_wrap = trainer.wrap_model_for_ddp

    def counting_wrap(module, dist_ctx):
        wraps.append(dist_ctx.rank)
        wrapped = real_wrap(module, dist_ctx)
        # After the wrap, on purpose: a buffer set here differs between the
        # ranks, so the only things that can make them agree again are DDP's
        # per-forward broadcast of the owning rank's buffers (rank 0's copy
        # overwrites rank 1's at the first forward) and the eval-pinning walk
        # (which stops the statistics updating, so the value cannot drift once
        # it has been synced). Setting it *before* the wrap would instead be
        # healed at construction and would pin nothing.
        if dist_ctx.is_distributed:
            with torch.no_grad():
                bn = wrapped.module.backbone.bn
                bn.running_mean.fill_(float(dist_ctx.rank) + 1.0)
                bn.num_batches_tracked.fill_(100 + dist_ctx.rank)
        return wrapped

    trainer.wrap_model_for_ddp = counting_wrap
    try:
        result = trainer.train_model_for_epochs(
            model=model,
            epochs=2,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            logger=_logger(),
            test_mode=False,
            warmup=True,
            finetune=False,
            coco=False,
            task="classification",
            checkpoint_every_epoch=False,
            distributed=ctx,
        )
    finally:
        trainer.wrap_model_for_ddp = real_wrap

    core = trainer.unwrap_or_self(model)
    final = core.state_dict()
    if ctx.is_main and result["best_state"] is not None:
        # The artifact the run actually writes to MinIO, so the parent can load
        # it into a plain model from a run that had two ranks.
        torch.save(result["best_state"], os.path.join(os.environ["ECS_HCM_DDP_OUT"], "best_state.pt"))
    return {
        "checksum": _checksum(train_loader),
        "wraps": len(wraps),
        "layer4_grads": grads["layer4"],
        "stem_grads": grads["stem"],
        "layer4_grad": _jsonable(core.backbone.layer4.weight.grad),
        "layer4_moved": not torch.allclose(
            initial["backbone.layer4.weight"], final["backbone.layer4.weight"]
        ),
        "stem_frozen": torch.equal(initial["backbone.stem.weight"], final["backbone.stem.weight"]),
        "last_epoch": result["last_epoch"],
        "best_state_keys": sorted(result["best_state"] or {}),
        "params": {k: _jsonable(v) for k, v in final.items()},
        # Item 8: the running statistics, which the ranks set differently and
        # must still agree on at the end of the run.
        "bn_running_mean": _jsonable(core.backbone.bn.running_mean),
        "bn_tracked": int(core.backbone.bn.num_batches_tracked),
    }


def reduce_arithmetic(ctx, device):
    """The reduction helpers, on values that make a wrong answer a different
    number: rank 0 trains 1 batch summing to 2.0, rank 1 trains 3 summing to
    12.0, so the pooled mean is 14/4 = 3.5 where a mean of the two ranks' means
    (2.0 and 4.0) would be 3.0.

    The counters are the same shape the epoch reduces: a mix of integers that
    must come back as integers and floats that must come back summed.
    """
    local = {
        "batch_count": 1 if ctx.rank == 0 else 3,
        "attempted": 1 if ctx.rank == 0 else 3,
        "oom_skipped": 0,
        "running_loss": 2.0 if ctx.rank == 0 else 12.0,
        "running_acc1": 1.0,
        "running_acc5": 2.0,
        "train_failed_images": ctx.rank,
    }
    reduced = trainer._reduce_epoch_numbers(device, dict(local))
    consecutive = trainer._all_reduce_max_int(device, 5 if ctx.rank == 0 else 21)
    fewest = trainer._all_reduce_min_float(device, 12.0 if ctx.rank == 0 else 4.5)
    return {
        "local": local,
        "reduced": reduced,
        "pooled_mean": reduced["running_loss"] / reduced["batch_count"],
        "mean_of_means": (2.0 + 4.0) / 2.0,
        "consecutive_oom": consecutive,
        "fewest_free_gib": fewest,
    }


def validation_once(ctx, device):
    """Validation must be measured on one rank, on the unwrapped model, and the
    result handed to everyone -- not measured twice and averaged.

    The stub is rank-specific, so a second measurement on rank 1 would leave a
    different number in its result, and a mean of the two would be a third.
    """
    calls = []

    def fake_evaluate(model, val_loader, device, max_batches=0, is_segmentation=False):
        calls.append(ctx.rank)
        return {
            "val_loss": 1.0 + ctx.rank,
            "val_acc1": 50.0 + ctx.rank,
            "val_acc5": 60.0 + ctx.rank,
        }

    real_evaluate = trainer.evaluate
    trainer.evaluate = fake_evaluate
    try:
        model = _seeded()
        train_loader, val_loader = _loaders(ctx, n_batches=2, batch_size=3)
        result = trainer.train_model_for_epochs(
            model=model,
            epochs=1,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            logger=_logger(),
            test_mode=False,
            warmup=False,
            finetune=False,
            coco=False,
            task="classification",
            distributed=ctx,
        )
    finally:
        trainer.evaluate = real_evaluate

    return {
        "evaluate_calls": calls,
        "best_epoch": result["best_epoch"],
        "best_metrics": {k: _jsonable(v) for k, v in (result["best_metrics"] or {}).items()},
    }


def uneven_epoch(ctx, device):
    """The ranks' shards are different lengths, so their epochs have a different
    number of batches: rank 1 runs out first and must leave the epoch without
    either rank waiting in a collective the other will never enter.

    The epochs' mean loss is pooled from both ranks, so both ranks log the same
    number and it is the mean over all four batches rather than the mean of the
    two ranks' means.
    """
    n_batches = 4 if ctx.rank == 0 else 3
    model = _seeded()
    train_loader, val_loader = _loaders(ctx, n_batches=n_batches, batch_size=3)
    logger = _logger()
    result = trainer.train_model_for_epochs(
        model=model,
        epochs=1,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        logger=logger,
        test_mode=False,
        warmup=False,
        finetune=False,
        coco=False,
        task="classification",
        distributed=ctx,
    )
    return {
        "n_batches": n_batches,
        "train_loss": [v for k, v, _ in logger.metrics if k == "train_loss"],
        "last_epoch": result["last_epoch"],
        "checksum": _checksum(train_loader),
    }


def _first_param_name(model):
    return sorted(trainer.unwrap_or_self(model).state_dict())[0]


def _first_param(model):
    state = trainer.unwrap_or_self(model).state_dict()
    return state[_first_param_name(model)]


def _model_checksum(model):
    """A byte-exact fingerprint of the model's parameters.

    Two ranks that all-reduce the same two tensors compute the same bits here
    (measured on this harness), so exact equality is a fair bar: a mismatch means
    the ranks are holding genuinely different weights, not that a float rounded a
    different way.
    """
    state = trainer.unwrap_or_self(model).state_dict()
    digest = hashlib.sha256()
    for name in sorted(state):
        digest.update(name.encode())
        digest.update(state[name].detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def uneven_epochs(ctx, device):
    """Two uneven epochs back to back, which is what the epoch boundary is about.

    Rank 1 runs out of data first, so rank 0 takes its extra step alone while
    rank 1 is shadowed by join(). The question is whether the ranks then begin
    the second epoch as the same model or as two different ones, so the run's
    final parameters are compared byte for byte across ranks.
    """
    n_batches = 4 if ctx.rank == 0 else 3
    model = _seeded()
    train_loader, val_loader = _loaders(ctx, n_batches=n_batches, batch_size=3)
    logger = _logger()
    result = trainer.train_model_for_epochs(
        model=model,
        epochs=2,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        logger=logger,
        test_mode=False,
        warmup=False,
        finetune=False,
        coco=False,
        task="classification",
        distributed=ctx,
    )
    return {
        "n_batches": n_batches,
        "train_loss": [v for k, v, _ in logger.metrics if k == "train_loss"],
        "last_epoch": result["last_epoch"],
        "param_checksum": _model_checksum(model),
        # A readable fingerprint for the failure message, in case the checksums
        # differ and someone needs to see how far apart the ranks actually are.
        "first_param": [
            _first_param_name(model),
            [float(x) for x in _first_param(model).flatten()[:6]],
        ],
    }


def accum_sync(ctx, device):
    """A group of micro-batches pays one all-reduce, at its last backward.

    Measured rather than inferred from the source: inside the accumulation
    context the wrapper's own ``require_backward_grad_sync`` flag is exactly what
    decides whether the coming backward all-reduces, so reading it once per
    micro-batch says which micro-batches sync. Five micro-batches at a group of
    two step twice -- and the trailing single micro-batch, which would be a
    different group on each rank, is discarded and logged.
    """
    flags = []
    steps = []
    real_accum = trainer.accum_step_sync

    @contextlib.contextmanager
    def probing_accum(model, is_group_end):
        with real_accum(model, is_group_end):
            flags.append(
                None
                if not trainer._is_wrapped(model)
                else bool(model.require_backward_grad_sync)
            )
            yield

    real_step = trainer._optimizer_step

    def counting_step(*args, **kwargs):
        steps.append(1)
        return real_step(*args, **kwargs)

    trainer.accum_step_sync = probing_accum
    trainer._optimizer_step = counting_step
    try:
        model = _seeded()
        train_loader, val_loader = _loaders(ctx, n_batches=5, batch_size=3)
        logger = _logger()
        trainer.train_model_for_epochs(
            model=model,
            epochs=1,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            logger=logger,
            test_mode=False,
            warmup=False,
            finetune=False,
            coco=False,
            task="classification",
            grad_accum_steps=2,
            distributed=ctx,
        )
    finally:
        trainer.accum_step_sync = real_accum
        trainer._optimizer_step = real_step
    return {
        "flags": flags,
        "steps": len(steps),
        "discarded": logger.scalar_values().get("train_discarded_partial_group"),
    }


def one_writer(ctx, device):
    """Exactly one rank writes the epoch's checkpoint, even when every epoch asks
    for one."""
    written = []
    handed_wrapped = []
    real_save = trainer.save_epoch_checkpoint

    def recording_save(**kwargs):
        written.append((ctx.rank, kwargs.get("epoch")))
        # What the loop hands the saver, not just that it called it: the artifact
        # must not carry DDP's `module.` prefix, and this is the one place the
        # loop's own hand-off is visible.
        handed_wrapped.append(trainer._is_wrapped(kwargs.get("model")))
        return None

    trainer.save_epoch_checkpoint = recording_save
    try:
        model = _seeded()
        train_loader, val_loader = _loaders(ctx, n_batches=2, batch_size=3)
        trainer.train_model_for_epochs(
            model=model,
            epochs=2,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            logger=_logger(),
            test_mode=False,
            warmup=False,
            finetune=False,
            coco=False,
            task="classification",
            checkpoint_every_epoch=True,
            distributed=ctx,
        )
    finally:
        trainer.save_epoch_checkpoint = real_save
    return {"writes": written, "handed_wrapped": handed_wrapped}


def wrap_without_find_unused(ctx, device):
    """The other half of find_unused_parameters, from the failing side.

    The same warmup freeze under a wrapper built *without* the flag must fail, so
    the flag is load-bearing rather than decoration. Both ranks freeze the same
    stage and run the same number of iterations, so both reach the error and
    neither is left waiting.

    The error is recorded rather than propagated: this scenario is *about* it.
    """
    raised = []
    model = _seeded().to(device)
    wrapper = torch.nn.parallel.DistributedDataParallel(model)
    trainer.set_requires_grad(model.backbone, False)
    train_loader, _ = _loaders(ctx, n_batches=2, batch_size=3)
    try:
        for images, labels in train_loader:
            loss = torch.nn.functional.cross_entropy(wrapper(images), labels)
            loss.backward()
    except RuntimeError as exc:
        raised.append(str(exc))
    return {"raised": raised}


def rank_failure(ctx, device):
    """A terminal error on one rank must take the other rank down with it.

    Only rank 1 fails, and it fails inside its first forward -- before its first
    backward completes the all-reduce its peer is already waiting in. The path
    under test is the real one: the peer is blocked in a collective the failing
    rank will never reach, and the group has to turn that into an error rather
    than into a wait. Nothing is caught here on the failing rank: it writes a
    marker and lets the exception end the process, which is what a real rank does.
    """
    marker = os.path.join(os.environ["ECS_HCM_DDP_OUT"], "injected.json")

    class Exploding(_model_class()):
        def forward(self, x):
            if ctx.rank == 1:
                with open(marker, "w", encoding="utf-8") as handle:
                    json.dump({"rank": 1, "error": "injected-terminal-error"}, handle)
                raise ValueError("injected-terminal-error")
            return super().forward(x)

    model = _seeded(Exploding)
    train_loader, val_loader = _loaders(ctx, n_batches=3, batch_size=3)
    trainer.train_model_for_epochs(
        model=model,
        epochs=1,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        logger=_logger(),
        test_mode=False,
        warmup=False,
        finetune=False,
        coco=False,
        task="classification",
        distributed=ctx,
    )
    return {"rank": ctx.rank, "error": None}


def oom_fraction_gate(ctx, device):
    """Show that the end-of-epoch OOM fraction gate judges the RUN totals summed
    across the ranks, not each rank's own counters.

    Each rank attempts 104 batches -- under the 200-batch floor, so no rank is
    ever judged on its own fraction -- and skips 2, at calls 10 and 60. The run
    sums to 4 of 208, past the floor and over the 1% limit, so the gate fails
    both ranks. The two ranks use the same call indices on purpose: an
    asymmetric plan would leave the ranks' DDP backward all-reduces mismatched,
    which would hang. Every number this returns is measured from the run:
    ``per_rank_batches`` is what the rank's loader handed the trainer and
    ``oom_fired_at`` is the OOM calls that actually fired.
    """
    plan = (10, 60)
    call_counts = [0]
    batches_pulled = [0]
    oom_fired = []

    class OomClassifier(_model_class()):
        def forward(self, x):
            call_counts[0] += 1
            if call_counts[0] in plan:
                oom_fired.append(call_counts[0])
                raise RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB")
            return super().forward(x)

    model = _seeded(OomClassifier)
    train_loader, val_loader = _loaders(ctx, n_batches=104, batch_size=2)
    train_loader = _CountingLoader(train_loader, batches_pulled)
    raised = False
    error = None
    try:
        trainer.train_model_for_epochs(
            model=model,
            epochs=1,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            logger=_logger(),
            test_mode=False,
            warmup=False,
            finetune=False,
            coco=False,
            task="classification",
            distributed=ctx,
        )
    except RuntimeError as exc:
        raised = True
        # Bound here: Python unbinds `exc` when the except block ends.
        error = str(exc)
    return {
        "rank": ctx.rank,
        "raised": raised,
        "error": error,
        "per_rank_batches": batches_pulled[0],
        "per_rank_skips": len(oom_fired),
        "oom_plan": list(plan),
        "oom_fired_at": oom_fired,
        # The limits the gate applies, read from the module this worker already
        # imports, so the parent test process needs no trainer import of its own.
        "min_batches_for_fraction": trainer.MIN_BATCHES_FOR_OOM_FRACTION,
        "max_skip_fraction": trainer.MAX_OOM_SKIP_FRACTION,
    }


def test_mode_after_wrap(ctx, device):
    """``test_mode``'s diagnostics read the model, so they must outlive the wrap.

    Two of them go to the model rather than through ``forward``: the smoke-mode
    block prints the segmentation head's share of the optimizer, and the epoch's
    activation maxima are folded out of the model's own counters. A DDP wrapper
    forwards neither -- ``segmentation_head`` raises AttributeError, and
    ``activation_absmax`` is not reachable at all, so a fold through the wrapper
    finds nothing and reports no maxima. Only ``test_mode`` reaches the first,
    which is why a single-process run never met it.

    Both are observable from outside: the printed line carries the head's
    parameter count, and the module's counters are seeded with a value that only
    the module itself holds, so the epoch metric says which object was read.
    """
    model = _seeded()
    # Distinct values per stage: the fold keeps a stage only when its value is
    # above the accumulator's, so three different numbers prove it read the
    # module's dict for every stage rather than once for one of them.
    model.activation_absmax.update({"isc": 3.5, "csc": 1.25, "isc3": 0.5})

    train_loader, val_loader = _loaders(ctx, n_batches=2, batch_size=3)
    logger = _logger()

    steps = []
    real_step = trainer._optimizer_step

    def counting_step(*args, **kwargs):
        steps.append(1)
        return real_step(*args, **kwargs)

    printed = io.StringIO()
    trainer._optimizer_step = counting_step
    try:
        with contextlib.redirect_stdout(printed):
            trainer.train_model_for_epochs(
                model=model,
                epochs=1,
                train_loader=train_loader,
                val_loader=val_loader,
                device=device,
                logger=logger,
                test_mode=True,
                warmup=True,
                finetune=False,
                coco=False,
                task="classification",
                distributed=ctx,
            )
    finally:
        trainer._optimizer_step = real_step

    head_lines = [
        line for line in printed.getvalue().splitlines()
        if line.startswith("seg params in optimizer:")
    ]
    return {
        "rank": ctx.rank,
        "seg_params_line": head_lines[0] if head_lines else None,
        "steps": len(steps),
        "activation_absmax": {
            key: value
            for key, value in logger.scalar_values().items()
            if key.endswith("_absmax")
        },
        # The printed line counts parameter *tensors* (weight, bias), not numel.
        "head_param_tensors": len(list(model.segmentation_head.parameters())),
    }


def outline_unlabelled_batch(ctx, device):
    """A segmentation step whose second batch has no full-outline label at all.

    The outline term is a masked mean over the labelled samples, so on a batch
    where none is labelled it is exactly zero -- and a zero from a *compute*
    whose result is unused would leave the head's parameters outside the graph
    and DDP would refuse the backward ("Expected to have finished reduction").
    The first batch is labelled, so the labels-arrived check passes and the
    epoch really runs; the second is the case under test.

    The outline "head" here is a Linear multiplied into the forward, so its
    gradient is a real (zero) tensor rather than an absent one: this measures
    the trainer's guarantee, not the fake's.
    """
    from conftest import TinySegmenter, make_segmentation_batches

    class OutlineSeg(TinySegmenter):
        def __init__(self):
            super().__init__(in_features=8)
            # Same storage name the real model uses, so the code under test
            # finds it the same way.
            self.outline_head = torch.nn.Linear(1, 1)
            self.amodal_outline = True

        def forward(self, x, return_outline=False):
            out = super().forward(x)
            if not return_outline:
                return out
            scale = self.outline_head.weight.view(1, 1, 1, 1)
            bias = self.outline_head.bias.view(1, 1, 1, 1)
            return out, out[:, 1:2] * scale + bias

    model = _seeded(OutlineSeg)
    torch.manual_seed(1000 + ctx.rank)      # each rank its own shard

    train_batches = make_segmentation_batches(n_batches=2, batch_size=3, in_features=8)
    for i, batch in enumerate(train_batches):
        labelled = i == 0
        visible = batch["segmentation_mask"][:, 0] > 0.5      # [B,H,W], as a loader gives it
        outline = visible if labelled else torch.zeros_like(visible)
        batch["outline_mask"] = outline.to(torch.int64)
        batch["has_full_outline"] = torch.full((3,), labelled, dtype=torch.bool)
        batch["outline_unknown_images"] = torch.tensor(
            0 if labelled else 3, dtype=torch.int64)

    val_batches = make_segmentation_batches(n_batches=1, batch_size=3, in_features=8)
    for batch in val_batches:
        batch["outline_mask"] = (batch["segmentation_mask"][:, 0] > 0.5).to(torch.int64)
        batch["has_full_outline"] = torch.ones(3, dtype=torch.bool)
        batch["outline_unknown_images"] = torch.tensor(0, dtype=torch.int64)

    logger = _logger()
    result = trainer.train_model_for_epochs(
        model=model,
        epochs=1,
        train_loader=_ListLoader(train_batches, 3),
        val_loader=_ListLoader(val_batches, 3),
        device=device,
        logger=logger,
        test_mode=False,
        warmup=False,
        finetune=True,
        coco=False,
        task="segmentation",
        distributed=ctx,
        amodal_outline=True,
    )
    scalars = {k: v for k, v, _ in logger.scalars}
    return {
        "rank": ctx.rank,
        # log_scalar lands in .scalars; .metrics carries the tracked DataInst.
        "losses": [v for k, v, _ in logger.scalars if k == "train_outline_loss"],
        "labelled": [v for k, v, _ in logger.scalars
                     if k == "train_outline_labelled_images"],
        "train_loss": [v for k, v, _ in logger.metrics if k == "train_loss"],
        "val_full_iou": scalars.get("val_full_iou"),
        # The head moved on the labelled batch and not on the unlabelled one.
        "outline_weight": trainer.unwrap_or_self(model).outline_head.weight.detach().cpu().tolist(),
        "last_epoch": result["last_epoch"],
    }


SCENARIOS = {
    "test_mode_after_wrap": test_mode_after_wrap,
    "outline_unlabelled_batch": outline_unlabelled_batch,
    "staged_unfreeze": staged_unfreeze,
    "reduce_arithmetic": reduce_arithmetic,
    "validation_once": validation_once,
    "uneven_epoch": uneven_epoch,
    "uneven_epochs": uneven_epochs,
    "accum_sync": accum_sync,
    "one_writer": one_writer,
    "rank_failure": rank_failure,
    "wrap_without_find_unused": wrap_without_find_unused,
    "oom_fraction_gate": oom_fraction_gate,
}


def main(argv):
    scenario, out_dir = argv[1], argv[2]
    ctx = trainer.distributed_context_from_env()
    payload = {"scenario": scenario, "rank": ctx.rank, "world_size": ctx.world_size}
    try:
        # The real process group, brought up before any scenario runs: every
        # scenario below is about what the group does, so a scenario without one
        # would silently measure the single-process path.
        _setup()
        payload["data"] = SCENARIOS[scenario](ctx, torch.device("cpu"))
        payload["ok"] = True
    except BaseException as exc:  # pylint: disable=broad-exception-caught
        payload["ok"] = False
        payload["error"] = f"{type(exc).__name__}: {exc}"
        payload["traceback"] = traceback.format_exc()
    finally:
        # The group is torn down here, as the entry point does in its finally,
        # and the resulting state is recorded rather than assumed.
        trainer.destroy_process_group_if_initialized()
        payload["group_destroyed"] = not torch.distributed.is_initialized()

    with open(os.path.join(out_dir, f"rank-{ctx.rank}.json"), "w", encoding="utf-8") as handle:
        json.dump(payload, handle)
    return 0 if payload.get("ok") else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
