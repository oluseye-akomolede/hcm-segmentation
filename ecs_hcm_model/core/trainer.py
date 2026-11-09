
import contextlib
import inspect
import math
import os
import signal
import sys
import time
import torch
import torch.distributed as dist
import torch.nn.functional as F
import traceback
from dataclasses import dataclass
from torch.utils.data import DataLoader as Dataloader
from huggingface_hub.utils import HfHubHTTPError
from typing import Literal

# pylint: disable=import-error disable=no-name-in-module
from ecs_hcm_model.core.model_logger import (
    data_obj,
    MLFlowHCMLogger,
    logger_config,
    TASK_CLASSIFICATION,
    TASK_SEGMENTATION,
    normalise_task,
)

from ecs_hcm_model.core.bdd import BDD_DATASET_NAME, fetch_prepared_tree
from ecs_hcm_model.core.model_utils import (
    load_hcm_model,
    warm_start_state,
    download_checkpoint_state,
    parse_checkpoint_spec,
    get_training_dataloader,
    get_validation_dataloader,
    DEFAULT_INPUT_SIZE,
)

from ecs_hcm_model.core.model_upload_download import (
    ModelUploaderAndDownloader,
    resolve_task,
    checkpoint_target,
)
from ecs_hcm_model.core.hcm_nn import (
    DECODER_V2,
    DECODER_V1,
    DEFAULT_BACKBONE,
    MASK_STRIDE_DEFAULT,
    arch_record_of,
    hcm_nn,
    validate_arch,
)

from tqdm import tqdm
from torch.cuda.amp import GradScaler, autocast
from colorama import Fore, Style

# Define custom exception
class KilledException(Exception):
    pass


class SignalKilled(KilledException):
    """Raised from a signal handler so the run ends KILLED rather than vanishing.

    Airflow deletes the pod on cancellation, which delivers SIGTERM. Without a
    handler the default disposition terminates the process immediately: the
    MLflow run it had just created stayed RUNNING forever, because no exit path
    ran. Carries the signal name so the KILLED run can be tagged with why.
    """

    def __init__(self, signum: int):
        self.signum = signum
        self.signame = signal.Signals(signum).name
        super().__init__(f"Received {self.signame} ({signum})")


# Previous handlers, restored on the way out so importing/driving the trainer
# from a test or a notebook does not leave the process with a handler installed.
_previous_signal_handlers = {}


def _signal_handler(signum, _frame):
    raise SignalKilled(signum)


def install_kill_handlers():
    """Install SIGTERM/SIGINT handlers that raise KilledException.

    Only the main thread may install signal handlers; a worker thread (or an
    embedding process that already owns the handler) is left alone rather than
    crashing the run at startup. Returns True if the handlers are installed.

    Idempotent: a second call keeps the originally-saved handlers, so
    restore_kill_handlers() always puts back the true originals rather than the
    previous install's own handler.
    """
    import threading

    if _previous_signal_handlers:
        # Already installed (possibly by an outer caller in this process). Do
        # not re-save: signal.getsignal would now return our own handler, and
        # restoring that would leave the process permanently patched.
        return True

    if threading.current_thread() is not threading.main_thread():
        return False

    saved = {}
    try:
        for sig in (signal.SIGTERM, signal.SIGINT):
            saved[sig] = signal.getsignal(sig)
            signal.signal(sig, _signal_handler)
    except (ValueError, OSError):
        # Not installable here (e.g. a restricted runtime); undo anything this
        # call already changed so the failure leaves no partial patch behind.
        for sig, previous in saved.items():
            try:
                signal.signal(sig, previous)
            except (ValueError, OSError):
                pass
        return False

    _previous_signal_handlers.update(saved)
    return True


def restore_kill_handlers():
    """Put back whatever handlers were in place before install_kill_handlers()."""
    while _previous_signal_handlers:
        sig, previous = _previous_signal_handlers.popitem()
        try:
            signal.signal(sig, previous)
        except (ValueError, OSError):
            pass


# Human-readable dataset name per checkpoint task, recorded in metadata.
DATASET_FOR_TASK = {
    "classification": "imagenet-1k",
    "coco_seg": "MSCOCO",
    "cocoa_seg": "COCOA",
}

# The validation metric that decides which epoch is "best".
BEST_METRIC_FOR_TASK = {
    "classification": "val_acc1",
    "coco_seg": "val_iou",
    "cocoa_seg": "val_iou",
}

# --- COCO mixed into a COCOA refine ------------------------------------------
# The refine sees COCOA alone, and the 2026-09-30 run forgot COCO for it: val
# IoU 69.8 -> 44.5. A ratio above zero interleaves COCO batches into every COCOA
# epoch and scores both validation sets, so the forgetting is visible while it
# happens instead of only afterwards. A ratio of zero is the untouched run.

# The two sources of one mixed epoch's batch schedule.
MIX_SOURCE_PRIMARY = "primary"
MIX_SOURCE_MIX = "mix"

# Validation metric suffixes: the refine's own dataset, and the one mixed in.
# A COCOA refine's own data is COCOA; a BDD refine's is BDD, so its suffix is
# passed through rather than hardcoded -- `val_iou_cocoa` on a BDD run would
# name a dataset the run never scored.
MIX_SUFFIX_PRIMARY = "cocoa"
MIX_SUFFIX_MIX = "coco"
MIX_SUFFIX_BDD = "bdd"

# The names the two datasets are scored under, in the order they are combined.
_MIXED_VAL_METRICS = ("val_loss", "val_iou", "val_dice")


class InitFromError(ValueError):
    """An --init-from warm start that cannot mean anything for its run.

    Its own type, like CocoaMixRatioError, so the command line reports it as a
    usage error rather than catching every ValueError the training stack raises.
    """


class CocoaMixRatioError(ValueError):
    """A COCO mix ratio that cannot mean anything for the run it was given to.

    Its own type so the command line can report it as a usage error -- the
    cause is the invocation -- without catching every ValueError the training
    stack might raise for unrelated reasons.
    """


class AugmentConfigError(ValueError):
    """A --scale-jitter/--color-jitter value that cannot mean anything.

    Its own type, like CocoaMixRatioError, so the command line reports it as a
    usage error (exit 2, no traceback) instead of starting a run and failing
    part-way through the first epoch, when the augmentation is first applied.
    """


class GradAccumError(ValueError):
    """A --grad-accum-steps value that cannot mean anything.

    Its own type for the same reason as AugmentConfigError: the value is the
    invocation's, so it is a usage error rather than a mid-training crash.
    """


class LrDecayError(ValueError):
    """An --lr-decay-start value that cannot be an epoch number.

    Its own type for the same reason as GradAccumError: the value is the
    invocation's, so it is a usage error rather than a mid-training crash.
    """


def lr_decay_start_for(lr_decay_start, *, warmup: bool) -> int:
    """The effective decay start: the flag if given, else today's default.

    Called before any training work so a nonsensical value fails the
    invocation rather than part-way through the first epoch. ``None`` means the
    flag was omitted, and resolves to DEFAULT_DECAY_START with warmup and 0
    without -- the schedule that existed before the flag.
    """
    if lr_decay_start is None:
        return DEFAULT_DECAY_START if warmup else 0
    if isinstance(lr_decay_start, bool) or not isinstance(lr_decay_start, int):
        raise LrDecayError(
            f"--lr-decay-start must be an integer epoch number, got {lr_decay_start!r}"
        )
    if lr_decay_start < 0:
        raise LrDecayError(
            f"--lr-decay-start must be an epoch number >= 0, got {lr_decay_start}"
        )
    return lr_decay_start


def aug_accum_steps(grad_accum_steps) -> int:
    """The accumulation count as a positive int, or ``GradAccumError``.

    Called at the top of training so a nonsensical value fails the invocation
    rather than part-way through the first epoch.
    """
    try:
        steps = int(grad_accum_steps)
    except (TypeError, ValueError):
        raise GradAccumError(
            f"--grad-accum-steps must be an integer, got {grad_accum_steps!r}"
        ) from None
    if steps < 1:
        raise GradAccumError(f"--grad-accum-steps must be at least 1, got {steps}")
    return steps


def validate_augmentation(scale_jitter, color_jitter, *, trains_coco: bool):
    """The jitter settings as usable values, or ``AugmentConfigError``.

    Refused up front, like a bad mix ratio: a range that cannot mean anything
    is the caller's fault, and the jitter is refused on a run whose train
    transform is not the COCO/COCOA one, where it would be silently ignored.

    Returns the parsed ``(scale_jitter, color_jitter)`` so the caller records
    the values the transform actually used.
    """
    if scale_jitter is None and color_jitter is None:
        return None, None
    if not trains_coco:
        raise AugmentConfigError(
            "--aug-scale-jitter/--aug-color-jitter are only implemented for the "
            "COCO/COCOA train transform; this run trains the ImageNet path, where "
            "they would be silently ignored"
        )
    parsed_scale = None
    if scale_jitter is not None:
        try:
            lo, hi = (float(value) for value in scale_jitter)
        except (TypeError, ValueError):
            raise AugmentConfigError(
                f"--aug-scale-jitter must be LO,HI (got {scale_jitter!r})"
            ) from None
        if not (math.isfinite(lo) and math.isfinite(hi)):
            raise AugmentConfigError(
                f"--aug-scale-jitter must be finite (got {lo},{hi})"
            )
        if lo <= 0 or hi <= 0:
            raise AugmentConfigError(
                f"--aug-scale-jitter must be positive (got {lo},{hi})"
            )
        if lo > hi:
            raise AugmentConfigError(
                f"--aug-scale-jitter LO must not exceed HI (got {lo},{hi})"
            )
        parsed_scale = (lo, hi)
    parsed_color = None
    if color_jitter is not None:
        parsed_color = float(color_jitter)
        if not math.isfinite(parsed_color) or parsed_color < 0:
            raise AugmentConfigError(
                f"--aug-color-jitter must be finite and non-negative "
                f"(got {color_jitter!r})"
            )
    return parsed_scale, parsed_color


def _optimizer_step(model, optimizer, scaler) -> None:
    """Clip once and take one optimizer step -- the whole of a group's step.

    With accumulation the group's batches have already summed their gradients,
    so the gradients are clipped once (on the scaled gradients, as before) and
    the optimizer is stepped once, per group, never per batch. There is no
    explicit unscale: clipping scaled gradients is the pre-existing behaviour
    and is left unchanged. Both training branches clipped at the same max_norm,
    so one helper serves both.
    """
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    scaler.step(optimizer)
    scaler.update()


def _augmentation_record(scale_jitter, color_jitter, grad_accum_steps) -> dict:
    """The augmentation/accumulation settings, as recorded in run metadata.

    An ablation run has to be self-describing: two runs of the same task can
    differ only by these, and nothing else in the checkpoint says which is
    which. Plain JSON values (a list, not a tuple) so the metadata round-trips.
    """
    return {
        "scale_jitter": list(scale_jitter) if scale_jitter is not None else None,
        "color_jitter": float(color_jitter) if color_jitter is not None else None,
        "grad_accum_steps": int(grad_accum_steps),
    }


def mix_ratio_for_share(share: float) -> float:
    """A share of the epoch's samples from the mix, as the trainer's mix ratio.

    The trainer's ``coco_mix_ratio`` is extra mix batches *per primary batch*,
    not a share of the total. A share ``s`` needs ``s/(1-s)`` of them, so 0.5 --
    the plan's 50/50 by sample count -- is one for one.
    """
    if share <= 0:
        return 0.0
    return share / (1.0 - share)


def monitored_key_from(monitor: str, *, mixed: bool, best_on_primary: bool,
                       primary_suffix: str = MIX_SUFFIX_PRIMARY) -> str:
    """The validation-metric key a run reads for ``monitor`` this epoch.

    Under a mix, the unsuffixed metric is the mean of the two datasets, which is
    the rule a COCOA refine wants. A run selected on its own dataset (a BDD
    refine) reads the primary's suffixed value instead. Early stopping uses
    this, not the bare name, so the stopper watches the series the best epoch
    is chosen on: stopping on the mean while selection reads BDD alone would let
    a falling COCO IoU cut a BDD refine short while its own IoU was still rising.
    """
    if best_on_primary and mixed:
        return f"{monitor}_{primary_suffix}"
    return monitor


def best_value_from(val_metrics: dict, best_metric_name: str | None,
                    *, mixed: bool, best_on_primary: bool,
                    primary_suffix: str = MIX_SUFFIX_PRIMARY):
    """The value best-epoch selection reads for this epoch. See the key rule in
    ``monitored_key_from``, which the stopper shares."""
    if not best_metric_name:
        return None
    key = monitored_key_from(best_metric_name, mixed=mixed,
                             best_on_primary=best_on_primary,
                             primary_suffix=primary_suffix)
    return val_metrics.get(key)


def dataset_name_for(task: str, coco_mix_ratio: float = 0.0) -> str:
    """The dataset label recorded for a run, naming the mix when there is one."""
    name = DATASET_FOR_TASK.get(task, "unknown")
    if coco_mix_ratio and coco_mix_ratio > 0:
        return f"{name}+{DATASET_FOR_TASK['coco_seg']}"
    return name


def mix_batches_for(n_primary: int, ratio: float) -> int:
    """How many mixed-in batches one epoch of ``n_primary`` batches gets."""
    if ratio <= 0 or n_primary <= 0:
        return 0
    return int(round(ratio * n_primary))


def mix_schedule(n_primary: int, n_mix: int) -> list[str]:
    """One source label per batch slot of a mixed epoch.

    The mixed batches are spread *through* the epoch rather than appended after
    it: by the end of the i-th primary batch, ``round(n_mix * i / n_primary)``
    mixed batches are due in total, and any not yet emitted are emitted there.
    A ratio of 1 alternates the two sources; 0.5 puts one mixed batch after
    every second primary one.
    """
    n_primary = max(0, int(n_primary))
    n_mix = max(0, int(n_mix))
    if n_primary == 0:
        # Degenerate, but defined: with no primary batches the epoch is the mix.
        return [MIX_SOURCE_MIX] * n_mix
    schedule = []
    placed = 0
    for i in range(1, n_primary + 1):
        schedule.append(MIX_SOURCE_PRIMARY)
        due = (n_mix * i + n_primary // 2) // n_primary
        while placed < due:
            schedule.append(MIX_SOURCE_MIX)
            placed += 1
    return schedule


# The per-epoch validation counts: images the collate dropped, COCOA regions
# the mask builder left out of a target, and the amodal-outline accounting
# (images whose hidden part was scored, images whose full-outline target is
# unknown, images that carried a full-outline label). Merged and logged the
# same way, zero included.
_VAL_COUNTS = ("val_failed_images", "val_dropped_regions",
               "val_hidden_images", "val_outline_unknown_images",
               "val_outline_labelled_images")

# The amodal-outline validation numbers. Not a mean across a mixed validation:
# the outline target exists only on COCOA rows, so a COCOA-primary run has them
# on its primary set alone, and a BDD-primary run (whose mix set is MSCOCO too)
# has none anywhere.
_OUTLINE_VAL_METRICS = ("val_full_iou", "val_hidden_iou",
                        "val_full_iou_baseline", "val_hidden_iou_baseline")


def _batch_dropped_regions(batch) -> int:
    """The COCOA regions left out of this batch's targets (0 if not reported)."""
    if not isinstance(batch, dict) or batch.get("dropped_regions") is None:
        return 0
    return int(torch.as_tensor(batch["dropped_regions"]).sum())


def _batch_failed_images(batch, loader_batch_size):
    """The images that failed to load into this segmentation batch.

    The loaders all use drop_last=True, so a batch shorter than the loader's
    batch size is short by exactly its failures, and a batch whose every image
    failed arrives as a one-sample placeholder marked valid=False.
    """
    if not isinstance(batch, dict) or "valid" not in batch:
        return 0
    if not bool(torch.as_tensor(batch["valid"]).any()):
        return loader_batch_size or 1
    if loader_batch_size:
        return max(0, loader_batch_size - int(batch["pixel_values"].shape[0]))
    return 0


def _interleave(primary_loader, mix_loader, schedule):
    """Yield the batches ``schedule`` names, cycling the mix loader as needed."""
    primary_iter = iter(primary_loader)
    mix_iter = iter(mix_loader)
    for source in schedule:
        if source == MIX_SOURCE_PRIMARY:
            try:
                yield next(primary_iter)
            except StopIteration:
                # The primary loader ended before its length said it would; the
                # epoch ends with it, as it did before mixing existed.
                return
            continue
        try:
            batch = next(mix_iter)
        except StopIteration:
            # Shorter than the epoch needs: start it again, so the epoch never
            # runs dry just because COCO is the smaller dataset.
            mix_iter = iter(mix_loader)
            try:
                batch = next(mix_iter)
            except StopIteration:
                raise RuntimeError(
                    "the COCO training loader was non-empty but yielded no batches"
                ) from None
        yield batch


def mixed_epoch_iterator(primary_loader, mix_loader, ratio: float):
    """One epoch's batches: the primary pass with the mix interleaved.

    Returns an iterator over ``len(primary_loader) + round(ratio * n_primary)``
    batches. A zero ratio returns the primary loader untouched, so an unmixed
    run iterates exactly as it did before.
    """
    try:
        n_primary = len(primary_loader)
    except TypeError as exc:
        raise TypeError(
            "a COCO mix ratio needs a sized primary loader to know how many "
            "COCO batches an epoch gets"
        ) from exc
    n_mix = mix_batches_for(n_primary, ratio)
    if n_mix <= 0:
        return iter(primary_loader)
    if len(mix_loader) == 0:
        raise ValueError(
            f"a COCO mix ratio of {ratio} asks for {n_mix} COCO batches per "
            f"epoch, but the COCO training loader is empty"
        )
    return _interleave(primary_loader, mix_loader, mix_schedule(n_primary, n_mix))


def merge_mixed_val_metrics(primary: dict, mixed: dict,
                            *, primary_suffix: str = MIX_SUFFIX_PRIMARY) -> dict:
    """Combine two segmentation validation sets into one metric dict.

    The unsuffixed names become the *mean* of the two sets, which is what a
    COCOA refine is selected and stopped on. A run selected on its primary
    dataset alone (a BDD refine) does not read them: ``monitored_key_from``
    points both best-epoch selection and early stopping at the primary's
    suffixed value. The per-dataset values are kept alongside the mean under
    the suffixed names, which is also what puts both IoUs into the checkpoint
    metadata (built from this same dict).

    The amodal-outline IoUs are the exception to "unsuffixed is the mean". An
    outline target exists only on COCOA rows, and the mix set is always MSCOCO,
    so a COCOA-primary run has them on its primary set alone (and a BDD-primary
    run nowhere): reading either side with ``[name]`` would abort a validation
    for a number it never promised. Each side is written only where it exists,
    and the unsuffixed value is the primary's when it has one, else the mix
    set's.
    ``val_outline_loss`` does follow ``val_loss``'s rule -- the mean of the sets
    that carry the target -- since a loss is comparable across sets.
    """
    merged = {}
    for name in _MIXED_VAL_METRICS:
        merged[f"{name}_{primary_suffix}"] = primary[name]
        merged[f"{name}_{MIX_SUFFIX_MIX}"] = mixed[name]
        merged[name] = (primary[name] + mixed[name]) / 2.0
    # Counts, so the unsuffixed value is the total rather than the mean.
    for count in _VAL_COUNTS:
        if count in primary or count in mixed:
            a = primary.get(count, 0)
            b = mixed.get(count, 0)
            merged[f"{count}_{primary_suffix}"] = a
            merged[f"{count}_{MIX_SUFFIX_MIX}"] = b
            merged[count] = a + b
    # Outline numbers: per-set where present, primary-first unsuffixed.
    for name in _OUTLINE_VAL_METRICS:
        if name in primary:
            merged[f"{name}_{primary_suffix}"] = primary[name]
        if name in mixed:
            merged[f"{name}_{MIX_SUFFIX_MIX}"] = mixed[name]
        if name in primary:
            merged[name] = primary[name]
        elif name in mixed:
            merged[name] = mixed[name]
    if "val_outline_loss" in primary or "val_outline_loss" in mixed:
        present = [d["val_outline_loss"] for d in (primary, mixed)
                   if "val_outline_loss" in d]
        merged["val_outline_loss"] = sum(present) / len(present)
        for label, part in ((primary_suffix, primary), (MIX_SUFFIX_MIX, mixed)):
            if "val_outline_loss" in part:
                merged[f"val_outline_loss_{label}"] = part["val_outline_loss"]
    return merged


scaler = GradScaler()

EARLY_STOP_MONITORS = ("val_loss", "val_iou")


class EarlyStopping:
    """Stop when the monitored validation metric stops improving.

    The default monitor is validation loss, which must fall -- that is the
    behaviour that existed before the monitor was a choice. ``val_iou`` must
    rise instead, so the mode follows the monitor: min for a loss, max for a
    score. ``patience=0`` (or any non-positive value) disables stopping.
    """

    def __init__(self, patience=5, monitor="val_loss"):
        if monitor not in EARLY_STOP_MONITORS:
            raise ValueError(
                f"unknown early-stop monitor {monitor!r}; "
                f"expected one of {EARLY_STOP_MONITORS}")
        self.patience = patience
        self.monitor = monitor
        self.mode = "max" if monitor == "val_iou" else "min"
        # The attribute keeps its old name: it is the best value seen for the
        # monitored metric, whichever metric that is.
        self.best_loss = float("inf") if self.mode == "min" else float("-inf")
        self.counter = 0

    def step(self, current):
        if self.patience <= 0:  # disabled: never stop, whatever the metric does
            return False
        if self.mode == "max":
            improved = current > self.best_loss
        else:
            improved = current < self.best_loss
        if improved:
            self.best_loss = current
            self.counter = 0
            return False
        self.counter += 1
        return self.counter >= self.patience

def __accuracy(output, target, topk=(1,)):
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        return [correct[:k].reshape(-1).float().sum(0).mul_(100.0 / batch_size) for k in topk]


def calculate_iou(pred_mask, gt_mask, threshold=0.5):
    """
    Calculate Intersection over Union (IoU) for segmentation masks.

    Args:
        pred_mask: Predicted logits (B,C,H,W) or (B,1,H,W) or (B,H,W)
        gt_mask: Ground truth mask tensor (B,H,W) or (B,1,H,W) or list/PIL
        threshold: Threshold for binarizing sigmoid predictions (binary case)

    Returns:
        IoU score as a float (0-100 scale)
    """
    with torch.no_grad():
        # ---- normalize gt -> tensor on same device ----
        if not isinstance(gt_mask, torch.Tensor):
            import torchvision.transforms as transforms
            to_tensor = transforms.ToTensor()
            if isinstance(gt_mask, list):
                items = []
                for m in gt_mask:
                    if m is None:
                        continue
                    if hasattr(m, "size"):
                        t = to_tensor(m)  # [1,H,W] float 0..1
                    else:
                        t = m
                    if isinstance(t, torch.Tensor):
                        items.append(t)
                if len(items) == 0:
                    return 0.0
                gt_mask = torch.stack(items, dim=0)
            else:
                gt_mask = to_tensor(gt_mask)

        gt_mask = gt_mask.to(pred_mask.device)

        # squeeze/shape gt to [B,H,W]
        if gt_mask.dim() == 4 and gt_mask.size(1) == 1:
            gt_mask = gt_mask.squeeze(1)
        if gt_mask.dim() == 3:
            pass
        else:
            gt_mask = gt_mask.view(-1, gt_mask.shape[-2], gt_mask.shape[-1])

        # if gt is 0/255 or floats, make binary 0/1
        if gt_mask.dtype != torch.bool:
            gt_bin = (gt_mask > 0.5).to(torch.bool) if gt_mask.is_floating_point() else (gt_mask > 0).to(torch.bool)
        else:
            gt_bin = gt_mask

        # ---- normalize pred -> binary mask [B,H,W] ----
        if pred_mask.dim() == 4 and pred_mask.size(1) > 1:
            # multi-class: argmax of logits (NO sigmoid)
            pred_cls = pred_mask.argmax(dim=1)  # [B,H,W] long
            pred_bin = (pred_cls > 0)
        elif pred_mask.dim() == 4:
            # binary logits [B,1,H,W]
            prob = torch.sigmoid(pred_mask).squeeze(1)
            pred_bin = (prob > threshold)
        else:
            # [B,H,W] logits
            prob = torch.sigmoid(pred_mask)
            pred_bin = (prob > threshold)

        # ---- resize gt to pred if needed (NEAREST) ----
        if gt_bin.shape[-2:] != pred_bin.shape[-2:]:
            gt_bin = F.interpolate(
                gt_bin.unsqueeze(1).float(),
                size=pred_bin.shape[-2:],
                mode="nearest",
            ).squeeze(1).to(torch.bool)

        # special-case: both empty -> perfect
        pred_sum = pred_bin.sum(dim=(1,2))
        gt_sum   = gt_bin.sum(dim=(1,2))
        both_empty = (pred_sum == 0) & (gt_sum == 0)

        # ---- IoU per-batch then mean ----
        inter = (pred_bin & gt_bin).sum(dim=(1,2)).float()
        union = (pred_bin | gt_bin).sum(dim=(1,2)).float()
        iou_per = torch.where(both_empty, torch.ones_like(inter), inter / union.clamp_min(1.0))
        iou = iou_per.mean() * 100.0
        return iou.item()


def calculate_dice_coefficient(pred_mask, gt_mask, threshold=0.5):
    """
    Calculate Dice Coefficient (F1 score) for segmentation masks.

    Args:
        pred_mask: Predicted logits (B,C,H,W) or (B,1,H,W) or (B,H,W)
        gt_mask: Ground truth mask tensor (B,H,W) or (B,1,H,W) or list/PIL
        threshold: Threshold for binarizing sigmoid predictions (binary case)

    Returns:
        Dice coefficient as a float (0-100 scale)
    """
    with torch.no_grad():
        # ---- normalize gt -> tensor on same device ----
        if not isinstance(gt_mask, torch.Tensor):
            import torchvision.transforms as transforms
            to_tensor = transforms.ToTensor()
            if isinstance(gt_mask, list):
                items = []
                for m in gt_mask:
                    if m is None:
                        continue
                    if hasattr(m, "size"):
                        t = to_tensor(m)  # [1,H,W]
                    else:
                        t = m
                    if isinstance(t, torch.Tensor):
                        items.append(t)
                if len(items) == 0:
                    return 0.0
                gt_mask = torch.stack(items, dim=0)
            else:
                gt_mask = to_tensor(gt_mask)

        gt_mask = gt_mask.to(pred_mask.device)

        if gt_mask.dim() == 4 and gt_mask.size(1) == 1:
            gt_mask = gt_mask.squeeze(1)
        if gt_mask.dim() != 3:
            gt_mask = gt_mask.view(-1, gt_mask.shape[-2], gt_mask.shape[-1])

        gt_bin = (gt_mask > 0.5).to(torch.bool) if gt_mask.is_floating_point() else (gt_mask > 0).to(torch.bool)

        # ---- normalize pred -> binary ----
        if pred_mask.dim() == 4 and pred_mask.size(1) > 1:
            pred_cls = pred_mask.argmax(dim=1)
            pred_bin = (pred_cls > 0)
        elif pred_mask.dim() == 4:
            pred_bin = (torch.sigmoid(pred_mask).squeeze(1) > threshold)
        else:
            pred_bin = (torch.sigmoid(pred_mask) > threshold)

        # ---- resize gt to pred if needed (NEAREST) ----
        if gt_bin.shape[-2:] != pred_bin.shape[-2:]:
            gt_bin = F.interpolate(
                gt_bin.unsqueeze(1).float(),
                size=pred_bin.shape[-2:],
                mode="nearest",
            ).squeeze(1).to(torch.bool)

        # special-case: both empty -> perfect
        pred_sum = pred_bin.sum(dim=(1,2))
        gt_sum   = gt_bin.sum(dim=(1,2))
        both_empty = (pred_sum == 0) & (gt_sum == 0)

        inter = (pred_bin & gt_bin).sum(dim=(1,2)).float()
        denom = (pred_sum.float() + gt_sum.float())
        dice_per = torch.where(both_empty, torch.ones_like(inter), (2*inter) / denom.clamp_min(1.0))
        dice = dice_per.mean() * 100.0
        return dice.item()

def dice_loss_from_logits(logits, target, eps=1e-6):
    probs = torch.softmax(logits, dim=1)[:, 1]  # [B,H,W]
    tgt = target.float()
    inter = (probs * tgt).sum(dim=(1,2))
    denom = probs.sum(dim=(1,2)) + tgt.sum(dim=(1,2))
    dice = (2*inter + eps) / (denom + eps)
    return (1 - dice).mean()

def sanity_check_mask(mask_np, name="mask"):
    # mask_np: HxW {0,1}
    import numpy as np

    h, w = mask_np.shape
    frac = mask_np.mean()
    ys, xs = np.where(mask_np > 0)
    if len(xs) == 0:
        print(name, "EMPTY")
        return

    x1, x2 = xs.min(), xs.max()
    y1, y2 = ys.min(), ys.max()
    box_area = (x2 - x1 + 1) * (y2 - y1 + 1)
    print(
        f"{name}: fg_frac={frac:.3f} bbox=({x1},{y1})-({x2},{y2}) "
        f"bbox_area_frac={box_area/(h*w):.3f}"
    )

    # simple heuristic: if fg_frac is huge AND bbox covers almost whole image -> suspicious
    if frac > 0.6 and box_area/(h*w) > 0.9:
        print("⚠️  Mask likely inverted or includes most of the image as FG.")
    

def calculate_segmentation_loss(output, target_masks, device, test_mode=False,
                                on_metric_inputs=None, target_follows_logits=False):
    """
    Segmentation loss + metrics for binary labels.

    Expected:
      - output: [B,C,h,w] logits (C=2 preferred)
      - target_masks: [B,H,W] with {0,1} (int/long/bool/0-255 ok)

    Grid:
      - ``target_follows_logits=False`` (default, today's training path):
        upsample the logits to the target masks' resolution (not the other way
        around) and compute the loss and metrics there.
      - ``target_follows_logits=True``: bring the masks down to the logits'
        own grid with ``nearest`` (labels must not be interpolated) and compute
        the loss and metrics there. This is what lets a stride-2 model be
        trained at 176x176 for a 352 input instead of having its 176x176
        prediction stretched to 352x352. The default stays False so no caller
        changes behaviour unless it asks.

    ``on_metric_inputs``, when given, is called with ``(pred_for_metrics,
    gt_for_metrics)`` -- the logits and the target the IoU/Dice below are
    computed from, on whichever grid the loss is computed on (see above). A
    caller that measures something else about the same prediction (a
    per-object-size breakdown, say) reads those tensors instead of re-deriving
    the prediction, so its numbers are the metric's own rather than a second
    opinion about them. Default None: the training path calls this function
    exactly as it did before.
    """

    if not isinstance(output, torch.Tensor):
        raise TypeError(f"output must be torch.Tensor, got {type(output)}")
    output = output.to(device)

    # ---- build target tensor on device ----
    if isinstance(target_masks, list) and len(target_masks) > 0:
        import torchvision.transforms as transforms
        to_tensor = transforms.ToTensor()

        processed = []
        for m in target_masks:
            if m is None:
                continue
            if hasattr(m, "size"):  # PIL
                t = to_tensor(m)  # [1,H,W] float 0..1
                if t.dim() == 3 and t.size(0) == 1:
                    t = t.squeeze(0)  # [H,W]
            elif isinstance(m, torch.Tensor):
                t = m
                if t.dim() == 3 and t.size(0) == 1:
                    t = t.squeeze(0)
            else:
                continue
            processed.append(t)

        if len(processed) == 0:
            return torch.tensor(0.0, device=device), 0.0, 0.0

        target = torch.stack(processed, dim=0).to(device)  # [B,H,W]
    elif isinstance(target_masks, torch.Tensor):
        target = target_masks.to(device)
        if target.dim() == 4 and target.size(1) == 1:
            target = target.squeeze(1)  # [B,H,W]
    else:
        return torch.tensor(0.0, device=device), 0.0, 0.0

    # Ensure shapes
    if output.dim() == 3:
        output = output.unsqueeze(1)
    if output.dim() != 4:
        raise ValueError(f"output must be [B,C,h,w], got {tuple(output.shape)}")

    if target.dim() != 3:
        target = target.view(-1, target.shape[-2], target.shape[-1])

    # Convert target to {0,1} long
    if target.dtype == torch.bool:
        target_long = target.long()
    else:
        # if 0/255 or 0..1 float, normalize to 0/1
        if target.is_floating_point():
            target_long = (target > 0.5).long()
        else:
            target_long = (target > 0).long()

    B, C, h, w = output.shape
    B.bit_count()
    Ht, Wt = target_long.shape[-2], target_long.shape[-1]

    # ---- the loss is computed on one grid ----
    if target_follows_logits:
        # Train at the model's own resolution: the masks come down to the
        # logits' grid with nearest -- labels must not be interpolated -- so a
        # stride-2 model is trained at 176x176 instead of having its 176x176
        # prediction stretched to the masks' 352x352 for the loss.
        if (h, w) != (Ht, Wt):
            target_long = F.interpolate(target_long.unsqueeze(1).float(), size=(h, w),
                                        mode="nearest").squeeze(1).long()
            Ht, Wt = target_long.shape[-2], target_long.shape[-1]
    elif (h, w) != (Ht, Wt):
        output = F.interpolate(output, size=(Ht, Wt), mode="bilinear", align_corners=False)

    # ---- loss ----
    if C == 1:
        # If you ever use 1-channel logits, use BCEWithLogits
        logits = output[:, 0]  # [B,Ht,Wt]
        loss = F.binary_cross_entropy_with_logits(logits, target_long.float())
        pred_for_metrics = logits
        gt_for_metrics = target_long.float()
    else:
        # C=2 case: CE directly with class ids {0,1}
        if C != 2:
            raise ValueError("Expected C=2 for weighted CE path")
        #logits_fg = output[:, 1, :, :]   # <-- foreground logit only
        #with torch.no_grad():
        #    fg = (target_long == 1).float().mean()
        #    bg = 1.0 - fg
        #    pos_weight = (bg / (fg + 1e-6)).clamp(1.0, 1.5)

        pos_weight = torch.tensor([1.0,1.5], device=output.device)

        loss_ce = F.cross_entropy(output, target_long, weight=pos_weight)
        loss_d  = dice_loss_from_logits(output, target_long)
        probs = torch.softmax(output, dim=1)
        #entropy = -(probs * (probs + 1e-8).log()).sum(dim=1).mean()
        loss = loss_ce + 0.2*loss_d #+ 0.01*entropy
        pred_for_metrics = output
        gt_for_metrics = target_long

    # Debugging info for segmentation outputs
    if test_mode:
        with torch.no_grad():
            probs = torch.softmax(output, dim=1)
            p_fg = probs[:, 1]
            p_fg_mean = float(p_fg.mean().item())
            pred_fg_frac_80 = float((p_fg > 0.8).float().mean().item())
            pred_fg_frac_70 = float((p_fg > 0.7).float().mean().item())
            pred_fg_frac_50 = float((p_fg > 0.5).float().mean().item())
            pred_fg_frac_20 = float((p_fg > 0.2).float().mean().item())
            pred_fg_frac_10 = float((p_fg > 0.1).float().mean().item())
            gt_fg_frac = float((target_long == 1).float().mean().item())
            probs = torch.softmax(output, dim=1)[:, 1]   # [B,H,W]
            gt   = (target_long == 1)

            for t in (0.5, 0.7, 0.8):
                pred = probs > t
                tp = (pred & gt).float().sum()
                fp = (pred & ~gt).float().sum()
                fn = (~pred & gt).float().sum()
                prec = (tp/(tp+fp+1e-6)).item()
                rec  = (tp/(tp+fn+1e-6)).item()
                f1   = (2*prec*rec/(prec+rec+1e-6))
                print(f"@{t} prec={prec:.3f} rec={rec:.3f} f1={f1:.3f}")

            # find best dice over thresholds
            ths = torch.linspace(0.05, 0.95, 19, device=p_fg.device)
            best = (-1, None)
            for t in ths:
                pred = (p_fg > t).to(torch.float32)
                # dice = 2TP/(2TP+FP+FN)
                tp = (pred * gt).sum()
                fp = (pred * (1-gt.int())).sum()
                fn = ((1-pred) * gt).sum()
                dice = (2*tp) / (2*tp + fp + fn + 1e-6)
                if dice.item() > best[0]:
                    best = (dice.item(), float(t))
            print("best_dice", best[0]*100, "best_t", best[1])

            p = torch.sigmoid(output) if output.shape[1] == 1 else torch.softmax(output, dim=1)[:,1]
            y = target_long.float()

            # calibration-ish summary
            fg_p = p[y > 0.5].mean().item() if (y > 0.5).any() else float("nan")
            bg_p = p[y < 0.5].mean().item() if (y < 0.5).any() else float("nan")
            print(f"[VAL] fg_p={fg_p:.3f} bg_p={bg_p:.3f} p_mean={p.mean().item():.3f}")


            probs = torch.softmax(output, dim=1)  # [B,2,H,W]
            p0 = probs[:, 0]
            p1 = probs[:, 1]
            y01 = (target_long > 0)

            def s(p, name):
                fg = p[y01].mean().item() if y01.any() else float("nan")
                bg = p[~y01].mean().item() if (~y01).any() else float("nan")
                print(f"{name}: fg={fg:.3f} bg={bg:.3f} mean={p.mean().item():.3f}")

            s(p0, "p(class0)")
            s(p1, "p(class1)")
            print("gt_fg_frac:", y01.float().mean().item())

            y = target_long
            print("target unique:", torch.unique(y).tolist())
            print("gt_fg_frac:", (y==1).float().mean().item())
            # per-sample foreground fraction (catches “all ones” or “all zeros” samples)
            frac = (y==1).float().flatten(1).mean(dim=1)
            print("gt_fg_frac per sample:", frac[:8].tolist())

            #m = target_long[0].detach().cpu().numpy().astype("uint8")
            #sanity_check_mask(m, f"sample")

            if output.shape[1] == 2:
                logit_fg = output[:, 1] - output[:, 0]   # log-odds for FG vs BG, shape [B,H,W]
            else:
                logit_fg = output[:, 0]

            y = target_long.float()
            fg_logit = logit_fg[y > 0.5].mean().item() if (y > 0.5).any() else float("nan")
            bg_logit = logit_fg[y < 0.5].mean().item() if (y < 0.5).any() else float("nan")
            print(f"logit_fg: fg={fg_logit:.3f} bg={bg_logit:.3f} mean={logit_fg.mean().item():.3f}")

        print(
            f"mean p_fg={p_fg_mean:.4f} "
            f"pred_fg_frac@.8={pred_fg_frac_80:.4f} "
            f"@.7={pred_fg_frac_70:.4f} "
            f"@.5={pred_fg_frac_50:.4f} "
            f"@.2={pred_fg_frac_20:.4f} "
            f"@.1={pred_fg_frac_10:.4f} "
            f"gt_fg_frac={gt_fg_frac:.4f}"
        )

    # ---- metrics ----
    if on_metric_inputs is not None:
        # Read the tensors the metric is about to consume, not the model's
        # output: they are the logits and the target on one grid, whichever of
        # the two the grid was chosen from.
        on_metric_inputs(pred_for_metrics, gt_for_metrics)
    iou_score, dice_score = iou_dice_fg_from_logits(pred_for_metrics, gt_for_metrics)
    if test_mode:
        soft_dice = soft_dice_fg_from_logits(pred_for_metrics, gt_for_metrics)
        print(f"Segmentation Metrics - IoU: {iou_score*100:.2f}, Dice: {dice_score*100:.2f}, Soft Dice: {soft_dice*100:.2f}")
    return loss, iou_score, dice_score


def set_requires_grad(module, flag: bool):
    for p in module.parameters():
        p.requires_grad = flag

def freeze_all_backbone(backbone):
    set_requires_grad(backbone, False)

def unfreeze_backbone_stages(backbone, stages):
    """
    stages: iterable of {"layer4","layer3","layer2","layer1","conv1","bn1"}
    """
    for s in stages:
        if hasattr(backbone, s):
            set_requires_grad(getattr(backbone, s), True)

def backbone_bn_eval(backbone):
    # Keep BN stable during finetune (often helps a lot).
    #
    # Both batch-norm families are matched: BatchNorm2d and SyncBatchNorm are
    # siblings under torch.nn.modules.batchnorm._BatchNorm, so the plain
    # `isinstance(m, torch.nn.BatchNorm2d)` this walk used to be is False for a
    # SyncBatchNorm layer. A model converted with convert_sync_batchnorm would
    # then have kept its running statistics updating through the stages that
    # mean to pin them.
    for m in backbone.modules():
        if isinstance(m, (torch.nn.BatchNorm2d, torch.nn.SyncBatchNorm)):
            m.eval()


def _bind_batch(on_metric_inputs, batch):
    """``on_metric_inputs`` with this batch bound as its third argument."""
    return lambda logits, target: on_metric_inputs(logits, target, batch)


def target_grid_follows_logits(model):
    """Whether the segmentation loss and IoU/Dice are computed on the model's
    own logits grid rather than on the target masks' grid.

    True for a model whose mask output is finer than the masks: ``mask_stride``
    2 at a 352 input is a 176x176 logit map, so the masks come down to 176 with
    ``nearest`` and the loss, the IoU/Dice and the ``on_metric_inputs`` tensors
    are all computed there. The stride-2 prediction is then trained and scored
    at its own resolution instead of having its 176x176 map stretched to the
    masks' 352x352.

    The stride-4 default -- today's model, the live ResNet-50 checkpoints -- is
    deliberately left on the old grid: a 4x nearest-downsample of the masks to
    88x88 would drop small instances out of the ground truth, which is the one
    thing this model is measured on, so its training, its checkpoints and its
    recorded ``val_iou`` stay exactly as they are.

    The read goes through the DDP wrapper: the epoch loop rebinds ``model`` to
    the wrapper (see ``wrap_model_for_ddp``) and this helper is called from the
    loop, and a ``DistributedDataParallel`` has no ``__getattr__`` to delegate
    with, so a bare read would fall back to the default 4 on every multi-rank
    run -- a stride-2 run would train and score its loss and IoU on the masks'
    grid while its own ``evaluate`` (which unwraps) scored the logits' grid.
    """
    return int(getattr(unwrap_or_self(model), "mask_stride", MASK_STRIDE_DEFAULT)) == 2


@torch.no_grad()
def evaluate(model, val_loader, device, max_batches=0, is_segmentation=False,
             use_amp=True, on_metric_inputs=None):
    """Validate ``model`` on ``val_loader``.

    ``on_metric_inputs``, when given, is called once per scored segmentation
    batch as ``on_metric_inputs(logits, target, batch)`` with the tensors
    ``calculate_segmentation_loss`` computes the IoU/Dice from -- the same
    forward pass, on whichever grid the loss itself uses, which
    ``target_grid_follows_logits(model)`` decides: the masks' grid for a
    stride-4 model, the logits' own for a stride-2 one -- so a caller's
    breakdown of that prediction is the metric's own and not a second
    measurement. Default None, i.e. the training path is unchanged. The batch is
    passed so the caller can align per-image work with ``batch["image_id"]``.
    """
    model.eval()

    # use_amp=False runs the forward in fp32: autocast cannot be switched off
    # from outside, because the autocast() opened here would re-enable it.
    running_loss = 0.0
    running_acc1 = 0.0
    running_acc5 = 0.0
    n = 0
    presented = 0
    # Images the collate dropped because they failed to load or transform.
    # The val loaders all use drop_last=True, so any batch shorter than the
    # loader's batch size is short by exactly its failures, and a batch whose
    # every image failed arrives as a one-sample placeholder marked
    # valid=False -- a blank that, scored, would average into val_iou as data.
    loader_batch_size = getattr(val_loader, "batch_size", None)
    failed_images = 0
    dropped_regions = 0
    # The outline terms, when the loader carries them. A batch without the key
    # is a run (or a set) with no amodal labels and contributes nothing.
    outline_loss_sum = 0.0
    outline_loss_batches = 0
    outline_labelled = 0
    outline_unknown = 0
    outline_sums = {}
    outline_metric_batches = 0

    for batch_idx, batch in enumerate(val_loader):
        if max_batches and batch_idx >= max_batches:
            break

        # Skip None batches that may come from collate_fn when data is invalid
        if batch is None:
            continue

        if is_segmentation and isinstance(batch, dict) and "valid" in batch:
            failed_images += _batch_failed_images(batch, loader_batch_size)
            if not bool(torch.as_tensor(batch["valid"]).any()):
                continue

        presented += 1
        if is_segmentation:
            dropped_regions += _batch_dropped_regions(batch)

        if is_segmentation:
            # COCO Segmentation Validation with Binary Masks
            if isinstance(batch, dict):
                images = batch["pixel_values"].to(device, non_blocking=True)
                segmentation_masks = batch["segmentation_mask"].to(device, non_blocking=True)
            else:
                images, segmentation_masks = batch
                images = images.to(device, non_blocking=True)
                segmentation_masks = segmentation_masks.to(device, non_blocking=True)

            # Validation never skips a batch: a check that silently covers a
            # subset of the held-out set reports a healthy-looking number that
            # is not the number it claims to be. An OOM here is fatal, and it
            # names the batch so the size at which it happened is recoverable.
            outline_batch = isinstance(batch, dict) and "outline_mask" in batch
            follows = target_grid_follows_logits(model)

            try:
                with autocast(enabled=use_amp):
                    if outline_batch:
                        # One decoder pass, two heads: the outline logits come
                        # from the same forward the IoU below is scored on.
                        output, outline_logits = model(images, return_outline=True)
                    else:
                        output = model(images)
                    batch_metric_inputs = None
                    if on_metric_inputs is not None:
                        # Bind this batch, so the callback can see its image_ids.
                        batch_metric_inputs = _bind_batch(on_metric_inputs, batch)
                    loss, iou_score, dice_score = calculate_segmentation_loss(
                        output, segmentation_masks, device,
                        on_metric_inputs=batch_metric_inputs,
                        target_follows_logits=follows,
                    )
                    if outline_batch:
                        outline_target = batch["outline_mask"].to(device, non_blocking=True)
                        has_full = batch["has_full_outline"]
                        outline_loss, outline_labelled_batch = outline_loss_from_logits(
                            outline_logits, outline_target, has_full,
                            target_follows_logits=follows,
                        )
                        # Measured outside autocast: this is a metric, not a
                        # gradient, and a half-precision activation probability
                        # is not what `val_iou` reports either.
                        with autocast(enabled=False):
                            batch_outline_metrics = outline_metrics_from_logits(
                                outline_logits.float(), output.float(), outline_target,
                                segmentation_masks, has_full,
                                target_follows_logits=follows,
                            )
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    raise RuntimeError(
                        f"validation OOM at batch {batch_idx}: {str(e)[:200]}"
                    ) from e
                raise

            running_loss += loss.item()
            running_acc1 += iou_score
            running_acc5 += dice_score
            n += 1

            if outline_batch:
                outline_loss_sum += float(outline_loss.item())
                outline_loss_batches += 1
                outline_labelled += int(outline_labelled_batch)
                outline_unknown += int(torch.as_tensor(
                    batch["outline_unknown_images"]).sum())
                if batch_outline_metrics:
                    outline_metric_batches += 1
                    for key, value in batch_outline_metrics.items():
                        outline_sums[key] = outline_sums.get(key, 0.0) + float(value)
        else:
            # Standard Classification Validation
            if isinstance(batch, dict):
                images = batch["pixel_values"].to(device, non_blocking=True)
                target = batch["label"].to(device, non_blocking=True)
            else:
                images, target = batch
                images = images.to(device, non_blocking=True)
                target = target.to(device, non_blocking=True)

            try:
                with autocast(enabled=use_amp):
                    output = model(images)
                    loss = F.cross_entropy(output, target)
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    raise RuntimeError(
                        f"validation OOM at batch {batch_idx}: {str(e)[:200]}"
                    ) from e
                raise

            acc1, acc5 = __accuracy(output, target, topk=(1, 5))

            running_loss += loss.item()
            running_acc1 += acc1.item()
            running_acc5 += acc5.item()
            n += 1

    # A validation that validated nothing must not average to a number. Without
    # this, `running_loss / max(n, 1)` reports 0.0 for a wholly skipped epoch --
    # a value that reads as a perfect result.
    if n == 0:
        raise RuntimeError(
            f"validation ran 0 batches out of {presented} presented "
            f"({failed_images} images failed to load); there is no validation "
            f"metric to report. The validation loader is empty or every batch "
            f"was discarded."
        )

    _warn_if_partial_validation(n, presented)

    metric_name_1 = "val_iou" if is_segmentation else "val_acc1"
    metric_name_2 = "val_dice" if is_segmentation else "val_acc5"

    # Percentages out of evaluate() are always 0-100, for both tasks. The
    # segmentation path accumulates raw fractions (iou_score/dice_score), while
    # __accuracy already returns percentages -- scale the fractions here so
    # every consumer sees one unit and nobody multiplies by 100 again.
    scale = 100.0 if is_segmentation else 1.0

    metrics = {
        "val_loss": running_loss / n,
        metric_name_1: (running_acc1 / n) * scale,
        metric_name_2: (running_acc5 / n) * scale,
    }
    if is_segmentation:
        metrics["val_failed_images"] = failed_images
        metrics["val_dropped_regions"] = dropped_regions
        if outline_loss_batches:
            # The loss only where the set carries at least one label. The term
            # is a masked mean, so an unlabelled set's is exactly 0, and a
            # mixed validation would average that zero in as if the set had a
            # target (it exists only on some sets). The counts are logged
            # either way, zero included.
            if outline_labelled:
                metrics["val_outline_loss"] = outline_loss_sum / outline_loss_batches
            metrics["val_outline_labelled_images"] = outline_labelled
            metrics["val_outline_unknown_images"] = outline_unknown
            for key, total in outline_sums.items():
                if key == "val_hidden_images":
                    metrics[key] = int(total)
                else:
                    metrics[key] = total / outline_metric_batches
            # Every count every epoch, zero included: a labelled set whose
            # hidden targets were all empty still reports the zero.
            metrics.setdefault("val_hidden_images", 0)
    return metrics

def iou_dice_fg_from_pred(pred_fg, tgt_fg, eps=1e-6, ignore_empty_gt=False):
    """IoU and Dice of two already-thresholded boolean masks, averaged per image.

    ``pred_fg`` and ``tgt_fg`` are ``[B,H,W]`` bool tensors on one grid. This is
    the per-image core every IoU/Dice number in this module goes through, so a
    metric scored on its own prediction (the outline terms) is the same
    arithmetic as ``val_iou``.
    """
    if not torch.is_tensor(tgt_fg):
        tgt_fg = torch.as_tensor(tgt_fg)
    tgt_fg = tgt_fg.to(dtype=torch.bool)
    pred_fg = pred_fg.to(dtype=torch.bool)

    # ---- optionally ignore empty-GT images ----
    gt_sum = tgt_fg.flatten(1).sum(dim=1)            # [B]
    if ignore_empty_gt:
        valid = gt_sum > 0
        if valid.sum() == 0:
            return 0.0, 0.0
        pred_fg = pred_fg[valid]
        tgt_fg  = tgt_fg[valid]

    # ---- IoU/Dice per image, then average ----
    inter = (pred_fg & tgt_fg).flatten(1).sum(dim=1).float()
    union = (pred_fg | tgt_fg).flatten(1).sum(dim=1).float()
    pred_sum = pred_fg.flatten(1).sum(dim=1).float()
    tgt_sum  = tgt_fg.flatten(1).sum(dim=1).float()

    iou  = (inter + eps) / (union + eps)
    dice = (2*inter + eps) / (pred_sum + tgt_sum + eps)

    return float(iou.mean().item()), float(dice.mean().item())


def iou_dice_fg_from_logits(logits, target, thresh=0.5, eps=1e-6, ignore_empty_gt=False):
    """
    logits:
      - [B,2,H,W] (two-class logits: bg, fg), OR
      - [B,1,H,W] / [B,H,W] (foreground logits)
    target: [B,H,W] with {0,1}
    thresh: threshold on foreground probability
    ignore_empty_gt: if True, skip images with no FG in GT (common for COCO)
    """
    if target.dtype != torch.bool:
        tgt_fg = (target == 1)
    else:
        tgt_fg = target

    # ---- get foreground probability ----
    if logits.ndim == 4 and logits.size(1) == 2:
        # two-class: softmax and take fg channel
        p_fg = F.softmax(logits, dim=1)[:, 1]          # [B,H,W]
    elif logits.ndim == 4 and logits.size(1) == 1:
        p_fg = torch.sigmoid(logits[:, 0])            # [B,H,W]
    elif logits.ndim == 3:
        p_fg = torch.sigmoid(logits)                  # [B,H,W]
    else:
        raise ValueError(f"Unexpected logits shape: {tuple(logits.shape)}")

    return iou_dice_fg_from_pred(
        (p_fg > thresh), tgt_fg, eps=eps, ignore_empty_gt=ignore_empty_gt)

# Weight of the dice half of the outline term, matching the 0.2 the visible
# loss gives its own dice.
OUTLINE_DICE_WEIGHT = 0.2


def _on_one_grid(pred, target, target_follows_logits):
    """Put a prediction and its target on one grid, by the segmentation loss's rule.

    ``target_follows_logits`` sends the target down to the prediction's grid with
    nearest (a label is never interpolated); otherwise the prediction is stretched
    to the target's grid with bilinear.
    """
    if pred.shape[-2:] == target.shape[-2:]:
        return pred, target
    if target_follows_logits:
        target = F.interpolate(target.unsqueeze(1).float(), size=pred.shape[-2:],
                               mode="nearest").squeeze(1).long()
    else:
        pred = F.interpolate(pred, size=target.shape[-2:], mode="bilinear",
                             align_corners=False)
    return pred, target


def outline_loss_from_logits(outline_logits, outline_masks, has_full_outline,
                             target_follows_logits=False):
    """The amodal-outline term: per-sample BCE-with-logits + 0.2*dice, averaged
    over the samples whose full-outline target is known.

    Returns ``(loss, labelled)``. With no labelled sample the loss is 0 but is
    still a function of ``outline_logits`` (the per-sample terms are multiplied
    by the zero mask), so the head stays in the graph and DDP does not report
    it unused. A sample without a known target contributes nothing, never a
    zero target standing in for a label.
    """
    if outline_logits.dim() == 3:
        outline_logits = outline_logits.unsqueeze(1)
    target = outline_masks
    if target.dim() == 4 and target.size(1) == 1:
        target = target.squeeze(1)
    target = (target > 0).long()
    logits, target = _on_one_grid(outline_logits, target, target_follows_logits)
    logits = logits[:, 0].float()
    y = target.float()

    bce = F.binary_cross_entropy_with_logits(logits, y, reduction="none")
    bce = bce.flatten(1).mean(dim=1)                     # [B]
    prob = torch.sigmoid(logits)
    inter = (prob * y).flatten(1).sum(dim=1)
    denom = prob.flatten(1).sum(dim=1) + y.flatten(1).sum(dim=1)
    dice = 1.0 - (2 * inter + 1e-6) / (denom + 1e-6)     # [B]
    per_sample = bce + OUTLINE_DICE_WEIGHT * dice

    m = torch.as_tensor(has_full_outline, device=logits.device,
                        dtype=per_sample.dtype)
    labelled = int(m.sum().item())
    loss = (per_sample * m).sum() / m.sum().clamp(min=1.0)
    return loss, labelled


def outline_metrics_from_logits(outline_logits, visible_logits, outline_target,
                                visible_target, has_full_outline,
                                target_follows_logits=False):
    """The outline validation numbers, scored on the labelled images only.

    ``outline_target`` is the amodal target and ``visible_target`` the visible
    one (``segmentation_mask``), both ``[B,H,W]``; ``visible_logits`` are the
    2-class segmentation logits from the same forward pass, so the baselines
    score exactly the prediction ``val_iou`` scores. All four use the
    ``val_iou`` skip rule, ``ignore_empty_gt=True``:

    - ``val_full_iou``: ``sigmoid(outline) > 0.5`` against the amodal target.
    - ``val_hidden_iou``: ``outline_pred AND NOT visible_pred`` against
      ``outline_target AND NOT visible_target``.
    - ``val_full_iou_baseline``: the visible prediction against the amodal target.
    - ``val_hidden_iou_baseline``: a 9x9 dilation of the visible prediction,
      outside it, against the hidden target -- a real guess at the hidden part.

    Returns an empty dict when no labelled image is in the batch, else the four
    numbers plus ``val_hidden_images`` (the images whose hidden target was not
    empty, and so the ones ``val_hidden_iou`` averaged over).
    """
    if outline_logits.dim() == 3:
        outline_logits = outline_logits.unsqueeze(1)
    if outline_target.dim() == 4 and outline_target.size(1) == 1:
        outline_target = outline_target.squeeze(1)
    if visible_target.dim() == 4 and visible_target.size(1) == 1:
        visible_target = visible_target.squeeze(1)
    outline_target = (outline_target > 0)
    visible_target = (visible_target > 0)

    ol, ot = _on_one_grid(outline_logits, outline_target.long(), target_follows_logits)
    vl, vt = _on_one_grid(visible_logits, visible_target.long(), target_follows_logits)
    ot = ot.bool()
    vt = vt.bool()

    m = torch.as_tensor(has_full_outline, device=ol.device, dtype=torch.bool)
    if m.dim() == 0:
        m = m.repeat(ol.shape[0])
    if not bool(m.any()):
        return {}

    pred_full = (torch.sigmoid(ol[:, 0]) > 0.5)[m]
    p_visible = F.softmax(vl, dim=1)[:, 1] if vl.size(1) == 2 else torch.sigmoid(vl[:, 0])
    pred_visible = (p_visible > 0.5)[m]
    gt_full = ot[m]
    gt_visible = vt[m]

    gt_hidden = gt_full & ~gt_visible
    pred_hidden = pred_full & ~pred_visible
    dilated = F.max_pool2d(pred_visible.float().unsqueeze(1).contiguous(),
                           kernel_size=9, stride=1, padding=4).squeeze(1) > 0.5
    pred_hidden_baseline = dilated & ~pred_visible

    hidden_images = int(gt_hidden.flatten(1).sum(dim=1).gt(0).sum().item())

    full_iou, _ = iou_dice_fg_from_pred(pred_full, gt_full, ignore_empty_gt=True)
    hidden_iou, _ = iou_dice_fg_from_pred(pred_hidden, gt_hidden, ignore_empty_gt=True)
    full_base, _ = iou_dice_fg_from_pred(pred_visible, gt_full, ignore_empty_gt=True)
    hidden_base, _ = iou_dice_fg_from_pred(pred_hidden_baseline, gt_hidden,
                                           ignore_empty_gt=True)

    return {
        "val_full_iou": full_iou * 100.0,
        "val_hidden_iou": hidden_iou * 100.0,
        "val_full_iou_baseline": full_base * 100.0,
        "val_hidden_iou_baseline": hidden_base * 100.0,
        "val_hidden_images": hidden_images,
    }


def soft_dice_fg_from_logits(logits, target, eps=1e-6):
    probs = torch.softmax(logits, dim=1)[:, 1]  # [B,H,W]
    tgt = target.float()
    valid = (tgt.flatten(1).sum(dim=1) > 0)
    if valid.sum() == 0:
        return 0.0
    probs = probs[valid]
    tgt = tgt[valid]
    inter = (probs * tgt).flatten(1).sum(dim=1)
    denom = probs.flatten(1).sum(dim=1) + tgt.flatten(1).sum(dim=1)
    dice = (2 * inter + eps) / (denom + eps)
    return float(dice.mean().item())



def _as_json_number(value):
    """Coerce a metric to a JSON-safe number, leaving anything else as a string.

    Checkpoint metadata is written as JSON; a value that is not numeric must not
    blow up the save path (which, on the interruption path, would mask the real
    error that is already being handled).
    """
    if value is None or isinstance(value, (int, float, str, bool)):
        return value
    try:
        return float(value)
    except (TypeError, ValueError):
        return str(value)


def init_from_of(model):
    """The checkpoint a warm-started model was initialised from (None otherwise)."""
    return getattr(getattr(model, "module", model), "init_from", None)


def warm_start_from(model, init_from, logger):
    """Copy the checkpoint ``init_from`` names into the fresh ``model``.

    Every tensor whose name and shape still fit is copied; a tensor whose shape
    changed, or one only the model has (v2's fuse convs), keeps its init. The
    model is marked as a new lineage: ``init_from`` set, ``resumed_from`` None.
    A source that shares nothing with the model is refused, since the run would
    be a fresh model carrying a misleading ``init_from``.
    """
    warm = warm_start_state(model, download_checkpoint_state(init_from))
    if not warm["copied"]:
        raise InitFromError(
            f"--init-from {init_from!r} shares no tensor of a matching name and "
            f"shape with the model; nothing would be warm-started"
        )
    target = getattr(model, "module", model)
    target.init_from = init_from
    target.resumed_from = None
    print(
        f"Warm start from {init_from} into decoder {decoder_arch_of(model)}: "
        f"{len(warm['copied'])} tensors copied, {len(warm['skipped'])} "
        f"skipped (shape changed), {len(warm['initialised'])} initialised "
        f"(fuse [I|0]), {len(warm['new'])} new (left at init), "
        f"{len(warm['dropped'])} unused from the source; "
        f"{warm['copied_fraction']:.1%} of the model's elements copied"
    )
    if warm["skipped"]:
        print(f"  skipped: {warm['skipped']}")
    if warm["initialised"]:
        print(f"  initialised: {warm['initialised']}")
    print(f"  new: {warm['new']}")
    for name, value in (
        ("init_copied_keys", len(warm["copied"])),
        ("init_skipped_keys", len(warm["skipped"])),
        ("init_initialised_keys", len(warm["initialised"])),
        ("init_new_keys", len(warm["new"])),
        ("init_copied_fraction", warm["copied_fraction"]),
    ):
        try:
            logger.log_scalar(name, value, step=0)
        except Exception:  # pylint: disable=broad-exception-caught
            print(f"{Fore.RED}[ERROR] Could not log {name}: {traceback.format_exc()}{Style.RESET_ALL}")
    return warm


def resumed_from_of(model):
    """The checkpoint key ``model`` was resumed from, or None for a fresh model."""
    return getattr(getattr(model, "module", model), "resumed_from", None)


def out_stride_of(model):
    """The backbone stride ``model`` was built with, or None if it records none."""
    stride = getattr(getattr(model, "module", model), "out_stride", None)
    return int(stride) if stride is not None else None


def decoder_arch_of(model):
    """The decoder version; a model that records none is the original "v1"."""
    return getattr(getattr(model, "module", model), "decoder_arch", None) or DECODER_V1


# --- Learning-rate decay after warmup ---------------------------------------
# The real run plateaued after epoch 5: warmup sets the backbone lr by stage
# (0.0, 1e-4, 1e-4, 5e-5, 5e-5) and then nothing ever changed either group, so
# both sat flat for the rest of the run. This decays each group from whatever it
# holds when the decay phase starts down to MIN_LR_FACTOR of that value.
#
# The warmup stages occupy epochs 0-4, so decay starts at epoch 5 when warmup is
# on and at epoch 0 when it is off. --lr-decay-start moves it; a start inside
# warmup tapers each stage's value rather than replacing it (staged_lr_for_epoch).
WARMUP_EPOCHS = 5
DEFAULT_DECAY_START = WARMUP_EPOCHS
MIN_LR_FACTOR = 0.05


def save_epoch_checkpoint(model, task, epoch, val_metrics, avg_loss,
                          val_loss, run_id=None, is_best=False, is_last=False,
                          is_smoke=False, input_size=DEFAULT_INPUT_SIZE,
                          coco_mix_ratio: float = 0.0,
                          augmentation: dict | None = None,
                          lr_decay_start: int = DEFAULT_DECAY_START,
                          dataset_name: str | None = None):
    """Persist one epoch's checkpoint to that task's bucket.

    Metadata carries the run id, task, dataset, this epoch's val metrics, both
    losses, the git SHA of the image, the best/last flags, the model input size,
    the COCO mix ratio, the augmentation/gradient-accumulation settings and the
    lr decay start, so a
    checkpoint is self-describing without cross-referencing MLflow. Under a mix
    the dataset name names both datasets and ``val_metrics`` carries each one's
    IoU as well as their mean.
    """
    bucket, model_name = checkpoint_target(task)
    metadata = {
        "mlflow_run_id": run_id,
        "task": task,
        "dataset_name": dataset_name or dataset_name_for(task, coco_mix_ratio),
        "coco_mix_ratio": float(coco_mix_ratio),
        # The augmentation and gradient-accumulation settings this epoch trained under.
        # An ablation differs from its baseline only here, so the checkpoint has
        # to carry them or the two are indistinguishable afterwards.
        "augmentation": dict(augmentation) if augmentation else _augmentation_record(None, None, 1),
        # The epoch the lr decay starts at, so two checkpoints differing only in
        # the schedule are distinguishable afterwards.
        "lr_decay_start": int(lr_decay_start),
        "epoch": epoch,
        # The spatial size the run trained at, so a consumer (the Jetson demo)
        # reads it rather than assuming 192.
        "input_size": int(input_size),
        # The parent checkpoint (null for a fresh model), so a later rule can
        # tell a fresh start from a continuation.
        "resumed_from": resumed_from_of(model),
        # The checkpoint a warm start copied its weights from (null otherwise):
        # a new lineage, not a continuation, so it never appears as resumed_from.
        "init_from": init_from_of(model),
        # The backbone stride, so a consumer rebuilds the same architecture
        # instead of assuming 16.
        "out_stride": out_stride_of(model),
        # The decoder version ("v1" or "v2"), so a consumer
        # rebuilds the decoder the weights were trained in.
        "decoder_arch": decoder_arch_of(model),
        # The ladder and the mask branch (backbone, neck_channels,
        # mask_stride), so a consumer rebuilds the architecture the weights were
        # trained in. The model's own record, so it cannot drift from the model
        # config; a model that records none contributes nothing.
        **arch_record_of(model),
        # val_metrics is usually {name: number}, but the interrupted-save path
        # passes descriptive entries (e.g. "interruption_reason"), so coerce
        # only what is numeric rather than assuming every value will convert.
        "val_metrics": {k: _as_json_number(v) for k, v in val_metrics.items()},
        "train_loss": _as_json_number(avg_loss),
        "val_loss": _as_json_number(val_loss),
        "git_commit": os.environ.get("GIT_SHA", "unknown"),
        "is_best": bool(is_best),
        "is_last": bool(is_last),
        "is_smoke": bool(is_smoke),
    }
    uploader = ModelUploaderAndDownloader(bucket_name=bucket)
    return uploader.upload_model(
        model=model,
        model_name=f"{model_name}_epoch{epoch}",
        architecture="hcm",
        metadata=metadata,
    )


def count_trainable_params(model) -> tuple[int, int]:
    """(trainable, total) parameter counts for the current freeze state."""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total


def print_trainable_params(model, when: str = None) -> None:
    """Report the trainable/total split for the CURRENT freeze state.

    Called after freezing rather than before it: counting at the top of
    train_model_for_epochs reported the pre-freeze (unfrozen backbone) numbers,
    so the figure was wrong for the configuration training actually started in
    -- and the warmup path re-freezes per epoch, which is the thing the owner
    wants to watch change.
    """
    trainable, total = count_trainable_params(model)
    pct = (100 * trainable / total) if total else 0.0
    suffix = f" ({when})" if when else ""
    print(f"  Trainable Parameters{suffix}: {trainable:,} / {total:,} ({pct:.2f}%)")


# How often the running activation maxima are flushed to MLflow, in batches.
# The model accumulates a max across forwards, so this is the *logging* cadence,
# not the measurement cadence: a spike anywhere in the window is still caught.
ACTIVATION_LOG_EVERY = 200

# MLflow metric keys for the activation magnitudes. These are raw magnitudes,
# not percentages, so they go through logger.log_scalar (which skips the 0-100
# guard) rather than the data_obj/track path. Kept in step with the model's
# hcm_nn.ACTIVATION_STAGES: "isc3" is isc_output[3], the only ISC output that
# reaches the decoder and therefore the one the fp16 boundary depends on.
_ACTIVATION_STAGES = ("isc", "csc", "isc3")

# --- Out-of-memory circuit breakers -----------------------------------------
# The V100 run `real-aa519f5` was double-booked onto a GPU another workload
# held, so every one of its 14,664 training batches raised CUDA OOM. The
# per-batch handlers printed and `continue`d, which meant the epoch "completed"
# having trained nothing and the run looked healthy. These limits turn that
# into a loud failure.
MAX_CONSECUTIVE_OOM_SKIPS = 20
MAX_OOM_SKIP_FRACTION = 0.01
# The fraction is only judged once this many batches have been attempted: at
# 1/1 a single early skip is 100% and would kill a healthy run.
MIN_BATCHES_FOR_OOM_FRACTION = 200
# A run with less than this much free GPU memory cannot train at all, and
# starting anyway costs hours of failed batches. Refused up front with numbers.
MIN_FREE_GPU_GIB = 8.0

def lr_decay_factor(epoch: int, decay_start: int, epochs: int, f: float) -> float:
    """The cosine multiplier both lr functions apply, in ``[f, 1]``.

    1.0 before the decay starts (``k < 0``), 1.0 when the decay phase is empty
    (``K <= 0``) or a single epoch (``K == 1``, where ``K - 1`` would divide by
    zero), and otherwise the ramp from 1.0 at the first decay epoch down to
    ``f`` at the last:

        factor_k = f + (1 - f) * 0.5 * (1 + cos(pi * k / (K - 1)))

    for k = 0..K-1, where K is the number of decay epochs. Defined once so the
    two schedules below cannot drift apart.
    """
    k = epoch - decay_start
    K = epochs - decay_start
    if K <= 0 or k < 0:
        return 1.0
    if K == 1:
        return 1.0
    return f + (1.0 - f) * 0.5 * (1.0 + math.cos(math.pi * k / (K - 1)))


def lr_for_epoch(base: float, epoch: int, decay_start: int, epochs: int, f: float) -> float:
    """The learning rate for ``epoch``, given the decay phase ``base`` lr.

    Cosine decay of one param group from ``base`` at the first decay epoch down
    to ``base * f`` at the last: ``base * lr_decay_factor(epoch, ...)``.

    Everything before ``decay_start`` is returned as ``base`` unchanged: during
    warmup the caller owns the value (it re-sets the backbone lr by stage before
    this is called), and epochs before decay_start but after warmup do not exist
    by construction. ``K == 1`` returns ``base`` (the single decay epoch is both
    the first and the last) and ``K <= 0`` (a run that ends during or before
    warmup) returns ``base`` too, so such a run is bit-for-bit what it was
    before this existed.

    Pure by design: no optimizer, no env, no global state, so the schedule can
    be checked against its own definition without building a model.
    """
    return base * lr_decay_factor(epoch, decay_start, epochs, f)


def staged_lr_for_epoch(stage_lr: float, epoch: int, decay_start: int,
                        epochs: int, f: float) -> float:
    """The lr a group runs at when the warmup stage owns it and decay has begun.

    The decay factor is applied on top of what the warmup stage sets for the
    epoch: ``stage_lr * lr_decay_factor(epoch, ...)``, with the same cosine and
    minimum-lr factor as lr_for_epoch. Before ``decay_start`` the factor is 1.0,
    so the stage's value passes through unchanged; with ``decay_start`` at or
    after the last warmup epoch this equals ``lr_for_epoch(stage_lr, ...)``.

    Separate from lr_for_epoch only because its first argument means something
    different: that function's ``base`` is the lr read off the optimizer at the
    first decay epoch, while ``stage_lr`` here is the value the warmup stage
    owns *this* epoch. Multiplying the group's live lr instead would compound
    the factor across every decay epoch.

    Pure by design, for the same reason as lr_for_epoch.
    """
    return stage_lr * lr_decay_factor(epoch, decay_start, epochs, f)


def _min_lr_factor() -> float:
    """MIN_LR_FACTOR, or the env override, if the override is usable.

    A factor outside (0, 1] is not a smaller-lr knob -- 0 collapses the run to
    zero lr and >1 raises it -- so a bad value is warned about and ignored
    rather than obeyed. NaN and inf are rejected by _env_float.
    """
    f = _env_float("ECS_HCM_MIN_LR_FACTOR", MIN_LR_FACTOR)
    if not (0.0 < f <= 1.0):
        print(
            f"{Fore.YELLOW}[WARNING] ECS_HCM_MIN_LR_FACTOR={f!r} is outside "
            f"(0, 1]; using the default {MIN_LR_FACTOR}{Style.RESET_ALL}"
        )
        return MIN_LR_FACTOR
    return f


def _env_int(name: str, default: int, minimum: int | None = None) -> int:
    """Read an int threshold from the environment, falling back to the default.

    Resolved per call rather than at import so an operator (or a test) can set
    the variable on a running process without reloading this module.

    ``minimum``, when given, is the smallest value that makes sense for this
    threshold; anything below it is warned about and ignored. A threshold of
    zero or less silently disables the guard it configures, which is the same
    failure as a bad parse and gets the same treatment.
    """
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    try:
        value = int(raw)
    except ValueError:
        print(
            f"{Fore.YELLOW}[WARNING] {name}={raw!r} is not an integer; "
            f"using the default {default}{Style.RESET_ALL}"
        )
        return default
    if minimum is not None and value < minimum:
        print(
            f"{Fore.YELLOW}[WARNING] {name}={raw!r} is below the smallest "
            f"usable value ({minimum}); using the default "
            f"{default}{Style.RESET_ALL}"
        )
        return default
    return value


def _env_float(name: str, default: float, minimum: float | None = None) -> float:
    """Read a float threshold from the environment, falling back to the default.

    ``float("nan")`` and ``float("inf")`` both parse, so a plain try/except lets
    them through -- and a NaN threshold makes every comparison against it False,
    which silently disables the guard rather than failing loudly. Both are
    rejected here, along with ``-inf`` and anything below ``minimum``.
    """
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    try:
        value = float(raw)
    except ValueError:
        print(
            f"{Fore.YELLOW}[WARNING] {name}={raw!r} is not a number; "
            f"using the default {default}{Style.RESET_ALL}"
        )
        return default
    if not math.isfinite(value):
        print(
            f"{Fore.YELLOW}[WARNING] {name}={raw!r} is not a finite number; "
            f"using the default {default}{Style.RESET_ALL}"
        )
        return default
    if minimum is not None and value < minimum:
        print(
            f"{Fore.YELLOW}[WARNING] {name}={raw!r} is below the smallest "
            f"usable value ({minimum}); using the default "
            f"{default}{Style.RESET_ALL}"
        )
        return default
    return value


def _oom_limits() -> tuple[int, float]:
    """The (max consecutive skips, max skip fraction) in force for this call."""
    return (
        _env_int("ECS_HCM_MAX_CONSECUTIVE_OOM", MAX_CONSECUTIVE_OOM_SKIPS, minimum=1),
        _env_float("ECS_HCM_MAX_OOM_FRACTION", MAX_OOM_SKIP_FRACTION, minimum=0.0),
    )


def _check_oom_limits(skipped: int, attempted: int, consecutive: int) -> None:
    """Raise if the OOM skips so far mean this run cannot make progress.

    Called after every skipped batch. The message has to stand alone in an
    MLflow `error` tag, so it names both the observed numbers and the limits
    rather than only what tripped.
    """
    max_consecutive, max_fraction = _oom_limits()
    if consecutive > max_consecutive:
        raise RuntimeError(
            f"run is out of GPU memory: {consecutive} consecutive training "
            f"batches skipped (limit {max_consecutive}). Another process is "
            f"probably holding the GPU; free it and restart, or raise "
            f"ECS_HCM_MAX_CONSECUTIVE_OOM to skip more before failing."
        )
    if attempted >= MIN_BATCHES_FOR_OOM_FRACTION and attempted > 0:
        fraction = skipped / attempted
        if fraction > max_fraction:
            raise RuntimeError(
                f"training is out of GPU memory: {skipped}/{attempted} batches "
                f"skipped for OOM ({fraction * 100:.2f}%), above the "
                f"{max_fraction * 100:.2f}% limit. The run is not making real "
                f"progress; free the GPU and restart, or raise "
                f"ECS_HCM_MAX_OOM_FRACTION."
            )


def _warn_if_partial_validation(validated: int, presented: int) -> None:
    """Say so when validation stopped short of the batches it was handed.

    Not an error -- only a run that validates *nothing* is fatal -- but a
    validation quietly averaged over a fraction of its batches reports a
    plausible number that nobody can tell apart from a healthy one.
    """
    if 0 < validated < presented:
        print(
            f"{Fore.YELLOW}[WARNING] validation completed {validated} of "
            f"{presented} batches; val metrics cover only the first "
            f"{validated}{Style.RESET_ALL}"
        )


def _account_oom_skip(logger, batch_idx: int, epoch: int, batch_count: int,
                      attempted: int, oom_skipped: int, consecutive: int,
                      skipped_total: int) -> tuple[int, int, int]:
    """Count a skipped-for-OOM batch, log it, and fail if the run is stuck.

    Returns ``(consecutive, oom_skipped, skipped_total)`` for the caller to
    rebind. Raises via ``_check_oom_limits`` once the skips say the run cannot
    make progress -- the caller's enclosing ``except Exception`` turns that into
    an MLflow run that ends FAILED with an ``error`` tag.

    ``attempted`` is counted at the top of the batch loop rather than here, so
    a batch that OOMs still counts as attempted; counting only successes as
    attempts would make the skip fraction look smaller than it is.
    """
    consecutive += 1
    oom_skipped += 1
    skipped_total += 1

    # Log outside any `batch_count > 0` gate: the whole point is that a run
    # which trained *nothing* still reports that it skipped everything. The
    # first skip is logged immediately so a stuck run is visible from MLflow
    # without waiting for the periodic cadence.
    if logger is not None and (oom_skipped == 1 or oom_skipped % ACTIVATION_LOG_EVERY == 0):
        try:
            logger.log_scalar("batch_oom_skipped", oom_skipped, step=batch_idx)
        except Exception:  # pylint: disable=broad-exception-caught
            # A diagnostic must never take the run down.
            print(f"{Fore.RED}[ERROR] Could not log batch_oom_skipped: "
                  f"{traceback.format_exc()}{Style.RESET_ALL}")

    _check_oom_limits(oom_skipped, attempted, consecutive)
    return consecutive, oom_skipped, skipped_total


def _log_epoch_oom_metrics(logger, epoch: int, oom_skipped: int) -> None:
    """Log how many of this epoch's batches were skipped for OOM.

    Logged unconditionally -- including when it is 0 -- and from outside any
    ``batch_count > 0`` gate, because a run that skipped every batch is the case
    this metric exists to expose and that run has ``batch_count == 0``.
    """
    if logger is None:
        return
    try:
        logger.log_scalar("train_oom_skipped", oom_skipped, step=epoch)
    except Exception:  # pylint: disable=broad-exception-caught
        print(f"{Fore.RED}[ERROR] Could not log train_oom_skipped: "
              f"{traceback.format_exc()}{Style.RESET_ALL}")


class _TimedBatches:
    """Iterate ``batches``, adding the time each ``next()`` blocks to ``wait_s``.

    One pair of perf_counter calls per batch: the per-epoch data-wait figure
    without touching the hot loop otherwise.
    """

    def __init__(self, batches):
        self._it = iter(batches)
        self.wait_s = 0.0

    def __iter__(self):
        return self

    def __next__(self):
        t0 = time.perf_counter()
        try:
            return next(self._it)
        finally:
            self.wait_s += time.perf_counter() - t0


def _sync_if_cuda(device) -> None:
    """Flush queued CUDA work so a wall-clock edge covers it. Called at epoch
    and validation edges only, never per batch."""
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize(device)


EPOCH_TIMING_KEYS = (
    "epoch_train_s", "epoch_data_wait_s", "epoch_step_s", "epoch_val_s",
    "train_samples_per_s",
)


def _log_epoch_timing(logger, epoch: int, train_s: float, data_wait_s: float,
                      step_s: float, val_s: float, samples: int) -> None:
    """The epoch's timing split, at the epoch metrics' step."""
    if logger is None:
        return
    values = (train_s, data_wait_s, step_s, val_s,
              samples / train_s if train_s > 0 else 0.0)
    for key, value in zip(EPOCH_TIMING_KEYS, values):
        logger.log_scalar(key, float(value), step=epoch)


def _activation_snapshot(model) -> dict:
    """Read the model's running activation maxima, or {} if it has none.

    A DDP wrapper forwards no arbitrary attributes, so the caller may hand the
    wrapper here and the maxima are still read from the module that owns them.
    """
    model = unwrap_or_self(model)
    current = getattr(model, "activation_absmax", None)
    if not isinstance(current, dict):
        return {}
    return dict(current)


def _reset_activation_absmax(model) -> None:
    # Same reason as _activation_snapshot: a wrapper has neither attribute.
    model = unwrap_or_self(model)
    reset = getattr(model, "reset_activation_absmax", None)
    if callable(reset):
        reset()
        return
    current = getattr(model, "activation_absmax", None)
    if isinstance(current, dict):
        for stage in current:
            current[stage] = 0.0


def _fold_activation_absmax(model, running: dict) -> dict:
    """Merge the model's running maxima into an accumulator (does not reset)."""
    for stage, value in _activation_snapshot(model).items():
        if value > running.get(stage, 0.0):
            running[stage] = float(value)
    return running


def _log_activation_absmax(logger, model, prefix: str, step: int) -> None:
    """Log each stage's running max as ``{prefix}_{stage}_absmax`` and reset."""
    for stage, value in _activation_snapshot(model).items():
        try:
            logger.log_scalar(f"{prefix}_{stage}_absmax", float(value), step=step)
        except Exception:  # pylint: disable=broad-exception-caught
            # A diagnostic must never take the run down.
            print(f"{Fore.RED}[ERROR] Could not log {prefix}_{stage}_absmax: "
                  f"{traceback.format_exc()}{Style.RESET_ALL}")
    _reset_activation_absmax(model)


OUTLINE_LABEL_PEEK_BATCHES = 20


def require_outline_labels(train_loader):
    """Refuse an outline run whose training set carries no amodal labels.

    Reads up to ``OUTLINE_LABEL_PEEK_BATCHES`` batches (all of them if the epoch
    is shorter) and counts samples with ``has_full_outline``. This catches the
    mistake no other check can: a run pointed at a set with visible masks only
    (COCO, BDD) with the outline flag on. There every sample falls back to
    ``has_full_outline=False``, the outline term is a weighted zero, and the run
    spends its hours training a head on nothing.

    A batch without the key is a loader built without ``with_outline``: an
    outline run's own plumbing is wrong, so that is a refusal too.

    The loaders here are re-iterable (map-style, or an HF ``IterableDataset``
    whose shuffle is seeded), so reading the first batches does not consume
    them: the epoch that follows sees the same stream from its start.

    No all-reduce: the training loaders are sharded across ranks (a distributed
    sampler, or ``split_dataset_by_node`` for the HuggingFace path), so each
    rank peeks its own shard and the verdict is per-rank, not per-run. The peek
    happens before any collective, so a rank that finds no labelled sample
    raises while the run is still single-minded: the process dies, and no rank
    trains the head on nothing.
    """
    seen = 0
    labelled = 0
    for i, batch in enumerate(train_loader):
        if i >= OUTLINE_LABEL_PEEK_BATCHES:
            break
        if not isinstance(batch, dict) or "has_full_outline" not in batch:
            raise RuntimeError(
                "amodal outline is on but the training batches carry no outline "
                "target; build the training loader with with_outline=True")
        flags = torch.as_tensor(batch["has_full_outline"]).reshape(-1)
        labelled += int(flags.sum().item())
        seen += int(flags.numel())
    if labelled == 0:
        raise RuntimeError(
            f"amodal outline is on but none of the {seen} training samples "
            f"checked (up to {OUTLINE_LABEL_PEEK_BATCHES} batches) has a full-"
            f"outline label; point the run at a set that carries amodal labels")
    print(f"[OUTLINE] {labelled}/{seen} sampled training images carry a "
          f"full-outline label")


def train_model_for_epochs(model: hcm_nn , epochs: int, train_loader: Dataloader, val_loader: Dataloader, device, logger=None, test_mode=False, max_batches_per_epoch=0, warmup: bool = True, finetune: bool = False, coco: bool = False, task: str = None, checkpoint_every_epoch: bool = False, input_size: int = DEFAULT_INPUT_SIZE, coco_mix_ratio: float = 0.0, mix_train_loader: Dataloader = None, mix_val_loader: Dataloader = None, augmentation: dict | None = None, grad_accum_steps: int = 1, lr_decay_start: int | None = None, distributed: "DistributedContext | None" = None, best_on_primary: bool = False, primary_suffix: str = MIX_SUFFIX_PRIMARY, dataset_label: str | None = None, early_stop_monitor: str = "val_loss", early_stop_patience: int = 5, full_outline_weight: float = 1.0, amodal_outline: bool = False):
    # Validate before any work: a bad value is the invocation's fault, so report
    # it as a usage error rather than failing part-way through the first epoch.
    accum_steps = aug_accum_steps(grad_accum_steps)
    if early_stop_monitor not in EARLY_STOP_MONITORS:
        raise ValueError(
            f"unknown early-stop monitor {early_stop_monitor!r}; "
            f"expected one of {EARLY_STOP_MONITORS}")
    if early_stop_patience < 0:
        raise ValueError(
            f"early-stop patience cannot be negative, got {early_stop_patience}; "
            f"use 0 to disable")
    # The epoch the cosine decay starts at: the flag if given, else 5 with
    # warmup and 0 without (the schedule that existed before the flag).
    decay_start = lr_decay_start_for(lr_decay_start, warmup=warmup)
    # What every epoch checkpoint records. Rebuilt from the caller's jitter and
    # this loop's own count, so the record cannot claim an accumulation the
    # loop did not use.
    aug_record = _augmentation_record(
        (augmentation or {}).get("scale_jitter"),
        (augmentation or {}).get("color_jitter"),
        accum_steps,
    )
    # Logged here, next to the record, and before the first epoch: these are run
    # *inputs*, not measurements over time, so they are params rather than
    # scalars. The identity values are recorded too -- "no jitter" has to be
    # readable, not inferred from a missing key.
    if logger is not None:
        try:
            jitter_lo, jitter_hi = aug_record["scale_jitter"] or (1.0, 1.0)
            settings = [
                ("aug_scale_jitter_lo", jitter_lo),
                ("aug_scale_jitter_hi", jitter_hi),
                ("aug_color_jitter", aug_record["color_jitter"] or 0.0),
                ("grad_accum_steps", aug_record["grad_accum_steps"]),
                ("lr_decay_start", decay_start),
                ("init_from", init_from_of(model) or "none"),
                ("git_commit", os.environ.get("GIT_SHA", "unknown")),
                ("image_tag", os.environ.get("IMAGE_TAG", "unknown")),
                # A run input, so an outline run and its control are told apart
                # from the run record alone. The weight is recorded only when
                # there is a head for it to weigh.
                ("amodal_outline", bool(amodal_outline)),
                ("full_outline_weight",
                 float(full_outline_weight) if amodal_outline else "none"),
            ]
            # The architecture the run is training -- the ladder, the neck width
            # and the mask stride -- the same three the checkpoint carries, so a
            # from-ImageNet run and a fine-tune are told apart, and a bake-off's
            # arms, from the run record alone. "none" is the pre-change 2048
            # widening ("no new neck"), which is how the checkpoint spells it too.
            settings += [(name, value if value is not None else "none")
                         for name, value in arch_record_of(model).items()]
            for key, value in settings:
                logger.log_run_param(key, value)
        except Exception:  # pylint: disable=broad-exception-caught
            print(f"{Fore.RED}[ERROR] Could not log the run's settings: {traceback.format_exc()}{Style.RESET_ALL}")
    print(f"\n{'=' * 70}")
    print("Training Configuration:")
    print(f"  Device: {device}")
    print(f"  Epochs: {epochs}")
    print(f"{'=' * 70}\n")

    if amodal_outline:
        # Before the first optimiser step, not during it: a run with no labels
        # must fail in seconds, not after an epoch of nothing.
        require_outline_labels(train_loader)

    model = model.to(device)
    # The model itself, under whatever wrapper this run needs. It is the object
    # that owns the backbone, the head and the state dict; `model` below may be a
    # DDP wrapper after the wrap, and a wrapper has `.module` instead of those.
    module = unwrap_or_self(model)
    if test_mode:
        n = sum(p.numel() for p in module.parameters() if p.requires_grad)
        n_head = sum(p.numel() for p in module.segmentation_head.parameters() if p.requires_grad)
        print("trainable:", n, "seg_head_trainable:", n_head)


    _scaler = GradScaler()
    early_stopper = EarlyStopping(patience=early_stop_patience, monitor=early_stop_monitor)
    model.train()
    # The DDP wrap, and it has to be here: BEFORE the warmup freeze below. DDP
    # fixes the set of parameters it synchronises at construction, so a backbone
    # frozen at that moment would never be all-reduced when a stage later
    # unfreezes it. Wrapping while everything still requires grad puts every
    # parameter in the reducer, and find_unused_parameters=True (see
    # wrap_model_for_ddp) is what keeps the frozen warmup epochs legal. The
    # optimizer is built from `module`, so its parameter groups see the model's
    # own names rather than DDP's `module.` prefix.
    if distributed is not None:
        model = wrap_model_for_ddp(model, distributed)
    if warmup:
        for p in module.backbone.parameters():
            p.requires_grad = False
        module.backbone.eval()
        if coco or finetune:
            optimizer = create_model_optimizer(module, 0.0, 3e-5)
        else:
            optimizer = create_model_optimizer(module, 0.0, 3e-4)
    else:
        for p in module.backbone.parameters():
            p.requires_grad = True
        module.backbone.train()
        if coco or finetune:
            optimizer = create_model_optimizer(module, 1e-5, 3e-5)
        else:
            optimizer = create_model_optimizer(module, 1e-4, 3e-4)

    # Report the split for the state training actually begins in. The warmup
    # branch above freezes the backbone, so counting any earlier reported the
    # unfrozen numbers and overstated what was being trained.
    print_trainable_params(model, "start of training")

    if test_mode:
        seg_param_ids = {id(p) for p in module.segmentation_head.parameters()}
        opt_param_ids = {id(p) for g in optimizer.param_groups for p in g["params"]}

        print("seg params in optimizer:", len(seg_param_ids & opt_param_ids), "/", len(seg_param_ids))

    
    # The task decides which metrics exist, which bucket the checkpoint goes to
    # and which metric picks the best epoch. Derive it once, before the loop:
    # reading it from the flags in one place and from `task` in another made the
    # two disagree (a segmentation epoch logged as classification and then
    # looked up `val_acc1`, which a segmentation evaluate() never returns).
    epoch_task = task if task is not None else (
        TASK_SEGMENTATION if (finetune or coco) else TASK_CLASSIFICATION
    )
    is_segmentation = normalise_task(epoch_task) == TASK_SEGMENTATION

    # Best-epoch tracking, populated inside the loop and returned so the caller
    # can upload the best checkpoint at the end of the run.
    best_epoch = None
    best_score = float("-inf")
    best_state = None
    best_metrics = None
    best_train_loss = None
    checkpoint_upload_failures = 0
    # Run-total OOM skips, returned so the caller can tag the run with it.
    oom_skipped_total = 0
    # Spans epochs on purpose: an OOM that repeats across an epoch boundary is
    # still one continuous run of failures, which is the signal that the GPU is
    # gone rather than that one batch was unlucky.
    consecutive_oom = 0

    # ---- decay phase setup, resolved once per call so tests and operators can
    # ---- set the env without reloading the module (same rule as _oom_limits).
    # decay_start was resolved at the top of the call: the flag, or where warmup
    # ends when it was omitted.
    min_lr_factor = _min_lr_factor()
    # The stage lr per group that the factor multiplies. It cannot be read up
    # front: with warmup on, the backbone lr during epochs 0-4 is set by the
    # stage and after epoch 4 is held at whatever stage 3 left there (5e-5), not
    # the creation value. Captured before the factor is applied, so it never
    # carries the previous epoch's decay.
    decay_base = [None] * len(optimizer.param_groups)
    if decay_start >= epochs:
        print(
            f"{Fore.YELLOW}[WARNING] {epochs} epoch(s) ends at or before "
            f"decay_start (decay would start at epoch {decay_start}); lr decay "
            f"is inactive for this run{Style.RESET_ALL}"
        )

    for epoch in range(epochs):
        model.train()
        # A DistributedSampler reshuffles per epoch only when it is told which
        # epoch this is; without it every epoch would repeat the one split. A
        # plain loader's sampler has no set_epoch, and there may be no mix
        # loader at all, so this is a no-op for a single-process run.
        for _loader in (train_loader, mix_train_loader):
            _sampler = getattr(_loader, "sampler", None)
            if hasattr(_sampler, "set_epoch"):
                _sampler.set_epoch(epoch)
        # A mixed epoch walks one interleaved stream (COCOA with COCO worked in
        # between) rather than a concatenation, so each optimizer step sees a
        # batch from somewhere in the epoch instead of a COCOA half followed by
        # a COCO half. With no mix loader this is the plain loader, exactly as
        # before -- the generator is not even built.
        epoch_batches = (
            mixed_epoch_iterator(train_loader, mix_train_loader, coco_mix_ratio)
            if mix_train_loader is not None
            else train_loader
        )
        # The source of each batch slot, in yield order: _interleave yields one
        # batch per schedule entry, so batch_idx indexes it directly.
        epoch_sources = None
        if mix_train_loader is not None:
            n_primary = len(train_loader)
            epoch_sources = mix_schedule(
                n_primary, mix_batches_for(n_primary, coco_mix_ratio)
            )
        source_batch_size = {
            MIX_SUFFIX_PRIMARY: getattr(train_loader, "batch_size", None),
            MIX_SUFFIX_MIX: getattr(mix_train_loader, "batch_size", None),
        }
        data_iterator = tqdm(epoch_batches, leave=True, unit='batch', desc=f"Epoch {epoch+1}/{epochs}")
        running_loss = 0.0
        running_acc1 = 0.0
        running_acc5 = 0.0
        # The amodal-outline term, summed over the batches that carried an
        # outline target (a mixed-lane COCO batch carries none), plus the
        # number of images with a known full-outline label.
        running_outline_loss = 0.0
        outline_labelled = 0
        outline_batches = 0
        batch_count = 0
        # Batches the loop pulled from the loader this epoch, whether they
        # trained or were skipped for OOM. This is the denominator the skip
        # fraction is judged against.
        attempted = 0
        oom_skipped = 0
        epoch_dropped_regions = 0
        # Per-source train counts. Under a mix each batch is attributed to its
        # dataset by its slot in the epoch's schedule; unmixed, all are primary.
        epoch_train_counts = {
            name: {MIX_SUFFIX_PRIMARY: 0, MIX_SUFFIX_MIX: 0}
            for name in ("train_dropped_regions", "train_failed_images")
        }
        avg_loss = 0.0
        avg_acc1 = 0.0
        avg_acc5 = 0.0
        # Epoch-level activation maxima, accumulated the same way as the batch
        # ones so the epoch figure is the epoch's true max, not a sample.
        epoch_activation_absmax = {}

        if warmup:
            # ---- staged unfreezing ----
            if epoch == 0:
                # head warmup
                freeze_all_backbone(module.backbone)
                module.backbone.eval()
                optimizer.param_groups[0]["lr"] = 0.0

            elif epoch == 1 or epoch == 2:
                # unfreeze layer4
                freeze_all_backbone(module.backbone)
                unfreeze_backbone_stages(module.backbone, ["layer4"])
                module.backbone.train()
                backbone_bn_eval(module.backbone)  # optional but recommended
                optimizer.param_groups[0]["lr"] = 1e-4

            elif epoch == 3 or epoch == 4:
                # unfreeze layer3 + layer4
                freeze_all_backbone(module.backbone)
                unfreeze_backbone_stages(module.backbone, ["layer3", "layer4"])
                module.backbone.train()
                backbone_bn_eval(module.backbone)
                optimizer.param_groups[0]["lr"] = 5e-5

            else:
                # model.train() at the top of the epoch left the whole model in
                # training mode. The warmup stages end at epoch 4, so nothing
                # else re-pins batch norm. This changes only the batch-norm mode:
                # the trainable stages and the backbone learning rate are left
                # exactly as the epoch-4 stage set them.
                backbone_bn_eval(module.backbone)

            # The staged unfreeze changes which parameters are trainable, so
            # report the split again rather than leaving the reader with the
            # start-of-training figure.
            print_trainable_params(model, f"epoch {epoch + 1} stage")

        # ---- learning-rate decay ----
        # Each group decays from what it holds at decay_start, so an epoch before
        # decay_start keeps exactly today's value -- a run that ends at or before
        # decay_start is bit-for-bit unchanged by this feature.
        #
        # A decay that starts *inside* warmup is the one case where "what it
        # holds at decay_start" is the wrong base: warmup owns the backbone lr
        # and re-sets it by stage every epoch (then stops at epoch 4), so the
        # factor is applied on top of the stage's value for the epoch instead
        # (staged_lr_for_epoch) -- a base that changes per warmup epoch and is
        # held at its last stage value once the stages end. Never the group's
        # live lr, which already carries the previous epoch's taper and would
        # compound it. The head (group 1) warmup never touches, so its base is
        # its creation lr, captured once.
        #
        # Outside that case the path below is exactly the code that ran before
        # the flag existed, so omitting --lr-decay-start cannot change a value.
        if warmup and decay_start < WARMUP_EPOCHS:
            for i, group in enumerate(optimizer.param_groups):
                if i == 0 and epoch < WARMUP_EPOCHS:
                    # The stage just set the backbone lr: that is this epoch's base.
                    decay_base[i] = group["lr"]
                elif decay_base[i] is None:
                    decay_base[i] = group["lr"]
                group["lr"] = staged_lr_for_epoch(
                    decay_base[i], epoch, decay_start, epochs, min_lr_factor
                )
        elif epoch >= decay_start:
            for i, group in enumerate(optimizer.param_groups):
                if decay_base[i] is None:
                    # First epoch of the decay phase: record the group's entry
                    # lr and decay from it on every later epoch.
                    decay_base[i] = group["lr"]
                group["lr"] = lr_for_epoch(
                    decay_base[i], epoch, decay_start, epochs, min_lr_factor
                )

        # Log both groups every epoch, including the warmup epochs, so the
        # schedule is visible in MLflow as one continuous curve rather than
        # only from the point decay starts.
        lr_backbone = optimizer.param_groups[0]["lr"]
        lr_head = optimizer.param_groups[1]["lr"]
        if logger is not None:
            logger.log_scalar("lr_backbone", lr_backbone, step=epoch)
            logger.log_scalar("lr_head", lr_head, step=epoch)
        print(f"[lr] epoch {epoch} backbone={lr_backbone:.6g} head={lr_head:.6g}")

        final_batch_idx = 0
        # Timing split for the epoch: time blocked fetching batches, time in
        # forward/backward/step, and the wall time around both. CUDA is
        # synchronised at the edges only; inside the loop, loss.item() already
        # waits for each step's kernels.
        epoch_step_s = 0.0
        epoch_samples = 0
        # Batches accumulated into the group that is still open. Zero means the
        # next batch starts a group and zeroes the gradients.
        accum_count = 0
        timed_batches = _TimedBatches(data_iterator)
        _sync_if_cuda(device)
        epoch_t0 = time.perf_counter()
        # The ranks' shards differ in size by up to one example, so at a batch
        # size that divides unevenly they can produce a different number of
        # batches. Inside DDP's join the rank that runs out leaves the epoch
        # cleanly while the other finishes, instead of waiting in an all-reduce
        # the other rank will never enter.
        #
        # Entered by hand rather than with a `with` statement: the body below is
        # the epoch's existing try/except, and this way it does not have to be
        # re-indented by four spaces. `sys.exc_info()` inside the finally is the
        # same information a `with` would hand to __exit__, and join() has to see
        # the exception type so that a failing rank skips the joining protocol
        # (which is a collective) instead of blocking on a peer that is gone.
        join_epoch = join_uneven_epochs(model)
        join_epoch.__enter__()
        try:
            for batch_idx, batch in enumerate(timed_batches):
                # A cap applies whenever one was given, not only in test_mode:
                # requiring test_mode here meant --max_batches_per_epoch was
                # silently ignored on a real run. 0 means uncapped.
                if max_batches_per_epoch > 0 and batch_idx >= max_batches_per_epoch:
                    break

                # Skip None batches that may come from collate_fn when data is invalid
                if batch is None:
                    continue

                attempted += 1

                if finetune or coco:
                    batch_dropped = _batch_dropped_regions(batch)
                    epoch_dropped_regions += batch_dropped
                    source = MIX_SUFFIX_PRIMARY
                    if epoch_sources is not None and batch_idx < len(epoch_sources):
                        if epoch_sources[batch_idx] == MIX_SOURCE_MIX:
                            source = MIX_SUFFIX_MIX
                    epoch_train_counts["train_dropped_regions"][source] += batch_dropped
                    epoch_train_counts["train_failed_images"][source] += (
                        _batch_failed_images(batch, source_batch_size[source])
                    )
                    # COCO Segmentation Training with Binary Masks
                    if isinstance(batch, dict):
                        # Extract images and binary segmentation masks
                        images = batch['pixel_values'].to(device, non_blocking=True)
                        segmentation_masks = batch['segmentation_mask'].to(device, non_blocking=True)
                    else:
                        images, segmentation_masks = batch
                        images = images.to(device, non_blocking=True)
                        segmentation_masks = segmentation_masks.to(device, non_blocking=True)

                    step_t0 = time.perf_counter()
                    if accum_count == 0:
                        # Zero at the start of a group rather than of every
                        # batch: the group's batches accumulate into one step.
                        # With accum_steps 1 this is every batch, as before.
                        optimizer.zero_grad(set_to_none=True)

                    # The last micro-batch of a group synchronises: the ones
                    # before it run under no_sync so the group pays one
                    # all-reduce, at its final backward, rather than one per
                    # micro-batch. A no-op when the model is not DDP-wrapped.
                    group_end = (accum_count + 1) >= accum_steps

                    # Unwrapped like this function's other hand-offs (the
                    # evaluate/save calls below): `model` here is the DDP
                    # wrapper, which has no `__getattr__`. Read at each use, so
                    # the grid flag is the model's and not a wrapper's default.
                    outline_batch = isinstance(batch, dict) and "outline_mask" in batch

                    try:
                        with autocast(), accum_step_sync(model, group_end):
                            if outline_batch:
                                # One decoder pass, two heads. Both terms share
                                # the pass, so the outline head costs no forward.
                                output, outline_logits = model(images, return_outline=True)
                            else:
                                output = model(images)
                            loss, iou_score, dice_score = calculate_segmentation_loss(
                                output, segmentation_masks, device, test_mode=test_mode,
                                target_follows_logits=target_grid_follows_logits(unwrap_or_self(model)),
                            )
                            if outline_batch:
                                outline_loss, outline_labelled_batch = outline_loss_from_logits(
                                    outline_logits, batch["outline_mask"].to(
                                        device, non_blocking=True),
                                    batch["has_full_outline"],
                                    target_follows_logits=target_grid_follows_logits(unwrap_or_self(model)),
                                )
                                loss = loss + full_outline_weight * outline_loss
                            # Divided by the group size so the step applies the
                            # mean gradient of the group's batches. The reported
                            # loss stays undivided, so the epoch average is the
                            # mean of the per-batch losses either way. The
                            # backward is inside the no_sync block on purpose:
                            # that is where DDP hangs its all-reduce.
                            _scaler.scale(loss / accum_steps).backward()

                        # Track segmentation metrics
                        running_loss += loss.detach().item()
                        running_acc1 += iou_score*100  # IoU as primary metric
                        running_acc5 += dice_score*100  # Dice coefficient as secondary metric
                        if outline_batch:
                            # The term on its own, without the weight: the
                            # weight is a run param, so the logged value stays
                            # comparable between runs that tune it.
                            running_outline_loss += float(outline_loss.detach().item())
                            outline_labelled += int(outline_labelled_batch)
                            outline_batches += 1
                        batch_count += 1
                        consecutive_oom = 0
                        accum_count += 1
                        if accum_count >= accum_steps:
                            _optimizer_step(model, optimizer, _scaler)
                            accum_count = 0
                        epoch_step_s += time.perf_counter() - step_t0
                        epoch_samples += images.shape[0]

                    except RuntimeError as e:
                        if "out of memory" in str(e).lower():
                            print(f"\n{Fore.RED}[ERROR] Batch {batch_idx}: {str(e)[:100]}{Style.RESET_ALL}")
                            optimizer.zero_grad(set_to_none=True)
                            # The skipped batch abandons its group, so the next
                            # batch starts a fresh one instead of adding to it.
                            accum_count = 0
                            if 'loss' in locals():
                                del loss
                            if 'output' in locals():
                                del output
                            torch.cuda.empty_cache()
                            consecutive_oom, oom_skipped, oom_skipped_total = _account_oom_skip(
                                logger, batch_idx, epoch, batch_count,
                                attempted, oom_skipped, consecutive_oom, oom_skipped_total,
                            )
                            continue
                        raise

                else:
                    if isinstance(batch, dict):
                        images = batch['pixel_values'].to(device, non_blocking=True)
                        target = batch['label'].to(device, non_blocking=True)
                    else:
                        # assume (images, labels) tuple/list
                        images, target = batch
                        images = images.to(device, non_blocking=True)
                        target = target.to(device, non_blocking=True)

                    step_t0 = time.perf_counter()
                    if accum_count == 0:
                        optimizer.zero_grad(set_to_none=True)

                    # As in the segmentation branch: only the group's last
                    # micro-batch synchronises, and the backward has to be inside
                    # the no_sync block for that to hold.
                    group_end = (accum_count + 1) >= accum_steps

                    try:
                        with autocast(), accum_step_sync(model, group_end):
                            output = model(images)
                            loss = F.cross_entropy(output, target)
                            _scaler.scale(loss / accum_steps).backward()

                        running_loss += loss.detach().item()
                        accum_count += 1
                        if accum_count >= accum_steps:
                            _optimizer_step(model, optimizer, _scaler)
                            accum_count = 0
                        with torch.no_grad():
                            acc1, acc5 = __accuracy(output, target, topk=(1, 5))
                        running_acc1 += acc1.item()
                        running_acc5 += acc5.item()
                        batch_count += 1
                        consecutive_oom = 0
                        epoch_step_s += time.perf_counter() - step_t0
                        epoch_samples += images.shape[0]

                    except RuntimeError as e:
                        if "out of memory" in str(e).lower():
                            print(f"\n{Fore.RED}[ERROR] Batch {batch_idx}: {str(e)[:100]}{Style.RESET_ALL}")
                            optimizer.zero_grad(set_to_none=True)
                            accum_count = 0
                            # An OOM can fire before `loss`/`output` are bound
                            # (e.g. on the forward pass), so only delete what
                            # actually exists here.
                            if 'loss' in locals():
                                del loss
                            if 'output' in locals():
                                del output
                            torch.cuda.empty_cache()
                            consecutive_oom, oom_skipped, oom_skipped_total = _account_oom_skip(
                                logger, batch_idx, epoch, batch_count,
                                attempted, oom_skipped, consecutive_oom, oom_skipped_total,
                            )
                            continue
                        raise

                # Update progress bar with metrics (outside if-else block)
                if batch_count > 0:
                    avg_loss = running_loss / batch_count
                    avg_acc1 = running_acc1 / batch_count
                    avg_acc5 = running_acc5 / batch_count

                    if finetune or coco:
                        data_iterator.set_postfix(
                            loss=f"{avg_loss:.4f}",
                            IoU=f"{avg_acc1:.2f}%",
                            Dice=f"{avg_acc5:.2f}%"
                        )
                    else:
                        data_iterator.set_postfix(
                            loss=f"{avg_loss:.4f}",
                            top1=f"{avg_acc1:.2f}%",
                            top5=f"{avg_acc5:.2f}%"
                        )

                    # Log batch metrics. The task decides which metric names
                    # exist; there is no longer an IoU=acc1 alias. Use the
                    # concrete task so the batch keys agree with the epoch keys.
                    if logger:
                        logger.track_batch(
                            data_obj(lv=loss.detach().item(), ac1=avg_acc1,
                                     ac5=avg_acc5, prefix="batch", task=epoch_task),
                            batch_idx,
                        )

                    # Log the activation magnitudes every N batches, then reset
                    # the running max. The running max is what catches a single
                    # spike between logging points -- sampling only the batch
                    # that lands on a multiple of N would miss exactly the kind
                    # of one-off blow-up that killed the run.
                    if logger and batch_idx > 0 and batch_idx % ACTIVATION_LOG_EVERY == 0:
                        _log_activation_absmax(
                            logger, model, prefix="batch", step=batch_idx
                        )
                        epoch_activation_absmax = _fold_activation_absmax(
                            model, epoch_activation_absmax
                        )

                    # Print milestone updates (less frequent)
                    if batch_idx > 0 and batch_idx % 1000 == 0:
                        if finetune or coco:
                            print(f"\n{Fore.CYAN}[Epoch {epoch+1} | Batch {batch_idx}] Loss: {avg_loss:.4f} | IoU: {avg_acc1:.2f}% | Dice: {avg_acc5:.2f}%{Style.RESET_ALL}")
                        else:
                            print(f"\n{Fore.CYAN}[Epoch {epoch+1} | Batch {batch_idx}] Loss: {avg_loss:.4f} | Top-1: {avg_acc1:.2f}% | Top-5: {avg_acc5:.2f}%{Style.RESET_ALL}")
                final_batch_idx = batch_idx

            if accum_count > 0:
                if distributed is not None and distributed.is_distributed:
                    # Under DDP the trailing group is not the same group on every
                    # rank: its micro-batches ran under no_sync, so stepping here
                    # would take a rank-local step on gradients the peers never
                    # saw, and all-reducing the open group by hand is not safe
                    # either -- the set of parameters that actually carries a
                    # gradient can differ between ranks. Discard it, and log the
                    # size so a run at an awkward accumulation/batch ratio is
                    # visible rather than mysterious.
                    print(
                        f"{Fore.YELLOW}[WARNING] Epoch {epoch+1}: discarding a "
                        f"partial accumulation group of {accum_count} batch(es) "
                        f"under distributed training{Style.RESET_ALL}"
                    )
                    if logger:
                        with contextlib.suppress(Exception):
                            logger.log_scalar(
                                "train_discarded_partial_group",
                                float(accum_count),
                                step=epoch,
                            )
                    optimizer.zero_grad(set_to_none=True)
                    accum_count = 0
                else:
                    # The epoch ended with fewer than accum_steps batches in the
                    # open group (its last group is partial). Step on what it has
                    # rather than leaving those gradients unapplied. With
                    # accum_steps 1 this cannot happen: every batch stepped.
                    _optimizer_step(model, optimizer, _scaler)
                    accum_count = 0

        except (HfHubHTTPError, Exception) as e: # pylint: disable=broad-exception-caught
            # Check if it's a HuggingFace HTTP error (like 504 Gateway Timeout).
            # No epoch gate: a run that dies at any epoch should still leave a
            # recoverable checkpoint behind.
            is_hf_error = isinstance(e, HfHubHTTPError) or "HfHubHTTPError" in str(type(e)) or "504" in str(e)
            if is_hf_error and not (distributed is not None and distributed.is_distributed and not distributed.is_main):
                print(f"\n{Fore.YELLOW}[WARNING] Caught HuggingFace HTTP error after epoch {epoch+1}. Saving model to MinIO...{Style.RESET_ALL}")
                print(f"Error details: {str(e)[:200]}")

                try:
                    # Save to the task's own bucket under the task's name.
                    save_epoch_checkpoint(
                        model=unwrap_or_self(model),
                        task=resolve_task(finetune=finetune, coco=coco),
                        epoch=epoch + 1,
                        val_metrics={
                            "of_epoch": epoch + 1,
                            "interruption_reason": "HuggingFace HTTP error",
                        },
                        avg_loss=avg_loss if batch_count > 0 else None,
                        val_loss=None,
                        run_id=logger.run_id if logger is not None else None,
                        is_best=False,
                        is_last=True,
                        is_smoke=test_mode,
                        input_size=input_size,
                        coco_mix_ratio=coco_mix_ratio,
                        dataset_name=dataset_label,
                        augmentation=aug_record,
                        lr_decay_start=decay_start,
                    )
                    print(f"{Fore.GREEN}Model successfully saved to MinIO{Style.RESET_ALL}")
                except Exception as save_error: # pylint: disable=broad-exception-caught
                    print(f"{Fore.RED}[ERROR] Failed to save model to MinIO: {str(save_error)}{Style.RESET_ALL}")
                    print(traceback.format_exc())

                # Re-raise the original error
                raise e
            else:
                # If it's not the error we're looking for, re-raise
                raise e
        finally:
            # Closes the join opened above. On the normal path this is where the
            # ranks agree the epoch is over; on a failing rank sys.exc_info() is
            # the in-flight exception, which tells join() not to run its
            # protocol -- a collective on an error path is the hang this whole
            # section exists to avoid.
            join_epoch.__exit__(*sys.exc_info())

        _sync_if_cuda(device)
        epoch_train_s = time.perf_counter() - epoch_t0

        # One all-reduce, before anything reads these numbers. Every decision
        # below -- the zero-batch failure, the OOM limits, early stopping, the
        # best-epoch comparison and the epoch's logged metrics -- is taken on the
        # run's totals rather than one rank's share of them, and it is THE place
        # where the ranks agree, so no rank can fail the run on a sum its peers
        # do not have.
        #
        # Sums, then a division, and never a mean of means: the run's mean over
        # its batches is `sum(loss) / sum(count)`, so counts of 1 and 3 with
        # losses of 2.0 and 4.0 give 3.5, where averaging the two ranks' means
        # gives 3.0. The val metric is a per-image mean, a different quantity, and
        # is deliberately not pooled this way -- validation runs once, on the
        # rank that owns the run, and its result is broadcast whole.
        epoch_numbers = {
            "batch_count": batch_count,
            "attempted": attempted,
            "oom_skipped": oom_skipped,
            "epoch_samples": epoch_samples,
            "epoch_dropped_regions": epoch_dropped_regions,
            "running_loss": running_loss,
            "running_acc1": running_acc1,
            "running_acc5": running_acc5,
            "running_outline_loss": running_outline_loss,
            "outline_labelled": outline_labelled,
            "outline_batches": outline_batches,
        }
        for _name in ("train_dropped_regions", "train_failed_images"):
            for _suffix in (MIX_SUFFIX_PRIMARY, MIX_SUFFIX_MIX):
                epoch_numbers[f"{_name}_{_suffix}"] = epoch_train_counts[_name][_suffix]
        epoch_numbers = _reduce_epoch_numbers(device, epoch_numbers)
        batch_count = epoch_numbers["batch_count"]
        attempted = epoch_numbers["attempted"]
        oom_skipped = epoch_numbers["oom_skipped"]
        epoch_samples = epoch_numbers["epoch_samples"]
        epoch_dropped_regions = epoch_numbers["epoch_dropped_regions"]
        running_loss = epoch_numbers["running_loss"]
        running_acc1 = epoch_numbers["running_acc1"]
        running_acc5 = epoch_numbers["running_acc5"]
        running_outline_loss = epoch_numbers["running_outline_loss"]
        outline_labelled = epoch_numbers["outline_labelled"]
        outline_batches = epoch_numbers["outline_batches"]
        for _name in ("train_dropped_regions", "train_failed_images"):
            for _suffix in (MIX_SUFFIX_PRIMARY, MIX_SUFFIX_MIX):
                epoch_train_counts[_name][_suffix] = epoch_numbers[f"{_name}_{_suffix}"]
        # One rank's unbroken run of OOM skips is not the run's: the limit is
        # about the run's worst stretch, so the ranks' consecutive counters are
        # combined by max rather than by sum.
        consecutive_oom = _all_reduce_max_int(device, consecutive_oom)

        # The progress bar's average is per-rank and per-iteration; the epoch's
        # reported loss is pooled over the whole run, so it is computed here,
        # once, from the reduced sums. Single-process this is the same division
        # the progress bar already made.
        if batch_count > 0:
            avg_loss = running_loss / batch_count
            avg_acc1 = running_acc1 / batch_count
            avg_acc5 = running_acc5 / batch_count

        if test_mode and not finetune and not coco: # Debugging info in test mode
            total_norm = 0.0
            for p in module.parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
            total_norm = total_norm ** 0.5
            print(f"[Epoch {epoch+1} | Batch {final_batch_idx}] Grad L2 norm: {total_norm:.4f}")

            batch = next(iter(train_loader))

            # Skip None batches that may come from collate_fn when data is invalid
            imgs, labels = None, None
            if batch is None:
                print("Warning: Skipping None batch in label inspection")
            else:
                if isinstance(batch, dict):
                    imgs = batch["pixel_values"]
                    labels = batch["label"]
                else:
                    imgs, labels = batch

                print("labels dtype:", labels.dtype)
                print("labels min:", labels.min().item(), "max:", labels.max().item())
                print("unique labels (first batch):", labels.unique()[:10])
                print("imgs shape:", imgs.shape, "imgs min/max:", imgs.min().item(), imgs.max().item())

        # An epoch that trained nothing is a failed run, not a quiet one. This
        # is gated on `attempted` rather than on `batch_count` alone so that a
        # loader which simply had no batches is not mistaken for a run whose
        # every batch blew up -- the second is the failure, the first is not a
        # training epoch at all. Checked before validation so the run fails on
        # the real cause instead of on whatever evaluate() hits next.
        if batch_count == 0 and attempted > 0:
            # Reports the epoch's OOM figure before dying. This and the call
            # below the `raise` are NOT duplicates: whichever path the epoch
            # takes reaches exactly one of them. Removing this call would drop
            # the number from the only path that needs it most -- the epoch
            # where the skips ate the whole run.
            _log_epoch_oom_metrics(logger, epoch, oom_skipped)
            raise RuntimeError(
                f"epoch {epoch + 1} trained 0 batches: all {attempted} attempted "
                f"batches failed with CUDA OOM. The epoch reported as complete "
                f"would have contained no training."
            )

        # Per-epoch OOM total, logged whether or not any batch was skipped so a
        # healthy epoch produces the same key a failed one does. Deliberately
        # outside the `batch_count > 0` block below: a zero-batch epoch is
        # exactly the case that must still report its skips. This is the call
        # the normal path reaches; the zero-batch `raise` above reaches its own
        # copy and never gets here. Keep both.
        _log_epoch_oom_metrics(logger, epoch, oom_skipped)
        if (finetune or coco) and logger is not None:
            # Every epoch, zero included: COCOA regions the mask builder left
            # out of a training target. Used to be dropped without a trace.
            logger.log_scalar("train_dropped_regions", float(epoch_dropped_regions), step=epoch)
            # Images a builder exception dropped from a training batch, zero
            # included, as validation does with val_failed_images.
            logger.log_scalar(
                "train_failed_images",
                float(sum(epoch_train_counts["train_failed_images"].values())),
                step=epoch,
            )
            if outline_batches:
                # The unweighted term, so runs that tune the weight compare.
                logger.log_scalar(
                    "train_outline_loss",
                    float(running_outline_loss / outline_batches), step=epoch)
                # Zero included: a run pointed at a set with no amodal labels
                # must show the zero rather than omit the key.
                logger.log_scalar(
                    "train_outline_labelled_images", float(outline_labelled), step=epoch)
            # Under a mix, each count per dataset too, as validation does.
            if epoch_sources is not None:
                for name, per_source in epoch_train_counts.items():
                    for suffix in (MIX_SUFFIX_PRIMARY, MIX_SUFFIX_MIX):
                        label = primary_suffix if suffix == MIX_SUFFIX_PRIMARY else suffix
                        logger.log_scalar(
                            f"{name}_{label}", float(per_source[suffix]), step=epoch
                        )

        # Re-judge the skip fraction now that `attempted` is final. The in-loop
        # check only runs when a batch is skipped, so a run whose skips all
        # landed early would never be judged against the 200-batch floor it has
        # since passed -- the ratio only becomes meaningful once enough batches
        # have been attempted, and by then there may be no further skip to
        # trigger the check.
        _check_oom_limits(oom_skipped, attempted, consecutive_oom)

        val_t0 = time.perf_counter()
        # Validation is measured once, by the rank that owns the run, and on the
        # *unwrapped* module: a DDP-wrapped forward is itself a collective (it
        # syncs buffers from rank 0), so running it on one rank while the others
        # sat in the reduction would hang. The result is then handed to every
        # rank whole, so all of them decide on bit-identical numbers instead of
        # recomputing a metric that is meant to be measured once.
        val_metrics = None
        if distributed is None or not distributed.is_distributed or distributed.is_main:
            val_metrics = evaluate(model=unwrap_or_self(model), val_loader=val_loader, device=device, max_batches=100 if test_mode else 0, is_segmentation=is_segmentation)

            # A mixed run is scored on BOTH datasets, and the unsuffixed names --
            # the ones best selection and early stopping read -- become their mean.
            # Scoring only the refine's own dataset is what let the 09-30 forgetting
            # go unseen until the run was over.
            if mix_val_loader is not None:
                mix_val_metrics = evaluate(
                    model=unwrap_or_self(model), val_loader=mix_val_loader, device=device,
                    max_batches=100 if test_mode else 0,
                    is_segmentation=is_segmentation,
                )
                val_metrics = merge_mixed_val_metrics(
                    val_metrics, mix_val_metrics, primary_suffix=primary_suffix
                )
        val_metrics = _broadcast_object(device, val_metrics)
        _sync_if_cuda(device)
        _log_epoch_timing(
            logger, epoch,
            train_s=epoch_train_s, data_wait_s=timed_batches.wait_s,
            step_s=epoch_step_s, val_s=time.perf_counter() - val_t0,
            samples=epoch_samples,
        )

        if is_segmentation:
            print(
                f"[VAL] loss={val_metrics['val_loss']:.4f} "
                f"IoU={val_metrics['val_iou']:.2f}% "
                f"Dice={val_metrics['val_dice']:.2f}%"
                + (f" failed_images={val_metrics['val_failed_images']}"
                   if val_metrics.get("val_failed_images") else "")
            )
            if mix_val_loader is not None:
                print(
                    f"[VAL]   {primary_suffix} IoU="
                    f"{val_metrics[f'val_iou_{primary_suffix}']:.2f}% | "
                    f"{MIX_SUFFIX_MIX} IoU="
                    f"{val_metrics[f'val_iou_{MIX_SUFFIX_MIX}']:.2f}% "
                    f"(the line above is their mean)"
                )
        else:
            print(
                f"[VAL] loss={val_metrics['val_loss']:.4f} "
                f"top1={val_metrics['val_acc1']:.2f}% "
                f"top5={val_metrics['val_acc5']:.2f}%"
            )

        # Log epoch metrics. Train and val go to distinct key namespaces
        # (train_*/val_*) so they no longer overwrite each other at the same
        # step, and each task logs only the metrics it actually has.
        if logger:
            logger.track(
                data_obj(avg_loss, avg_acc1, avg_acc5,
                         prefix="train", task=epoch_task),
                epoch,
            )
            logger.track(
                data_obj(
                    val_metrics["val_loss"],
                    val_metrics["val_iou"] if is_segmentation else val_metrics["val_acc1"],
                    val_metrics["val_dice"] if is_segmentation else val_metrics["val_acc5"],
                    prefix="val", task=epoch_task,
                ),
                epoch,
            )
            for count in _VAL_COUNTS:
                if count in val_metrics:
                    # Every epoch, zero included, so a run with no failures is
                    # distinguishable from one that never counted them.
                    logger.log_scalar(count, float(val_metrics[count]), step=epoch)
            # The amodal-outline numbers, when the validation set carried an
            # outline target. Unweighted loss so runs that tune the weight
            # compare; the four IoUs on the labelled images only. The primary
            # set's values unsuffixed and under its suffix, a mix set's under
            # its own, as weight 0/1 of the merge.
            for name in ("val_outline_loss",) + _OUTLINE_VAL_METRICS:
                if name in val_metrics:
                    logger.log_scalar(name, float(val_metrics[name]), step=epoch)
                if mix_val_loader is not None:
                    for suffix in (primary_suffix, MIX_SUFFIX_MIX):
                        key = f"{name}_{suffix}"
                        if key in val_metrics:
                            logger.log_scalar(key, float(val_metrics[key]), step=epoch)
            if mix_val_loader is not None:
                # Each dataset on its own, so the forgetting is visible in
                # MLflow while it happens. The unsuffixed values logged just
                # above are the mean of these.
                for name in ("loss", "iou", "dice"):
                    for suffix in (primary_suffix, MIX_SUFFIX_MIX):
                        logger.log_scalar(
                            f"val_{name}_{suffix}",
                            val_metrics[f"val_{name}_{suffix}"],
                            step=epoch,
                        )
                # The failed-image count per dataset too, so a rising total
                # says which dataset is losing images.
                for count in _VAL_COUNTS:
                    for suffix in (primary_suffix, MIX_SUFFIX_MIX):
                        key = f"{count}_{suffix}"
                        if key in val_metrics:
                            logger.log_scalar(key, float(val_metrics[key]), step=epoch)
            # The epoch's activation maxima. Fold in whatever the model is
            # still holding so the tail of the epoch is not dropped, then log
            # the epoch figure under train_*_absmax.
            _fold_activation_absmax(model, epoch_activation_absmax)
            _reset_activation_absmax(model)
            for stage in _ACTIVATION_STAGES:
                if stage in epoch_activation_absmax:
                    logger.log_scalar(
                        f"train_{stage}_absmax",
                        epoch_activation_absmax[stage],
                        step=epoch,
                    )

        # Track the best epoch by the task's validation metric so the best
        # checkpoint can be uploaded at the end.
        best_metric_name = BEST_METRIC_FOR_TASK.get(epoch_task)
        # A BDD refine keeps COCO in the mix to hold what the model already
        # knows, but its own target is BDD: the best epoch is chosen on BDD val
        # alone. merge_mixed_val_metrics makes the unsuffixed value the mean of
        # the two, which is the right rule for a COCOA refine and the wrong one
        # here, so read the primary's own figure out of the same dict.
        current_best_value = best_value_from(
            val_metrics, best_metric_name,
            mixed=mix_val_loader is not None, best_on_primary=best_on_primary,
            primary_suffix=primary_suffix,
        )
        is_best = (
            current_best_value is not None
            and (best_epoch is None or current_best_value > best_score)
        )
        if is_best:
            best_epoch = epoch
            best_score = float(current_best_value)
            # The unwrapped module's state dict: a wrapped one carries DDP's
            # `module.` prefix on every key, which the single-process loader this
            # checkpoint is written for would not find. Captured on every rank --
            # the ranks' parameters are identical after DDP's reduction, so this
            # keeps the returned result the same on all of them -- and written by
            # the rank that owns the run.
            best_state = {
                k: v.detach().cpu().clone()
                for k, v in unwrap_or_self(model).state_dict().items()
            }
            best_metrics = dict(val_metrics)
            best_train_loss = avg_loss

        # Save a checkpoint at the end of every epoch to the task's bucket. Only
        # the rank that owns the run writes: two ranks uploading the same epoch
        # number would be two writers racing for one MinIO key.
        if checkpoint_every_epoch and (
            distributed is None or not distributed.is_distributed or distributed.is_main
        ):
            try:
                save_epoch_checkpoint(
                    model=unwrap_or_self(model), task=epoch_task, epoch=epoch,
                    val_metrics=val_metrics, avg_loss=avg_loss,
                    val_loss=val_metrics["val_loss"],
                    run_id=logger.run_id if logger is not None else None,
                    is_best=is_best, is_last=(epoch == epochs - 1),
                    is_smoke=test_mode, input_size=input_size,
                    coco_mix_ratio=coco_mix_ratio,
                    augmentation=aug_record,
                    lr_decay_start=decay_start,
                    # What this epoch actually trained on, when it is not what
                    # the task key implies (a BDD refine runs under the refine
                    # task key but on its own data).
                    dataset_name=dataset_label,
                )
            except Exception as ckpt_err:  # pylint: disable=broad-exception-caught
                # A checkpoint that fails to persist must not silently pass as
                # a healthy epoch; surface it, but let training continue so the
                # run can still finish and report the failure. Count it and tag
                # the run, because the print is invisible from MLflow and a run
                # whose checkpoints all failed looked healthy there.
                checkpoint_upload_failures += 1
                print(f"{Fore.RED}[ERROR] Failed to save epoch {epoch+1} checkpoint: {ckpt_err}{Style.RESET_ALL}")
                print(traceback.format_exc())
                if logger is not None:
                    try:
                        logger.tag_checkpoint_upload_failures(checkpoint_upload_failures)
                    except Exception:  # pylint: disable=broad-exception-caught
                        print(f"{Fore.RED}[ERROR] Could not tag checkpoint_upload_failures: {traceback.format_exc()}{Style.RESET_ALL}")

        # Print epoch summary
        print(f"\n{Fore.YELLOW}{'=' * 70}")
        print(f"Epoch {epoch+1}/{epochs} Complete:")
        print(f"  Average Loss: {avg_loss:.4f}")
        if is_segmentation:
            print(f"  IoU Score: {avg_acc1:.2f}%")
            print(f"  Dice Coefficient: {avg_acc5:.2f}%")
        else:
            print(f"  Top-1 Accuracy: {avg_acc1:.2f}%")
            print(f"  Top-5 Accuracy: {avg_acc5:.2f}%")
        print(f"{'=' * 70}{Style.RESET_ALL}\n")

        # Check early stopping on the monitored validation metric. The default
        # is validation loss, not train loss: stopping on train loss keeps
        # training while the model is already overfitting. Under a mix of a
        # run selected on its primary dataset, the key is the primary's
        # suffixed value, not the mean -- the same key the best epoch reads.
        monitor_key = monitored_key_from(
            early_stop_monitor, mixed=mix_val_loader is not None,
            best_on_primary=best_on_primary, primary_suffix=primary_suffix)
        if monitor_key not in val_metrics:
            raise ValueError(
                f"early-stop monitor {early_stop_monitor!r} resolves to "
                f"{monitor_key!r}, which is not among this run's validation "
                f"metrics {sorted(val_metrics)}")
        monitored_value = val_metrics[monitor_key]
        if early_stopper.step(monitored_value):
            print(f"{Fore.RED}Early stopping triggered at epoch {epoch+1} "
                  f"({monitor_key} did not improve){Style.RESET_ALL}\n")
            break

    return {
        "best_epoch": best_epoch,
        "best_score": None if best_epoch is None else best_score,
        "best_state": best_state,
        "best_metrics": best_metrics,
        "best_train_loss": best_train_loss,
        "last_epoch": epoch,
        "oom_skipped_total": oom_skipped_total,
    }

# ---------------------------------------------------------------------------
# Distributed (two-GPU DDP) support.
#
# Everything in this section is inert unless WORLD_SIZE > 1: a single-process
# run creates no process group, wraps no model and reduces no number, so its
# behaviour -- and its checkpoints -- are what they were before this section
# existed. The gate is `world_size > 1` and nothing else.
# ---------------------------------------------------------------------------


class _NullLogger:
    """A logger that does nothing, for the ranks that do not own the run.

    A distributed run has exactly one MLflow run and one set of MinIO keys, so
    every rank but the owner must write nothing. Rather than thread `if
    model_logger is not None` through every call site, the non-owning ranks get
    this: the call sites stay as they are, and the writes stop here.
    """

    def __getattr__(self, name):
        if name.startswith("__") and name.endswith("__"):
            raise AttributeError(name)

        def _noop(*args, **kwargs):
            return None

        return _noop


@dataclass(frozen=True)
class DistributedContext:
    """Which rank this process is, out of how many, and how they talk."""

    rank: int = 0
    local_rank: int = 0
    world_size: int = 1
    backend: str = "gloo"

    @property
    def is_distributed(self) -> bool:
        return self.world_size > 1

    @property
    def is_main(self) -> bool:
        """Whether this rank owns the run, its artifacts and its log."""
        return self.rank == 0


def distributed_context_from_env() -> DistributedContext:
    """Read torchrun's environment, defaulting to a plain single-process run.

    torchrun sets WORLD_SIZE, RANK and LOCAL_RANK for every worker; a run
    started as `python -m ecs_hcm_model.core.trainer` has none of them and gets
    the world size of one that keeps the single-process path intact. LOCAL_RANK
    falls back to RANK because torchrun sets both, and a launcher that sets only
    RANK still means "rank r on this node's device r".
    """
    world_size = _env_int("WORLD_SIZE", 1, minimum=1)
    rank = _env_int("RANK", 0, minimum=0)
    local_rank = _env_int("LOCAL_RANK", rank, minimum=0)
    if rank >= world_size:
        raise ValueError(
            f"RANK={rank} is outside WORLD_SIZE={world_size}; the launcher's "
            f"rank numbering and its world size disagree."
        )
    # The backend is only ever consumed by `init_process_group_for`, which a
    # single-process run never reaches, so asking about CUDA here would be a
    # GPU probe that a one-rank run does not need -- and it would happen before
    # `set_env_vars` has set CUDA_DEVICE_ORDER, which is exactly the kind of
    # ordering the run's own record must be able to assume.
    backend = (
        ("nccl" if torch.cuda.is_available() else "gloo")
        if world_size > 1
        else "gloo"
    )
    return DistributedContext(
        rank=rank,
        local_rank=local_rank,
        world_size=world_size,
        backend=backend,
    )


def _init_process_group_accepts_device_id() -> bool:
    """Whether this torch's init_process_group takes a device_id keyword.

    Read from the installed function's signature, never from a version string:
    the keyword appeared in torch 2.3, and the trainer image pins 2.1.
    """
    try:
        return "device_id" in inspect.signature(dist.init_process_group).parameters
    except (TypeError, ValueError):
        return False


def process_group_kwargs(backend: str, device: torch.device) -> dict:
    """The keyword arguments init_process_group_for hands to torch.

    The single place the trainer builds them, so a test can build the same
    kwargs and call the real init_process_group with them.
    """
    kwargs = {"backend": backend}
    if backend == "nccl" and _init_process_group_accepts_device_id():
        kwargs["device_id"] = device
    return kwargs


def init_process_group_for(ctx: DistributedContext, device: torch.device) -> None:
    """Bring up the process group for a multi-rank run. A no-op otherwise.

    Must run *after* `torch.cuda.set_device` (nccl binds the rank's device at
    init) and *before* the GPU preflight, which is a collective agreement across
    the ranks.
    """
    if not ctx.is_distributed:
        return
    if dist.is_available() and dist.is_initialized():
        return
    dist.init_process_group(**process_group_kwargs(ctx.backend, device))
    print(
        f"Process group up: backend={ctx.backend} world_size={ctx.world_size} "
        f"rank={ctx.rank} local_rank={ctx.local_rank}"
    )


def destroy_process_group_if_initialized() -> None:
    """Tear the process group down, whatever state the run ended in.

    Called from the run's `finally` on every rank: a process that finishes (or
    fails) while its group is still up leaves the peers' collectives to time out
    rather than fail fast.
    """
    if not dist.is_available() or not dist.is_initialized():
        return
    try:
        dist.destroy_process_group()
    except Exception:  # pylint: disable=broad-exception-caught
        print(f"{Fore.RED}[ERROR] destroy_process_group failed: {traceback.format_exc()}{Style.RESET_ALL}")


def _is_wrapped(model) -> bool:
    return isinstance(model, torch.nn.parallel.DistributedDataParallel)


def unwrap_or_self(model):
    """The module under a DDP wrapper, or the model itself.

    Validation, state-dict capture and checkpoint writing go through this. A
    DDP-wrapped forward is a *collective* (it syncs buffers), so validating
    through the wrapper would drag the non-validating ranks into broadcasts they
    are not expecting; and a wrapped state_dict carries DDP's `module.` prefix,
    which the single-process loader cannot read.
    """
    return getattr(model, "module", model)


def wrap_model_for_ddp(model, ctx: DistributedContext):
    """Wrap the model for a multi-rank run, or return it untouched.

    `find_unused_parameters=True` is required here, not merely permitted: the
    warmup stages freeze the backbone (and unfreeze it stage by stage) *after*
    this wrap, and DDP fixes the set of parameters it synchronises at
    construction -- it is `[p for p in module.parameters() if p.requires_grad]`
    at that moment. Wrapping before the freeze puts every parameter in the
    reducer, so an unfrozen stage is synchronised; without the flag, DDP then
    raises "Expected to have finished reduction in the prior iteration" on the
    first warmup epoch, because parameters it was told to expect produced no
    gradient.

    The alternative -- re-wrapping after each unfreeze -- is rejected
    (design.md §7): it also works, but it replaces the module object mid-run,
    and the tests pin the wrapper's identity as stable across an unfreeze.
    """
    if not ctx.is_distributed:
        return model
    if _is_wrapped(model):
        return model
    if ctx.backend == "nccl":
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[ctx.local_rank],
            output_device=ctx.local_rank,
            find_unused_parameters=True,
        )
    else:
        model = torch.nn.parallel.DistributedDataParallel(
            model, find_unused_parameters=True
        )
    print(
        f"Rank {ctx.rank}/{ctx.world_size}: DDP wrap "
        f"(find_unused_parameters=True, backend={ctx.backend})"
    )
    return model


@contextlib.contextmanager
def accum_step_sync(model, is_group_end: bool):
    """Defer DDP's gradient all-reduce to the last micro-batch of a group.

    Micro-batches that do not end the accumulation group run under `no_sync()`,
    so a group of N pays one all-reduce at its final backward instead of one per
    micro-batch. A no-op for an unwrapped model, so single-process accumulation
    is untouched.
    """
    if is_group_end or not _is_wrapped(model):
        yield
        return
    with model.no_sync():
        yield


@contextlib.contextmanager
def join_uneven_epochs(model):
    """DDP's uneven-inputs context manager, or a no-op for a plain model.

    The ranks' shards differ in size by up to one example, so one rank can
    produce a different number of batches than another (about 1 in B at batch
    size B). Inside this block the rank that runs out of batches leaves the
    epoch cleanly while its peer finishes the last one, instead of waiting in an
    all-reduce the peer will never enter.
    """
    if _is_wrapped(model):
        with model.join():
            yield
        return
    yield


def _reduce_epoch_numbers(device, numbers):
    """Sum a mapping of the epoch's counters across the ranks.

    ONE all-reduce per epoch carries everything the epoch accumulates, so the
    numbers that decide something (counts, the pooled losses) and the numbers
    that are only logged (samples, dropped regions, failed images) cannot
    disagree about which reduction they came from. SUM is the right reduction for
    all of them: the losses are running sums whose ratios are pooled means, and
    the counts sum because every rank's batches are disjoint.

    An integer comes back an integer -- a summed count of 0 still compares equal
    to 0 -- and a single-rank run gets its own values back untouched.
    """
    keys = list(numbers)
    if (
        not dist.is_available()
        or not dist.is_initialized()
        or dist.get_world_size() < 2
    ):
        return {key: numbers[key] for key in keys}
    tensor = torch.tensor(
        [float(numbers[key]) for key in keys], dtype=torch.float64, device=device
    )
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return {
        key: int(round(value)) if isinstance(numbers[key], int) else float(value)
        for key, value in zip(keys, tensor.tolist())
    }


def failed_rank_marker_path() -> str:
    """Where a dying rank leaves its number for a peer that outlives it."""
    return os.environ.get(
        "ECS_HCM_FAILED_RANK_FILE", "/tmp/ecs_hcm_failed_rank"
    )


def clear_failed_rank(path: str | None = None) -> None:
    """Drop a marker file left by an earlier run on this node."""
    target = path or failed_rank_marker_path()
    try:
        os.remove(target)
    except OSError:
        pass


def record_failed_rank(rank: int, path: str | None = None) -> None:
    """Leave this rank's number for a peer to read after this process dies.

    A failing rank cannot log its own death: only the rank that owns the run has
    a logger, and the collective that would tell it is exactly the hang this
    design avoids. So the number goes in a file. The ranks of one training pod
    share a node, so /tmp is common to them; the write is atomic
    (write-then-rename) because the reader may be in the middle of a crash too.
    """
    target = path or failed_rank_marker_path()
    tmp = f"{target}.{os.getpid()}"
    try:
        with open(tmp, "w", encoding="utf-8") as handle:
            handle.write(str(int(rank)))
        os.replace(tmp, target)
    except OSError as exc:
        print(
            f"{Fore.YELLOW}[WARNING] could not record the failing rank "
            f"{rank}: {exc}{Style.RESET_ALL}"
        )


def read_failed_rank(path: str | None = None):
    """The rank a peer recorded as failing, or None if none did."""
    target = path or failed_rank_marker_path()
    try:
        with open(target, encoding="utf-8") as handle:
            return int(handle.read().strip())
    except (OSError, ValueError):
        return None


def _all_reduce_min_float(device, value: float) -> float:
    """The smallest of this number across the ranks. A no-op for a single rank."""
    if not dist.is_available() or not dist.is_initialized():
        return float(value)
    if dist.get_world_size() < 2:
        return float(value)
    tensor = torch.tensor([float(value)], dtype=torch.float64, device=device)
    dist.all_reduce(tensor, op=dist.ReduceOp.MIN)
    return float(tensor.item())


def _all_reduce_max_int(device, value: int) -> int:
    """The largest of this number across the ranks. A no-op for a single rank."""
    if not dist.is_available() or not dist.is_initialized():
        return int(value)
    if dist.get_world_size() < 2:
        return int(value)
    tensor = torch.tensor([int(value)], dtype=torch.int64, device=device)
    dist.all_reduce(tensor, op=dist.ReduceOp.MAX)
    return int(tensor.item())


def fetch_prepared_tree_once_per_node(ctx: DistributedContext, root):
    """Fetch the prepared tree once per node, and let the node's other ranks in.

    Every process on a node shares that node's /tmp, so two ranks downloading
    the same objects into the same directory race: MinIO writes each object to
    `<file>.<hash>.part.minio` and renames it, and the losing rank's rename
    raises `FileNotFoundError` on the file its peer just moved. The node's
    local rank 0 therefore does the download alone; every rank then waits at a
    barrier; and the rest fetch afterwards, which returns at once on the
    manifest rank 0 has written and doubles as their proof the tree is whole.

    local rank, not global rank: the unit of a shared download directory is one
    node. A single-process run, or a run with no process group, has nothing to
    coordinate and takes the plain path.
    """
    if not ctx.is_distributed or not dist.is_available() or not dist.is_initialized():
        return fetch_prepared_tree(root)

    if ctx.local_rank == 0:
        # If this raises, the peers below stay parked; torchrun tears the job
        # down when one rank exits non-zero, which releases them. Nothing is
        # left to clean up here.
        fetched = fetch_prepared_tree(root)
    dist.barrier()
    if ctx.local_rank == 0:
        return fetched
    return fetch_prepared_tree(root)


def _broadcast_object(device, obj, src: int = 0):
    """Give every rank rank `src`'s object, bit for bit.

    Used for the validation result, so every rank decides on the same numbers
    instead of recomputing a metric that is only meant to be measured once.
    """
    if not dist.is_available() or not dist.is_initialized():
        return obj
    if dist.get_world_size() < 2:
        return obj
    payload = [obj]
    dist.broadcast_object_list(payload, src=src, device=device)
    return payload[0]


def set_env_vars(world_size: int = 1):
    os.environ['HF_HUB_ETAG_TIMEOUT'] = "500"
    os.environ['CURL_CA_BUNDLE'] = ''
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    torch.set_float32_matmul_precision('high')
    torch.backends.cudnn.benchmark = True  # Enable cudnn autotuner for better performance
    _apply_omp_thread_limit(world_size)


def _apply_omp_thread_limit(world_size: int = 1) -> None:
    """Honour OMP_NUM_THREADS by telling torch how many CPU threads to use.

    OpenMP's own variable is read by the runtime at init, but torch keeps a
    separate intra-op thread count that defaults to the machine's core count.
    On a shared or CPU-capped pod that default oversubscribes badly, so the
    operator's OMP_NUM_THREADS is treated as the intent for both.

    In a distributed run the variable is the *node's* budget and every rank on
    that node would otherwise take all of it, so each rank takes its integer
    share (floored at one -- a rank with no threads is slower than an
    oversubscribed node, and an integer division to zero must not become
    `torch.set_num_threads(0)`). A single-process run divides by one, which
    leaves the value exactly as it was.

    A malformed value must not take the run down -- it is a performance knob,
    not a correctness one -- so it is warned about and ignored.
    """
    raw = os.environ.get("OMP_NUM_THREADS")
    if raw is None or raw == "":
        return
    try:
        threads = int(raw)
    except ValueError:
        print(
            f"{Fore.YELLOW}[WARNING] OMP_NUM_THREADS={raw!r} is not an integer; "
            f"leaving torch's thread count at its default "
            f"({torch.get_num_threads()}){Style.RESET_ALL}"
        )
        return
    if threads < 1:
        print(
            f"{Fore.YELLOW}[WARNING] OMP_NUM_THREADS={raw!r} is below 1; "
            f"leaving torch's thread count at its default "
            f"({torch.get_num_threads()}){Style.RESET_ALL}"
        )
        return
    if world_size > 1:
        threads = max(1, threads // world_size)
        print(
            f"OMP_NUM_THREADS is the node's budget; rank's share is {threads} "
            f"of the world size {world_size}"
        )
    torch.set_num_threads(threads)
    print(f"torch.set_num_threads({threads}) from OMP_NUM_THREADS")

def create_model_optimizer(model, backbone_learning_rate = 0.0, head_learning_rate = 3e-4):
    backbone_params = list(model.backbone.parameters())
    head_params = [p for n,p in model.named_parameters() if "backbone" not in n]

    model_optimizer = torch.optim.AdamW(
        [
            {"params": backbone_params, "lr": backbone_learning_rate},
            {"params": head_params, "lr": head_learning_rate},
        ],
        weight_decay=0.01,
    )
    return model_optimizer

def train_hcm_imagenet1k(
        new_model: bool = True,
        environment: str = "development",
        finetune: bool = False,
        coco: bool = False,
        model_base: Literal["models", "finetuned-models", "coco-models"] = "models",
        max_epochs: int = 5,
        batch_size: int = 8,
        test_mode: bool = False,
        warmup: bool = True,
        max_batches_per_epoch: int = 0,
        input_size: int = DEFAULT_INPUT_SIZE,
        coco_mix_ratio: float = 0.0,
        init_from: str | None = None,
        init_decoder_arch: str = DECODER_V2,
        backbone: str = DEFAULT_BACKBONE,
        neck_channels: int | None = None,
        mask_stride: int = MASK_STRIDE_DEFAULT,
        scale_jitter: tuple[float, float] | None = None,
        color_jitter: float | None = None,
        grad_accum_steps: int = 1,
        lr_decay_start: int | None = None,
        bdd: bool = False,
        bdd_data_root: str | None = None,
        bdd_coco_mix_share: float = 0.5,
        early_stop_monitor: str = "val_loss",
        early_stop_patience: int = 5,
        amodal_outline: bool = False,
        full_outline_weight: float = 1.0,
):
    # A warm start is its own decision, never a resume: it builds a fresh model
    # and copies weights into it, so it cannot also continue a lineage.
    if init_from and not new_model:
        raise InitFromError(
            f"init_from={init_from!r} starts a new lineage from a fresh model, so "
            f"it cannot be combined with a resume (new_model=False)"
        )
    if init_from and init_decoder_arch not in (DECODER_V1, DECODER_V2):
        raise InitFromError(
            f"init_decoder_arch must be one of {(DECODER_V1, DECODER_V2)} "
            f"(got {init_decoder_arch!r})"
        )
    # The architecture choices are refused here for the same reason as the
    # augmentation and the ratio below: a mistyped backbone or mask stride is a
    # caller error and should cost a stack trace, not an MLflow run, a GPU
    # preflight and an ImageNet download. validate_arch is the constructor's own
    # check, so the two cannot disagree.
    neck_channels = validate_arch(backbone, neck_channels, mask_stride)
    if init_from:
        # A typo in the name is a usage error before MLflow or preflight start.
        try:
            parse_checkpoint_spec(init_from)
        except ValueError as exc:
            raise InitFromError(str(exc)) from exc
    # Refused before anything is installed, configured or downloaded: a ratio
    # that cannot mean anything is a caller error, and it should cost a stack
    # trace rather than an MLflow run, a GPU preflight and an HF download.
    # Finite first: nan compares False both ways, so it would slip past the
    # sign checks, run as 0 and log NaN; inf would fail only after the COCO
    # download, when the batch count overflows.
    if not math.isfinite(coco_mix_ratio):
        raise CocoaMixRatioError(
            f"coco_mix_ratio must be a finite number (got {coco_mix_ratio})"
        )
    if coco_mix_ratio < 0:
        raise CocoaMixRatioError(
            f"coco_mix_ratio must not be negative (got {coco_mix_ratio})"
        )
    if coco_mix_ratio > 0 and not finetune:
        raise CocoaMixRatioError(
            f"coco_mix_ratio={coco_mix_ratio} mixes COCO batches into a COCOA "
            f"refine, so it requires finetune=True (this run has "
            f"finetune={finetune}, coco={coco})"
        )
    # BDD100K is a segmentation refine with its own data, so it takes the same
    # flag the refine path takes and refuses every other task outright rather
    # than pick one of them silently.
    if bdd and not finetune:
        raise ValueError(
            f"bdd=True trains a BDD100K segmentation refine, so it requires "
            f"finetune=True (this run has finetune={finetune}, coco={coco})"
        )
    if bdd and coco:
        raise ValueError(
            f"coco=True selects the COCO-only task and bdd=True selects the "
            f"BDD100K refine; a run trains one or the other"
        )
    # The share is a fraction of the epoch's samples, which is what the plan
    # fixed at half. It is converted to the trainer's mix ratio below (that
    # ratio is extra mix batches per primary batch, not a share), so it is
    # validated here on its own terms: 0 is off, and 1 would leave no primary
    # data to train on.
    if bdd:
        if not math.isfinite(bdd_coco_mix_share) or not 0.0 <= bdd_coco_mix_share < 1.0:
            raise ValueError(
                f"bdd_coco_mix_share must be a finite number in [0, 1) "
                f"(got {bdd_coco_mix_share})"
            )
        if coco_mix_ratio != 0:
            raise ValueError(
                f"bdd_coco_mix_share and coco_mix_ratio both set the COCO mix; "
                f"pass only one (got bdd_coco_mix_share={bdd_coco_mix_share}, "
                f"coco_mix_ratio={coco_mix_ratio})"
            )
        # share s means s of the epoch's samples come from COCO, so the mix
        # needs s/(1-s) batches per primary batch: 0.5 -> one for one.
        coco_mix_ratio = mix_ratio_for_share(bdd_coco_mix_share)
    # Same reasoning as the ratio, and validated here rather than in the
    # trainer loop: a refused jitter costs no MLflow run, no preflight and no
    # download.
    scale_jitter, color_jitter = validate_augmentation(
        scale_jitter, color_jitter, trains_coco=bool(finetune or coco)
    )
    accum_steps = aug_accum_steps(grad_accum_steps)
    # Resolved here, like the accumulation count, so a refused value costs no
    # MLflow run, no preflight and no download.
    decay_start = lr_decay_start_for(lr_decay_start, warmup=warmup)
    # What the transform and the loop were told to do, as recorded in each
    # epoch checkpoint and logged below. Built here so the record cannot
    # disagree with the values the run was given.
    augmentation_record = _augmentation_record(scale_jitter, color_jitter, accum_steps)

    model_logger = None
    # Bound before the try so the finally block can tell "training never
    # returned" (an OOM trip, a kill) from "training finished".
    train_result = None
    # torchrun's environment, read once, before anything is created: WORLD_SIZE=1
    # (the default) is the plain single-process run, and it takes the path that
    # existed before this change -- no process group, no wrapping, no reduction.
    dist_ctx = distributed_context_from_env()
    if dist_ctx.is_distributed:
        # A marker left by an earlier run on this node would name a rank that is
        # not the one that loses this time.
        clear_failed_rank()
    # Install before anything else: a SIGTERM arriving after the MLflow run is
    # created but before the handler exists would kill the process outright and
    # strand the run in RUNNING, which is the failure this prevents.
    if not install_kill_handlers():
        # Not fatal: training proceeds, but a pod deletion now terminates the
        # process outright and leaves the run RUNNING. Say so rather than let it
        # look protected.
        print(
            f"{Fore.YELLOW}[WARNING] Could not install SIGTERM/SIGINT handlers "
            f"(not the main thread, or the runtime forbids it). A cancellation "
            f"will kill this process without ending the MLflow run."
            f"{Style.RESET_ALL}"
        )
    try:

        # The thread budget is the node's, shared by every rank on it; at a
        # world size of one this is the value it always was.
        set_env_vars(world_size=dist_ctx.world_size)

        # The logger is created BEFORE the device is chosen and before the GPU
        # preflight. The preflight exists to fail a run that cannot possibly
        # train, and that failure is only useful if it is recorded: with the
        # logger created here, the preflight's RuntimeError reaches the
        # `except Exception` arm below and the run ends FAILED with an `error`
        # tag. Creating it after the preflight meant a refused GPU left no
        # trace in MLflow at all.
        datasent_name = ""
        if bdd:
            # The refine's own data, and what it was mixed with, exactly as the
            # COCOA line states its mix.
            datasent_name = "BDD100K+MSCOCO" if coco_mix_ratio > 0 else "BDD100K"
        elif finetune:
            # "COCOA", or "COCOA+MSCOCO" when COCO is mixed back in, so the
            # tracking run says which data it actually trained on.
            datasent_name = dataset_name_for("cocoa_seg", coco_mix_ratio)
        elif coco:
            datasent_name = "MSCOCO"
        else:
            datasent_name = "imagenet-1k"
        log_config = logger_config(
            experiment_name="hcm_imagenet1k_experiment",
            dataset=datasent_name,
            max_epochs=max_epochs,
            batch_size=batch_size,
            # The size this run trains at, which is this function's parameter --
            # the same value the loaders and the checkpoint metadata get. Not
            # DEFAULT_INPUT_SIZE: a run launched with --input-size 352 must not
            # be recorded as a 192 run.
            input_size=input_size,
            environment=environment,
        )

        # Exactly one rank owns the run. On every other rank the logger is a
        # sink: constructing a real one there would open a second MLflow run for
        # the same training, and every `track`/`log_*`/tag below would be written
        # twice.
        model_logger = (
            MLFlowHCMLogger(config = log_config) if dist_ctx.is_main else _NullLogger()
        )
        model_logger.set_logger(config = log_config)

        # The mix is part of what the run *is*, so it is recorded at step 0
        # whether or not this host has a GPU, rather than only living in the
        # launch command. The augmentation and the accumulation are logged the
        # same way, as params, by train_model_for_epochs, which owns their record.
        try:
            model_logger.log_scalar("coco_mix_ratio", float(coco_mix_ratio), step=0)
        except Exception:  # pylint: disable=broad-exception-caught
            print(f"{Fore.RED}[ERROR] Could not log coco_mix_ratio at step 0: {traceback.format_exc()}{Style.RESET_ALL}")

        # The device this rank drives. torchrun hands each process a local rank,
        # and a single-process run's local rank is 0, so this is cuda:0 as before
        # for every run that does not ask for more.
        if torch.cuda.is_available():
            device = torch.device(f"cuda:{dist_ctx.local_rank}")
            torch.cuda.set_device(device)
        else:
            device = torch.device("cpu")
            print("Using CPU device")

        # The group comes up after set_device (nccl binds the rank's device at
        # init) and before the preflight, which is itself a collective.
        init_process_group_for(dist_ctx, device)

        if torch.cuda.is_available():
            # Preflight BEFORE any cache is cleared or any data is loaded: an
            # under-resourced GPU (the V100 double-booking) otherwise costs the
            # whole run -- 14,664 failed batches -- before anyone sees a number.
            # mem_get_info returns (free, total); free is the figure that
            # decides whether this run can start at all.
            free_bytes, total_bytes = torch.cuda.mem_get_info()
            free_gib = free_bytes / (1024 ** 3)
            total_gib = total_bytes / (1024 ** 3)
            min_free_gib = _env_float("ECS_HCM_MIN_FREE_GPU_GIB", MIN_FREE_GPU_GIB, minimum=0.0)
            # In a distributed run the run's free memory is the least any rank
            # has: every rank reads its own device and they agree on the
            # smallest, so all of them refuse together or all of them proceed.
            # A rank that raised on its own reading would leave its peers
            # blocked in the next collective -- a hang, not a failure.
            effective_free_gib = (
                _all_reduce_min_float(device, free_gib)
                if dist_ctx.is_distributed
                else free_gib
            )
            print(
                f"GPU memory: {free_gib:.1f} GiB free of {total_gib:.1f} GiB "
                f"(minimum to start: {min_free_gib:.1f} GiB)"
                f"{' [least free across the ranks: ' + format(effective_free_gib, '.1f') + ' GiB]' if dist_ctx.is_distributed else ''}"
            )
            # Record the numbers as params so a refused run is diagnosable from
            # MLflow, not only from the terminal it was launched in.
            for name, value in (("gpu_free_gib", free_gib), ("gpu_total_gib", total_gib)):
                try:
                    model_logger.log_scalar(name, value, step=0)
                except Exception:  # pylint: disable=broad-exception-caught
                    print(f"{Fore.RED}[ERROR] Could not log {name}: {traceback.format_exc()}{Style.RESET_ALL}")
            if effective_free_gib < min_free_gib:
                raise RuntimeError(
                    f"only {effective_free_gib:.1f} GiB free of {total_gib:.1f} GiB on "
                    f"{device}, but {min_free_gib:.1f} GiB is required. Another "
                    f"process is probably holding this GPU -- free it and "
                    f"restart, or run on a different device. (Override with "
                    f"ECS_HCM_MIN_FREE_GPU_GIB if this threshold is wrong.)"
                )
            torch.cuda.empty_cache()
            print(f"Using CUDA device: {torch.cuda.get_device_name(device)}")

        finetune_config = False
        if coco or finetune:
            finetune_config = True

        model = load_hcm_model(
            new_model=new_model,
            environment=environment,
            finetune=finetune_config,
            device=device,
            model_base=model_base,
            num_segmentation_classes=2,
            out_stride=16,
            task=resolve_task(finetune=finetune, coco=coco),
            input_size=input_size,
            # A fresh model gets the fixed v2 decoder; a resume keeps
            # the version of the checkpoint it loads.
            decoder_arch=init_decoder_arch if init_from else DECODER_V2,
            # The ladder and the mask branch of a fresh model. A resume keeps
            # what its checkpoint recorded, so these reach only the new-model
            # path (see load_hcm_model).
            backbone=backbone,
            neck_channels=neck_channels,
            mask_stride=mask_stride,
            # Reaches the new-model path only; a resume keeps what its
            # checkpoint recorded.
            amodal_outline=amodal_outline,
        )
        if init_from:
            warm_start_from(model, init_from, model_logger)

        model = model.to(device)
        # The flag and the model have to agree. A resume keeps its checkpoint's
        # architecture, so the flag alone cannot add a head -- and a head with
        # no outline targets in its batches would be an unused parameter (DDP
        # refuses the backward) as well as a head trained on nothing. Refused
        # here, before the loaders are built and the datasets downloaded.
        has_outline_head = getattr(unwrap_or_self(model), "outline_head", None) is not None
        if amodal_outline and not has_outline_head:
            raise InitFromError(
                "--amodal-outline builds the outline head on a NEW model, but this "
                "run loads a checkpoint without one. Start a new model "
                "(--new-model, or --init-from for a warm start) or drop the flag."
            )
        if has_outline_head and not amodal_outline:
            raise InitFromError(
                "the loaded checkpoint carries an amodal outline head but "
                "--amodal-outline is off, so nothing in the batches would train "
                "it. Pass --amodal-outline to keep training the head."
            )
        train_loader = None
        val_loader = None
        # The COCO data mixed into a refine, and the validation set that makes
        # the forgetting visible. Both stay None unless a ratio was asked for,
        # so the unmixed path builds no extra loaders and downloads no COCO.
        mix_train_loader = None
        mix_val_loader = None
        if finetune:
            # A BDD refine reads its own prepared tree; a COCOA refine reads the
            # HuggingFace dataset. Everything else about the refine -- the
            # optimizer, the segmentation metrics, the best-metric rule -- is
            # the same code path, which is why --bdd also requires --finetune.
            primary_dataset = BDD_DATASET_NAME if bdd else "shunk031/COCOA"
            if bdd:
                # A pod starts with an empty disk: the prepared tree is fetched
                # once and kept across restarts (fetch_prepared_tree skips what
                # is already there). A readable manifest of the right format is
                # the signal that the copy finished, so a half-download is redone.
                # The ranks of a node share its /tmp, so the download is done by
                # one process per node; see fetch_prepared_tree_once_per_node.
                local = (bdd_data_root or os.getenv("BDD100K_DATA_ROOT")
                         or "/tmp/bdd100k/prepared")
                bdd_data_root = str(fetch_prepared_tree_once_per_node(dist_ctx, local))
            train_loader = get_training_dataloader(
                dataset_name=primary_dataset,
                is_dataset_local=False,
                bdd_data_root=bdd_data_root,
                batch_size=batch_size,
                workers=4,
                # Training data is sharded across the ranks: each rank reads its
                # own disjoint slice of the epoch's stream.
                global_rank=dist_ctx.rank,
                world_size=dist_ctx.world_size,
                input_size=input_size,
                with_outline=amodal_outline,
                # Both None unless the flags were given (validate_augmentation
                # normalises them), and refused outright on a non-COCO path.
                scale_jitter=scale_jitter,
                color_jitter=color_jitter,
            )
            val_loader = get_validation_dataloader(
                dataset_name=primary_dataset,
                is_dataset_local=False,
                bdd_data_root=bdd_data_root,
                batch_size=batch_size,
                workers=1,
                # Validation is NOT sharded in this version: it is replicated,
                # and only the rank that owns the run scores it. The result is
                # broadcast, so every rank still decides on the same numbers.
                global_rank=0,
                world_size=1,
                input_size=input_size,
                with_outline=amodal_outline,
            )
            if coco_mix_ratio > 0:
                # The same loaders the COCO-only task trains and validates on,
                # at the same input size, so a mixed epoch sees both datasets in
                # one geometry. Both datasets are built by the same code path --
                # get_*_dataloader selects on "coco" appearing in the name, which
                # is true of "shunk031/COCOA" and "shunk031/MSCOCO" alike -- so
                # each yields the same (pixel_values, segmentation_mask) batch.
                mix_train_loader = get_training_dataloader(
                    dataset_name="shunk031/MSCOCO",
                    is_dataset_local=False,
                    batch_size=batch_size,
                    workers=4,
                    # Training data is sharded across the ranks: each rank reads
                    # its own disjoint slice of the epoch's stream.
                    global_rank=dist_ctx.rank,
                    world_size=dist_ctx.world_size,
                    input_size=input_size,
                    with_outline=amodal_outline,
                    scale_jitter=scale_jitter,
                    color_jitter=color_jitter,
                )
                mix_val_loader = get_validation_dataloader(
                    dataset_name="shunk031/MSCOCO",
                    is_dataset_local=False,
                    batch_size=batch_size,
                    workers=1,
                    global_rank=0,
                    world_size=1,
                    input_size=input_size,
                    with_outline=amodal_outline,
                )
        else:
            if coco:
                train_loader = get_training_dataloader(
                    dataset_name="shunk031/MSCOCO",
                    is_dataset_local=False,
                    batch_size=batch_size,
                    workers=4,
                    # Training data is sharded across the ranks: each rank reads
                    # its own disjoint slice of the epoch's stream.
                    global_rank=dist_ctx.rank,
                    world_size=dist_ctx.world_size,
                    input_size=input_size,
                    with_outline=amodal_outline,
                    scale_jitter=scale_jitter,
                    color_jitter=color_jitter,
                )

                val_loader = get_validation_dataloader(
                    dataset_name="shunk031/MSCOCO",
                    is_dataset_local=False,
                    batch_size=batch_size,
                    workers=1,
                    global_rank=0,
                    world_size=1,
                    input_size=input_size,
                    with_outline=amodal_outline,
                )
            else:
                train_loader = get_training_dataloader(
                    dataset_name="imagenet-1k",
                    is_dataset_local=False,
                    batch_size=batch_size,
                    workers=4,
                    # Training data is sharded across the ranks: each rank reads
                    # its own disjoint slice of the epoch's stream.
                    global_rank=dist_ctx.rank,
                    world_size=dist_ctx.world_size,
                    input_size=input_size,
                    with_outline=amodal_outline,
                    # Always None here: validate_augmentation refuses jitter on
                    # this path. Passed through so the loader's backstop raises
                    # if a caller ever reaches it with jitter set.
                    scale_jitter=scale_jitter,
                    color_jitter=color_jitter,
                )

                val_loader = get_validation_dataloader(
                    dataset_name="imagenet-1k",
                    is_dataset_local=False,
                    batch_size=batch_size,
                    workers=1,
                    global_rank=0,
                    world_size=1,
                    input_size=input_size,
                    with_outline=amodal_outline,
                )

        # test_mode keeps its long-standing 32-batch smoke default, but only
        # when no explicit cap was asked for: an explicit --max_batches_per_epoch
        # wins, which is what makes a *longer* smoke run (and any capped
        # diagnostic run outside test_mode) possible.
        if test_mode and not max_batches_per_epoch:
            max_batches_per_epoch = 32
        print(
            f"Batch cap: "
            f"{max_batches_per_epoch if max_batches_per_epoch > 0 else 'none (full epoch)'}"
        )
        # Begin training - use custom training loop
        print("Beginning training...")


        task = resolve_task(finetune=finetune, coco=coco)

        train_result = train_model_for_epochs(
            model=model,
            epochs=max_epochs,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            logger=model_logger,
            test_mode=test_mode,
            max_batches_per_epoch=max_batches_per_epoch,
            warmup=warmup,
            finetune=finetune,
            coco=coco,
            task=task,
            checkpoint_every_epoch=True,
            input_size=input_size,
            coco_mix_ratio=coco_mix_ratio,
            mix_train_loader=mix_train_loader,
            mix_val_loader=mix_val_loader,
            # A BDD refine is selected on BDD val, not on the BDD/COCO mean.
            best_on_primary=bdd,
            # ...and its own validation keys say `bdd`, not `cocoa`.
            primary_suffix=MIX_SUFFIX_BDD if bdd else MIX_SUFFIX_PRIMARY,
            # A BDD refine records its own dataset name rather than the refine
            # task's COCOA label.
            dataset_label=datasent_name if bdd else None,
            augmentation=augmentation_record,
            grad_accum_steps=accum_steps,
            lr_decay_start=decay_start,
            # Which rank this is, so the loop knows whether to wrap, reduce and
            # cooperate. None means a plain single-process run.
            distributed=dist_ctx,
            early_stop_monitor=early_stop_monitor,
            early_stop_patience=early_stop_patience,
            amodal_outline=amodal_outline,
            full_outline_weight=full_outline_weight,
        )

        # Upload the best checkpoint (by the task's val metric) and the last.
        # This runs for smoke runs too, so the upload path is exercised there.
        # Only the rank that owns the run: the ranks' weights are identical, so a
        # second upload would be the same bytes written twice under the same key,
        # and its "best" tag would race the first.
        run_id = model_logger.run_id if model_logger is not None else None
        model_uri = None
        if dist_ctx.is_main:
            bucket, model_name = checkpoint_target(task)
            model_uploader = ModelUploaderAndDownloader(bucket_name=bucket)
        if dist_ctx.is_main and train_result["best_state"] is not None:
            # Restore the best weights so the uploaded checkpoint is the best
            # epoch, not whichever epoch training happened to stop on.
            # A copy, not model.state_dict(): that returns the live tensors,
            # which the best-weights load below overwrites in place, so the
            # "restore" in the finally would reload the best weights.
            current_state = {
                k: v.detach().clone() for k, v in model.state_dict().items()
            }
            model.load_state_dict(train_result["best_state"])
            try:
                model_uri = model_uploader.upload_model(
                    model=model,
                    model_name=f"{model_name}_best",
                    architecture="hcm",
                    metadata={
                        "mlflow_run_id": run_id,
                        "task": task,
                        "dataset_name": datasent_name or dataset_name_for(task, coco_mix_ratio),
                        "coco_mix_ratio": float(coco_mix_ratio),
                        "epoch": train_result["best_epoch"],
                        # Under a mix this carries each dataset's IoU as well as
                        # their mean (see merge_mixed_val_metrics), so the best
                        # checkpoint says what it was best at on both.
                        "val_metrics": {
                            k: float(v) for k, v in (train_result["best_metrics"] or {}).items()
                        },
                        "train_loss": train_result["best_train_loss"],
                        "git_commit": os.environ.get("GIT_SHA", "unknown"),
                        "is_best": True,
                        "is_last": False,
                        "is_smoke": test_mode,
                        # The spatial size this run trained at (see
                        # save_epoch_checkpoint); a consumer reads it rather than
                        # assuming 192.
                        "input_size": int(input_size),
                        "resumed_from": resumed_from_of(model),
                        "init_from": init_from_of(model),
                        "out_stride": out_stride_of(model),
                        "decoder_arch": decoder_arch_of(model),
                        # The ladder and the mask branch (backbone,
                        # neck_channels, mask_stride), so a consumer rebuilds the
                        # architecture the weights were trained in. The model's
                        # own record, so it cannot drift from the model config.
                        **arch_record_of(model),
                        # How the run that produced this model was trained:
                        # the promoted checkpoint on the Jetson has to say
                        # whether its augmentation and accumulation differed
                        # from the run it is being compared against.
                        "augmentation": dict(augmentation_record),
                        # The decay start this run trained under: a schedule
                        # ablation differs from its baseline only here, so the
                        # promoted model has to carry it.
                        "lr_decay_start": decay_start,
                    },
                )
            finally:
                # Leave the model as it was so the caller isn't surprised by a
                # weights swap as a side effect of uploading.
                model.load_state_dict(current_state)

        if model_logger is not None:
            model_logger.register(
                                model_uri=model_uri,
                                finish_status="FINISHED"
                                )
    except KilledException as exc:
        # Arrives from a SIGTERM/SIGINT handler (SignalKilled) or from the
        # explicit kill path. The run ends KILLED and is tagged with why, so
        # the dashboard shows a cancellation rather than a mystery.
        #
        # ORDER MATTERS: tag_* writes to the live run, kill() closes it, and
        # MLflow silently drops set_tag on a closed run. Tagging after kill()
        # loses the reason with no error -- which is exactly what it did.
        if model_logger is not None:
            signame = getattr(exc, "signame", None)
            if signame:
                try:
                    model_logger.tag_kill_reason(signame)
                except Exception: # pylint: disable=broad-exception-caught
                    print(f"{Fore.RED}[ERROR] Could not tag kill reason: {traceback.format_exc()}{Style.RESET_ALL}")
            model_logger.kill()

        raise exc

    except Exception as exc: # pylint: disable=broad-exception-caught
        print(f"Error: {str(exc)}")
        print(traceback.format_exc())
        if model_logger is not None:
            # Any failure leaves the run FAILED with the error recorded --
            # previously the run was simply abandoned in RUNNING.
            try:
                model_logger.register(
                    model_uri=None,
                    finish_status="FAILED",
                    exc_msg=str(exc),
                )
            except Exception: # pylint: disable=broad-exception-caught
                print(f"{Fore.RED}[ERROR] Could not mark run FAILED: {traceback.format_exc()}{Style.RESET_ALL}")
        # A distributed failure is never one rank's business alone. Each failing
        # rank leaves its number in the shared marker file, and the rank that
        # owns the run names the loss in the run's tags -- read BEFORE writing
        # its own, so a peer that died first is the one recorded. Every rank then
        # re-raises, so every process exits non-zero; the peers find out as soon
        # as their next collective fails, and nothing here waits for a rank that
        # has already stopped.
        if dist_ctx.is_distributed:
            try:
                loser = read_failed_rank() if dist_ctx.is_main else None
                record_failed_rank(dist_ctx.rank)
                if dist_ctx.is_main:
                    model_logger.tag_failed_rank(
                        loser if loser is not None else dist_ctx.rank
                    )
            except Exception:  # pylint: disable=broad-exception-caught
                print(f"{Fore.RED}[ERROR] Could not record the failing rank: {traceback.format_exc()}{Style.RESET_ALL}")
        raise exc
    finally:
        # Belt-and-braces: if anything above failed to end the run, end it now.
        # _end_run is idempotent, so the normal FINISHED path is unaffected and
        # a KILLED run is not downgraded to FAILED.
        if model_logger is not None:
            # Tag how many batches were skipped for OOM. Set before _end_run so
            # it lands on the live run -- MLflow drops set_tag on a closed run.
            # `train_result` is only bound once training returns, so an OOM
            # trip (which raises) reports 0 here; the batch_/train_ metrics
            # logged on the way out carry the real counts for that case.
            if train_result is not None:
                try:
                    model_logger.tag_oom_skipped_total(
                        train_result.get("oom_skipped_total", 0)
                    )
                except Exception:  # pylint: disable=broad-exception-caught
                    print(f"{Fore.RED}[ERROR] Could not tag oom_skipped_total: {traceback.format_exc()}{Style.RESET_ALL}")
            model_logger._end_run("FAILED")
        # Put the process's signal disposition back as we found it.
        restore_kill_handlers()
        # And take the process group down on every path out of this function,
        # including the failing ones: a rank that leaves while its peers are
        # still in the group leaves them waiting out a collective timeout
        # instead of failing fast.
        destroy_process_group_if_initialized()
