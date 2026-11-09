# training-lr-schedule Specification

## Purpose

Everything that decides what learning rate each parameter group trains at: the staged warmup that unfreezes the backbone from the head outwards, the constant head learning rate, and the cosine decay that takes over after warmup. A run's lr curve is the difference between a model that plateaus at epoch 5 and one that keeps improving, and it is the first thing to check when a run stops learning.

## Requirements

### Requirement: Warmup is a fixed five-epoch staged lr and unfreeze schedule in epoch order

When `warmup=True`, `train_model_for_epochs` MUST run five warmup epochs before the decay phase: epoch 0 trains the head alone with the backbone fully frozen and backbone lr 0.0; epochs 1 and 2 unfreeze backbone `layer4` at backbone lr 1e-4; epochs 3 and 4 unfreeze `layer3` and `layer4` at backbone lr 5e-5. Each stage MUST be applied at the START of its epoch, and each stage MUST re-freeze the whole backbone and unfreeze exactly its named stages, so stage membership is absolute rather than cumulative.

#### Scenario: Epoch 0 freezes the backbone and zeroes its lr

- **WHEN** a warmup run begins epoch 0
- **THEN** the backbone's parameters have `requires_grad=False`
- **AND** `optimizer.param_groups[0]["lr"]` is set to 0.0

#### Scenario: Epochs 1 and 2 unfreeze layer4 at 1e-4

- **WHEN** a warmup run reaches epoch 1 or epoch 2
- **THEN** `layer4` is unfrozen and every other backbone stage is frozen
- **AND** `optimizer.param_groups[0]["lr"]` is set to 1e-4

#### Scenario: Epochs 3 and 4 unfreeze layer3 and layer4 at 5e-5

- **WHEN** a warmup run reaches epoch 3 or epoch 4
- **THEN** `layer3` and `layer4` are both unfrozen and stages below them remain frozen
- **AND** `optimizer.param_groups[0]["lr"]` is set to 5e-5

#### Scenario: No warmup leaves the backbone trainable for the whole run

- **WHEN** `warmup=False`
- **THEN** every backbone parameter has `requires_grad=True`
- **AND** the backbone is left in train mode with no per-epoch re-freezing

### Requirement: The head learning rate is constant and set at optimizer creation

The optimizer MUST hold exactly two parameter groups, backbone at index 0 and head at index 1. Warmup MUST NOT modify the head group's lr. At creation the head lr MUST be 3e-5 for COCO segmentation and finetuning runs and 3e-4 for classification runs, and the backbone lr at creation MUST be 0.0 when warmup is on and 1e-5 (COCO/finetune) or 1e-4 (classification) when warmup is off.

#### Scenario: Classification with warmup starts the head at 3e-4

- **WHEN** a classification run with `warmup=True` builds its optimizer
- **THEN** `optimizer.param_groups[1]["lr"]` is 3e-4
- **AND** `optimizer.param_groups[0]["lr"]` is 0.0

#### Scenario: COCO segmentation starts the head at 3e-5

- **WHEN** a COCO segmentation run builds its optimizer
- **THEN** `optimizer.param_groups[1]["lr"]` is 3e-5

#### Scenario: The head lr survives every warmup epoch unchanged

- **WHEN** a warmup run logs `lr_head` at the start of each of epochs 0 through 4
- **THEN** every logged value equals the head lr the optimizer was created with

### Requirement: Cosine lr decay applies to each param group from the epoch warmup ends

When `warmup=True` the decay phase MUST start at epoch 5 by default, and when `warmup=False` at
epoch 0, unless `--lr-decay-start N` names a different epoch, in which case `N` MUST win. From its
first decay epoch, each parameter group MUST take the lr it holds at that moment as its decay base
and then follow, for decay-epoch index k of K, `base * (f + (1 - f) * 0.5 * (1 + cos(pi * k / (K - 1))))`,
reaching `base * f` at the final epoch. The decay base MUST be read off the optimizer at the first
decay epoch, not recomputed from the creation arguments. A negative `N` MUST be refused as a usage
error before any training starts, and a run that omits the flag MUST take, every epoch, exactly the
lr this schedule gave before the flag existed.

#### Scenario: The first decay epoch is exactly the base

- **WHEN** a warmup run reaches epoch 5 after stage 3 left the backbone at 5e-5
- **THEN** the backbone lr at epoch 5 is exactly 5e-5

#### Scenario: The last epoch is exactly base times the factor

- **WHEN** a 15-epoch warmup run reaches epoch 14
- **THEN** the backbone lr is 5e-5 times the minimum-lr factor
- **AND** the head lr is 3e-4 times that same factor

#### Scenario: The decay is non-increasing across the decay phase

- **WHEN** the backbone lr is read at every epoch from `decay_start` to the last epoch
- **THEN** each epoch's value is less than or equal to the previous epoch's value

#### Scenario: Epochs during warmup are untouched by the decay

- **WHEN** a warmup run of 15 or more epochs logs the backbone lr at epochs 0 through 4
- **THEN** the logged values are 0.0, 1e-4, 1e-4, 5e-5 and 5e-5, identical to a run without the decay feature

#### Scenario: A run that ends at or before decay_start is unchanged

- **WHEN** a warmup run is given 5 epochs, so the last epoch equals `decay_start`
- **THEN** the decay schedule is never applied to either param group
- **AND** the run's lr values are bit-for-bit the pre-decay values

#### Scenario: No warmup decays from epoch zero

- **WHEN** a 4-epoch run has `warmup=False`
- **THEN** the decay applies at epochs 0, 1, 2 and 3
- **AND** the epoch-0 lr is the group's creation lr unchanged

#### Scenario: The decay start is configurable

- **WHEN** a warmup run is given `--lr-decay-start 2`
- **THEN** the decay's first epoch is epoch 2, whose lr is the group's lr at that moment

#### Scenario: A negative decay start is refused

- **WHEN** a run is given `--lr-decay-start -1`
- **THEN** it is refused as a usage error before any training starts

#### Scenario: Omitting the flag changes nothing

- **WHEN** a run omits `--lr-decay-start`
- **THEN** every epoch's lr equals the value the pre-flag schedule gives for that epoch, exactly

### Requirement: The decay handles runs too short for a multi-epoch ramp

`lr_for_epoch` MUST return the decay base unchanged when the decay phase has no epochs (K <= 0) and unchanged when it has exactly one (K == 1), because the cosine formula divides by K - 1 and a one-epoch phase is both its own first and last epoch.

#### Scenario: A run entirely inside warmup does not decay

- **WHEN** `lr_for_epoch` is called with `epoch < decay_start` or with `epochs <= decay_start`
- **THEN** it returns the base it was given

#### Scenario: A single decay epoch returns the base

- **WHEN** `lr_for_epoch` is called with exactly one decay epoch available
- **THEN** it returns the base rather than dividing by zero

### Requirement: The minimum-lr factor is configurable and rejects unusable values

The fraction of the base lr that the decay ramps down to MUST default to 0.05 and MUST be overridable with `ECS_HCM_MIN_LR_FACTOR`. A value outside the half-open interval (0, 1] MUST be warned about and replaced by the default, never obeyed and never fatal.

#### Scenario: A factor of one produces a flat schedule

- **WHEN** `ECS_HCM_MIN_LR_FACTOR=1` and a warmup run decays
- **THEN** every decay epoch's lr equals the decay base, so the schedule is flat

#### Scenario: Values at or below zero and above one fall back

- **WHEN** `ECS_HCM_MIN_LR_FACTOR` is one of 0, 0.0, -0.1, 1.5 or 2
- **THEN** the effective factor is the default 0.05
- **AND** a warning naming the offending value is printed

#### Scenario: A non-finite factor falls back without crashing

- **WHEN** `ECS_HCM_MIN_LR_FACTOR` is `nan`, `inf` or `-inf`
- **THEN** the effective factor is the default 0.05
- **AND** the run continues rather than raising

#### Scenario: A usable override is honoured

- **WHEN** `ECS_HCM_MIN_LR_FACTOR` is a value strictly between 0 and 1
- **THEN** the decay's final epoch is the decay base times that value

### Requirement: A run whose length makes the decay dead says so

When the requested epochs end at or before `decay_start`, the trainer MUST print a warning naming the epoch count and the unrealised decay start, and MUST still run every epoch with today's lr values.

#### Scenario: A five-epoch warmup run warns that decay is inactive

- **WHEN** `warmup=True` and `epochs=5`
- **THEN** a warning states that decay would start at epoch 5 and is therefore inactive
- **AND** all five epochs run to completion

### Requirement: A decay start inside warmup tapers each group's stage lr

When `--lr-decay-start N` is earlier than the last warmup epoch, the decay factor for epoch e MUST
be applied on top of whatever the warmup stage sets for that epoch, for every parameter group,
using the same cosine and minimum-lr factor: `lr_e = stage_lr_e * factor(e)`. The backbone's stage
lr for an epoch MUST be the value the stage sets for that epoch, and after the stages end MUST be
held at the last stage value rather than re-read from the group's already-decayed lr, so the factor
is applied once per epoch and never compounded. The head group, which warmup never modifies, MUST
be tapered off its creation lr by the same factor.

#### Scenario: A decay start at epoch zero tapers the stage values

- **WHEN** a 5-epoch warmup run has `--lr-decay-start 0`
- **THEN** at the last epoch each group's lr is its stage lr times the minimum-lr factor

#### Scenario: The head group is tapered too

- **WHEN** the same run reaches its last epoch
- **THEN** the head lr is its creation lr times the same factor

#### Scenario: The taper does not compound

- **WHEN** consecutive decay epochs are logged during warmup
- **THEN** each epoch's lr is that epoch's stage lr times that epoch's factor alone, not the previous epoch's lr times a factor

#### Scenario: A start at or after the last stage matches the untapered schedule

- **WHEN** `N` is 5 or later
- **THEN** every epoch's lr equals the value the decay requirement above gives for the same epoch

### Requirement: Every epoch checkpoint records the lr decay start

Every epoch checkpoint's metadata MUST carry an `lr_decay_start` key holding the run's effective
decay start -- the value given on the command line, or the default (5 with warmup, 0 without) when
it was omitted -- beside the augmentation settings, so two checkpoints differing only in the decay
start are distinguishable after the fact.

#### Scenario: An epoch checkpoint carries the decay start

- **WHEN** an epoch checkpoint is saved by a run started with `--lr-decay-start 2`
- **THEN** its metadata contains `lr_decay_start` equal to 2

#### Scenario: An omitted flag is recorded as the default

- **WHEN** a warmup run that omitted the flag saves an epoch checkpoint
- **THEN** its metadata contains `lr_decay_start` equal to 5

### Requirement: The backbone's batch norm stays in eval after the warmup stages

When `warmup=True`, the backbone's batch-normalisation layers MUST be in eval mode during every epoch, not only the five warmup epochs. The loop puts the whole model back in training mode at the start of each epoch, so an epoch at or past `WARMUP_EPOCHS` MUST pin them again, exactly as the warmup stages do. An epoch at or past `WARMUP_EPOCHS` MUST otherwise be unchanged by this: it MUST NOT re-freeze or unfreeze a stage, and the backbone group's learning rate MUST stay where the decay phase holds it.

#### Scenario: An epoch past warmup pins the backbone's batch norm again

- **WHEN** a warmup run begins an epoch at or past `WARMUP_EPOCHS`
- **THEN** every batch-normalisation module in the backbone is in eval mode while that epoch trains

#### Scenario: The post-warmup pin changes nothing but the batch-norm mode

- **WHEN** a warmup run reaches an epoch at or past `WARMUP_EPOCHS`
- **THEN** the same backbone stages are trainable as at the last warmup stage
- **AND** the backbone group's learning rate is the one the decay phase holds

#### Scenario: Without warmup the batch norm is left as the trainer set it

- **WHEN** `warmup=False`
- **THEN** no epoch re-pins the backbone's batch norm to eval
