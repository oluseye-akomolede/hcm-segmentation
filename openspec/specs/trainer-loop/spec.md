# trainer-loop Specification

## Purpose
How the per-batch training loop turns batches into optimizer steps: when gradients are zeroed,
clipped and applied, including gradient accumulation across batches, so a small per-device batch
can train with a larger effective batch while per-epoch scheduling, logging and checkpointing keep
their meaning.

## Requirements

### Requirement: The optimizer can step every N micro-batches

The training loop SHALL accept a gradient-accumulation setting, `--grad-accum-steps N`, default
1. With N greater than 1, the batch loss MUST be divided by N and the optimizer MUST be stepped
once per N batches rather than once per batch, so the step applies the mean gradient of the N
batches. Gradients MUST be zeroed at the start of a group rather than at the start of every
batch, so they accumulate across the group, and MUST be zeroed at the start of the next group
so no gradient is counted twice. When the epoch ends with fewer than N batches accumulated, the
loop MUST step on the last batch of the epoch rather than leaving those gradients unapplied;
under distributed training, where a rank's trailing group is not shared with the other ranks,
the loop MUST instead discard the trailing group and log its size, because a rank-local step on
a group the other ranks do not share would take a different step on each rank. A batch skipped
for CUDA out-of-memory MUST abandon its group. With the AMP gradient scaler enabled, the group
MUST be clipped once (on the scaled gradients, as before) and stepped once, per group, not once
per micro-batch. The learning-rate schedule, early stopping, epoch logging and checkpointing
MUST keep their per-epoch semantics at any N. With N of 1 the training loop MUST be unchanged.

#### Scenario: One accumulated step matches one step on the concatenated batch

- **WHEN** a toy model takes one step at N=2 over two equal batches, and an identical copy takes one step at N=1 over the two batches concatenated
- **THEN** the two models' parameters agree within float tolerance

#### Scenario: The batch loss is still reported undivided

- **WHEN** an epoch runs at N=2
- **THEN** the reported train loss is the mean of the per-batch losses, not of the per-batch losses divided by 2

#### Scenario: A partial final group steps

- **WHEN** a single-process epoch of 3 batches runs at N=2
- **THEN** the optimizer has stepped twice by the end of the epoch, the second time from the gradients of the single leftover batch

#### Scenario: A distributed epoch's trailing group is discarded

- **WHEN** a distributed epoch ends with fewer than N micro-batches accumulated
- **THEN** the trailing group is discarded rather than stepped, and its size is logged

#### Scenario: N of 1 is unchanged

- **WHEN** an epoch runs at the default N of 1
- **THEN** every batch takes exactly one optimizer step, zeroed on its own boundary, as before
