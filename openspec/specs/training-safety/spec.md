# training-safety Specification

## Purpose

The guards that decide whether a training run is allowed to start, kept running, or failed loudly. A run that cannot make progress must cost seconds rather than hours, and a run that trains nothing must never be reported as a completed epoch.

## Requirements

### Requirement: A GPU preflight refuses an under-resourced device before training

When CUDA is available, `train_hcm_imagenet1k` MUST read free and total GPU memory before
clearing the cache or building any dataloader. Free memory below the minimum MUST raise, and
the minimum MUST default to 8.0 GiB with an `ECS_HCM_MIN_FREE_GPU_GIB` override. The free and
total figures MUST be logged to MLflow as `gpu_free_gib` and `gpu_total_gib` before the check,
so a refused run is diagnosable after the fact. In a distributed run each rank MUST read its
own device's free memory and the ranks MUST agree on the smallest free figure before any rank
raises, so every rank refuses together or every rank proceeds together; a rank that raised
alone would leave its peers blocked in the next collective. The figures logged to the run
belong to the rank that owns the run and MUST be that rank's own device's.

#### Scenario: A GPU with too little free memory is refused

- **WHEN** the device reports 4 GiB free of 16 GiB and the minimum is 8.0 GiB
- **THEN** the run raises with a message naming the free figure, the total and the required minimum

#### Scenario: A healthy GPU proceeds

- **WHEN** the device reports 12 GiB free of 16 GiB and the minimum is 8.0 GiB
- **THEN** training continues past the preflight

#### Scenario: The preflight does not touch GPU memory APIs on CPU

- **WHEN** CUDA is not available
- **THEN** the device is CPU and no GPU memory query is made

#### Scenario: The refused run is recorded as failed

- **WHEN** the preflight raises
- **THEN** the MLflow run ends FAILED carrying an `error` tag with the message

#### Scenario: All ranks refuse together

- **WHEN** one rank of a distributed run reports free memory below the minimum and its peers do not
- **THEN** every rank raises on the reduced figure rather than only the rank that read it

### Requirement: Skipped-for-OOM batches are counted and bounded

The training loop MUST count every batch it pulled from the loader as attempted, including the
ones that then failed, and MUST count the OOM failures separately. The run MUST be failed when
the consecutive OOM skips exceed a limit defaulting to 20 (`ECS_HCM_MAX_CONSECUTIVE_OOM`), and
when the OOM skip fraction exceeds 0.01 (`ECS_HCM_MAX_OOM_FRACTION`) once at least 200 batches
have been attempted. A batch that trains successfully MUST reset the consecutive counter. The
200-batch floor MUST keep a single early skip from failing a healthy run. In a distributed run
the limits are judged twice, on two different sets of counts, and only one of them is the run's.
The run-level judgement is the end-of-epoch gate: the attempted and skipped counts MUST be summed
across ranks and the consecutive-skip length MUST be the largest run of any one rank, so the limit
is applied once to the whole run's numbers -- a run of skips split across ranks is not one
unbroken run. The per-batch check inside the loop is a rank-local early guard over that rank's own
counts: it MAY fail the run sooner than the run-level numbers alone would, and it MUST NOT itself
be reduced, because that branch is entered only by the rank that hit the OOM, and an all-reduce
there would leave that rank in a collective its peers never enter -- the mismatched collectives
the distributed-training requirement forbids.

#### Scenario: Consecutive skips cross the limit

- **WHEN** more than the configured maximum batches fail in an unbroken run
- **THEN** a `RuntimeError` names the consecutive count and the limit

#### Scenario: A successful batch resets the consecutive count

- **WHEN** OOM failures are interleaved with successful batches and never reach the limit unbroken
- **THEN** no error is raised and the skips are counted

#### Scenario: The fraction limit only applies past the floor

- **WHEN** the skip fraction is above the limit but fewer than 200 batches have been attempted
- **THEN** no fraction error is raised
- **AND** the same fraction after 200 attempts does raise

#### Scenario: The skip fraction is re-judged at the end of the epoch

- **WHEN** every OOM skip landed early in the epoch and no later skip triggered the in-loop check
- **THEN** the fraction is still evaluated once the epoch's attempted count is final

#### Scenario: Scattered single skips do not fail a run

- **WHEN** isolated OOM skips occur under both limits
- **THEN** training continues and the skips are counted

#### Scenario: A split across ranks crosses the fraction limit at the epoch gate

- **WHEN** in a distributed run no rank is judged on its own fraction because each is under the 200-batch floor, while the run's summed attempts pass the floor and its summed skips cross the fraction limit
- **THEN** the run fails at the end-of-epoch gate

#### Scenario: Split skips are not one unbroken run

- **WHEN** two ranks of a distributed run each skip 15 consecutive batches, under the limit of 20
- **THEN** no consecutive-limit error is raised, because the run's consecutive length is the largest run on any one rank -- two ranks at 15 do not make 30

### Requirement: An epoch that trains nothing fails the run

When an epoch attempted at least one batch and trained none, the trainer MUST log that epoch's
OOM metrics and then raise a `RuntimeError` naming the epoch and the attempted count. The check
MUST apply whether or not the run is in test mode. A loader that presented no batches at all
MUST NOT trip this check. In a distributed run the attempted and trained counts MUST be summed
across ranks before the check, so an epoch in which every rank trained some batches is not an
epoch that trained nothing and one in which no rank did is caught.

#### Scenario: An all-OOM epoch raises

- **WHEN** every batch of an epoch fails with CUDA OOM
- **THEN** the run raises with a message stating the epoch trained 0 batches

#### Scenario: The zero-batch failure applies in test mode too

- **WHEN** the same all-OOM epoch happens with `test_mode=True`
- **THEN** the run still raises

#### Scenario: An empty loader is not a failed epoch

- **WHEN** an epoch's loader yields no batches
- **THEN** the zero-batch check does not fire

#### Scenario: A distributed epoch is judged on the run's batch count

- **WHEN** in a distributed run no rank trains a batch but at least one attempted one
- **THEN** the run raises on the summed count rather than only the ranks that trained nothing

### Requirement: Validation never silently skips a batch

`evaluate` MUST NOT skip or break out of a validation batch on CUDA OOM; it MUST raise a `RuntimeError` naming the batch index. When validation completes with zero batches it MUST raise, because an average over nothing reads as a perfect result. A partial validation is not an error but MUST warn, naming how many batches the reported metrics actually cover.

For segmentation, `evaluate` MUST NOT score a batch whose `valid` flags are all false (the collate's placeholder for a batch in which every image failed); it MUST skip it and count it as failed images: the loader's batch size, or at least one image when the loader does not state a batch size. It MUST also count, for every scored batch shorter than the loader's batch size, the shortfall as failed images. The returned segmentation metrics MUST include `val_failed_images`, the training loop MUST log it every epoch including when it is zero, and in a mixed-dataset run the unsuffixed value MUST be the sum of the per-dataset counts, which MUST also be logged individually. A validation consisting only of placeholders presents zero batches and MUST therefore raise, with an error that states the failed-image count.

#### Scenario: A validation OOM names the batch

- **WHEN** a validation batch raises CUDA OOM
- **THEN** a `RuntimeError` containing that batch index is raised

#### Scenario: A validation that ran nothing raises

- **WHEN** the validation loop completes having validated zero batches out of those presented
- **THEN** a `RuntimeError` is raised rather than returning a zero loss

#### Scenario: A partial validation warns but returns

- **WHEN** validation validates some but not all presented batches
- **THEN** a warning names the validated and presented counts
- **AND** metrics are returned averaged over the batches that ran

#### Scenario: A placeholder batch is not scored

- **WHEN** a segmentation validation batch has every `valid` flag false
- **THEN** it does not contribute to `val_loss`, `val_iou` or `val_dice`
- **AND** `val_failed_images` increases by the loader's batch size, or by at least one when the loader states no batch size

#### Scenario: A short batch counts its shortfall

- **WHEN** a scored segmentation batch has fewer samples than the loader's batch size
- **THEN** `val_failed_images` increases by the difference

#### Scenario: Only placeholders raises

- **WHEN** every segmentation validation batch is a placeholder
- **THEN** a `RuntimeError` is raised rather than returning metrics
- **AND** its message states how many images failed to load

#### Scenario: The count is logged every epoch and summed across datasets

- **WHEN** a segmentation epoch validates, with or without failures
- **THEN** `val_failed_images` is logged for that epoch
- **AND** in a mixed run it equals the sum of the per-dataset counts, each also logged

### Requirement: Early stopping is driven by validation loss

Early stopping MUST be evaluated against a validation metric, never a training one, and the
default monitor MUST be validation loss, so a run that does not ask for another metric keeps
the behaviour it had: patience five epochs, and an epoch that does not improve on the best
validation loss seen so far MUST count against the patience, so a plateau counts as
non-improvement. Training MUST stop when the counter reaches the patience.

The monitor MUST be selectable, with `val_iou` the one alternative, because a segmentation
refine is judged by IoU and the two curves do not move together at the end of a fine-tune.
The comparison direction MUST follow the monitor — a loss improves downward, a score upward
— and the run MUST NOT accept a direction flag that can disagree with the chosen monitor. The
monitor MUST read the same validation series the run selects its best epoch on: when the run is
selected on its primary dataset while also validating a mixed dataset, the stopper MUST follow
that dataset's own metric rather than the two-set mean, so a run whose own metric is still
improving is not stopped because the mixed dataset's is not. A monitor the run's validation
metrics do not carry MUST raise an error naming the monitor and the metrics that were present,
rather than stopping never or failing with a bare lookup error. A patience of zero MUST disable
stopping; a negative patience MUST be refused as a usage error before any epoch runs.

#### Scenario: A plateau eventually stops a run

- **WHEN** validation loss fails to improve for five consecutive epochs
- **THEN** training breaks out of the epoch loop

#### Scenario: An improving epoch resets the counter

- **WHEN** validation loss improves on the best seen so far
- **THEN** the non-improving counter is reset to zero

#### Scenario: The monitor selects the direction

- **WHEN** the monitor is `val_iou` and it rises above the best seen so far
- **THEN** the counter is reset, and an epoch that does not exceed the best counts against
  the patience

#### Scenario: The default is validation loss with patience five

- **WHEN** a run passes no early-stop flag
- **THEN** the monitor is `val_loss` and the patience is five

#### Scenario: A disabled patience never stops the run

- **WHEN** the patience is zero
- **THEN** the run is never stopped for a non-improving epoch, whatever the monitor does

#### Scenario: A monitor the run does not produce is named

- **WHEN** the selected monitor is absent from the run's validation metrics
- **THEN** an error names the monitor and the metrics that were present

#### Scenario: A mixed run stops on the series it is selected on

- **WHEN** a run is selected on its primary dataset's IoU and also validates a mixed dataset
- **THEN** the stopper is stepped with the primary dataset's own IoU, not the mean of the two

### Requirement: Torch's CPU thread count follows OMP_NUM_THREADS

`set_env_vars` MUST apply the operator's `OMP_NUM_THREADS` to `torch.set_num_threads`, because
torch's intra-op default is the machine's core count and oversubscribes a shared or
CPU-capped pod. An unset, non-integer or sub-one value MUST warn and leave torch's thread
count at its default rather than take the run down, since this is a performance knob and not a
correctness one. In a distributed run the value is the node's budget shared by every rank, so
each rank MUST apply its integer share of it, floored at one; a single-process run MUST apply
the value undivided.

#### Scenario: A valid thread count is applied

- **WHEN** `OMP_NUM_THREADS` is a positive integer
- **THEN** `torch.set_num_threads` is called with it

#### Scenario: A malformed thread count is ignored

- **WHEN** `OMP_NUM_THREADS` is not an integer or is below 1
- **THEN** a warning is printed and the run continues

#### Scenario: A distributed run divides the thread budget

- **WHEN** `OMP_NUM_THREADS` is 8 and the world size is 2
- **THEN** each rank sets torch's thread count to 4, and a world size of 1 would set 8

### Requirement: A killed run ends KILLED rather than RUNNING

The trainer MUST install SIGTERM and SIGINT handlers before creating work, and MUST end the MLflow run as KILLED with the signal name tagged as the kill reason. The kill reason MUST be tagged before the run is ended, because MLflow silently drops a tag written to a closed run. The run end MUST be idempotent so a normal finish is not downgraded to FAILED.

#### Scenario: A SIGTERM ends the run as KILLED

- **WHEN** SIGTERM arrives during training
- **THEN** the run ends KILLED with the signal named in its kill reason tag

#### Scenario: A normal finish is not overwritten

- **WHEN** training completes and the run is then ended again by the exit path
- **THEN** the second end is a no-op and the status stays FINISHED
