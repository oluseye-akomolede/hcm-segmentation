# experiment-logging Specification

## Purpose

What a training run records in MLflow and how a reader finds it: which metric keys exist for which task, what step each is recorded at, and how a post-mortem finds the evidence for a run that failed. A metric that is never written is a question nobody can answer later.

## Requirements

### Requirement: Metric keys are namespaced by phase and by task

Training and validation metrics MUST be written under distinct `train_*` and `val_*` key prefixes so a train value and a val value never overwrite each other at the same step. A classification run MUST log `acc1` and `acc5` and MUST NOT log IoU or Dice; a segmentation run MUST log `iou` and `dice` and MUST NOT log accuracies. Every metric block MUST also carry a `loss`.

#### Scenario: Classification logs accuracies and no IoU

- **WHEN** a classification epoch's metrics are recorded
- **THEN** the keys are `train_acc1`, `train_acc5`, `train_loss` and the matching `val_*` keys
- **AND** no `train_iou` or `train_dice` key is written

#### Scenario: Segmentation logs IoU and Dice and no accuracies

- **WHEN** a segmentation epoch's metrics are recorded
- **THEN** the keys are `train_iou`, `train_dice`, `train_loss` and the matching `val_*` keys
- **AND** no `train_acc1` key is written

#### Scenario: A caller that passes segmentation numbers positionally still logs IoU and Dice

- **WHEN** a segmentation metric object is built with IoU and Dice supplied positionally rather than by keyword
- **THEN** the IoU and Dice keys are still produced

### Requirement: Percentage metrics are guarded into the 0-100 range

Any metric named in the percentage set (`acc1`, `acc5`, `iou`, `dice`) MUST be checked to be finite and within 0 to 100 inclusive before it is written, and a value outside that range MUST raise rather than be recorded. This turns a double-scaled value into a loud failure instead of garbage in the tracking store.

#### Scenario: A double-scaled accuracy raises

- **WHEN** an accuracy of 6006 is logged
- **THEN** an out-of-range error is raised and the value is not written

#### Scenario: A non-numeric percentage raises

- **WHEN** a percentage metric carries a value that cannot be converted to a number
- **THEN** an out-of-range error is raised

#### Scenario: A boundary value passes

- **WHEN** a percentage metric is exactly 0 or exactly 100
- **THEN** it is written

### Requirement: Raw magnitudes bypass the percentage guard

The activation magnitude metrics and the learning rates MUST be written through a scalar path that writes the key verbatim and applies no percentage range check, because these are raw magnitudes that legitimately exceed 100.

#### Scenario: A large activation magnitude is recorded

- **WHEN** an activation maximum of 60000 is logged
- **THEN** the value is written under its exact key rather than rejected as an out-of-range percentage

#### Scenario: A learning rate is written under its exact key

- **WHEN** `lr_backbone` is logged at a step
- **THEN** the value reaches MLflow under the key `lr_backbone`

### Requirement: The learning rate of both param groups is logged every epoch

At the start of every epoch, after the schedule has set the lrs, the trainer MUST log `lr_backbone` and `lr_head` at a step equal to the epoch index, and MUST print one line naming the epoch and both values. This MUST happen for warmup epochs too, so the schedule reads as one continuous curve rather than only from where decay begins.

#### Scenario: Both groups are logged once per epoch

- **WHEN** an epoch begins with a logger attached
- **THEN** exactly one `lr_backbone` and one `lr_head` value are recorded for that epoch

#### Scenario: The step is the epoch index

- **WHEN** the lr is recorded for epoch 5
- **THEN** the recorded step is 5

#### Scenario: A run with no logger does not fail

- **WHEN** the trainer runs with no logger attached
- **THEN** the lr print still happens and nothing raises

### Requirement: OOM skips are reported from outside the batch-count gate

The per-batch skip counter MUST be logged as `batch_oom_skipped` on the first skip of an epoch and thereafter at the periodic cadence, and the per-epoch total MUST be logged as `train_oom_skipped` every epoch including when it is zero. Both MUST be emitted even when the epoch trained no batches at all, because a run that skipped every batch is exactly the case these metrics exist to expose. At run end the run-total MUST be written as a tag named `oom_skipped_total`.

#### Scenario: A batch skip is visible immediately

- **WHEN** the first batch of an epoch is skipped for OOM
- **THEN** `batch_oom_skipped` is logged at that batch's step without waiting for the periodic cadence

#### Scenario: A zero-batch epoch still reports its skips

- **WHEN** every batch of an epoch fails with OOM so no batch trained
- **THEN** `batch_oom_skipped` and `train_oom_skipped` are both logged for that epoch

#### Scenario: A healthy epoch reports a zero skip count

- **WHEN** an epoch skips no batches
- **THEN** `train_oom_skipped` is logged with the value 0

#### Scenario: The run total is tagged at run end

- **WHEN** a run finishes after skipping batches
- **THEN** the run carries an `oom_skipped_total` tag with the run-wide count

### Requirement: Activation magnitudes are logged per batch and folded per epoch

The activation maxima MUST be logged under batch-scoped keys at the periodic batch cadence and folded into an epoch maximum, which MUST then be logged under `train_<stage>_absmax` at the epoch's step. The model's running maxima MUST be reset after each epoch's fold so an epoch's figure is its own maximum rather than a running total over the run.

#### Scenario: The epoch figure includes the tail of the epoch

- **WHEN** the epoch ends with activation maxima not yet flushed
- **THEN** whatever the model is still holding is folded in before the epoch figure is logged

#### Scenario: Reset keeps epochs independent

- **WHEN** a new epoch begins after the previous epoch's fold
- **THEN** the model's recorded maxima start empty so the new epoch cannot inherit the previous one's peak

### Requirement: The run ends idempotently and a failure is recorded on it

Every exit path MUST funnel through one idempotent run-end so a run is never left RUNNING and a second end is a no-op. A failure MUST end the run FAILED with the exception message stored both as an `error` parameter and, truncated, as an `error` tag.

#### Scenario: A failure leaves a diagnosable run

- **WHEN** training raises
- **THEN** the run ends FAILED and carries the exception message as an `error` tag

#### Scenario: A second end does not change the status

- **WHEN** the run has already been ended and the exit path ends it again
- **THEN** the recorded status is unchanged

#### Scenario: A checkpoint upload failure is countable from the run

- **WHEN** a per-epoch checkpoint fails to persist
- **THEN** the failure is counted and the count is written as a `checkpoint_upload_failures` tag

### Requirement: The tracking environment selects the MLflow server

The logger MUST take an environment of `development` or `production` and MUST point at the development tracking server for the former and the production server for the latter, and an unrecognised environment MUST raise rather than silently default. The run MUST be tagged and parametrised with the dataset, batch size, input size, epoch count and environment.

#### Scenario: Development points at the development server

- **WHEN** the logger is created with the development environment
- **THEN** the tracking URI is the development server

#### Scenario: An unknown environment raises

- **WHEN** the logger is created with an environment that is neither development nor production
- **THEN** a `ValueError` is raised

#### Scenario: The run is tagged with its configuration

- **WHEN** a run starts
- **THEN** its tags carry the batch size, input size, max epochs, environment and dataset

#### Scenario: The run records the size it trained at

- **WHEN** a training run is started for an input size other than the module default
- **THEN** that size is the run's `input_size` parameter and `input_size` tag, and not the default the trainer falls back to when unset

### Requirement: Run-level tags are written to the run that produced them

Every run-level tag the logger writes MUST be addressed to that logger's own run id through
the run-addressed client API, so the tag lands on the run that produced it whether that run
is still open or has already ended, and so the write MUST NOT create a run. A logger that
holds no run id MUST write no tag rather than let the call attach itself to a new run. This
covers the tag methods the trainer calls from its exit path, which all run after the run
may already have been ended.

#### Scenario: A tag after the run has ended lands on the original run and creates no run

- **WHEN** a run is ended and the logger is then asked to tag it
- **THEN** the tag is readable on the original run with the value asked for
- **AND** no additional run exists in the experiment

#### Scenario: A tag while the run is live lands on that run

- **WHEN** the logger is asked to tag a still-open run
- **THEN** the tag is readable on that run

#### Scenario: The trainer's exit path does not spawn a scrap run

- **WHEN** a training run finishes and the exit path records its OOM skip total
- **THEN** that total is a tag on the finished run itself
- **AND** the experiment holds exactly the one run the training produced

#### Scenario: A logger with no run writes nothing and creates nothing

- **WHEN** a tag method is called on a logger that never had a run started
- **THEN** it returns without raising and no run is created

#### Scenario: A checkpoint upload failure count survives the run end

- **WHEN** per-epoch checkpoint uploads failed during a run and the run has ended
- **THEN** the count is readable as a tag on that same run

### Requirement: No logger write after a run end may create a run

Any write the logger performs after the run has ended MUST be a no-op or a run-addressed
write, and MUST NOT be performed through an API that picks a run implicitly. The rule is
about the class of call, not one tag: a single implicit-run write left in the post-run path
recreates the defect for that key. Writes that are guaranteed to precede the run end may
remain implicit. Metrics and params MUST be written by run id. A `register()` or `kill()`
after the run has ended MUST NOT change the run's recorded status or params. A logger that
never started a run MUST write nothing.

#### Scenario: The metric path is not the offender

- **WHEN** the audit for implicit-run writes reachable after a run end is performed
- **THEN** the metric, param and dataset writes are run-addressed or guarded by the run
  end, and the only implicit write left (`set_logger`'s params) runs while its run is live

#### Scenario: A late metric lands on the ended run

- **WHEN** a scalar or an epoch metric set is logged after `register()` ended the run
- **THEN** it is stored on that run and the experiment still holds one run

#### Scenario: A kill during teardown changes nothing

- **WHEN** `kill()` is called after `register(FINISHED)`
- **THEN** the run stays FINISHED, gains no `error` param, and no run is created

#### Scenario: A second register changes nothing

- **WHEN** `register()` is called again after the run has ended
- **THEN** the first outcome, its params and its tags are unchanged, and no run is created

#### Scenario: A late dataset log is dropped

- **WHEN** `log_dataset` is called after the run has ended
- **THEN** nothing is written and no run is created

#### Scenario: The stray run's key is gone from a finished run

- **WHEN** a finished run is inspected
- **THEN** its OOM skip total is present on the run itself rather than only on a separate
  run that holds nothing else

### Requirement: Every training run records its COCO mix ratio

Every training run MUST log a `coco_mix_ratio` metric at step 0, after the run is opened and before any data is loaded, including runs that do not mix (where the value MUST be 0.0). "No mix" is therefore a recorded value on the run, never an absence.

#### Scenario: A run without a mix logs zero

- **WHEN** a training run starts with the default mix ratio
- **THEN** the metric `coco_mix_ratio` is logged with value 0.0 at step 0

#### Scenario: A mixed refine logs its ratio

- **WHEN** a finetune run starts with a mix ratio of 1.0
- **THEN** the metric `coco_mix_ratio` is logged with value 1.0 at step 0

### Requirement: Runs are compared with a baseline epoch by epoch, read-only

`tools/compare_runs.py` SHALL compare a run with a baseline run (default carefree-rook-316) epoch by epoch, using only MLflow's read endpoints. It SHALL take the tracking URI from `MLFLOW_TRACKING_URI`, defaulting to the LAN dev server, and SHALL NOT create or modify runs.

#### Scenario: The table compares each epoch with the baseline's same epoch
- **WHEN** the tool runs on a run with N logged epochs
- **THEN** it prints one row per epoch with every `val_*` metric the baseline logged, `train_loss`, the epoch's wall time, and each value's delta to the baseline at that epoch
- **AND** an epoch the baseline never reached shows no delta rather than a delta against another epoch

#### Scenario: The metric keys come from the baseline
- **WHEN** the baseline logged no `val_*` metric
- **THEN** the tool exits non-zero instead of guessing key names

#### Scenario: A finish time is projected
- **WHEN** the run's planned epoch count is known from `--epochs` or its `max epochs` param, and the run is not finished
- **THEN** the tool prints the projected finish as the last val timestamp plus the mean epoch time times the remaining epochs, in UTC and Chicago time

#### Scenario: Only read endpoints are used
- **WHEN** the tool runs
- **THEN** every request is `runs/search`, `runs/get` or `metrics/get-history`
- **AND** any other endpoint is refused by an explicit error, not an assert that `python -O` would strip

#### Scenario: A RUNNING run that stopped logging is flagged
- **WHEN** the run is RUNNING and either its projected finish is already past or its last val log is older than twice the mean epoch time
- **THEN** the tool prints a line starting `stale?` saying which, since MLflow leaves a run RUNNING when its pod dies
- **AND** a run with no epochs left (finalising) is judged by the age check alone, since its projected finish is already past

#### Scenario: An unreachable or failing server is one error line
- **WHEN** a request fails with a URLError or HTTPError, times out, or gets a 200 whose body is not JSON
- **THEN** the tool prints `error: ...` naming the server and exits 2, with no traceback

#### Scenario: The printed tracking URI carries no credentials
- **WHEN** `MLFLOW_TRACKING_URI` contains `user:pass@`
- **THEN** the URI is printed, in the tracking line and in errors, without it

### Requirement: Each epoch logs its timing split

For every train epoch that reaches validation, the trainer MUST log these keys through the run's logger, once per epoch, at a step equal to the epoch index:

- `epoch_train_s`: wall time of the epoch's training loop.
- `epoch_data_wait_s`: total time blocked fetching the next batch.
- `epoch_step_s`: total time from `zero_grad` through the optimizer step and the loss read-back, for batches that trained.
- `epoch_val_s`: wall time of the epoch's validation, mixed-dataset validation included.
- `train_samples_per_s`: samples in batches that trained, divided by `epoch_train_s`.

On CUDA the device MUST be synchronised at the edges of the training loop and of validation, and MUST NOT be synchronised per batch for timing. The split MUST NOT add any per-batch logger call.

#### Scenario: Time blocked on the loader is data wait

- **WHEN** the train loader sleeps a known time before each batch
- **THEN** `epoch_data_wait_s` is at least the total sleep, and `epoch_step_s` does not include it

#### Scenario: Validation is timed apart from training

- **WHEN** the val loader sleeps and the train loader does not
- **THEN** that sleep shows up in `epoch_val_s` and not in `epoch_train_s`

#### Scenario: One value per key per epoch

- **WHEN** a run with a logger trains N epochs
- **THEN** each of the five keys is logged exactly N times, at steps 0 to N-1

#### Scenario: CUDA is synchronised at the edges only

- **WHEN** a run trains N epochs of any number of batches
- **THEN** the timing split synchronises the device three times per epoch (epoch start, end of training, end of validation), and only when the device is CUDA

### Requirement: compare_runs shows the timing split when the run has it

`tools/compare_runs.py` MUST add a column for each timing key the compared run logged (seconds in hours, samples per second as is), showing that run's own value for each epoch and `-` for an epoch without one. A run that logged none of the keys MUST get the report it got before.

#### Scenario: A run with the split

- **WHEN** the compared run logged the timing keys
- **THEN** each epoch row carries its train, wait, step and val hours and its samples per second

#### Scenario: An older run

- **WHEN** the compared run logged none of the timing keys
- **THEN** the header and rows are exactly those of a report without the split

### Requirement: A mixed run's validation time covers both datasets

For a run with a mixed-dataset validation loader, `epoch_val_s` MUST include the time of both evaluations: the run's own validation set and the mixed one.

#### Scenario: The mixed val loader takes a known time

- **WHEN** only the mixed val loader blocks, for a known total time
- **THEN** `epoch_val_s` includes that time

### Requirement: compare_runs names the untimed remainder of training

`tools/compare_runs.py` MUST show an `other_h` column, `epoch_train_s - epoch_data_wait_s - epoch_step_s` in hours, when the compared run logged all three of those keys, and MUST NOT show it otherwise. It is the part of the training loop that is neither data wait nor step: the host-to-device copy of each batch and the loop's own overhead. An epoch missing any of the three values MUST show `-`.

#### Scenario: All three parts logged

- **WHEN** an epoch logged train 7200 s, data wait 5400 s and step 1440 s
- **THEN** its `other_h` is 0.10

#### Scenario: A part never logged

- **WHEN** the run never logged `epoch_step_s`
- **THEN** the report has no `other_h` column

#### Scenario: A part missing for one epoch

- **WHEN** one epoch lacks `epoch_data_wait_s`
- **THEN** that epoch's `other_h` is `-`

### Requirement: compare_runs has a one-line brief for owner reports

`tools/compare_runs.py --brief` MUST print exactly one line, using only the same read endpoints as the full report. For a run with at least one epoch it MUST give the latest epoch as `Epoch N/P`, where N is the run's last logged MLflow step plus one (steps count from 0, so step 7 is epoch 8) and P is the planned epoch count, the run's `val_iou` at that epoch, the baseline's `val_iou` at the same epoch and the baseline's best `val_iou`, the run's mean epoch time, and the projected finish. With no epoch logged it MUST say so and give the time since the run started, projecting the finish from the baseline's mean epoch and saying that it does. Times MUST be in America/Chicago on a 12-hour clock with the zone abbreviation in force on that date (CDT or CST), and MUST name the weekday when the date is not today in Chicago. A run that is not RUNNING, or whose planned epoch count is unknown, MUST get no projected finish; a run that is not RUNNING MUST be named with its status (e.g. `Run is FAILED.`). A RUNNING run with no epoch logged after more than twice the baseline's mean epoch MUST get a note to check it is alive.

#### Scenario: Epochs logged

- **WHEN** the run's last logged step is N-1, P epochs are planned, and it is RUNNING
- **THEN** the line reads like `Epoch N/P: COCO val 69.8 (baseline at epoch N: 69.1, best 70.86). ~3.3 h/epoch, done ~7:10 PM CDT Sat.`

#### Scenario: No epochs yet

- **WHEN** the run has logged no epoch
- **THEN** the line says no epoch is logged yet, gives the elapsed time, and gives a finish projected from the baseline's mean epoch, naming it as the baseline's

#### Scenario: A finish across the DST change

- **WHEN** the projected finish falls after Chicago leaves daylight saving time
- **THEN** the finish is shown in CST

#### Scenario: A resume logging into a new MLflow run

- **WHEN** the run's first and only `val_iou` is at step 7
- **THEN** the line says `Epoch 8/P` and compares with the baseline's `val_iou` at step 7

#### Scenario: A failed run with no epochs

- **WHEN** the run is FAILED and has logged no epoch
- **THEN** the line says no epoch is logged yet and `Run is FAILED.`, with no projected finish

#### Scenario: No first epoch after twice the baseline's epoch

- **WHEN** the run is RUNNING, has logged no epoch, and has been running more than twice the baseline's mean epoch
- **THEN** the line ends with a note to check it is alive; at or under that, it has no such note

### Requirement: Every training run records its augmentation and accumulation settings

Every training run MUST record the train-time augmentation, gradient-accumulation and
learning-rate-decay-start settings it ran with as MLflow run parameters, logged once the run is
open, including a run that uses none of them. A setting that is off MUST be recorded as its
identity value, so the parameter says what the transform or the loop did rather than being
absent: the scale-jitter bounds as `1.0` and `1.0` (a factor that changes nothing), the
colour-jitter strength as `0.0`, the accumulation count as `1`, and the decay start as its
effective value -- the epoch given on the command line, or the default (5 with warmup, 0 without)
when it was omitted.

#### Scenario: A run with no augmentation logs the identity values

- **WHEN** a training run starts with the default augmentation and accumulation settings
- **THEN** the parameters `aug_scale_jitter_lo` and `aug_scale_jitter_hi` are 1.0, `aug_color_jitter` is 0.0 and `grad_accum_steps` is 1

#### Scenario: An augmented run logs what it ran with

- **WHEN** a run starts with a scale jitter of `0.75,1.5`, a colour jitter of 0.3 and an accumulation of 4
- **THEN** the same four parameters read 0.75, 1.5, 0.3 and 4

#### Scenario: An omitted decay start logs the effective default

- **WHEN** a warmup run starts without `--lr-decay-start`
- **THEN** the parameter `lr_decay_start` is 5

#### Scenario: A given decay start is logged

- **WHEN** a run starts with `--lr-decay-start 2`
- **THEN** the parameter `lr_decay_start` is 2

### Requirement: An epoch's metrics are logged once per run, not once per rank

In a distributed run each epoch's metrics, and the run's tags, MUST be written once for the
run, by the rank that owns it, so a run's history holds one value per key per step rather
than one per rank. The values written MUST be the run's reduced values, so the single
history that is kept is the whole run's and not one rank's share of it. A rank other than
the owner MUST NOT write metrics or tags, including through any logger fallback.

#### Scenario: An epoch's metric appears once

- **WHEN** a two-rank run completes its first epoch
- **THEN** each per-epoch metric key has one value for that step, not two

#### Scenario: A non-owning rank writes nothing

- **WHEN** a rank other than the owner reaches a logging point
- **THEN** no metric or tag is written by that rank

#### Scenario: The logged value is the run's, not a rank's

- **WHEN** a distributed epoch's training counters are reduced across ranks
- **THEN** the value logged for the epoch is the reduced one

### Requirement: The run records the model it started from and the code that ran it

Every run MUST record, as params on its own run, the `init_from` checkpoint it was warm-started from or the literal `none` when it started from a fresh model, the `git_commit` revision the environment reports or `unknown` when it reports none, and the `image_tag` the environment reports or `unknown` when it reports none. Together with the input size and the lr decay start this spec already requires, these MUST be enough to tell two runs apart from MLflow alone when they differ only in their starting weights or in the code that ran them.

#### Scenario: A warm-started run names its source

- **WHEN** a run is warm-started from an init-from checkpoint
- **THEN** its `init_from` param is that checkpoint's location

#### Scenario: A fresh run records no source explicitly

- **WHEN** a run starts from a fresh model
- **THEN** its `init_from` param is the literal `none`

#### Scenario: The run records the revision and image that ran it

- **WHEN** a run starts and the environment provides a revision and an image tag
- **THEN** the run's `git_commit` and `image_tag` params are those values

#### Scenario: A missing revision or image tag is recorded as unknown

- **WHEN** a run starts and the environment provides neither
- **THEN** both params are the literal `unknown`
