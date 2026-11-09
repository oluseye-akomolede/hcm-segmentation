# distributed-training Specification

## Purpose
Lets one training run use more than one GPU on a node (torchrun with NCCL): each rank trains on
its own shard of the data, gradients are synchronised across ranks, only rank 0 saves and logs,
and the OOM and failure guards judge the run as a whole. A single-GPU run is unchanged.

## Requirements

### Requirement: A distributed run activates only when the world size is greater than one

The trainer SHALL read `RANK`, `LOCAL_RANK` and `WORLD_SIZE` from the environment, defaulting
to `0`, `0` and `1`, and SHALL place its device on `cuda:LOCAL_RANK` with the process group
initialised after the device is set and before any GPU preflight. The distributed path MUST
be gated on `WORLD_SIZE > 1`: at a world size of one the run MUST take the single-process
path exactly, with no process group created, no module wrapped, no stream sharded and no
value reduced. The process group MUST be destroyed on every exit path, including the error
path, before the run's handlers are restored.

#### Scenario: A single-process run is untouched

- **WHEN** `WORLD_SIZE` is unset or one
- **THEN** no process group is created, the model is not wrapped, the loader is not
  sharded, and the device, metrics and artifacts are what they were before this change

#### Scenario: A multi-rank run is initialised in the right order

- **WHEN** `WORLD_SIZE` is greater than one and CUDA is available
- **THEN** the device is `cuda:LOCAL_RANK` and the process group exists before the GPU
  preflight runs

#### Scenario: The process group is destroyed on the error path

- **WHEN** a rank raises before training completes
- **THEN** the process group is destroyed before the run's handlers are restored

### Requirement: Each training rank reads its own shard of the shuffled stream

With a world size greater than one the training stream MUST be split into disjoint per-rank
shards after the dataset's map and shuffle stages, so no example is trained on by two ranks
and none is dropped; the split MUST be applied only when the world size is greater than one,
so the single-process loader is unchanged. Validation MUST stay replicated: it runs on the
one rank that owns the run and its result is broadcast to the others, rather than being
sharded in this version. A training loader that is not offered across ranks MUST refuse a
world size greater than one loudly, naming the loader, the world size and the rank, rather
than iterate a stream its ranks duplicate. The dataset-local `GeneratorDataset` path is not
offered across ranks -- not because it cannot shard, which it can, since its shard index is
the rank's and its worker's over the world size times the worker count, but because it has no
cross-rank test coverage and no production caller, unlike the HuggingFace path -- and that is
the reason its refusal MUST give.

#### Scenario: Two ranks see disjoint halves

- **WHEN** a two-rank run splits a shuffled streaming dataset
- **THEN** the two ranks' example sets are disjoint and their union is the source set

#### Scenario: A single-process loader is not split

- **WHEN** the world size is one
- **THEN** the sharding call is not made and the loader yields the same stream as before

#### Scenario: An unshardable loader is refused

- **WHEN** a training loader that is not offered across ranks is asked for a world size
  greater than one
- **THEN** it raises, naming the loader, the world size and the rank, and its message gives the missing
  cross-rank test coverage and the absent production caller as the reason rather than
  duplicated examples

### Requirement: Exactly one rank owns the run and its artifacts

In a distributed run a single rank MUST own the MLflow run, the tags written to it, the
per-epoch and best checkpoint saves, and the model uploads and registrations; the other
ranks MUST NOT write to the run or to the checkpoint keys. A rank that is not the owner MUST
not reinstate a logger through any fallback. The failure of the run MUST be recorded on that
one run, naming the rank that failed.

#### Scenario: Only one rank writes

- **WHEN** a two-rank run reaches a checkpoint-save boundary
- **THEN** one rank writes the checkpoint and the other writes nothing

#### Scenario: A failing rank is named on the run

- **WHEN** one rank of a two-rank run raises
- **THEN** the run is marked FAILED with a message naming that rank

### Requirement: Every input to a training decision is reduced across ranks

Every value a training decision reads MUST be reduced across all ranks before the decision is
taken: the epoch's training accumulators are reduced as sums and divided once so the reported
figure is a pooled mean and not a mean of per-rank means; the OOM skip counters are reduced
as sums and the consecutive-skip length as the maximum; the trained-nothing check reads the
reduced batch count; and the validation metrics every rank decides on MUST be identical,
which the single broadcast validation result supplies. The validation metric MUST be computed
once, by one process, and MUST NOT be averaged once per rank and again across ranks.

#### Scenario: The training figures are pooled, not averaged twice

- **WHEN** two ranks report different batch counts and losses
- **THEN** the reported loss is the sum of losses divided by the sum of batch counts, not the
  mean of the two per-rank means

#### Scenario: A distributed epoch's metric matches the single-process value

- **WHEN** a two-rank run pools the training figures its ranks accumulated
- **THEN** the pooled loss equals the loss one process would compute over the same batches
  (the summed losses over the summed batch count), where averaging the ranks' means would
  give a different number

#### Scenario: The validation metric is the one process's value

- **WHEN** a two-rank run validates a replicated validation set
- **THEN** the metric is computed once by the owning rank over the whole set -- the same
  computation a single-process run performs -- and every rank decides on that value

#### Scenario: Validation is measured once

- **WHEN** a two-rank run validates
- **THEN** one process computes the metric over all validation images and every other rank
  decides on that same value

### Requirement: A failure on any rank ends every rank

While the ranks still have a collective ahead of them, a terminal error on any rank MUST end all
ranks, none exiting zero, and MUST NOT leave a rank waiting in a collective. No collective
operation may sit on an error path, so a rank that has failed does not enter one and its peers
fail on their next collective rather than block. After the last collective the main rank finishes
alone -- upload and registration -- and the non-main rank has already returned and exited zero, so
no peer is left to abort and no barrier or watchdog is added: a failure in that window exits the
main rank non-zero and tags the run FAILED, and that exit code is the run's failure signal.

#### Scenario: One rank's error ends the other

- **WHEN** one rank of a two-rank run raises a terminal error while the ranks still have a
  collective ahead
- **THEN** both rank processes exit non-zero within a bounded time, with neither left alive

### Requirement: Gradients are synchronised once per optimizer step

In a distributed run the per-batch micro-steps of an accumulation group MUST NOT each
all-reduce; the group's gradients MUST be synchronised once, at the optimizer step, so an
accumulation group costs one all-reduce rather than one per micro-batch. The epoch MUST NOT
end with an unsynchronised trailing group: under distributed training a trailing partial group
MUST be discarded and logged as an observable figure, because a rank-local step on a group the
other rank does not share would take a different step per rank.

#### Scenario: An accumulation group all-reduces once

- **WHEN** a distributed run accumulates over several micro-batches per step
- **THEN** the gradient synchronisation happens once per optimizer step, not once per
  micro-batch

#### Scenario: A trailing partial group is discarded, not stepped

- **WHEN** a distributed epoch ends with fewer micro-batches than the accumulation setting
- **THEN** the trailing group is discarded and its size is logged

### Requirement: A parameter unfrozen after wrapping is still synchronised

Distributed wrapping MUST synchronise every parameter the model could train, including
parameters frozen at the moment of wrapping and unfrozen later, so a staged unfreeze does not
silently leave a parameter rank-local. The wrapped module object MUST be the same object
across epochs, including across an unfreeze boundary, and MUST be wrapped only once; the
trainable parameter set is permitted to change between epochs.

#### Scenario: An unfrozen parameter receives the same gradient on every rank

- **WHEN** a parameter frozen during the wrapping epochs is unfrozen and the next epoch
  trains
- **THEN** that parameter's gradient is present on every rank and equal across ranks

#### Scenario: The wrapper is not replaced

- **WHEN** the backbone is unfrozen mid-run
- **THEN** the wrapped module is the same object before and after the unfreeze

### Requirement: Replicated parameters and buffers keep every rank's model identical

Parameters and buffers MUST be identical on every rank once the model is wrapped, and MUST
stay identical. The wrap broadcasts rank 0's parameters and buffers at construction, so
ranks that built their models independently still enter the loop with one model between
them; the all-reduce of gradients keeps the parameters equal thereafter, and the buffers are
laid down from the owning rank on every forward.
Layers with running statistics MUST NOT be left in training mode by an eval-pinning walk over
the backbone, whether they are plain batch normalisation or its synchronised form, since a
diverging statistic would make the ranks' models differ. The backbone's batch normalisation
MUST NOT be replaced by its synchronised form in this version: that form hides collectives
inside the forward, which an uneven epoch's `join()` cannot carry without early termination,
and the per-forward broadcast of the owning rank's buffers is what keeps the statistics
identical across ranks here. The eval-pinning walk MUST nevertheless recognise the
synchronised form, so the pin stays correct if it is adopted later. A normalisation block with no running
statistics -- the grouped form the model can be converted to -- has no buffer to diverge, so it
is already identical across ranks, and the walk MUST leave it training rather than pin it as
though it held a statistic.

#### Scenario: An eval-pinning walk reaches the synchronised form of batch normalisation

- **WHEN** the backbone contains a synchronised batch-normalisation layer
- **THEN** the eval-pinning walk puts it in eval mode, as it does for the plain form

#### Scenario: Buffers do not diverge

- **WHEN** a buffer is set differently on each rank after the wrapper is built and the ranks
  then train
- **THEN** every rank ends the run holding the owning rank's buffers, so a statistic that
  moved during training is the same on every rank and counted the same number of batches,
  rather than each rank keeping its own batch-bumped copy

#### Scenario: A stateless normalisation block is left training

- **WHEN** the backbone's batch normalisation has been converted to the grouped form
- **THEN** it holds no buffers, and the eval-pinning walk leaves it in training mode

### Requirement: An uneven epoch ends rather than hangs

When the ranks of an epoch produce different numbers of batches, the epoch MUST end cleanly
and the run MUST continue, rather than one rank blocking in a collective its peer will never
enter. Every rank MUST begin the next epoch holding the same parameter values, so that the
gradients all-reduced from it are gradients of one model rather than of several.

#### Scenario: A rank with one fewer batch ends the epoch

- **WHEN** one rank produces one fewer batch than its peer in an epoch
- **THEN** both ranks finish the epoch and the run continues, with neither hanging

#### Scenario: The next epoch starts from the same state

- **WHEN** an epoch ends with unequal per-rank batch counts, the rank with more batches having taken a step its peer did not take
- **THEN** every rank begins the next epoch holding byte-identical parameters

### Requirement: Each rank's thread count is its share of the node's

When the operator's `OMP_NUM_THREADS` is applied to torch's thread count, a distributed run
MUST divide it by the number of ranks so the node's thread budget is shared rather than
multiplied, with a floor of one; a single-process run MUST apply the value undivided.

#### Scenario: A two-rank run halves the thread budget

- **WHEN** `OMP_NUM_THREADS` is 8 and the world size is two
- **THEN** each rank sets torch's thread count to 4

#### Scenario: A single-process run is unchanged

- **WHEN** `OMP_NUM_THREADS` is 8 and the world size is one
- **THEN** torch's thread count is 8

### Requirement: A stream split inside DataLoader workers yields each example once

A training stream that is iterated inside a DataLoader worker MUST take the worker's identity
into account when it splits the stream: the shard count MUST be the world size multiplied by
the number of workers, and the shard a worker reads MUST be derived from its rank and its
worker id, so that a rank's workers together yield each example exactly once rather than each
worker replaying the whole stream. Outside a worker process the split MUST be unchanged: with
no DataLoader worker the shard count is the world size and the index is the rank, yielding
exactly the examples a single-worker iteration yielded before.

#### Scenario: A rank's workers share its shard once

- **WHEN** one rank iterates its stream through four DataLoader workers
- **THEN** the four workers' outputs are pairwise disjoint and their union is the whole stream

#### Scenario: Two ranks with two workers each stay disjoint

- **WHEN** a two-rank run iterates its stream through two DataLoader workers per rank
- **THEN** the four iterators' outputs are pairwise disjoint and their union is the source set

#### Scenario: Outside a worker the split is unchanged

- **WHEN** that same stream is iterated with no DataLoader worker
- **THEN** the rank yields exactly the examples a single-worker iteration of it yields; the
  only split is across ranks

### Requirement: A process group is initialised only with keywords the installed torch declares

The trainer SHALL decide whether `torch.distributed.init_process_group` accepts a
`device_id` keyword by inspecting that function's own signature, not by comparing torch
version numbers, and SHALL pass `device_id` only when the backend is `nccl` and the
inspection says the keyword exists. On a torch whose signature declares no `device_id`,
a distributed start MUST still initialise its group, with the backend alone, rather than
fail on an unexpected keyword. The keywords MUST be built in one place — the module's
own builder — so that a test can build exactly what the trainer passes.

#### Scenario: A torch without device_id initialises the group

- **WHEN** the installed `init_process_group` declares no `device_id` parameter and the
  backend is `nccl`
- **THEN** the keywords carry `backend` only, and initialising a group with them
  succeeds with the expected world size

#### Scenario: A torch with device_id gets the rank's device

- **WHEN** the installed `init_process_group` declares `device_id` and the backend is
  `nccl`
- **THEN** the keywords carry `device_id` set to the rank's own device

#### Scenario: A non-nccl backend carries no device_id

- **WHEN** the backend is `gloo`
- **THEN** the keywords carry `backend` only, whether or not the installed signature
  declares `device_id`

### Requirement: Every read of the model after the wrap reads the module

A distributed run MUST treat the wrapper as a wrapper and not as the model. After the wrap,
any read of the model's own attributes, counters or sub-objects MUST resolve to the module
the wrapper holds, because `DistributedDataParallel` forwards only the module's methods --
`forward`, `train`, `eval`, `parameters`, `state_dict`, `no_sync`, `join` -- and no arbitrary
attribute. A read MUST NOT be made against the wrapper whether it would raise (an owned
object such as the segmentation head) or silently yield a default (a counter such as the
running activation maxima): both are reads of the model, and both MUST see the module. The
reads the wrapper does forward MUST keep being made on the object the loop trains with, so
that gradient synchronisation and uneven-epoch handling are unchanged. Reading through the
module MUST NOT change a single-process run, where the model has no module of its own.

#### Scenario: The smoke-mode diagnostics survive the wrap

- **WHEN** a multi-rank run trains with the trainer's smoke mode enabled
- **THEN** the run reports the segmentation head's share of the optimizer from the model's own
  head, rather than failing on an attribute the wrapper does not forward

#### Scenario: The epoch's activation maxima survive the wrap

- **WHEN** a multi-rank run reaches the end of an epoch with the model's activation counters
  set
- **THEN** the epoch's activation maxima are folded from the model's counters and logged with
  the values the model holds, rather than being reported as no maxima at all

#### Scenario: A single-process run reads the object it always read

- **WHEN** the run is not distributed
- **THEN** every read resolves to the model itself, so the diagnostics and metrics are
  unchanged
