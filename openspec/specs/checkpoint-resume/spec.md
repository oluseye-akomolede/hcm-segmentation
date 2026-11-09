# checkpoint-resume Specification

## Purpose

Where each task's checkpoints are stored, what they are named, what metadata makes them self-describing, and which of them a new run is allowed to resume from. This is the contract that decides whether a resumed run continues from the right weights or silently starts over from a smoke-test artifact.

## Requirements

### Requirement: Each task maps to exactly one checkpoint bucket and model name

The system MUST resolve a training configuration to one of three task keys (`classification`, `coco_seg`, `cocoa_seg`) and each task key MUST have exactly one bucket and one base model name: `classification` to bucket `models` named `hcm_imagenet1k`, `coco_seg` to bucket `coco-models` named `hcm_coco_seg`, and `cocoa_seg` to bucket `finetuned-models` named `hcm_cocoa_seg`. An unknown task key MUST raise rather than fall back to a default bucket.

#### Scenario: Finetune resolves to the cocoa bucket

- **WHEN** the task is resolved with `finetune=True`
- **THEN** the task key is `cocoa_seg`
- **AND** the target bucket is `finetuned-models` with model name `hcm_cocoa_seg`

#### Scenario: An unknown task raises

- **WHEN** a bucket is requested for a task key not in the table
- **THEN** a `ValueError` naming the task and the accepted keys is raised

### Requirement: Checkpoint metadata is written on every save path

Every checkpoint save MUST write a metadata document carrying the MLflow run id, the task, the dataset name, the epoch, that epoch's validation metrics, the training and validation losses, the image's `GIT_SHA`, and the boolean flags `is_best`, `is_last` and `is_smoke`. Values that are not JSON-safe numbers MUST be coerced to a JSON-safe form rather than raising, because the interrupted-save path passes descriptive strings where metrics normally go.

#### Scenario: A normal epoch checkpoint is self-describing

- **WHEN** an epoch checkpoint is saved with validation metrics
- **THEN** its metadata contains the run id, task, epoch, metrics, both losses, the git commit and all three boolean flags

#### Scenario: A non-numeric metric does not break the save

- **WHEN** the save path is given a descriptive string such as an interruption reason in place of a metric
- **THEN** the value is written as a string and the save completes

### Requirement: Only best checkpoints are resume candidates

A stored checkpoint MUST be eligible to be resumed from only when it is not a smoke checkpoint and it is a best checkpoint. A checkpoint whose `is_smoke` is true MUST be rejected. A checkpoint with no `is_best` field at all MUST be accepted, because those are legacy artifacts written before the field existed and they are the only checkpoints some buckets contain.

#### Scenario: A smoke checkpoint is never resumed

- **WHEN** candidate metadata has `is_smoke` true
- **THEN** the checkpoint is not a resume candidate

#### Scenario: A legacy checkpoint without is_best stays loadable

- **WHEN** candidate metadata has no `is_best` field
- **THEN** the checkpoint is accepted as a resume candidate

#### Scenario: A non-best per-epoch checkpoint is rejected

- **WHEN** candidate metadata has `is_best` false and no smoke flag
- **THEN** the checkpoint is not a resume candidate

### Requirement: Resume selection picks the newest eligible candidate

From the set of resume candidates, the system MUST select the one with the latest `upload_timestamp`. A candidate with a missing timestamp MUST sort oldest, so any well-formed checkpoint wins. A candidate whose `metadata.json` was read but whose `upload_timestamp` cannot be parsed MUST make the selection raise, naming the value, rather than sort oldest. When there are no eligible candidates the selection MUST return none, and the caller MUST distinguish "the bucket is empty" from "every candidate was unsuitable" by naming the rejection reason for each.

#### Scenario: The newest candidate wins

- **WHEN** two eligible candidates exist with different upload timestamps
- **THEN** the one with the later timestamp is selected

#### Scenario: A val-selected best displaces the legacy base

- **WHEN** a bucket holds a legacy base checkpoint and a newer `_best` checkpoint flagged `is_best`
- **THEN** the newer `_best` checkpoint is selected

#### Scenario: An unparseable timestamp sorts oldest

- **WHEN** one candidate has no timestamp and another has a valid one
- **THEN** the candidate with the valid timestamp is selected

#### Scenario: An unparseable timestamp raises

- **WHEN** a candidate's `upload_timestamp` is present but not a valid timestamp
- **THEN** the selection raises an error naming that value

#### Scenario: Only unsuitable candidates produces a naming error

- **WHEN** the bucket contains matching checkpoints but none is a resume candidate
- **THEN** the raised error names each rejected prefix and its reason

### Requirement: Epoch checkpoints and the best checkpoint are named distinctly

A per-epoch save MUST be named `<model_name>_epoch<epoch>`, and the run-end best save MUST be named `<model_name>_best`. The run-end best upload MUST restore the best epoch's weights before uploading and MUST restore the model's prior weights afterwards, so the upload does not leave the caller's model swapped. Training MUST be configured to checkpoint every epoch.

#### Scenario: A per-epoch checkpoint carries its epoch number

- **WHEN** epoch 3 of a classification run is saved
- **THEN** the uploaded checkpoint is named `hcm_imagenet1k_epoch3`

#### Scenario: The best checkpoint is uploaded under the best name

- **WHEN** a run finishes with a best epoch recorded
- **THEN** the best weights are loaded and uploaded as `hcm_imagenet1k_best` with `is_best` true
- **AND** the model is left holding the weights it had before the upload

#### Scenario: The upload is skipped when no epoch was best

- **WHEN** a run finishes with no best epoch recorded
- **THEN** no best checkpoint is uploaded

### Requirement: A downloaded checkpoint is rebuilt from its class info

When a checkpoint carries `model_info.pkl`, the loader SHALL rebuild the model
from it and return a built model, not the raw state dict. A config value of
`None` SHALL be treated as absent. A missing `num_segmentation_classes` SHALL
fall back to the count inferred from the state dict, and a missing `out_stride`
SHALL fall back to 16. `hcm_nn` SHALL record its `out_stride` and
`num_segmentation_classes` so the uploader pickles real values.

The loader SHALL report non-empty missing or unexpected keys as a warning. If
the class info is absent or cannot be used, the loader SHALL return the state
dict as before, and its warning MUST name the exception type and repr.

#### Scenario: A legacy checkpoint with a None config still loads

- **WHEN** `model_info.pkl` records `out_stride` and `num_segmentation_classes` as `None`, and the state dict holds a two-class segmentation head
- **THEN** the loader returns an `hcm_nn` with a two-class head and all weights loaded

#### Scenario: A new upload round-trips

- **WHEN** an `hcm_nn` with two segmentation classes and out_stride 16 is uploaded and downloaded again
- **THEN** the loader returns an `hcm_nn` with the same head size and out_stride, and identical weights

#### Scenario: Unusable class info keeps the state-dict fallback

- **WHEN** the class info cannot be used to build a model
- **THEN** the loader returns the state dict and warns with the exception's type name and repr

### Requirement: The caller decides the mode and the checkpoint decides the architecture

`load_hcm_model` SHALL set the returned model's `enable_finetuning` to the
caller's `finetune` flag, whether the checkpoint recorded true or false.

When the caller asks for classification (`finetune` false) and the checkpoint's
weights hold no classification head (no `roi_classifier.*` key), the loader MUST
raise a `ValueError` naming both the requested mode and what the checkpoint
lacks, rather than return a model with an untrained classifier.

The checkpoint's architecture SHALL win, because its weights only load into
it: the segmentation class count recorded in, or inferred from, the checkpoint,
and on the class-info path its `out_stride`. When the caller passes a
`num_segmentation_classes` or `out_stride` that disagrees with the checkpoint's,
the loader MUST warn, naming both values. A caller's class count SHALL add a
head only when the checkpoint has none. A bare state dict carries no
`out_stride`, so on that path the caller's value is used. Likewise, when a
checkpoint's class info records `out_stride` as `None` (a legacy checkpoint),
the checkpoint has no stride to win: the loader MUST use the caller's
`out_stride` without a warning, and MUST fall back to 16 only when the caller
passes none.

#### Scenario: A classification run from a segmentation checkpoint

- **WHEN** the checkpoint recorded `enable_finetuning` true and the caller passes `finetune` false
- **THEN** the returned model has `enable_finetuning` false and returns classification logits

#### Scenario: A fine-tune from a classification checkpoint

- **WHEN** the checkpoint recorded `enable_finetuning` false and the caller passes `finetune` true
- **THEN** the returned model has `enable_finetuning` true

#### Scenario: A legacy checkpoint follows the caller

- **WHEN** the checkpoint's class info records `None` for its config and the caller passes either flag
- **THEN** the returned model's mode is the caller's

#### Scenario: Classification from a checkpoint with no classifier

- **WHEN** the caller passes `finetune` false and the checkpoint has no `roi_classifier` weights
- **THEN** a `ValueError` names classification and the missing classification head

#### Scenario: A disagreeing class count

- **WHEN** the caller passes a `num_segmentation_classes` different from the checkpoint's
- **THEN** the returned model keeps the checkpoint's class count and its weights
- **AND** a warning names both counts

#### Scenario: A disagreeing out_stride

- **WHEN** the checkpoint records an `out_stride` and the caller passes a different one
- **THEN** the returned model keeps the checkpoint's `out_stride`
- **AND** a warning names both values

#### Scenario: A legacy checkpoint that recorded no out_stride

- **WHEN** the checkpoint's class info records `out_stride` as `None` and the caller passes 8
- **THEN** the returned model has `out_stride` 8 and the checkpoint's weights
- **AND** no precedence warning is raised

#### Scenario: Neither the checkpoint nor the caller gives an out_stride

- **WHEN** the checkpoint's class info records `out_stride` as `None` and the caller passes none
- **THEN** the returned model has `out_stride` 16

### Requirement: There is one resume search path

The checkpoint store MUST expose exactly one way to find a checkpoint to resume from: the architecture search that applies the resume-candidate rules. A name-based search that bypasses those rules MUST NOT exist, so no caller can resume from a smoke or non-best checkpoint by reaching for it.

#### Scenario: The name-based search is gone

- **WHEN** the checkpoint store's public methods are listed
- **THEN** `search_and_download_model_by_architecture` is present and `search_and_download_model` is not

### Requirement: metadata.json is the last object a checkpoint upload writes

A checkpoint upload MUST write `model.pth` first, then `model_info.pkl` (when it could be serialised), and `metadata.json` last. The resume search MUST only consider prefixes that have a `metadata.json`, so a checkpoint whose upload failed or is still in flight is never a resume candidate. A failure uploading any object MUST raise from the upload.

#### Scenario: The upload order

- **WHEN** a checkpoint is uploaded
- **THEN** the objects are written in the order `model.pth`, `model_info.pkl`, `metadata.json`

#### Scenario: A save that fails before metadata.json is not resumable

- **WHEN** the upload of `model.pth` or `model_info.pkl` fails
- **THEN** the upload raises
- **AND** no `metadata.json` exists under that prefix, so the resume search does not see it

### Requirement: Unreadable checkpoint metadata is counted and never skipped past

A `metadata.json` that exists but cannot be fetched or parsed MUST be counted, and the resume summary line MUST report how many there were. If any such checkpoint is newer, by the timestamp in its prefix, than the checkpoint selected, the resume MUST be refused with an error naming each one, rather than resuming from an older checkpoint. A missing `metadata.json` is an upload in flight and MUST NOT count.

#### Scenario: A newer unreadable checkpoint refuses the resume

- **WHEN** the newest checkpoint's `metadata.json` is unreadable and an older readable best checkpoint exists
- **THEN** the resume is refused and the error names the unreadable prefix

#### Scenario: An older unreadable checkpoint is reported, not fatal

- **WHEN** an unreadable checkpoint is older than the selected one
- **THEN** the resume proceeds and the summary line counts one checkpoint with unreadable metadata

### Requirement: A failed resume names the bucket and the real reason

When there is no checkpoint to resume from, the loader MUST raise an error naming the bucket it searched and why: the bucket has no checkpoint of the architecture, every candidate was unsuitable (each named with its reason), checkpoint metadata was unreadable, or the selected checkpoint's download failed.

#### Scenario: An empty bucket

- **WHEN** the resume bucket holds no checkpoint of the architecture
- **THEN** the error names the bucket and says it has no checkpoint

#### Scenario: A failed download

- **WHEN** a checkpoint is selected but its weights cannot be downloaded
- **THEN** the error names the bucket and the prefix whose download failed

### Requirement: The resume bucket comes from the run's task and --model-base

The bucket a resume searches MUST be the one `--model-base` names, and only if that bucket holds one task's checkpoints and that task may resume into the run's task (from `resolve_task`). The allowed transitions, as (checkpoint task, run task), MUST be exactly:

| Checkpoint task | Run task | Why |
| --- | --- | --- |
| classification | classification | continue the lineage |
| coco_seg | coco_seg | continue the lineage |
| cocoa_seg | cocoa_seg | continue the lineage |
| classification | coco_seg | ImageNet to COCO; the segmentation head is added |
| coco_seg | cocoa_seg | COCO to COCOA; the refine |

Any other combination, and any `--model-base` that is not a checkpoint bucket, MUST raise an error naming the flags and the bucket(s) that would work. It MUST NOT fall back to `models`. A fresh model (no resume) MUST NOT be checked.

#### Scenario: The refine resumes from the COCO bucket

- **WHEN** a `--finetune` run resumes with `--model-base coco-models`
- **THEN** it searches `coco-models`

#### Scenario: A combination outside the table raises

- **WHEN** a `--finetune` run resumes with `--model-base models`
- **THEN** it raises naming `--finetune` and `--model-base coco-models or finetuned-models`

#### Scenario: No fallback to models

- **WHEN** a classification run resumes with `--model-base finetuned-models`
- **THEN** it raises instead of searching `models`

### Requirement: A resume reports and checks what it resumes
The resume SHALL log the chosen checkpoint's recorded task, input size and COCO mix ratio, printing `unknown` for a field a legacy checkpoint lacks. A recorded task the run may not resume from (per the allowed-transition table) MUST raise before any training starts. A recorded input size that differs from the run's MUST warn loudly but MUST NOT refuse, because fine-tuning at a new size is legitimate. A legacy checkpoint without these fields MUST NOT be checked.

#### Scenario: Refining a 352 COCO baseline
- **WHEN** a cocoa_seg run resumes a coco_seg checkpoint recorded at input size 352
- **THEN** the log names task coco_seg, input size 352 and the mix ratio, and nothing warns

#### Scenario: A checkpoint from the wrong task
- **WHEN** a coco_seg run finds a cocoa_seg checkpoint
- **THEN** the load raises naming both tasks

#### Scenario: A new input size
- **WHEN** a run at 352 resumes a checkpoint recorded at 192
- **THEN** a warning names both sizes and the model still loads

### Requirement: A resume that passes over a better checkpoint says so, and every upload records its parent

Resume selection SHALL stay newest-wins among eligible candidates: a resume continues the lineage. When an eligible candidate in the same bucket recorded a higher `val_iou` than the chosen one, the resume MUST log a WARNING naming both checkpoints, their epochs and their `val_iou`, and MUST NOT change the selection. Every checkpoint upload (per-epoch and best) MUST record `resumed_from` in its custom metadata: the bucket and prefix of the checkpoint the run resumed from, or null for a fresh model.

#### Scenario: A dead run's epoch 0 is newer than a long-trained best

- **WHEN** a bucket holds a 09-30 `_best` at epoch 14 with val_iou 0.61 and a newer epoch-0 checkpoint with val_iou 0.22
- **THEN** the resume still loads the epoch-0 checkpoint, and a warning names both checkpoints with their epochs and val_iou

#### Scenario: A fresh model

- **WHEN** a run that resumed nothing uploads a checkpoint
- **THEN** its metadata records `resumed_from` as null

### Requirement: A checkpoint records its backbone stride

Every checkpoint upload (per-epoch and best) MUST record `out_stride` in its custom metadata, taken from the model being saved (unwrapping a DDP wrapper), so a consumer rebuilds the same architecture instead of assuming 16. A model that records no stride SHALL write null.

#### Scenario: A stride-8 model

- **WHEN** a per-epoch checkpoint is uploaded for a model built with `out_stride` 8
- **THEN** its metadata records `out_stride` 8

#### Scenario: The demo reads it

- **WHEN** a demo's sidecar carries the trainer's `out_stride`
- **THEN** the demo reports the stride as coming from the metadata, not the default

### Requirement: Uploading the best checkpoint leaves the model's weights as they were

When `train_hcm_imagenet1k` uploads the `_best` checkpoint it SHALL load the best epoch's weights into the model for the upload, and afterwards it MUST restore the weights the model held before, from a copy taken before the swap. A reference to the model's live state (`model.state_dict()`) MUST NOT serve as that copy, because loading the best weights overwrites it in place.

#### Scenario: The best checkpoint carries the best weights
- **WHEN** the run's best epoch is not its last
- **THEN** the uploaded `_best` checkpoint holds the best epoch's weights

#### Scenario: The model holds the last weights after the upload
- **WHEN** the `_best` upload has finished, successfully or not
- **THEN** the model holds the last epoch's weights, not the best epoch's

### Requirement: The decoder version travels with the checkpoint

New training SHALL build `v2` models. Uploads MUST record `decoder_arch` in model_info and in checkpoint metadata. Every loader, demo and the eval tool SHALL build the version the checkpoint is: from recorded metadata when present, otherwise from the weights, where the presence of hcm_rrd fuse weights means `v2` and their absence means `v1`. A recorded version that the weights contradict, or an unknown recorded version, MUST stop the load with an error. This holds on every path: rebuilding from model_info (which loads non-strictly, so it MUST compare the recorded version with the weights before building), and the downloader, which MUST propagate the error rather than fall back to a bare state dict or to no checkpoint. When metadata states the version in more than one place, every stated value MUST agree with the weights.

#### Scenario: An unrecorded checkpoint is read from its weights
- **WHEN** a checkpoint without a recorded `decoder_arch` is loaded
- **THEN** it is built as `v1` if it has no fuse weights and `v2` if it has them

#### Scenario: A contradicted version is refused
- **WHEN** the metadata or model_info records `v2` but the weights have no fuse convolutions, or records `v1` but the weights have them
- **THEN** the load stops with an error naming both

#### Scenario: The downloader does not swallow a contradicted version
- **WHEN** resuming downloads a checkpoint whose model_info records a version its weights contradict
- **THEN** the error reaches the caller; no state dict and no `None` is returned

#### Scenario: Disagreeing stated versions are refused
- **WHEN** `custom_metadata.decoder_arch` and the top-level `decoder_arch` disagree
- **THEN** the load stops with an error, even if one of them agrees with the weights

#### Scenario: An unknown version is named as unknown
- **WHEN** model_info records a `decoder_arch` that is neither `v1` nor `v2`, e.g. `v9`
- **THEN** the load stops with an error naming the value as an unknown decoder_arch, not as a disagreement with the weights

### Requirement: A warm start initialises a new lineage from a checkpoint

Training SHALL accept an explicit warm-start source, `--init-from <bucket>/<prefix>`, and a decoder version for the new model, `--init-decoder-arch` (default `v2`). A warm start MUST build a fresh model and copy into it every tensor of the source checkpoint, as saved, whose name and shape match; the source MUST NOT be read through a rebuilt model, whose non-strict load would invent the tensors the source lacks and lose those only it has. Tensors whose shape differs, and tensors the source lacks, MUST keep their initialisation, except that a `v1` source into a `v2` model MUST initialise each hcm_rrd fuse convolution to weight `[I | 0]` (identity on the normal branch, zero on the reversed) and bias 0. The run MUST report the number of copied, skipped, initialised and new tensors, the source tensors it did not use, and the copied fraction of the model's elements. Its checkpoints MUST record the source as `init_from` and MUST record `resumed_from` as null. A warm start MUST be refused when combined with a resume, when the decoder version is unknown, or when the source is not named as `<bucket>/<prefix>`, all before any MLflow run is created; and when the source shares no tensor with the model.

#### Scenario: A v1 checkpoint warm-starts a v2 model
- **WHEN** a run warm-starts a `v2` model from a `v1` checkpoint
- **THEN** every v1 tensor is copied, the hcm_rrd fuse convolutions are reported as initialised to `[I | 0]` with zero bias, none is new, and the counts are logged
- **AND** each hcm_rrd rung's output equals the v1 RCAB and convolution applied to the normal branch alone

#### Scenario: A changed shape is skipped
- **WHEN** a source tensor has the model's name but another shape
- **THEN** it is reported as skipped and the model's tensor keeps its init

#### Scenario: A warm start is not a resume
- **WHEN** a warm-started run saves a checkpoint
- **THEN** the metadata records `init_from` as the source and `resumed_from` as null

#### Scenario: A warm start combined with a resume is refused
- **WHEN** a run is asked to warm-start and to resume
- **THEN** it stops with an error before any MLflow run is created

#### Scenario: The source is the checkpoint as saved
- **WHEN** the source checkpoint lacks a tensor the model has, and has one the model lacks
- **THEN** the model's tensor is reported new and keeps its init, and the source-only tensor is reported as unused

#### Scenario: A malformed source name is a usage error
- **WHEN** `--init-from` is not of the form `<bucket>/<prefix>`
- **THEN** the run stops with a usage error before any MLflow run is created

### Requirement: A distributed run writes its checkpoints unwrapped

A checkpoint written by a distributed run MUST carry no distributed wrapper's prefix: its
keys MUST be the model's own, so the single-process loader that reads it is unchanged and a
model saved by two ranks loads into a plain model exactly as one saved by a single process
does. The state that is saved MUST therefore be taken from the model the run is training,
not from the wrapper around it.

#### Scenario: A two-rank checkpoint loads into a plain model

- **WHEN** a two-rank run saves a checkpoint and that checkpoint is loaded into a plain model
- **THEN** it loads without any missing or unexpected keys and no key carries a wrapper prefix

#### Scenario: A distributed best checkpoint is named and shaped as before

- **WHEN** a distributed run saves its best checkpoint
- **THEN** its name, its metadata and its key shape are what a single-process run would write

### Requirement: The recorded architecture names the backbone, the neck width and the mask stride

Every checkpoint upload MUST record, in `model_info` and in its custom metadata, the
`backbone` name, the `neck_channels` (or null) and the `mask_stride` of the model being
saved, taken from the model itself and unwrapped from a DDP wrapper. Every loader — the
rebuild from `model_info`, the downloader, the evaluation tool, the demo and the torch-free
shape gate — SHALL build the architecture the checkpoint is: from the record when present,
otherwise from the weights. An absent backbone record means `resnet50`, an absent neck
width means the 2048-channel widening, and an absent mask stride means 4.

A recorded value the weights contradict, or an unknown recorded backbone, MUST stop the
load with an error naming both values, on every path, including the rebuild that loads
non-strictly and the downloader that must not fall back to a bare state dict or to no
checkpoint.

#### Scenario: A legacy checkpoint with no architecture record loads as the old model

- **WHEN** a checkpoint whose `model_info` has no backbone, neck or mask-stride record is
  loaded
- **THEN** it is built as `resnet50` with the 2048 widening and a mask stride of 4, and all
  its weights load

#### Scenario: The architecture survives a round trip

- **WHEN** a model with a timm backbone, a neck width and a mask stride is uploaded and
  downloaded again
- **THEN** the returned model has the same backbone, neck width and mask stride, and
  identical weights

#### Scenario: A contradicted architecture is refused

- **WHEN** the record names a timm backbone but the weights hold the resnet50 backbone's
  keys, or the record names a neck width the weights do not have
- **THEN** the load stops with an error naming the recorded value and the value the weights
  show

#### Scenario: The downloader does not swallow a contradicted architecture

- **WHEN** a resume downloads a checkpoint whose architecture record its weights contradict
- **THEN** the error reaches the caller; no state dict and no `None` is returned

#### Scenario: An unknown backbone is named as unknown

- **WHEN** the record names a backbone outside the registry
- **THEN** the load stops with an error naming the value as an unknown backbone

### Requirement: The recorded architecture says whether the model has an outline head

Every checkpoint upload MUST record `amodal_outline` (true or false) in `model_info` and in its custom metadata, next to the backbone, neck width and mask stride, taken from the model itself and unwrapped from a DDP wrapper. Every loader that builds the architecture from the record (the rebuild from `model_info`, the downloader, the evaluation tool and the demo's model builder) SHALL build the outline head when the record says true, or, with no record, when the weights hold `outline_head.weight`; otherwise it SHALL build no outline head. A record the weights contradict MUST stop the load with an error naming both values, on the same paths as the other recorded architecture fields: the torch-free metadata gate (`ModelShapeError`, used by the demo's model builder and the evaluation tool) and the rebuild from `model_info` (`ArchMismatchError`). The torch-free gate MUST decide the value without importing torch. A warm start into a model with the outline head from a checkpoint without one SHALL load every shared tensor and report the outline head's tensors as newly initialised.

#### Scenario: A live checkpoint loads without an outline head
- **WHEN** a checkpoint with no `amodal_outline` record and no outline-head weights is loaded
- **THEN** it is built without an outline head and all its weights load

#### Scenario: An outline checkpoint loads strictly in the demo and evaluation
- **WHEN** a checkpoint recorded with `amodal_outline` true is loaded by the demo's model builder or by the evaluation tool
- **THEN** the model is built with the outline head and the strict load succeeds

#### Scenario: A contradicted outline record is refused
- **WHEN** the record says `amodal_outline` true but the weights have no outline head, or the reverse
- **THEN** the load stops with an error naming the recorded value and what the weights show

#### Scenario: A warm start adds the outline head
- **WHEN** a run with `--amodal-outline` warm-starts from a checkpoint without an outline head
- **THEN** every shared tensor is copied and the outline head's tensors are reported as new
