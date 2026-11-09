# seg-eval Specification

## Purpose
Score saved segmentation checkpoints offline on COCO or COCOA val, using the
trainer's own val transform and metrics, so models from different runs (and
different devices, e.g. the Jetson) can be compared on one scale.

## Requirements

### Requirement: Segmentation checkpoints are scored the way the trainer validates

The repository SHALL provide a tool (`tools/eval_seg.py`) that scores one or
more segmentation checkpoints on COCO val2017 (`--dataset coco`) or the COCOA
val split (`--dataset cocoa`) read from `--data-root`. It MUST use the
trainer's validation transform, the trainer's collate function and the
trainer's `evaluate()`, with `drop_last` and the training run's batch size
(`--batch-size`; default 2 = the 352 runs), so that its IoU is the same measure as
a training run's `val_iou`. The trainer's validation loader uses the run's own
`--batch_size`, so a run trained at another batch size is matched by passing it.
A checkpoint MUST be loaded strictly: a checkpoint without a segmentation head,
or whose weights do not match the model, is an error rather than a score.

#### Scenario: COCO-format data end to end

- **WHEN** the tool scores a model that predicts exactly the instances visible in the image, on a COCO-format set with several instances per image
- **THEN** the IoU and Dice are near 100, and the inverse model's IoU is near 0

#### Scenario: COCOA-format data end to end

- **WHEN** the rows come from the shunk031/COCOA builder's example generator and features
- **THEN** every region of an annotation is in the target, and the same near-100 / near-0 results hold

#### Scenario: Rows the transform cannot use

- **WHEN** some samples fail the transform
- **THEN** they are not scored, the count of failed images is reported, and a placeholder batch is never scored as data

### Requirement: The input size is the checkpoint's

The tool SHALL score each checkpoint at the input size recorded in its
metadata, resolved with the demo's resolver (`jetson/input_size.py`) from a
sidecar next to the checkpoint (`<name>.json`, `<name>.meta.json` or
`metadata.json`) or from `--metadata`, and at 192 when there is no metadata.
`--input-size` MUST override the metadata. A baseline is not a checkpoint: it
SHALL be scored at the native input size its adapter declares, and at
`--input-size` when one is given, so one model can be reported "as published" and
at our own resolution. The tool MUST report the size it used and whether it came
from the adapter or from `--input-size`, and the adapter MUST run at the size it
is handed rather than resizing to its own: a row that reads as having run at one
resolution and ran at another is the failure this requirement exists to prevent.
The demo's environment override MUST NOT change what an evaluation
measures. A sidecar that exists but cannot be read, or a size that cannot be
used, MUST stop the tool before any data is read.

#### Scenario: Metadata states 352

- **WHEN** a checkpoint's sidecar has `custom_metadata.input_size` 352
- **THEN** it is scored at 352 and the result says so and where the size came from

#### Scenario: An unreadable sidecar

- **WHEN** a checkpoint's sidecar is not valid JSON
- **THEN** the tool exits with an error and does not fall back to 192

#### Scenario: A baseline at its declared size

- **WHEN** a baseline is scored with no `--input-size`
- **THEN** it is scored at its adapter's declared size, the batch it receives is that size, and the result says the size came from the adapter

#### Scenario: A baseline at our own resolution

- **WHEN** a baseline whose adapter declares 640 is given `--input-size 352`
- **THEN** it is scored at 352, the batch it receives is 352x352, and the result says 352 was requested while the adapter's native size is 640

### Requirement: Results are reported one per checkpoint

The tool SHALL print one `RESULT` line per checkpoint carrying the IoU, the
Dice, the number of images scored, the input size, the batch size and the number
of tail images `drop_last` dropped, or one JSON object per
checkpoint with `--json`. A failure to score one checkpoint MUST NOT stop the
others, and MUST make the exit status non-zero.

#### Scenario: Two checkpoints

- **WHEN** the tool is given two checkpoints
- **THEN** it prints two results, each naming its checkpoint

### Requirement: The tool runs on the Jetson

The tool SHALL default to no DataLoader workers, MUST NOT import torchaudio,
and MUST work with `PYTORCH_NO_CUDA_MEMORY_CACHING` set.

#### Scenario: A broken torchaudio is installed

- **WHEN** the installed torchaudio raises on import
- **THEN** the tool still imports and runs, and torchaudio is never loaded

### Requirement: The validation data can be fetched

The repository SHALL provide a fetch helper (`tools/fetch_seg_val.py`) that
lays out COCO val2017 and the COCOA val split as the tool reads them, fetching
only the images each annotation file references and skipping every file
already present. The COCOA builder script MUST be pinned to a revision and
checked against its SHA-256. The COCOA annotations live in a gated HF repo:
the helper MUST download them from the URL the builder names, sending
`HF_TOKEN` from the environment to that host only, MUST NOT carry it across a
redirect to a different host (the gated download redirects to a CDN) or across
an https-to-http downgrade, MUST accept a local tarball instead, and MUST say
what to do when the repo refuses the request. Its error messages MUST name the
configured URL, never the redirected one, which carries a presigned signature.

Every request the helper makes MUST carry a finite timeout, and that timeout
MUST be applied where the request is made rather than at each call site, so no
function in the module can issue a request without one. It MUST bound
establishing the connection and every read on it, so a response that stops
arriving fails the request instead of blocking until the operating system's own
TCP timeout. A download that fails or times out SHALL be retried a bounded
number of times with a growing delay between tries, and every try MUST start
from an empty temporary file, so a retry never continues a half-written one; a
destination file MUST still appear only as the rename of a complete temporary
file, and a file already present MUST still be skipped. A transport failure
that survives its tries MUST be reported with the URL in the message; an HTTP
response's own error is re-raised unchanged, and its URL is on `e.url`. A
refusal (a 4xx) is the server's answer and MUST NOT be retried, and what the
helper reports for it MUST NOT change.

#### Scenario: A resumed fetch

- **WHEN** some images are already present
- **THEN** only the missing ones are downloaded

#### Scenario: The gated annotations are refused

- **WHEN** the annotation download answers 401 or 403
- **THEN** the helper stops and names `HF_TOKEN` and `--annotations-tar`

#### Scenario: The token stays on its host across a redirect

- **WHEN** the annotation download redirects to a different host
- **THEN** the redirected request carries no `Authorization` header
- **AND** a redirect to the same host keeps it, unless it downgrades https to http

#### Scenario: The redirected download fails

- **WHEN** the host a redirect led to refuses the request with any other status
- **THEN** the helper stops with a message naming the status, the configured
  URL and `--annotations-tar`, not a traceback
- **AND** the message does not contain the redirected URL

#### Scenario: A changed builder script

- **WHEN** `COCOA.py` does not match the pinned SHA-256
- **THEN** the helper stops and says so

#### Scenario: A stalled response is bounded

- **WHEN** a server accepts the request and then sends nothing
- **THEN** the download fails within about the configured timeout rather than
  blocking until the operating system's TCP timeout

#### Scenario: Every request carries the timeout

- **WHEN** any of the helper's fetch functions issues a request through the
  module's default opener
- **THEN** the request is handed to the opener with the configured timeout

#### Scenario: A transient failure is retried from scratch

- **WHEN** a try fails at the transport layer and a later try succeeds
- **THEN** the destination is written once, with no temporary file left
- **AND** no try starts with a temporary file already present

#### Scenario: Every try fails

- **WHEN** every try fails at the transport layer
- **THEN** the helper raises an error naming the URL
- **AND** neither the destination nor a temporary file is left behind

#### Scenario: A server error is retried and reported unchanged

- **WHEN** the server answers 5xx on every try
- **THEN** the request is made the bounded number of times
- **AND** the server's own error is raised unchanged, with its URL on the error
  object

#### Scenario: A refusal is not retried

- **WHEN** the server answers 4xx
- **THEN** the request is made once and the answer is reported as it is today

### Requirement: The COCO image set is the one the trainer validates on

The tool SHALL score only the val2017 images that have at least one annotation,
as the trainer's HF validation split (shunk031/MSCOCO) does (4952 of 5000). It
MUST report how many images it skipped for having no annotations, on stderr, so
the result and JSON lines on stdout stay machine-readable.

#### Scenario: An image with no annotations is skipped and reported

- **WHEN** the instances file lists an image with no annotations
- **THEN** that image is not among the rows scored
- **AND** stderr reports the number of skipped images and the number scored
- **AND** stdout carries only the result lines

### Requirement: Evaluation builds a state dict at the recorded stride

`ecs_hcm_model/eval/seg.py` SHALL build a bare state-dict checkpoint at the `out_stride` from its `--out-stride` flag, else the checkpoint's sidecar, else the checkpoint's own record, else 16, using the demos' resolver. Like the input size, the demos' environment variable MUST NOT change what an evaluation builds. An unusable recorded stride MUST fail that checkpoint with an error.

#### Scenario: The sidecar stride is used
- **WHEN** a state-dict checkpoint's sidecar states `out_stride` 8
- **THEN** the evaluated model is built at stride 8

#### Scenario: The demo environment is ignored
- **WHEN** `ECS_HCM_DEMO_OUT_STRIDE=32` is set and nothing records a stride
- **THEN** the evaluated model is built at stride 16

### Requirement: Evaluation refuses a datasets release that EXIF-transposes

The eval tool MUST refuse to run when the installed `datasets` is 2.19.0 or later, the first release whose `Image.decode_example` EXIF-transposes images, because the trainer pins `datasets==2.14.0` and its scores would not match `val_iou`. The check SHALL run before any data is read, and the tool SHALL exit with code 2 and an error naming the installed version and the release to install. The tool MUST read the version from package metadata and MUST NOT import `datasets` to do so. When `datasets` is not installed the tool SHALL NOT refuse.

#### Scenario: A newer datasets is installed
- **WHEN** the installed `datasets` is 2.19.0 or later
- **THEN** the tool exits with code 2 before reading any data, and the error names the installed version

#### Scenario: The pinned datasets is installed
- **WHEN** the installed `datasets` is 2.18.x or earlier
- **THEN** the tool runs as before

### Requirement: The forward precision is selectable and defaults to the trainer's

The tool SHALL accept `--precision` with the values `fp16` and `fp32`, defaulting to
`fp16`. Under `fp16` every forward MUST run under the same autocast the trainer's
validation uses, so the default reproduces every result the tool has reported before.
Under `fp32` the forward MUST run with autocast disabled. That autocast is CUDA-only, so
off CUDA every forward runs in fp32 whatever was requested. The result line and the JSON
output MUST record the effective precision: `fp16` only when fp16 was requested and the
device is CUDA, otherwise `fp32`. The JSON MUST also carry the requested value as
`precision_requested`, and the tool MUST note on stderr when the two differ. Training
validation MUST be unaffected.

#### Scenario: The default is the trainer's autocast
- **WHEN** the tool runs on a CUDA device without `--precision`
- **THEN** every forward runs under the trainer's autocast, enabled
- **AND** the result records `precision=fp16`

#### Scenario: fp32 is a cross-check with autocast off
- **WHEN** the tool runs with `--precision fp32`
- **THEN** every forward runs with autocast disabled
- **AND** the JSON output records `"precision": "fp32"`

#### Scenario: fp16 off CUDA is recorded as fp32
- **WHEN** the tool runs on the CPU with `--precision fp16`
- **THEN** the JSON output records `"precision": "fp32"` and `"precision_requested": "fp16"`
- **AND** stderr notes that fp16 ran as fp32 on that device

#### Scenario: An unknown precision is refused
- **WHEN** the tool runs with `--precision bf16`
- **THEN** it exits with a usage error before reading any data

### Requirement: Scoring time is reported, split into data and model

For each checkpoint the tool SHALL report the scoring wall time, the images scored per
second, the seconds spent waiting for the data loader to produce batches, and the
remaining seconds, attributed to the model. Data wait and model time MUST add up to the
scoring wall time. The model time is honest on CUDA because `evaluate()` reads each
batch's loss back to the host, which waits for the device.

#### Scenario: The timing adds up
- **WHEN** a checkpoint is scored
- **THEN** the result carries `score_seconds`, `data_wait_seconds`, `model_seconds` and `images_per_s`
- **AND** `data_wait_seconds + model_seconds` equals `score_seconds`
- **AND** `images_per_s` equals the images scored divided by `score_seconds`

#### Scenario: A slow loader shows as data time
- **WHEN** the loader takes 50 ms to produce each of three batches
- **THEN** at least 150 ms is counted as data wait

### Requirement: Evaluation reproduces an uploaded checkpoint of either decoder version

The eval tool SHALL build the decoder version the checkpoint's sidecar or weights state and load it strictly, so a checkpoint as uploaded by the trainer scores with the outputs it was trained with.

#### Scenario: A v2 checkpoint as uploaded is evaluated
- **WHEN** `load_checkpoint` is given a v2 `model.pth` with the `metadata.json` the trainer uploaded beside it
- **THEN** it builds v2 and its outputs equal the uploaded model's

### Requirement: Pixel recall can be broken down by object size

`tools/eval_seg.py` SHALL accept `--by-size`, which adds a per-object-size breakdown of
pixel recall to a score without changing the score: the IoU, Dice and loss MUST be the
same numbers with the flag off and on, bit for bit, because they are computed from the
same tensors in the same pass.

Each ground-truth instance SHALL be bucketed by the COCO area convention applied to its
pixel count in the **original** image, before the validation transform's resize and crop:
`small` below 32² = 1024 pixels, `medium` below 96² = 9216, `large` at or above it. A pixel
covered by more than one instance SHALL be charged to the smallest instance covering it, so
that every target pixel belongs to exactly one bucket and the result does not depend on the
order the annotations came back in. (Instances of equal pixel count are in the same bucket,
so their relative order cannot affect the result.) A bucket's recall SHALL be the fraction
of that bucket's ground-truth pixels the model also calls foreground, pooled over every
image scored -- the same foreground rule the IoU uses, on the same upsampled logits, from
the pass that produced the score. (It pools pixels, where the reported IoU averages
per-image ratios; the breakdown and the score therefore aggregate the same pass differently,
and the output SHALL say so rather than imply that one restates the other.)

The breakdown SHALL report per bucket the recall (null when the bucket has no ground-truth
pixels), the ground-truth pixel count and the instance count, and SHALL state the rule it
used. The buckets and the rule SHALL be in the JSON output and in the `RESULT` line.

The breakdown is defined for COCO instances, so `--by-size` with `--dataset cocoa` MUST be
refused with an error saying so, and the bucket map is built in the scoring process, so
`--by-size` with `--num-workers` above 0 MUST be refused too; both refusals MUST happen
before any image is read. When the instances the breakdown bucketed do not add up to the
target the metric scored, that checkpoint MUST fail with an error naming the image rather
than report a breakdown measured against a different ground truth.

#### Scenario: Three instances in three buckets, with a known answer

- **WHEN** rows hold a 24x24, a 64x64 and a 128x128 instance and the model calls foreground exactly the left half of the target
- **THEN** the small bucket's recall is 1.0, the medium's 0.0 and the large's 0.5
- **AND** each bucket reports its instance count and its ground-truth pixel count

#### Scenario: An overlapping pixel belongs to the smallest instance

- **WHEN** a 24x24 instance is painted on top of a 128x128 one, both centred on the same pixel
- **THEN** the small bucket holds 144 pixels and the large 4096 - 144
- **AND** reversing the annotation order gives the identical map

#### Scenario: A bucket with no ground truth

- **WHEN** every instance of the scored rows is `medium` and the run asks for the breakdown
- **THEN** the small and large buckets report a null recall with 0 pixels and 0 instances, not a recall of 0.0

#### Scenario: The bucket is the area before the resize

- **WHEN** an instance of 2000 original pixels is scored at an input size that shrinks it below 1024
- **THEN** it is counted in `medium`, not in `small`

#### Scenario: The score is untouched

- **WHEN** the same rows are scored with and without `--by-size`
- **THEN** the IoU, Dice, loss and image count are identical, bit for bit

#### Scenario: The recall comes from the pass that produced the score

- **WHEN** a checkpoint is scored with `--by-size`
- **THEN** the model runs one forward per scored batch, not two

#### Scenario: A refused dataset or worker count

- **WHEN** `--by-size` is given with `--dataset cocoa`, or with `--num-workers` above 0
- **THEN** the tool exits with an error naming the conflict before reading any image

#### Scenario: The buckets do not add up

- **WHEN** the instances the breakdown bucketed do not cover exactly the target the metric scored
- **THEN** that checkpoint fails with an error naming the image, and it prints no score

### Requirement: A public baseline is scored through the identical metric path

`tools/eval_seg.py` SHALL accept `--baseline <name>` to score a public segmentation
model instead of a checkpoint, through the same path a checkpoint takes: the trainer's
validation transform, the trainer's collate function, the same `--batch-size`, the same
`evaluate()` and, when asked for, the same `--by-size` breakdown. An adapter SHALL
present the baseline to `evaluate()` as the model it calls: a callable that takes the
batch's `pixel_values` and returns logits the metric accepts. The baseline's own
preprocessing that the transform does not perform SHALL be inside the adapter, so the
only difference between a baseline's run and a checkpoint's run is the model.

The registry SHALL be keyed by name, and an unknown name MUST be refused, naming the
names it knows, before any image is read. `--baseline` MUST be refused together with a
checkpoint path, `--metadata` or `--out-stride`, which describe a checkpoint and not a
baseline, before any image is read.

Each adapter SHALL import the baseline's package when it is used and no earlier, so
importing the adapter module MUST NOT import any baseline's package, and a package that
is not importable MUST be refused with an error naming the package and how to install
it, before any image is read. The repository MUST NOT vendor, copy or re-license a
baseline's source or weights: the adapter calls the installed package.

#### Scenario: A known-answer baseline scores its known answer

- **WHEN** an adapter whose model returns exactly the instances visible in the image is scored
- **THEN** the IoU and Dice are near 100, as they are for a checkpoint with that same output
- **AND** the result names the baseline and its input size source

#### Scenario: The breakdown covers a baseline

- **WHEN** a baseline is scored with `--by-size`
- **THEN** the per-size recall, its pixel counts and its instance counts are reported as they are for a checkpoint

#### Scenario: An unknown baseline name

- **WHEN** `--baseline yolo` is given
- **THEN** the tool exits with an error naming the baselines it knows, before reading any image

#### Scenario: A baseline mixed with a checkpoint's options

- **WHEN** `--baseline` is given with a checkpoint path, with `--metadata`, or with `--out-stride`
- **THEN** the tool exits with an error naming the conflict, before reading any image

#### Scenario: Importing the adapters imports no baseline package

- **WHEN** the adapter module is imported
- **THEN** no baseline's package is in `sys.modules`

#### Scenario: A baseline package that is not installed

- **WHEN** a baseline is named whose package is not importable
- **THEN** the tool exits with an error naming the package and how to install it, before reading any image

### Requirement: A baseline's number carries its caveat

A baseline that was not trained for the task being measured SHALL carry a caveat, and
the tool MUST report it with the number and not instead of it: the result MUST carry a
compact, space-free `caveat` id on the `RESULT` line and in the JSON, and the JSON MUST
also carry a `caveat_note` sentence saying what differs. A baseline with nothing to
declare MUST report no caveat at all rather than an empty one.

#### Scenario: The class-set handicap is on the number

- **WHEN** a baseline trained on another class set is scored
- **THEN** the RESULT line carries a space-free `caveat` id
- **AND** the JSON carries that id and a `caveat_note` naming what differs

#### Scenario: A comparable baseline declares nothing

- **WHEN** a baseline with no caveat is scored
- **THEN** neither the result nor the RESULT line carries a caveat field

### Requirement: A baseline's latency is measured with the bench tool's own method

`jetson/bench_model.py` SHALL accept `--baseline <name>` and time it with the method it
already uses for an hcm model: the same warmup, the same iterations, the same
device-synchronised per-iteration clock, and the same mean/p50/p90 over the timed
iterations. It MUST time the baseline at its adapter's declared size by default and at
`--input-size` when one is given, and MUST report which of the two it used. It MUST
refuse a `--checkpoint`/`--arch`/`--engine` that does not belong to a baseline. The
report SHALL name the baseline and its caveat, so a latency and a score describe the
same configuration.

#### Scenario: A baseline latency report

- **WHEN** a baseline is benchmarked
- **THEN** the report names the baseline, its input size and where that size came from, the batch size and its caveat, and carries the same timing fields an hcm model's report carries

#### Scenario: A refused baseline argument

- **WHEN** a baseline is given with `--arch`, `--checkpoint` or `--engine`
- **THEN** the tool refuses with an error naming the conflict, before timing anything
