# jetson-realtime-demo Specification

## Purpose

The real-time demo that runs on the Jetson and shows what the trained model does with a live camera. It is a demo rather than a service, and its contract is deliberately thin: it loads published weights from a fixed directory and serves an annotated stream.

## Requirements

### Requirement: The demo loads its weights from a fixed artifact directory

The demo MUST load its weights from a fixed artifact directory on the device rather than from a command-line argument or an object store; a metadata sidecar in that directory MAY be read when present and MUST NOT be required. The demo MUST fail at startup rather than serve a stream it cannot annotate when the weights are unusable.

#### Scenario: The weights come from the device's artifact directory

- **WHEN** the demo starts
- **THEN** it loads the weights from the fixed artifact directory on the device, reading metadata only if present

#### Scenario: Missing weights stop the demo

- **WHEN** the weights file is absent or unreadable
- **THEN** the demo raises at startup instead of starting a stream

### Requirement: Weights are unwrapped and the class count inferred from the state dict

The demo MUST accept a checkpoint whose tensors are nested under a wrapper key as well as a plain state dict, and MUST infer the output class count from the final classification tensor's shape rather than from a configured constant, so a checkpoint trained with a different class count still runs.

#### Scenario: A wrapped checkpoint loads

- **WHEN** the checkpoint nests its tensors under a wrapper key
- **THEN** the demo unwraps it and loads the tensors

#### Scenario: The class count follows the checkpoint

- **WHEN** the final tensor's shape implies a class count different from the default
- **THEN** the model is built with the inferred count

### Requirement: The demo serves the annotated stream over HTTP on port 5000

The demo MUST run an HTTP server on all interfaces on port 5000 and MUST serve the annotated frames at the path `/video` as an endless multipart stream of JPEG frames, so a browser or a viewer can consume it without polling. The server MUST run without the debug reloader, because the reloader would start a second process holding the camera.

#### Scenario: The stream is served at the video path

- **WHEN** a client requests the video path
- **THEN** the response is a multipart stream of JPEG frames at the documented port

#### Scenario: The server does not double-start

- **WHEN** the server thread starts
- **THEN** the debug reloader is off so the camera is opened by exactly one process

#### Scenario: The stream is endless

- **WHEN** a client stays connected
- **THEN** frames continue to be produced until the client disconnects

### Requirement: Frames are annotated with the model's prediction

Each frame MUST be read from the device camera at the demo's fixed capture settings, preprocessed to the model's fixed input size, and returned with the model's output rendered over it, so the stream shows the prediction rather than the raw camera image. A frame the model cannot be run on MUST NOT take the stream down.

#### Scenario: A served frame carries the prediction

- **WHEN** a frame is served on the stream
- **THEN** the model's output for that frame is rendered onto the frame

#### Scenario: The camera is opened at the configured index

- **WHEN** the demo starts and the camera is available
- **THEN** capture begins from the configured camera index at the demo's resolution and frame rate

#### Scenario: An unavailable camera stops the demo

- **WHEN** the camera cannot be opened
- **THEN** the demo raises rather than serving an empty stream

### Requirement: Each produced frame is served at most once per viewer

A viewer of the stream MUST receive each produced frame at most once, and MUST wait for a
newly produced frame rather than re-reading the most recent one in a loop. Delivery MUST NOT
spin: when there is no new frame the viewer blocks until the capture loop publishes one.
A viewer that connects between two frames MUST be sent the next frame produced rather than
waiting for the frame after it, so a slow viewer sees the newest frame and never a backlog.

#### Scenario: A viewer does not receive the same frame twice

- **WHEN** one frame is produced and a viewer reads from the stream repeatedly
- **THEN** it receives that frame once and its subsequent read blocks until another frame is
  produced

#### Scenario: Repeated reads with no new frame block rather than repeat

- **WHEN** the capture loop produces nothing for a period and a viewer reads during that
  period
- **THEN** no frame is delivered to it in that period

#### Scenario: A viewer receives every distinct frame once

- **WHEN** a sequence of distinct frames is produced while one viewer is reading
- **THEN** it receives each distinct frame in order and no frame more than once

#### Scenario: An idle stream does not burn a core

- **WHEN** the stream has no new frame to deliver
- **THEN** the viewer's read blocks on a wait rather than iterating, so an idle viewer costs
  no measurable CPU

#### Scenario: Two viewers each get each frame once

- **WHEN** two viewers read the same stream
- **THEN** each receives each produced frame once, independently of the other's pace

### Requirement: The demo is launched by a documented script with its required environment

The demo MUST be started by a launch script that sets the environment the demo needs to run
on this device: the repository root on the Python import path, and CUDA memory caching
disabled for the torch build that runs there, because the caching allocator fails on the
first forward with an NVML internal assertion on the device's torch and JetPack versions.
The script MUST run the demo from the repository's own virtual environment, MUST survive the
operator's shell exiting, and MUST write the demo's output to a log file rather than the
terminal. The reason for the caching setting MUST be recorded in the script, so it is not
mistaken for a tuning choice and removed.

#### Scenario: The script supplies both required variables

- **WHEN** the launch script starts the demo
- **THEN** the demo process sees the repository root on its import path and CUDA memory
  caching disabled

#### Scenario: The launch does not die with the shell

- **WHEN** the operator exits the shell they launched from
- **THEN** the demo keeps running and its output is available in the log file

#### Scenario: The caching setting is documented where it is set

- **WHEN** a reader opens the launch script
- **THEN** the reason for disabling caching is stated there, naming the failure it prevents

#### Scenario: The script uses the repository's own interpreter

- **WHEN** the script resolves which Python to run
- **THEN** it uses the virtual environment in the repository rather than an ambient one

### Requirement: The demo's input size follows the model it loaded

The demo MUST resize each camera frame to the input size its loaded weights were trained at
rather than to a size fixed in the demo's source. It MUST resolve that size at startup from
these sources, in this order: the environment variable `ECS_HCM_DEMO_INPUT_SIZE` when that
variable states a usable size; then an `input_size` entry inside the metadata sidecar's
`custom_metadata`; then an `input_size` entry at the top level of the same sidecar; then the
built-in default of 192, which is the size of models published before a checkpoint recorded
its own. The demo MUST print the chosen size and the source it came from once at startup, so
an operator can see which geometry is in use. The resolution MUST be reachable without a
camera, a GPU, or a model. This requirement concerns the input geometry only: the weights
continue to come from the fixed artifact directory, and no other setting is added to the
weights path.

The demo MUST keep resizing the whole frame to a square, as it does today, and MUST NOT crop
it: the demo shows the whole field of view rather than the training pipeline's window into it.

A size that a source states but that cannot be used MUST stop the demo at startup with an
error naming the source and the value, and MUST NOT be replaced by the default. A frame
resized to a size other than the one the model was trained at produces an overlay computed at
the wrong resolution, which is the failure this requirement exists to prevent. A value is
usable when it is a positive whole number; an absent or empty value is unstated and falls
through to the next source rather than failing.

#### Scenario: The environment overrides the model's own metadata

- **WHEN** `ECS_HCM_DEMO_INPUT_SIZE` states a usable size and the metadata sidecar states a
  different one
- **THEN** the demo uses the size from the environment and reports the environment as its
  source

#### Scenario: The model's metadata supplies the size

- **WHEN** no size is stated in the environment and the sidecar's `custom_metadata` carries an
  `input_size`
- **THEN** the demo uses that size and reports the metadata as its source

#### Scenario: A top-level size is read when `custom_metadata` has none

- **WHEN** no size is stated in the environment or in `custom_metadata`, and the sidecar has a
  top-level `input_size`
- **THEN** the demo uses that top-level size and reports where it came from

#### Scenario: A model published before the size was recorded uses the default

- **WHEN** no size is stated in the environment and the sidecar has no `input_size`, or no
  sidecar is present at all
- **THEN** the demo uses 192 and reports the default as its source

#### Scenario: A stated size that cannot be used stops the demo

- **WHEN** the environment or the sidecar states a size that is not a positive whole number
- **THEN** the demo raises at startup naming the source and the value, and does not fall back
  to the default or to another source

#### Scenario: The chosen size and its source are reported once

- **WHEN** the demo starts
- **THEN** it prints the resolved size and the source it came from before it serves a stream

#### Scenario: A 352-pixel model is fed 352-pixel frames

- **WHEN** the resolved size is 352
- **THEN** the frame handed to the model, the warmup tensor, and the logits are all at 352,
  and the overlay is drawn back onto the full camera frame

### Requirement: The offline video demo's input size follows the model it loaded

The offline frames-directory demo (`video/video_demo.py`) MUST resolve its input
size with the same resolver and in the same order as the real-time demo: the
`ECS_HCM_DEMO_INPUT_SIZE` environment override, then the metadata's
`custom_metadata.input_size`, then a top-level `input_size`, then 192. It MUST
reuse the real-time demo's resolver rather than a copy of it. The resolved size
MUST reach the frame resize, the warmup tensor, and the logits upsample, and the
demo MUST print the size and its source. A stated size that cannot be used MUST
stop the demo rather than fall back to the default.

#### Scenario: A 352 model is fed 352-pixel frames

- **WHEN** the metadata sidecar records `custom_metadata.input_size` as 352 and no override is set
- **THEN** the demo resizes each frame to 352x352 and prints `input size: 352 (from metadata.custom_metadata.input_size)`

#### Scenario: A model without the field keeps the old size

- **WHEN** the metadata has no `input_size` and no override is set
- **THEN** the demo uses 192, as before

### Requirement: A present but unreadable metadata sidecar stops the demo

Both the real-time demo and the offline video demo MUST load the `metadata.json` sidecar through the shared resolver module. A missing sidecar MUST remain legal and resolve as if there were no metadata. A sidecar that exists but cannot be read, is not valid JSON, or is not a JSON object MUST stop the demo at startup with a message naming the file and the error, in the same way a stated but unusable input size stops it; it MUST NOT be replaced by the default size.

#### Scenario: A missing sidecar falls back to the default

- **WHEN** no `metadata.json` exists and no environment override is set
- **THEN** the demo starts at the default input size of 192

#### Scenario: A truncated sidecar stops the demo

- **WHEN** `metadata.json` exists but its JSON is truncated
- **THEN** the demo exits at startup with a message naming the file and the parse error
- **AND** it does not run at 192

#### Scenario: A sidecar that is not an object stops the demo

- **WHEN** `metadata.json` holds a JSON array, number, string or null
- **THEN** the demo exits at startup naming the file

### Requirement: The single-image demo's input size follows the model unless overridden

`video/image_demo.py` SHALL use `--input-size` when it is given. When it is not given, the demo SHALL read the `metadata.json` next to `--weights` and resolve the size with the same rule as the other demos (environment, then `custom_metadata.input_size`, then top-level `input_size`, then 192), and SHALL print the size and its source. A given flag that is not a positive whole number, a stated size that cannot be used, or a sidecar that is present but unreadable MUST stop the demo.

#### Scenario: The flag wins
- **WHEN** the demo is run with `--input-size 288` against a model whose metadata says 352
- **THEN** it feeds the model 288-pixel images and reports the flag as the source

#### Scenario: Without the flag the model's size is used
- **WHEN** the demo is run without `--input-size` against a model whose metadata says 352
- **THEN** it feeds the model 352-pixel images and reports the metadata as the source

#### Scenario: A model without the field keeps the old size
- **WHEN** the demo is run without `--input-size` and the weights have no `metadata.json`
- **THEN** it uses 192

### Requirement: The size resolver refuses metadata that is not an object

The input-size resolver SHALL treat metadata that is neither absent nor a JSON object as corrupt and MUST raise rather than return the default. Absent metadata SHALL still resolve to the default.

#### Scenario: A list is refused
- **WHEN** the resolver is given metadata that is a list
- **THEN** it raises an error saying the metadata is not an object, and the demo stops

#### Scenario: No metadata still means the default
- **WHEN** the resolver is given no metadata and no environment override
- **THEN** it returns 192

### Requirement: The demos load only a binary head, at the model's own stride

Each demo (`jetson/realtime_demo.py`, `video/video_demo.py`, `video/image_demo.py`) SHALL refuse at load, with a message, a checkpoint whose segmentation head does not have exactly 2 classes. Each demo SHALL build the model at the `out_stride` taken, in this order, from its `--out-stride` flag, the `ECS_HCM_DEMO_OUT_STRIDE` environment variable (an empty value counts as unset), the sidecar's `custom_metadata.out_stride`, its top-level `out_stride`, the stride the checkpoint recorded, else 16. A null recorded stride is unstated and falls through. Each demo SHALL report the stride and its source on the same line as the input size and its source. A flag or environment value that disagrees with a recorded stride MUST be honoured with a warning, not an error. A stated stride that is not 8, 16 or 32 MUST stop the demo. `jetson/run_demo.sh` SHALL pass its arguments through to the camera demo.

#### Scenario: A non-binary checkpoint is refused
- **WHEN** a demo is given a checkpoint whose segmentation head has 3 classes
- **THEN** it stops at load with a message naming the class count

#### Scenario: The recorded stride is used
- **WHEN** the model's metadata states `out_stride` 8 and no override is given
- **THEN** the demo builds the model at stride 8 and reports the metadata as the source

#### Scenario: A model without a recorded stride uses 16
- **WHEN** neither the sidecar nor the checkpoint states an `out_stride`, or the sidecar states null
- **THEN** the demo builds the model at stride 16

#### Scenario: The flag beats the environment and the sidecar
- **WHEN** a demo runs with `--out-stride 32`, `ECS_HCM_DEMO_OUT_STRIDE=16` and a sidecar stating 8
- **THEN** it builds at 32, reports `--out-stride` as the source, and warns that 32 overrides the recorded 8

### Requirement: Camera and encode failures are visible

The realtime demo MUST retry a failed camera read a bounded number of times
before giving up, and when every try fails it MUST exit with a nonzero status
and a message naming the try count; it MUST NOT treat a lost camera as a clean
stop. A failed JPEG encode MUST produce a warning, rate-limited so a persistent
failure cannot flood the log, and the warning SHALL report how many failures it
suppressed since the last one.

#### Scenario: A brief camera hiccup is ridden out

- **WHEN** the camera fails two reads and then returns a frame
- **THEN** the demo continues with that frame

#### Scenario: A camera that stays gone fails the process

- **WHEN** every read in the retry budget fails
- **THEN** the demo exits nonzero with a message naming the number of tries

#### Scenario: Encode failures warn without flooding

- **WHEN** JPEG encoding fails repeatedly within one warning interval
- **THEN** one warning is printed, and the next one after the interval reports the count suppressed in between

### Requirement: Demo paths take environment overrides

The realtime and video demos MUST read their weights and metadata directory from
`ECS_HCM_DEMO_ART_DIR` when it is set, and the video demo MUST read its input
and output frame directories from `ECS_HCM_DEMO_FRAMES_DIR` and
`ECS_HCM_DEMO_OUT_DIR`. When a variable is unset the demo SHALL use the path it
used before this change. Neither demo MUST create a directory when its module
loads; the video demo SHALL create its output directory only after it has found
frames to process.

#### Scenario: Unset variables keep the Jetson paths

- **WHEN** none of the variables is set
- **THEN** the demos use `/home/user/models`, `~/clustering/videos/frames` and `~/clustering/videos/frames_out`

#### Scenario: A set variable moves the path

- **WHEN** `ECS_HCM_DEMO_ART_DIR` names another directory
- **THEN** the weights and metadata are read from that directory

#### Scenario: A run without frames creates nothing

- **WHEN** the video demo finds no frames
- **THEN** it stops before creating the output directory

### Requirement: The demos share one model loader

The realtime, video and image demos SHALL load their model through one function, `jetson/demo_model.py:load_demo_model`. That module MUST NOT import cv2, so the load the demos run can be exercised where cv2 is absent. The function SHALL apply the binary-head, stride and decoder-version rules of the demo requirements, build the resolved decoder version, and load the weights strictly.

#### Scenario: A v2 checkpoint as uploaded loads in the demos
- **WHEN** a v2 model is uploaded by the trainer's uploader and its `model.pth` and `metadata.json` are placed in a demo artifact directory
- **THEN** `load_demo_model` builds a v2 decoder, reports the metadata as the version's source, and its outputs equal the uploaded model's

#### Scenario: A v1 checkpoint still loads
- **WHEN** the same is done with a v1 model
- **THEN** it builds v1 and its outputs equal the uploaded model's

### Requirement: A checkpoint is promoted to the demo by a staged, scored, reversible tool

`jetson/promote_model.py` SHALL promote a checkpoint to the demo in three separate steps: `stage`, `accept` and `apply`. It SHALL match the demo process only with the bracket pattern `[r]ealtime_demo\.py`. It SHALL read MinIO credentials and the endpoint only from the environment and SHALL never print them; a failed fetch SHALL be reported by the S3 error's code and message, or otherwise by the exception's type only, never with the endpoint or the user.

#### Scenario: Stage copies a checkpoint and records its hashes
- **WHEN** `stage coco-models/hcm_coco_seg_best/<stamp>` runs with `MINIO_ENDPOINT`, `MINIO_USER` and `MINIO_PASSWORD` set
- **THEN** `model.pth` and `metadata.json` are written to `~/models_staging/<stamp>`, together with a manifest of their source and sha256

#### Scenario: Stage refuses rather than overwrite or guess
- **WHEN** a credential or the endpoint is missing, the endpoint is malformed, an object is missing, `metadata.json` does not parse, or the staging dir already exists
- **THEN** stage exits 2 and leaves no partial staging dir

#### Scenario: A failed fetch does not reveal the endpoint
- **WHEN** fetching an object fails with an error whose text contains the MinIO host, port or user (e.g. a connection failure)
- **THEN** neither the endpoint nor the user appears in the tool's output or its error

#### Scenario: Accept passes only a strictly better candidate and always restarts the demo
- **WHEN** `accept <staged dir>` runs while the demo is up
- **THEN** the demo is stopped, the candidate and the baseline are scored on COCO val with `ecs_hcm_model.eval.seg`, and the demo is restarted with the command, working dir and environment it had, even if scoring fails
- **AND** both IoUs are printed and the result is PASS (exit 0) only if the candidate's IoU is higher than the baseline's, else FAIL (exit 1)
- **AND** if eval gives no score for either checkpoint, does not report `n_failed` for both, or the two report a different `n_failed`, accept prints what it has and exits 2 without a verdict, still restarting the demo

#### Scenario: Apply refuses without confirmation or a v2-capable code dir
- **WHEN** `apply` runs without `--yes`, or `--code-dir` lacks `jetson/run_demo.sh`, `jetson/demo_model.py` or `.venv/bin/python`
- **THEN** it exits non-zero before stopping the demo or touching `~/models`

#### Scenario: Apply keeps the old weights and checks the stream
- **WHEN** `apply <staged dir> --code-dir <repo> --yes` runs
- **THEN** `~/models` is renamed to `~/models_prev_<UTC stamp>` and never deleted, the staged files become the new `~/models`, and `run_demo.sh` is started from `--code-dir`
- **AND** success requires HTTP 200 and a first stream part from `:5000/video` within the timeout
- **AND** a rollback command is printed right after `~/models` is renamed, before the copy or the restart can fail

#### Scenario: A demo process that cannot be read is refused
- **WHEN** `pgrep` matches a process whose `/proc/<pid>` entries cannot be read (another user's process)
- **THEN** accept and apply print a warning and exit 2 without stopping anything or touching `~/models`

### Requirement: A latency bench times the demo's own inference path
`jetson/bench_model.py` SHALL time the model through the same build and forward the realtime demo runs (`build_demo_model` / `load_demo_model` and `demo_forward`), not a copy of them, and SHALL NOT import cv2.

#### Scenario: Random weights, either decoder
- **WHEN** the bench runs with `--arch v1` or `--arch v2` and no `--checkpoint`
- **THEN** it builds the model with random weights through `build_demo_model` and times `--iters` calls of `demo_forward` after `--warmup` untimed ones

#### Scenario: A checkpoint is loaded as the demo loads it
- **WHEN** `--checkpoint DIR` is given
- **THEN** the model comes from `load_demo_model` on `DIR/model.pth` with `DIR/metadata.json`, so its decoder version and stride are the checkpoint's, and an `--arch` the checkpoint contradicts is refused with exit 2

#### Scenario: The report
- **WHEN** the bench finishes
- **THEN** it prints arch, input size, device, dtype, parameter count and per-iteration mean, p50 and p90 ms, plus peak CUDA memory on CUDA; with `--json` it prints one JSON object with the same fields

#### Scenario: fp16 is an explicit opt-in
- **WHEN** `--fp16` is given
- **THEN** the forward runs under float16 autocast on CUDA, which the demo itself does not use; on a non-CUDA device the bench refuses with exit 2

### Requirement: The demos have an opt-in fp16 mode
The realtime, video and image demos SHALL accept `--fp16` (the realtime demo through `run_demo.sh`'s pass-through arguments), SHALL run every forward, warmup included, through `jetson.demo_model.demo_forward`, and SHALL default to fp32.

#### Scenario: Default is fp32
- **WHEN** a demo runs without `--fp16`
- **THEN** its forward runs under `no_grad` with no autocast context, exactly as before

#### Scenario: fp16 on CUDA
- **WHEN** a demo runs with `--fp16` on a CUDA device
- **THEN** each forward runs under CUDA float16 autocast (the eval tool's fp16) and the logits handed to the overlay are fp32

#### Scenario: fp16 off CUDA is refused
- **WHEN** `--fp16` is given and the device is not CUDA
- **THEN** the demo stops before the camera or input is opened, saying `--fp16` needs a CUDA device

### Requirement: The latency bench takes a checkpoint's input size as the demo does

With `--checkpoint` and no `--input-size`, `jetson/bench_model.py` MUST resolve the input size with the realtime demo's resolver (`jetson.input_size.resolve_input_size`: the environment, then the metadata, then the default), not with code of its own. An explicit `--input-size` that differs from a size the checkpoint's metadata states MUST be refused with exit 2. An explicit `--input-size` MUST be accepted and used when the metadata states no size, and MUST win over `ECS_HCM_DEMO_INPUT_SIZE`. Without `--checkpoint`, the default input size MUST stay 352.

#### Scenario: Size from the metadata

- **WHEN** the bench runs `--checkpoint DIR` without `--input-size` and DIR's metadata states an input size
- **THEN** it times the model at that size and reports it

#### Scenario: A contradicting flag

- **WHEN** `--input-size` is given and differs from the input size DIR's metadata states
- **THEN** the bench prints one `error:` line naming both sizes and exits 2

#### Scenario: A flag with no stated size

- **WHEN** `--input-size` is given and DIR's metadata states no input size
- **THEN** the bench times the model at the flag's size and reports it

#### Scenario: The flag beats the environment

- **WHEN** `--input-size` is given and `ECS_HCM_DEMO_INPUT_SIZE` is set to a different size
- **THEN** the bench times the model at the flag's size and reports it

### Requirement: The latency bench reports a refused checkpoint as a refusal

When `load_demo_model` refuses the checkpoint, the bench MUST print the loader's message as one `error:` line and exit 2, the same as its own refusals.

#### Scenario: A checkpoint the loader refuses

- **WHEN** the checkpoint states a decoder version its weights contradict
- **THEN** the bench prints `error: Cannot use this checkpoint: ...` and exits 2

### Requirement: Each timed iteration of the latency bench is synchronised and percentiles are by rank

Every timed call MUST run with the model in eval mode and gradients disabled. On CUDA each timed iteration MUST include a device synchronise before its end time is taken. `p50_ms` and `p90_ms` MUST be the nearest-rank 50th and 90th percentiles of the timed iterations.

#### Scenario: A known distribution

- **WHEN** ten timed iterations take 1 to 10 ms
- **THEN** the bench reports p50 5 ms, p90 9 ms and mean 5.5 ms

### Requirement: Uncertainty capture is opt-in and leaves the feed unchanged

The realtime demo MUST capture nothing unless `--capture-uncertain DIR` is given. With
capture on, the feed's contract is unchanged: same port and route, same camera resolution,
same overlay, same input size and same model. With the flag on but the MinIO environment
incomplete, the demo MUST refuse to start, naming the missing variable, rather than run
with a capture that silently fails to upload. `MINIO_SECURE` is optional and MUST select
https when set to a true value; the chosen scheme MUST be reported once at startup, so a
TLS bucket is not silently attempted over plain http. The upload client MUST be constructed
with an explicit region — `MINIO_REGION`, default `us-east-1` — and MUST NOT rely on region
discovery: the upload-only credential is granted `PutObject` and `AbortMultipartUpload` on
the capture prefix and nothing else, so a client that asks the server for the bucket's
location before writing has every upload denied. The credential's grant MUST NOT be widened
to permit that lookup.

#### Scenario: No flag, no capture
- **WHEN** the demo is started without `--capture-uncertain`
- **THEN** no frame is written or uploaded, and the frame path is the one it runs today

#### Scenario: Capture on, feed unchanged
- **WHEN** the demo runs with capture on and a reachable bucket
- **THEN** the stream serves the same resolution and the overlay is drawn by the same code
  as with capture off

#### Scenario: The upload client never asks for the bucket location
- **WHEN** the demo starts with capture on and uploads a frame
- **THEN** the client is built with the configured region, `us-east-1` unless `MINIO_REGION`
  says otherwise, so the write needs no request beyond the granted put

#### Scenario: Half-configured capture is refused at startup
- **WHEN** the demo is started with `--capture-uncertain` and one of `MINIO_ENDPOINT`,
  `MINIO_USER`, `MINIO_PASSWORD` is unset
- **THEN** it exits before serving, with an error naming the variable

#### Scenario: The upload scheme is chosen and visible
- **WHEN** the demo starts with capture on, with `MINIO_SECURE` set and again without it
- **THEN** the first uses https and the second http, and the startup output names the scheme
  it chose

### Requirement: The per-frame uncertainty score comes from the model's own probabilities

For every frame the demo MUST compute, from the `p_fg` it already produces, the fraction of
model-grid pixels whose foreground probability lies strictly between 0.35 and 0.65, and the
maximum `p_fg`. The score MUST NOT be computed by a second model or by re-running the model.

#### Scenario: The score is the band fraction
- **WHEN** every `p_fg` is 0.5
- **THEN** the score is 1.0; when every `p_fg` is 0.9 the score is 0.0

#### Scenario: The score uses the live probabilities
- **WHEN** a frame is scored
- **THEN** the score equals the band fraction of the same `p_fg` tensor the overlay is drawn
  from in that iteration

### Requirement: Frame selection is sparse and capped

A frame MUST be saved only when its score is at least 0.02 AND at least the configured
minimum spacing has passed since the last saved frame AND neither the hourly cap nor the
daily cap is reached. The caps MUST be enforced as hard limits, and each drop MUST be
counted by reason.

#### Scenario: Consecutive uncertain frames are thinned
- **WHEN** two consecutive frames both qualify inside the minimum spacing
- **THEN** only the first is saved

#### Scenario: A frame below the score floor is never a candidate
- **WHEN** a frame's score is below the floor
- **THEN** it is not a candidate and nothing is written, whatever the caps and spacing allow

#### Scenario: The hourly cap stops saving
- **WHEN** the hourly cap is reached
- **THEN** further qualifying frames are not saved until the hour rolls over, and the drop
  is counted

#### Scenario: The daily cap stops saving
- **WHEN** a day's saved count reaches the daily cap
- **THEN** further qualifying frames are not saved until the next local day, and the drop
  is counted

### Requirement: A control frame is sampled independently of the score

Independently of the uncertainty score, the policy MUST save one control frame every 5
minutes — the first frame after each boundary — through the same writer and uploader, capped
at 288 per local day. Control frames MUST be named distinctly and recorded with a `kind` of
`control`, and MUST NOT count against the uncertain-frame caps.

#### Scenario: A confident frame can be the control
- **WHEN** the first frame after a 5-minute boundary has a score of 0.0
- **THEN** it is saved as a control frame

#### Scenario: Controls do not consume the uncertain caps
- **WHEN** a day has reached the uncertain daily cap
- **THEN** control frames are still sampled, up to their own cap

#### Scenario: Controls are dense enough for a per-night mean
- **WHEN** a night saves uncertain frames for 2.5 hours
- **THEN** about 30 control frames are available in that night, enough for that night's own
  control mean

### Requirement: Saving never blocks the inference loop

The camera frame MUST be handed to a bounded queue and written by a background thread. When
the queue is full the frame MUST be dropped and counted; the loop MUST NOT wait on the disk.

#### Scenario: A slow disk does not stall the stream
- **WHEN** the writer thread is blocked long enough for the queue to fill
- **THEN** the loop keeps reading frames, the excess is dropped, and the drop counter rises

#### Scenario: Capture adds no measurable slowdown
- **WHEN** the loop's frame rate is measured in three alternating off/on pairs — six 60 s
  windows in one session: off, on, off, on, off, on
- **THEN** the median of the capture-on windows is at least 95% of the median of the
  capture-off windows, and the spread of both is reported

### Requirement: Saved frames are the raw camera image with their selection metadata

Each saved frame MUST be the camera frame as read, before any overlay, stored as JPEG with
its UTC timestamp in the file name — control frames prefixed `control-` — and one JSON record
holding the file name, kind, timestamp, score, maximum probability, frame rate, model
identity and input size. The record MUST be
kept in the capture directory's index and uploaded beside the frame, so the analysis can
read it without the device. The capture directory MUST stay within a byte cap, pruning
uploaded frames oldest first, and MUST skip a capture (counting it) rather than exceed the
cap. The cap MUST default to 512 MiB and MUST be overridable with `CAPTURE_DISK_CAP_BYTES`
(a positive byte count; any other value refuses to start and names the variable), so a
verification run can reach the cap in minutes rather than hours.

#### Scenario: The overlay is not in the saved file
- **WHEN** a frame is saved while the overlay draws a heatmap and contours
- **THEN** the saved image shows the camera frame without the overlay

#### Scenario: The cap holds when uploads are blocked
- **WHEN** the bucket is unreachable and the directory reaches its byte cap
- **THEN** no further frame is written, the drop is counted, and the directory does not
  exceed the cap

#### Scenario: A verification run sets a cap it can reach
- **WHEN** the demo starts with `CAPTURE_DISK_CAP_BYTES` set to a small positive number
- **THEN** the writer uses that cap instead of 512 MiB, and a value that is not a positive
  byte count stops the demo before it serves, naming the variable

### Requirement: The session reports its counters when it stops

When the capture session stops, it MUST print one line `capture stats: <JSON object>` to
stdout holding the counters it collected — at least the frames written, the queue drops,
the cap refusals and the upload failures — so a run nobody watched can account for what it
saved and what it dropped without a debugger. A counter that never fired MAY be absent.

#### Scenario: A dropped frame is visible in the run's own log
- **WHEN** a capture run refuses a frame because the disk cap is reached and then stops
- **THEN** its log ends with one `capture stats:` line whose object counts that refusal

### Requirement: Upload runs off the loop, retries, and never raises into it

Uploads MUST run on a thread independent of the inference loop, retry with backoff while
the bucket is unreachable, and mark each object only after a confirmed upload. An upload
failure MUST NOT stop the loop, the capture, or the process.

#### Scenario: A failed upload is retried, not raised
- **WHEN** an upload attempt fails
- **THEN** the loop and capture continue, the failure is counted, and the object is retried
  after a backoff

#### Scenario: A captured frame is not deleted before it is stored
- **WHEN** an upload has not been confirmed
- **THEN** the local file is still present and is not pruned as uploaded

### Requirement: A stop request is honoured however the demo was started

The demo MUST install its own handlers for `SIGINT` and `SIGTERM` rather than rely on the
inherited disposition: a demo started as a background job inherits `SIGINT` set to
`SIG_IGN`, and CPython installs its default handler only when the disposition is not
`SIG_IGN`, so such a process ignores a terminal's Ctrl-C and a supervisor's stop outright.
The handlers MUST be in place before the demo does anything that can take time or block —
reading metadata, building the model, loading the engine, warming up, opening the camera —
so that a request arriving during start-up is held by the interpreter and acted on when
that call returns, instead of being discarded at delivery. The first request MUST be
remembered and return, so the loop finishes the frame it is on and leaves normally — the
capture session's `atexit` hook flushes its queue and prints its counters only on a normal
exit. A second request MUST raise `KeyboardInterrupt`, so a demo wedged inside a driver
call can still be stopped.

#### Scenario: A background demo stops instead of being killed
- **WHEN** a demo that was started with `SIGINT` ignored receives `SIGINT`
- **THEN** it leaves its loop and exits normally, and its capture session flushes its queue

#### Scenario: A stop during start-up is not lost
- **WHEN** a demo that was started with `SIGINT` ignored is sent `SIGINT` while it is still
  loading its model, engine or camera
- **THEN** the request is still held rather than discarded, and the demo leaves before it
  enters its frame loop

#### Scenario: A second request escapes a wedged loop
- **WHEN** a stop has been requested and has not been acted on, and another stop request
  arrives
- **THEN** the handler raises `KeyboardInterrupt`
