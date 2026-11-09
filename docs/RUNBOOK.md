# Runbook

Operational rules learned the hard way. Each section starts with the commands;
the code named in each is the source of truth if this file and it ever differ.

Standing rules: never launch cluster training, push to GitHub, or change the
Jetson's installed environment (JetPack, CUDA, system packages) without the
owner's explicit go-ahead.

## 1. Warm-start retrain (`--init-from`)

```sh
python main.py train_hcm_imagenet1k --coco --input-size 352 --batch_size 2 \
    --max_epochs <N> --init-from coco-models/hcm_coco_seg_best/<stamp>
```

- Always pass `--max_epochs` and `--batch_size`; the defaults (5 and 8) are not
  the 352 run's.
- `--init-from` builds a fresh model (`--init-decoder-arch`, default `v2`) and
  copies in every tensor whose name and shape fit. It implies `--new_model`
  and refuses to start if the source shares no tensor with the model or records
  a decoder version its weights contradict.
- **Check the startup line.** For a v1 checkpoint into v2 it must read:

  ```
  Warm start from <spec> into decoder v2: 502 tensors copied, 0 skipped (shape changed),
  8 initialised (fuse [I|0]), 0 new (left at init), ...
  ```

  Any `skipped` or `new` means the source is not the checkpoint you think it
  is. Stop the run. The same four counts are logged to MLflow as
  `init_*_keys`.
- **What gets uploaded, and when.** Each epoch uploads
  `hcm_coco_seg_epoch<N>/<stamp>` with `is_best` in its `metadata.json`
  (`core/trainer.py:2087-2094`, `save_epoch_checkpoint` at `:1030-1081`).
  `is_best` is true when that epoch's `val_iou` beats every earlier epoch's
  **in the same training process** (`:2073-2075`), so epoch 0 is `is_best`
  whenever it logs a `val_iou`. The single `hcm_coco_seg_best/<stamp>` is
  uploaded only after training ends, from the best epoch's weights
  (`:2503-2532`). **There is no new `_best` while the run is going.** A
  mid-epoch save on interruption is uploaded with `is_best=False` (`:1868`)
  and never counts below.
- **Every resume starts the count again.** Each training process starts with
  no best epoch (`:1506-1507`) and numbers its epochs from 0 (`:1536`). So
  after a resume, its first epoch is `is_best` whatever its `val_iou`, and it
  uploads `hcm_coco_seg_epoch0` again under a new stamp. `is_best` and the
  end-of-run `_best` mean "best of this process", not "best of the retrain".
  "Newest" below always means the newest `upload_timestamp`, never the
  highest `<N>`.
- **If the run dies before its first epoch checkpoint, restart it with the
  same `--init-from` command.** A plain resume reads every `metadata.json` in
  `coco-models` (`core/model_upload_download.py:443`) and picks the newest
  `is_best` one (`select_resume_checkpoint`, `:153-182`). Until
  `hcm_coco_seg_epoch0` exists that is still the old v1 `_best`, so a plain
  resume would continue the v1 lineage.
- **After the first epoch checkpoint a plain resume is fine.**
  `hcm_coco_seg_epoch0` is `is_best` and newer than the v1 `_best`, so it wins
  (newest `upload_timestamp`, `:180`). This holds only if epoch 0 logged a
  `val_iou`; check its `metadata.json` says `"is_best": true` first. From
  then on:

  ```sh
  python main.py train_hcm_imagenet1k --coco --model-base coco-models \
      --input-size 352 --batch_size 2 --max_epochs <N>
  ```

  It will warn that an older checkpoint (the v1 source) recorded a better
  `val_iou`. That is expected early in the retrain: newest wins by design.
- **If the run dies mid-way, a plain resume picks the newest
  `hcm_coco_seg_epoch<N>` whose `metadata.json` has `"is_best": true`.** That
  rule is unchanged, warning included. There is no `_best` for this run yet.
- **The model to promote is the best since the warm start, which is not
  always the newest.** If the retrain has never been resumed, it is the newest
  `is_best` epoch checkpoint. After any resume, compare every `is_best` epoch
  checkpoint since the warm start, and the `_best` of a resumed run that
  finished, and take the highest `val_iou`. A later process's `is_best` can be
  worse than one from before the death. Each `metadata.json` has what you
  need:
  - `upload_timestamp` (top level): keep those at or after the warm-started
    run's first `hcm_coco_seg_epoch0`;
  - `custom_metadata.decoder_arch` is `"v2"` for this retrain, which also
    excludes the v1 lineage;
  - `custom_metadata.is_best` must be `true`;
  - `custom_metadata.val_metrics.val_iou` is the score to compare.
- **Pace:** about 3.3 h per epoch at 352 / batch 2 (baseline
  `carefree-rook-316`: 9 epochs in 29.8 h).
- **Early stopping:** the run ends after 5 epochs without a lower `val_loss`,
  so it can stop before `--max_epochs`. The baseline planned 15 and stopped
  after 9.
- **`--lr-decay-start N` moves the cosine decay's first epoch** (omitted:
  today's schedule, 5 with warmup and 0 without). A start *inside* warmup
  (0-4) does not skip the stages: it tapers each group's lr from that epoch's
  stage value down to `MIN_LR_FACTOR` of it, the head group included. The
  factor always multiplies that epoch's stage value (the last one warmup set,
  once its stages stop at epoch 4), never the already-decayed lr, so the taper
  does not compound and reaches `MIN_LR_FACTOR` only at the final epoch: a
  start of 2 over 5 epochs is 0.525x at epoch 3 and 0.05x at epoch 4. The
  effective value is logged as the `lr_decay_start` run parameter and written
  into every epoch checkpoint's `custom_metadata`.
- **Compare against the baseline:**

  ```sh
  python tools/compare_runs.py <run name or id> [<baseline>] --epochs <N>
  ```

  It is read-only against MLflow, and the baseline defaults to
  `carefree-rook-316`. Pass `--epochs` when the run's `max epochs` param is not
  what you expect it to run (see early stopping), or the projected finish will
  be late.

## 2. Promoting a checkpoint to the Jetson demo

All three steps run on the Jetson, from a checkout that has
`jetson/promote_model.py`.

```sh
export MINIO_ENDPOINT=<host:port reachable FROM THE JETSON> MINIO_USER=... MINIO_PASSWORD=...
# MINIO_SECURE=1 for https
python jetson/promote_model.py stage  coco-models/hcm_coco_seg_best/<stamp>
python jetson/promote_model.py accept ~/models_staging/<stamp> --data-root <coco val root>
python jetson/promote_model.py apply  ~/models_staging/<stamp> --code-dir <checkout> --yes
```

- **What to stage.** If the run finished without ever being resumed, stage
  its `hcm_coco_seg_best/<stamp>`. While it is running, or after it died,
  there is no `_best` from this run (only the end of training uploads one),
  and the newest `hcm_coco_seg_best` is an older run's. After any resume,
  even a finished run's `_best` is only the best of its last process. In
  those cases stage the checkpoint with the highest `val_iou` since the warm
  start, chosen as in section 1, which may be an epoch checkpoint
  (`hcm_coco_seg_epoch<N>/<stamp>`) rather than a `_best`.
- **`MINIO_ENDPOINT`:** the trainer's in-cluster MinIO name does not resolve
  on the Jetson; use an address the Jetson can reach. The tool never prints the
  endpoint or the credentials, even on a failed fetch.
- **`stage`** writes `model.pth`, `metadata.json` and a sha256 manifest to
  `~/models_staging/<stamp>`. It refuses an existing dir and leaves nothing
  behind on failure.
- **`accept`** stops the demo, scores the staged candidate and `~/models` with
  `ecs_hcm_model.eval.seg` on COCO val (each at its own recorded input size),
  then restarts the demo as it was.
  - Exit 0 PASS: the candidate's IoU is strictly higher.
  - Exit 1 FAIL: a tie or worse.
  - Exit 2 no verdict: a missing score, or the two failed on different numbers
    of images.
  - Lay out the val set with `python tools/fetch_seg_val.py coco --data-root <dir>`.
- **Exit 2 from `stage` or `apply` means a refusal of any kind** (missing or
  malformed settings, a missing object, an existing dir, no `--yes`, an
  unusable `--code-dir`, an unreadable demo process). The one-line `error:`
  says which.
- **`--code-dir`** must contain `jetson/run_demo.sh`, `jetson/demo_model.py` (the
  v2-aware loader; older checkouts such as `~/repos/ecs_hcm_model-8a6360f` lack
  it and are refused), and `.venv/bin/python`. A fresh export has no venv, so
  symlink the shared one:

  ```sh
  ln -s ~/repos/ecs_hcm_model/.venv <checkout>/.venv
  ```

- **`apply` moves ALL of `~/models`, subdirectories included, to
  `~/models_prev_<UTC stamp>`.** Nothing is deleted, but backups kept inside
  `~/models` (e.g. `backup_imagenet_20260105`) end up under `models_prev_*`.
  Move any you want to keep at the top level first.
- **`apply` restarts `run_demo.sh` with no arguments.** A demo that was started
  with flags (e.g. `--out-stride 8`) comes back without them. Restart it by
  hand with its flags if so.
- **Success** needs HTTP 200 and a first frame on `:5000/video`. A 200 alone
  only proves the web server started, which happens before the model loads.
- **Rollback:** `apply` prints the exact commands right after it moves
  `~/models`, before anything else can fail. Copy them from its output. They
  have this shape:

  ```sh
  pkill -f '[r]ealtime_demo\.py'; { [ ! -e ~/models ] || mv ~/models ~/models_failed_<stamp>; } \
      && mv ~/models_prev_<stamp> ~/models
  cd <old demo cwd> && env <old demo env> nohup <old demo argv> >> realtime_demo_rollback.log 2>&1 &
  ```

- Promotion only swaps weights and code dirs. Anything touching the device's
  installed environment needs the owner's go-ahead (see the standing rules).

## 3. Launching the Jetson demo

```sh
<checkout>/jetson/run_demo.sh      # logs to <checkout>/run_demo.log; stream at :5000/video
```

- Always launch with `run_demo.sh`. Its header explains the two variables it
  sets (`PYTHONPATH` and `PYTORCH_NO_CUDA_MEMORY_CACHING=1`) and why neither is
  optional.
- **Over ssh, match processes with a bracketed pattern:**

  ```sh
  ssh jetson "pkill -f '[r]ealtime_demo\.py'"
  ```

  A plain `pgrep -f realtime_demo.py` inside `ssh host '...'` matches the
  remote shell running it, and a `pkill` would kill that shell. For anything
  longer than one command, put it in a script file and run that.

## 4. Running the demo from a TensorRT engine

One-off, on the Jetson, from the checkout (the export needs `onnx` and the
TensorRT Python bindings; both live in the system `dist-packages`):

```sh
PYTHONPATH=<checkout> python -m jetson.export_trt /home/user/models \
    --out /home/user/models/trt --verify \
    --verify-dir /home/user/video_run_20261005/stills_in
```

- The export checks both dependencies before it touches `--out`: a missing `onnx` or
  TensorRT is refused (`error: ...`, exit 2) and creates nothing, so a retry after
  installing it is not refused for overwriting a leftover directory.
- `--out` must not exist; the export refuses to overwrite it. Delete an old one
  first if you mean to rebuild.
- Two rewrites are installed for the export call and restored immediately after: the
  static `adaptive_avg_pool2d` (the legacy tracer loses the spatial extent after
  GroupNorm, so the operator has no size to derive its bins from), and the v2 decoder's
  pair of GroupNorms collapsed into one 16-group normalisation over the concatenated
  tensor (TensorRT 10.3 mis-compiles the pair as written -- the fp32 engine differed from
  the torch model by 6.9 in max abs logits). Both are the same arithmetic spelled for the
  tracer. The model, the training path and eager inference are untouched -- nothing but
  the export sees either swap, and the v1 decoder is not rewritten at all.
- `--fp16` is opt-in and off by default, as in the demo.
- `--verify` runs four seeded random tensors through both the torch model and the engine,
  plus the four jpg/png frames in `--verify-dir` (preprocessed exactly as the demo
  preprocesses a camera frame) when it is given. It **fails** on any NaN or infinite
  engine output, when fewer than 99.5% of pixels agree on their argmax class, or when the
  maximum absolute logits difference exceeds 0.25 -- the good fp32 engine measures about
  0.06 there and the mis-compiled one 6.9. Pass `--verify-dir` when there are frames to
  hand; without it the report says it verified on random input only. The mask IoU is
  printed for context but does **not** gate: the smallest mask measured is 6 of 7 union
  pixels, where a correct engine scores 0.8571, so a 0.99 IoU floor rejects a correct
  engine. A refusal from the argument or dependency checks -- a missing `--verify-dir`, an
  `--out` that already exists, no ONNX, no TensorRT -- leaves no output directory behind.
  A failure *during* the export or the verify cannot: `--out` then holds what had been
  written by the time it failed -- `model.onnx`, and `model.engine` if the build got that
  far, but no `engine.json` -- so a re-run needs a fresh `--out`, or delete the wreckage
  first. Either way the refusal prints how far off the engine was.
- `engine.json` records the checkpoint's sha256, the input size, the precision,
  the TensorRT version, and the dtype of the engine's input and outputs. That is
  what the loader checks: an engine built from another checkpoint, or at another
  size, is refused rather than served. An fp16 engine is declared fp16, and the
  forward converts the float32 tensor the demo pre-processes to fp16 before
  handing it over.

Time it, then run the demo on it:

```sh
PYTHONPATH=<checkout> python -m jetson.bench_model --checkpoint /home/user/models \
    --engine /home/user/models/trt
ECS_HCM_DEMO_ART_DIR=/home/user/models \
    <checkout>/jetson/run_demo.sh --engine /home/user/models/trt
```

- The bench with `--engine` needs `--checkpoint` (that is what the engine is
  validated against) and reports `backend engine`. Without `--engine` it times
  the torch forward exactly as before.
- **Rollback is dropping `--engine`:** both tools fall back to the torch forward,
  and nothing about the torch path depends on an engine existing.

## 5. CI

```sh
~/ecs_hcm-rig/tools/k8s_pytest.sh <git-ref> [image-tag] [-- pytest args]
```

- The full suite runs only this way, as a CPU-only Job in the `ci` namespace,
  never on the VM. On the VM, run only the test files a change touches.
- `<git-ref>` must exist in `~/ecs_hcm_model` (or `$REPO`).
- **What the Job ships:** `ecs_hcm_model jetson video tests main.py pytest.ini`,
  plus `dockerfiles` and `tools` for SHAs that have them. Tests that read a new
  top-level path fail in CI until it is added to the `git archive` line in
  `k8s_pytest.sh`.
- The archive must stay under 900 KB (it ships in a ConfigMap), so large
  fixtures do not belong in the repo.
- CI has no `.git`, no network, and no model downloads. Tests must not depend on
  any of them.

## 6. Scoring a checkpoint by object size (`tools/eval_seg.py --by-size`)

The question "would 448 or 512 help small or distant objects?" cannot be read off
`val_iou`, which reports a single number for the whole set: a gain on small
objects and an equal loss on large ones cancel. `--by-size` adds the breakdown.

```sh
# one size
.venv/bin/python tools/eval_seg.py --dataset coco --data-root <coco val root> \
    --num-workers 0 --by-size <run>/best.pth

# the comparison the owner asked for: the same checkpoint at three sizes
for s in 352 448 512; do
  .venv/bin/python tools/eval_seg.py --dataset coco --data-root <coco val root> \
      --input-size $s --num-workers 0 --by-size <run>/best.pth
done
```

`<coco val root>` is a directory laid out by `tools/fetch_seg_val.py coco
--data-root <dir>`, the same val set `promote_model.py accept` scores against.

- **The buckets.** An instance is `small` under 32² = 1024 pixels, `medium` under
  96² = 9216, `large` at or above it, by its pixel count in the **original** image
  -- before the resize -- so an object the resize enlarges is still bucketed as
  the small object it is. Where instances overlap, the pixel is charged to the
  smallest one covering it. The rule travels with the number: the `RESULT` line
  and the JSON both carry `rule=origpx:small<32^2,medium<96^2,smallest-wins`.
- **Recall, pooled.** A bucket's number is the fraction of that bucket's
  ground-truth pixels the model calls foreground, pooled over every image scored.
  Note this pools pixels, while `val_iou` on the same line is the mean of
  per-image ratios: the two aggregate the same prediction differently, so do not
  try to reconstruct one from the other. Each bucket reports its ground-truth
  pixel and instance counts beside the recall -- a recall over 40 pixels is
  visible as such.
- **The score does not move.** `iou`, `dice` and `loss` are bit-for-bit the same
  number with the flag off and on: the breakdown reads the metric's own tensors
  in the same pass and runs no second forward.
- **What it needs.** `--dataset coco` (COCOA's regions have no COCO area) and
  `--num-workers 0` (the bucket map is built in the scoring process); both are
  refused up front, before any image is read. Keep the batch size fixed across a
  comparison -- `drop_last` is on, so it decides how many images are scored.
- **When it refuses to report.** If the instances it bucketed do not add up to
  the target the metric scored, that checkpoint fails with an error naming the
  image and prints no score. That is the guard against a breakdown measured
  against a different ground truth. It has not run against real `val2017` yet, so
  if a first run does fire it, suspect the decomposition (crowd and RLE instances),
  not the data.

## 7. Scoring a public baseline beside ours (`--baseline`)

Our IoU means more next to someone else's. `--baseline NAME` scores a public
model through the identical path -- the trainer's validation transform, its
collate, the same batch size, the same `evaluate()`, the same `--by-size`
breakdown -- so the two numbers are the same measure of two different models.

```sh
# ours
.venv/bin/python tools/eval_seg.py --dataset coco --data-root <coco val root> <run>/best.pth
# theirs, same val set, same metric
.venv/bin/python tools/eval_seg.py --dataset coco --data-root <coco val root> \
    --num-workers 0 --by-size --baseline yolo11n-seg
```

Known names: `yolo11n-seg` (ultralytics, AGPL-3.0) and `torchvision-lraspp`
(torchvision, LRASPP-MobileNetV3-Large -- the small model, 3.2M params, for a
latency table beside ours). Each needs its package installed (`pip install
ultralytics`; torchvision is already a training dependency, but its weights
download on first use); the import happens when the baseline is used, so this
tool still runs where none is installed, and a missing package is an error
naming the install command. The packages are called, never vendored.

- **The size is the adapter's unless you ask for another one.** With no
  `--input-size`, `yolo11n-seg` runs at 640 and `torchvision-lraspp` at 520 --
  the sizes their packages serve. `--input-size 352` runs the same model at ours
  on purpose: that is the second row of the table, so it is allowed and it is
  reported (`input_size=352 (--input-size 352 (adapter yolo11n-seg's native size
  is 640))`). What is never allowed is a row that reads as one size and ran at
  another, which is why the adapter is handed the resolved size and reports it
  rather than resizing to its own.
- **The geometry is ours.** The transform resizes the short side by 4/3 and
  centre-crops to a square, always. So a baseline runs at our crop at its size,
  which is not the same as its published recipe (the torchvision weights resize
  the short side to 520 and do not crop): the price of one shared transform, and
  the reason each row reports the size it ran at.
- **Read the caveat with the number.** `torchvision-lraspp` is BSD code but its
  weights are trained on COCO with VOC's 21 labels (background + 20 classes, not
  COCO's 80 instance classes), and its foreground is the argmax over those 21
  being non-background -- the model's own decision, not ours; that row carries
  `caveat=voc21-argmax` on the `RESULT` line, with the sentence in the JSON
  (`caveat_note`) and on stderr. `yolo11n-seg` is COCO instance segmentation, so
  it has nothing to declare.
- **`loss` is not a comparable number for a baseline.** It is the trainer's own
  training objective (pos-weighted CE + Dice) evaluated on the baseline's
  probability. The comparable numbers are `iou` and `dice`.
- **Latency is a separate run, and a different measurement.** On the box:

  ```sh
  PYTHONPATH=<repo>:<deps dir> python -m jetson.bench_model --baseline yolo11n-seg
  ```

  It times the adapter's forward on a prepared tensor, fp32, batch 1, at the
  declared size, with the same warmup/iterations/percentiles as an hcm model's
  run. The report's `timed` line says what the number covers -- for
  `yolo11n-seg` the package's own preprocessing and NMS are inside it, which
  `demo_forward` never includes. Keep the package in its own `PYTHONPATH`
  directory; do not install it into the demo venv.
- **Not run yet.** This has been exercised with a known-answer adapter and with
  no real weights on this VM. On the first real run, confirm the adapter's input
  convention by scoring one image the package also processes itself (`YOLO(...)
  .predict(<same image>)`): if the mask area is wildly different -- near zero or
  near full -- the tensor convention is wrong, not the model.
