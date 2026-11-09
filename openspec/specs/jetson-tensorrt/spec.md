# jetson-tensorrt Specification

## Purpose
Run the Jetson demo's forward pass on a TensorRT engine built from a promoted checkpoint, for speed, while keeping the PyTorch pre/post-processing. An engine is served only beside a record that ties it to the checkpoint it was built from and the input size it expects; anything missing or mismatched is refused rather than run.

## Requirements

### Requirement: A demo checkpoint is exported to ONNX and a TensorRT engine

The repository SHALL provide `jetson/export_trt.py`, which loads a checkpoint directory
exactly as `jetson/demo_model.py` does (the same metadata-driven decoder version, stride and
input size) and writes `model.onnx`, `model.engine` and `engine.json` into an output
directory. The ONNX graph MUST be exported with the legacy exporter at a fixed batch of 1
and the checkpoint's input size, at opset 17 or above, and the engine MUST be built from it
with the TensorRT Python builder API. The build MUST accept an optional fp16 flag and a
workspace memory limit, and MUST default to fp32.

The export MUST refuse `--fp16` for this checkpoint's graph, naming the reason — an fp16 engine
of this model returns non-finite logits while its fp32 engine is finite — and MUST refuse it
before it loads the checkpoint, before it asks the builder for an engine, and before anything
under `--out` is written. The refusal stays until the non-finite output is diagnosed; the builder
function itself keeps accepting the flag.

The export MUST check that the ONNX package and the TensorRT Python bindings are both
importable before it creates, reads or writes anything under `--out`, and MUST refuse,
naming the missing one, when either is absent. The `--out` directory MUST NOT exist after
a refusal, including a refusal that happens once the export has begun to write — a failed
`--verify`, a builder failure, or the process dying mid-export. Nothing MUST be written into
`--out` itself until every step has succeeded.

The directory the export creates for `--out` MUST have the mode a plain `mkdir` would have
given it under the process umask (`0o777` minus the umask, as on the live `~/models/trt`),
not the owner-only mode of the staging directory the export builds in.

#### Scenario: A checkpoint is exported

- **WHEN** the tool is run on a checkpoint directory with an output directory that does not exist
- **THEN** `model.onnx`, `model.engine` and `engine.json` are written

#### Scenario: The engine defaults to fp32

- **WHEN** the tool is run without `--fp16`
- **THEN** the built engine is fp32 and `engine.json` records the precision as fp32

#### Scenario: fp16 is an explicit opt-in

- **WHEN** the tool is run without `--fp16`
- **THEN** the engine is built in fp32, and no fp16 engine is built implicitly — for this graph the flag has to be asked for, and asking for it is refused (see "A request for fp16 is refused")

#### Scenario: A request for fp16 is refused

- **WHEN** the tool is run with `--fp16`
- **THEN** it stops with a message naming the reason (an fp16 engine of this model has non-finite logits), the builder is never asked for an engine, and no `--out` directory or file under it is created

#### Scenario: TensorRT is absent

- **WHEN** the TensorRT Python bindings cannot be imported
- **THEN** the tool stops with a message naming TensorRT and writes nothing

#### Scenario: The ONNX package is absent

- **WHEN** the ONNX package cannot be imported
- **THEN** the tool stops with a message naming ONNX, and no `--out` directory or file under it is created

#### Scenario: A dependency refusal leaves no output directory

- **WHEN** the export refuses for a missing dependency and `--out` did not exist
- **THEN** it still does not, so a retry after installing the dependency is not refused for overwriting a half-written directory

#### Scenario: A dependency refusal precedes the overwrite check

- **WHEN** the export is given a missing dependency and an `--out` that already exists
- **THEN** it refuses for the missing dependency, and the existing directory is left untouched

#### Scenario: A refusal after the export has begun leaves no output directory

- **WHEN** the export refuses after it has exported the ONNX graph and built the engine — a `--verify` whose verdict fails, or one that cannot run on the device it was given — and `--out` did not exist
- **THEN** the tool exits non-zero, `--out` still does not exist, and the directory it did not create holds no leftover of the attempt, so a retry is not refused for overwriting a half-written directory

#### Scenario: The output directory has the mode a plain mkdir would give it

- **WHEN** the export creates an `--out` that did not exist, having built the files in a staging directory of its own
- **THEN** the created directory's mode is `0o777` minus the process umask, the same mode `mkdir` would have produced, and not the owner-only mode of the staging directory it was built in

### Requirement: The export records provenance and geometry in engine.json

`engine.json` SHALL record the sha256 of the source checkpoint, the input size, the
precision, the TensorRT version, the dtype of the engine's input tensor, and the engine's
output names, shapes and dtypes. The record MUST be readable without TensorRT installed.

#### Scenario: The record names the source

- **WHEN** a checkpoint is exported
- **THEN** `engine.json` carries the sha256 of that checkpoint's weights file

#### Scenario: The record names the geometry

- **WHEN** a checkpoint trained at 352 is exported
- **THEN** `engine.json` records an input size of 352 and the output shape the engine produces

#### Scenario: The record names the tensor dtypes

- **WHEN** a checkpoint is exported (the engine is fp32, because an fp16 one is refused for this graph)
- **THEN** `engine.json` records the input tensor's dtype as float32, alongside each output's dtype

### Requirement: The export refuses to overwrite an existing output directory

The export MUST stop rather than write into an `--out` that already exists.

#### Scenario: The output directory exists

- **WHEN** `--out` names a path that exists
- **THEN** the tool stops with an error and does not modify it

### Requirement: The export can verify the engine against the torch model

The export SHALL offer a `--verify` mode that runs both the torch model and the built engine
over the same inputs and reports the maximum absolute difference of their logits, the
fraction of pixels whose argmax class agrees, the foreground-mask IoU and the count of
non-finite outputs. It MUST fail when any output is not finite, when fewer than 99.5% of
pixels agree on their class, or when the maximum absolute logit difference exceeds 0.25. The
mask IoU MUST be reported and MUST NOT gate, because a correct engine scores 0.8571 on the
smallest mask this model produces (6 of 7 union pixels). The inputs MUST be seeded random
tensors and, when
`--verify-dir DIR` is given, also the jpg/png images in that directory preprocessed by the
demo's own preprocessing; without the flag the report MUST say that it verified on random
input only, and the report MUST distinguish the random from the image inputs.

#### Scenario: A faithful engine passes

- **WHEN** `--verify` compares an engine built from the checkpoint to the torch model
- **THEN** it reports the logits difference, the pixel agreement, the mask IoU and the non-finite count, and exits zero when the pixel agreement and the logits difference are within their limits

#### Scenario: A mask that disagrees fails loudly

- **WHEN** fewer than 99.5% of pixels agree on their argmax class between the engine and the torch model
- **THEN** the tool reports the agreement it measured and exits non-zero

#### Scenario: A mask that is slightly off does not fail

- **WHEN** the mask IoU between the engine and the torch model is below 0.99 but the argmax class agrees on at least 99.5% of pixels and the logits are within 0.25
- **THEN** the verification passes and reports the IoU with the reference it is compared against

#### Scenario: Logits that are wrong everywhere fail

- **WHEN** the engine's logits differ from the torch model's by more than 0.25 on any input
- **THEN** the tool reports the difference and exits non-zero, even when every pixel's argmax class agrees

#### Scenario: A non-finite output fails

- **WHEN** any engine output holds a NaN or an infinity
- **THEN** the tool reports how many values were non-finite and exits non-zero, whatever the remaining comparisons say

#### Scenario: Real images are verified when a directory is given

- **WHEN** `--verify-dir DIR` is given and DIR holds jpg/png images
- **THEN** those images, preprocessed as the demo preprocesses a camera frame, are verified alongside the seeded random inputs, and the report names how many of each it used

#### Scenario: Verifying without a directory is reported as such

- **WHEN** `--verify` runs without `--verify-dir`
- **THEN** the report notes that it verified on seeded random input only

#### Scenario: A directory that cannot be used is refused before anything is written

- **WHEN** `--verify-dir` names something that is not a directory, or a directory with no images
- **THEN** the tool refuses before the output directory is created

### Requirement: An engine loader refuses an engine that does not match what it is asked to serve

The engine loader SHALL read `engine.json` and refuse, with a message naming the field and
both values, an engine whose recorded input size or source-checkpoint sha256 does not match
the input size and checkpoint the caller states it is serving. All of the loader's
validation MUST work with TensorRT absent.

#### Scenario: The input size disagrees

- **WHEN** the loader is asked to serve an engine recorded at 192 with input size 352
- **THEN** it refuses and names the recorded and the requested size

#### Scenario: The checkpoint disagrees

- **WHEN** the loader is given an engine whose recorded sha256 is not the checkpoint's
- **THEN** it refuses and names the sha256 it expected

#### Scenario: The record is unreadable

- **WHEN** `engine.json` is missing, unreadable, not JSON, or not a JSON object
- **THEN** the loader refuses rather than falling back to the engine file

### Requirement: The demos run an engine when one is given and are unchanged without one

`realtime_demo.py` and `bench_model.py` SHALL each accept `--engine <dir>`. With it, the
forward pass MUST run through the engine using the same pre-processing and post-processing
those tools already use, and the bench MUST report which backend produced the timings.
Without `--engine`, both tools MUST behave exactly as they do today. Both MUST refuse an
engine that does not match the input size or the checkpoint they are about to serve.

The engine forward MUST feed the engine buffers of the dtypes the engine declares: when
the tensor a caller passes has another dtype, it MUST be converted to the engine's before
its pointer is handed over, so an fp16 engine is never fed a float32 buffer.

#### Scenario: The demo runs the engine

- **WHEN** `realtime_demo.py --engine <dir>` starts
- **THEN** the model forward is the engine's and the overlay is produced from its outputs

#### Scenario: The bench names the backend

- **WHEN** `bench_model.py --engine <dir>` runs
- **THEN** the report says the timings are the engine's

#### Scenario: Without the flag nothing changes

- **WHEN** either tool is run without `--engine`
- **THEN** it loads and runs the torch model exactly as before

#### Scenario: A mismatched engine is refused

- **WHEN** either tool is given an engine whose input size or source sha256 disagrees with what it is about to serve
- **THEN** it stops with the loader's refusal message and does not run the engine

#### Scenario: An fp16 engine is fed fp16

- **WHEN** the engine declares a float16 input tensor and the caller passes a float32 tensor
- **THEN** the tensor is converted to float16 before the engine reads it

### Requirement: Adaptive pooling is exported as a static equivalent, for the export only

The ONNX export MUST replace `torch.nn.functional.adaptive_avg_pool2d` with a
static-shape equivalent for the duration of the export call only, and MUST restore the
original function before returning -- including when the export raises. No caller other
than the export may observe the replacement: training and eager inference MUST behave
exactly as they do without this change.

The equivalent MUST reproduce PyTorch's adaptive bin rule -- for output index `i` of
`out` over an input extent `n`, the bin spans `start = floor(i*n/out)` to
`end = ceil((i+1)*n/out)`, and the output is the mean over that span -- so that
non-divisible extents (where bins are uneven and may overlap) match the operator it
replaces. It MUST accept either an int or a `(height, width)` output size, and the
result MUST take the input's dtype and device.

#### Scenario: A model whose shape is lost after GroupNorm is exported

- **WHEN** a model reaches `F.adaptive_avg_pool2d` through a path whose spatial extent
  the legacy tracer cannot see (GroupNorm, then an addition of an interpolated tensor)
- **THEN** the export writes the ONNX graph instead of failing with `SymbolicValueError`,
  and the graph passes `onnx.checker`

#### Scenario: The replacement is confined to the export call

- **WHEN** an export runs, and the `torch.onnx.export` call is observed from inside
- **THEN** `torch.nn.functional.adaptive_avg_pool2d` is the static equivalent for the
  duration of that call, and is the original function again before and after it

#### Scenario: The original operator is restored when the export raises

- **WHEN** the code under the replacement raises
- **THEN** `torch.nn.functional.adaptive_avg_pool2d` is the original function again

#### Scenario: The static pool agrees with the operator it replaces

- **WHEN** the static equivalent is applied to the same input and output size as
  `F.adaptive_avg_pool2d`, for a divisible extent, a non-divisible extent, and an extent
  larger than the output, given as an int and as a `(height, width)` tuple, with a batch
  greater than one
- **THEN** the two results agree to within floating-point summation order

### Requirement: The v2 decoder's GroupNorm pair is fused for the export only

The export SHALL rewrite the v2 decoder's two `group_norm(..., num_groups=8, eps=1e-5)`
calls in `hcm_rrd.forward` as a single `group_norm(..., num_groups=16, eps=1e-5)` over the
channel-concatenated tensor, for the duration of the export call and no longer, because
TensorRT 10.3 mis-compiles the pair as written when it consumes the concat buffer. The
rewrite MUST be exactly the same function: sixteen groups of four channels over the
64-channel concat are the same groups as eight and eight over the two 32-channel halves, and
neither call has affine parameters. The v1 decoder MUST be left alone -- it concatenates on
the width, where the joint spelling would change the normalisation statistics -- and the
original forward MUST be restored before the export call returns, including when it raises.

#### Scenario: The fused forward computes the same function as the original

- **WHEN** the v2 decoder's fused forward and its original forward are run on the same inputs
- **THEN** their outputs are equal bit for bit, at a 2x upsample, at equal sizes and at a non-square pair

#### Scenario: The rewrite is confined to the export call

- **WHEN** a model is exported
- **THEN** `hcm_rrd.forward` is the fused call for the duration of `torch.onnx.export` and the original afterwards

#### Scenario: The original forward is restored when the export raises

- **WHEN** the export call raises
- **THEN** `hcm_rrd.forward` is the original forward again

#### Scenario: The v1 decoder is unaffected

- **WHEN** the swap is installed and a v1 decoder's forward runs
- **THEN** it delegates to the original and its output is unchanged, because a joint normalisation over v1's width-concat is a different function from its two separate normalisations

### Requirement: An engine is built with TF32 disabled, and its record says so

TensorRT is free to select TF32 tactics for an fp32 network unless the builder config refuses
them, and a TF32 tactic computes fp32 operands with a 10-bit mantissa — so without an explicit
refusal, an engine the record calls "fp32" is not fp32. `jetson/export_trt.py` MUST clear
`BuilderFlag.TF32` on the builder config before it builds the network, so that a build recorded
as fp32 is fp32. The clear MUST be unconditional, including when the fp16 flag is set, because
TF32 governs the fp32 arithmetic left in any build and clearing it there keeps a single build
path and a single recorded truth. `engine.json` MUST record that TF32 was disabled, so an
engine's record says how its arithmetic was built. That field is descriptive and MUST NOT be
one of `trt_engine.REQUIRED_KEYS`, so a record written before the field existed still loads.

#### Scenario: The builder clears TF32 before it builds

- **WHEN** an engine is built
- **THEN** `clear_flag(BuilderFlag.TF32)` is called on the builder config, and it is called before `build_serialized_network` is handed that config

#### Scenario: TF32 is cleared for an fp16 build too

- **WHEN** `build_engine` is called with `fp16=True`
- **THEN** the config is given both `clear_flag(BuilderFlag.TF32)` and `set_flag(BuilderFlag.FP16)`, with the clear before the build

#### Scenario: The record says TF32 was disabled

- **WHEN** a checkpoint is exported
- **THEN** `engine.json` records `tf32: false`

#### Scenario: A record written before the field existed still loads

- **WHEN** the loader reads an `engine.json` that carries no `tf32` key
- **THEN** it reads the record back, because `tf32` is not one of the keys a loader requires

### Requirement: The verification's torch reference is computed with TF32 disabled

An engine this export builds refuses TF32 tactics, but the torch model `--verify` compares it
against does not: `torch.backends.cudnn.allow_tf32` is true by default, so the reference's
convolutions run with a 10-bit mantissa and the gate charges the engine for a difference the
reference contributed. How far over the gate a correct engine lands then depends on the weights
rather than on the engine, so the same export passes one checkpoint and refuses another.
`jetson/export_trt.py` MUST disable TF32 for the torch reference around every reference forward
`--verify` computes, so that the gate compares two fp32 computations. It MUST restore each flag
to the value it held before the call, including when the reference forward raises, so a caller
that set either flag keeps its own process. It MUST NOT make the reference fp32 by setting
`NVIDIA_TF32_OVERRIDE` in the process environment: TensorRT records that variable in the engine
it builds and refuses to run an engine whose build-time and run-time values differ, so an engine
built under it can only be served by a process that sets it too.

#### Scenario: The reference runs with TF32 refused

- **WHEN** `verify` computes a reference forward
- **THEN** `torch.backends.cudnn.allow_tf32` and `torch.backends.cuda.matmul.allow_tf32` are both false at that moment

#### Scenario: The flags are restored after the comparison

- **WHEN** `verify` returns
- **THEN** both flags hold the values they held before the call

#### Scenario: The flags are restored when the reference forward raises

- **WHEN** the reference forward raises
- **THEN** both flags hold the values they held before the call, and the error propagates

#### Scenario: A caller's own flag values survive the comparison

- **WHEN** a caller has set either flag to a value that is not its default before calling `verify`
- **THEN** each flag holds that caller's value again after `verify` returns

#### Scenario: The reference is made fp32 without the process environment

- **WHEN** `verify` computes a reference forward
- **THEN** `NVIDIA_TF32_OVERRIDE` in the process environment is unchanged from what it was before the call
