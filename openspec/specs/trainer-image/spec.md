# trainer-image Specification

## Purpose

The contract between the image that runs training and the cluster that launches it: which CUDA and torch libraries are used at runtime, what the container starts, how the CLI selects a task and its scale, and where secrets come from. Getting any of this wrong produces a run that looks like a numerics problem, so it is spec'd as a contract rather than left as Dockerfile detail.

## Requirements

### Requirement: The image ships a slim CUDA runtime base with a pinned torch toolchain

The training image MUST be built in two stages: the build stage uses a CUDA development base to compile or install the toolchain, and the final stage uses the CUDA runtime base rather than the development base, so the shipped image does not carry the compiler layer. The final image MUST pin the torch and torchvision versions and install them from the CUDA 12.1 wheel index matching the CUDA 12.1 base, so the wheel's CUDA major version matches the driver-facing runtime.

#### Scenario: The runtime stage does not inherit the development base

- **WHEN** the final image's base is examined
- **THEN** it is the CUDA runtime base, not the development base used by the build stage

#### Scenario: torch versions are pinned rather than floating

- **WHEN** the image is rebuilt at a later date
- **THEN** the same torch and torchvision versions are installed as before

### Requirement: torch's bundled CUDA libraries take precedence on the library path

The runtime image MUST place torch's own CUDA library directory ahead of the CUDA toolkit directory on `LD_LIBRARY_PATH`, so any CUDA library torch vendors in the wheel is resolved from the wheel rather than from the system CUDA, because a GPU library leak across the two silently produces a numeric failure instead of a startup error. A path built from an unset variable would produce a relative or empty component and so MUST NOT be allowed to shadow the intended directories.

#### Scenario: The torch library directory is first

- **WHEN** the runtime library search path is inspected inside a container built from this image
- **THEN** torch's CUDA library directory appears before the CUDA toolkit directory

#### Scenario: The load-bearing library is actually present

- **WHEN** the image is built
- **THEN** the build fails if torch's vendor library directory does not contain the CUDA runtime library it is expected to provide

### Requirement: The image's git commit is a build argument, not a build-time clone

The image MUST accept the commit being built as a build argument, MUST default it to a known placeholder when the build does not supply one, and MUST expose it to the process inside the image as an environment variable so a checkpoint written by the container can record the commit that produced it without the container reading git.

#### Scenario: The supplied commit reaches the process

- **WHEN** the image is built with a commit build argument
- **THEN** a process inside the container sees that value in its environment

#### Scenario: A build with no commit argument still succeeds

- **WHEN** the image is built without the commit build argument
- **THEN** the build succeeds and the value is the placeholder

#### Scenario: No git checkout is required at runtime

- **WHEN** the container runs
- **THEN** the recorded commit comes from the environment rather than from a git command

### Requirement: The container runs as a non-root user with the training directory on the import path

The image MUST create and switch to a non-root user before declaring its entrypoint, and MUST place the training code in the working directory with that directory on the Python import path, so the entrypoint resolves the package without an install step. The workspace itself cannot have drifted from the code that was copied in, so the code that runs is the code that was built.

#### Scenario: The container does not run as root

- **WHEN** the process runs inside the image
- **THEN** its user is the non-root training user

#### Scenario: The training package imports from the working directory

- **WHEN** the entrypoint starts
- **THEN** the package is importable from the container's working directory

### Requirement: Secrets are read from the environment and never baked into the image

The image and the code it carries MUST read object-store credentials and any other secret from environment variables at run time and MUST NOT contain credentials, keys or tokens in any layer. A missing credential MUST raise at the point of use rather than fall back to a default, so a misconfigured launch fails immediately instead of writing to the wrong place.

#### Scenario: Credentials come from the environment

- **WHEN** the uploader is constructed inside the container
- **THEN** it reads the user and password from the environment variables the launcher supplies

#### Scenario: A missing credential raises

- **WHEN** the credential variables are absent
- **THEN** construction raises rather than proceeding with a default

### Requirement: The entrypoint exposes one training command with the task's scale as flags

The image's entrypoint MUST dispatch on a single training command and MUST accept the flags that set the training configuration: whether to start from a new model, the tracking environment, finetune mode, COCO mode, the model base to load from and upload to, the epoch budget, the batch size, the model input size, test mode, whether warmup runs, a per-epoch batch cap, and the COCO mix ratio for a refine. The defaults MUST be the runnable development configuration: development environment, the models base bucket, a five-epoch budget, batch size eight, an input size of 192 and a mix ratio of zero. Warmup MUST be off unless the warmup flag is passed.

The input size MUST be the single control for the spatial size of the COCO/COCOA segmentation pipeline, and it MUST reach the training command's own transform configuration rather than being settable only inside the library.

The mix ratio MUST reach the training command's own refine configuration, MUST default to zero so that omitting it is the existing run, and MUST be refused when it is negative, when it is not a finite number, or when it is set on a run that is not a finetune.

#### Scenario: The upstream launcher's flags keep working

- **WHEN** the training command is invoked with the flags the cluster launcher supplies
- **THEN** each flag selects the configuration it names

#### Scenario: A bare invocation is runnable

- **WHEN** the training command is invoked with no flags
- **THEN** it runs the development configuration with a five-epoch budget, batch size eight and an input size of 192

#### Scenario: The model base is restricted to the known buckets

- **WHEN** an unrecognised model base is passed
- **THEN** the argument parser rejects it rather than allowing an arbitrary bucket

#### Scenario: A caller reaches the trainer through the package, not only the command line

- **WHEN** a caller wants to start a run without the CLI
- **THEN** there is a public entry point in the training package that the command line delegates to

#### Scenario: The input size is settable from the command line

- **WHEN** the training command is invoked with an input size
- **THEN** that size is what the COCO/COCOA segmentation transforms build tensors at

#### Scenario: The input size defaults without being given

- **WHEN** the training command is invoked without an input size
- **THEN** the size is 192

#### Scenario: The mix ratio is settable from the command line

- **WHEN** the training command is invoked for a finetune with a mix ratio
- **THEN** each training epoch interleaves COCO batches in that proportion and validation scores both datasets

#### Scenario: The mix ratio defaults to the existing run

- **WHEN** the training command is invoked without a mix ratio
- **THEN** the ratio is zero and the run trains on COCOA alone

#### Scenario: A mix ratio that cannot mean anything is refused

- **WHEN** the training command is given a negative, NaN or infinite mix ratio, or one above zero without a finetune
- **THEN** the run fails with an error naming the ratio and the mode it requires

### Requirement: No image layer carries more than about 3 GB

The trainer image MUST NOT ship its Python environment as a single layer large enough to outlast the registry push timeout. The build stage MUST split the environment into a fixed number of parts, each holding at most 2.5 GB of files, and the runtime stage MUST copy every part onto the same environment path so the installed tree is identical to an unsplit copy, including each directory's permission mode. If a single file or the whole environment does not fit the parts allowed, the build MUST fail rather than produce a larger layer. The torch library path ordering and the bundled `libcublas` check MUST run after the environment is reassembled.

#### Scenario: The environment layers stay under the cap

- **WHEN** the built image's layer history is inspected
- **THEN** no layer is larger than about 3 GB

#### Scenario: The reassembled environment matches the original

- **WHEN** the parts are copied onto the environment path in order
- **THEN** every file, symlink and empty directory of the original environment is present with the same content

#### Scenario: An environment that outgrows the parts fails the build

- **WHEN** the environment, or one file in it, exceeds what the configured parts can hold
- **THEN** the split step exits non-zero and the image is not built

#### Scenario: Directory modes survive the split

- **WHEN** a directory in the environment has a mode other than the build default, such as 0700
- **THEN** after the parts are copied onto the environment path that directory still has the same mode

### Requirement: The venv split never moves files across layers

The builder stage SHALL install every package into `/opt/venv` and split it into its
parts within a single RUN instruction, so that no file the split renames was created by
an earlier layer. No other builder RUN may install into `/opt/venv`. pip's download cache
MUST be a BuildKit cache mount, not part of any layer. A failed install SHALL fail the build:
no tolerated failure in that RUN (`|| true`) may let the cleanup or the split run after a
failed `pip install`, so every tolerated command MUST be grouped as `{ cmd || true; }`.

#### Scenario: One RUN installs and splits
- **WHEN** the builder stage's RUN instructions are read
- **THEN** exactly one of them runs `split_venv.py` on `/opt/venv`
- **AND** every venv `pip install`, torch's included, is in that same RUN, before the split

#### Scenario: The pip cache is a mount
- **WHEN** that RUN is read
- **THEN** it mounts a cache at `/root/.cache/pip`

#### Scenario: A failed install fails the build
- **WHEN** either `pip install` in that RUN fails
- **THEN** the RUN exits non-zero and `split_venv.py` never runs

### Requirement: The image carries the source trees its evaluation entry points need

The image SHALL carry every module its documented evaluation entry points (`tools/eval_seg.py`,
`tools/fetch_seg_val.py`) run from, at the path their launcher resolves them at: the directory the
entry point lives in and the packages the modules it imports are in. The build context MUST NOT drop
any of them, so a `COPY` that names a tree and an ignore rule that removes it cannot both be true of
the same tree. A module the image carries but no entry point reaches MAY import packages the image
does not install, because the image is not expected to run it. From an entry point and everything
reachable from it, every import MUST name a package the image installs, except that a package the
image deliberately does not ship MUST be imported only from inside a function body and declared
where the module names what to install without it -- so reaching that import reports the install
rather than failing the entry point.

#### Scenario: A documented entry point is in the image

- **WHEN** the built image is inspected for the documented evaluation entry points
- **THEN** each one is present at the path its launcher uses

#### Scenario: A tree an entry point imports is in the image

- **WHEN** the entry points' imports are followed inside the image
- **THEN** every module of this repository they reach is present in the image

#### Scenario: The build context does not undo a COPY

- **WHEN** the final stage's copy instructions and the build context's ignore file are read together
- **THEN** no tree carried for an entry point is removed by an ignore rule

#### Scenario: A package an entry point reaches is not in the image

- **WHEN** the imports reachable from the documented evaluation entry points are collected and
  compared with the distributions the image installs
- **THEN** each one either names a package the image installs or is imported only inside a function
  body and declared by the module that imports it, and an import at module scope of a package the
  image does not install is a failure

### Requirement: The build context carries no bytecode or weights from the trees the image ships

The build context MUST NOT carry Python bytecode or model-weight files from any depth of the
trees the image ships. `__pycache__` directories and `*.py[cod]` files, `*.so` objects,
`build/` and `lib/` directories, and `*.pth`, `*.onnx` or `*.torchscript` files anywhere
under those trees MUST be dropped by the ignore file, not only at the context root. Docker
anchors every ignore pattern at the context root, so a rule that reaches a shipped tree's
subdirectories has to ask for the depth.

#### Scenario: A nested cache directory is not carried

- **WHEN** the final stage's ignore file meets a tree that has run Python, so a
  `__pycache__/` directory holding `.pyc` files sits below the top level of a directory the
  image ships
- **THEN** those files are absent from the image

#### Scenario: Nested weight files are not carried

- **WHEN** the ignore file meets a `.pth`, `.onnx` or `.torchscript` file below the top level
  of a shipped tree
- **THEN** it is absent from the image

#### Scenario: The trees an entry point needs are still carried

- **WHEN** the imports of the documented evaluation entry points are followed
- **THEN** every module of this repository they reach is still present in the image

### Requirement: The image pins timm and never bakes backbone weights

The trainer image SHALL install `timm` at an exact pinned version, so the ImageNet backbone
entries can be built inside a training container. No image layer MUST carry backbone
weights: the build MUST NOT download or vendor an ImageNet checkpoint, and no module MUST
fetch weights at import time. When a run asks for pretrained weights, they SHALL be
resolved from the environment's cache or a path the caller passes — the PVC-backed cache, or
a mirrored file — and never from a credential written in code.

#### Scenario: The image pins timm

- **WHEN** the Dockerfile is inspected
- **THEN** it installs `timm` at an exact `==` version, and no line downloads a weights
  file or copies one into the image
- **AND** the check is a text assertion, so it holds in CI before the image is rebuilt

#### Scenario: Weights are provisioned, not fetched at import

- **WHEN** the backbone module is imported
- **THEN** no network request is made and no weight file is read; a build with pretrained
  weights disabled downloads nothing, and one with weights enabled takes them from the
  cache or the path the caller passes
