# model-decoder Specification

## Purpose
Define the segmentation decoder's geometry and its versions, so that new checkpoints get a decoder that keeps each pixel's position and fine detail, while every existing checkpoint still loads and runs exactly as it was trained.

## Requirements

### Requirement: The decoder architecture is versioned

hcm_nn, hcm_rrd and hcm_csc SHALL take a `decoder_arch` of `v1` or `v2`, defaulting to `v1`, and SHALL refuse any other value. Under `v1` the decoder MUST compute exactly what it computed before versioning, so legacy weights load strictly and produce bit-identical outputs. `v2` MUST add only the hcm_rrd fuse convolutions to the parameter set.

#### Scenario: A legacy checkpoint runs unchanged
- **WHEN** a state dict from the pre-versioning hcm_nn is loaded strictly into a `v1` hcm_nn
- **THEN** the forward pass output is bit-identical to the original model's
- **AND** the original model is a frozen, hash-pinned copy of the pre-versioning source in the test fixtures, so the check needs no git, network or download and never skips

#### Scenario: An unknown version is refused
- **WHEN** a decoder module is built with `decoder_arch` other than `v1` or `v2`
- **THEN** construction raises ValueError

### Requirement: The v2 decoder preserves spatial geometry

Under `v2`, hcm_rrd SHALL concatenate its gated copies along the channel axis and fuse them with a 1x1 convolution, so its output keeps the input width and moving an input column moves the response by the same amount. hcm_csc SHALL add the full-resolution input to its output as a residual. All four ISC outputs SHALL be resized to one working resolution.

#### Scenario: hcm_rrd is shift-equivariant
- **WHEN** a single-column perturbation is applied at several columns of a `v2` hcm_rrd input
- **THEN** the response peaks at a constant offset from each input column

#### Scenario: Fine detail reaches the hcm_csc output
- **WHEN** one pixel of a `v2` hcm_csc input is perturbed
- **THEN** the output response at that pixel dominates the mean response

#### Scenario: The ISC rungs share one size
- **WHEN** a `v2` hcm_nn runs a forward pass
- **THEN** all four ISC outputs have the same spatial size

### Requirement: The backbone is a named registry entry

The decoder SHALL be buildable from a named backbone, resolved through one registry of
frozen specifications. Each specification MUST state the rung order, each rung's channel
count and its stride, and whether the backbone exposes a stride-2 tap. A name outside the
registry MUST raise a `ValueError` naming the name and the registered ones. The `resnet50`
entry MUST build the pre-change backbone class unchanged, so a model built through the
registry with no other setting is bit-identical (same parameter names, same outputs) to one
built before this change, and SHALL remain the default when no backbone is named.

A timm-backed entry MUST be built by asking the library for the registered rungs
(`features_only=True` with the spec's indices), and the rung channels the library reports
MUST be compared against the specification, so a drifted library stops the build instead
of silently changing a shape.

#### Scenario: A registered backbone builds without weights

- **WHEN** each registered entry is built with pretrained weights disabled
- **THEN** the model constructs, and a batch-1 forward pass produces a segmentation output
  whose spatial size is the input divided by the mask stride
- **AND** no network access or weight download is attempted during construction

#### Scenario: An unknown backbone is refused and named

- **WHEN** a backbone name outside the registry is requested
- **THEN** construction raises `ValueError` naming the requested name and listing the
  registered names

#### Scenario: The resnet50 entry is bit-identical to the pre-change model

- **WHEN** a model is built through the registry as `resnet50` and the pre-change
  `hcm_nn` is built with the same `out_stride`, decoder version and class count, sharing
  one copied state dict
- **THEN** their state-dict key sets are equal and their forward outputs are exactly equal

### Requirement: The neck narrows each rung to a recorded width

The decoder SHALL take a `neck_channels`: when it is an integer, each backbone rung MUST be
projected by a 1x1 convolution to that width and normalised by a `GroupNorm` with 32
groups, or -- when 32 does not divide that width -- the largest divisor of it no larger
than 32, so that any positive width builds and no caller needs a separate rule about
multiples of 32. When it is `None`, the decoder MUST build the pre-change
widening (a bare 1x1 convolution to 2048 channels, and an identity for the rung that is
already 2048), so a `neck_channels=None` model is bit-identical to one built before this
change. The two paths MUST be distinct modules, so no state dict loads across them.

The normalisation MUST be per-sample (GroupNorm, not BatchNorm): a batch statistic would
make a batch-1 evaluation differ from a batch-N training step, and would mask an activation
scale that is the reason this change exists.

#### Scenario: A narrowed neck runs end to end

- **WHEN** a model is built with `neck_channels` 128 and run on a batch-1 input
- **THEN** every rung the decoder consumes carries 128 channels and the forward pass
  returns logits of the expected shape

#### Scenario: A width that 32 does not divide still builds

- **WHEN** a model is built with `neck_channels` 33 (32 does not divide it) and run on a
  batch-1 input
- **THEN** each rung's `GroupNorm` carries 33 channels over 11 groups, the forward pass
  returns logits of the expected shape, and a state dict saved from that model reloads
  strictly into a second one built the same way

#### Scenario: No neck means the pre-change model

- **WHEN** a model is built with `neck_channels=None` and the pre-change `hcm_nn` is built
  with the same settings, sharing one copied state dict
- **THEN** their state-dict key sets are equal and their forward outputs are exactly equal

### Requirement: The ISC front and the ASPP input follow the neck width

The 1x1 convolutions inside the ISC that read the incoming rung, and the ASPP's input
projection, SHALL take their input width from what feeds them (the neck width, or the
2048-channel widening when there is no neck), and MUST NOT hardcode 2048. The ISC's inner
widths, its 32-channel output, and everything downstream of the ISC (CSC, RRD, the class
head) MUST keep their existing widths and behaviour under `decoder_arch=v2`.

#### Scenario: The ISC front accepts the narrowed rung

- **WHEN** a model is built with `neck_channels` 128
- **THEN** the ISC rung-reading convolutions and the ASPP accept 128-channel input, the
  ASPP input is 128 channels, and the ISC output is 32 channels

#### Scenario: A neck-less model keeps the 2048 front

- **WHEN** a model is built with `neck_channels=None`
- **THEN** the ISC rung-reading convolutions and the ASPP still take 2048 channels, as they
  did before this change

### Requirement: The mask path can run at half the input stride

The decoder SHALL take a `mask_stride` of 4 or 2 and MUST refuse any other value. At 4 the
mask head SHALL read the stride-4 decoder output, as it does before this change. At 2 the
model MUST upsample that output by 2 with bilinear interpolation, concatenate a 32-channel
1x1 projection of the backbone's stride-2 tap, apply two 3x3 `GroupNorm`+ReLU
convolutions, and pass the result to the existing class head, so the logits are at half the
input resolution (176 at 352). For a backbone whose stem downsamples 4x and therefore has
no stride-2 tap, the concatenated branch MUST instead be a 2x-upsampled projection of the
first rung.

A backbone that exposes no stride-2 tap MUST NOT be an error; the first-rung branch is the
stated behaviour, because that backbone genuinely has no finer signal to offer.

The output grid MUST be derived from the input and the model's own tensors, never from a
hardcoded 4, on every consumer: the training loss, the validation IoU, the evaluation
bucket map and the demo's display path all take the grid from the tensors they are handed.

#### Scenario: Logits land at half resolution

- **WHEN** a model with `mask_stride` 2 runs a 352x352 input
- **THEN** its logits are 176x176, and a model with `mask_stride` 4 on the same input
  produces 88x88

#### Scenario: An unsupported mask stride is refused

- **WHEN** a model is built with a `mask_stride` other than 2 or 4
- **THEN** construction raises `ValueError` naming the value

#### Scenario: A finer mask reaches the validation metric without a stride constant

- **WHEN** a `mask_stride=2` model's logits and a full-resolution ground truth are scored
  through the training path's validation IoU
- **THEN** the ground truth is resized to the prediction's grid and the score is computed
  there, with no 4 in the call path

### Requirement: The ConvNeXt V2 backbones are registered with their licence

The registry MUST carry the ConvNeXt V2 family as named entries — the Nano, the Tiny and
the Base — each built from the timm checkpoint tag the specification names and resolved
through the same registry as every other backbone. The Tiny entry MUST be
`convnextv2_tiny.fcmae_ft_in22k_in1k` and the Base entry MUST be
`convnextv2_base.fcmae_ft_in22k_in1k`, chosen because the timm pinned in the trainer image
accepts them; each MUST use the same rung order and no stride-2 tap as the Nano, and its
rung channels MUST be the ones timm reports for that tag, so a drifted library stops the
build rather than changing a shape silently. Every entry MUST carry a licence note stating
that the upstream weights are CC-BY-NC-4.0 and therefore non-commercial.

The torch-free name list in `jetson/model_shape.py` MUST carry the same backbone names, and
it MUST stay equal to the registry's, so a shape config naming a backbone the Jetson cannot
build is refused there rather than at run time on the device.

#### Scenario: The Tiny entry builds and forwards offline

- **WHEN** the Tiny entry is built with pretrained weights disabled at a 352 input
- **THEN** the model constructs, and a forward pass produces a segmentation output whose
  spatial size is the input divided by the mask stride
- **AND** no weight download is attempted during construction

#### Scenario: The Base entry builds and forwards offline

- **WHEN** the Base entry is built with pretrained weights disabled at a 352 input
- **THEN** the model constructs with the rung channels timm reports for that tag, and a
  forward pass produces a segmentation output whose spatial size is the input divided by
  the mask stride
- **AND** no weight download is attempted during construction

#### Scenario: A name missing from the Jetson mirror is refused there

- **WHEN** a shape config names a backbone outside `jetson/model_shape.py`'s list
- **THEN** the shape check raises its error naming the value and the accepted ones

#### Scenario: The registry and the Jetson mirror agree

- **WHEN** the backbone names are read from the registry and from `jetson/model_shape.py`
- **THEN** the two lists are equal

### Requirement: The model can carry a separate full-outline head

A model built with `amodal_outline=True` SHALL have, besides its segmentation head, a one-channel outline head reading the same feature map, and `forward(x, return_outline=True)` SHALL return the segmentation logits and the `[B, 1, H, W]` outline logits from one decoder pass. Plain `forward(x)` MUST return only the segmentation logits, with the same shape and values as a model built without the outline head from the same weights. The segmentation head MUST keep its 2-class [background, foreground] meaning; its loss, `val_iou`, and every consumer that reads it are unchanged. `return_outline=True` on a model without the outline head MUST raise. Building with `amodal_outline=True` and no segmentation head MUST raise. The default is `amodal_outline=False`.

#### Scenario: Plain forward is unchanged by the outline head
- **WHEN** a model with the outline head and a model without it are given the same weights for every shared tensor and the same input
- **THEN** their plain `forward` outputs are equal and have 2 channels

#### Scenario: One pass returns both outputs
- **WHEN** `forward(x, return_outline=True)` runs on a model with the outline head
- **THEN** it returns the 2-channel segmentation logits and 1-channel outline logits on the same grid

#### Scenario: Asking a plain model for the outline fails
- **WHEN** `forward(x, return_outline=True)` runs on a model built with `amodal_outline=False`
- **THEN** it raises an error saying the model has no outline head

#### Scenario: An outline head needs a segmentation head
- **WHEN** a model is built with `amodal_outline=True` and no segmentation classes
- **THEN** the build raises an error saying the outline head needs a segmentation head
