# numerics-precision Specification

## Purpose

Which parts of the forward pass run in fp32 and which run under the caller's autocast, and the magnitude measurements that make an overflow visible before it becomes a NaN. This is the code that decides whether the model trains at all, and it is where a range mistake shows up as an inf loss rather than an error.

## Requirements

### Requirement: The ISC to CSC chain runs in fp32 with autocast disabled

The whole ISC to CSC chain MUST run inside an autocast-disabled block in fp32. The CSC loop MUST be inside that block, not merely handed fp32 tensors, because autocast casts a convolution's inputs down to fp16 even when the caller passes fp32 values. Each ISC stage MUST be called on its input cast to float32.

#### Scenario: The chain is executed with autocast off

- **WHEN** the model's forward pass reaches the ISC list
- **THEN** autocast is disabled for the ISC and CSC loops
- **AND** each ISC stage receives a float32 input

#### Scenario: Disabling autocast covers the CSC loop too

- **WHEN** the CSC stages are evaluated
- **THEN** they run inside the same autocast-disabled block as the ISC stages rather than after it

### Requirement: The rung that consumes a raw ISC output runs in fp32

Of the four rrd rungs, the one that consumes a raw ISC output MUST run inside an autocast-disabled block, because autocast would cast that convolution's inputs down to fp16 and lose the range one step before the guard could see it. All other rungs consume CSC outputs and may run under the caller's autocast.

#### Scenario: The raw-ISC rung is guarded by fp32

- **WHEN** the rrd rung that consumes the ISC output directly is evaluated
- **THEN** autocast is disabled for that call

#### Scenario: Every rung output is checked against the fp16 boundary

- **WHEN** an rrd rung produces an output
- **THEN** its magnitude is checked before the value is handed to code that runs under autocast

### Requirement: Handing an unsafe magnitude to fp16 code is refused

Before a tensor is passed to code running under autocast or fp16, its magnitude MUST be compared against the fp16-safe maximum of 65504. A tensor whose magnitude exceeds that MUST raise rather than be cast, so an overflow becomes a loud failure instead of an inf that kills the run later.

#### Scenario: An over-large tensor raises at the boundary

- **WHEN** a tensor whose absolute maximum exceeds the fp16-safe maximum reaches the guard
- **THEN** an error is raised naming the tensor and its magnitude

#### Scenario: An in-range tensor passes

- **WHEN** a tensor's magnitude is within the fp16-safe maximum
- **THEN** the guard passes it through and the run continues

### Requirement: Activation magnitudes are measured as running maxima

The model MUST record a running absolute maximum for the `isc`, `csc` and `isc3` stages, and these MUST be true maxima over everything seen, not a sample of the batches that happened to land on a logging boundary. `isc3` MUST be tracked separately from `isc` because it is the only ISC output that reaches the decoder and therefore the one the rrd boundary has to survive.

#### Scenario: A spike between logging points is still caught

- **WHEN** a batch produces a large activation and the next logging boundary is several batches away
- **THEN** the recorded maximum for that stage includes the spike

#### Scenario: isc3 is recorded alongside isc

- **WHEN** the fourth ISC output is produced
- **THEN** a maximum is recorded under both the `isc` and `isc3` stage keys

### Requirement: The rrd_list rung list is indexed from the decoder side

The rrd outputs MUST be built by iterating the rung indices in descending order, with the raw-ISC rung as the last rung in that order, so that each rung's second argument is the output of the rung below it. The decoder consumes the first rrd output.

#### Scenario: Rungs are built decoder-first

- **WHEN** the rrd outputs are constructed
- **THEN** the index the raw-ISC rung is handled at is the deepest index, and the loop proceeds from deepest to shallowest

#### Scenario: The decoder receives the shallowest rung

- **WHEN** the forward pass reaches the decoder
- **THEN** it consumes the first element of the rrd output list
