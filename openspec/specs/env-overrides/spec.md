# env-overrides Specification

## Purpose

The environment variables an operator can use to move a threshold without rebuilding the image, and the rule that a bad value is warned about and ignored rather than obeyed. These knobs exist so an incident like a double-booked GPU can be worked around without a code change, which only works if an unusable value cannot make things worse.

## Requirements

### Requirement: Thresholds are read per call, not at import

Every overridable threshold MUST be resolved each time it is used rather than bound at module import, so an operator or a test can set the variable on a running process without reloading the module. An unset or empty variable MUST yield the module constant.

#### Scenario: Setting the variable on a live process takes effect

- **WHEN** a threshold variable is set after the module is already imported
- **THEN** the next read of that threshold uses the new value

#### Scenario: An unset variable yields the default

- **WHEN** a threshold variable is unset
- **THEN** the module's default constant is used

#### Scenario: An empty variable is treated as unset

- **WHEN** a threshold variable is set to the empty string
- **THEN** the module's default constant is used

### Requirement: Non-finite numbers are rejected rather than obeyed

A float threshold MUST be rejected when its value is NaN or infinite, because both parse successfully and a NaN threshold makes every comparison against it false, which silently disables the guard the variable configures. A rejected value MUST produce a warning and fall back to the default.

#### Scenario: NaN falls back to the default

- **WHEN** a float threshold is set to `nan`
- **THEN** the default is used and a warning is printed

#### Scenario: Infinite values fall back to the default

- **WHEN** a float threshold is set to `inf` or `-inf`
- **THEN** the default is used and a warning is printed

#### Scenario: A NaN does not reach the OOM fraction limit

- **WHEN** the OOM fraction limit is set to `nan`
- **THEN** the effective limit is the module default and is not NaN

### Requirement: Integer thresholds reject non-numeric and special values

An integer threshold MUST be rejected when it does not parse as an integer, which covers both plain text and the non-numeric `nan` and `inf` spellings, and MUST fall back to the default with a warning.

#### Scenario: A malformed integer falls back

- **WHEN** an integer threshold is set to a non-numeric string
- **THEN** the default is used and a warning is printed

#### Scenario: nan and inf never parse as integers

- **WHEN** an integer threshold is set to `nan`, `inf` or `NaN`
- **THEN** the default is used and no exception escapes

### Requirement: A threshold below its smallest usable value falls back

Each threshold MUST declare the smallest value that makes sense for it, and any value below that MUST be warned about and replaced by the default rather than accepted, because a threshold of zero or less silently disables the guard it configures. The declared minimums MUST be: at least 1 for the consecutive-OOM skip limit, and at least 0 for the OOM skip fraction and the minimum free GPU memory.

#### Scenario: A zero skip limit falls back

- **WHEN** the consecutive-OOM limit is set to 0, 0.0 or a negative integer
- **THEN** the default is used and a warning is printed

#### Scenario: A negative free-memory threshold falls back

- **WHEN** the minimum free GPU memory is set below zero
- **THEN** the default is used and a warning is printed

#### Scenario: A zero fraction limit is accepted

- **WHEN** the OOM skip fraction is set to 0
- **THEN** the value is used, because zero is a meaningful floor for a fraction

### Requirement: A bad override is never fatal

Resolving any overridable threshold MUST NOT raise for any string an operator could set, including unparseable text, non-finite numbers and values below the declared minimum. Each of these MUST warn and fall back so a typo in an environment variable cannot take down a training run.

#### Scenario: Arbitrary garbage does not crash the run

- **WHEN** a threshold variable is set to arbitrary unparseable text
- **THEN** the run continues with the default and a warning is printed

#### Scenario: Every rejection names the variable

- **WHEN** any override is rejected
- **THEN** the warning names the variable, the value received and the default being used

### Requirement: The float helper returns a usable float or the default

The float helper MUST return a value the caller can compare against directly: a finite number at or above the declared minimum, or the default. It MUST NOT return a value whose comparisons against it would be silently false.

#### Scenario: A finite value is returned as given

- **WHEN** a float threshold is set to a finite value at or above its minimum
- **THEN** that value is returned

#### Scenario: A finite value below the minimum is rejected

- **WHEN** a float threshold is set below its declared minimum
- **THEN** the default is returned instead
