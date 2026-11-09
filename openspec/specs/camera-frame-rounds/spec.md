# camera-frame-rounds Specification

## Purpose
Collect camera frames the deployed segmentation model is unsure about, in bounded rounds, so they can be
labelled and used to measure and improve the model on its real deployment scenes.

## Requirements

### Requirement: Captured frames live in one bucket, one prefix per round, and are never overwritten

Captured frames MUST be stored in the home-cluster MinIO bucket
`camera-uncertain-frames`, under `round0/<YYYYmmdd>/`, one immutable JPEG object per frame,
where `<YYYYmmdd>` is the frame's America/Chicago calendar day (the day the daily cap
counts). A frame's object key MUST be its UTC timestamp in milliseconds, so that a retried
upload of the same frame rewrites one key instead of duplicating the frame, and two saved
frames — which the enforced spacing always separates — never share a key. Each object's
per-frame metadata record MUST be uploaded beside it, so the analysis does not depend on
the device's local index. No frame may be written to a git repository, the cloud repository,
or cloud sessions, and none may leave the home cluster.

#### Scenario: A retried upload does not duplicate the frame
- **WHEN** the uploader retries a frame whose object was already stored
- **THEN** it writes the same key and the bucket still holds one object for that frame

#### Scenario: Two saved frames never share a key
- **WHEN** a session saves many frames
- **THEN** every object key is distinct and no frame overwrites another

#### Scenario: The round's prefix is fixed
- **WHEN** round 0 uploads a frame
- **THEN** its key starts with `round0/`

#### Scenario: No frame leaves the home cluster
- **WHEN** capture and upload run for a round
- **THEN** every object is written to the home-cluster MinIO service, and no frame or record
  is written to a git repository, the cloud repository, or a cloud session

### Requirement: The camera's upload credential can only add round-0 frames

The client credential used on the device MUST be a dedicated identity whose only grants are
`PutObject` and `AbortMultipartUpload` on `camera-uncertain-frames/round0/*`. It MUST NOT be
able to read, delete or list, and it MUST NOT reach another bucket.

#### Scenario: The camera credential can add a frame
- **WHEN** a put is attempted into `camera-uncertain-frames/round0/` with the device credential
- **THEN** it succeeds

#### Scenario: The camera credential cannot read, delete or stray
- **WHEN** a get or a delete is attempted with the device credential, or a put into any other
  bucket or prefix
- **THEN** each is refused

### Requirement: Captured frames expire after 30 days

The bucket MUST carry a server-side expiry rule giving objects under `round0/` a lifetime of
30 days, and it MUST be verified from outside the rule: a scheduled check MUST fail, and
report, when the rule is absent or when any object is older than 31 days.

#### Scenario: The rule is present
- **WHEN** the scheduled check runs
- **THEN** it reports the 30-day rule on `round0/` and passes

#### Scenario: An expired object is caught
- **WHEN** any object under `round0/` is older than 31 days
- **THEN** the check fails and reports the object's key and age

### Requirement: Round 0 measures the gap between the live model and the strongest model, against a control

The measurement MUST run as a bounded Kubernetes Job with resource limits and matching
thread counts, MUST use the two exact checkpoints (the deployed live checkpoint and the
strongest 2-class checkpoint), each verified against a recorded sha256, MUST preprocess
frames exactly as the live demo does, and MUST score each frame with the shared
`iou_dice_fg_from_pred` arithmetic on argmax masks. It MUST compute the same quantity on a
control sample of camera frames the same policy saved independently of their score, over the
same nights and grouped by night so each night's gap can use its own control mean; that
camera control is the gate's baseline. It MUST also compute the same
quantity on a random sample of ordinary validation frames as a secondary reference. It MUST
log all three means to one MLflow run with the checkpoints' identities and a per-frame
artefact. Both models MUST be scored on one pre-registered comparison grid (352 px, the live
model's input size when the gate was written) and the Job MUST refuse, before scoring, when
the live model's configured input size is not that grid: reading the grid off whichever live
checkpoint happens to be deployed would let a re-pointed live model change every number the
gate compares. It MUST read the bucket with a client built with an explicit region
(`MINIO_REGION`, default `us-east-1`), never left to discovery — the region is a property of
the endpoint, and a client that asks the server for the bucket's location needs a grant the
scoped credentials here do not carry.

#### Scenario: The same arithmetic scores both sets
- **WHEN** the Job scores an uncertain frame and a control frame
- **THEN** both go through the same preprocessing, the same two models and the same IoU
  function

#### Scenario: The gate's control comes from the same camera
- **WHEN** the Job computes the control mean the gate uses
- **THEN** it is computed over camera frames selected by elapsed time rather than by score,
  and the validation-frame mean is reported separately and does not gate

#### Scenario: The comparison grid does not follow the live checkpoint
- **WHEN** the live model's configured input size is not the pre-registered 352
- **THEN** the Job refuses before scoring any frame, naming both sizes, rather than
  rescaling the comparison quietly

#### Scenario: The measurement client never asks for the bucket location
- **WHEN** the Job opens the bucket
- **THEN** the client is built with the configured region, `us-east-1` unless `MINIO_REGION`
  says otherwise, so no grant beyond the bucket's data path is needed

#### Scenario: A checkpoint that does not match its sha256 stops the Job
- **WHEN** a checkpoint file's digest differs from the recorded value
- **THEN** the Job fails before scoring any frame, naming the file

#### Scenario: Both results reach one run
- **WHEN** the Job finishes
- **THEN** its MLflow run holds the uncertain-set and control-set aggregates, both checkpoint
  identities, and the per-frame table

### Requirement: Round 0 recommends, and only the owner starts a later round

Round 0 MUST end at its measurement. Its collection window MUST be fixed before capture
starts — five counted nights — and only frames captured before the stop MUST enter the gate.
The measurement MUST therefore take the round's recorded last night, as `YYYYMMDD` (the last
night the round counts), and MUST drop every frame on a later night before scoring and
gating, because the capture can run past the stop and a night written after it would
otherwise be counted. It MUST report how many frames it dropped and on which night, and
record the last night and those counts with the run. A value that is not a `YYYYMMDD` date
MUST be refused before any frame is read. When no last night is given, every night under
the round's prefix counts, as before.
It MUST recommend round 1 only when all of these hold: at least 500 saved uncertain frames;
exactly 5 counted nights, where a counted night is an America/Chicago calendar day with at
least 100 saved uncertain frames; at least 25 camera control frames in each counted night;
the pooled uncertain mean IoU at least 0.10 below the pooled camera
control mean IoU; and, in every counted night, that night's uncertain mean IoU at least 0.10
below **that same night's** camera control mean IoU. A per-night gap MUST NOT be computed
against the pooled control mean. The result MUST record each counted night's control count,
mean and standard deviation, and the control frames' mean score. Insufficient exposure MUST
be a stated stop for a **counted** night only: a counted night with fewer than 25 controls
MUST end the round with a stop naming that night and its count. If five counted
nights have not accumulated within ten local days, the round MUST still end and report a stop
on the night count with the numbers it has. A night with fewer than 100 uncertain frames MUST
be reported but MUST neither count toward the minimum nor fail the gate. The validation-frame reference mean MUST be reported and MUST
NOT gate. Otherwise the result MUST recommend stopping. No training, labelling round,
checkpoint promotion, camera update or automatic change may follow from this capability
without a new decision from the owner.

#### Scenario: A passing gate is reported, not acted on
- **WHEN** the measured gap meets every condition, pooled and against each counted night's
  own controls
- **THEN** the result is reported as a recommendation and nothing is trained or deployed

#### Scenario: One bad night fails the gate
- **WHEN** the pooled gap meets the rule and one counted night's gap against its own controls
  does not
- **THEN** the result is reported as a stop, naming that night

#### Scenario: Each night is compared against its own controls
- **WHEN** two counted nights have very different control means
- **THEN** each night's gap is computed against that night's own control mean, not the
  pooled one

#### Scenario: Too few controls in a counted night stops the round
- **WHEN** a counted night holds fewer than 25 control frames
- **THEN** the result is a stop naming that night and its control count

#### Scenario: A short night's exposure does not stop the round
- **WHEN** a night holds fewer than 100 uncertain frames and fewer than 25 control frames
- **THEN** it is reported, not counted, and the round continues

#### Scenario: A failing gate is reported as a stop
- **WHEN** the measured gap does not meet the rule, or the frame, control-frame or night
  minimum is not met
- **THEN** the result is reported as a stop with the numbers

#### Scenario: A short round ends on time
- **WHEN** five counted nights have not accumulated within ten local days
- **THEN** the round still ends and the gate reports a stop on the night count, using only
  the frames captured before the stop

#### Scenario: Frames captured after the stop do not count
- **WHEN** the bucket holds a night later than the round's recorded last night and the
  measurement is given that last night
- **THEN** the later night's frames are dropped before scoring, the gate counts only the
  nights up to the stop, and the dropped frames are reported by night

#### Scenario: A malformed last night is refused
- **WHEN** the recorded last night is not a valid `YYYYMMDD` date
- **THEN** the measurement refuses before reading any frame
