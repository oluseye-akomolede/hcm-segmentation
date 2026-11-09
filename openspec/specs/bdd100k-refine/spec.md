# bdd100k-refine Specification

## Purpose
Refine the COCO segmentation model on BDD100K street scenes: prepare the BDD10k seg set once into
a binary foreground tree in object storage, score any checkpoint on its validation split with the
same IoU as COCO, and fine-tune on BDD mixed with COCO, selecting the best checkpoint on BDD
validation while recording COCO validation each epoch.

## Requirements

### Requirement: One named constant defines the BDD100K foreground

The target of a BDD100K refine MUST be the objects the model is for — person, rider, car,
truck, bus, train, motorcycle, bicycle and traffic light — and everything else in the label
MUST be background. That mapping MUST live in one named constant in
`ecs_hcm_model/core/bdd.py`, pairing each foreground class with its BDD label index, and no
other module MUST carry a second copy of the class list. A class label index the constant does
not name, including the upstream unknown index 255, MUST map to background.

#### Scenario: The foreground is the named objects only

- **WHEN** a BDD label contains a person, a car and a traffic light
- **THEN** the object mask is 1 on those three and 0 everywhere else

#### Scenario: Background classes are background

- **WHEN** a BDD label contains road, building, sky, vegetation and a traffic sign
- **THEN** the object mask is 0 on all of them

#### Scenario: The unknown index is background

- **WHEN** a label pixel is the upstream unknown index (255)
- **THEN** the object mask is 0 there

#### Scenario: The mapping is unit-tested

- **WHEN** the test suite runs
- **THEN** it asserts the constant's names against the plan's list and every name against its
  BDD label index

### Requirement: A BDD refine mixes COCO back in and validates on both

A BDD100K refine MUST select the existing segmentation-refine code path and MUST require the
refine mode. It MUST train on the prepared BDD training split with COCO training batches mixed
back in, at a mix measured as a share of each epoch's samples and defaulting to **half from
each dataset**, so a refine on street scenes does not unlearn what the model already knows. It
MUST validate on the BDD validation split and the COCO validation split every epoch, log each
under its own suffix (`val_loss_bdd`/`val_iou_bdd`/`val_dice_bdd` alongside the `_coco` ones),
and keep the unsuffixed `val_*` names as the mean of the two.

#### Scenario: The default mix is half and half

- **WHEN** a BDD refine runs with the default mix share
- **THEN** half of each epoch's samples come from BDD and half from COCO

#### Scenario: A zero share is BDD alone

- **WHEN** a BDD refine runs with a mix share of 0
- **THEN** no COCO batches are trained on and only BDD val is scored

#### Scenario: Both validation sets are scored every epoch

- **WHEN** an epoch of a mixed BDD refine finishes
- **THEN** BDD val and COCO val are each evaluated, and each dataset's loss, IoU and dice are
  logged under its own suffix

### Requirement: A BDD refine selects its best epoch on BDD val

Best-checkpoint selection and early stopping for a BDD refine MUST read the BDD validation
figure, not the mean of BDD and COCO. COCO is in the mix to hold what the model already knows;
the new target is BDD, and a COCO gain MUST NOT be able to pay for a BDD loss. The unsuffixed
metrics MUST still be logged as the mean, and the best checkpoint's metadata MUST record the
mix ratio, both per-dataset IoUs, and a dataset label naming the datasets actually trained on
(`BDD100K` or `BDD100K+MSCOCO`) rather than the refine task's COCOA label.

#### Scenario: A BDD gain wins over a COCO loss

- **WHEN** an epoch raises BDD val IoU while COCO val IoU falls by more, so the mean falls
- **THEN** the epoch is still the best so far for a BDD refine

#### Scenario: The checkpoint names the data

- **WHEN** the best checkpoint of a mixed BDD refine is uploaded
- **THEN** its metadata carries the mix ratio, both per-dataset IoUs, and `BDD100K+MSCOCO`

#### Scenario: A COCOA refine still decides on the mean

- **WHEN** a COCOA refine runs with a mix
- **THEN** its best epoch is still chosen on the unsuffixed mean

### Requirement: The BDD data is prepared once, outside the trainer image

The raw BDD100K label tiles MUST be converted to binary object masks by a prepare step that
runs as a bounded Job and writes a prepared tree — images, masks, split lists and a manifest of
counts and source checksums — to the object store. The per-tile masks MUST NOT be added to the
trainer image. The prepare step MUST refuse an image with no matching label and MUST refuse a
tree whose file counts disagree with the manifest unless an explicit subset is allowed, so a
half-downloaded release cannot pass as data. A pair whose image and label do not share a frame
MUST NOT be written as a sample: the prepare step MUST drop it from the split list and record
the split, stem, both sizes and the reason in the manifest, and the published-total check MUST
count written plus excluded so a truncated release still fails. The published archives' own
layout MUST be the basis of the match: images at `10k/<split>/<id>.jpg`, class-index labels at
`labels/<split>/<id>_train_id.png`, and a colour preview at
`color_labels/<split>/<id>_train_color.png` that is not a label. The split MUST be taken
from the directory, never from the filename: the published archive names every map with a
`_train_` token, the val split's included, so a filename token cannot identify a split.

#### Scenario: The prepare step counts what it wrote

- **WHEN** the prepare step finishes
- **THEN** its manifest records the train and val counts it wrote and the source checksums

#### Scenario: A missing label is refused

- **WHEN** an image has no matching label tile
- **THEN** the prepare step fails naming that image

#### Scenario: A truncated tree is refused

- **WHEN** the label count disagrees with the image count
- **THEN** the prepare step fails unless an explicit subset is allowed

#### Scenario: A pair whose frames disagree is excluded

- **WHEN** an image's decoded size differs from its label's shape
- **THEN** the pair is left out of the split list and never written, and the manifest
  records its split, stem, both sizes and the reason
- **AND** the pair still counts towards the published total, so a release missing other
  pairs fails the count check

#### Scenario: The label archive's two maps per image are told apart

- **WHEN** the label archive holds, for one image, both `labels/<split>/<id>_train_id.png`
  (class indices) and `color_labels/<split>/<id>_train_color.png` (a colourised preview)
- **THEN** only the `_id` map is the label, joined to `10k/<split>/<id>.jpg` with its
  `_train_id` suffix stripped, and the colour preview is never counted as a label

#### Scenario: A val label is named with the train token

- **WHEN** the val split's labels are read, each named `labels/val/<id>_train_id.png`
- **THEN** each pairs with `10k/val/<id>.jpg` because the split came from the `val`
  directory, with no rows left unmatched

### Requirement: A supplied binary mask is the sample's target

A segmentation transform that is handed a sample's own binary mask MUST use it as the target
instead of building one from COCO annotations, so a dataset whose rows carry a prepared mask
trains through the same transforms as COCO. The mask MUST be thresholded at greater than zero,
so an 8-bit 0/255 PNG and a 0/1 array produce the same target, and a mask whose size disagrees
with its image MUST be refused rather than resized or silently emptied.

#### Scenario: A prepared row trains through the COCO transform

- **WHEN** a row carries a binary mask and no annotations
- **THEN** the transform uses that mask as its target

#### Scenario: A mismatched mask is refused

- **WHEN** a row's mask is not the size of its image
- **THEN** the transform refuses it with an error naming both sizes

### Requirement: A BDD checkpoint can be scored on the prepared val split

The segmentation evaluation tool MUST accept `--dataset bdd` and score a checkpoint on the
prepared BDD validation split through the same metric path the trainer validates with, so the
project's reported number and the trainer's number are the same measurement.

#### Scenario: The tool scores BDD val

- **WHEN** the tool runs with `--dataset bdd` against a prepared tree
- **THEN** it scores the BDD validation split through the trainer's metric path

### Requirement: The prepared tree is fetched by one process per node

In a multi-process run, exactly one process per node MUST fetch the prepared tree, and no rank
MUST read it before that fetch completes. The fetching process MUST be the node's local rank 0,
not its global rank 0, because the shared download directory is the node's `/tmp`: every node
needs its own fetch. The node's other ranks MUST wait on a process-group barrier until that
fetch returns and MUST then fetch themselves, which MUST return at once on a complete tree and
so confirm it is whole on their side. A single-process run, or a run with no process group,
MUST take the plain fetch path with no barrier.

#### Scenario: One rank downloads while its peers wait

- **WHEN** the BDD branch runs with a process group of more than one rank on a node
- **THEN** local rank 0 calls the fetch before the barrier, every rank calls the barrier once,
  and the other ranks call the fetch only after the barrier

#### Scenario: A second node fetches for itself

- **WHEN** a rank with a global rank above 0 but a local rank of 0 reaches the BDD branch
- **THEN** it fetches before the barrier like any node's first process

#### Scenario: A single-process run never barriers

- **WHEN** the world size is one, or no process group is initialised
- **THEN** the tree is fetched once and no barrier is called
