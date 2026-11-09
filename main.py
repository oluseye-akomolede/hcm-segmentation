from argparse import ArgumentParser, Namespace, _SubParsersAction

from ecs_hcm_model.core.services.trainer import run_train_hcm_imagenet1k
from ecs_hcm_model.core.backbones import DEFAULT_BACKBONE, backbone_names
from ecs_hcm_model.core.hcm_nn import MASK_STRIDE_DEFAULT, MASK_STRIDES, NECK_CHANNELS
from ecs_hcm_model.core.trainer import (
    AugmentConfigError,
    CocoaMixRatioError,
    GradAccumError,
    InitFromError,
    LrDecayError,
)

def setup_parser(subparsers: _SubParsersAction, action = "train_hcm_imagenet1k") -> ArgumentParser:
    parser = subparsers.add_parser(
        action,
        help="Train HCM model on ImageNet1k dataset",
    )
    parser.add_argument(
        "--new_model",
        action="store_true",
        help="Whether to create a new model instance",
    )
    parser.add_argument(
        "--environment",
        type=str,
        default="development",
        help="Environment to determine MLflow tracking URI",
    )
    parser.add_argument(
        "--finetune",
        action="store_true",
        help="Whether to enable finetuning",
    )
    parser.add_argument(
        "--coco",
        action="store_true",
        help="Whether to enable COCO dataset training",
    )
    parser.add_argument(
        "--model-base",
        type=str,
        choices=["models", "finetuned-models", "coco-models"],
        default="models",
        help="Base bucket to load model from for finetuning",
    )
    parser.add_argument(
        "--max_epochs",
        type=int,
        default=5,
        help="Maximum number of training epochs",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Batch size for training",
    )
    parser.add_argument(
        "--test_mode",
        action="store_true",
        help="Whether to enable test mode and test the training on a small dataset.",
    )
    parser.add_argument(
        "--warmup",
        action="store_true",
        help="Whether to enable warmup phase where backbone is frozen.",
    )
    parser.add_argument(
        "--max_batches_per_epoch",
        type=int,
        default=0,
        help=(
            "Cap batches per training epoch (0 = no cap). In test mode the "
            "32-batch smoke default applies only when this is not given."
        ),
    )
    parser.add_argument(
        "--input-size",
        type=int,
        default=192,
        help=(
            "Model input H=W for the COCO/COCOA segmentation pipeline "
            "(default: 192). The shorter-side resize scales with it, keeping the "
            "256/192 ratio."
        ),
    )
    parser.add_argument(
        "--coco-mix-ratio",
        type=float,
        default=0.0,
        help=(
            "COCOA refine only: interleave this many COCO training batches per "
            "COCOA batch and score both validation sets (default: 0.0, which "
            "refines on COCOA alone). Requires --finetune."
        ),
    )
    parser.add_argument(
        "--bdd",
        action="store_true",
        help=(
            "Train a BDD100K street-scene refine instead of the COCOA refine: "
            "eight object classes (person, rider, car, truck, bus, train, "
            "motorcycle, bicycle) plus traffic lights are the target and every "
            "other class is background. Requires --finetune. The tree comes from "
            "--bdd-data-root, else $BDD100K_DATA_ROOT, else "
            "/tmp/bdd100k/prepared, fetched from the object store if absent. "
            "Checkpoints use the refine task key."
        ),
    )
    parser.add_argument(
        "--bdd-data-root",
        default=None,
        help=(
            "The prepared BDD100K tree (images/, masks/, train.txt, val.txt). "
            "Defaults to $BDD100K_DATA_ROOT, else /tmp/bdd100k/prepared, which "
            "is fetched from the object store when it is not already there."
        ),
    )
    parser.add_argument(
        "--bdd-coco-mix-share",
        type=float,
        default=0.5,
        help=(
            "BDD refines only: the share of each epoch's samples drawn from "
            "COCO, so the model does not forget what it knows while it learns "
            "the street-scene target. 0.5 (the default) is 50/50 by sample "
            "count; 0 trains on BDD alone. Validation runs on BDD val and COCO "
            "val every epoch under separate MLflow keys, and the best epoch is "
            "chosen on BDD val alone."
        ),
    )
    parser.add_argument(
        "--init-from",
        default=None,
        help=(
            "Warm start: build a fresh model and copy into it every tensor of "
            "this checkpoint (<bucket>/<prefix>) whose shape still fits. Starts "
            "a new lineage, so it implies --new_model and never resumes."
        ),
    )
    parser.add_argument(
        "--init-decoder-arch",
        choices=("v1", "v2"),
        default="v2",
        help="Decoder version of the warm-started model (default: v2).",
    )
    parser.add_argument(
        "--backbone",
        choices=backbone_names(),
        default=DEFAULT_BACKBONE,
        help=(
            "Backbone of a fresh model: the ladder every rung of the decoder "
            f"reads (default: {DEFAULT_BACKBONE})."
        ),
    )
    parser.add_argument(
        "--neck-channels",
        type=int,
        default=None,
        metavar="C",
        help=(
            "Width each backbone rung is projected to before the decoder "
            f"(default: {NECK_CHANNELS} for the v2 backbones, and the backbone's "
            "own pre-change width for resnet50, which is what its checkpoints "
            "were trained with)."
        ),
    )
    parser.add_argument(
        "--mask-stride",
        type=int,
        choices=MASK_STRIDES,
        default=MASK_STRIDE_DEFAULT,
        help=(
            "Stride of the segmentation output: 4 is the pre-change model and "
            "its checkpoints, 2 adds the finer-mask branch and trains and scores "
            f"at twice that resolution (default: {MASK_STRIDE_DEFAULT})."
        ),
    )
    parser.add_argument(
        "--aug-scale-jitter",
        type=float,
        nargs=2,
        metavar=("LO", "HI"),
        default=None,
        help=(
            "COCO/COCOA train only: draw each image's short-side resize from "
            "[LO, HI] times the configured size, mask resized with it "
            "(default: off, a single size as before)."
        ),
    )
    parser.add_argument(
        "--aug-color-jitter",
        type=float,
        default=None,
        help=(
            "COCO/COCOA train only: jitter brightness, contrast and saturation "
            "by +-S, and hue by +-S/4, on the image only and never the mask "
            "(default: off)."
        ),
    )
    parser.add_argument(
        "--grad-accum-steps",
        type=int,
        default=1,
        help=(
            "Step the optimizer every N batches, applying the mean gradient of "
            "the N with the loss divided by N, so N*batch_size is the effective "
            "batch size (default: 1, one step per batch)."
        ),
    )
    parser.add_argument(
        "--lr-decay-start",
        type=int,
        default=None,
        metavar="N",
        help=(
            "Epoch the cosine lr decay starts at (default: 5 with warmup, 0 "
            "without). Earlier than the last warmup epoch, the decay factor is "
            "applied on top of the stage's lr rather than replacing it."
        ),
    )
    parser.add_argument(
        "--early-stop-monitor",
        choices=("val_loss", "val_iou"),
        default="val_loss",
        help=(
            "Which validation metric drives early stopping (default: val_loss, "
            "which must fall). val_iou must rise instead. val_iou needs a "
            "segmentation run, since only those validation metrics carry it."
        ),
    )
    parser.add_argument(
        "--early-stop-patience",
        type=int,
        default=5,
        metavar="N",
        help=(
            "Consecutive non-improving epochs before training stops (default: "
            "5, the behaviour before this flag). 0 disables early stopping."
        ),
    )
    parser.add_argument(
        "--amodal-outline",
        action="store_true",
        help=(
            "Add the amodal outline head and train it on each sample's full-"
            "extent target. Needs a set that carries COCOA's full polygons; a "
            "COCO or BDD training set has none and the run is refused before "
            "the first step (default: off, no head, no extra target)."
        ),
    )
    parser.add_argument(
        "--full-outline-weight",
        type=float,
        default=1.0,
        metavar="W",
        help=(
            "Weight of the outline term added to the segmentation loss "
            "(default: 1.0). Logged as a run param; the term is a per-image "
            "mean against the visible loss's per-pixel mean, so 1.0 is a "
            "starting point, not a balanced one."
        ),
    )
    parser.set_defaults(func=train_hcm_1k)
    
def train_hcm_1k(args: Namespace) -> None:
    run_train_hcm_imagenet1k(
        # --init-from builds a fresh model by definition; it never resumes.
        new_model=args.new_model or bool(args.init_from),
        environment=args.environment,
        finetune=args.finetune,
        coco=args.coco,
        model_base=args.model_base,
        max_epochs=args.max_epochs,
        batch_size=args.batch_size,
        test_mode=args.test_mode,
        warmup=args.warmup,
        max_batches_per_epoch=args.max_batches_per_epoch,
        input_size=args.input_size,
        coco_mix_ratio=args.coco_mix_ratio,
        init_from=args.init_from,
        init_decoder_arch=args.init_decoder_arch,
        backbone=args.backbone,
        neck_channels=args.neck_channels,
        mask_stride=args.mask_stride,
        scale_jitter=tuple(args.aug_scale_jitter) if args.aug_scale_jitter else None,
        color_jitter=args.aug_color_jitter,
        grad_accum_steps=args.grad_accum_steps,
        lr_decay_start=args.lr_decay_start,
        bdd=args.bdd,
        bdd_data_root=args.bdd_data_root,
        bdd_coco_mix_share=args.bdd_coco_mix_share,
        early_stop_monitor=args.early_stop_monitor,
        early_stop_patience=args.early_stop_patience,
        amodal_outline=args.amodal_outline,
        full_outline_weight=args.full_outline_weight,
    )

def setup_subpparsers() -> ArgumentParser:
    parser = ArgumentParser(
        description="ECS HCM Model Training and Evaluation CLI"
    )
    parser.set_defaults(func=lambda args: parser.print_help())
    subparsers = parser.add_subparsers(
        dest="action", help="Available actions"
    )
    setup_parser(subparsers, action="train_hcm_imagenet1k")
    return parser

def main() -> None:
    parser = setup_subpparsers()
    args = parser.parse_args()
    try:
        args.func(args)
    except (AugmentConfigError, CocoaMixRatioError, GradAccumError, InitFromError,
            LrDecayError) as exc:
        # A flag that cannot mean anything for this invocation is a usage
        # error: report it as one rather than as a traceback out of training.
        parser.error(str(exc))


if __name__ == "__main__":
    main()