"""Count the COCOA train regions the old polygon guards threw away.

Until 9a36bfcb, build_binary_mask_coco_all dropped a COCOA polygon region when it
spanned >= 98% of the image on both axes, when its polygon area was > 80% of the
image, or when its rasterised fill was > 95%. COCOA val (PIL masks) only ever met a
0.995 fill check, so those regions were missing from the train target alone, and
nothing counted them. This scan replays the old guards over an amodal annotation
file (annotations only, no images) and reports how much the earlier runs lost,
plus how many regions the one remaining guard (fill > 0.995) still drops.

    python tools/cocoa_region_scan.py --annotations COCO_amodal_train2014.json
    python tools/cocoa_region_scan.py --annotations annotations.tar.gz

A tarball is searched for ``*_amodal_train2014.json`` unless ``--member`` names
another file. Prints one ``RESULT {...}`` JSON line.
"""
import argparse
import json
import sys
import tarfile

import numpy as np
from pycocotools import mask as mask_utils

from ecs_hcm_model.core.model_utils import _seg_to_polys, poly_area, poly_span_reject

OLD_SPAN_THRESH = 0.98
OLD_AREA_FRAC = 0.8
OLD_FILL_FRAC = 0.95
FILL_FRAC = 0.995


def _fill(poly, h, w):
    pts = [float(v) for v in np.asarray(poly, dtype=np.float64).reshape(-1)]
    return float(mask_utils.decode(mask_utils.merge(mask_utils.frPyObjects([pts], h, w))).mean())


def _rle_fill(rle, h, w):
    if isinstance(rle.get("counts"), list):
        rle = mask_utils.frPyObjects(rle, h, w)
    return float(mask_utils.decode(rle).mean())


def scan(data: dict) -> dict:
    """Replay the old and current COCOA polygon guards over an amodal annotation dict.

    A region is counted once, by the first old guard that would have dropped it, in
    the old order (span, area, fill). The old fill check is measured with the
    current (pycocotools) rasteriser; the old one drew about 1 px larger, which
    cannot move a fill across 95% on any real outline.

    Each region is read as the trainer reads it: its ``visible_mask`` first (an
    RLE in the raw amodal file, present when the region is occluded), then its
    ``segmentation`` polygon. A visible-mask region only ever met the fill >
    0.995 check (the old span/area/0.95 guards were polygon-only), so the counts
    are exact rather than upper bounds.
    """
    sizes = {im["id"]: (int(im["height"]), int(im["width"])) for im in data.get("images", [])}
    out = {
        "images": 0, "regions": 0, "polygons": 0, "empty_segmentations": 0,
        "old_span_rejects": 0, "old_area_rejects": 0, "old_fill_rejects": 0,
        "old_rejected_regions": 0, "old_images_with_a_rejected_region": 0,
        "current_fill_rejects": 0, "images_without_size": 0, "visible_masks": 0,
    }
    for ann in data.get("annotations", []):
        hw = sizes.get(ann.get("image_id"))
        if hw is None:
            out["images_without_size"] += 1
            continue
        h, w = hw
        out["images"] += 1
        image_lost = False
        for region in ann.get("regions") or []:
            out["regions"] += 1
            visible = region.get("visible_mask")
            if isinstance(visible, dict) and "counts" in visible:
                out["visible_masks"] += 1
                if _rle_fill(visible, h, w) > FILL_FRAC:
                    out["current_fill_rejects"] += 1
                continue
            polys = _seg_to_polys(visible or region.get("segmentation"))
            if not polys:
                out["empty_segmentations"] += 1
                continue
            region_lost = False
            for poly in polys:
                out["polygons"] += 1
                fill = _fill(poly, h, w)
                if fill > FILL_FRAC:
                    out["current_fill_rejects"] += 1
                if poly_span_reject(poly, h, w, span_thresh=OLD_SPAN_THRESH):
                    out["old_span_rejects"] += 1
                elif poly_area(poly) / (h * w) > OLD_AREA_FRAC:
                    out["old_area_rejects"] += 1
                elif fill > OLD_FILL_FRAC:
                    out["old_fill_rejects"] += 1
                else:
                    continue
                region_lost = True
            if region_lost:
                out["old_rejected_regions"] += 1
                image_lost = True
        if image_lost:
            out["old_images_with_a_rejected_region"] += 1
    return out


def _load(path: str, member: str = "") -> dict:
    if not tarfile.is_tarfile(path):
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    with tarfile.open(path) as tar:
        names = [m.name for m in tar.getmembers() if m.isfile()]
        want = [n for n in names if (n.endswith(member) if member else n.endswith("_amodal_train2014.json"))]
        if len(want) != 1:
            raise SystemExit(f"{path}: expected one member matching {member or '*_amodal_train2014.json'}, found {want}")
        return json.load(tar.extractfile(want[0]))


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--annotations", required=True, help="an amodal JSON file, or the annotations tarball")
    p.add_argument("--member", default="", help="tarball member to scan (default: *_amodal_train2014.json)")
    args = p.parse_args(argv)
    print("RESULT " + json.dumps(scan(_load(args.annotations, args.member)), sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
