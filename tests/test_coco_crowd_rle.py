"""Raw COCO crowd segmentations are uncompressed RLE: a list of run lengths.

pycocotools.decode only accepts compressed counts, so the shared decoder must
convert a run-length list with frPyObjects first. Before that it raised inside
the decoder, returned None, and the builder fell back to the annotation's bbox.
"""
import numpy as np
from PIL import Image
from pycocotools import mask as mask_utils

from ecs_hcm_model.core.model_utils import build_binary_mask_coco_all, decode_coco_rle_robust

H, W = 60, 80


def _crowd_mask():
    m = np.zeros((H, W), np.uint8)
    m[10:25, 5:30] = 1      # two separate blobs, so the bbox fallback
    m[40:55, 50:75] = 1     # (which spans both) cannot pass for the mask
    return m


def _uncompressed(m):
    """The raw instances_val2017.json form: column-major run lengths, zeros first."""
    flat = m.flatten(order="F")
    counts, cur, run = [], 0, 0
    for v in flat:
        if v != cur:
            counts.append(run)
            cur, run = v, 0
        run += 1
    counts.append(run)
    return {"counts": counts, "size": [H, W]}


def _compressed(m):
    rle = mask_utils.encode(np.asfortranarray(m))
    return {"counts": rle["counts"].decode("ascii"), "size": [H, W]}


def _sample(seg):
    ys, xs = np.nonzero(_crowd_mask())
    bbox = [float(xs.min()), float(ys.min()), float(xs.max() - xs.min() + 1), float(ys.max() - ys.min() + 1)]
    return {
        "image": Image.new("RGB", (W, H)),
        "annotations": [{"iscrowd": 1, "segmentation": seg, "bbox": bbox, "category_id": 1}],
    }


def test_the_decoder_reads_uncompressed_run_lengths():
    m, mode = decode_coco_rle_robust(_uncompressed(_crowd_mask()), expected_h=H, expected_w=W)
    assert mode == "uncompressed_rl"
    assert np.array_equal(m, _crowd_mask())


def test_uncompressed_and_compressed_crowd_give_the_same_target():
    raw = build_binary_mask_coco_all(_sample(_uncompressed(_crowd_mask())))
    packed = build_binary_mask_coco_all(_sample(_compressed(_crowd_mask())))
    assert np.array_equal(packed, _crowd_mask())
    assert np.array_equal(raw, packed)
