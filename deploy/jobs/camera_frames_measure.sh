#!/bin/bash
# Camera round 0 measurement (design D8). Read-only with respect to MinIO: it
# fetches the saved frames and two checkpoints, scores agreement, logs one
# MLflow run. The two sha256 pins come from the manifest's env; without them the
# tool refuses before downloading anything.
set -euo pipefail
# The pod untars the source ConfigMap into /work/src and runs this from there, so
# the repo layout (ecs_hcm_model/, jetson/, tools/) is the working directory.
cd "$(dirname "$0")"
echo "START $(date -u +%FT%TZ) git=${GIT_SHA_UNDER_TEST:-unknown}"
python -c 'import torch; print("torch", torch.__version__, "cuda", torch.cuda.is_available())'
nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader || true

: "${CAMERA_ROUND0_SHA_LIVE:?record the live checkpoint's sha256 and set CAMERA_ROUND0_SHA_LIVE}"
: "${CAMERA_ROUND0_SHA_STRONG:?record the strong checkpoint's sha256 and set CAMERA_ROUND0_SHA_STRONG}"
# The round's recorded last night. The capture ran past the stop and the bucket
# holds a later night; without the pin the tool would count six nights and the gate
# would stop the round for the wrong reason.
: "${CAMERA_ROUND0_LAST_NIGHT:?record the last counted night (YYYYMMDD) and set CAMERA_ROUND0_LAST_NIGHT}"

echo "--- prepared BDD tree (secondary reference only) ---"
python - <<'PY'
from ecs_hcm_model.core.bdd import fetch_prepared_tree
print("BDD_ROOT", fetch_prepared_tree("/work/bdd"), flush=True)
PY

echo "--- measure ---"
python tools/measure_camera_frames.py \
    --sha256 "live=${CAMERA_ROUND0_SHA_LIVE}" \
    --sha256 "strong=${CAMERA_ROUND0_SHA_STRONG}" \
    --last-night "${CAMERA_ROUND0_LAST_NIGHT}" \
    --bdd-root /work/bdd \
    --out-dir /work/out
echo "DONE $(date -u +%FT%TZ)"
