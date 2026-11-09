#!/usr/bin/env bash
#
# Launch the HCM realtime demo. This is the documented way to start it -- the
# two environment variables below are load-bearing, not preferences.
#
#   PYTHONPATH=<repo root>
#     realtime_demo.py imports `ecs_hcm_model.core.hcm_nn` and
#     `jetson.frame_stream`, so the repository root has to be on the import
#     path. Without it the first import fails.
#
#   PYTORCH_NO_CUDA_MEMORY_CACHING=1
#     REQUIRED on this device. With caching on, torch 2.8.0 on JetPack 6.2
#     aborts on the first forward pass with:
#         NVML_SUCCESS == r INTERNAL ASSERT FAILED at
#         .../c10/cuda/CUDACachingAllocator.cpp:1131
#     With caching disabled the forward works. Measured footprint with it off
#     is about 250 MB, which this Orin Nano has. This is a workaround for the
#     allocator/JetPack pairing, not a performance knob -- do not remove it.
#
# Usage:  ./jetson/run_demo.sh
# Logs:   <repo>/run_demo.log   (tail it; the stream is at http://<JETSON_IP>:5000/video)
#
set -euo pipefail

# Resolve the repository root from this script's own location, so the launcher
# works from any working directory and always uses this checkout.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname -- "$SCRIPT_DIR")"

PYTHON="$REPO_ROOT/.venv/bin/python"
DEMO="$SCRIPT_DIR/realtime_demo.py"
LOG="$REPO_ROOT/run_demo.log"

if [[ ! -x "$PYTHON" ]]; then
    echo "error: no interpreter at $PYTHON -- run from the repository's own venv" >&2
    exit 1
fi

export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export PYTORCH_NO_CUDA_MEMORY_CACHING=1

# nohup + a detached process: the demo outlives the shell it was launched from,
# and its output lands in the log file rather than the terminal.
# Arguments pass through, e.g. ./run_demo.sh --out-stride 8
nohup "$PYTHON" -u "$DEMO" "$@" >>"$LOG" 2>&1 &
DEMO_PID=$!

echo "demo started: pid $DEMO_PID"
echo "log:          $LOG"
echo "stream:       http://<JETSON_IP>:5000/video"
