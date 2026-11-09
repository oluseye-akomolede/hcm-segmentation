#!/usr/bin/env bash
# Build and submit the round-0 measurement Job (design D8, task 5.4).
#
# Usage:
#   deploy/jobs/launch_camera_frames_measure.sh \
#     --live-sha256 <64 hex> --strong-sha256 <64 hex> --last-night <YYYYMMDD> \
#     [--image-tag TAG] [--ref REV] [--dry-run]
#
# The measurement cannot run from a bare `kubectl apply`: it mounts two ConfigMaps
# that are not in the repo -- the source tarball (this repo's import closure from
# tools/measure_camera_frames.py, plus deploy/jobs/camera_frames_measure.sh as
# run.sh) and camera-round0-pins (the two checkpoint sha256s the tool refuses to
# start without). This script builds both, creates the Job with the image tag
# injected, and prints the rollback.
#
# Env: KUBE_HOST (default controlplane) -- kubectl lives there, not on this VM.
#      REPO      (default: this script's git toplevel)
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# REPO is resolved below, after the arguments are checked: a bad flag must refuse
# with its own message anywhere, including a copy of this script that is not in a
# git checkout (CI stages an export at /tmp/src, where rev-parse exits 128).
KUBE_HOST=${KUBE_HOST:-controlplane}
# A ConfigMap holds at most 1 MiB of key data (measured 2026-10-27: 1,000,000
# bytes accepted, 1,050,000 rejected). Over this, no Job is created at all.
CM_LIMIT=1000000

REF=HEAD
TAG=a0cceab
DRY=0
LIVE=""
STRONG=""
LAST_NIGHT=""

die() { echo "REFUSE $*" >&2; exit 2; }

while [[ $# -gt 0 ]]; do
  case "$1" in
    --live-sha256)   LIVE=$2; shift 2 ;;
    --strong-sha256) STRONG=$2; shift 2 ;;
    --last-night)    LAST_NIGHT=$2; shift 2 ;;
    --image-tag)     TAG=$2; shift 2 ;;
    --ref)           REF=$2; shift 2 ;;
    --dry-run)       DRY=1; shift ;;
    -h|--help)       sed -n '2,19p' "${BASH_SOURCE[0]}"; exit 0 ;;
    *)               die "unknown argument: $1" ;;
  esac
done

[[ $LIVE =~ ^[0-9a-f]{64}$ ]] \
  || die "--live-sha256 must be the 64-hex sha256 of the live checkpoint"
[[ $STRONG =~ ^[0-9a-f]{64}$ ]] \
  || die "--strong-sha256 must be the 64-hex sha256 of the strong checkpoint"

# The round's recorded last night: the night directory under round0/ (YYYYMMDD).
# Checked as a real date, not just eight digits, so a 20261131 refuses here rather
# than in a running Job.
[[ $LAST_NIGHT =~ ^[0-9]{8}$ ]] && [[ $(date -d "$LAST_NIGHT" +%Y%m%d 2>/dev/null) == "$LAST_NIGHT" ]] \
  || die "--last-night must be the round's last counted night as a real YYYYMMDD date"

REPO=${REPO:-$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel)}
SHA=$(git -C "$REPO" rev-parse --short=7 "$REF^{commit}")
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT
mkdir -p "$TMP/tree" "$TMP/src"

# Unpack $REF first and walk *that* tree: listing from the checkout while copying
# from the commit is how a stale file decides what the pod gets -- a module the
# working tree still imports would be packed, and one it dropped would be missing.
git -C "$REPO" archive --format=tar "$SHA" | tar -x -C "$TMP/tree" \
  || die "cannot unpack $SHA"
mapfile -t FILES < <(cd "$TMP/tree" && python3 tools/src_closure.py tools/measure_camera_frames.py)
[[ ${#FILES[@]} -gt 0 ]] || die "the source closure is empty"
for f in "${FILES[@]}" deploy/jobs/camera_frames_measure.sh; do
  [[ -f "$TMP/tree/$f" ]] || die "$f does not exist at $SHA"
  mkdir -p "$TMP/src/$(dirname "$f")"
  cp "$TMP/tree/$f" "$TMP/src/$f"
done
# The Job runs this as run.sh; it is the repo's own script, not a copy.
mv "$TMP/src/deploy/jobs/camera_frames_measure.sh" "$TMP/src/run.sh"
rmdir "$TMP/src/deploy/jobs" "$TMP/src/deploy"
tar --sort=name --mtime=@0 --owner=0 --group=0 --numeric-owner \
    -czf "$TMP/src.tgz" -C "$TMP/src" .
BYTES=$(stat -c %s "$TMP/src.tgz")
[[ $BYTES -lt $CM_LIMIT ]] \
  || die "source tarball is $BYTES bytes, over the $CM_LIMIT-byte ConfigMap limit"
echo "src: ${#FILES[@]} files (+run.sh), $BYTES bytes tarball (limit $CM_LIMIT)"

# The manifest is the record of the pod; the launcher only supplies what varies per
# run -- the image tag and the commit being measured. Both substitutions are
# checked, so a manifest edit cannot silently leave a placeholder in place.
sed -e "s|ecs_hcm_trainer:[A-Za-z0-9._-]*|ecs_hcm_trainer:$TAG|" \
    -e "s|{name: GIT_SHA_UNDER_TEST, value: \"[^\"]*\"}|{name: GIT_SHA_UNDER_TEST, value: \"$SHA\"}|" \
    "$REPO/deploy/jobs/camera_frames_measure_job.yaml" > "$TMP/job.yaml"
grep -q "ecs_hcm_trainer:$TAG" "$TMP/job.yaml" || die "could not inject image tag $TAG"
grep -q "GIT_SHA_UNDER_TEST, value: \"$SHA\"" "$TMP/job.yaml" || die "could not inject sha $SHA"

if [[ $DRY == 1 ]]; then
  echo "--- dry run: nothing sent to $KUBE_HOST ---"
  echo "ConfigMap camera-frames-round0-measure-src  key src.tgz ($BYTES bytes)"
  echo "ConfigMap camera-round0-pins  live_sha256/strong_sha256/last_night=$LAST_NIGHT"
  cat "$TMP/job.yaml"
  exit 0
fi

scp -q "$TMP/src.tgz" "$TMP/job.yaml" "$KUBE_HOST:/tmp/"

ssh "$KUBE_HOST" bash -s <<EOF
set -euo pipefail
cd /tmp
mv src.tgz camera-round0-src-$SHA.tgz
mv job.yaml camera-round0-job-$SHA.yaml
kubectl -n default create configmap camera-frames-round0-measure-src \
  --from-file=src.tgz=camera-round0-src-$SHA.tgz --dry-run=client -o yaml | kubectl apply -f - >/dev/null
kubectl -n default create configmap camera-round0-pins \
  --from-literal=live_sha256=$LIVE --from-literal=strong_sha256=$STRONG \
  --from-literal=last_night=$LAST_NIGHT \
  --dry-run=client -o yaml | kubectl apply -f - >/dev/null
if ! kubectl create -f camera-round0-job-$SHA.yaml; then
  echo "the Job already exists; delete it first:" >&2
  echo "  kubectl -n default delete job camera-frames-round0-measure" >&2
  exit 1
fi
rm -f camera-round0-src-$SHA.tgz camera-round0-job-$SHA.yaml
echo "job default/camera-frames-round0-measure submitted (sha $SHA, image ...:$TAG)"
EOF

cat <<EOF
ROLLBACK: kubectl -n default delete job camera-frames-round0-measure
          kubectl -n default delete configmap camera-frames-round0-measure-src camera-round0-pins
LOGS:     kubectl -n default logs -f job/camera-frames-round0-measure
EOF
