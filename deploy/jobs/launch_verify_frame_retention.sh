#!/usr/bin/env bash
# Install the daily retention verifier (design D7, task 4.3).
#
# Usage:
#   deploy/jobs/launch_verify_frame_retention.sh [--image-tag TAG] [--ref REV]
#                                                [--dry-run] [--run-now]
#
# The check cannot run from a bare `kubectl apply`: it mounts a ConfigMap that is
# not in the repo (the verifier script itself) and reads a Secret this script does
# not create. Both are checked here, because the alternative is a CronJob that
# exists, runs nightly, and fails for a reason nobody reads. The ConfigMap's key is
# checked against the file the CronJob's command runs: kubectl's --from-file=KEY=PATH
# names the key KEY, and a mismatch is invisible until the first scheduled run.
#
# The Secret `hcm-frame-retention-creds` holds MINIO_USER / MINIO_PASSWORD for a
# MinIO user allowed ListBucket and the lifecycle read on camera-uncertain-frames
# and nothing else -- deploy/minio/hcm-frame-retention-policy.json is that policy.
# The lead creates the user and the Secret; this script only checks they are there.
#
# Env: KUBE_HOST (default controlplane) -- kubectl lives there, not on this VM.
#      REPO      (default: this script's git toplevel)
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=${REPO:-$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel)}
KUBE_HOST=${KUBE_HOST:-controlplane}
# The same 1 MiB of key data a ConfigMap holds; this one is one small file, but the
# guard is here so a future edit that grows the check cannot fail silently either.
CM_LIMIT=1000000
SECRET=hcm-frame-retention-creds
CM_NAME=verify-frame-retention-src
# The key the ConfigMap holds, which must be the basename the CronJob's command runs
# (checked below, and again against the object kubectl would create). One value feeds
# the creation, the check and the dry run, so the three cannot drift apart.
CM_KEY=verify_frame_retention.py

REF=HEAD
TAG=a0cceab
DRY=0
RUN_NOW=0
JOB=

die() { echo "REFUSE $*" >&2; exit 2; }

while [[ $# -gt 0 ]]; do
  case "$1" in
    --image-tag) TAG=$2; shift 2 ;;
    --ref)       REF=$2; shift 2 ;;
    --dry-run)   DRY=1; shift ;;
    --run-now)   RUN_NOW=1; shift ;;
    -h|--help)   sed -n '2,/^set -euo/p' "${BASH_SOURCE[0]}" | sed '$d'; exit 0 ;;
    *)           die "unknown argument: $1" ;;
  esac
done

SHA=$(git -C "$REPO" rev-parse --short=7 "$REF^{commit}")
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT

# Both halves come from $SHA, not from the checkout: a working tree ahead of or
# behind --ref would install a pair nobody reviewed.
git -C "$REPO" show "$SHA:tools/verify_frame_retention.py" > "$TMP/$CM_KEY" \
  || die "tools/verify_frame_retention.py does not exist at $SHA"
BYTES=$(stat -c %s "$TMP/$CM_KEY")
[[ $BYTES -lt $CM_LIMIT ]] \
  || die "the verifier is $BYTES bytes, over the $CM_LIMIT-byte ConfigMap limit"
echo "src: $CM_KEY, $BYTES bytes (limit $CM_LIMIT)"

# The manifest is the record; the launcher supplies what varies -- the image tag.
git -C "$REPO" show "$SHA:deploy/jobs/verify_frame_retention_cronjob.yaml" > "$TMP/cronjob.in.yaml" \
  || die "deploy/jobs/verify_frame_retention_cronjob.yaml does not exist at $SHA"
sed -e "s|ecs_hcm_trainer:[A-Za-z0-9._-]*|ecs_hcm_trainer:$TAG|" \
    "$TMP/cronjob.in.yaml" > "$TMP/cronjob.yaml"
grep -q "ecs_hcm_trainer:$TAG" "$TMP/cronjob.yaml" || die "could not inject image tag $TAG"

# The wiring: the key the ConfigMap gets must be the file the command runs. There is
# no items: mapping on the mount, so a name that differs is a file the Job cannot
# find -- nightly, silently, and only once installed.
TARGET_PATH=$(grep -o '/opt/verify/[A-Za-z0-9._-]*' "$TMP/cronjob.yaml" | head -n1) \
  || true
[[ -n $TARGET_PATH ]] || die "the CronJob names no file under /opt/verify"
TARGET=${TARGET_PATH##*/}
[[ $CM_KEY == "$TARGET" ]] \
  || die "the ConfigMap key is '$CM_KEY' but the CronJob runs '$TARGET_PATH'"

# The Secret keys the Job reads, taken from the manifest rather than from a list here:
# a manifest that starts reading another key must not pass on a Secret that lacks it.
SECRET_REFS=$(grep -o 'secretKeyRef: *{[^}]*}' "$TMP/cronjob.yaml")
[[ -n $SECRET_REFS ]] || die "the CronJob reads no Secret; 4.3 runs as its own identity"
SECRET_NAMES=$(sed -e 's/.*name: *//' -e 's/,.*//' <<<"$SECRET_REFS" | sort -u | paste -sd,)
SECRET_KEYS=$(sed -e 's/.*key: *//' -e 's/}"*$//' <<<"$SECRET_REFS" | sort -u | paste -sd,)
[[ $SECRET_NAMES == "$SECRET" ]] \
  || die "the CronJob reads Secret '$SECRET_NAMES', but this installs '$SECRET'"

if [[ $RUN_NOW == 1 ]]; then
  # Unique per invocation: a retry after a failed first run must not die on
  # AlreadyExists once the CronJob has been applied.
  JOB="verify-frame-retention-manual-$(date -u +%Y%m%d%H%M%S)"
fi

if [[ $DRY == 1 ]]; then
  echo "--- dry run: nothing sent to $KUBE_HOST ---"
  echo "ConfigMap $CM_NAME  key $CM_KEY ($BYTES bytes) -> $TARGET_PATH"
  echo "Secret    $SECRET  keys ${SECRET_KEYS//,/ } (must exist; not created here)"
  [[ -n $JOB ]] && echo "Job       $JOB (one run, after the apply)"
  cat "$TMP/cronjob.yaml"
  exit 0
fi

scp -q "$TMP/$CM_KEY" "$TMP/cronjob.yaml" "$KUBE_HOST:/tmp/"

# Values reach the remote shell as environment assignments rather than through heredoc
# expansion, so the go-template below stays literal -- and every value is comma- or
# space-free, because ssh joins its arguments with spaces.
ssh "$KUBE_HOST" \
    CM_NAME="$CM_NAME" CM_KEY="$CM_KEY" SECRET="$SECRET" SECRET_KEYS="$SECRET_KEYS" \
    JOB="$JOB" SHA="$SHA" TAG="$TAG" RUN_NOW="$RUN_NOW" \
    bash -s <<'REMOTE'
set -euo pipefail
cd /tmp
trap 'rm -f "$CM_KEY" cronjob.yaml' EXIT

# The Secret must hold the keys the Job reads, not merely exist: a Secret with other
# keys installs cleanly and exits 2 every night. Key names only -- never the values.
if ! HELD=$(kubectl -n default get secret "$SECRET" \
      -o go-template='{{range $k, $v := .data}}{{$k}}{{"\n"}}{{end}}' 2>/dev/null); then
  echo "REFUSE the Secret default/$SECRET does not exist. The verifier needs a MinIO" >&2
  echo "       user with ListBucket + the lifecycle read on camera-uncertain-frames." >&2
  exit 2
fi
for key in ${SECRET_KEYS//,/ }; do
  if ! grep -qx "$key" <<<"$HELD"; then
    echo "REFUSE the Secret default/$SECRET has no key $key" >&2
    echo "       it holds: $(echo "$HELD" | tr '\n' ' ')" >&2
    exit 2
  fi
done

# The wiring, checked against the object kubectl would create rather than against the
# argument we meant to pass: the ConfigMap's keys must be exactly the file the
# CronJob's command runs. Nothing is applied until this passes.
WANT_PATH=$(grep -o '/opt/verify/[A-Za-z0-9._-]*' cronjob.yaml | head -n1)
WANT=${WANT_PATH##*/}
KEYS=$(kubectl -n default create configmap "$CM_NAME" \
         --from-file="$CM_KEY=$CM_KEY" --dry-run=client \
         -o go-template='{{range $k, $v := .data}}{{$k}}{{"\n"}}{{end}}' \
       | sort | tr '\n' ' ') \
  || { echo "REFUSE could not render the ConfigMap $CM_NAME" >&2; exit 2; }
if [[ "$KEYS" != "$WANT " ]]; then
  echo "REFUSE the ConfigMap $CM_NAME would hold [$KEYS] but the CronJob runs $WANT_PATH" >&2
  exit 2
fi

kubectl -n default create configmap "$CM_NAME" --from-file="$CM_KEY=$CM_KEY" \
  --dry-run=client -o yaml | kubectl apply -f - >/dev/null
kubectl -n default apply -f cronjob.yaml
echo "configmap default/$CM_NAME key $CM_KEY -> $WANT_PATH"

if [[ "$RUN_NOW" == 1 ]]; then
  # The install has already succeeded; a manual run that cannot be created is a
  # warning, not a failed install.
  if kubectl -n default create job --from=cronjob/verify-frame-retention "$JOB"; then
    echo "one run submitted: kubectl -n default logs job/$JOB"
  else
    echo "WARNING the CronJob is installed but the manual run was not created" >&2
  fi
fi
echo "cronjob default/verify-frame-retention applied (sha $SHA, image ...:$TAG)"
REMOTE

if [[ -n $JOB ]]; then
  LOGS="kubectl -n default logs job/$JOB"
else
  LOGS="kubectl -n default get jobs -l app=verify-frame-retention"
fi
cat <<EOF
ROLLBACK: kubectl -n default delete cronjob verify-frame-retention
          kubectl -n default delete configmap $CM_NAME
LOGS:     $LOGS
EOF
