#!/usr/bin/env bash
# Convenience wrapper for EC2 training.
# Pins each worker to 1 BLAS thread so CPU cores aren't over-subscribed.
# --s3-bucket is captured here for the SIGTERM trap and passed through to train.py.
#
# Usage:
#   ./train.sh --config medium --workers 12 --experiment baseline
#   ./train.sh --config medium --workers 12 --s3-bucket my-bucket && sudo shutdown -h now
set -euo pipefail
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

S3_BUCKET=""
PYTHON_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --s3-bucket)
      S3_BUCKET="$2"
      PYTHON_ARGS+=("$1" "$2")
      shift 2
      ;;
    *)
      PYTHON_ARGS+=("$1")
      shift
      ;;
  esac
done

# Emergency upload on spot termination (SIGTERM).
# train.py handles SIGTERM gracefully — it finishes the current game and saves
# a checkpoint before exiting. Bash defers this trap until train.py exits, so
# by the time _s3_upload runs the checkpoint is already on disk.
_s3_upload() {
  if [[ -z "$S3_BUCKET" ]]; then return; fi
  echo "--- Emergency upload to s3://${S3_BUCKET} ---"
  aws s3 cp mlflow.db "s3://${S3_BUCKET}/mlflow.db" || true
  aws s3 sync runs/ "s3://${S3_BUCKET}/runs/" \
    --exclude "*" \
    --include "*/checkpoints/checkpoint_best.pt" \
    --include "*/checkpoints/checkpoint_best.json" || true
  echo "--- Upload complete ---"
}
trap '_s3_upload' TERM

python train.py "${PYTHON_ARGS[@]}"

if [[ -n "$S3_BUCKET" ]]; then
  echo "--- Uploading results to s3://${S3_BUCKET} ---"
  aws s3 cp mlflow.db "s3://${S3_BUCKET}/mlflow.db"
  aws s3 sync runs/ "s3://${S3_BUCKET}/runs/" \
    --exclude "*" \
    --include "*/checkpoints/checkpoint_best.pt" \
    --include "*/checkpoints/checkpoint_best.json"
  echo "--- Upload complete ---"
fi
