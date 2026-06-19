#!/usr/bin/env bash
# Convenience wrapper for EC2 training.
# Pins each worker to 1 BLAS thread so CPU cores aren't over-subscribed.
#
# --s3-bucket  captured here for the SIGTERM trap and passed through to train.py.
# --shutdown   halt the machine when training completes (for direct invocations;
#              ec2_manager.py handles shutdown via the script it writes on EC2).
#
# Usage:
#   ./train.sh --config medium --workers 12 --experiment baseline
#   ./train.sh --config full --workers 28 --s3-bucket my-bucket --shutdown
set -euo pipefail
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

S3_BUCKET=""
SHUTDOWN_AFTER=0
PYTHON_ARGS=()

# Track the effective config.name for correctly namespacing the S3 MLflow DB.
# --name overrides --config; matches the logic in train.py and trainer.py.
_CONFIG_NAME="dev"
_RUN_NAME=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --s3-bucket)
      S3_BUCKET="$2"
      PYTHON_ARGS+=("$1" "$2")
      shift 2
      ;;
    --shutdown)
      SHUTDOWN_AFTER=1
      shift
      ;;
    --config)
      _CONFIG_NAME="$2"
      PYTHON_ARGS+=("$1" "$2")
      shift 2
      ;;
    --name)
      _RUN_NAME="$2"
      PYTHON_ARGS+=("$1" "$2")
      shift 2
      ;;
    *)
      PYTHON_ARGS+=("$1")
      shift
      ;;
  esac
done

# The trainer uploads mlflow.db to s3://{bucket}/mlflow-{config.name}.db.
# Replicate that naming here so the emergency upload uses the same key.
_EFFECTIVE_NAME="${_RUN_NAME:-$_CONFIG_NAME}"

# Emergency upload on spot termination (SIGTERM).
# train.py handles SIGTERM gracefully — it finishes the current game and saves
# a checkpoint before exiting. Bash defers this trap until train.py exits, so
# by the time _s3_upload runs the checkpoint is already on disk.
_s3_upload() {
  if [[ -z "$S3_BUCKET" ]]; then return; fi
  echo "--- Emergency upload to s3://${S3_BUCKET} ---"
  aws s3 cp mlflow.db "s3://${S3_BUCKET}/mlflow-${_EFFECTIVE_NAME}.db" || true
  aws s3 sync runs/ "s3://${S3_BUCKET}/runs/" \
    --exclude "*" \
    --include "*/checkpoints/checkpoint_latest.pt" \
    --include "*/checkpoints/checkpoint_latest.json" \
    --include "*/checkpoints/checkpoint_best.pt" \
    --include "*/checkpoints/checkpoint_best.json" || true
  echo "--- Upload complete ---"
}
trap '_s3_upload' TERM

python train.py "${PYTHON_ARGS[@]}"

if [[ -n "$S3_BUCKET" ]]; then
  echo "--- Uploading results to s3://${S3_BUCKET} ---"
  aws s3 cp mlflow.db "s3://${S3_BUCKET}/mlflow-${_EFFECTIVE_NAME}.db"
  aws s3 sync runs/ "s3://${S3_BUCKET}/runs/" \
    --exclude "*" \
    --include "*/checkpoints/checkpoint_best.pt" \
    --include "*/checkpoints/checkpoint_best.json"
  echo "--- Upload complete ---"
fi

if [[ "$SHUTDOWN_AFTER" -eq 1 ]]; then
  echo "--- Shutting down ---"
  sudo shutdown -h now
fi
