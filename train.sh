#!/usr/bin/env bash
# Convenience wrapper for EC2 training.
# Pins each worker to 1 BLAS thread so CPU cores aren't over-subscribed.
# Polls the AWS spot 2-minute termination warning so train.py has time to
# save an emergency checkpoint before the instance is killed.
#
# --s3-bucket  captured here for uploads and passed through to train.py.
# --shutdown   halt the machine when training completes (for direct invocations;
#              ec2_manager.py handles shutdown via the script it writes on EC2).
#
# Usage:
#   ./train.sh --config medium --workers 12 --experiment baseline
#   ./train.sh --config full --workers 14 --s3-bucket my-bucket --shutdown
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

# Upload checkpoints and MLflow DB — called on termination and on normal completion.
_s3_upload() {
  if [[ -z "$S3_BUCKET" ]]; then return; fi
  echo "--- Uploading to s3://${S3_BUCKET} ---"
  aws s3 cp mlflow.db "s3://${S3_BUCKET}/mlflow-${_EFFECTIVE_NAME}.db" || true
  aws s3 sync runs/ "s3://${S3_BUCKET}/runs/" \
    --exclude "*" \
    --include "*/checkpoints/checkpoint_latest.pt" \
    --include "*/checkpoints/checkpoint_latest.json" \
    --include "*/checkpoints/checkpoint_best.pt" \
    --include "*/checkpoints/checkpoint_best.json" || true
  echo "--- Upload complete ---"
}

# Run training in the background so the spot watcher can run concurrently.
python train.py "${PYTHON_ARGS[@]}" &
TRAIN_PID=$!

# Background watcher: polls the 2-minute AWS spot termination metadata endpoint.
# When the notice appears, SIGTERM is sent to this script (which then forwards
# to train.py), giving Python ~2 minutes to finish its current game and write
# an emergency checkpoint before the instance is killed.
_spot_watcher() {
  while kill -0 "$TRAIN_PID" 2>/dev/null; do
    if curl -sf --max-time 2 \
        http://169.254.169.254/latest/meta-data/spot/termination-time \
        >/dev/null 2>&1; then
      echo "$(date -u +%H:%M:%S) Spot termination warning -- triggering graceful shutdown"
      kill -TERM $$ 2>/dev/null || true
      exit 0
    fi
    sleep 5
  done
}
_spot_watcher &
WATCHER_PID=$!

# SIGTERM handler: forward to train.py, wait for it to save its emergency
# checkpoint, then do a belt-and-suspenders S3 upload.
_on_term() {
  trap '' TERM  # ignore further TERMs while we're shutting down
  echo "--- SIGTERM: forwarding to training (PID ${TRAIN_PID}) ---"
  kill -TERM "$TRAIN_PID" 2>/dev/null || true
  wait "$TRAIN_PID" 2>/dev/null || true
  kill "$WATCHER_PID" 2>/dev/null || true
  _s3_upload
  exit 0
}
trap '_on_term' TERM

# Wait for training to complete normally.
wait "$TRAIN_PID" || true
kill "$WATCHER_PID" 2>/dev/null || true

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
