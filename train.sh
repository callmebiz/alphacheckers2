#!/usr/bin/env bash
# Convenience wrapper for EC2 training.
# Pins each worker to 1 BLAS thread so CPU cores aren't over-subscribed.
# All extra args are forwarded to train.py, e.g.:
#   ./train.sh --config medium --workers 12 --experiment baseline
#   ./train.sh --config medium --workers 12 --sims 800 --iters 300
#   ./train.sh --config medium --workers 12 --resume --experiment baseline && sudo shutdown -h now
set -euo pipefail
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
python train.py "$@"
