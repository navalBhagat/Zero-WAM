#!/bin/bash
#SBATCH --job-name=noiseicl
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --cpus-per-task=18
#SBATCH --time=03:00:00
#SBATCH --output=noiseicl_%j.log
set -eo pipefail
export ZERO_WAM_ROOT="${ZERO_WAM_ROOT:-$HOME/Zero-WAM}"
export EXPERIMENT_CLUSTER=snellius
exec bash "$ZERO_WAM_ROOT/experiments/scripts/run_evaluation.sh"
