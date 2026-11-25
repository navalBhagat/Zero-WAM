#!/bin/bash
#SBATCH --job-name=noiseicl
#SBATCH --partition=defq
#SBATCH --gres=gpu:A6000:1
#SBATCH --cpus-per-task=4
#SBATCH --time=03:00:00
#SBATCH --output=noiseicl_%j.log
set -eo pipefail
source "$HOME/nbhagat-workspace-env.sh"
export EXPERIMENT_CLUSTER=das6
exec bash "$ZERO_WAM_ROOT/experiments/scripts/run_evaluation.sh"
