#!/bin/bash
#SBATCH --job-name=zerowam-eval
#SBATCH --partition=defq
#SBATCH --gres=gpu:A6000:1
#SBATCH --cpus-per-task=4
#SBATCH --time=01:30:00
#SBATCH --output=zerowam-eval-%j.log
set -eo pipefail
source "$HOME/nbhagat-workspace-env.sh"
export EXPERIMENT_CLUSTER=das6
exec bash "${ZERO_WAM_ROOT:?Workspace helper must set ZERO_WAM_ROOT}/experiments/scripts/run_evaluation.sh"
