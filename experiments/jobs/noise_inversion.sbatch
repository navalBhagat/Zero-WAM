#!/bin/bash
#SBATCH --partition=gpu_a100
#SBATCH --job-name=noise_inversion
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=18
#SBATCH --time=03:00:00
#SBATCH --output=noise_inversion_%j.out
set -euo pipefail
export NOISE_INVERSION_PILOT=1
export VIDEO_STATE_FP32=1
export USE_GUIDANCE=0
export USE_TEXT=1
export USE_ICL=1
export TASK=place_bread_basket
export PHASE_SEPARATION="${PHASE_SEPARATION:-0}"
export COMPACT_OBSERVATION_CACHE="${COMPACT_OBSERVATION_CACHE:-1}"
export SAMPLING_SEED="${SAMPLING_SEED:-1234}"
export RENDER_SAMPLES_PER_PIXEL="${RENDER_SAMPLES_PER_PIXEL:-16}"
export CLIENT_TIMEOUT="${CLIENT_TIMEOUT:-9000}"
# Launch the existing environment/server lifecycle; the pilot runs both tasks.
bash "$(dirname "${BASH_SOURCE[0]}")/eval.sbatch"
