#!/bin/bash
#SBATCH --partition=gpu_a100
#SBATCH --job-name=noise_control
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=18
#SBATCH --time=01:00:00
#SBATCH --output=noise_control_%j.out
set -euo pipefail
export ZERO_WAM_ROOT="${ZERO_WAM_ROOT:-$HOME/Zero-WAM}"
export NOISE_INVERSION_CONTROL_SOURCE="${NOISE_INVERSION_CONTROL_SOURCE:-$ZERO_WAM_ROOT/experiments/stages/noise_inversion_pilot/runs/place_bread_basket_29492432/noise_inversion}"
for task in place_bread_basket place_object_scale; do
    if [[ ! -f "$NOISE_INVERSION_CONTROL_SOURCE/$task/capture/config.json" ||
          ! -f "$NOISE_INVERSION_CONTROL_SOURCE/$task/summary.json" ||
          ! -f "$NOISE_INVERSION_CONTROL_SOURCE/$task/chunks/chunk_0000.pt" ]]; then
        echo "Missing pilot artifacts for $task in $NOISE_INVERSION_CONTROL_SOURCE" >&2
        exit 1
    fi
done
echo "Appending controls to: $NOISE_INVERSION_CONTROL_SOURCE"
export NOISE_INVERSION_PILOT=1
export VIDEO_STATE_FP32=1
export PHASE_SEPARATION=0
export COMPACT_OBSERVATION_CACHE=1
export USE_ICL=0
export USE_TEXT=0
export USE_GUIDANCE=0
export TASK=place_bread_basket
export SAMPLING_SEED=1234
export RENDER_SAMPLES_PER_PIXEL=16
export CLIENT_TIMEOUT=3000
EVAL_JOB="$ZERO_WAM_ROOT/experiments/jobs/eval.sbatch"
if [[ ! -f "$EVAL_JOB" ]]; then
    echo "Submit from the repository root or set ZERO_WAM_ROOT." >&2
    exit 1
fi
exec bash "$EVAL_JOB"
