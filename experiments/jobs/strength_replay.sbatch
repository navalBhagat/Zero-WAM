#!/bin/bash
#SBATCH --partition=gpu_a100
#SBATCH --job-name=guidance_strength_replay
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=18
#SBATCH --time=00:30:00
#SBATCH --output=strength_replay_%j.out

set -eo pipefail

module purge
module load 2024 Anaconda3/2024.06-1 CUDA/12.6.0

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate zerowam
export PATH="${CONDA_PREFIX:?}/bin:$PATH"
hash -r

export ZERO_WAM_ROOT="${ZERO_WAM_ROOT:-$SLURM_SUBMIT_DIR}"
export HF_HUB_OFFLINE=1
export PYTHONUNBUFFERED=1
export MPLBACKEND=Agg

cd "$ZERO_WAM_ROOT"

CAPTURE_RUN="${CAPTURE_RUN:-$ZERO_WAM_ROOT/experiments/stages/03_temporal_guidance_same_seed/runs/place_object_scale_29419404}"
MODEL_PATH="${MODEL_PATH:-$ZERO_WAM_ROOT/checkpoints/zero-wam-posttrain-robotwin}"
GUIDANCE_CHECKPOINT="${GUIDANCE_CHECKPOINT:-$ZERO_WAM_ROOT/checkpoints/dinov2-small}"
OUTPUT="$ZERO_WAM_ROOT/experiments/strength_replay/job_${SLURM_JOB_ID}"

# Arguments after the script path are strengths, e.g. sbatch ... 0 0.1 0.3 1.0.
# Keep the environment override for existing callers; explicit arguments win.
if (( $# > 0 )); then
    REPLAY_STRENGTHS=("$@")
else
    read -r -a REPLAY_STRENGTHS <<< "${STRENGTHS:-0 0.01 0.03 0.1}"
fi

python experiments/scripts/precision_replay.py \
    --run "$CAPTURE_RUN" \
    --model "$MODEL_PATH" \
    --dino "$GUIDANCE_CHECKPOINT" \
    --decoder-dtypes float32 \
    --strengths "${REPLAY_STRENGTHS[@]}" \
    --repeats 1 \
    --output "$OUTPUT"
