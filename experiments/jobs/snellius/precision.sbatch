#!/bin/bash
#SBATCH --partition=gpu_a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=18
#SBATCH --time=00:30:00
#SBATCH --output=precision_%j.out
set -eo pipefail
module purge
module load 2024 Anaconda3/2024.06-1 CUDA/12.6.0
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate zerowam
export PATH="${CONDA_PREFIX:?}/bin:$PATH"
hash -r
: "${ZERO_WAM_ROOT:?Set ZERO_WAM_ROOT}"
: "${CAPTURE_RUN:?Set CAPTURE_RUN to a stage-3 run folder}"
: "${GUIDANCE_CHECKPOINT:?Set local DINO path}"
export PYTHONPATH="$ZERO_WAM_ROOT:${PYTHONPATH:-}"
export HF_HUB_OFFLINE=1
output="$ZERO_WAM_ROOT/experiments/precision/replay_${SLURM_JOB_ID}"
args=()
if [[ -n "${REFERENCE_LATENT:-}" ]]; then args+=(--reference "$REFERENCE_LATENT"); fi
if [[ -n "${GUIDANCE_REFERENCE_FRAMES_OVERRIDE:-}" ]]; then args+=(--frames "$GUIDANCE_REFERENCE_FRAMES_OVERRIDE"); fi
python "$ZERO_WAM_ROOT/experiments/scripts/precision_replay.py" \
 --run "$CAPTURE_RUN" --model "${MODEL_PATH:-$ZERO_WAM_ROOT/checkpoints/zero-wam-posttrain-robotwin}" \
 --dino "$GUIDANCE_CHECKPOINT" --output "$output" "${args[@]}"

python -m pip freeze > "$output/packages.txt"
nvidia-smi -q > "$output/gpu_hardware.txt"
git -C "$ZERO_WAM_ROOT" rev-parse HEAD > "$output/revision.txt"
