#!/bin/bash
#SBATCH --partition=gpu_a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=18
set -euo pipefail
: "${ZERO_WAM_ROOT:?Submit through noise_inversion_campaign.py}"
: "${NOISE_INVERSION_JOB_ROOT:?Missing grouped-job output directory}"
: "${NOISE_INVERSION_JOB_SETTINGS:?Missing grouped-job settings}"
export NOISE_INVERSION_PILOT=1
export VIDEO_STATE_FP32=1
export PHASE_SEPARATION=0
export COMPACT_OBSERVATION_CACHE=1
export USE_ICL=1
export USE_TEXT=1
export USE_GUIDANCE=0
export COUNT_POLICY_FLOPS=0
export RUN_DIAGNOSTICS=0
export SAVE_GUIDANCE_CAPTURE=0
export TEXT_ENCODER_OFFLOAD=1
export PROFILE_TIMING=1
export SAMPLING_SEED=1234
export CLIENT_TIMEOUT="${CLIENT_TIMEOUT:-10500}"
# eval.sbatch uses TASK for its task-specific patch/preflight setup.
: "${TASK:?Missing task in submission environment}"
export TASK
export MODEL_PATH="${MODEL_PATH:-$ZERO_WAM_ROOT/checkpoints/zero-wam-posttrain-robotwin}"
exec bash "$ZERO_WAM_ROOT/experiments/jobs/eval.sbatch"
