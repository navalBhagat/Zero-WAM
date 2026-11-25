#!/bin/bash
#SBATCH --job-name=noise-inversion
#SBATCH --partition=defq
#SBATCH --gres=gpu:A6000:1
#SBATCH --cpus-per-task=4
#SBATCH --time=01:30:00
#SBATCH --output=noise-inversion-%j.log
set -euo pipefail
source "$HOME/nbhagat-workspace-env.sh"
: "${NOISE_INVERSION_JOB_ROOT:?Submit through noise_inversion_campaign.py}"
: "${NOISE_INVERSION_JOB_SETTINGS:?Missing grouped-job settings}"
: "${TASK:?Missing task in submission environment}"
export NOISE_INVERSION_PILOT=1
export VIDEO_STATE_FP32=1 COMPACT_OBSERVATION_CACHE=1
export PHASE_SEPARATION=1 PHASE_OFFLOAD_TRANSFORMER=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export USE_ICL=1 USE_TEXT=1 USE_GUIDANCE=0
export RUN_DIAGNOSTICS=1 PROFILE_TIMING=1 COUNT_POLICY_FLOPS=0 SAVE_GUIDANCE_CAPTURE=0
export TEXT_ENCODER_OFFLOAD=1
export RENDER_SAMPLES_PER_PIXEL=16
export SAMPLING_SEED="${SAMPLING_SEED:-1234}"
export NOISE_INVERSION_STEP_TIMEOUT=90 NOISE_INVERSION_PHASE_TIMEOUT=1800
export CLIENT_TIMEOUT="${CLIENT_TIMEOUT:-5100}"
unset CAMERA_STRESS_TEST NOISE_INVERSION_CONTINUE NOISE_INVERSION_REPAIR_PILOT NOISE_INVERSION_CONTROL_SOURCE
export EXPERIMENT_CLUSTER=das6
exec bash "$ZERO_WAM_ROOT/experiments/scripts/run_evaluation.sh"
