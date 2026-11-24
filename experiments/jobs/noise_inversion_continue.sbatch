#!/bin/bash
#SBATCH --partition=gpu_a100
#SBATCH --job-name=noise_stack_continue
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=18
#SBATCH --time=01:30:00
#SBATCH --output=noise_continue_%j.out
set -euo pipefail
export ZERO_WAM_ROOT="${ZERO_WAM_ROOT:-$HOME/Zero-WAM}"
export NOISE_INVERSION_REPAIR_SOURCE="${NOISE_INVERSION_REPAIR_SOURCE:-$ZERO_WAM_ROOT/experiments/stages/noise_inversion_full}"
export NOISE_INVERSION_CAPTURE_SNAPSHOT="$NOISE_INVERSION_REPAIR_SOURCE/repair_jobs/job_29648542/repair_source.zip"
export NOISE_INVERSION_JOB_ROOT="$NOISE_INVERSION_REPAIR_SOURCE/continuation_jobs/job_${SLURM_JOB_ID}"
export NOISE_INVERSION_CONTINUE=1
# Match the completed source's serving settings; rendering remains the baseline.
export TASK=stack_blocks_three SAMPLING_SEED=1234
export PHASE_SEPARATION=1 PHASE_OFFLOAD_TRANSFORMER=1 COMPACT_OBSERVATION_CACHE=1
export VIDEO_STATE_FP32=1 TEXT_ENCODER_OFFLOAD=1 RENDER_SAMPLES_PER_PIXEL=16
export USE_ICL=1 USE_TEXT=1 USE_GUIDANCE=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NOISE_INVERSION_STEP_TIMEOUT=90 NOISE_INVERSION_PHASE_TIMEOUT=1800
export CLIENT_TIMEOUT=5100 RUN_DIAGNOSTICS=1 PROFILE_TIMING=1 COUNT_POLICY_FLOPS=0 SAVE_GUIDANCE_CAPTURE=0
unset CAMERA_STRESS_TEST CAMERA_SHADER CAMERA_DENOISER NOISE_INVERSION_REPAIR_PILOT NOISE_INVERSION_PILOT
export EXPERIMENT_SOURCE_DIGEST
EXPERIMENT_SOURCE_DIGEST=$(cd "$ZERO_WAM_ROOT" && python3 -c 'from experiments.campaign.provenance import fingerprint; print(fingerprint()[0])')
exec bash "$ZERO_WAM_ROOT/experiments/jobs/eval.sbatch"
