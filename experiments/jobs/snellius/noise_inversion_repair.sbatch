#!/bin/bash
#SBATCH --partition=gpu_a100
#SBATCH --job-name=noise_repair_pilot
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=18
#SBATCH --time=00:45:00
#SBATCH --output=noise_repair_%j.out
set -euo pipefail
export ZERO_WAM_ROOT="${ZERO_WAM_ROOT:-$HOME/Zero-WAM}"
export NOISE_INVERSION_REPAIR_SOURCE="${NOISE_INVERSION_REPAIR_SOURCE:-$ZERO_WAM_ROOT/experiments/stages/noise_inversion_full}"
export NOISE_INVERSION_JOB_ROOT="$NOISE_INVERSION_REPAIR_SOURCE/repair_jobs/job_${SLURM_JOB_ID}"
export NOISE_INVERSION_REPAIR_PILOT=1
export NOISE_INVERSION_PILOT=1
export PHASE_SEPARATION=1
export PHASE_OFFLOAD_TRANSFORMER=1
export VIDEO_STATE_FP32=1
export COMPACT_OBSERVATION_CACHE=1
export NOISE_INVERSION_STEP_TIMEOUT=90
export NOISE_INVERSION_PHASE_TIMEOUT="${NOISE_INVERSION_PHASE_TIMEOUT:-1200}"
export USE_ICL=1 USE_TEXT=1 USE_GUIDANCE=0
export COUNT_POLICY_FLOPS=0 RUN_DIAGNOSTICS=1 PROFILE_TIMING=1
export SAVE_GUIDANCE_CAPTURE=0 TEXT_ENCODER_OFFLOAD=1
export TASK=move_stapler_pad SAMPLING_SEED=1234
export RENDER_SAMPLES_PER_PIXEL=16
export CLIENT_TIMEOUT="${CLIENT_TIMEOUT:-2400}"
# A dedicated capture-only pilot, not a submission against the old manifest.
unset NOISE_INVERSION_JOB_SETTINGS
export EXPERIMENT_SOURCE_DIGEST
EXPERIMENT_SOURCE_DIGEST=$(cd "$ZERO_WAM_ROOT" && python3 -c 'from experiments.campaign.provenance import fingerprint; print(fingerprint()[0])')
exec bash "$ZERO_WAM_ROOT/experiments/jobs/snellius/eval.sbatch"
