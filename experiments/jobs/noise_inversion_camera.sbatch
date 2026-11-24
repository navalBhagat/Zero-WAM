#!/bin/bash
#SBATCH --partition=gpu_a100
#SBATCH --job-name=noise_camera_test
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=18
#SBATCH --time=00:15:00
#SBATCH --output=noise_camera_%j.out
set -euo pipefail
export ZERO_WAM_ROOT="${ZERO_WAM_ROOT:-$HOME/Zero-WAM}"
export CAMERA_STRESS_TEST=1 CAMERA_LOAD_MODEL="${CAMERA_LOAD_MODEL:-0}"
export CAMERA_SHADER="${CAMERA_SHADER:-rt}"
case "$CAMERA_SHADER" in rt|default) ;; *) echo "CAMERA_SHADER must be rt or default" >&2; exit 1 ;; esac
export CAMERA_DENOISER="${CAMERA_DENOISER:-oidn}"
case "$CAMERA_DENOISER" in oidn|none) ;; *) echo "CAMERA_DENOISER must be oidn or none" >&2; exit 1 ;; esac
export NOISE_INVERSION_JOB_ROOT="$ZERO_WAM_ROOT/experiments/stages/noise_inversion_full/camera_tests/shader_${CAMERA_SHADER}_denoiser_${CAMERA_DENOISER}_model_${CAMERA_LOAD_MODEL}_job_${SLURM_JOB_ID}"
export TASK=move_stapler_pad RENDER_SAMPLES_PER_PIXEL=16
export USE_ICL=1 USE_TEXT=1 USE_GUIDANCE=0
export VIDEO_STATE_FP32=1 COMPACT_OBSERVATION_CACHE=1
export PHASE_SEPARATION=1 PHASE_OFFLOAD_TRANSFORMER=1
export RUN_DIAGNOSTICS=1 SERVER_STARTUP_TIMEOUT=240
exec bash "$ZERO_WAM_ROOT/experiments/jobs/eval.sbatch"
