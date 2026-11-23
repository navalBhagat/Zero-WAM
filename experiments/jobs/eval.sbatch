#!/bin/bash
#SBATCH --partition=gpu_a100
#SBATCH --job-name=zerowam_smoke
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=18
#SBATCH --time=01:00:00
#SBATCH --output=zerowam_smoke_%j.out

set -eo pipefail
JOB_STARTED=$(date +%s)
SERVER_STARTED=0
SERVER_READY=0
CLIENT_STARTED=0
CLIENT_FINISHED=0

module purge
module load 2024
module load Anaconda3/2024.06-1
module load CUDA/12.6.0
source "$(conda info --base)/etc/profile.d/conda.sh"

export ZERO_WAM_ROOT="${ZERO_WAM_ROOT:-$HOME/Zero-WAM}"
if [[ -n "${EXPERIMENT_SOURCE_DIGEST:-}" ]]; then
    python3 "$ZERO_WAM_ROOT/experiments/scripts/provenance.py" --verify "$EXPERIMENT_SOURCE_DIGEST"
fi
export ROBOTWIN_ROOT="${ROBOTWIN_ROOT:-$HOME/Robotwin}"
export MODEL_PATH="${MODEL_PATH:-$ZERO_WAM_ROOT/checkpoints/zero-wam-posttrain-robotwin}"
export ICL_LATENT_ROOT="${ICL_LATENT_ROOT:-$ZERO_WAM_ROOT/data/HumanGen/human_latents/robotwin}"
export TEST_NUM="${TEST_NUM:-1}"
export SEED="${SEED:-0}"
export USE_ICL="${USE_ICL:-1}"
export ICL_GUIDANCE_SCALE="${ICL_GUIDANCE_SCALE:-5}"
export USE_TEXT="${USE_TEXT:-0}"
export USE_GUIDANCE="${USE_GUIDANCE:-0}"
export TEXT_GUIDANCE_SCALE="${TEXT_GUIDANCE_SCALE:-1}"
export GUIDANCE_STRENGTH="${GUIDANCE_STRENGTH:-0.01}"
export GUIDANCE_EVERY="${GUIDANCE_EVERY:-10}"
export GUIDANCE_MAX_NORM="${GUIDANCE_MAX_NORM:-1}"
export GUIDANCE_CHECKPOINT="${GUIDANCE_CHECKPOINT:-$ZERO_WAM_ROOT/checkpoints/dinov2-small}"
export GUIDANCE_REFERENCE_SOURCE="${GUIDANCE_REFERENCE_SOURCE:-auto}"
export GUIDANCE_DECODER_DTYPE="${GUIDANCE_DECODER_DTYPE:-float32}"
export VIDEO_STATE_FP32="${VIDEO_STATE_FP32:-1}"
export PROFILE_TIMING="${PROFILE_TIMING:-1}"
export COUNT_POLICY_FLOPS="${COUNT_POLICY_FLOPS:-0}"
export RUN_DIAGNOSTICS="${RUN_DIAGNOSTICS:-0}"
export SAVE_GUIDANCE_CAPTURE="${SAVE_GUIDANCE_CAPTURE:-0}"
export PROFILE_SYNC_GPU="${PROFILE_SYNC_GPU:-0}"
export MONITOR_GPU="${MONITOR_GPU:-1}"
export RENDER_SAMPLES_PER_PIXEL="${RENDER_SAMPLES_PER_PIXEL:-32}"
export GUIDANCE_REFERENCE_FRAMES="${GUIDANCE_REFERENCE_FRAMES:-0}"
case "$MONITOR_GPU" in
    0|1) ;;
    *) echo "MONITOR_GPU must be 0 or 1" >&2; exit 1 ;;
esac
# Keep only the text encoder on CPU; VAE and transformer remain on GPU.
export TEXT_ENCODER_OFFLOAD="${TEXT_ENCODER_OFFLOAD:-1}"
# Reserve adjacent pairs: consecutive jobs must not overlap server/rendezvous ports.
export PORT=$((20000 + 2 * (SLURM_JOB_ID % 10000)))
export MASTER_PORT=$((PORT + 1))
export PYTHONUNBUFFERED=1
export ZERO_WAM_DIAGNOSTICS=1
export PYTHONFAULTHANDLER=1
# Diagnostic cap after server startup; timeout exits nonzero and cleanup runs.
CLIENT_TIMEOUT="${CLIENT_TIMEOUT:-900}"
export MPLBACKEND=Agg
export OMP_NUM_THREADS=4
export NO_PROXY="localhost,127.0.0.1,0.0.0.0,${NO_PROXY:-}"
export no_proxy="$NO_PROXY"

TASK="${TASK:-place_empty_cup}"
case "$TASK" in
    place_empty_cup|place_object_scale|stamp_seal|open_microwave|move_stapler_pad|place_bread_basket|stack_blocks_three) ;;
    *) echo "Not one of the seven unseen tasks: $TASK"; exit 1 ;;
esac
RUN_DIR="${NOISE_INVERSION_JOB_ROOT:-${EXPERIMENT_OUTPUT_ROOT:-$ZERO_WAM_ROOT/experiments/runs}/${TASK}_${SLURM_JOB_ID}}"
if [[ -n "${NOISE_INVERSION_JOB_ROOT:-}" && -d "$RUN_DIR/diagnostics" ]]; then
    mkdir -p "$RUN_DIR/attempts/job_${SLURM_JOB_ID}"
    mv "$RUN_DIR/diagnostics" "$RUN_DIR/attempts/job_${SLURM_JOB_ID}/diagnostics"
fi
mkdir -p "$RUN_DIR/diagnostics"
# ICL serving does not write latent dumps. Reuse the run root rather than
# creating an empty generated/ directory.
export SAVE_ROOT="$RUN_DIR"
# One replayable actual prediction, with its selected reference metadata.
if [[ "${USE_GUIDANCE}" == "1" && "${SAVE_GUIDANCE_CAPTURE}" == "1" ]]; then
    export GUIDANCE_CAPTURE="$RUN_DIR/diagnostics/capture_step.pt"
else
    unset GUIDANCE_CAPTURE
fi
export TASK
cd "$ZERO_WAM_ROOT"
python3 -m evaluation.robotwin.run_metadata "$RUN_DIR/config.json"
if [[ "$RUN_DIAGNOSTICS" == "1" ]]; then python3 -m pip freeze > "$RUN_DIR/diagnostics/bootstrap_packages.txt"; fi
# Record memory/utilization throughout setup, model loading, and evaluation.
MONITOR_PID=""
SERVER_PID=""
cleanup() {
    local status=$?
    trap - EXIT
    if [[ -n "$SERVER_PID" ]]; then
        kill "$SERVER_PID" 2>/dev/null || true
        wait "$SERVER_PID" 2>/dev/null || true
    fi
    if [[ -n "$MONITOR_PID" ]]; then
        kill "$MONITOR_PID" 2>/dev/null || true
        wait "$MONITOR_PID" 2>/dev/null || true
    fi
    # Write partial timing even when startup/client fails. Timing writer uses stdlib only.
    python3 "$ZERO_WAM_ROOT/experiments/scripts/job_report.py" timing "$RUN_DIR/diagnostics/job_timing.json" \
      "$status" "$JOB_STARTED" "$SERVER_STARTED" "$SERVER_READY" \
      "$CLIENT_STARTED" "$CLIENT_FINISHED" "$(date +%s)" || true
    python3 -m evaluation.robotwin.run_metadata --refresh "$RUN_DIR/config.json" || true
    exit "$status"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

test -d "$MODEL_PATH"
if [[ "$USE_ICL" == "1" || "$USE_GUIDANCE" == "1" ]]; then
    test -d "$ICL_LATENT_ROOT"
fi
test -f "$ROBOTWIN_ROOT/policy/ACT/deploy_policy.yml"
command -v curl >/dev/null
nvidia-smi
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv > "$RUN_DIR/diagnostics/hardware.csv"


if [[ "$MONITOR_GPU" == "1" ]]; then
    nvidia-smi --query-gpu=timestamp,index,memory.used,memory.total,utilization.gpu --format=csv -l 2 \
        > "$RUN_DIR/diagnostics/gpu.csv" &
    MONITOR_PID=$!
fi


# These released task overrides are irrelevant to scale/bread evaluations.
# Serialize patching when explicitly evaluating one of the overridden tasks.
case "$TASK" in
    move_stapler_pad|stamp_seal)
        (
            flock -x 9
            cp -n "$ROBOTWIN_ROOT/envs/$TASK.py" "$ROBOTWIN_ROOT/envs/$TASK.py.upstream"
            if ! cmp -s "evaluation/robotwin/envs/$TASK.py" "$ROBOTWIN_ROOT/envs/$TASK.py"; then
                cp "evaluation/robotwin/envs/$TASK.py" "$ROBOTWIN_ROOT/envs/$TASK.py"
            fi
        ) 9>"$ROBOTWIN_ROOT/.zerowam-task-patches.lock"
        ;;
esac

# Fail early on client imports/render initialization, before loading the WAM.
source activate robotwin
# Module Python can precede an inherited active Conda environment.
export PATH="${CONDA_PREFIX:?Environment activation failed}/bin:$PATH"
hash -r
if [[ "$RUN_DIAGNOSTICS" == "1" ]]; then python -m pip freeze > "$RUN_DIR/diagnostics/client_packages.txt"; fi
# Show the interpreter selected inside the compute job before client imports.
echo "CONDA_PREFIX=${CONDA_PREFIX:-unset}"
type -a python
python -c "import sys; print('EXECUTABLE:', sys.executable, flush=True); print('PREFIX:', sys.prefix, flush=True)"
export PYTHONPATH="$ZERO_WAM_ROOT:$ROBOTWIN_ROOT:${PYTHONPATH:-}"
python experiments/scripts/job_report.py preflight

run_camera_checks() {
    local failed=0
    local camera_task camera_seed
    for camera_task in move_stapler_pad place_object_scale; do
        camera_seed=100002
        if [[ "$camera_task" == "place_object_scale" ]]; then camera_seed=100006; fi
        mkdir -p "$RUN_DIR/$camera_task"
        timeout --kill-after=15s 300s python setup/robotwin/check_camera.py \
            --task "$camera_task" --seed "$camera_seed" --frames 200 --timeout 280 \
            --output "$RUN_DIR/$camera_task" > "$RUN_DIR/$camera_task/client.log" 2>&1 || failed=1
    done
    return "$failed"
}
if [[ "${CAMERA_STRESS_TEST:-0}" == "1" && "${CAMERA_LOAD_MODEL:-0}" == "0" ]]; then
    run_camera_checks
    exit 0
fi

SERVER_STARTED=$(date +%s)
# Separate subprocess environments, sharing the same allocated GPU and node.
(
    source activate zerowam
    # Module Python can precede an inherited active Conda environment.
    export PATH="${CONDA_PREFIX:?Environment activation failed}/bin:$PATH"
    hash -r
    if [[ "$RUN_DIAGNOSTICS" == "1" ]]; then python -m pip freeze > "$RUN_DIR/diagnostics/server_packages.txt"; fi
    exec bash evaluation/robotwin/launch_server.sh
) > "$RUN_DIR/diagnostics/server.log" 2>&1 &
SERVER_PID=$!

deadline=$((SECONDS + ${SERVER_STARTUP_TIMEOUT:-1800}))
until curl --noproxy '*' --fail --silent --max-time 2 "http://127.0.0.1:$PORT/healthz" >/dev/null; do
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
        echo "Server exited. See $RUN_DIR/diagnostics/server.log"
        tail -n 40 "$RUN_DIR/diagnostics/server.log"
        exit 1
    fi
    if (( SECONDS >= deadline )); then
        echo "Server startup timed out. See $RUN_DIR/diagnostics/server.log"
        exit 1
    fi
    sleep 5
done

SERVER_READY=$(date +%s)
CLIENT_STARTED=$(date +%s)
if [[ "${CAMERA_STRESS_TEST:-0}" == "1" ]]; then
    run_camera_checks
    CLIENT_FINISHED=$(date +%s)
    exit 0
fi
if [[ "${NOISE_INVERSION_PILOT:-0}" == "1" ]]; then
    if [[ "${NOISE_INVERSION_REPAIR_PILOT:-0}" == "1" ]]; then
        timeout --kill-after=15s "${CLIENT_TIMEOUT}s" python experiments/scripts/noise_inversion_repair.py \
            --campaign "$NOISE_INVERSION_REPAIR_SOURCE" --job-output "$RUN_DIR"
        CLIENT_FINISHED=$(date +%s)
        exit 0
    elif [[ -n "${NOISE_INVERSION_JOB_SETTINGS:-}" ]]; then
        retry_args=()
        if [[ "${NOISE_INVERSION_RETRY:-0}" == "1" ]]; then retry_args+=(--retry-incomplete); fi
        timeout --kill-after=15s "${CLIENT_TIMEOUT}s" python experiments/scripts/noise_inversion_campaign.py run \
            --directory "$RUN_DIR" --settings "$NOISE_INVERSION_JOB_SETTINGS" \
            --digest "$NOISE_INVERSION_SOURCE_DIGEST" "${retry_args[@]}"
        CLIENT_FINISHED=$(date +%s)
        exit 0
    elif [[ -n "${NOISE_INVERSION_CONTROL_SOURCE:-}" ]]; then
        pilot_args=(--control-source "$NOISE_INVERSION_CONTROL_SOURCE")
    else
        pilot_args=(--output "$RUN_DIR/noise_inversion" --sampling-seed "$SAMPLING_SEED" --simulator-seed "$SEED")
    fi
    timeout --kill-after=15s "${CLIENT_TIMEOUT}s" python experiments/scripts/noise_inversion_pilot.py "${pilot_args[@]}"
    CLIENT_FINISHED=$(date +%s)
    exit 0
fi

echo "Server ready. Evaluating $TASK for $TEST_NUM episode(s). Results: $RUN_DIR"
timeout --kill-after=15s "${CLIENT_TIMEOUT}s" bash evaluation/robotwin/launch_client.sh "$RUN_DIR" "$TASK" \
    2>&1 | tee "$RUN_DIR/diagnostics/client.log"

CLIENT_FINISHED=$(date +%s)

# Require the requested number of completed episodes, not just a partial file.
python experiments/scripts/job_report.py complete "$RUN_DIR" "$TEST_NUM"
echo "Evaluation finished. Results: $RUN_DIR"
