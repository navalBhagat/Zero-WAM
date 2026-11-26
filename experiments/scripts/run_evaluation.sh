#!/usr/bin/env bash
# Shared lifecycle; scientific methods live in wan_va/noise_inversion/.
set -eo pipefail
: "${ZERO_WAM_ROOT:?Set the repository root}"
: "${NOISE_CAMPAIGN:?Submit a planned NoiseICL campaign}"
case "${EXPERIMENT_CLUSTER:-}" in das6|snellius) ;; *) echo "Unknown cluster" >&2; exit 1 ;; esac
source "$ZERO_WAM_ROOT/experiments/scripts/clusters/$EXPERIMENT_CLUSTER.sh"
activate_experiment_environment robotwin
if [[ -n "${NOISE_ARRAY_MAP:-}" ]]; then
    NOISE_JOB=$(python -c 'import json,os; print(json.load(open(os.environ["NOISE_ARRAY_MAP"]))[int(os.environ["SLURM_ARRAY_TASK_ID"])])')
    export NOISE_JOB
fi
: "${NOISE_JOB:?Missing campaign job name}"
cd "$ZERO_WAM_ROOT"
export PYTHONPATH="$ZERO_WAM_ROOT:$ROBOTWIN_ROOT:${PYTHONPATH:-}"
python experiments/scripts/provenance.py --verify "$EXPERIMENT_SOURCE_DIGEST"
RUN_DIR="$NOISE_CAMPAIGN/jobs/$NOISE_JOB"
mkdir -p "$RUN_DIR/diagnostics"
export SAVE_ROOT="$RUN_DIR"
export MODEL_PATH="${MODEL_PATH:-$ZERO_WAM_ROOT/checkpoints/zero-wam-posttrain-robotwin}"
export ICL_LATENT_ROOT="${ICL_LATENT_ROOT:-$ZERO_WAM_ROOT/data/HumanGen/human_latents/robotwin}"
export PORT=$((20000 + 2 * (SLURM_JOB_ID % 10000)))
export MASTER_PORT=$((PORT + 1))
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export PYTHONUNBUFFERED=1 PYTHONFAULTHANDLER=1 MPLBACKEND=Agg
export VIDEO_STATE_FP32=1 COMPACT_OBSERVATION_CACHE=1
export PHASE_SEPARATION=1 PHASE_OFFLOAD_TRANSFORMER=1 TEXT_ENCODER_OFFLOAD=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PROFILE_TIMING=1 RUN_DIAGNOSTICS=1 COUNT_POLICY_FLOPS="${COUNT_POLICY_FLOPS:-0}"
export RENDER_SAMPLES_PER_PIXEL=16
export NO_PROXY="localhost,127.0.0.1,${NO_PROXY:-}" no_proxy="localhost,127.0.0.1,${no_proxy:-}"
STARTED=$(date +%s)
SERVER_STARTED=0 SERVER_READY=0 CLIENT_STARTED=0 CLIENT_FINISHED=0
SERVER_PID="" MONITOR_PID=""
cleanup() {
    local status=$?
    trap - EXIT
    if [[ -n "$SERVER_PID" ]]; then kill "$SERVER_PID" 2>/dev/null || true; wait "$SERVER_PID" 2>/dev/null || true; fi
    if [[ -n "$MONITOR_PID" ]]; then kill "$MONITOR_PID" 2>/dev/null || true; wait "$MONITOR_PID" 2>/dev/null || true; fi
    python3 experiments/scripts/job_report.py timing "$RUN_DIR/diagnostics/job_timing.json" \
        "$status" "$STARTED" "$SERVER_STARTED" "$SERVER_READY" "$CLIENT_STARTED" "$CLIENT_FINISHED" "$(date +%s)" || true
    exit "$status"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv > "$RUN_DIR/diagnostics/hardware.csv"
nvidia-smi --query-gpu=timestamp,index,memory.used,utilization.gpu --format=csv -l 2 > "$RUN_DIR/diagnostics/gpu.csv" &
MONITOR_PID=$!
KIND=$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["kind"])' "$RUN_DIR/settings.json")
if [[ "$KIND" == "train" ]]; then
    activate_experiment_environment zerowam
else
    # Both released success overrides are installed before the client imports.
    (
        flock -x 9
        for task in move_stapler_pad stamp_seal; do
            target="$ROBOTWIN_ROOT/envs/$task.py"
            if [[ ! -e "$target.upstream" ]]; then cp "$target" "$target.upstream"; fi
            if ! cmp -s "evaluation/robotwin/envs/$task.py" "$target"; then
                temporary="$target.noiseicl_${SLURM_JOB_ID}.tmp"
                cp "evaluation/robotwin/envs/$task.py" "$temporary"
                mv "$temporary" "$target"
            fi
        done
    ) 9>"$ROBOTWIN_ROOT/.noiseicl-patches.lock"
    if [[ "$KIND" == "evaluate" ]]; then python experiments/scripts/job_report.py preflight; fi
    python -m pip freeze > "$RUN_DIR/diagnostics/client_packages.txt"
    SERVER_STARTED=$(date +%s)
    (
        activate_experiment_environment zerowam
        python -m pip freeze > "$RUN_DIR/diagnostics/server_packages.txt"
        exec bash evaluation/robotwin/launch_server.sh
    ) > "$RUN_DIR/diagnostics/server.log" 2>&1 &
    SERVER_PID=$!
    deadline=$((SECONDS + 600))
    until curl --noproxy '*' --fail --silent --max-time 2 "http://127.0.0.1:$PORT/healthz" >/dev/null; do
        if ! kill -0 "$SERVER_PID" 2>/dev/null || ((SECONDS >= deadline)); then
            echo "Server startup failed; inspect $RUN_DIR/diagnostics/server.log" >&2
            exit 1
        fi
        sleep 5
    done
    SERVER_READY=$(date +%s)
fi
CLIENT_STARTED=$(date +%s)
python -u experiments/scripts/noiseicl.py run --directory "$NOISE_CAMPAIGN" --job "$NOISE_JOB" \
    > "$RUN_DIR/diagnostics/client.log" 2>&1
CLIENT_FINISHED=$(date +%s)
