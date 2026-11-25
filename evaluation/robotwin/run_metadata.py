"""Small, atomic JSON run records; suitable for later W&B config/summary logging.

No simulation, Torch, credentials, or full environment dump is needed here.
"""
import argparse
import csv
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
ENV_KEYS = (
    'NOISE_METHOD',
    'NOISE_OUTPUT',
    'COUNT_POLICY_FLOPS',
    'RUN_DIAGNOSTICS',
    'TASK',
    'TEST_NUM',
    'SEED',
    'SAMPLING_SEED',
    'ICL_SEED',
    'USE_ICL',
    'USE_TEXT',
    'ICL_GUIDANCE_SCALE',
    'TEXT_GUIDANCE_SCALE',
    'TEXT_ENCODER_OFFLOAD',
    'MODEL_PATH',
    'ROBOTWIN_ROOT',
    'ICL_LATENT_ROOT',
    'VIDEO_STATE_FP32',
    'PROFILE_TIMING',
    'PROFILE_SYNC_GPU',
    'MONITOR_GPU',
    'SLURM_JOB_ID',
    'RENDER_SAMPLES_PER_PIXEL',
    'COMPACT_OBSERVATION_CACHE',
    'SAVE_ROLLOUT_SNAPSHOTS',
    'PHASE_COMPARE_ACTIONS',
    'PHASE_SEPARATION',
    'PHASE_OFFLOAD_TRANSFORMER',
    'STAGE_REQUIRE_SUCCESS',
    'TARGET_TEXT_SOURCE',
    'ICL_TEXT_SOURCE',
    'EXPERIMENT_LABEL',
    'EXPERIMENT_STAGE',
    'EXPERIMENT_REPEAT',
)

def _git(root, *args):
    try:
        return subprocess.check_output(['git', '-C', str(root), *args], text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return None

def write_json_atomic(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(data, indent=2) + '\n', encoding='utf-8')
    temporary.replace(path)

def create_run(path, env, project_root):
    root = Path(project_root).resolve()
    settings = {key: env.get(key) for key in ENV_KEYS}
    resolved = {key: str(Path(value).expanduser().resolve()) for (key, value) in settings.items() if value and key in ('MODEL_PATH', 'ROBOTWIN_ROOT', 'ICL_LATENT_ROOT')}
    status = _git(root, 'status', '--porcelain')
    data = dict(schema_version=2, created_at_utc=datetime.now(timezone.utc).isoformat(), **settings, resolved_paths=resolved, code=dict(revision=_git(root, 'rev-parse', 'HEAD'), dirty=None if status is None else bool(status), robotwin_revision=_git(env.get('ROBOTWIN_ROOT', ''), 'rev-parse', 'HEAD')), episodes=[])
    write_json_atomic(path, data)
    return data

def record_episode(path, episode_id, **updates):
    """Single client writer per run. Preserve run settings and earlier episodes."""
    path = Path(path)
    data = json.loads(path.read_text()) if path.exists() else dict(schema_version=2, episodes=[])
    episodes = data.setdefault('episodes', [])
    record = next((item for item in episodes if item['episode_id'] == episode_id), None)
    if record is None:
        record = dict(episode_id=episode_id)
        episodes.append(record)
    record.update(updates)
    write_json_atomic(path, data)
    write_run_results(path.parent, data)
RESULT_COLUMNS = (
    'noise_method',
    'episode',
    'status',
    'task',
    'simulator_seed',
    'sampling_seed',
    'success',
    'use_text',
    'use_icl',
    'actions',
    'chunks',
    'preparation_seconds',
    'rollout_seconds',
    'policy_counted_tflops',
    'policy_estimated_tflops',
    'flops_status',
    'unsupported_op_types',
    'text_encoder_tflops',
    'transformer_tflops',
    'reference_decode_tflops',
    'observation_encode_tflops',
    'video',
)

def write_run_results(root, data):
    """Readable per-episode summary, written atomically by the client only."""
    root = Path(root)
    rows = []
    for e in data.get('episodes', []):
        compute = e.get('compute') or {}
        profile = {}
        profile_path = e.get('profile_path')
        if profile_path and Path(profile_path).is_file():
            profile = json.loads(Path(profile_path).read_text())
        timings = profile.get('timings', {})
        reset = timings.get('server_reset', {}).get('exclusive_seconds')
        wall = profile.get('wall_seconds')
        counted = compute.get('counted_flops')
        known = not compute.get('unsupported_ops') and compute.get('status') == 'estimated_all_observed_ops'
        server = e.get('server') or {}
        components = compute.get('components_flops', {})
        rows.append(dict(
            noise_method=server.get('noise_method', data.get('NOISE_METHOD')),
            episode=e['episode_id'],
            status=e.get('status'),
            task=e.get('task', data.get('TASK')),
            simulator_seed=e.get('simulator_seed'),
            sampling_seed=(e.get('server') or {}).get('sampling_seed', data.get('SAMPLING_SEED')),
            use_text=server.get('use_text'),
            use_icl=server.get('use_icl'),
            text_encoder_tflops=components.get('text_encoder', 0) / 1000000000000.0 if compute else None,
            transformer_tflops=components.get('transformer', 0) / 1000000000000.0 if compute else None,
            reference_decode_tflops=components.get('reference_vae_decode', 0) / 1000000000000.0 if compute else None,
            observation_encode_tflops=(components.get('observation_vae_encode', 0) + components.get('initial_observation_vae_encode', 0)) / 1000000000000.0 if compute else None,
            success=e.get('success'),
            actions=e.get('policy_action_calls'),
            chunks=timings.get('server_sampling_roundtrip', {}).get('calls'),
            preparation_seconds=reset,
            rollout_seconds=wall - reset if wall is not None and reset is not None else None,
            policy_counted_tflops=counted / 1000000000000.0 if counted is not None else None,
            policy_estimated_tflops=counted / 1000000000000.0 if known and counted is not None else None,
            flops_status=compute.get('status', 'not_recorded'),
            unsupported_op_types=len(compute.get('unsupported_ops', {})),
            video=str(Path('videos') / Path(e['video_path']).name) if e.get('video_path') else None,
        ))
    target = root / 'results.csv'
    temporary = root / 'results.csv.tmp'
    root.mkdir(parents=True, exist_ok=True)
    with temporary.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=RESULT_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(target)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    parser.add_argument('--refresh', action='store_true', help='Refresh results.csv and compute totals from completed/partial diagnostics')
    parser.add_argument('--project-root', type=Path, default=Path.cwd())
    args = parser.parse_args()
    if args.refresh:
        data = json.loads(args.output.read_text())
        flops = args.output.parent / 'diagnostics/flops.jsonl'
        if flops.is_file():
            for line in flops.read_text().splitlines():
                entry = json.loads(line)
                ordinal = entry.get('episode', 0)
                if 0 <= ordinal < len(data.get('episodes', [])):
                    data['episodes'][ordinal]['compute'] = entry['cumulative']
        write_json_atomic(args.output, data)
        write_run_results(args.output.parent, data)
    else:
        data = create_run(args.output, os.environ, args.project_root)
        write_run_results(args.output.parent, data)
if __name__ == '__main__':
    main()
