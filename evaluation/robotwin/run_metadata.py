"""Small, atomic JSON run records; suitable for later W&B config/summary logging.

No simulation, Torch, credentials, or full environment dump is needed here.
"""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess


ENV_KEYS = (
    'TASK', 'TEST_NUM', 'SEED', 'SAMPLING_SEED', 'ICL_SEED', 'USE_ICL', 'USE_TEXT',
    'USE_GUIDANCE', 'ICL_GUIDANCE_SCALE', 'TEXT_GUIDANCE_SCALE', 'TEXT_ENCODER_OFFLOAD',
    'MODEL_PATH', 'ROBOTWIN_ROOT', 'ICL_LATENT_ROOT', 'GUIDANCE_CHECKPOINT',
    'GUIDANCE_STRENGTH', 'GUIDANCE_EVERY', 'GUIDANCE_MAX_NORM',
    'GUIDANCE_REFERENCE_SOURCE',
    'GUIDANCE_DECODER_DTYPE', 'VIDEO_STATE_FP32', 'GUIDANCE_CAPTURE', 'PROFILE_TIMING',
    'PROFILE_SYNC_GPU', 'MONITOR_GPU', 'SLURM_JOB_ID', 'RENDER_SAMPLES_PER_PIXEL',
    'GUIDANCE_REFERENCE_FRAMES',
)


def _git(root, *args):
    try:
        return subprocess.check_output(['git', '-C', str(root), *args], text=True,
                                       stderr=subprocess.DEVNULL).strip()
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
    resolved = {key: str(Path(value).expanduser().resolve()) for key, value in settings.items()
                if value and key in ('MODEL_PATH', 'ROBOTWIN_ROOT', 'ICL_LATENT_ROOT',
                                     'GUIDANCE_CHECKPOINT', 'GUIDANCE_CAPTURE')}
    status = _git(root, 'status', '--porcelain')
    data = dict(schema_version=2, created_at_utc=datetime.now(timezone.utc).isoformat(),
                **settings, resolved_paths=resolved,
                code=dict(revision=_git(root, 'rev-parse', 'HEAD'),
                          dirty=None if status is None else bool(status),
                          robotwin_revision=_git(env.get('ROBOTWIN_ROOT', ''), 'rev-parse', 'HEAD')),
                episodes=[])
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    parser.add_argument('--project-root', type=Path, default=Path.cwd())
    args = parser.parse_args()
    create_run(args.output, os.environ, args.project_root)


if __name__ == '__main__':
    main()
