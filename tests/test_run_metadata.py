"""Run/episode provenance and optional GPU monitor; no simulator needed."""
import ast
import importlib.util
import json
import os
from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('run_metadata', ROOT / 'evaluation/robotwin/run_metadata.py')
metadata = importlib.util.module_from_spec(spec)
spec.loader.exec_module(metadata)


def test_run_metadata_records_revision_resolved_paths_not_secrets(tmp_path, monkeypatch):
    monkeypatch.setattr(metadata, '_git', lambda root, *args: '' if args[0] == 'status' else 'revision')
    real = tmp_path / 'real-model'
    real.mkdir()
    link = tmp_path / 'model'
    link.symlink_to(real, target_is_directory=True)
    path = tmp_path / 'config.json'
    data = metadata.create_run(path, dict(MODEL_PATH=str(link), MONITOR_GPU='0',
                                         USE_TEXT='1', SECRET_TOKEN='never-save'), ROOT)
    assert data['code'] == dict(revision='revision', dirty=False, robotwin_revision='revision')
    assert data['resolved_paths']['MODEL_PATH'] == str(real)
    assert data['MONITOR_GPU'] == '0' and data['USE_TEXT'] == '1'
    assert 'SECRET_TOKEN' not in data and 'never-save' not in path.read_text()
    assert not path.with_name('config.json.tmp').exists()


def test_episode_updates_preserve_initialization_and_other_episodes(tmp_path):
    path = tmp_path / 'config.json'
    path.write_text(json.dumps(dict(USE_TEXT='1', code={'revision':'abc'}, episodes=[])))
    metadata.record_episode(path, 0, status='initializing', task_instruction='Open door', simulator_seed=100000)
    assert json.loads(path.read_text())['episodes'][0]['status'] == 'initializing'
    metadata.record_episode(path, 0, status='running', model_text='Open door')
    metadata.record_episode(path, 1, status='initializing', task_instruction='Other instruction')
    metadata.record_episode(path, 0, status='completed', success=True)
    data = json.loads(path.read_text())
    assert data['code']['revision'] == 'abc'
    assert len(data['episodes']) == 2
    assert data['episodes'][0]['model_text'] == 'Open door'
    assert data['episodes'][0]['simulator_seed'] == 100000
    assert data['episodes'][1]['status'] == 'initializing'


@pytest.mark.parametrize('use_text', [False, True])
def test_server_reports_effective_text_and_guidance(use_text):
    server = next(n for n in ast.parse((ROOT / 'wan_va/wan_va_server.py').read_text()).body
                  if isinstance(n, ast.ClassDef) and n.name == 'VA_Server')
    method = next(n for n in server.body if isinstance(n, ast.FunctionDef) and n.name == '_episode_metadata')
    scope = dict(os=os)
    exec(compile(ast.Module(body=[method], type_ignores=[]), '<metadata-method>', 'exec'), scope)
    job = SimpleNamespace(external_every=10, external_max_norm=1, dino_checkpoint='/dino',
                          reference_start=65, reference_stop=66, model_path='/model',
                          num_inference_steps=50, action_num_inference_steps=50,
                          frame_chunk_size=2, video_state_fp32=True)
    mock = SimpleNamespace(use_text=use_text, use_icl=True, use_guidance=True,
                           external_strength=.01, external_reference_kind='reconstructed_latent',
                           job_config=job, dtype=torch.bfloat16, icl_guidance_scale=5,
                           video_guidance_scale=1, vae=SimpleNamespace(decoder=torch.nn.Linear(1,1)))
    result = scope['_episode_metadata'](mock, 'Open the door', dict(icl_latent_path='/human.pth'))
    assert result['model_text'] == ('Open the door' if use_text else '')
    assert result['icl_context_text'] == result['model_text']
    assert result['cached_caption_used'] is False
    assert result['guidance']['reference_start'] == 65
    assert result['guidance']['decoder_dtype'] == 'torch.float32'
    mock.use_guidance = False
    mock.external_strength = 0
    assert scope['_episode_metadata'](mock, 'Open the door', {})['guidance'] is None


@pytest.mark.skipif(not (ROOT / 'implementation/scripts/eval_one_task.sbatch').exists(),
                    reason='Local Slurm recipe is intentionally Git-ignored')
@pytest.mark.parametrize('enabled', ['0', '1'])
def test_monitor_gpu_flag_controls_csv_without_a_gpu(tmp_path, enabled):
    script = (ROOT / 'implementation/scripts/eval_one_task.sbatch').read_text()
    section = script.split('MONITOR_PID=""', 1)[1].split('SERVER_PID=""', 1)[0]
    binary = tmp_path / 'nvidia-smi'
    binary.write_text('#!/bin/sh\necho "fake GPU measurement"\n')
    binary.chmod(0o755)
    env = dict(os.environ, PATH=str(tmp_path) + os.pathsep + os.environ['PATH'],
               MONITOR_GPU=enabled, RUN_DIR=str(tmp_path))
    subprocess.run(['bash', '-c', 'MONITOR_PID=""\n' + section +
                    '\nif [[ -n "$MONITOR_PID" ]]; then wait "$MONITOR_PID"; fi'],
                   env=env, check=True)
    assert (tmp_path / 'gpu.csv').exists() == (enabled == '1')
