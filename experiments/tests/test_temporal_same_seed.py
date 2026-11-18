import importlib.util
from pathlib import Path

from experiments.scripts.robotwin_suite import parser, specifications

ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('same_seed',ROOT/'experiments/scripts/run_temporal_guidance_same_seed.py')
launcher=importlib.util.module_from_spec(spec);spec.loader.exec_module(launcher)


def test_same_seed_launch_is_one_paired_distance_episode(tmp_path):
    cmd=launcher.command(tmp_path/'manifest.json')
    assert '--submit' not in cmd
    assert '--distance-increment' not in cmd

    args=parser().parse_args(cmd[2:])
    run,=specifications(args)
    env=run['environment']
    expected=dict(TASK='place_object_scale',SEED='0',ICL_SEED='0',SAMPLING_SEED='1234',
        TEST_NUM='1',USE_ICL='0',USE_TEXT='0',USE_GUIDANCE='1',GUIDANCE_SELECTION='window',
        GUIDANCE_LOOKAHEAD_MODE='dino_distance',GUIDANCE_DISTANCE_INCREMENT='',
        GUIDANCE_DISTANCE_SMOOTHING='3',GUIDANCE_MAX_LOOKAHEAD_FRAMES='24',
        GUIDANCE_REFERENCE_SOURCE='latent',GUIDANCE_MATCH_WINDOW='24',GUIDANCE_LOOKAHEAD='4',
        GUIDANCE_STRENGTH='0.01',RENDER_SAMPLES_PER_PIXEL='8',
        PHASE_SEPARATION='1',COMPACT_OBSERVATION_CACHE='1',VIDEO_STATE_FP32='1')
    assert all(env[k]==v for k,v in expected.items())
    assert '--submit' in launcher.command(tmp_path/'submit.json',submit=True)
