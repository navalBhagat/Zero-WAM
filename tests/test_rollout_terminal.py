"""Exercise the real client loop without importing the external simulator."""
import ast
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest


@pytest.mark.parametrize('success_at', [2, 4, 5])
def test_terminal_chunk_stops_and_skips_cache_update(success_at):
    tree = ast.parse(Path('evaluation/robotwin/eval_policy_client_openpi.py').read_text())
    loop = next(n for n in ast.walk(tree) if isinstance(n, ast.While)
                and 'TASK_ENV.take_action_cnt' in ast.unparse(n.test))
    calls = []
    env = SimpleNamespace(take_action_cnt=0, step_lim=100, eval_success=False)
    def take_action(action, action_type):
        env.take_action_cnt += 1
        env.eval_success = env.take_action_cnt == success_at
    env.take_action = take_action
    env.get_obs = lambda: env.take_action_cnt
    def infer(payload):
        calls.append(payload)
        return {'action': np.zeros((14, 2, 4))}
    snapshots = []
    scope = dict(TASK_ENV=env, first=False, first_obs=None, model=SimpleNamespace(infer=infer),
                 diagnostic_call=lambda label, fn, *args, **kwargs: fn(*args, **kwargs),
                 prompt='test', save_visualization=False, video_guidance_scale=1,
                 action_guidance_scale=1, full_action_history=[], full_obs_list=[],
                 gen_video_list=[], guidance_selections=[],
                 rollout_chunk=0, snapshots=SimpleNamespace(write=lambda *args: snapshots.append(args)),
                 np=np, euler2quat=lambda *args: np.zeros(4), format_obs=lambda obs, prompt: obs,
                 succ=False)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[loop], type_ignores=[])), '<rollout>', 'exec'), scope)
    assert env.take_action_cnt == success_at
    assert len(scope['full_action_history']) == success_at
    assert scope['succ'] is True
    assert len(calls) == 1  # Inference only: no partial/terminal cache update.
    assert scope['full_obs_list'][-1] == success_at
    assert snapshots[-1][0] == success_at and snapshots[-1][-1] is True
