"""Optional task instrumentation; never sets task success or modifies physics."""
import json
import os


class MicrowaveDiagnostic:
    def __init__(self, path):
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)

    def record(self, env, chunk):
        angle = float(env.microwave.get_qpos()[0])
        upper = float(env.microwave.get_qlimits()[0][1])
        row = dict(chunk=chunk, simulator_action_count=int(env.take_action_cnt),
                   door_angle=angle, upper_limit=upper, threshold=0.6 * upper,
                   opening_fraction=angle / upper if upper else None,
                   check_success=bool(env.check_success()),
                   eval_success=bool(env.eval_success))
        with self.path.open('a') as stream:
            stream.write(json.dumps(row) + '\n')


def make_task_diagnostic(task, run_dir, episode):
    if os.environ.get('TASK_DIAGNOSTICS') == 'microwave' and task == 'open_microwave':
        return MicrowaveDiagnostic(run_dir / 'diagnostics' / f'microwave_episode_{episode:03d}.jsonl')
    return None
