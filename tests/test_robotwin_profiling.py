import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

spec = importlib.util.spec_from_file_location(
    'profiling', Path(__file__).resolve().parents[1] / 'evaluation/robotwin/profiling.py',
)
profiling = importlib.util.module_from_spec(spec)
spec.loader.exec_module(profiling)


def test_nested_times_are_not_double_counted(tmp_path):
    times = iter([0., 1., 2., 4., 6., 7.])
    profile = profiling.EpisodeProfiler(tmp_path / 'profile.json', enabled=True, clock=lambda: next(times))
    with profile.measure('action'):
        with profile.measure('camera'):
            pass
    profile.close(True)
    data = json.loads((tmp_path / 'profile.json').read_text())
    assert data['timings']['action']['inclusive_seconds'] == 5
    assert data['timings']['action']['exclusive_seconds'] == 3
    assert data['timings']['camera']['exclusive_seconds'] == 2


def test_wrappers_preserve_results_restore_and_report_failures(tmp_path):
    profile = profiling.EpisodeProfiler(tmp_path / 'profile.json', enabled=True)
    obj = SimpleNamespace(capture=lambda value: value + 1)
    original = obj.capture
    profile.wrap(obj, 'capture', 'camera')
    profile.wrap(obj, 'absent', 'missing')
    assert obj.capture(3) == 4
    profile.close(False)
    assert obj.capture is original
    data = json.loads((tmp_path / 'profile.json').read_text())
    assert data['success'] is False
    assert data['timings']['camera']['calls'] == 1
    assert data['hooks']['missing'] == 'unavailable'


def test_disabled_profiling_does_not_wrap(tmp_path):
    profile = profiling.EpisodeProfiler(tmp_path / 'profile.json')
    obj = SimpleNamespace(capture=lambda: 3)
    original = obj.capture
    profile.wrap(obj, 'capture', 'camera')
    assert obj.capture is original
    assert profile.call('capture', obj.capture) == 3
    profile.close(True)
    assert not (tmp_path / 'profile.json').exists()


def test_attach_counts_real_calls_without_extra_observations(tmp_path):
    class Scene:
        __slots__ = ()  # emulate a read-only native method
        def step(self):
            pass
    class Cameras:
        def update_picture(self):
            pass
        def get_rgb(self):
            return 'rgb'
    class Robot:
        def left_plan_path(self):
            pass
        def right_plan_path(self):
            pass
    class Task:
        def __init__(self):
            self.scene, self.cameras, self.robot = Scene(), Cameras(), Robot()
        def _update_render(self):
            pass
        def get_obs(self):
            self._update_render()
            self.cameras.update_picture()
            return self.cameras.get_rgb()
        def take_action(self):
            self.scene.step()
            self.robot.left_plan_path()
            self._update_render()
    task = Task()
    model = SimpleNamespace(infer=lambda payload: payload)
    profile = profiling.EpisodeProfiler(tmp_path / 'profile.json', enabled=True)
    profile.attach(task, model)
    task.take_action()
    assert task.get_obs() == 'rgb'
    for payload in ({'reset': True}, {'compute_kv_cache': True}, {}):
        assert model.infer(payload) == payload
    profile.close(True)
    assert profile.totals['observation']['calls'] == 1
    assert profile.totals['camera_capture']['calls'] == 1
    assert profile.totals['render_state_update']['calls'] == 2
    assert profile.totals['server_sampling_roundtrip']['calls'] == 1
    assert profile.coverage['physics_step'] == 'not writable'
