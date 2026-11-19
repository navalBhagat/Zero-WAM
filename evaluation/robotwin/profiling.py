"""Opt-in wall-clock timings, with exclusive totals to avoid nested double counting.

Camera capture may submit GPU work; readback usually waits for completion. Use
PROFILE_SYNC_GPU=1 for additional synchronization (adds profiling overhead).
"""
import atexit
from contextlib import contextmanager
import json
from pathlib import Path
import time


class EpisodeProfiler:
    def __init__(self, output, enabled=False, sync_gpu=False, clock=time.perf_counter):
        self.output = Path(output)
        self.enabled, self.sync_gpu, self.clock = enabled, sync_gpu, clock
        self.totals, self.stack, self.originals = {}, [], []
        self.coverage = {}
        self.finished = False
        self.started = clock()
        if enabled:
            atexit.register(self.dump)

    def sync(self):
        if self.sync_gpu:
            import torch
            if torch.cuda.is_available():
                torch.cuda.synchronize()

    @contextmanager
    def measure(self, name):
        if not self.enabled:
            yield
            return
        self.sync()
        state = {'start': self.clock(), 'children': 0.0}
        self.stack.append(state)
        try:
            yield
        finally:
            self.sync()
            elapsed = self.clock() - state['start']
            self.stack.pop()
            if self.stack:
                self.stack[-1]['children'] += elapsed
            record = self.totals.setdefault(name, dict(calls=0, inclusive_seconds=0.0,
                                                       exclusive_seconds=0.0, max_seconds=0.0))
            record['calls'] += 1
            record['inclusive_seconds'] += elapsed
            record['exclusive_seconds'] += max(0.0, elapsed - state['children'])
            record['max_seconds'] = max(record['max_seconds'], elapsed)

    def call(self, name, function, *args, **kwargs):
        with self.measure(name):
            return function(*args, **kwargs)

    def wrap(self, obj, method, name):
        """Only wrap writable Python methods; report unsupported hooks explicitly."""
        if not self.enabled:
            return
        key = name.__name__ if callable(name) else str(name)
        original = getattr(obj, method, None)
        if not callable(original):
            self.coverage[key] = 'unavailable'
            return
        def wrapped(*args, **kwargs):
            label = name(*args, **kwargs) if callable(name) else name
            return self.call(label, original, *args, **kwargs)
        try:
            setattr(obj, method, wrapped)
        except (AttributeError, TypeError):
            self.coverage[key] = 'not writable'
        else:
            self.originals.append((obj, method, original))
            self.coverage[key] = 'enabled'

    def attach(self, task, model):
        for method, label in [('_update_render', 'render_state_update'),
                              ('get_obs', 'observation'),
                              ('take_action', 'action_execution')]:
            self.wrap(task, method, label)
        for method, label in [('update_picture', 'camera_capture'), ('get_rgb', 'camera_readback')]:
            self.wrap(task.cameras, method, label)
        for method in ('left_plan_path', 'right_plan_path'):
            self.wrap(task.robot, method, 'planning_' + method)
        # Pybind methods may be read-only: absence is reported, never claimed timed.
        self.wrap(task.scene, 'step', 'physics_step')
        def server_label(payload, *args, **kwargs):
            if payload.get('reset'):
                return 'server_reset'
            if payload.get('compute_kv_cache'):
                return 'server_cache_update'
            return 'server_sampling_roundtrip'
        self.wrap(model, 'infer', server_label)

    def dump(self, success=None):
        if not self.enabled or self.finished:
            return
        if success is not None:
            self.finished = True
        self.output.parent.mkdir(parents=True, exist_ok=True)
        data = dict(success=success, wall_seconds=self.clock() - self.started,
                    sync_gpu=self.sync_gpu, timings=self.totals, hooks=self.coverage,
                    note='Wall-clock costs; inclusive times overlap. Sum exclusive times only. '
                         'Action exclusive time includes any uninstrumented planning/physics. '
                         'Expert feasibility/setup is excluded; rollout scene setup is timed.')
        self.output.write_text(json.dumps(data, indent=2) + '\n')
        print('[PROFILE] ' + json.dumps(data), flush=True)

    def close(self, success):
        self.dump(success)
        for obj, method, original in reversed(self.originals):
            setattr(obj, method, original)
        self.originals.clear()
        if self.enabled:
            atexit.unregister(self.dump)
