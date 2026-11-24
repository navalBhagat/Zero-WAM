"""Per-camera timing without modifying SAPIEN's read-only native methods."""
import json
import time

from evaluation.robotwin.watchdog import guarded_call


class TimedCamera:
    def __init__(self, camera, label, stream):
        self.camera, self.label, self.stream = camera, label, stream

    def __getattr__(self, name):
        return getattr(self.camera, name)

    def take_picture(self):
        started = time.perf_counter()
        self.record("begin")
        try:
            result = guarded_call(self.label, self.camera.take_picture)
        except BaseException as error:
            self.record("error", error=repr(error))
            raise
        self.record("end", seconds=time.perf_counter() - started)
        return result

    def record(self, event, **details):
        self.stream.write(json.dumps(dict(camera=self.label, event=event,
                                          monotonic_seconds=time.perf_counter(), **details)) + "\n")
        self.stream.flush()


def trace_cameras(cameras, stream):
    """Temporarily proxy only references used by the original update_picture."""
    original = cameras.update_picture

    def update_picture():
        static = cameras.static_camera_list
        wrist = {}
        try:
            cameras.static_camera_list = [TimedCamera(camera, f"static[{i}]", stream)
                                          for i, camera in enumerate(static)]
            if cameras.collect_wrist_camera:
                for name in ("left_camera", "right_camera"):
                    wrist[name] = getattr(cameras, name)
                    setattr(cameras, name, TimedCamera(wrist[name], name, stream))
            return original()
        finally:
            cameras.static_camera_list = static
            for name, camera in wrist.items():
                setattr(cameras, name, camera)

    cameras.update_picture = update_picture


def configure_diagnostic_denoiser(render, denoiser):
    """Keep RoboTwin setup from overriding this diagnostic's denoiser choice."""
    if denoiser not in ("oidn", "none"):
        raise ValueError("Diagnostic denoiser must be oidn or none")
    original = render.set_ray_tracing_denoiser

    def enforce(_scene_default):
        return original(denoiser)

    render.set_ray_tracing_denoiser = enforce
    enforce(denoiser)
