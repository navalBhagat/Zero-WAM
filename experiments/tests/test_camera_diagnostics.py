import io
import json
from types import SimpleNamespace
import unittest
from evaluation.robotwin.camera_diagnostics import trace_cameras


class CameraDiagnosticsTests(unittest.TestCase):
    def test_original_order_and_references_are_preserved(self):
        calls = []
        def camera(name):
            return SimpleNamespace(take_picture=lambda: calls.append(name))
        left, right, static = camera("left"), camera("right"), camera("static")
        cameras = SimpleNamespace(collect_wrist_camera=True, left_camera=left,
                                  right_camera=right, static_camera_list=[static])
        def update():
            cameras.left_camera.take_picture()
            cameras.right_camera.take_picture()
            for item in cameras.static_camera_list:
                item.take_picture()
        cameras.update_picture = update
        stream = io.StringIO()
        trace_cameras(cameras, stream)
        cameras.update_picture()
        self.assertEqual(calls, ["left", "right", "static"])
        self.assertIs(cameras.left_camera, left)
        self.assertIs(cameras.static_camera_list[0], static)
        rows = [json.loads(line) for line in stream.getvalue().splitlines()]
        self.assertEqual([row["event"] for row in rows], ["begin", "end"] * 3)

    def test_error_restores_camera_references(self):
        def fail():
            raise RuntimeError("render failed")
        original = SimpleNamespace(take_picture=fail)
        cameras = SimpleNamespace(collect_wrist_camera=False, static_camera_list=[original])
        cameras.update_picture = lambda: cameras.static_camera_list[0].take_picture()
        stream = io.StringIO()
        trace_cameras(cameras, stream)
        with self.assertRaisesRegex(RuntimeError, "render failed"):
            cameras.update_picture()
        self.assertIs(cameras.static_camera_list[0], original)
        self.assertIn('"event": "error"', stream.getvalue())
