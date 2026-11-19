"""Numerical and cache contracts, runnable on CPU without WAM dependencies."""
import importlib
import sys
import tempfile
import types
import unittest
from pathlib import Path

import torch

parent = types.ModuleType("_noise_test_root")
parent.__path__ = [str(Path(__file__).resolve().parents[1] / "wan_va")]
sys.modules[parent.__name__] = parent
PACKAGE = parent.__name__ + ".noise_inversion"
package = types.ModuleType(PACKAGE)
package.__path__ = [str(Path(__file__).resolve().parents[1] / "wan_va/noise_inversion")]
sys.modules[PACKAGE] = package
solver = importlib.import_module(PACKAGE + ".solver")
ChunkStore = importlib.import_module(PACKAGE + ".artifacts").ChunkStore


class InversionTests(unittest.TestCase):
    def test_constant_flow_round_trip_and_endpoints(self):
        clean = torch.zeros(1, 2, 2, 1, 1)
        grid = torch.tensor([0., .2, .7, 1.])
        visited = []

        def velocity(state, sigma):
            visited.append(sigma)
            return torch.full_like(state, 3.)

        noise = solver.integrate(clean, grid, velocity)
        torch.testing.assert_close(noise, clean + 3)
        reconstruction = solver.integrate(noise, grid.flip(0), velocity)
        torch.testing.assert_close(reconstruction, clean, atol=1e-6, rtol=0)
        self.assertEqual(visited[0], 0.)
        self.assertEqual(visited[3], 1.)

    def test_observed_frame_is_clamped_before_every_evaluation(self):
        clean = torch.zeros(1, 1, 2, 1, 1)
        observed = torch.ones(1, 1, 1, 1, 1) * 7

        def velocity(state, sigma):
            torch.testing.assert_close(state[:, :, :1], observed)
            return torch.ones_like(state)

        result = solver.integrate(clean, torch.tensor([0., .5, 1.]), velocity, observed)
        torch.testing.assert_close(result[:, :, :1], observed)
        self.assertEqual(solver.relative_error(result, result, True), 0.)

    def test_invalid_grid_is_rejected(self):
        with self.assertRaises(ValueError):
            solver.integrate(torch.zeros(1), torch.tensor([0., .5, .2]), lambda x, s: x)

    def test_store_round_trip_and_missing_chunk(self):
        with tempfile.TemporaryDirectory() as directory:
            store = ChunkStore(directory)
            store.save(0, video_noise=torch.ones(1))
            torch.testing.assert_close(store.load(0)["video_noise"], torch.ones(1))
            with self.assertRaises(FileExistsError):
                store.save(0, video_noise=torch.ones(1))
            store.save(2, video_noise=torch.ones(1))
            with self.assertRaises(ValueError):
                store.count()

    def test_episode_inversion_rebuilds_only_past_robot_history(self):
        import json
        module_name = parent.__name__ + ".modules.icl_model"
        module = types.ModuleType(module_name)
        module.OBSERVATION_CACHE_TYPE = 0
        sys.modules[module_name] = module
        runtime = importlib.import_module(PACKAGE + ".runtime")
        calls = []

        class Transformer:
            def clear_prediction_cache(self, *args):
                calls.append("clear")

            def __call__(self, payload, **kwargs):
                calls.append((payload[0], kwargs["update_cache"]))

        def reset(**kwargs):
            self.assertFalse(kwargs["use_icl"])
            self.assertFalse(kwargs["use_text"])
            self.assertEqual(kwargs["prompt"], "")
            self.assertEqual(kwargs["target_text_source"], "empty")
            calls.append("reset")

        scheduler = types.SimpleNamespace(sigmas=torch.tensor([1., .5]),
            num_train_timesteps=1000, set_timesteps=lambda steps: None)
        server = types.SimpleNamespace(_reset=reset, target_text_source="empty",
            use_text=False, use_icl=False, use_guidance=False,
            scheduler=scheduler, job_config=types.SimpleNamespace(num_inference_steps=2),
            device="cpu", transformer=Transformer(), cache_name="test",
            _prepare_icl_stream=lambda data, time, mode, cache: (mode, data))

        def velocity(state, timestep):
            calls.append(("velocity", server.chunk_idx))
            return torch.ones_like(state)

        server._video_velocity = velocity
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for index in range(2):
                ChunkStore(root / "chunks").save(index,
                    initial_frame=torch.zeros(1, 1, 1, 1, 1),
                    video=torch.zeros(1, 1, 2, 1, 1), action_noise=torch.ones(1))
            ChunkStore(root / "history").save(0, video=torch.zeros(1), actions=torch.ones(1))
            report = runtime.invert_episode(server, directory)
            self.assertEqual(report["relative_errors"], [0., 0.])
            self.assertEqual(ChunkStore(root / "inverted").count(), 2)
            self.assertTrue((root / "inversion.json").exists())
        history = calls.index(("video", 1))
        self.assertTrue(all(call != ("velocity", 1) for call in calls[:history]))
        self.assertEqual(calls.count(("video", 1)), 1)
        self.assertEqual(calls.count(("action", 1)), 1)

    def test_replay_exhaustion_never_samples_fresh_noise(self):
        import ast
        source = Path(__file__).resolve().parents[1] / "wan_va/wan_va_server.py"
        tree = ast.parse(source.read_text())
        cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "VA_Server")
        method = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "infer")
        method.decorator_list = []
        namespace = dict(__package__=parent.__name__, Path=Path)
        exec(compile(ast.Module(body=[method], type_ignores=[]), str(source), "exec"), namespace)
        with tempfile.TemporaryDirectory() as directory:
            ChunkStore(Path(directory) / "inverted").save(0, video_noise=torch.ones(1))
            server = types.SimpleNamespace(use_icl_model=True, use_icl=False,
                use_guidance=False, use_text=False, target_text_source="empty", chunk_idx=1)
            result = namespace["infer"](server, dict(noise_replay_directory=directory))
            server.use_text = True
            with self.assertRaisesRegex(ValueError, "language, ICL"):
                namespace["infer"](server, dict(noise_replay_directory=directory))
        self.assertEqual(result, dict(noise_exhausted=True))

    def test_velocity_cache_write_is_explicit(self):
        import ast
        source = Path(__file__).resolve().parents[1] / "wan_va/wan_va_server.py"
        tree = ast.parse(source.read_text())
        cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "VA_Server")
        method = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "_video_velocity")
        namespace = dict(PREDICTION_CACHE_TYPE=1, data_seq_to_patch=lambda *a: a[1])
        exec(compile(ast.Module(body=[method], type_ignores=[]), str(source), "exec"), namespace)
        calls = []

        def transformer(payload, **kwargs):
            calls.append(kwargs["update_cache"])
            return payload

        server = types.SimpleNamespace(dtype=torch.float32, chunk_idx=1,
            transformer=transformer, cache_name="test", use_icl_cfg=False,
            target_text_cfg_active=False, job_config=types.SimpleNamespace(frame_chunk_size=2),
            latent_height=1, latent_width=1, _prepare_icl_stream=lambda *a: a[0])
        transformer.patch_size = (1, 1, 1)
        method_fn = namespace["_video_velocity"]
        method_fn(server, torch.ones(1, 1, 2, 1, 1), 0)
        method_fn(server, torch.ones(1, 1, 2, 1, 1), 0, update_cache=True)
        self.assertEqual(calls, [0, 1])


if __name__ == "__main__":
    unittest.main()
