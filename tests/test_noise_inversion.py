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
