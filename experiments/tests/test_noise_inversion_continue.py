import hashlib
import json
from pathlib import Path
import tempfile
import unittest
import zipfile

from experiments.scripts.noise_inversion_continue import continue_trial, validate_runtime
from experiments.noise_inversion.records import save_json
from experiments.tests.test_noise_inversion_pilot import episode, tensors


class ContinuationTests(unittest.TestCase):
    def test_failed_source_is_reused_and_original_identity_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            identity = dict(settings=dict(task="stack_blocks_three", scene_seed=100000,
                                          sampling_seed=1234, icl_seed=0, render_samples=16), source_digest="original")
            save_json(root / "trial.json", identity)
            save_json(root / "capture/config.json", dict(episodes=[episode("stack_blocks_three", 100000, False, "capture")]))
            tensors(root / "chunks")
            tensors(root / "history")
            before = (root / "capture/config.json").read_bytes()
            events = []
            def launch(folder, task, env):
                phase = env["NOISE_INVERSION_MODE"]
                events.append(phase)
                self.assertNotEqual(phase, "capture")
                save_json(folder / "config.json", dict(episodes=[episode(task, 100000, True, phase)]))
            def infer(request):
                events.append("invert")
                tensors(root / "inverted")
                return dict(inversion=dict(chunks=3))
            result = continue_trial(root, infer, digest="new", launch=launch)
            self.assertEqual(events, ["control", "invert", "replay"])
            self.assertEqual((root / "capture/config.json").read_bytes(), before)
            self.assertEqual(json.loads((root / "trial.json").read_text()), identity)
            self.assertFalse(result["source_success"])
            self.assertEqual(result["status"], "completed")
            continue_trial(root, infer, digest="new", launch=launch)
            self.assertEqual(events, ["control", "invert", "replay"])

    def test_missing_history_does_not_launch_any_phase(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            save_json(root / "trial.json", dict(settings=dict(task="stack_blocks_three", scene_seed=100000,
                      sampling_seed=1234), source_digest="original"))
            save_json(root / "capture/config.json", dict(episodes=[episode("stack_blocks_three", 100000, False, "capture")]))
            tensors(root / "chunks")
            with self.assertRaisesRegex(ValueError, "Incomplete tensor sequence"):
                continue_trial(root, lambda _: self.fail("must not infer"), digest="new",
                               launch=lambda *args: self.fail("must not launch"))

    def test_scientific_code_changes_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "source.zip"
            hashes = {"wan_va/model.py": "old", "evaluation/robotwin/camera_diagnostics.py": "diagnostic"}
            digest = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()
            with zipfile.ZipFile(path, "w") as archive:
                archive.writestr("source_hashes.json", json.dumps(hashes))
            validate_runtime(path, digest, dict(hashes, **{"evaluation/robotwin/camera_diagnostics.py": "new"}))
            with self.assertRaisesRegex(ValueError, "runtime changed"):
                validate_runtime(path, digest, dict(hashes, **{"wan_va/model.py": "changed"}))
            with self.assertRaisesRegex(ValueError, "does not belong"):
                validate_runtime(path, "wrong", hashes)
