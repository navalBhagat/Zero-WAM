"""Paired trial sequencing, failure semantics, reproducibility and safe retries."""
import json
import tempfile
import unittest
from pathlib import Path

from experiments.noise_inversion.config import Trial
from experiments.noise_inversion.pilot import run_trial, run_controls, ensure_inversion
from experiments.noise_inversion.records import save_json
from experiments.noise_inversion.scenes import SceneSearch, ProbeResult, select_scene


def episode(task, seed, success, phase):
    return dict(status="completed", success=success, task=task, simulator_seed=seed,
        icl_seed=0, task_instruction="Put object in target", rollout_chunks=3,
        stopping_policy="same", rendering=dict(samples_per_pixel_effective=16),
        server=dict(model_path="checkpoint", video_state_dtype="torch.float32",
            transformer_dtype="torch.bfloat16", video_inference_steps=50,
            action_inference_steps=50, frame_chunk_size=2, sampling_seed=1234,
            compact_observation_cache=True, phase_separation=False, phase_offload_transformer=True,
            use_text=phase == "capture", use_icl=phase == "capture", use_guidance=False, model_text="",
            cached_caption_used=phase == "capture", target_text_source="empty"))


def tensors(directory, count=3):
    directory.mkdir(parents=True, exist_ok=True)
    for index in range(count):
        (directory / f"chunk_{index:04d}.pt").touch()


class PilotTests(unittest.TestCase):
    def exercise(self, source_success):
        events = []
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            trial = Trial("place_bread_basket", 100000, 1234)

            def launch(folder, task, env):
                phase = env["NOISE_INVERSION_MODE"]
                events.append(phase)
                if phase == "capture":
                    tensors(root / "chunks")
                self.assertEqual(env["SAMPLING_SEED"], "1234")
                self.assertEqual(env["START_SEED"], "100000")
                self.assertEqual(env["NOISE_INVERSION_CONTINUATION"], "gaussian")
                self.assertEqual(env["USE_TEXT"], "1" if phase == "capture" else "0")
                save_json(folder / "config.json", dict(episodes=[
                    episode(task, 100000, source_success if phase == "capture" else True, phase)]))

            def infer(request):
                events.append("invert")
                tensors(root / "inverted")
                self.assertNotIn("prompt", request)
                return dict(inversion=dict(chunks=3, relative_errors=[0.] * 3))

            result = run_trial(root, trial, infer, launch, digest="test")
            run_trial(root, trial, infer, launch, digest="test")
            self.assertEqual(events, ["capture", "control", "invert", "replay"])
            with self.assertRaisesRegex(ValueError, "settings/source"):
                run_trial(root, trial, infer, launch, digest="different")
            return result

    def test_successful_source_and_resume(self):
        self.assertTrue(self.exercise(True)["source_success"])

    def test_failed_source_is_inverted_and_replayed(self):
        result = self.exercise(False)
        self.assertFalse(result["source_success"])
        self.assertTrue(result["replay_success"])
        self.assertEqual(result["status"], "completed")

    def test_interrupted_rollout_requires_explicit_retry_and_is_archived(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "capture").mkdir()
            (root / "capture/client.log").write_text("interrupted")
            trial = Trial("place_object_scale", 100000, 1234)
            def launch(folder, task, env):
                if env["NOISE_INVERSION_MODE"] == "capture":
                    tensors(root / "chunks")
                save_json(folder / "config.json", dict(episodes=[episode(task, 100000, False, env["NOISE_INVERSION_MODE"])]))
            with self.assertRaisesRegex(ValueError, "retry-incomplete"):
                run_trial(root, trial, lambda r: {}, launch, digest="test")
            def invert(request):
                tensors(root / "inverted")
                return dict(inversion=dict(chunks=3))
            result = run_trial(root, trial, invert, launch, digest="test", retry=True)
            self.assertEqual(result["status"], "completed")
            self.assertEqual(len(list(root.glob("attempts/*/capture/client.log"))), 1)

    def test_completed_pair_mismatch_is_incomplete_not_task_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            def launch(folder, task, env):
                phase = env["NOISE_INVERSION_MODE"]
                if phase == "capture":
                    tensors(root / "chunks")
                value = episode(task, 100000, False, phase)
                if phase == "control":
                    value["server"]["sampling_seed"] = 999
                save_json(folder / "config.json", dict(episodes=[value]))
            with self.assertRaisesRegex(ValueError, "sampling_seed"):
                run_trial(root, Trial("place_object_scale", 100000, 1234), lambda r: {}, launch, digest="test")
            self.assertEqual(json.loads((root / "summary.json").read_text())["status"], "incomplete")

    def test_partial_inversion_is_preserved_and_rebuilt_on_explicit_retry(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            tensors(root / "inverted", count=1)
            save_json(root / "inversion.json", dict(chunks=3))
            def infer(request):
                tensors(root / "inverted", count=3)
                return dict(inversion=dict(chunks=3))
            with self.assertRaisesRegex(ValueError, "retry-incomplete"):
                ensure_inversion(root, 3, infer, False)
            ensure_inversion(root, 3, infer, True)
            self.assertEqual(len(list((root / "inverted").glob("*.pt"))), 3)
            self.assertEqual(len(list(root.glob("attempts/*/inversion.json"))), 1)

    def test_replacement_seed_is_used_by_both_paired_rollouts(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "trial"
            search = SceneSearch(Path(temporary) / "scenes.json", 100000, (100000, 100001))
            actual = []
            def launch(folder, task, env):
                phase = env["NOISE_INVERSION_MODE"]
                if phase == "capture":
                    self.assertEqual(env["NOISE_INVERSION_SCENE_REGISTRY"], str(search.registry))
                    seed, _ = select_scene(search, lambda candidate: ProbeResult(candidate == 100002, {}))
                    tensors(root / "chunks")
                else:
                    seed = int(env["START_SEED"])
                actual.append(seed)
                save_json(folder / "config.json", dict(episodes=[episode(task, seed, False, phase)]))
            def infer(request):
                tensors(root / "inverted")
                return dict(inversion=dict(chunks=3))
            summary = run_trial(root, Trial("place_bread_basket", 100000, 1234), infer,
                                launch, digest="test", scene_search=search)
            self.assertEqual(actual, [100002, 100002, 100002])
            self.assertEqual(summary["requested_scene_seed"], 100000)
            self.assertEqual(summary["simulator_seed"], 100002)

    def test_legacy_controls_accept_failed_source(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            task = root / "place_bread_basket"
            save_json(task / "capture/config.json", dict(episodes=[episode("place_bread_basket", 100000, False, "capture")]))
            save_json(task / "summary.json", dict(sampling_seed=1234))
            def launch(folder, name, env):
                value = episode(name, 100000, True, "control")
                value["rollout_chunks"] = 5
                save_json(folder / "config.json", dict(episodes=[value]))
            run_controls(root, ["place_bread_basket"], launch)
            summary = json.loads((task / "summary.json").read_text())
            self.assertTrue(summary["control_extended_beyond_saved_noise"])
            with self.assertRaises(FileExistsError):
                run_controls(root, ["place_bread_basket"], launch)


if __name__ == "__main__":
    unittest.main()
