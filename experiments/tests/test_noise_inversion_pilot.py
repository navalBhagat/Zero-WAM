"""Pilot sequencing without a GPU or simulator."""
import json
import tempfile
import unittest
from pathlib import Path

from experiments.noise_inversion.pilot import run_pilot, run_controls


class PilotTests(unittest.TestCase):
    def exercise(self, successful):
        events = []

        def launch(directory, task, env):
            events.append((env["NOISE_INVERSION_MODE"], env))
            directory.mkdir()
            episode = dict(status="completed", success=successful, simulator_seed=100002,
                           task_instruction="Put bread in basket", rollout_chunks=3)
            (directory / "config.json").write_text(json.dumps(dict(episodes=[episode])))

        def infer(request):
            events.append(("invert", request))
            return dict(inversion=dict(chunks=3, relative_errors=[0., 0., 0.]))

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "pilot"
            run_pilot(root, ["place_bread_basket"], 1234, 0, infer, launch)
            summary = json.loads((root / "place_bread_basket/summary.json").read_text())
        return events, summary

    def test_success_runs_three_phases(self):
        events, summary = self.exercise(True)
        self.assertEqual([event[0] for event in events], ["capture", "invert", "replay"])
        self.assertEqual(events[0][1]["SAMPLING_SEED"], "1234")
        self.assertEqual(events[0][1]["USE_TEXT"], "1")
        self.assertEqual(events[2][1]["USE_ICL"], "0")
        self.assertEqual(events[2][1]["USE_TEXT"], "0")
        self.assertNotIn("prompt", events[1][1])
        self.assertEqual(events[2][1]["USE_GUIDANCE"], "0")
        self.assertEqual(summary["simulator_seed"], 100002)

    def test_control_appends_to_existing_pilot(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            task = root / "place_bread_basket"
            capture = task / "capture"
            capture.mkdir(parents=True)
            episode = dict(status="completed", success=True, simulator_seed=100007, icl_seed=0,
                server=dict(video_state_dtype="torch.float32", compact_observation_cache=True,
                            phase_separation=False),
                rendering=dict(samples_per_pixel_effective=16))
            (capture / "config.json").write_text(json.dumps(dict(episodes=[episode])))
            (task / "summary.json").write_text(json.dumps(dict(sampling_seed=1234, source_chunks=5)))

            def launch(output, name, environment):
                self.assertEqual(output, task / "control")
                self.assertEqual(environment["USE_TEXT"], "0")
                self.assertEqual(environment["START_SEED"], "100007")
                self.assertEqual(environment["NOISE_INVERSION_MODE"], "control")
                output.mkdir()
                result = dict(status="completed", success=True, rollout_chunks=7)
                (output / "config.json").write_text(json.dumps(dict(episodes=[result])))

            run_controls(root, ["place_bread_basket"], launch)
            summary = json.loads((task / "summary.json").read_text())
            self.assertTrue(summary["control_success"])
            self.assertFalse(summary["control_success_within_source_horizon"])
            self.assertTrue(summary["control_extended_beyond_saved_noise"])
            with self.assertRaises(FileExistsError):
                run_controls(root, ["place_bread_basket"], launch)

    def test_source_failure_skips_inversion_and_replay(self):
        events, summary = self.exercise(False)
        self.assertEqual(len(events), 1)
        self.assertEqual(summary["status"], "source_failed_inversion_skipped")


if __name__ == "__main__":
    unittest.main()
