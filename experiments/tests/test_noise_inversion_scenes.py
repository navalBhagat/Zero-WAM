"""Feasibility replacement preserves paired scenes and never masks crashes."""
import tempfile
import unittest
from pathlib import Path

from experiments.io import read_json
from experiments.noise_inversion.scenes import ProbeResult, SceneSearch, select_scene, setup_rejection


class SceneSearchTests(unittest.TestCase):
    def test_rejects_are_recorded_and_reserved_slots_are_not_consumed(self):
        with tempfile.TemporaryDirectory() as temporary:
            search = SceneSearch(Path(temporary) / "scenes.json", 100000, (100000, 100001), 5)
            seen = []
            def probe(seed):
                seen.append(seed)
                return ProbeResult(seed == 100003, dict(seed=seed), "expert failed")
            seed, info = select_scene(search, probe)
            self.assertEqual(seen, [100000, 100002, 100003])
            self.assertEqual(seed, 100003)
            self.assertEqual(info["seed"], seed)
            self.assertEqual(search.accepted_seed(), seed)
            entries = read_json(search.registry)["slots"]["100000"]["attempts"]
            self.assertEqual([a["status"] for a in entries], ["infeasible", "infeasible", "accepted"])
            self.assertEqual(entries[0]["reason"], "expert failed")

    def test_second_noise_uses_the_same_accepted_scene(self):
        with tempfile.TemporaryDirectory() as temporary:
            search = SceneSearch(Path(temporary) / "scenes.json", 10, (10, 11))
            select_scene(search, lambda seed: ProbeResult(seed != 10, {}))
            seen = []
            def recheck(seed):
                seen.append(seed)
                return ProbeResult(True, {})
            select_scene(search, recheck)
            self.assertEqual(seen, [12])

    def test_different_slots_cannot_claim_the_same_replacement(self):
        with tempfile.TemporaryDirectory() as temporary:
            registry = Path(temporary) / "scenes.json"
            accepted = []
            for slot in (10, 11):
                seed, _ = select_scene(SceneSearch(registry, slot, (10, 11)),
                                       lambda seed: ProbeResult(seed > 11, {}))
                accepted.append(seed)
            self.assertEqual(accepted, [12, 13])

    def test_crashes_do_not_advance_the_candidate(self):
        with tempfile.TemporaryDirectory() as temporary:
            search = SceneSearch(Path(temporary) / "scenes.json", 10, (10,))
            def crash(seed):
                raise RuntimeError("renderer crashed")
            with self.assertRaisesRegex(RuntimeError, "renderer crashed"):
                select_scene(search, crash)
            entry = read_json(search.registry)["slots"]["10"]
            self.assertEqual(entry["candidate_seed"], 10)
            self.assertEqual(entry["attempts"][0]["status"], "error")
            seed, _ = select_scene(search, lambda seed: ProbeResult(True, {}))
            self.assertEqual(seed, 10)

    def test_attempt_limit_is_persistent_across_retries(self):
        with tempfile.TemporaryDirectory() as temporary:
            search = SceneSearch(Path(temporary) / "scenes.json", 10, (10,), 2)
            for _ in range(2):
                with self.assertRaisesRegex(ValueError, "exhausted 2"):
                    select_scene(search, lambda seed: ProbeResult(False, {}, "infeasible"))
            self.assertEqual(len(read_json(search.registry)["slots"]["10"]["attempts"]), 2)

    def test_only_the_observed_grasp_assertion_is_infeasibility(self):
        self.assertTrue(setup_rejection(AssertionError("target_pose cannot be None for move action.")))
        self.assertFalse(setup_rejection(AssertionError("unexpected internal bug")))
        self.assertFalse(setup_rejection(RuntimeError("CUDA out of memory")))

    def test_capture_receipt_survives_lost_registry_and_recovers_other_noise(self):
        from experiments.noise_inversion.records import save_json
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            receipt = root / "scene_10_noise_1/capture/scene_selection.json"
            first = SceneSearch(root / "scenes.json", 10, (10, 11), receipt=receipt)
            select_scene(first, lambda seed: ProbeResult(seed == 12, {}))
            save_json(root / "scenes.json", dict(schema_version=1, slots={}))
            self.assertEqual(first.accepted_seed(), 12)
            second = SceneSearch(root / "scenes.json", 10, (10, 11),
                                 receipt=root / "scene_10_noise_2/capture/scene_selection.json")
            checked = []
            def probe(seed):
                checked.append(seed)
                return ProbeResult(True, {})
            select_scene(second, probe)
            self.assertEqual(checked, [12])
            self.assertEqual(second.accepted_seed(), 12)

    def test_historical_completed_capture_recovers_missing_acceptance(self):
        from experiments.noise_inversion.records import save_json
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            receipt = root / "scene_10_noise_1/capture/scene_selection.json"
            save_json(receipt.parent / "config.json", dict(episodes=[
                dict(status="completed", success=False, simulator_seed=10)]))
            search = SceneSearch(root / "scenes.json", 10, (10, 11), receipt=receipt)
            self.assertEqual(search.accepted_seed(), 10)
            self.assertEqual(read_json(receipt)["accepted_seed"], 10)

    def test_previously_accepted_scene_is_never_silently_replaced(self):
        with tempfile.TemporaryDirectory() as temporary:
            search = SceneSearch(Path(temporary) / "scenes.json", 10, (10,))
            select_scene(search, lambda seed: ProbeResult(True, {}))
            with self.assertRaisesRegex(ValueError, "no longer feasible"):
                select_scene(search, lambda seed: ProbeResult(False, {}, "changed setup"))
            self.assertEqual(search.accepted_seed(), 10)


if __name__ == "__main__":
    unittest.main()
