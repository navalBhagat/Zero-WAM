"""Submission, accounting and reporting contracts without Slurm/GPU access."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from experiments.noise_inversion import campaign
from experiments.noise_inversion.config import TASKS, Trial
from experiments.noise_inversion.records import save_json
from experiments.noise_inversion.report import collect, report


def fake_snapshot(path):
    path.write_text("snapshot")
    return "digest"


class CampaignTests(unittest.TestCase):
    def make_plan(self, root):
        with patch.object(campaign, "snapshot", side_effect=fake_snapshot):
            return campaign.plan(root, list(TASKS), [100000, 100001], [1234, 5678])

    def test_plan_grid_and_first_calibration_batch(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "campaign"
            manifest = self.make_plan(root)
            self.assertEqual(len(manifest["runs"]), 28)
            self.assertEqual(len(manifest["jobs"]), 14)
            self.assertTrue(all(len(job["run_paths"]) == 2 for job in manifest["jobs"]))
            self.assertEqual([r["settings"]["task"] for r in manifest["runs"][:7]], list(TASKS))
            self.assertEqual(manifest["runs"][0]["path"], "trials/place_bread_basket/scene_100000_noise_1234")
            self.assertTrue(all(not job["submissions"] for job in manifest["jobs"]))
            with self.assertRaises(FileExistsError):
                self.make_plan(root)

    def test_submit_limit_account_and_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "campaign"
            self.make_plan(root)
            with patch.object(campaign, "fingerprint", return_value=("digest", {})), \
                 patch.object(campaign.subprocess, "check_output", side_effect=["11\n", "12\n", "13\n"]) as sbatch:
                jobs = campaign.submit(root, "account_a", limit=2)
                self.assertEqual(jobs, ["11", "12"])
                command = sbatch.call_args_list[1].args[0]
                self.assertIn("--account=account_a", command)
                self.assertIn("--dependency=afterany:11", command)
                self.assertNotIn("--dependency=afterany:11", sbatch.call_args_list[0].args[0])
                jobs = campaign.submit(root, "account_b", limit=1)
                self.assertEqual(jobs, ["13"])
                data = json.loads((root / "manifest.json").read_text())
                self.assertEqual(data["jobs"][2]["submissions"][0]["account"], "account_b")
                self.assertEqual(sbatch.call_args.kwargs["env"]["NOISE_INVERSION_JOB_ROOT"], str(root / "jobs" / data["jobs"][2]["name"]))

    def test_account_is_optional_and_login_labels_remain_separate(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "campaign"
            self.make_plan(root)
            with patch.object(campaign, "fingerprint", return_value=("digest", {})), \
                 patch.object(campaign.getpass, "getuser", return_value="login_a"), \
                 patch.object(campaign.subprocess, "check_output", return_value="11\n") as sbatch:
                campaign.submit(root, None, limit=1)
                self.assertFalse(any(value.startswith("--account") for value in sbatch.call_args.args[0]))
                manifest = json.loads((root / "manifest.json").read_text())
                self.assertEqual(manifest["jobs"][0]["submissions"][0]["account"], "default-login:login_a")
                with patch.object(campaign.getpass, "getuser", return_value="login_b"):
                    campaign.submit(root, None, limit=1)

    def test_active_job_is_not_retried(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "campaign"
            self.make_plan(root)
            with patch.object(campaign, "fingerprint", return_value=("digest", {})), \
                 patch.object(campaign.subprocess, "check_output", return_value="11\n") as sbatch:
                campaign.submit(root, "account_a", limit=1)
                with patch.object(campaign, "active_job", return_value=True):
                    campaign.submit(root, "account_a", limit=1, retry=True)
                second = sbatch.call_args.kwargs["env"]["NOISE_INVERSION_JOB_ROOT"]
                self.assertNotEqual(second, str(root / "jobs/place_bread_basket_noise_1234"))

    def test_group_skips_completed_scenes_and_reuses_one_inference_callback(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "campaign"
            manifest = self.make_plan(root)
            group = manifest["jobs"][0]
            save_json(root / group["run_paths"][0] / "summary.json", dict(status="completed"))
            calls = []
            infer = object()
            def runner(path, trial, callback, **kwargs):
                calls.append((path, callback, kwargs["retry"]))
            result = campaign.run_group(root, group, manifest, infer, retry=True, runner=runner)
            self.assertEqual(result["skipped"], [group["run_paths"][0]])
            self.assertEqual(len(calls), 1)
            self.assertIs(calls[0][1], infer)
            self.assertTrue(calls[0][2])

    def test_group_keeps_trying_other_scenes_after_trial_error(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "campaign"
            manifest = self.make_plan(root)
            group = manifest["jobs"][0]
            def runner(path, trial, infer, **kwargs):
                if trial.scene_seed == 100000:
                    raise RuntimeError("interrupted scene")
            result = campaign.run_group(root, group, manifest, object(), runner=runner)
            self.assertEqual(len(result["incomplete"]), 1)
            self.assertEqual(result["completed"], [group["run_paths"][1]])

    def test_costs_count_shared_job_once(self):
        import csv
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "campaign"
            manifest = self.make_plan(root)
            group = manifest["jobs"][0]
            save_json(root / "jobs" / group["name"] / "diagnostics/job_timing.json",
                      dict(allocated_seconds=3600, exit_code=1))
            report(root, plots=False)
            with (root / "costs.csv").open() as stream:
                costs = list(csv.DictReader(stream))
            self.assertEqual(len(costs), 14)
            self.assertEqual(sum(float(row["estimated_sbus"]) for row in costs), 128)
            _, tables = collect(root)
            self.assertEqual(tables[0]["incomplete"], 2)

    def test_provenance_includes_noise_package(self):
        from experiments.campaign.provenance import files
        self.assertTrue(any(str(path).endswith("experiments/noise_inversion/pilot.py") for path in files()))

    def test_source_drift_rejected_before_submission(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "campaign"
            self.make_plan(root)
            with patch.object(campaign, "fingerprint", return_value=("changed", {})):
                with self.assertRaisesRegex(ValueError, "Source changed"):
                    campaign.submit(root, "account_a")

    def test_common_denominator_rescue_and_failed_source_breakdown(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            runs = []
            outcomes = [(True, False, True), (False, True, False), (False, False, True)]
            for index in range(4):
                trial = Trial("place_bread_basket", 100000 + index, 1234)
                path = trial.directory(root)
                runs.append(dict(settings=trial.to_dict(), path=str(path.relative_to(root)), submissions=[]))
                if index == 3:
                    save_json(path / "summary.json", dict(status="incomplete", error="GPU crash"))
                    continue
                source, control, replay = outcomes[index]
                save_json(path / "summary.json", dict(status="completed", task=trial.task,
                    simulator_seed=trial.scene_seed, source_success=source, control_success=control,
                    replay_success=replay, replay_success_within_source_horizon=replay,
                    replay_extended_beyond_saved_noise=False))
            save_json(root / "manifest.json", dict(runs=runs, sbu_per_hour=128))
            _, table = collect(root)
            row = table[0]
            self.assertEqual(row["valid_pairs"], 3)
            self.assertEqual(row["incomplete"], 1)
            self.assertEqual(row["rescues"], 2)
            self.assertEqual(row["control_failures"], 2)
            self.assertEqual(row["regressions"], 1)
            self.assertEqual(row["source_failure_pairs"], 2)
            self.assertEqual(row["source_failure_replay_successes"], 1)
            report(root, plots=False)
            for name in ("trials.csv", "results.csv", "results.md", "costs.csv"):
                self.assertTrue((root / name).is_file())

    def test_feasibility_attempts_are_reported_separately_from_policy_failures(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            trial = Trial("place_object_scale", 100000, 1234)
            save_json(root / "manifest.json", dict(sbu_per_hour=128, runs=[dict(
                settings=trial.to_dict(), path=str(trial.directory(root).relative_to(root)), submissions=[])]))
            save_json(root / "trials/place_object_scale/scenes.json", dict(slots={"100000": dict(
                accepted_seed=100001, attempts=[dict(seed=100000, status="infeasible", reason="expert failed"),
                                                dict(seed=100001, status="accepted")])}))
            _, tables = collect(root)
            self.assertEqual(tables[0]["infeasible_scene_checks"], 1)
            self.assertEqual(tables[0]["valid_pairs"], 0)
            report(root, plots=False)
            self.assertIn("expert failed", (root / "scene_attempts.csv").read_text())

    def test_startup_failure_is_incomplete(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            trial = Trial("place_object_scale", 100000, 1234)
            path = trial.directory(root)
            save_json(path / "diagnostics/job_timing.json", dict(exit_code=1))
            save_json(root / "manifest.json", dict(runs=[dict(settings=trial.to_dict(), path=str(path.relative_to(root)), submissions=[])]))
            _, table = collect(root)
            self.assertEqual(table[0]["incomplete"], 1)
            self.assertIsNone(table[0]["source_success_pct"])


if __name__ == "__main__":
    unittest.main()
