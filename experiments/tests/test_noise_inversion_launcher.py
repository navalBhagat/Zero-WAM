"""Exercise the actual launcher from a simulated Slurm spool directory."""
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


LAUNCHER = Path(__file__).resolve().parents[1] / "jobs/noise_inversion.sbatch"


class LauncherTests(unittest.TestCase):
    def run_launcher(self, explicit_root: bool, valid_checkout: bool = True):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkout = root / "checkout with spaces"
            jobs = checkout / "experiments/jobs"
            jobs.mkdir(parents=True)
            if valid_checkout:
                (jobs / "eval.sbatch").write_text(
                    'test "$NOISE_INVERSION_PILOT" = 1 || exit 2\n'
                    'printf "%s\\n" "$ZERO_WAM_ROOT"\n')
            spool = root / "spool"
            spool.mkdir()
            script = spool / "slurm_script"
            shutil.copyfile(LAUNCHER, script)
            environment = dict(os.environ, SLURM_JOB_ID="29492245")
            environment.pop("ZERO_WAM_ROOT", None)
            environment["SLURM_SUBMIT_DIR"] = str(checkout)
            if explicit_root:
                environment["ZERO_WAM_ROOT"] = str(checkout)
                environment["SLURM_SUBMIT_DIR"] = str(spool)
            result = subprocess.run(["bash", str(script)], cwd=spool,
                                    env=environment, text=True, capture_output=True)
            return result, str(checkout)

    def test_submit_directory_resolves_checkout_from_spool(self):
        result, checkout = self.run_launcher(False)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip(), checkout)

    def test_explicit_checkout_overrides_submit_directory(self):
        result, checkout = self.run_launcher(True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip(), checkout)

    def test_missing_checkout_has_actionable_error(self):
        result, _ = self.run_launcher(False, valid_checkout=False)
        self.assertEqual(result.returncode, 1)
        self.assertIn("set ZERO_WAM_ROOT", result.stderr)


if __name__ == "__main__":
    unittest.main()
