"""Exercise spooled DAS6 launchers and per-process environment selection."""
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]


class ClusterLauncherTests(unittest.TestCase):
    def helper(self, home, checkout):
        helper = home / "nbhagat-workspace-env.sh"
        # Paths intentionally contain spaces to cover shell quoting.
        helper.write_text(f'''export ZERO_WAM_ROOT='{checkout}'
export ROBOTWIN_ROOT='{home}/Robotwin'
export NBHAGAT_WORKSPACE='{home}/workspace'
export XDG_CACHE_HOME="$NBHAGAT_WORKSPACE/cache"
export TMPDIR="$NBHAGAT_WORKSPACE/tmp"
module() {{ printf '%s\\n' "$*" >> "$HOME/modules.log"; }}
conda() {{ export CONDA_PREFIX="$NBHAGAT_WORKSPACE/conda/envs/$2"; }}
''')

    def test_spooled_campaign_sources_workspace_and_dispatches_shared_runner(self):
        with tempfile.TemporaryDirectory() as temporary:
            home = Path(temporary)
            checkout = home / "checkout with spaces"
            scripts = checkout / "experiments/scripts"
            scripts.mkdir(parents=True)
            self.helper(home, checkout)
            (scripts / "run_evaluation.sh").write_text('''test "$EXPERIMENT_CLUSTER" = das6 || exit 2
test "$NOISE_INVERSION_PILOT" = 1 || exit 3
test "$PHASE_SEPARATION" = 1 || exit 4
printf '%s\\n' "$ZERO_WAM_ROOT"
''')
            spool = home / "slurm_script"
            shutil.copyfile(ROOT / "experiments/jobs/das6/noise_inversion_campaign.sbatch", spool)
            env = dict(os.environ, HOME=str(home), TASK="stack_blocks_three",
                       NOISE_INVERSION_JOB_ROOT=str(home / "output"), NOISE_INVERSION_JOB_SETTINGS="settings.json")
            result = subprocess.run(["bash", str(spool)], cwd=home, env=env, text=True, capture_output=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(result.stdout.strip(), str(checkout))

    def test_cuda_and_conda_switch_in_separate_server_subprocess(self):
        with tempfile.TemporaryDirectory() as temporary:
            home = Path(temporary)
            self.helper(home, ROOT)
            program = f'''source "{ROOT}/experiments/scripts/clusters/das6.sh"
activate_experiment_environment robotwin
client_prefix="$CONDA_PREFIX"
(activate_experiment_environment zerowam; test "$CONDA_PREFIX" = "$NBHAGAT_WORKSPACE/conda/envs/zerowam") || exit 3
test "$CONDA_PREFIX" = "$client_prefix" || exit 4
'''
            result = subprocess.run(["bash", "-c", program], env=dict(os.environ, HOME=str(home)),
                                    text=True, capture_output=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            modules = (home / "modules.log").read_text()
            self.assertIn("load cuda12.1/toolkit/12.1", modules)
            self.assertIn("load cuda12.6/toolkit/12.6", modules)

    def test_provenance_covers_nested_cluster_jobs_and_shared_scripts(self):
        from experiments.campaign.provenance import files
        recorded = {str(path.relative_to(ROOT)) for path in files()}
        self.assertIn("experiments/jobs/das6/noise_inversion_campaign.sbatch", recorded)
        self.assertIn("experiments/scripts/clusters/das6.sh", recorded)
        self.assertIn("experiments/scripts/run_evaluation.sh", recorded)

    def test_shared_lifecycle_propagates_campaign_exit_and_cleans_up(self):
        for exit_code in (0, 7):
            with self.subTest(exit_code=exit_code), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                checkout = root / "checkout"
                scripts = checkout / "experiments/scripts"
                (scripts / "clusters").mkdir(parents=True)
                (checkout / "evaluation/robotwin").mkdir(parents=True)
                (checkout / "checkpoints/zero-wam-posttrain-robotwin").mkdir(parents=True)
                (checkout / "data/HumanGen/human_latents/robotwin").mkdir(parents=True)
                robotwin = root / "Robotwin"
                (robotwin / "policy/ACT").mkdir(parents=True)
                (robotwin / "policy/ACT/deploy_policy.yml").touch()
                binary = root / "bin"
                binary.mkdir()
                (scripts / "clusters/das6.sh").write_text('''activate_experiment_environment() {
    export CONDA_PREFIX="$HOME"
    export PATH="$HOME/bin:$PATH"
}
''')
                (checkout / "evaluation/robotwin/launch_server.sh").write_text('exec sleep 30\n')
                python = binary / "python"
                python.write_text('''#!/bin/bash
printf '%s\\n' "$*" >> "$HOME/python_calls.log"
if [[ "$1" == "experiments/scripts/noise_inversion_campaign.py" ]]; then exit "$TEST_EXIT"; fi
''')
                python.chmod(0o755)
                (binary / "python3").symlink_to(python)
                timeout = binary / "timeout"
                timeout.write_text('#!/bin/bash\nshift 2\nexec "$@"\n')
                timeout.chmod(0o755)
                for command in ("curl", "nvidia-smi"):
                    path = binary / command
                    path.write_text('#!/bin/bash\nexit 0\n')
                    path.chmod(0o755)
                env = dict(os.environ, HOME=str(root), ZERO_WAM_ROOT=str(checkout), ROBOTWIN_ROOT=str(robotwin),
                           EXPERIMENT_CLUSTER="das6", SLURM_JOB_ID="12345", TASK="place_empty_cup",
                           NOISE_INVERSION_JOB_ROOT=str(root / "output"), NOISE_INVERSION_PILOT="1",
                           NOISE_INVERSION_JOB_SETTINGS="settings.json", NOISE_INVERSION_SOURCE_DIGEST="digest",
                           MONITOR_GPU="0", RUN_DIAGNOSTICS="0", TEST_EXIT=str(exit_code))
                for key in ("EXPERIMENT_SOURCE_DIGEST", "CAMERA_STRESS_TEST", "NOISE_INVERSION_CONTINUE",
                            "NOISE_INVERSION_REPAIR_PILOT"):
                    env.pop(key, None)
                result = subprocess.run(["bash", str(ROOT / "experiments/scripts/run_evaluation.sh")],
                                        env=env, text=True, capture_output=True, timeout=10)
                self.assertEqual(result.returncode, exit_code, result.stderr)
                calls = (root / "python_calls.log").read_text()
                self.assertIn("noise_inversion_campaign.py run", calls)
                self.assertIn("job_report.py timing", calls)
                self.assertIn("--refresh", calls)
