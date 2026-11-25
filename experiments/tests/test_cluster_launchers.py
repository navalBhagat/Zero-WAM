"""Cluster bootstrapping works from Slurm spool copies without a GPU."""
import os
from pathlib import Path
import shutil
import subprocess

from experiments.runtime.provenance import files

ROOT = Path(__file__).resolve().parents[2]


def workspace(home, checkout):
    (home / "nbhagat-workspace-env.sh").write_text(f'''export ZERO_WAM_ROOT='{checkout}'
export ROBOTWIN_ROOT='{home}/Robotwin'
export NBHAGAT_WORKSPACE='{home}/workspace'
export XDG_CACHE_HOME="$NBHAGAT_WORKSPACE/cache"
export TMPDIR="$NBHAGAT_WORKSPACE/tmp"
module() {{ printf '%s\\n' "$*" >> "$HOME/modules.log"; }}
conda() {{ export CONDA_PREFIX="$NBHAGAT_WORKSPACE/conda/envs/$2"; }}
''')


def test_spooled_das6_job_honors_workspace_helper(tmp_path):
    checkout = tmp_path / "checkout with spaces"
    scripts = checkout / "experiments/scripts"
    scripts.mkdir(parents=True)
    workspace(tmp_path, checkout)
    (scripts / "run_evaluation.sh").write_text('test "$EXPERIMENT_CLUSTER" = das6 || exit 2\nprintf "%s" "$ZERO_WAM_ROOT"\n')
    spool = tmp_path / "slurm_script"
    shutil.copyfile(ROOT / "experiments/jobs/das6/noiseicl.sbatch", spool)
    result = subprocess.run(["bash", str(spool)], cwd=tmp_path, env=dict(os.environ, HOME=str(tmp_path)),
                            text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    assert result.stdout == str(checkout)


def test_server_environment_switch_does_not_change_client_environment(tmp_path):
    workspace(tmp_path, ROOT)
    program = f'''source "{ROOT}/experiments/scripts/clusters/das6.sh"
activate_experiment_environment robotwin
original="$CONDA_PREFIX"
(activate_experiment_environment zerowam; test "$CONDA_PREFIX" = "$NBHAGAT_WORKSPACE/conda/envs/zerowam") || exit 3
test "$CONDA_PREFIX" = "$original" || exit 4
'''
    result = subprocess.run(["bash", "-c", program], env=dict(os.environ, HOME=str(tmp_path)), capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    modules = (tmp_path / "modules.log").read_text()
    assert "load cuda12.1/toolkit/12.1" in modules
    assert "load cuda12.6/toolkit/12.6" in modules


def test_source_snapshot_covers_new_runtime_and_cluster_jobs():
    recorded = {str(path.relative_to(ROOT)) for path in files()}
    assert "experiments/jobs/das6/noiseicl.sbatch" in recorded
    assert "experiments/scripts/run_evaluation.sh" in recorded
    assert "experiments/noiseicl/training.py" in recorded
    assert "wan_va/noise_inversion/online.py" in recorded
