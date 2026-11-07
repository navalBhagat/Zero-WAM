#!/bin/bash
#SBATCH --job-name=zerowam_download
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --time=00:15:00
#SBATCH --output=zerowam_download_%j.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=navaltbhagat@gmail.com

# Portable CPU-only template: set partition/time/account at submission to match
# local policy. Initialize cluster modules in your shell before submitting.
set -eo pipefail
: "${CONDA_BASE:?Set CONDA_BASE to your Conda installation directory}"
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate "${ZEROWAM_ENV:-zerowam}"
# Slurm spools this file; use the explicitly exported repository path instead.
: "${ZERO_WAM_ROOT:?Set ZERO_WAM_ROOT to your checkout before sbatch}"
cd "$ZERO_WAM_ROOT"
bash setup/zerowam/download.sh "$@"
