# Download Zero-WAM and HumanGen later

These scripts require an activated `zerowam` environment with `huggingface_hub`.
No GPU is needed. They do not install packages, modify environments, or start
inference. Do not start a download until you have enough personal quota.

## Authenticate once

```bash
conda activate zerowam
hf auth login
hf auth whoami
```

The Python downloader uses the saved token or HF_TOKEN. Never embed credentials
in a job file. It does not change HF_HOME, so batch jobs can use the same saved
login as the interactive shell. Xet chunks default to the chosen storage root
unless HF_XET_CACHE is already explicitly set.

## Plan first

From the repository root, specify a shared storage path with adequate quota:

```bash
bash setup/zerowam/download.sh all --storage-root /path/to/shared/zero-wam
```

This authenticates, retrieves metadata, and prints selected sizes and estimated
additional content. It creates the storage directory but downloads no file
content. `checkpoint` or `dataset` can be used instead of `all`.

Profiles:

| Profile | HumanGen selection |
| --- | --- |
| `eval` (default) | Fixed seven human MP4s and matching VAE latents from the repository's task map, plus ICL manifests |
| `robotwin` | All `robotwin_data/**`, `human_latents/robotwin/**`, `human_data/robotwin/**`, and ICL manifests |

Live evaluation does not need recorded robot videos/actions/latents. The full
simulation profile supports offline comparisons and subsequent project work;
robot-video VAE latents are nested inside `robotwin_data/<task>/latents/`.

## Download after storage is available

Check `quota -s`. Convert the remaining allowance to GiB; choose a conservative
value if the quota tool uses different units. `df -h` alone cannot tell you your
quota. Pass that value explicitly:

```bash
bash setup/zerowam/download.sh all \
  --storage-root /path/to/shared/zero-wam \
  --quota-free-gib YOUR_REMAINING_GIB --download --link
```

`--download` opts into transferring file content. Without it, the command only
plans. The required allowance includes selected missing files and a 2 GiB
temporary/cache margin. This estimate cannot reserve storage or account for
simultaneous writes from other jobs. It conservatively counts a mismatched or
incomplete file as needing its whole size. Quota failures can still occur.

The checkpoint is around 35 GB. Our last DAS5 scratch usage was about 33 GiB
against a 40 GiB soft quota: that was insufficient. Ask for more storage or move
to an appropriate project filesystem before downloading the checkpoint.

`--link` creates repository-relative links only after successful downloads. It
refuses to overwrite existing directories or conflicting links. Move existing
content deliberately first or omit `--link` and manage paths yourself.

Downloaded layout:

```text
/path/to/shared/zero-wam/
  zero-wam-posttrain-robotwin/
  HumanGen/
  download-manifest.json
  .xet-cache/                 # if Xet is used and not overridden
```

The manifest pins resolved Hub commits and records the selection/completion
state. Later runs resume the same revisions. To intentionally use a new revision,
use a new storage root; optional `--checkpoint-revision` and `--dataset-revision`
select an initial tag/commit. Switching from `eval` to `robotwin` reuses already
downloaded references at the saved dataset revision. Completion means requested
files exist with expected sizes, not a successful model/rollout test or an
independent cryptographic integrity audit.

Keep destination `.cache/huggingface` metadata. Rerun the same command to reuse
completed files. If 429 rate-limited, wait and check authentication; use only one
downloader per storage root. Default concurrency is two workers; `--workers 1`
reduces bursts further but cannot guarantee avoidance of quota/rate limits.

## CPU-only Slurm template

`download.sbatch` does not hard-code cluster modules or partitions. Initialize
required modules in the submission shell and export:

```bash
export ZERO_WAM_ROOT="$HOME/Zero-WAM"
export CONDA_BASE=/var/scratch/nbhagat/miniforge3
export ZEROWAM_ENV=zerowam
```

Plan job (metadata only):

```bash
sbatch setup/zerowam/download.sbatch all --storage-root /path/to/shared/zero-wam
```

Actual download (after increasing storage):

```bash
sbatch --time=YOUR_ALLOWED_WALLTIME setup/zerowam/download.sbatch all \
  --storage-root /path/to/shared/zero-wam \
  --quota-free-gib YOUR_REMAINING_GIB --download --link
```

Replace placeholders; set the CPU partition/account permitted on your cluster
via `sbatch` options. The template defaults to 15 minutes, which may be too short
for transfers. DAS5 has site/time-of-day limits; obtain the appropriate allowed
long-job arrangement instead of copying Snellius's `rome` partition or arbitrary
long walltimes. Jobs must have outbound network access and access to the same
credentials/storage. No GPU directives are present.

See [INSTALL.md](../../INSTALL.md) for the complete environment, simulator asset,
GPU-check, and evaluation sequence. Do not rerun completed installation stages
merely to download data.
