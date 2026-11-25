# Running and resuming

See [campaign commands](../README.md) for the five stages. Plan on the cluster
where execution will occur so absolute data/output paths resolve correctly.
Use a new campaign directory after code changes. Plan creation does not submit.

Jobs are independent except stage-5 evaluation, which depends on training.
Inspect Slurm state before retrying; do not run two workers against the same
trial. Failed policy execution is a completed episode, whereas setup, renderer,
solver or infrastructure failures leave the job incomplete. Failures stop the
job instead of spending its remaining allocation on a broken server.

Completed jobs are skipped by `submit`. For inactive incomplete jobs, use
`submit --retry --limit N`. Existing completed phases are reused; incomplete
phase/pair directories move to attempts/. Checkpoint resume restores optimizer
and RNG state. Completed numerical inference does not establish task success.

GPU validation is required on the target cluster. CUDA/Diffusers/LeRobot backbone
tests must run in their installed environments; the CPU-only checks cover the
mathematics, orchestrator, grouping, adapter and metadata boundaries.
