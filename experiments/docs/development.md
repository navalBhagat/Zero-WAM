# Code organization and development

## Responsibilities

| Location | Responsibility |
| --- | --- |
| `scripts/` | Compatible CLI entry points; importing them does not submit jobs |
| `campaign/config.py`, `validation.py` | Grouped CLI flags, defaults and input validation |
| `campaign/plans.py`, `presets.py` | Ordered condition/task/seed expansion and named experiment presets |
| `campaign/cli.py`, `slurm.py` | Validate, budget, snapshot, save manifest and optionally submit |
| `campaign/resume.py` | Completed-prefix reuse and strict source/settings compatibility |
| `campaign/provenance.py` | Recursive source fingerprints, archives and queued-job drift checks |
| `campaign/budget.py`, `job_report.py` | Shared budget math and job lifecycle reporting |
| `replay/artifacts.py`, `reference.py`, `runtime.py` | Capture/reference loading and lazy guidance access |
| `replay/precision.py`, `temporal.py` | Captured-step measurement and recorded-observation target replay |
| `replay/temporal_plots.py` | Temporal diagnostic figures |
| `analysis/collect.py`, `metrics.py` | Artifact normalization, actual-scene pairing, timing and statistics |
| `analysis/compute.py` | Operator totals or coverage-labelled historical arithmetic estimates |
| `analysis/report.py`, `plots.py` | Compact summary rows and shared plotting backend |
| `io.py`, `paths.py` | Explicit serialization and repository-root utilities |
| `jobs/` | Slurm resources/environment activation; Python logic lives in named helpers |

Existing command paths and import APIs are retained through small wrappers. The
budget calculator and timing wrapper delegate to shared implementations. Compact
plots are no longer reimplemented by the diagnostic plotting path.

## Contracts to preserve

Manifest schema, flag defaults, environment exports, seed order, render repetition
rotation, CFG conditioning and resume behavior are compatibility contracts. Use
`Variant`, `BudgetEstimate` and `CapturedStep` for internal named data; serialized
run/episode rows remain dictionaries for historical compatibility.

Replay imports model dependencies lazily. CPU temporal replay and CLI help must not
import WAM modules, initialize distributed inference or require CUDA. GPU replay
loads only the reward models; numerical flow/reward code stays in `wan_va/guidance/`.

Source snapshot discovery must cover every nested source module and exclude
`stages/`, generated outputs, caches, checkpoints and data. Keep archive discovery
sorted/deterministic. Do not loosen resume checks to accommodate a refactor.

## Regression checks

From the repository root in an environment with PyTorch, pytest, Pillow and
Matplotlib:

```bash
python -m pytest -q experiments/tests tests/test_policy_flops.py \
  tests/test_guidance_temporal.py tests/test_run_metadata.py \
  tests/test_phase_separation.py tests/test_conditioning.py
bash -n experiments/jobs/eval.sbatch
```

When Ruff is available, use `ruff format --check experiments` and
`ruff check experiments --select F821` for formatting/undefined names.

Tests cover command generation without real Slurm submission, source drift and
nested-module snapshots, historical/compact layouts, missing/duplicate episodes,
CPU replay, and frozen-network forward/backward accounting. CUDA-only checks skip
without hardware. They do not replace a short GPU compatibility/coverage smoke test.
