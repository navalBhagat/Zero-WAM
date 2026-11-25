# Code layout and checks

- `wan_va/noise_inversion/online.py`: teacher/student context switching and actions.
- `collection.py`, `solver.py`: fixed-history targets and numerical integration.
- `adapter.py`: residual ViT/MLP, independent of backbone model imports.
- `experiments/noiseicl/`: configuration, indexing, planning, submission, execution,
  training and reporting, with no implicit submissions during imports/planning.
- `experiments/runtime/`: atomic records, feasibility search, source snapshots and timing.
- `experiments/scripts/clusters/`: existing Conda/workspace/module initialization.
- `experiments/jobs/{das6,snellius}/`: resource requests only.
- `wan_va/diagnostics/`: workload/FLOP accounting, independent of reward guidance.

The non-guidance tests remain in `tests/` and `experiments/tests/`. Run pytest in
the installed inference/training environment; backbone/dataset tests additionally
require the pinned Torch, Diffusers and LeRobot dependencies. CPU-only adapter,
solver, grouping, metadata and orchestration tests avoid loading model weights.

Useful diagnostics remain: per-camera timing and camera-only checks, faulthandler
watchdogs, episode timing profiles, GPU utilization/memory samples, cache-size
logs, component/FLOP accounting, package snapshots and source hashes. Renderer
initialization alone does not establish camera stability. Matched runs retain
ray tracing, OIDN, depth 8 and 16 samples/pixel; rasterization is diagnostic only.
