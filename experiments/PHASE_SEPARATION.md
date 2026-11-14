# Stage 00: observation-encoding phase separation


Run this before restarting the rendering benchmark. Runtime code lives in `wan_va/modules/phase_separation.py`. It moves the **transformer parameters and registered buffers** to CPU only during initial/new-observation VAE encoding, then restores them before prediction/cache updates. KV tensors and cache metadata remain on GPU and their identity/device/dtype are checked on every transfer. No history, resolution, noise seed, conditioning or denoising count is shortened. Sampling state remains FP32, model inputs/weights remain BF16, text encoder remains on CPU.

This is an opt-in **single-GPU unsharded path**. The normal server FSDP-wraps even at world size one; moving those managed parameters with `.to()` is inappropriate. `PHASE_SEPARATION=1` bypasses that wrapping and rejects world sizes other than one. The default path remains unchanged. Both stage-00 cup arms use the unsharded path, so their comparison isolates offloading rather than mixing it with wrapping. Success is also checked against the archived cup case; that alone is not a proof of equivalence to the FSDP path on every input. CPU RAM must accommodate transformer weights alongside the already-offloaded text encoder.

From the Snellius checkout, with the same working environment/model/data paths:

```bash
# Preview: manifest/source snapshot only; no sbatch calls.
python3 experiments/scripts/run_phase_separation.py \\
  --manifest experiments/stages/00_test_phase_separation_preview/manifest.json

# Submit: three one-episode jobs, each one A100 + 18 cores, one-hour limit.
python3 experiments/scripts/run_phase_separation.py --submit
```

Default output: `experiments/stages/00_test_phase_separation/`. Supply `--account ACCOUNT` if required. Use a new `--manifest` directory for another attempt. To first spend only on the two cup checks, use `--cup-only` with its own manifest directory; do not then overwrite that manifest to add the third job.

Jobs, in order:

1. **Cup control:** unsharded, no weight movement. Original-caption conditioning, sampling seed 1234, run/demo seeds 0.
2. **Cup with offload:** identical settings, transformer offloaded only while encoding observations. Compare action hashes, task outcome and memory/latency.
3. **Microwave with offload:** original-caption conditioning, sampling seed 100000, unchanged history window. This is a memory-capacity test; a completed task failure is still useful. It must pass the earlier OOM region. Reaching 32 observation updates provides evidence at the configured history bound; a shorter successful episode does not establish that.

Jobs use **afterok** dependencies. Cup jobs exit nonzero if the episode does not succeed; subsequent jobs then remain dependency-blocked without consuming GPU runtime. Cancel blocked jobs using their recorded IDs if a cup check fails. The offload cup job also requires exact action-hash parity with its control; a mismatch exits nonzero and blocks the microwave job. Inspect the report before adopting offloading. No stages 1–4 are auto-submitted.

The approximate planning allowance is 8 minutes for each cup + 25 minutes for microwave + five setup minutes/job, about 119.5 SBUs total. The three requested one-hour wall limits have a 384-SBU ceiling. These estimates are not transfer-performance measurements. Full transformer transfers can be expensive; stage 00 measures them before scale-up.

```bash
python experiments/analysis/analyze_phase_separation.py \\
  experiments/stages/00_test_phase_separation/manifest.json

# Existing general timing/outcome/accounting plots also work:
python experiments/analysis/analyze.py \\
  experiments/stages/00_test_phase_separation/manifest.json --sacct
```

Each run contains `phase_separation.jsonl` with allocated/reserved/free GPU memory before movement, during encoding and after weight restoration; actual KV/metadata bytes; encoding peak allocation; offload/restore/encoding times; and exception status. `phase_actions.jsonl` contains per-chunk hashes of returned actions. Existing `gpu.csv` captures whole-GPU usage including the simulator. PyTorch allocation and whole-GPU usage are different quantities.

The analysis writes `phase_summary.json`, `phase_runs.csv`, `phase_events.csv` and `phase_separation.png` under the stage's `analysis/`. It checks exact cup action-hash parity and reports whether microwave reached the history bound. Missing/incomplete results do not pass. `ready_for_stage_1_to_4` deliberately remains false pending manual review: stage 00 validates **observation encoding only**, not memory during external reward backward. Inspect any parity failure, remaining OOM, or unacceptable transfer overhead before proceeding.

After accepting stage-00 results, use `--phase-separation` on `robotwin_suite.py` commands to explicitly retain the validated placement for later runs, e.g.:

```bash
python3 experiments/scripts/robotwin_suite.py render --phase-separation \\
  --manifest experiments/stages/01-render-phase/manifest.json --submit
```

This does not yet offload around DINO-guidance backward passes; those need their own memory validation before scaling stages 3–4. Do not infer guidance safety from unguided completion. Baselines and guided comparisons must ultimately use a documented consistent configuration.

Local tests exercise restored placement on encoding exceptions, cache identity/dtype preservation, the no-op disabled path, and stage submission/gating. A CUDA test is provided but skipped when CUDA is unavailable:

```bash
python -m pytest -q tests/test_phase_separation.py experiments/tests/test_phase_stage.py
```

Do not change source while stage-00 jobs are queued: source verification rejects drift. Existing run logs/manifests are not modified by this implementation.
