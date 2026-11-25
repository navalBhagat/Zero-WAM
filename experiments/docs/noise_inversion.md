# Matched-history noise inversion

For the current robot history o, human demo h and Gaussian video noise z:

1. Rebuild ICL context: demo KV plus the actual robot observations/prior actions.
2. Generate the teacher video latent y with z. The teacher uses no language.
3. Clear that context and rebuild the same robot history without the human demo.
4. Integrate the unconditioned flow backward from y to recovered noise z*.
5. Forward-integrate z* to measure relative L2 reconstruction error.
6. Generate video/actions without ICL or text, starting from z* and the original
   action noise. Execute only these actions in the simulator.
7. Encode the actual resulting observations and retain the executed normalized
   action chunk for the next history.

The teacher still uses human-demo KV. Contexts are rebuilt for isolation rather
than retaining two large GPU caches. The streaming VAE feature cache is preserved
across switches so encoding the next observation does not restart the video.
Rebuilding full contexts costs time; profile before optimizing cache reuse.

ICL, unconditioned control and oracle rollouts share scene slots and chunk-indexed
Gaussian seeds, but each follows its own closed-loop history. Within every oracle
chunk, teacher/inversion/student use identical history. This differs from the old
method, which replayed noises recovered from a separate source rollout.

Euler inversion and reconstruction use FP32 state and BF16 model evaluations.
The first observed frame is clamped and excluded from target regression/error.
Inversion error is not proof of task success. Failed policy episodes remain valid
results; solver failures, GPU errors and incomplete phases remain explicit errors.
There is no Gaussian fallback in the online oracle.

## Training targets

Collection queries the same oracle at fixed, recorded seen-task histories. The
shared context index references human latents, initial frame and history tensors;
per-seed files save original/recovered noise, teacher video, original action noise
and a valid-frame mask. Numerical error, seed, solver settings, code digest and
conditioning flags are recorded. Invalid/nonfinite targets and fallback are not
used for training; unsuccessful teacher trajectories are not filtered by success.

The adapter predicts `z + residual(z, demo, robot_history, prior_actions, chunk)`.
Noise is patchified, with spatial/temporal coordinates; context modalities have
separate embeddings. The ViT uses self-attention, context cross-attention,
RMSNorm and SwiGLU; its output head starts at zero. A patch MLP is also available.
The frozen backbone is not in the training graph. Text/task IDs are not adapter
inputs. Shuffled-demo evaluation checks whether the adapter uses its demo.

Keep all seven unseen tasks out of training/tuning. Use held-out seen scenes and
unseen tasks for closed-loop evaluation. Per-task success uses complete matched
trials as the denominator; report incompletes, rescues, regressions, paired
scene-level intervals, reconstruction quality, GPU time and peak/cache logs.
The smoke campaign is a technical check, not a success-rate estimate.
