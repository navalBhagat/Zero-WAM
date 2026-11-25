# NoiseICL on Zero-WAM

Transfer human-demonstration information into the initial video noise of a frozen
Zero-WAM. [Installation](INSTALL.md) · [Upstream model/training](old_readme.md) ·
[Campaigns](experiments/README.md) · [Method](experiments/docs/noise_inversion.md)

At each robot chunk, the oracle generates an ICL video prediction using the human
demo and current robot history. It removes the human context, rebuilds that same
robot history, reverses the predicted video into noise, and generates actions
without ICL or language. Only the final actions advance the simulator.

The learned adapter replaces this teacher/reversal operation with a residual
patch transformer. It receives Gaussian noise, human video latents, robot history
and prior actions; the Zero-WAM backbone still receives no human context or text.

External DINO/reward guidance is removed. Native ICL classifier-free guidance
remains part of the teacher. The new campaigns use empty text embeddings even for
the teacher; they must not be pooled with older cached-caption experiments.

Five campaigns are implemented: seven-task smoke evaluation, multi-scene signal
evaluation, a 100-pair collection benchmark, grouped dataset collection, and
adapter training with seen/unseen evaluation. Historical results are preserved;
old guidance campaigns are not supported.

Run `python experiments/scripts/noiseicl.py --help`. Planning never submits jobs;
submission is an explicit, bounded command. Start with one full task before
scaling. GPU execution and task success must be checked on the target cluster.
