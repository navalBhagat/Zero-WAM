"""Ordered Snellius experiments. Dry-run by default; never submits later stages."""
import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

# Also works when invoked as python experiments/scripts/robotwin_suite.py.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from experiments.scripts.provenance import snapshot

ROOT = Path(__file__).resolve().parents[2]
MODES = {'text': (0, 1, 0), 'icl': (1, 0, 0), 'guidance': (0, 0, 1),
         'icl_guidance': (1, 0, 1)}
# Do not let an earlier interactive experiment silently contaminate a new stage.
CONTROLLED = ('COMPACT_OBSERVATION_CACHE', 'SAVE_ROLLOUT_SNAPSHOTS', 'PHASE_COMPARE_ACTIONS', 'PHASE_SEPARATION', 'PHASE_OFFLOAD_TRANSFORMER', 'STAGE_REQUIRE_SUCCESS', 'GUIDANCE_CAPTURE', 'GUIDANCE_SELECTION', 'GUIDANCE_REFERENCE_FRAMES',
              'GUIDANCE_MATCH_WINDOW', 'GUIDANCE_LOOKAHEAD', 'GUIDANCE_STRENGTH',
              'GUIDANCE_EVERY', 'GUIDANCE_MAX_NORM', 'GUIDANCE_REFERENCE_SOURCE',
              'GUIDANCE_DECODER_DTYPE', 'GUIDANCE_STATE_DTYPE', 'VIDEO_STATE_FP32',
              'USE_ICL', 'USE_TEXT', 'USE_GUIDANCE', 'TEXT_GUIDANCE_SCALE',
              'ICL_GUIDANCE_SCALE', 'TARGET_TEXT_SOURCE', 'ICL_TEXT_SOURCE', 'SEED', 'SAMPLING_SEED', 'ICL_SEED')


def specifications(a):
    tasks = a.tasks or (['place_object_scale'] if a.stage == 'render' else
                       ['place_object_scale', 'place_bread_basket'])
    seeds = a.seeds or ([0] if a.stage == 'render' else [0, 1, 2])
    specs = []
    for task in tasks:
        for seed in seeds:
            if a.stage == 'render':
                # Rotate submission order; three exact repetitions of each paired setting.
                variants = [(f'spp{s}_r{r}', 'icl', s, 'fixed', 'float32', r)
                            for r in range(a.repeats)
                            for s in ((32, 16, 8)[r % 3:] + (32, 16, 8)[:r % 3])]
            elif a.stage == 'baselines':
                variants = [(m, m, a.spp, 'fixed', 'float32', 0) for m in ('text', 'icl')]
            elif a.stage == 'guidance':
                variants = [(f'{m}_{sel}', m, a.spp, sel, 'float32', 0)
                            for m in ('guidance', 'icl_guidance') for sel in ('fixed', 'window')]
            else:
                variants = [(f'{m}_{dtype}', m, a.spp, 'window', dtype, 0)
                            for m in ('guidance', 'icl_guidance') for dtype in ('float32', 'float16')]
            for label, mode, spp, selector, dtype, repeat in variants:
                icl, text, guide = MODES[mode]
                env = dict(ZERO_WAM_ROOT=str(ROOT), TASK=task, SEED=str(seed),
                    ICL_SEED=str(seed), SAMPLING_SEED=str(a.sampling_seed),
                    TEST_NUM=str(1 if a.stage == 'render' else a.episodes),
                    USE_ICL=str(icl), USE_TEXT=str(int(bool(text or icl))), USE_GUIDANCE=str(guide),
                    ICL_TEXT_SOURCE='cached' if icl else 'target', TARGET_TEXT_SOURCE='instruction' if text else 'empty',
                    COMPACT_OBSERVATION_CACHE=str(int(a.compact_cache)),
                    ICL_GUIDANCE_SCALE='5', TEXT_GUIDANCE_SCALE='5' if mode=='text' else '1',
                    RENDER_SAMPLES_PER_PIXEL=str(spp), PROFILE_TIMING='1', PROFILE_SYNC_GPU='0',
                    PHASE_SEPARATION=str(int(getattr(a, 'phase_separation', False))), PHASE_OFFLOAD_TRANSFORMER='1', STAGE_REQUIRE_SUCCESS='0', MONITOR_GPU='1', TEXT_ENCODER_OFFLOAD='1', VIDEO_STATE_FP32='1',
                    GUIDANCE_DECODER_DTYPE=dtype, GUIDANCE_SELECTION=selector,
                    GUIDANCE_MATCH_WINDOW=str(a.window), GUIDANCE_LOOKAHEAD=str(a.lookahead),
                    GUIDANCE_REFERENCE_FRAMES=str(a.fixed_frame), GUIDANCE_STRENGTH=str(a.strength),
                    GUIDANCE_EVERY='10', GUIDANCE_MAX_NORM='1', GUIDANCE_REFERENCE_SOURCE='auto',
                    CLIENT_TIMEOUT=str(a.client_timeout), EXPERIMENT_LABEL=label,
                    EXPERIMENT_STAGE=a.stage, EXPERIMENT_REPEAT=str(repeat),
                    EXPERIMENT_OUTPUT_ROOT=str(a.manifest.resolve().parent/'runs'))
                specs.append(dict(task=task, seed=seed, label=label, mode=mode, repeat=repeat,
                                  environment=env, status='planned'))
    return specs


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage', choices=['render', 'baselines', 'guidance', 'precision'])
    p.add_argument('--tasks', nargs='+', choices=['place_object_scale', 'place_bread_basket'])
    p.add_argument('--sampling-seed', type=int, default=1234)
    p.add_argument('--compact-cache', action='store_true', help='Opt in after stage-00 validation')
    p.add_argument('--seeds', nargs='+', type=int)
    p.add_argument('--episodes', type=int, default=10)
    p.add_argument('--repeats', type=int, default=3)
    p.add_argument('--spp', type=int, choices=[8, 16, 32], default=32)
    p.add_argument('--window', type=int, default=24)
    p.add_argument('--lookahead', type=int, default=4)
    p.add_argument('--fixed-frame', default='last', help='Zero-based RGB index or last')
    p.add_argument('--strength', type=float, default=.01)
    p.add_argument('--partition', default='gpu_a100')
    p.add_argument('--time', default=None, help='Wall limit per job; raise for multi-episode runs')
    p.add_argument('--client-timeout', type=int, default=None)
    p.add_argument('--account')
    p.add_argument('--job-script', type=Path, default=ROOT/'experiments/jobs/eval.sbatch')
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--minutes-per-episode', type=float, default=10)
    p.add_argument('--startup-minutes', type=float, default=5, help='Planning allowance, not measured')
    p.add_argument('--sbu-per-hour', type=float, default=128)
    p.add_argument('--budget-sbu', type=float, default=12202 + 52/60)
    p.add_argument('--submit', action='store_true')
    p.add_argument('--phase-separation', action='store_true', help='Opt into single-GPU observation-encoding offload after stage-00 validation')
    return p


def main():
    p = parser(); a = p.parse_args()
    a.time = a.time or ('01:00:00' if a.stage == 'render' else '03:00:00')
    a.client_timeout = a.client_timeout or (2400 if a.stage == 'render' else 9000)
    if a.sampling_seed < 0:
        p.error('Sampling seed must be nonnegative')
    if min(a.episodes, a.repeats, a.window, a.client_timeout) < 1 or a.lookahead < 0:
        p.error('Invalid counts/window/frame settings')
    if a.fixed_frame != 'last':
        try:
            if int(a.fixed_frame) < 0: raise ValueError
        except ValueError:
            p.error('--fixed-frame must be last or a nonnegative integer')
    if min(a.minutes_per_episode, a.sbu_per_hour, a.budget_sbu, a.strength) <= 0 or a.startup_minutes < 0:
        p.error('Invalid timing, budget, or guidance strength')
    if a.manifest.exists():
        p.error('Use a new manifest path; existing manifests are never overwritten')
    if not a.job_script.is_file():
        p.error(f'Job script missing: {a.job_script}')
    specs = specifications(a)
    count = sum(int(r['environment']['TEST_NUM']) for r in specs)
    estimated = (count*a.minutes_per_episode + len(specs)*a.startup_minutes)/60*a.sbu_per_hour
    try:
        parts = [int(v) for v in a.time.split(':')]
        if len(parts) != 3 or min(parts) < 0 or parts[1] >= 60 or parts[2] >= 60:
            raise ValueError
        wall_hours = parts[0] + parts[1]/60 + parts[2]/3600
        if wall_hours <= 0: raise ValueError
    except ValueError:
        p.error('--time must be positive HH:MM:SS')
    if max(int(s['environment']['TEST_NUM']) for s in specs)*a.minutes_per_episode+a.startup_minutes > wall_hours*60:
        p.error('Estimated per-job runtime exceeds --time; reduce episodes or increase the limit')
    reservation_ceiling = len(specs)*wall_hours*a.sbu_per_hour
    manifest = dict(schema_version=1, stage=a.stage, submitted=a.submit, runs=specs,
                    estimate=dict(episodes=count, sbu=estimated, budget_sbu=a.budget_sbu,
                                  sbu_per_hour=a.sbu_per_hour, startup_minutes=a.startup_minutes,
                                  requested_walltime_sbu_ceiling=reservation_ceiling,
                                  note='Guidance may cost more; estimate assumes supplied episode time.'))
    print(f'{len(specs)} jobs / {count} episodes; estimated {estimated:.1f} SBUs '
          f'({estimated/a.budget_sbu:.1%} of budget); includes assumed startup.')
    print(f'Requested wall-time ceiling: {reservation_ceiling:.1f} SBUs '
          '(submission may reserve this amount; actual charge uses elapsed allocation).')
    if reservation_ceiling > a.budget_sbu:
        p.error('Requested wall-time ceiling exceeds --budget-sbu; reduce jobs or --time')
    if estimated > a.budget_sbu:
        p.error('Estimate exceeds --budget-sbu; reduce the experiment')
    if a.submit and a.stage in ('guidance', 'precision'):
        checkpoint = Path(os.environ.get('GUIDANCE_CHECKPOINT', ROOT/'checkpoints/dinov2-small'))
        if not (checkpoint/'config.json').is_file():
            p.error('Set GUIDANCE_CHECKPOINT to the local DINOv2 model directory')
    a.manifest.parent.mkdir(parents=True, exist_ok=True)
    bundle = a.manifest.with_suffix('.source.zip')
    if bundle.exists():
        p.error(f'Source bundle already exists: {bundle}')
    digest = snapshot(bundle)
    manifest['source_bundle'] = str(bundle.resolve())
    manifest['source_digest'] = digest
    for spec in specs:
        spec['environment']['EXPERIMENT_SOURCE_DIGEST'] = digest
    # Save complete planned set before submitting anything. Preserve provenance.
    def save():
        a.manifest.write_text(json.dumps(manifest, indent=2)+'\n')
    manifest['revision'] = subprocess.check_output(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], text=True).strip()
    manifest['dirty_diff'] = subprocess.check_output(['git', '-C', str(ROOT), 'diff'], text=True)
    save()
    previous = None
    for spec in specs:
        cmd = ['sbatch', '--parsable', '--export=ALL', f'--partition={a.partition}',
               f'--time={a.time}', '--output='+str(a.manifest.resolve().parent/'slurm_%j.out'),
               f'--job-name=zw_{spec["task"]}_{spec["label"]}', f'--chdir={ROOT}']
        if a.account:
            cmd += [f'--account={a.account}']
        # Sequential within this campaign: avoid shared Robotwin patch races and GPU contention.
        if previous:
            cmd += [f'--dependency=afterany:{previous}']
        cmd += [str(a.job_script.resolve())]
        spec['command'] = cmd
        print(' '.join(f'{k}={shlex.quote(v)}' for k, v in spec['environment'].items()), shlex.join(cmd))
        if a.submit:
            env = {k:v for k,v in os.environ.items() if k not in CONTROLLED}
            env.update(spec['environment'])
            try:
                spec['job_id'] = subprocess.check_output(cmd, text=True, env=env).strip().split(';')[0]
                spec['status'] = 'submitted'
                previous = spec['job_id']
            except (OSError, subprocess.CalledProcessError) as error:
                spec['status'] = 'submission_failed'; spec['error'] = str(error); save(); raise
        save()
    print(f'Manifest: {a.manifest}; '+('submitted sequentially' if a.submit else 'DRY RUN: nothing submitted'))


if __name__ == '__main__':
    main()
