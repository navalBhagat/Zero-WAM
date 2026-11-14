"""Analyze one campaign, including failed/missing runs. CPU-only; no simulator imports."""
import argparse
import ast
import csv
from collections import defaultdict
import json
import math
from pathlib import Path
import statistics
import subprocess


COMPONENTS = ['sampling', 'server_reset', 'cache_update', 'camera_render', 'planning',
              'physics', 'action_other', 'scene_setup', 'video_encoding', 'other_profiled', 'unprofiled']


def read(path, default=None):
    return json.loads(path.read_text()) if path.is_file() else default


def write_csv(path, rows):
    keys = list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader(); writer.writerows(rows)


def breakdown(profile):
    result = dict.fromkeys(COMPONENTS, 0.)
    for name, timing in profile.get('timings', {}).items():
        key = {'server_sampling_roundtrip':'sampling', 'server_reset':'server_reset',
               'server_cache_update':'cache_update', 'physics_step':'physics',
               'action_execution':'action_other', 'scene_setup':'scene_setup',
               'video_encoding':'video_encoding'}.get(name)
        if name in ('camera_capture', 'camera_readback', 'render_state_update'):
            key = 'camera_render'
        if name.startswith('planning_'):
            key = 'planning'
        result[key or 'other_profiled'] += timing.get('exclusive_seconds', 0.)
    result['unprofiled'] = max(0., profile['wall_seconds'] - sum(result.values()))
    return result


def log_records(path, marker):
    if not path.is_file():
        return []
    records = []
    for line in path.read_text(errors='replace').splitlines():
        if marker not in line:
            continue
        try:
            value = ast.literal_eval(line.split(marker, 1)[1].strip())
            if isinstance(value, dict):
                records.append(value)
        except (ValueError, SyntaxError):
            pass
    return records


def wilson(successes, n):
    if not n:
        return None, None
    z = 1.96; p = successes/n; d = 1 + z*z/n
    mid = (p+z*z/(2*n))/d
    half = z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/d
    return mid-half, mid+half


def collect(manifest, runs_root):
    runs, episodes, chunks, guidance = [], [], [], []
    for number, spec in enumerate(manifest['runs']):
        env = spec['environment']; job = spec.get('job_id')
        root = runs_root or Path(env['EXPERIMENT_OUTPUT_ROOT'])
        folder = root/f'{spec["task"]}_{job}'
        base = dict(run=number, job_id=job, task=spec['task'], label=(spec['label'].split('_r')[0] if manifest['stage']=='render' else spec['label']),
                    run_label=spec['label'],
                    seed=spec['seed'], repeat=spec.get('repeat', 0), mode=spec['mode'],
                    spp=int(env['RENDER_SAMPLES_PER_PIXEL']), selector=env['GUIDANCE_SELECTION'],
                    decoder_dtype=env['GUIDANCE_DECODER_DTYPE'])
        config = read(folder/'config.json', {})
        timing = read(folder/'job_timing.json', {})
        requested = int(env['TEST_NUM'])
        completed = [e for e in config.get('episodes', []) if e.get('status') == 'completed']
        run = dict(base, path=str(folder), requested=requested, completed=len(completed),
                   missing=max(0, requested-len(completed)), exit_code=timing.get('exit_code'),
                   allocated_seconds=timing.get('allocated_seconds'),
                   estimated_sbu=(timing['allocated_seconds']/3600*manifest['estimate']['sbu_per_hour']
                                  if timing.get('allocated_seconds') is not None else None),
                   startup_seconds=timing.get('server_startup_seconds'),
                   pre_server_seconds=timing.get('pre_server_seconds'),
                   client_seconds=timing.get('client_seconds'),
                   status=('complete' if len(completed)==requested and timing.get('exit_code')==0
                           else 'incomplete_or_failed'))
        runs.append(run)
        for episode in completed:
            profile_path = next(folder.glob(f'**/profile_episode_{episode["episode_id"]}.json'), None)
            profile = read(profile_path, {}) if profile_path else {}
            row = dict(base, episode_id=episode['episode_id'],
                       simulator_seed=episode.get('simulator_seed'),
                       sampling_seed=env['SAMPLING_SEED'], icl_seed=episode.get('icl_seed'),
                       success=int(episode['success']), actions=episode.get('policy_action_calls'),
                       wall_seconds=profile.get('wall_seconds'), profile_available=bool(profile),
                       step_limit_counter=episode.get('simulator_action_count'))
            if profile:
                row.update(breakdown(profile))
                t = profile.get('timings', {})
                calls = t.get('server_sampling_roundtrip', {}).get('calls', 0)
                row['chunks'] = calls
                row['sampling_seconds_per_chunk'] = row['sampling']/calls if calls else None
                row['seconds_per_action'] = profile['wall_seconds']/row['actions'] if row['actions'] else None
                row['camera_hooks'] = json.dumps({k:v for k,v in profile.get('hooks', {}).items()
                                                if 'camera' in k or 'physics' in k})
            episodes.append(row)
        profiled = [e['wall_seconds'] for e in episodes if e['run']==number and e.get('wall_seconds') is not None]
        run['profiled_episode_seconds'] = sum(profiled)
        run['client_outside_episode_seconds'] = (max(0,timing['client_seconds']-sum(profiled))
                                                if timing.get('client_seconds') is not None else None)
        if (folder/'gpu.csv').is_file():
            with (folder/'gpu.csv').open() as stream:
                samples=list(csv.DictReader(stream,skipinitialspace=True))
            for column,target,operation in [('memory.used [MiB]','gpu_peak_used_mib',max),
                                            ('utilization.gpu [%]','gpu_mean_utilization',statistics.mean)]:
                numbers=[]
                for sample in samples:
                    try: numbers.append(float(sample.get(column,'')))
                    except (ValueError,TypeError): pass
                run[target]=operation(numbers) if numbers else None
        for marker, target in [('[inference-profile]', chunks), ('[external-guidance]', guidance)]:
            for i, entry in enumerate(log_records(folder/'server.log', marker)):
                target.append(dict(base, call=i, **entry))
    return runs, episodes, chunks, guidance


def summaries(episodes):
    groups = defaultdict(list)
    for row in episodes:
        groups[(row['task'], row['label'], row['spp'])].append(row)
    rows = []
    for (task,label,spp), values in sorted(groups.items()):
        n = len(values); successes = sum(v['success'] for v in values)
        low,high = wilson(successes,n)
        row = dict(task=task,label=label,spp=spp,n=n,successes=successes,
                   success_rate=successes/n,wilson_low=low,wilson_high=high)
        for key in ['wall_seconds','seconds_per_action','sampling_seconds_per_chunk', *COMPONENTS]:
            xs = [v[key] for v in values if v.get(key) is not None]
            row[key+'_mean'] = statistics.mean(xs) if xs else None
            row[key+'_sd'] = statistics.stdev(xs) if len(xs)>1 else None
        rows.append(row)
    return rows


def pairs(episodes, stage):
    groups = defaultdict(list)
    for row in episodes:
        # Use actual accepted simulator seeds, not only nominal run seeds.
        key = (row['task'], row['simulator_seed'], row['sampling_seed'], row['icl_seed'], row['repeat'])
        groups[key].append(row)
    result = []
    for key, values in groups.items():
        for left in values:
            for right in values:
                if stage == 'render':
                    compare = left['spp']==32 and right['spp'] in (16,8)
                elif stage == 'textcheck':
                    compare = left['mode']=='icl' and right['mode'] in ('icl_legacy_text','icl_original_caption_only')
                elif stage == 'baselines':
                    compare = left['mode']=='text' and right['mode']=='icl'
                elif stage == 'guidance':
                    compare = left['mode']==right['mode'] and left['selector']=='fixed' and right['selector']=='window'
                else:
                    compare = left['mode']==right['mode'] and left['decoder_dtype']=='float32' and right['decoder_dtype']=='float16'
                if not compare:
                    continue
                row = dict(task=key[0],simulator_seed=key[1],repeat=key[4],left=left['label'],right=right['label'],
                           success_delta=right['success']-left['success'],
                           action_count_equal=left['actions']==right['actions'])
                for field in ('wall_seconds','seconds_per_action','sampling_seconds_per_chunk'):
                    a,b=left.get(field),right.get(field)
                    row[field+'_delta'] = b-a if a is not None and b is not None else None
                result.append(row)
    return result


def plots(output, runs, episodes, chunks, guidance):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    groups=defaultdict(list)
    for e in episodes:
        if e.get('wall_seconds') is not None:
            groups[(e['task'], e['label'])].append(e)
    if groups:
        keys=sorted(groups); x=list(range(len(keys)))
        fig,axes=plt.subplots(3,1,figsize=(max(9,len(keys)*.65),13),layout='constrained')
        bottom=[0.]*len(keys)
        for component in COMPONENTS:
            heights=[statistics.mean(e.get(component,0) for e in groups[k]) for k in keys]
            axes[0].bar(x,heights,bottom=bottom,label=component)
            bottom=[b+h for b,h in zip(bottom,heights)]
        axes[0].set_ylabel('Mean episode seconds'); axes[0].legend(fontsize=7,ncol=4)
        axes[0].set_title('Exclusive wall time: camera costs can include GPU waits; physics hooks may be unavailable')
        for axis,field,label in [(axes[1],'seconds_per_action','Episode seconds / executed action'),
                                 (axes[2],'sampling_seconds_per_chunk','Sampling roundtrip seconds / chunk')]:
            for i,k in enumerate(keys):
                ys=[e[field] for e in groups[k] if e.get(field) is not None]
                axis.scatter([i]*len(ys),ys,alpha=.7)
                if ys: axis.plot(i,statistics.mean(ys),'k_',markersize=18)
            axis.set_ylabel(label)
        for axis in axes:
            axis.set_xticks(x,[f'{t}\n{l}' for t,l in keys],rotation=60,ha='right',fontsize=7)
        fig.savefig(output/'timing.png',dpi=170); plt.close(fig)
    summary=summaries(episodes)
    if summary:
        fig,ax=plt.subplots(figsize=(max(8,len(summary)*.65),5),layout='constrained')
        for i,s in enumerate(summary):
            ax.errorbar(i,s['success_rate'],yerr=[[s['success_rate']-s['wilson_low']],
                        [s['wilson_high']-s['success_rate']]],fmt='o')
        ax.set_xticks(range(len(summary)),[f'{s["task"]}\n{s["label"]}\nn={s["n"]}' for s in summary],rotation=60,ha='right',fontsize=7)
        ax.set_ylim(-.05,1.05); ax.set_ylabel('Success rate (95% Wilson interval)')
        ax.set_title('Completed episodes only; missing/failed runs listed separately. Seed replicates are clustered.')
        fig.savefig(output/'success.png',dpi=170); plt.close(fig)
    if chunks:
        fig,ax=plt.subplots(figsize=(9,5),layout='constrained')
        byrun=defaultdict(list)
        for c in chunks: byrun[c['run']].append(c)
        for run,values in byrun.items():
            ax.plot([v['call'] for v in values],[v['total_seconds'] for v in values],label=f'run {run}')
        ax.set(xlabel='Inference call within job',ylabel='Server inference seconds',title='Server sampling (excludes client simulation)')
        ax.legend(fontsize=6,ncol=3); fig.savefig(output/'server_chunks.png',dpi=170); plt.close(fig)
    if guidance:
        fig,axes=plt.subplots(2,1,figsize=(10,7),layout='constrained')
        byrun=defaultdict(list)
        for g in guidance: byrun[g['run']].append(g)
        for run,values in byrun.items():
            axes[0].plot([v['call'] for v in values],[v['reward'] for v in values],label=f'run {run}')
            axes[1].plot([v['call'] for v in values],[v['seconds'] for v in values])
        axes[0].set(ylabel='DINO reward before correction',title='Descriptive only: targets and robot histories can differ across calls')
        axes[1].set(xlabel='Guidance call within job',ylabel='Guidance seconds'); axes[0].legend(fontsize=6,ncol=3)
        fig.savefig(output/'guidance.png',dpi=170); plt.close(fig)
    if runs:
        fig,ax=plt.subplots(figsize=(max(8,len(runs)*.4),5),layout='constrained')
        bottom=[0.]*len(runs)
        for field,label in [('pre_server_seconds','Pre-server setup'),('startup_seconds','Server startup'),('client_seconds','Client total')]:
            heights=[r.get(field) or 0 for r in runs]
            ax.bar(range(len(runs)),heights,bottom=bottom,label=label)
            bottom=[b+h for b,h in zip(bottom,heights)]
        residual=[max(0,(r.get('allocated_seconds') or 0)-b) for r,b in zip(runs,bottom)]
        ax.bar(range(len(runs)),residual,bottom=bottom,label='Cleanup / failed or unclassified time')
        ax.set(xlabel='Manifest run number',ylabel='Job-script seconds',title='Startup and client costs; missing timings are not zero-cost runs')
        ax.legend(fontsize=8); fig.savefig(output/'job_timing.png',dpi=170); plt.close(fig)


def accounting_from_sacct(raw, rate):
    rows=[]
    for record in csv.DictReader(raw.splitlines(), delimiter='|'):
        try:
            seconds=int(record['ElapsedRaw'])
        except (ValueError,KeyError,TypeError):
            continue
        resources={}
        for item in record.get('AllocTRES','').split(','):
            if '=' in item:
                key,value=item.split('=',1); resources[key]=value
        count=resources.get('gres/gpu')
        try: count=float(count)
        except (ValueError,TypeError): count=None
        rows.append(dict(job_id=record.get('JobIDRaw'),state=record.get('State'),
                         elapsed_seconds=seconds,gpus=count,exit_code=record.get('ExitCode'),
                         estimated_sbu=seconds/3600*count*rate if count is not None else None))
    return rows


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('manifest',type=Path); p.add_argument('--runs-root',type=Path)
    p.add_argument('--output',type=Path); p.add_argument('--sacct',action='store_true')
    a=p.parse_args(); manifest=read(a.manifest)
    output=a.output or a.manifest.parent/'analysis'; output.mkdir(parents=True,exist_ok=True)
    runs,episodes,chunks,guidance=collect(manifest,a.runs_root)
    paired=pairs(episodes,manifest['stage'])
    for name,rows in [('runs',runs),('episodes',episodes),('summary',summaries(episodes)),
                      ('paired',paired),('server_chunks',chunks),('guidance',guidance)]:
        write_csv(output/f'{name}.csv',rows)
    # Keep between-seed variability visible rather than treating every episode as independent.
    seed_rows=[]
    for seed in sorted({e['seed'] for e in episodes}):
        seed_rows += [dict(seed=seed,**s) for s in summaries([e for e in episodes if e['seed']==seed])]
    write_csv(output/'per_seed.csv',seed_rows)
    accounting=[]
    if a.sacct:
        ids=[str(r['job_id']) for r in runs if r['job_id']]
        if ids:
            result=subprocess.run(['sacct','-j',','.join(ids),'-X','-P','--units=G',
                '--format=JobIDRaw,State,ElapsedRaw,AllocTRES,ExitCode'],text=True,capture_output=True)
            (output/'sacct.txt').write_text(result.stdout+result.stderr)
            if result.returncode: print('WARNING: sacct failed; see sacct.txt')
            else: accounting=accounting_from_sacct(result.stdout,manifest['estimate']['sbu_per_hour'])
            write_csv(output/'accounting.csv',accounting)
    notes=['# Campaign analysis',f'Stage: {manifest["stage"]}',
           f'Requested episodes: {sum(r["requested"] for r in runs)}; completed: {len(episodes)}; '
           f'missing: {sum(r["missing"] for r in runs)}.',
           f'Matched comparisons: {len(paired)}. Inspect paired.csv; actual simulator seeds must match.',
           f'Job-script estimated cost for {sum(r["estimated_sbu"] is not None for r in runs)}/{len(runs)} timed jobs: '
           f'{sum(r["estimated_sbu"] or 0 for r in runs):.2f} SBUs. Missing timings are unaccounted; use sacct.',
           'Timing bars use exclusive durations; unavailable hooks remain in parent or unprofiled costs.',
           'Compare per-action and per-chunk costs alongside totals and action counts. '
           'Changing SPP can change behavior, not only rendering time.',
           'Wilson intervals are descriptive pooled intervals; use per_seed.csv for seed-level variability. '
           'Repeated rendering trials are not independent success-rate evidence.',
           'Do not compare absolute DINO reward across changing targets as evidence of improvement.']
    if accounting:
        notes.append(f"sacct elapsed-time estimated charge: {sum(r['estimated_sbu'] or 0 for r in accounting):.2f} SBUs "
                     f"across {sum(r['estimated_sbu'] is not None for r in accounting)}/{len(accounting)} "
                     "accounting rows with GPU counts. Pending/running rows are not final charges.")
    try:
        plots(output,runs,episodes,chunks,guidance)
    except ImportError as error:
        notes.append(f'Plots skipped: {error}. Install experiments/analysis/requirements.txt and rerun.')
    (output/'README.md').write_text('\n\n'.join(notes)+'\n')
    print('\n'.join(notes)); print(f'Outputs: {output}')


if __name__=='__main__': main()
