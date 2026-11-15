"""Stage 00: compact masked observation padding without shortening real history.

Dry-run unless --submit. Uses one GPU/job, unchanged policy context/precision.
Cache controls use afterok; --diagnostics uses afterany dependencies. No stage 1-4 jobs are submitted automatically.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from experiments.scripts.provenance import snapshot
from experiments.scripts.robotwin_suite import CONTROLLED, parser, specifications


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',type=Path)
    group=p.add_mutually_exclusive_group()
    group.add_argument('--diagnostics',action='store_true',help='One microwave, scale and bread-basket episode; no cup reruns')
    p.add_argument('--submit',action='store_true'); p.add_argument('--account')
    group.add_argument('--cup-only',action='store_true',help='Only the two cup validation jobs')
    a=p.parse_args()
    stage='task_diagnostics' if a.diagnostics else 'cache_compaction'
    manifest=(a.manifest or ROOT/f'experiments/stages/00_{stage}/manifest.json').resolve()
    if manifest.exists() or manifest.with_suffix('.source.zip').exists(): p.error('Choose a new manifest directory')
    defaults=parser().parse_args(['baselines','--manifest',str(manifest)])
    defaults.client_timeout=3000
    template=next(s for s in specifications(defaults) if s['mode']=='icl')
    import copy
    runs=[]
    cases=[('place_empty_cup',1234,False),('place_empty_cup',1234,True)]
    if not a.cup_only: cases.append(('open_microwave',1234,True))
    if a.diagnostics:
        cases=[(task,1234,True) for task in ('open_microwave','place_object_scale','place_bread_basket')]
    for task,seed,compact in cases:
        spec=copy.deepcopy(template); label=f'{task}_cache_{"compact" if compact else "control"}'
        if a.diagnostics: label=f'{task}_diagnostic'
        spec.update(task=task,label=label,mode=stage,seed=0,repeat=0,status='planned')
        spec['environment'].update(TEST_NUM='1',TASK=task,SEED='0',SAMPLING_SEED=str(seed),ICL_SEED='0',
            EXPERIMENT_LABEL=label,EXPERIMENT_STAGE=stage,
            EXPERIMENT_OUTPUT_ROOT=str(manifest.parent/'runs'),PHASE_SEPARATION='1',
            PHASE_OFFLOAD_TRANSFORMER='1',COMPACT_OBSERVATION_CACHE=str(int(compact)),SAVE_ROLLOUT_SNAPSHOTS='1',STAGE_REQUIRE_SUCCESS=str(int(task=='place_empty_cup')),
            PROFILE_TIMING='1',PROFILE_SYNC_GPU='0')
        spec['environment']['TASK_DIAGNOSTICS']='microwave' if a.diagnostics and task=='open_microwave' else ''
        runs.append(spec)
    data=dict(schema_version=1,stage=stage,submitted=a.submit,runs=runs,
              estimate=dict(episodes=len(runs),sbu_per_hour=128,
                            sbu=((3*5+25+2*10) if a.diagnostics else (len(runs)*5+16+(25 if not a.cup_only else 0)))/60*128,
                            requested_walltime_sbu_ceiling=len(runs)*128),
              note=('Task diagnostics use afterany so an OOM does not block the other tasks.' if a.diagnostics else 'Both cup arms use observation offload; only padding compaction differs. '
                   'Microwave task failure is acceptable for memory validation, infrastructure failure is not.'))
    manifest.parent.mkdir(parents=True,exist_ok=True)
    data['source_digest']=snapshot(manifest.with_suffix('.source.zip'))
    data['source_bundle']=str(manifest.with_suffix('.source.zip'))
    data['revision']=subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip()
    data['dirty_diff']=subprocess.check_output(['git','-C',str(ROOT),'diff'],text=True)
    def save(): manifest.write_text(json.dumps(data,indent=2)+'\n')
    save(); previous=None
    for spec in runs:
        env=spec['environment']; env['EXPERIMENT_SOURCE_DIGEST']=data['source_digest']
        cmd=['sbatch','--parsable','--export=ALL','--partition=gpu_a100','--time=01:00:00',
             '--job-name='+spec['label'],'--output='+str(manifest.parent/'slurm_%j.out'),f'--chdir={ROOT}']
        if a.account: cmd += ['--account='+a.account]
        if previous: cmd += ['--dependency='+('afterany:' if a.diagnostics else 'afterok:')+previous]
        cmd += [str(ROOT/'experiments/jobs/eval.sbatch')]
        spec['command']=cmd
        print(spec['label'],f'sampling_seed={env["SAMPLING_SEED"]}',cmd)
        if a.submit:
            child={k:v for k,v in os.environ.items() if k not in CONTROLLED};child.update(env)
            try:
                spec['job_id']=subprocess.check_output(cmd,text=True,env=child).strip().split(';')[0]
                spec['status']='submitted'; previous=spec['job_id']
            except (OSError,subprocess.CalledProcessError) as error:
                spec.update(status='submission_failed',error=str(error));save();raise
        save()
    print(f'{len(runs)} one-episode jobs; estimate {data["estimate"]["sbu"]:.1f} SBUs, '
          f'wall-limit ceiling {len(runs)*128} SBUs. '+('SUBMITTED' if a.submit else 'DRY RUN'))
    print(manifest)


if __name__=='__main__': main()
