"""Stage 00: cup parity control, cup offload, then a long microwave memory test.

Dry-run unless --submit. Uses one GPU/job, afterok dependencies, unchanged policy
context/precision. No stage 1-4 jobs are submitted automatically.
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
    p.add_argument('--manifest',type=Path,default=ROOT/'experiments/stages/00_test_phase_separation/manifest.json')
    p.add_argument('--submit',action='store_true'); p.add_argument('--account')
    p.add_argument('--cup-only',action='store_true',help='Only the two cup parity jobs')
    a=p.parse_args(); manifest=a.manifest.resolve()
    if manifest.exists() or manifest.with_suffix('.source.zip').exists(): p.error('Choose a new manifest directory')
    defaults=parser().parse_args(['textcheck','--manifest',str(manifest)])
    defaults.client_timeout=3000
    template=next(s for s in specifications(defaults) if s['mode']=='icl_legacy_text')
    import copy
    runs=[]
    cases=[('place_empty_cup',1234,False),('place_empty_cup',1234,True)]
    if not a.cup_only: cases.append(('open_microwave',100000,True))
    for task,seed,offload in cases:
        spec=copy.deepcopy(template); label=f'{task}_phase_{"offload" if offload else "control"}'
        spec.update(task=task,label=label,mode='phase_separation',seed=0,repeat=0,status='planned')
        spec['environment'].update(TASK=task,SEED='0',SAMPLING_SEED=str(seed),ICL_SEED='0',
            EXPERIMENT_LABEL=label,EXPERIMENT_STAGE='phase_separation',
            EXPERIMENT_OUTPUT_ROOT=str(manifest.parent/'runs'),PHASE_SEPARATION='1',
            PHASE_OFFLOAD_TRANSFORMER=str(int(offload)),STAGE_REQUIRE_SUCCESS=str(int(task=='place_empty_cup')),
            PROFILE_TIMING='1',PROFILE_SYNC_GPU='0')
        runs.append(spec)
    data=dict(schema_version=1,stage='phase_separation',submitted=a.submit,runs=runs,
              estimate=dict(episodes=len(runs),sbu_per_hour=128,
                            sbu=(len(runs)*5+16+(25 if not a.cup_only else 0))/60*128,
                            requested_walltime_sbu_ceiling=len(runs)*128),
              note='Single-GPU unsharded path in both cup arms; only transformer transfers differ. '
                   'Microwave task failure is acceptable for memory validation, infrastructure failure is not.')
    manifest.parent.mkdir(parents=True,exist_ok=True)
    data['source_digest']=snapshot(manifest.with_suffix('.source.zip'))
    data['source_bundle']=str(manifest.with_suffix('.source.zip'))
    data['revision']=subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip()
    data['dirty_diff']=subprocess.check_output(['git','-C',str(ROOT),'diff'],text=True)
    def save(): manifest.write_text(json.dumps(data,indent=2)+'\n')
    save(); previous=None; cup_control=None
    for spec in runs:
        env=spec['environment']; env['EXPERIMENT_SOURCE_DIGEST']=data['source_digest']
        if spec['label']=='place_empty_cup_phase_offload' and cup_control:
            env['PHASE_COMPARE_ACTIONS']=str(manifest.parent/'runs'/f'place_empty_cup_{cup_control}'/'phase_actions.jsonl')
        cmd=['sbatch','--parsable','--export=ALL','--partition=gpu_a100','--time=01:00:00',
             '--job-name='+spec['label'],'--output='+str(manifest.parent/'slurm_%j.out'),f'--chdir={ROOT}']
        if a.account: cmd += ['--account='+a.account]
        if previous: cmd += ['--dependency=afterok:'+previous]
        cmd += [str(ROOT/'experiments/jobs/eval.sbatch')]
        spec['command']=cmd
        print(spec['label'],f'sampling_seed={env["SAMPLING_SEED"]}',cmd)
        if a.submit:
            child={k:v for k,v in os.environ.items() if k not in CONTROLLED};child.update(env)
            try:
                spec['job_id']=subprocess.check_output(cmd,text=True,env=child).strip().split(';')[0]
                spec['status']='submitted'; previous=spec['job_id']
                if spec['label']=='place_empty_cup_phase_control': cup_control=spec['job_id']
            except (OSError,subprocess.CalledProcessError) as error:
                spec.update(status='submission_failed',error=str(error));save();raise
        save()
    print(f'{len(runs)} one-episode jobs; estimate {data["estimate"]["sbu"]:.1f} SBUs, '
          f'wall-limit ceiling {len(runs)*128} SBUs. '+('SUBMITTED' if a.submit else 'DRY RUN'))
    print(manifest)


if __name__=='__main__': main()
