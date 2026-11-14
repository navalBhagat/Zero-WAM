"""One scale-placement episode with the current offload setup; no dependencies.

Dry-run by default. Saves crash-surviving camera snapshots after each action chunk.
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
from experiments.scripts.robotwin_suite import CONTROLLED,parser,specifications


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',type=Path,default=ROOT/'experiments/stages/00_test_place_object_scale/manifest.json')
    p.add_argument('--submit',action='store_true');p.add_argument('--account')
    p.add_argument('--sampling-seed',type=int,default=100000)
    a=p.parse_args();path=a.manifest.resolve()
    if a.sampling_seed<0:p.error('Sampling seed must be nonnegative')
    if path.exists() or path.with_suffix('.source.zip').exists():p.error('Choose a fresh --manifest directory')
    args=parser().parse_args(['textcheck','--tasks','place_object_scale','--seeds','0','--manifest',str(path)])
    args.client_timeout=3000
    run=next(r for r in specifications(args) if r['mode']=='icl_legacy_text')
    run.update(label='place_object_scale_phase_smoke',mode='icl_original_caption_only')
    env=run['environment'];env.update(SAMPLING_SEED=str(a.sampling_seed),
        EXPERIMENT_LABEL=run['label'],EXPERIMENT_STAGE='scale_smoke',PHASE_SEPARATION='1',
        PHASE_OFFLOAD_TRANSFORMER='1',STAGE_REQUIRE_SUCCESS='0',SAVE_ROLLOUT_SNAPSHOTS='1')
    data=dict(schema_version=1,stage='scale_smoke',submitted=a.submit,runs=[run],
        estimate=dict(episodes=1,sbu_per_hour=128,sbu=25/60*128,requested_walltime_sbu_ceiling=128),
        note='Current observation-encoding offload; task failure may complete normally. '
             'Long-history OOM remains possible. No guidance or history-window change.')
    path.parent.mkdir(parents=True,exist_ok=True)
    data['source_bundle']=str(path.with_suffix('.source.zip'))
    data['source_digest']=snapshot(path.with_suffix('.source.zip'))
    data['revision']=subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip()
    data['dirty_diff']=subprocess.check_output(['git','-C',str(ROOT),'diff'],text=True)
    env['EXPERIMENT_SOURCE_DIGEST']=data['source_digest']
    cmd=['sbatch','--parsable','--export=ALL','--partition=gpu_a100','--time=01:00:00',
         '--job-name=zw_scale_smoke','--output='+str(path.parent/'slurm_%j.out'),f'--chdir={ROOT}']
    if a.account:cmd+=['--account='+a.account]
    cmd+=[str(ROOT/'experiments/jobs/eval.sbatch')];run['command']=cmd
    def save():path.write_text(json.dumps(data,indent=2)+'\n')
    save();print('ONE job; 1 A100 + 18 CPU cores; one-hour maximum; no dependencies.');print(cmd)
    if a.submit:
        child={k:v for k,v in os.environ.items() if k not in CONTROLLED};child.update(env)
        try:
            run['job_id']=subprocess.check_output(cmd,text=True,env=child).strip().split(';')[0]
            run['status']='submitted'
        except (OSError,subprocess.CalledProcessError) as error:
            run.update(status='submission_failed',error=str(error));save();raise
        save();print('Submitted',run['job_id'])
    else:print('DRY RUN: nothing submitted.')
    print('Manifest:',path)
    print('Camera PNGs: runs/place_object_scale_JOBID/snapshots/episode_000/')


if __name__=='__main__':main()
