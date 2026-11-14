"""Append ONE original-caption ICL job (default: documented FP32 cup test, seed 1234).

Dry-run unless --submit is passed. Existing runs and source metadata are preserved.
"""
import argparse
import copy
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import uuid

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from experiments.scripts.provenance import snapshot
from experiments.scripts.robotwin_suite import CONTROLLED


def atomic_write(path,data):
    temporary=path.with_name(path.name+'.tmp')
    temporary.write_text(json.dumps(data,indent=2)+'\n'); temporary.replace(path)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('manifest',type=Path)
    p.add_argument('--submit',action='store_true')
    p.add_argument('--phase-separation',action='store_true')
    p.add_argument('--client-timeout',type=int,help='Whole-client wall timeout in seconds; includes client setup')
    p.add_argument('--sampling-seed',type=int,default=1234)
    p.add_argument('--task',choices=['place_empty_cup','open_microwave'],default='place_empty_cup')
    p.add_argument('--partition',default='gpu_a100')
    p.add_argument('--account')
    a=p.parse_args(); path=a.manifest.resolve()
    if a.sampling_seed < 0: p.error('Sampling seed must be nonnegative')
    if a.client_timeout is not None and a.client_timeout <= 0: p.error('Client timeout must be positive')
    label=f'{a.task}_icl_original_caption_only_s{a.sampling_seed}'
    if not path.is_file(): p.error('Existing manifest not found')
    # An advisory lock serializes appenders; job submission is intentionally singular.
    with path.with_suffix('.append.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        data=json.loads(path.read_text())
        if data.get('stage')!='textcheck': p.error('Expected the existing textcheck manifest')
        if any(r.get('label')==label or (r.get('mode')=='icl_original_caption_only' and
               r.get('environment',{}).get('TASK',r.get('task'))==a.task and
               str(r.get('environment',{}).get('SAMPLING_SEED'))==str(a.sampling_seed)) for r in data['runs']):
            p.error('Original-caption entry for this sampling seed already exists; inspect its status/job ID before retrying')
        base=next((r for r in data['runs'] if r.get('mode')=='icl'),None)
        if base is None: p.error('Manifest has no pure-ICL control to copy')
        spec=copy.deepcopy(base)
        for key in ('job_id','command','error'): spec.pop(key,None)
        spec.update(task=a.task,seed=0,repeat=0,label=label,mode='icl_original_caption_only',status='planned')
        env=spec['environment']
        env.update(PHASE_SEPARATION=str(int(a.phase_separation)),PHASE_OFFLOAD_TRANSFORMER='1',
                   SAVE_ROLLOUT_SNAPSHOTS=str(int(a.phase_separation)),TASK=a.task,SEED='0',ICL_SEED='0',VIDEO_STATE_FP32='1',
                   RENDER_SAMPLES_PER_PIXEL='32',ICL_GUIDANCE_SCALE='5',
                   SAMPLING_SEED=str(a.sampling_seed),ZERO_WAM_ROOT=str(ROOT),TEST_NUM='1',USE_ICL='1',USE_TEXT='1',USE_GUIDANCE='0',
                   TARGET_TEXT_SOURCE='empty',ICL_TEXT_SOURCE='cached',TEXT_GUIDANCE_SCALE='1',
                   EXPERIMENT_LABEL=spec['label'],EXPERIMENT_OUTPUT_ROOT=str(path.parent/'runs'))
        if a.client_timeout is not None: env['CLIENT_TIMEOUT']=str(a.client_timeout)
        cmd=['sbatch','--parsable','--export=ALL',f'--partition={a.partition}','--time=01:00:00',
             f'--job-name=zw_{a.task}_original','--output='+str(path.parent/'slurm_%j.out'),
             f'--chdir={ROOT}']
        if a.account: cmd += [f'--account={a.account}']
        cmd += [str(ROOT/'experiments/jobs/eval.sbatch')]
        print(f'One {a.task} job: empty robot text + cached human caption; sampling seed {a.sampling_seed}; external guidance OFF.')
        print(' '.join(f'{k}={shlex.quote(v)}' for k,v in env.items() if k!='EXPERIMENT_SOURCE_DIGEST'))
        print(shlex.join(cmd))
        if not a.submit:
            print('DRY RUN: manifest unchanged; nothing submitted.'); return
        stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_'+uuid.uuid4().hex[:8]
        backup=path.with_name(f'manifest.before_original_{stamp}.json')
        backup.write_bytes(path.read_bytes())
        bundle=path.parent/f'original_caption_{stamp}.source.zip'
        digest=snapshot(bundle); env['EXPERIMENT_SOURCE_DIGEST']=digest
        spec.update(command=cmd,source_bundle=str(bundle),source_digest=digest,
                    appended_at_utc=stamp,revision=subprocess.check_output(
                        ['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip())
        spec['estimated_sbu'] = (20+5)/60*128
        spec['requested_walltime_sbu_ceiling'] = 128
        spec['dirty_diff'] = subprocess.check_output(['git','-C',str(ROOT),'diff'],text=True)
        previous_estimate = copy.deepcopy(data.get('estimate',{}))
        estimate = data.setdefault('estimate',{})
        estimate['episodes'] = int(estimate.get('episodes',0))+1
        estimate['sbu'] = float(estimate.get('sbu',0))+spec['estimated_sbu']
        estimate['requested_walltime_sbu_ceiling'] = float(estimate.get('requested_walltime_sbu_ceiling',0))+128
        data['runs'].append(spec)
        data.setdefault('amendments',[]).append(dict(kind='original_caption_only',task=a.task,sampling_seed=a.sampling_seed,backup=str(backup),previous_estimate=previous_estimate,
            note='Preserve previous arms; this adds empty target text plus cached human caption.'))
        atomic_write(path,data)  # Persist intent before submitting, so interruptions are visible.
        child_env={k:v for k,v in os.environ.items() if k not in CONTROLLED}
        child_env.update(env)
        try:
            spec['job_id']=subprocess.check_output(cmd,text=True,env=child_env).strip().split(';')[0]
            spec['status']='submitted'
        except (OSError,subprocess.CalledProcessError) as error:
            spec['status']='submission_failed'; spec['error']=str(error); atomic_write(path,data); raise
        atomic_write(path,data)
        print(f'Appended job {spec["job_id"]}. Logs/results remain under {path.parent}')
        print('Estimate updated with a 20-minute episode + 5-minute startup allowance; added ceiling: 128 SBUs.')


if __name__=='__main__': main()
