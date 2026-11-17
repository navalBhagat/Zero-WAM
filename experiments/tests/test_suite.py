"""Campaign design, interruption accounting, and matched-analysis checks."""
import json
from pathlib import Path
import sys
import pytest

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from experiments.scripts.robotwin_suite import parser, specifications
from experiments.analysis.analyze import breakdown, collect, pairs


def args(stage):
    a=parser().parse_args([stage,'--manifest','/tmp/manifest.json'])
    a.client_timeout=2400
    return a


def test_render_default_is_three_paired_episodes():
    specs=specifications(args('render'))
    assert len(specs)==3
    assert {s['task'] for s in specs}=={'place_object_scale'}
    assert [s['environment']['RENDER_SAMPLES_PER_PIXEL'] for s in specs]==['32','16','8']
    assert {s['environment']['SAMPLING_SEED'] for s in specs}=={'1234'}
    assert all(s['environment']['TEST_NUM']=='1' and s['repeat']==0 for s in specs)
    # Explicit repetitions still rotate order while preserving each paired set.
    a=args('render');a.repeats=3
    repeated=specifications(a)
    assert len(repeated)==9
    assert [s['environment']['RENDER_SAMPLES_PER_PIXEL'] for s in repeated[3:6]]==['16','8','32']


def test_exclusive_breakdown_not_inclusive_double_count():
    data=breakdown(dict(wall_seconds=10,timings={
        'action_execution':dict(inclusive_seconds=8,exclusive_seconds=3),
        'camera_capture':dict(inclusive_seconds=5,exclusive_seconds=5)}))
    assert data['camera_render']==5 and data['action_other']==3 and data['unprofiled']==2
    assert sum(data.values())==10


def test_missing_runs_and_real_seed_pairing(tmp_path):
    specs=specifications(args('render'))[:3]
    manifest=dict(stage='render',runs=specs,estimate=dict(sbu_per_hour=128))
    for i,s in enumerate(specs):
        s['job_id']=str(i+1)
        folder=tmp_path/f'place_object_scale_{i+1}'; folder.mkdir()
        if i==2: continue
        (folder/'gpu.csv').write_text('timestamp, memory.used [MiB], utilization.gpu [%]\nnow, 34295 MiB, 60 %\n')
        (folder/'config.json').write_text(json.dumps(dict(episodes=[dict(
            episode_id=0,status='completed',simulator_seed=123+i,icl_seed=0,
            success=True,policy_action_calls=16)])))
        (folder/'job_timing.json').write_text(json.dumps(dict(exit_code=0,allocated_seconds=600)))
    runs,episodes,_,_=collect(manifest,tmp_path)
    assert sum(r['missing'] for r in runs)==1
    assert len(episodes)==2 and pairs(episodes,'render')==[]
    episodes[1]['simulator_seed']=123
    assert len(pairs(episodes,'render'))==1
    assert runs[0]['estimated_sbu']==128/6
    assert runs[0]['gpu_peak_used_mib']==34295 and runs[0]['gpu_mean_utilization']==60


@pytest.mark.parametrize('parallel', [False, True])
def test_submission_dependencies_and_inherited_controls(tmp_path, monkeypatch, parallel):
    import experiments.scripts.robotwin_suite as suite
    real=suite.subprocess.check_output; calls=[]
    def check(cmd, **kwargs):
        if cmd[0]!='sbatch': return real(cmd,**kwargs)
        calls.append((cmd,kwargs['env']))
        return str(100+len(calls))+'\n'
    monkeypatch.setattr(suite.subprocess,'check_output',check)
    monkeypatch.setenv('GUIDANCE_CAPTURE','/must/not/leak.pt')
    manifest=tmp_path/'campaign/manifest.json'
    monkeypatch.setattr(sys,'argv',['suite','render','--manifest',str(manifest),'--submit'] + (['--parallel'] if parallel else []))
    suite.main()
    data=json.loads(manifest.read_text())
    assert len(calls)==3 and all(s['status']=='submitted' for s in data['runs'])
    assert not any(c.startswith('--dependency') for c in calls[0][0])
    if parallel:
        assert all(not any(c.startswith('--dependency') for c in cmd) for cmd, _ in calls)
    else:
        assert '--dependency=afterany:101' in calls[1][0]
    assert all('GUIDANCE_CAPTURE' not in env for _,env in calls)
    assert Path(data['source_bundle']).is_file()


def test_sacct_uses_gpu_count_once_and_keeps_missing_accounting():
    from experiments.analysis.analyze import accounting_from_sacct
    rows=accounting_from_sacct('JobIDRaw|State|ElapsedRaw|AllocTRES|ExitCode\n'
        '1|COMPLETED|600|cpu=18,gres/gpu=1,gres/gpu:a100=1|0:0\n'
        '2|FAILED|60|cpu=18|1:0\n',128)
    assert rows[0]['estimated_sbu']==128/6
    assert rows[1]['estimated_sbu'] is None


def test_consecutive_jobs_use_disjoint_server_and_rendezvous_ports():
    import subprocess
    script=(ROOT/'experiments/jobs/eval.sbatch').read_text()
    assignments='\n'.join(line for line in script.splitlines()
                          if line.startswith(('export PORT=', 'export MASTER_PORT=')))
    used=set()
    for job in range(29286680,29286690):
        result=subprocess.check_output(['bash','-c',f'SLURM_JOB_ID={job}\n'+assignments+
                                        '\nprintf "%s %s" "$PORT" "$MASTER_PORT"'],text=True)
        pair={int(v) for v in result.split()}
        assert len(pair)==2 and not pair & used
        used.update(pair)


def test_temporal_only_pilot_is_two_guidance_only_jobs():
    a=parser().parse_args(['guidance','--tasks','place_object_scale','place_bread_basket',
                          '--seeds','0','--episodes','1','--guidance-modes','guidance',
                          '--guidance-selectors','window',
                          '--manifest','/tmp/guidance.json'])
    a.client_timeout=3000
    runs=specifications(a)
    assert len(runs)==2
    assert {r['task'] for r in runs}=={'place_object_scale','place_bread_basket'}
    assert all(r['environment']['GUIDANCE_SELECTION']=='window' and
               r['environment']['USE_TEXT']=='0' and r['environment']['USE_ICL']=='0' and
               r['environment']['USE_GUIDANCE']=='1' for r in runs)
