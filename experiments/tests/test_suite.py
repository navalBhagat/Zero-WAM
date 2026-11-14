"""Campaign design, interruption accounting, and matched-analysis checks."""
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from experiments.scripts.robotwin_suite import parser, specifications
from experiments.analysis.analyze import breakdown, collect, pairs


def args(stage):
    a=parser().parse_args([stage,'--manifest','/tmp/manifest.json'])
    a.client_timeout=2400
    return a


def test_render_default_is_nine_paired_episodes():
    specs=specifications(args('render'))
    assert len(specs)==9
    assert {s['task'] for s in specs}=={'open_microwave'}
    for repeat in range(3):
        group=[s for s in specs if s['repeat']==repeat]
        assert {s['environment']['RENDER_SAMPLES_PER_PIXEL'] for s in group}=={'8','16','32'}
        assert len({s['environment']['SAMPLING_SEED'] for s in group})==1
        assert all(s['environment']['TEST_NUM']=='1' for s in group)
    assert specs[0]['environment']['RENDER_SAMPLES_PER_PIXEL']=='32'
    assert specs[3]['environment']['RENDER_SAMPLES_PER_PIXEL']=='16'


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
        folder=tmp_path/f'open_microwave_{i+1}'; folder.mkdir()
        if i==2: continue
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


def test_submission_is_serial_and_scrubs_inherited_controls(tmp_path, monkeypatch):
    import experiments.scripts.robotwin_suite as suite
    real=suite.subprocess.check_output; calls=[]
    def check(cmd, **kwargs):
        if cmd[0]!='sbatch': return real(cmd,**kwargs)
        calls.append((cmd,kwargs['env']))
        return str(100+len(calls))+'\n'
    monkeypatch.setattr(suite.subprocess,'check_output',check)
    monkeypatch.setenv('GUIDANCE_CAPTURE','/must/not/leak.pt')
    manifest=tmp_path/'campaign/manifest.json'
    monkeypatch.setattr(sys,'argv',['suite','render','--manifest',str(manifest),'--submit'])
    suite.main()
    data=json.loads(manifest.read_text())
    assert len(calls)==9 and all(s['status']=='submitted' for s in data['runs'])
    assert not any(c.startswith('--dependency') for c in calls[0][0])
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


def test_legacy_text_pair_matches_failed_render_settings():
    specs=specifications(args('textcheck'))
    assert len(specs)==2
    off,on=[s['environment'] for s in specs]
    assert off['USE_TEXT']=='0' and off['ICL_TEXT_SOURCE']=='target'
    assert on['USE_TEXT']=='1' and on['ICL_TEXT_SOURCE']=='cached'
    assert on['TARGET_TEXT_SOURCE']=='empty'
    for env in (off,on):
        assert env['TEST_NUM']=='1' and env['TASK']=='open_microwave'
        assert env['SEED']=='0' and env['SAMPLING_SEED']=='10000' and env['ICL_SEED']=='0'
        assert env['RENDER_SAMPLES_PER_PIXEL']=='32' and env['USE_GUIDANCE']=='0'
        assert env['ICL_GUIDANCE_SCALE']=='5' and env['TEXT_GUIDANCE_SCALE']=='1'
        assert env['VIDEO_STATE_FP32']=='1'


def test_append_preserves_old_jobs_and_uses_same_folder(tmp_path,monkeypatch):
    from experiments.scripts import append_original_icl as script
    import copy
    specs=specifications(args('textcheck'))
    previous=copy.deepcopy(specs[0])
    previous.update(label='icl_original_caption_only',mode='icl_original_caption_only',job_id='99999')
    previous['environment']['SAMPLING_SEED']='1234'
    specs.append(previous)
    old=copy.deepcopy(specs)
    manifest=tmp_path/'manifest.json'
    manifest.write_text(json.dumps(dict(stage='textcheck',runs=specs,
        source_digest='historical',estimate=dict(episodes=3,sbu=117.33,requested_walltime_sbu_ceiling=384))))
    real=script.subprocess.check_output; calls=[]
    def check(cmd,**kwargs):
        if cmd[0]!='sbatch': return real(cmd,**kwargs)
        calls.append((cmd,kwargs['env'])); return '12345\n'
    monkeypatch.setattr(script.subprocess,'check_output',check)
    monkeypatch.setattr(sys,'argv',['append',str(manifest),'--submit','--phase-separation','--client-timeout','1200'])
    script.main()
    result=json.loads(manifest.read_text())
    assert result['runs'][:3]==old and result['source_digest']=='historical'
    assert len(calls)==1 and len(result['runs'])==4
    added=result['runs'][-1]
    assert added['job_id']=='12345' and added['environment']['TARGET_TEXT_SOURCE']=='empty'
    assert added['environment']['EXPERIMENT_OUTPUT_ROOT']==str(tmp_path/'runs')
    assert f'--output={tmp_path}/slurm_%j.out' in calls[0][0]
    assert result['estimate']['episodes']==4
    assert added['environment']['PHASE_SEPARATION']=='1'
    assert added['environment']['CLIENT_TIMEOUT']=='1200'
    assert added['environment']['SAVE_ROLLOUT_SNAPSHOTS']=='1'
    assert added['environment']['SAMPLING_SEED']=='1234'
    assert added['task']=='place_empty_cup' and added['environment']['TASK']=='place_empty_cup'
    assert added['environment']['RENDER_SAMPLES_PER_PIXEL']=='32'
    assert added['environment']['ICL_GUIDANCE_SCALE']=='5'
    assert added['environment']['USE_GUIDANCE']=='0'
    assert added['environment']['SEED']=='0' and added['environment']['ICL_SEED']=='0'
    assert added['environment']['VIDEO_STATE_FP32']=='1'
    assert added['label']=='place_empty_cup_icl_original_caption_only_s1234'
    import pytest
    with pytest.raises(SystemExit): script.main()
    assert len(calls)==1
    assert list(tmp_path.glob('manifest.before_original_*.json'))
