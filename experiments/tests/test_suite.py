"""Campaign design, interruption accounting, and matched-analysis checks."""
import importlib.util
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from experiments.scripts.robotwin_suite import parser, specifications
from experiments.analysis.analyze import breakdown, collect, pairs, wilson, plots
from experiments.scripts.job_timing import record


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


def test_stage_counts_and_precision_factor():
    assert len(specifications(args('baselines')))==12
    assert len(specifications(args('guidance')))==24
    specs=specifications(args('precision'))
    assert len(specs)==24
    assert {s['environment']['GUIDANCE_DECODER_DTYPE'] for s in specs}=={'float16','float32'}
    assert all(s['environment']['VIDEO_STATE_FP32']=='1' for s in specs)


def test_failed_job_has_billable_time_without_completion():
    data=record([124,100,110,130,135,0,180])
    assert data['allocated_seconds']==80 and data['client_seconds'] is None
    assert not data['client_completed'] and data['server_startup_seconds']==20


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


def test_wilson_small_sample_is_not_certain():
    lo,hi=wilson(1,1)
    assert 0<lo<.5 and hi<=1


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


def test_plotting_smoke(tmp_path):
    profile=dict(wall_seconds=10,timings={'server_sampling_roundtrip':dict(calls=2,exclusive_seconds=6)})
    e=dict(run=0,task='open_microwave',label='spp32',spp=32,success=1,wall_seconds=10,
           seconds_per_action=1,sampling_seconds_per_chunk=3,**breakdown(profile))
    plots(tmp_path,[dict(allocated_seconds=20,pre_server_seconds=1,startup_seconds=3,client_seconds=15)],
          [e],[dict(run=0,call=0,total_seconds=3)],[])
    assert (tmp_path/'timing.png').stat().st_size>1000
    assert (tmp_path/'success.png').is_file()


def test_sacct_uses_gpu_count_once_and_keeps_missing_accounting():
    from experiments.analysis.analyze import accounting_from_sacct
    rows=accounting_from_sacct('JobIDRaw|State|ElapsedRaw|AllocTRES|ExitCode\n'
        '1|COMPLETED|600|cpu=18,gres/gpu=1,gres/gpu:a100=1|0:0\n'
        '2|FAILED|60|cpu=18|1:0\n',128)
    assert rows[0]['estimated_sbu']==128/6
    assert rows[1]['estimated_sbu'] is None


def test_precision_plot_keeps_failure_rows(tmp_path,monkeypatch):
    from experiments.analysis.plot_precision import main
    source=tmp_path/'precision.json'
    source.write_text(json.dumps(dict(measurements=[dict(decoder_dtype='float32',strength=.01,
        repeat=0,status='ok',reward_gain=.001,applied_correction_norm=.01,changed_fraction=1,seconds=2),
        dict(decoder_dtype='float16',strength=.01,repeat=0,status='failed',error='overflow')])) )
    monkeypatch.setattr(sys,'argv',['plot',str(source)])
    main()
    assert (tmp_path/'precision.png').is_file() and (tmp_path/'precision.pdf').is_file()
    assert len(json.loads((tmp_path/'failures.json').read_text()))==1


def test_matching_contact_sheet_and_trace(tmp_path,monkeypatch):
    from PIL import Image
    from experiments.analysis.plot_matching import main
    folder=tmp_path/'references/episode_000'; folder.mkdir(parents=True)
    for name in ('frame_0001.png','robot_chunk_0000.png'):
        Image.new('RGB',(20,20),'red').save(folder/name)
    (folder/'selection.jsonl').write_text(json.dumps(dict(chunk=0,frame_indices=[1],matched_frame=0,
        matched_similarity=.5,selection_seconds=.02,robot_preview_path='/old/location/robot_chunk_0000.png'))+'\n')
    monkeypatch.setattr(sys,'argv',['plot',str(tmp_path)])
    main()
    assert (tmp_path/'matching_analysis/episode_000_contact_sheet.jpg').is_file()
    assert (tmp_path/'matching_analysis/episode_000_matching.png').is_file()


def test_legacy_text_pair_matches_failed_render_settings():
    specs=specifications(args('textcheck'))
    assert len(specs)==2
    off,on=[s['environment'] for s in specs]
    assert off['USE_TEXT']=='0' and off['ICL_TEXT_SOURCE']=='target'
    assert on['USE_TEXT']=='1' and on['ICL_TEXT_SOURCE']=='cached'
    for env in (off,on):
        assert env['TEST_NUM']=='1' and env['TASK']=='open_microwave'
        assert env['SEED']=='0' and env['SAMPLING_SEED']=='10000' and env['ICL_SEED']=='0'
        assert env['RENDER_SAMPLES_PER_PIXEL']=='32' and env['USE_GUIDANCE']=='0'
        assert env['ICL_GUIDANCE_SCALE']=='5' and env['TEXT_GUIDANCE_SCALE']=='1'
        assert env['VIDEO_STATE_FP32']=='1'


def test_original_comparison_uses_empty_robot_text():
    off,on=specifications(args('textcheck'))
    assert off['environment']['USE_TEXT']=='0'
    assert on['environment']['TARGET_TEXT_SOURCE']=='empty'
    assert on['environment']['ICL_TEXT_SOURCE']=='cached'


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
    monkeypatch.setattr(sys,'argv',['append',str(manifest),'--submit'])
    script.main()
    result=json.loads(manifest.read_text())
    assert result['runs'][:3]==old and result['source_digest']=='historical'
    assert len(calls)==1 and len(result['runs'])==4
    added=result['runs'][-1]
    assert added['job_id']=='12345' and added['environment']['TARGET_TEXT_SOURCE']=='empty'
    assert added['environment']['EXPERIMENT_OUTPUT_ROOT']==str(tmp_path/'runs')
    assert f'--output={tmp_path}/slurm_%j.out' in calls[0][0]
    assert result['estimate']['episodes']==4
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
