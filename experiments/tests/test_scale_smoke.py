import json
import sys
from experiments.scripts import run_scale_smoke as script


def test_one_scale_job_with_snapshots_and_no_dependency(tmp_path,monkeypatch):
    real=script.subprocess.check_output;calls=[]
    def check(cmd,**kwargs):
        if cmd[0]!='sbatch':return real(cmd,**kwargs)
        calls.append((cmd,kwargs['env']));return '12345\n'
    monkeypatch.setattr(script.subprocess,'check_output',check)
    manifest=tmp_path/'manifest.json'
    monkeypatch.setattr(sys,'argv',['scale','--manifest',str(manifest),'--submit'])
    script.main();data=json.loads(manifest.read_text());run=data['runs'][0];env=run['environment']
    assert len(calls)==len(data['runs'])==1 and run['task']=='place_object_scale'
    assert env['TEST_NUM']=='1' and env['SAMPLING_SEED']=='1234'
    assert env['TARGET_TEXT_SOURCE']=='empty' and env['ICL_TEXT_SOURCE']=='cached'
    assert env['PHASE_SEPARATION']==env['SAVE_ROLLOUT_SNAPSHOTS']==env['VIDEO_STATE_FP32']=='1'
    assert env['USE_GUIDANCE']=='0'
    assert not any(c.startswith('--dependency') for c in calls[0][0])
