import json
from pathlib import Path
import sys
from experiments.scripts import run_phase_separation as stage


def test_stage00_is_three_gated_jobs_with_known_inputs(tmp_path,monkeypatch):
    manifest=tmp_path/'manifest.json';calls=[];real=stage.subprocess.check_output
    def check(cmd,**kwargs):
        if cmd[0]!='sbatch':return real(cmd,**kwargs)
        calls.append((cmd,kwargs['env']));return str(100+len(calls))+'\n'
    monkeypatch.setattr(stage.subprocess,'check_output',check)
    monkeypatch.setattr(sys,'argv',['stage','--manifest',str(manifest),'--submit'])
    stage.main();data=json.loads(manifest.read_text());runs=data['runs']
    assert len(calls)==len(runs)==3
    assert [r['environment']['PHASE_OFFLOAD_TRANSFORMER'] for r in runs]==['0','1','1']
    assert all(r['environment']['PHASE_SEPARATION']=='1' for r in runs)
    assert [r['environment']['SAMPLING_SEED'] for r in runs]==['1234','1234','100000']
    assert all(r['environment']['TARGET_TEXT_SOURCE']=='empty' and
               r['environment']['ICL_TEXT_SOURCE']=='cached' and
               r['environment']['VIDEO_STATE_FP32']=='1' and
               r['environment']['USE_GUIDANCE']=='0' for r in runs)
    assert [r['environment']['STAGE_REQUIRE_SUCCESS'] for r in runs]==['1','1','0']
    assert runs[1]['environment']['PHASE_COMPARE_ACTIONS'].endswith('place_empty_cup_101/phase_actions.jsonl')
    assert '--dependency=afterok:101' in calls[1][0]
    assert '--dependency=afterok:102' in calls[2][0]


def test_later_stages_require_explicit_offload_opt_in():
    from experiments.scripts.robotwin_suite import parser,specifications
    for enabled in (False,True):
        argv=['render','--manifest','/tmp/phase_manifest.json']+(['--phase-separation'] if enabled else [])
        a=parser().parse_args(argv);a.client_timeout=2400
        assert all(r['environment']['PHASE_SEPARATION']==str(int(enabled)) for r in specifications(a))


def test_job_parity_gate_rejects_different_actions(tmp_path,monkeypatch):
    import pytest
    source=(stage.ROOT/'experiments/jobs/eval.sbatch').read_text()
    block=source.rsplit("<<'PY'",1)[1].split('\nPY',1)[0]
    metrics=tmp_path/'stseed-100000/metrics/place_empty_cup/res.json'
    metrics.parent.mkdir(parents=True);metrics.write_text(json.dumps(dict(total_num=1,succ_num=1)))
    reference=tmp_path/'control.jsonl';reference.write_text(json.dumps(dict(chunk=0,sha256='abc'))+'\n')
    actions=tmp_path/'phase_actions.jsonl';actions.write_text(json.dumps(dict(chunk=0,sha256='different'))+'\n')
    monkeypatch.setenv('STAGE_REQUIRE_SUCCESS','1');monkeypatch.setenv('PHASE_COMPARE_ACTIONS',str(reference))
    monkeypatch.setattr(sys,'argv',['check',str(tmp_path),'place_empty_cup','1'])
    with pytest.raises(SystemExit,match='hashes differ'): exec(compile(block,'<job-check>','exec'),{})
    actions.write_text(reference.read_text())
    exec(compile(block,'<job-check>','exec'),{})
