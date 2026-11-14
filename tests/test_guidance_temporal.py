import json
from types import SimpleNamespace
import numpy as np
import pytest
import torch
# Load this standalone runtime module without importing the model dependencies.
import importlib.util
from pathlib import Path
spec = importlib.util.spec_from_file_location(
    'guidance_temporal', Path(__file__).resolve().parents[1] / 'wan_va/guidance/temporal.py')
temporal = importlib.util.module_from_spec(spec)
spec.loader.exec_module(temporal)
WindowFrameSelector = temporal.WindowFrameSelector
select_reference = temporal.select_reference


def test_latest_actual_observation_wins_over_stale_request(tmp_path,monkeypatch):
    monkeypatch.setenv('GUIDANCE_SELECTION','window')
    selected=[]
    reward=SimpleNamespace(reference_features=torch.eye(3),
        features=lambda image: image.mean((2,3)),
        select_reference=lambda indices: selected.append(indices),
        save_reference_frames=lambda output: [])
    stale={'external':np.full((4,4,3),[255,0,0],dtype=np.uint8)}
    latest={'external':np.full((4,4,3),[0,255,0],dtype=np.uint8)}
    server=SimpleNamespace(guidance_selector=WindowFrameSelector(3),external_reward=reward,
        chunk_idx=1,device='cpu',job_config=SimpleNamespace(obs_cam_keys=['external']),
        reference_preview_dir=str(tmp_path),latest_guidance_observation=[latest])
    result=select_reference(server,dict(obs=[stale]))
    assert selected==[[1]] and result['matched_frame']==1
    assert (tmp_path/'robot_chunk_0001.png').is_file()
    assert json.loads((tmp_path/'selection.jsonl').read_text())['frame_indices']==[1]


def test_nonfinite_matching_scores_rejected():
    with pytest.raises(ValueError,match='finite'):
        WindowFrameSelector(2).select(0,similarities=[0,float('nan')])


def test_window_pauses_and_does_not_jump_or_regress():
    selector=WindowFrameSelector(8,window=2,lookahead=1)
    assert selector.select(0,similarities=[0,1,3,0,0,0,0,99])==[3]
    assert selector.select(1,similarities=[99,0,4,1,0,0,0,99])==[3]
    assert selector.position==2
