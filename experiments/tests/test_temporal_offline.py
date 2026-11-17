import importlib.util
from pathlib import Path

import pytest
import torch

ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('offline_temporal',ROOT/'experiments/scripts/03_temporal_guidance_test.py')
offline=importlib.util.module_from_spec(spec);spec.loader.exec_module(offline)


def test_offline_comparison_uses_identical_alignment_without_cuda():
    features=torch.tensor([[1.,0.]]*6+[[.8,.6],[.6,.8],[0.,1.]])
    queries=torch.tensor([[1.,0.],[.8,.6],[0.,1.]])
    rows,selector=offline.replay(features,queries,[0,2,4],lookahead=4,
                                 distance=.15,smoothing=1,max_frames=8)
    assert rows[0]['matched_frame']==0
    assert rows[0]['frame_target']==4 and rows[0]['distance_target']==6
    assert rows[-1]['matched_frame']==8 and rows[-1]['distance_target']==8
    assert selector.position==8
    assert all(row['distance_target']>=row['matched_frame'] for row in rows)


def test_offline_single_observation_figures(tmp_path):
    pytest.importorskip('matplotlib')
    from PIL import Image
    features=torch.eye(3)
    rows,selector=offline.replay(features,features[:1],[0],distance=.1,smoothing=1)
    offline.figures(tmp_path,rows,selector,torch.zeros(3,3,16,16),[Image.new('RGB',(16,16))])
    assert (tmp_path/'targets.png').is_file() and (tmp_path/'comparison.png').is_file()


def test_replay_rejects_missing_observations():
    with pytest.raises(ValueError,match='One chunk'):
        offline.replay(torch.eye(3),torch.eye(3),[0])
