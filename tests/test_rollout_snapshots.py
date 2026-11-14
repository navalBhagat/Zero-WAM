import importlib.util
import json
from pathlib import Path
import numpy as np
from PIL import Image

spec=importlib.util.spec_from_file_location('snapshots',Path(__file__).resolve().parents[1]/'evaluation/robotwin/rollout_snapshots.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


def test_snapshots_are_readable_before_episode_completion(tmp_path):
    writer=module.RolloutSnapshots(tmp_path,True)
    obs={'observation.images.'+cam:np.zeros((4,6,3),dtype=np.uint8)
         for cam in ('cam_high','cam_left_wrist','cam_right_wrist')}
    writer.write(obs,0,16,False)
    assert len(list(tmp_path.glob('*.png')))==3
    row=json.loads((tmp_path/'frames.jsonl').read_text())
    assert row['chunk']==0 and row['simulator_action_count']==16
    for name in row['images'].values():
        with Image.open(tmp_path/name) as image:assert image.size==(6,4)
    disabled=module.RolloutSnapshots(tmp_path/'disabled',False);disabled.write({},0,0,False)
    assert not (tmp_path/'disabled').exists()
