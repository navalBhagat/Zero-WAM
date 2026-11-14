"""Opt-in CPU-only camera snapshots saved before each observation-cache update."""
import json
from pathlib import Path


class RolloutSnapshots:
    def __init__(self, output, enabled=False):
        self.output=Path(output);self.enabled=enabled;self.index=0
        if enabled:self.output.mkdir(parents=True,exist_ok=True)

    def write(self, observation, chunk, simulator_action_count, success):
        if not self.enabled:return
        import numpy as np
        from PIL import Image
        paths={}
        for camera in ('cam_high','cam_left_wrist','cam_right_wrist'):
            key='observation.images.'+camera
            array=np.asarray(observation[key])
            if array.dtype != np.uint8:
                if np.issubdtype(array.dtype,np.floating) and array.max()<=1:
                    array=array*255
                array=np.clip(array,0,255).astype(np.uint8)
            name=f'frame_{self.index:04d}_{camera}.png'
            temporary=self.output/(name+'.tmp')
            Image.fromarray(array).save(temporary,format='PNG')
            temporary.replace(self.output/name);paths[camera]=name
        with (self.output/'frames.jsonl').open('a') as stream:
            stream.write(json.dumps(dict(frame=self.index,chunk=chunk,
                simulator_action_count=int(simulator_action_count),success=bool(success),images=paths))+'\n')
        self.index+=1
