"""Small CPU training run checks validation selection and resumability."""
from pathlib import Path
import torch
from experiments.runtime.records import save_json
from experiments.noiseicl.training import train
from wan_va.noise_inversion.adapter import load_adapter
def test_training_checkpoint_restores_model_optimizer_and_rng(tmp_path):
 root=tmp_path;dataset=root/'dataset';out=root/'train';out.mkdir()
 for split in ('train','val','test'):
  demo=root/f'{split}_demo.pt';torch.save(dict(latent=torch.randn(1,48,2,4,4)),demo)
  initial=root/f'{split}_initial.pt';torch.save(dict(initial_frame=torch.randn(1,48,1,4,4)),initial)
  context=root/f'{split}.json';save_json(context,dict(task='seen_example',scene_id=split,simulator_seed=100000+('train','val','test').index(split),demo_id=split,demo_latent=str(demo),initial=str(initial),history=[],split=split,chunk=0))
  directory=dataset/'pairs'/split;directory.mkdir(parents=True)
  original=torch.randn(1,48,2,4,4);mask=torch.ones_like(original);mask[:,:,:1]=0
  pair=directory/'pair.pt';torch.save(dict(original=original,recovered=original*.5,mask=mask),pair)
  save_json(directory/'record.json',dict(status='completed',context=str(context),pair=str(pair),fallback=False))
 settings=dict(sampling_seed=1234,train_steps=2,effective_batch=1,width=32,layers=1,heads=4,architecture='vit',overfit_pairs=0)
 result=train(dataset,out,settings)
 assert result['status']=='completed'
 loaded=load_adapter(out/'adapter.pt','cpu')
 assert loaded.config.width==32
 result=train(dataset,out,dict(settings,train_steps=3))
 assert result['status']=='completed'
 assert (out/"latest.pt").exists()
