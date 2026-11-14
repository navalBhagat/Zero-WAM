"""Paired FP32/FP16 reward-decoder replay on one captured WAM prediction.

Persistent latent arithmetic and DINO remain FP32. No transformer/simulator loaded.
Reward changes measure the local stop-gradient surrogate, not episode success.
"""
import argparse
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True,help='Completed stage-3 run folder containing capture_step.pt')
    p.add_argument('--model',required=True); p.add_argument('--dino',required=True)
    p.add_argument('--reference',type=Path,help='Override cached human latent path, e.g. after moving results')
    p.add_argument('--frames',help='Override comma-separated zero-based decoded reference frame indices')
    p.add_argument('--strengths',nargs='+',type=float,default=[.001,.01,.1,1.])
    p.add_argument('--repeats',type=int,default=3)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.repeats<1 or not a.strengths or any(s<=0 for s in a.strengths): p.error('Use positive strengths/repeats')
    if a.output.exists(): p.error('Choose a new output directory')
    import torch
    from wan_va.modules.utils import load_vae, resolve_model_component
    from wan_va.guidance.dino_reward import DinoReward
    from wan_va.guidance.latent_guidance import guided_step
    from wan_va.guidance.reference import reconstructed_reference, decoder_on_cpu
    from wan_va.guidance.selection import validate_frames
    if not torch.cuda.is_available(): p.error('Run on a CUDA compute node with zerowam active')
    torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
    capture=torch.load(a.run/'capture_step.pt',map_location='cuda',weights_only=True)
    config=json.loads((a.run/'config.json').read_text())
    captured_rgb=capture.get('reference_rgb') if not a.reference and not a.frames else None
    reference=a.reference or Path(config['episodes'][0]['selected_reference']['latent'] or '')
    if captured_rgb is None and not reference.is_file():
        p.error('No captured reference RGB: pass --reference to the corresponding cached human latent')
    indices=([int(v) for v in a.frames.split(',')] if a.frames else capture.get('reference_frames'))
    if indices is None: p.error('Capture has no selected target metadata; pass --frames explicitly')
    step={k:capture[k] for k in ('sample','velocity','sigma','base_next','fixed_first_frame')}
    step={k:v.float() if torch.is_tensor(v) and v.is_floating_point() else v for k,v in step.items()}
    observed=capture.get('observed')
    if observed is not None: observed=observed.float()
    a.output.mkdir(parents=True)
    rows=[]
    for dtype_name in ('float32','float16'):
        vae=load_vae(resolve_model_component(a.model,'vae'),torch_dtype=torch.float32,torch_device='cuda')
        reward=DinoReward(vae,a.dino,torch.device('cuda'),decoder_dtype=getattr(torch,dtype_name))
        if captured_rgb is not None:
            if captured_rgb.ndim != 4 or len(captured_rgb) != len(indices):
                raise ValueError('Captured reference RGB and frame indices disagree')
            selected=list(indices)
            reward.set_reference(captured_rgb.float())
        else:
            with decoder_on_cpu(vae): frames=reconstructed_reference(vae,str(reference))
            selected=validate_frames(indices,len(frames)); reward.set_reference(frames[selected]); del frames
        def reward_fn(clean):
            if observed is not None:
                clean=clean.clone(); clean[:,:,:1]=observed
            return reward(clean)
        clean=step['sample']-step['sigma']*step['velocity']
        for strength in a.strengths:
            for repeat in range(a.repeats):
                row=dict(decoder_dtype=dtype_name,strength=strength,repeat=repeat,
                         reference_frames=selected,status='failed')
                try:
                    torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats(); started=time.perf_counter()
                    corrected,stats=guided_step(**step,reward_fn=reward_fn,strength=strength,max_norm=1.)
                    with torch.no_grad(): after=reward_fn(clean+(corrected-step['base_next'])).item()
                    torch.cuda.synchronize()
                    if not torch.isfinite(torch.tensor(after)): raise FloatingPointError('Non-finite post-correction reward')
                    row.update(stats,reward_after=after,reward_gain=after-stats['reward'],
                               seconds=time.perf_counter()-started,
                               peak_allocated_gib=torch.cuda.max_memory_allocated()/1024**3,status='ok')
                    del corrected
                except (FloatingPointError,RuntimeError) as error:
                    row['error']=str(error)
                rows.append(row)
                (a.output/'precision.json').write_text(json.dumps(dict(
                    arguments={k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()},
                    definition='Only reward VAE decoder changes precision. Latents and DINO are FP32.',
                    measurements=rows),indent=2)+'\n')
        del reward,vae,clean; torch.cuda.empty_cache()
    print(f'Results: {a.output}/precision.json. Plot using experiments/analysis/plot_precision.py')


if __name__=='__main__': main()
