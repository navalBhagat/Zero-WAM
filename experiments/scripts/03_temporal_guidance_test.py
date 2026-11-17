"""CPU-only replay of scale observations: +4 versus DINO-distance lookahead.

Loads only DINO and, for a reconstructed reference, the VAE on CPU. No WAM
transformer, simulator, CUDA, Slurm submission, or new policy trajectory.
Targets are counterfactual selections on ONE recorded robot trajectory.
"""
import argparse
import csv
import importlib
import json
import os
from pathlib import Path
import sys
import types

ROOT = Path(__file__).resolve().parents[2]


def guidance_modules():
    # Import the small shared modules without wan_va/__init__.py, which loads
    # training/distributed/transformer dependencies irrelevant to CPU replay.
    name = '_temporal_offline_guidance'
    if name not in sys.modules:
        package = types.ModuleType(name)
        package.__path__ = [str(ROOT/'wan_va/guidance')]
        sys.modules[name] = package
    return (importlib.import_module(name+'.temporal'),
            importlib.import_module(name+'.dino_reward'),
            importlib.import_module(name+'.reference'))


def replay(reference_features, observation_features, chunks, window=24, lookahead=4,
           distance=None, smoothing=3, max_frames=24):
    temporal, _, _ = guidance_modules()
    fixed = temporal.WindowFrameSelector(len(reference_features), window, lookahead)
    adaptive = temporal.DistanceFrameSelector(reference_features, window, lookahead,
                                              distance, smoothing, max_frames)
    if len(chunks) != len(observation_features):
        raise ValueError('One chunk ID per observation is required')
    rows = []
    for chunk, query in zip(chunks, observation_features):
        scores = (reference_features @ query).tolist()
        frame_target = fixed.select(chunk, similarities=scores)[0]
        distance_target = adaptive.select(chunk, similarities=scores)[0]
        assert fixed.position == adaptive.position  # Only lookahead differs.
        rows.append(dict(chunk=chunk, matched_frame=fixed.position,
            matched_similarity=scores[fixed.position], frame_target=frame_target,
            distance_target=distance_target, **adaptive.details))
    return rows, adaptive


def load_reference(args, episode, reference_module):
    import torch
    paths = episode.get('server', {}).get('reference_paths', {}) or episode.get('selected_reference', {})
    kind = (episode.get('server', {}).get('guidance') or {}).get('reference_kind')
    if args.reference_video or (kind == 'original_video' and not args.reference_latent):
        import numpy as np
        import imageio.v2 as imageio
        import torch.nn.functional as F
        path = args.reference_video or Path(paths.get('video') or '')
        if not path.is_file(): raise ValueError(f'Reference video missing: {path}; pass --reference-video')
        reader = imageio.get_reader(str(path))
        try:
            fps = float(reader.get_meta_data().get('fps', 0) or 0)
            frames = np.stack(list(reader))
        finally: reader.close()
        if fps > 0:
            count = max(1, int(round(len(frames)/fps*args.reference_fps)))
            indices = np.unique(np.clip(np.floor(np.arange(count)*fps/args.reference_fps).astype(int),0,len(frames)-1))
            frames = frames[indices]
        tensor = torch.from_numpy(frames).permute(0,3,1,2).float()
        tensor = F.interpolate(tensor, size=(args.reference_height,args.reference_width), mode='bilinear',align_corners=False)/255
        return tensor, dict(kind='original_video', path=str(path.resolve()),
            fps=args.reference_fps, height=args.reference_height, width=args.reference_width,
            source_override=bool(args.reference_video and kind != 'original_video'))
    if args.reference_latent: kind='reconstructed_latent'
    if kind not in (None, 'reconstructed_latent'):
        raise ValueError(f'Unknown reference kind: {kind}')
    latent = args.reference_latent or Path(paths.get('latent') or '')
    model = args.model_path or Path(episode.get('server', {}).get('model_path') or '')
    vae_path = args.vae_path or model/'vae'
    if not latent.is_file(): raise ValueError(f'Reference latent missing: {latent}; pass --reference-latent')
    if not (vae_path/'config.json').is_file(): raise ValueError(f'VAE missing: {vae_path}; pass --vae-path or --model-path')
    from diffusers import AutoencoderKLWan
    vae = AutoencoderKLWan.from_pretrained(str(vae_path),torch_dtype=torch.float32,local_files_only=True).to('cpu').eval()
    with torch.inference_mode():
        frames = reference_module.reconstructed_reference(vae, latent)
    del vae
    return frames, dict(kind='reconstructed_latent',path=str(latent.resolve()),vae_path=str(vae_path.resolve()))


def figures(output, rows, selector, reference, observations):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from PIL import Image, ImageDraw
    fig, axes = plt.subplots(2,1,figsize=(10,7),layout='constrained')
    x = [r['chunk'] for r in rows]
    for key,label in [('matched_frame','Matched frame'),('frame_target','Frame lookahead'),('distance_target','DINO-distance lookahead')]:
        axes[0].plot(x,[r[key] for r in rows],marker='o',label=label)
    axes[0].set(ylabel='Human reference frame',title='Same observations and alignment; different future targets')
    axes[0].legend()
    axes[1].plot(range(len(selector.coordinate)),selector.coordinate)
    axes[1].set(xlabel='Human reference frame',ylabel='Cumulative cosine distance',
        title=f'Distance threshold = {selector.distance:.6g}; smoothing = {selector.smoothing}')
    fig.savefig(output/'targets.png',dpi=170);plt.close(fig)
    # At most six evenly spaced observed chunks, including the endpoints.
    chosen=sorted({round(i*(len(rows)-1)/max(1,min(6,len(rows))-1)) for i in range(min(6,len(rows)))})
    sheet=Image.new('RGB',(960,205*len(chosen)), 'white');draw=ImageDraw.Draw(sheet)
    for line,index in enumerate(chosen):
        row=rows[index]; y=line*205
        images=[observations[index]]+[Image.fromarray((reference[row[key]].permute(1,2,0).clamp(0,1).numpy()*255).astype('uint8'))
                                     for key in ('frame_target','distance_target')]
        labels=[f'Robot chunk {row["chunk"]}',f'+frames target {row["frame_target"]}',f'Distance target {row["distance_target"]}']
        for column,(img,label) in enumerate(zip(images,labels)):
            img=img.copy();img.thumbnail((310,175));sheet.paste(img,(column*320,y+25));draw.text((column*320+5,y+5),label,fill='black')
    sheet.save(output/'comparison.png')


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,default=ROOT/'experiments/stages/03_guidance_pilot/runs/place_object_scale_29326930')
    p.add_argument('--episode',type=int,default=0,help='config.json episode ID')
    p.add_argument('--output',type=Path,default=ROOT/'experiments/stages/03_temporal_guidance_test')
    p.add_argument('--dino-checkpoint',type=Path,default=Path(os.environ.get('GUIDANCE_CHECKPOINT',ROOT/'checkpoints/dinov2-small')))
    p.add_argument('--reference-latent',type=Path);p.add_argument('--reference-video',type=Path)
    p.add_argument('--model-path',type=Path);p.add_argument('--vae-path',type=Path)
    p.add_argument('--reference-fps',type=float,default=12)
    p.add_argument('--reference-height',type=int,default=320);p.add_argument('--reference-width',type=int,default=480)
    p.add_argument('--window',type=int,default=24);p.add_argument('--lookahead',type=int,default=4)
    p.add_argument('--distance-increment',type=float);p.add_argument('--smoothing',type=int,default=3)
    p.add_argument('--max-lookahead-frames',type=int,default=24)
    p.add_argument('--threads',type=int,default=4)
    args=p.parse_args(argv)
    if args.reference_latent and args.reference_video:p.error('Choose one reference override')
    import math
    if args.window<1 or args.lookahead<0 or args.max_lookahead_frames<1 or args.smoothing<1 or args.smoothing%2!=1:
        p.error('Window/max lookahead must be positive; lookahead nonnegative; smoothing positive odd')
    if args.distance_increment is not None and (not math.isfinite(args.distance_increment) or args.distance_increment<=0):
        p.error('Distance increment must be finite and positive')
    if args.threads<1 or args.reference_fps<=0 or min(args.reference_height,args.reference_width)<1:p.error('Invalid CPU/reference settings')
    if args.output.exists() and any(args.output.iterdir()):p.error('Output is not empty; choose a new --output')
    config_path=args.run/'config.json'
    if not config_path.is_file():p.error(f'Run config missing: {config_path}; pass --run')
    config=json.loads(config_path.read_text())
    episode=next((e for e in config.get('episodes',[]) if e['episode_id']==args.episode),None)
    if episode is None:p.error('Episode ID not found')
    if episode.get('task',config.get('TASK'))!='place_object_scale':p.error('This phase is for place_object_scale')
    preview=(episode.get('server',{}).get('guidance') or {}).get('reference_preview_dir')
    # Preserve reference_episode ordinal rather than assuming it equals resumed episode ID.
    ordinal=config['episodes'].index(episode)
    directory=args.run/'references'/f'episode_{ordinal:03d}'
    if preview and (args.run/'references'/Path(preview).name).is_dir():directory=args.run/'references'/Path(preview).name
    trace=directory/'selection.jsonl'
    if not trace.is_file():p.error(f'Observation trace missing: {trace}')
    selected=[json.loads(line) for line in trace.read_text().splitlines() if line.strip()]
    chunks=[r['chunk'] for r in selected]
    if not chunks or chunks!=sorted(set(chunks)):p.error('Expected nonempty strictly increasing chunk IDs')
    image_paths=[directory/Path(r.get('robot_preview_path') or f'robot_chunk_{r["chunk"]:04d}.png').name for r in selected]
    if any(not path.is_file() for path in image_paths):p.error('Missing saved robot previews; copy the full references/episode_* folder')
    if not (args.dino_checkpoint/'config.json').is_file():p.error(f'DINO checkpoint missing: {args.dino_checkpoint}')
    # Everything below runs on CPU, even on a host with visible GPUs.
    import torch
    import numpy as np
    from PIL import Image
    from transformers import AutoModel
    torch.set_num_threads(args.threads)
    temporal,dino,reference_module=guidance_modules()
    try:reference,provenance=load_reference(args,episode,reference_module)
    except (ValueError,OSError) as error:p.error(str(error))
    print(f'CPU only: {len(reference)} human frames, {len(chunks)} saved robot observations.',flush=True)
    reward=dino.DinoReward.__new__(dino.DinoReward)
    reward.encoder=AutoModel.from_pretrained(str(args.dino_checkpoint),local_files_only=True).to('cpu').eval().requires_grad_(False)
    observations=[]
    for path in image_paths:
        with Image.open(path) as image: observations.append(image.convert('RGB').copy())
    with torch.inference_mode():
        features=torch.cat([reward.features(batch) for batch in reference.split(4)])
        queries=torch.cat([reward.features(torch.from_numpy(np.asarray(img).copy()).permute(2,0,1).unsqueeze(0).float()/255)
                           for img in observations])
    rows,selector=replay(features,queries,chunks,args.window,args.lookahead,args.distance_increment,args.smoothing,args.max_lookahead_frames)
    args.output.mkdir(parents=True,exist_ok=True)
    with (args.output/'results.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    figures(args.output,rows,selector,reference,observations)
    provenance.update(run=str(args.run.resolve()),episode_id=args.episode,device='cpu',
        dino_checkpoint=str(args.dino_checkpoint.resolve()),window=args.window,frame_lookahead=args.lookahead,
        distance_increment=selector.distance,smoothing=args.smoothing,max_forward_frames=args.max_lookahead_frames,
        fallback_chunks=sum(bool(r['fallback']) for r in rows),cap_or_end_limited_chunks=sum(r['horizon_limited'] for r in rows),
        interpretation='Counterfactual target selection on a recorded trajectory; does not measure policy success.')
    (args.output/'provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
    print(f'Threshold: {selector.distance:.6g}; bounded chunks: {provenance["cap_or_end_limited_chunks"]}/{len(rows)}')
    print(f'Outputs: {args.output}/results.csv, targets.png, comparison.png, provenance.json')


if __name__=='__main__':main()
