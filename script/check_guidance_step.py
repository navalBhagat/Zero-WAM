"""Replay one captured WAM step with real VAE/DINO, without loading WAM or SAPIEN."""
import argparse
import json
from pathlib import Path
import time

import imageio.v2 as imageio
import torch
from PIL import Image

from wan_va.modules.utils import load_vae, resolve_model_component
from wan_va.guidance.dino_reward import DinoReward
from wan_va.guidance.latent_guidance import guided_step


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--capture', required=True)
    parser.add_argument('--model', required=True)
    parser.add_argument('--dino', required=True)
    parser.add_argument('--reference', required=True)
    parser.add_argument('--reference-start', type=int, default=0)
    parser.add_argument('--reference-stop', type=int, default=1)
    parser.add_argument('--strength', type=float, default=0.01)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    if args.strength <= 0:
        parser.error('strength must be positive')
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    step = torch.load(args.capture, map_location='cuda', weights_only=True)
    observed = step.pop('observed', None)
    vae = load_vae(resolve_model_component(args.model, 'vae'),
                   torch_dtype=step['sample'].dtype, torch_device='cuda')
    reward = DinoReward(vae, args.dino, torch.device('cuda'))
    reader = imageio.get_reader(args.reference)
    fps = float(reader.get_meta_data().get('fps', 12))
    frames = list(reader)
    reader.close()
    # Match server reference indexing after 12 FPS resampling.
    indices = torch.arange(max(1, int(len(frames) / fps * 12))) * fps / 12
    indices = indices.long().clamp(max=len(frames) - 1).unique()
    selected = indices[args.reference_start:args.reference_stop]
    import numpy as np
    reference = torch.from_numpy(np.stack([frames[i] for i in selected.tolist()]))
    reward.set_reference(reference.permute(0, 3, 1, 2).float() / 255)
    def reward_fn(clean):
        if observed is not None:
            clean = clean.clone()
            clean[:, :, :1] = observed
        return reward(clean)

    reports = []
    # Repetition helps expose retained graphs or cache growth.
    for repeat in range(3):
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        corrected, stats = guided_step(**step, reward_fn=reward_fn, strength=args.strength)
        with torch.no_grad():
            clean = step['sample'] - step['sigma'] * step['velocity']
            if observed is not None:
                clean[:, :, :1] = observed
            direction = corrected - step['base_next']
            # Hold velocity fixed: this tests the implemented approximation.
            stats['perturbed_reward'] = reward(clean + direction).item()
            for label, latent in [('before', clean), ('after', clean + direction)]:
                frame = reward.decode(latent)[0, :, -1].float().cpu()
                rgb = ((frame + 1) * 127.5).clamp(0, 255).byte().permute(1, 2, 0).numpy()
                Image.fromarray(rgb).save(output / f'{label}.png')
        torch.cuda.synchronize()
        stats.update(repeat=repeat, seconds=time.perf_counter() - started,
                     peak_allocated=torch.cuda.max_memory_allocated(),
                     peak_reserved=torch.cuda.max_memory_reserved(),
                     current_allocated=torch.cuda.memory_allocated())
        reports.append(stats)
        assert stats['gradient_norm'] > 0, 'Zero editable gradient'
        assert all(p.grad is None for p in vae.parameters())
        assert all(p.grad is None for p in reward.encoder.parameters())
    (output / 'diagnostics.json').write_text(json.dumps(
        dict(arguments=vars(args), measurements=reports), indent=2))
    print(json.dumps(reports, indent=2))


if __name__ == '__main__':
    main()
