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
from wan_va.guidance.reference import reconstructed_reference, decoder_on_cpu
from wan_va.guidance.selection import FixedFrameSelector



def check_direction(step, reward_fn, captured_dtype):
    """Test the raw ascent direction, separately from Euler-update rounding."""
    with torch.enable_grad():
        x = step['sample'].detach().float().requires_grad_(True)
        clean = x - step['sigma'].float() * step['velocity'].detach().float()
        score = reward_fn(clean)
        grad, = torch.autograd.grad(score, x)
    if step['fixed_first_frame']:
        grad = grad.clone()
        grad[:, :, :1] = 0
    clean, grad = clean.detach(), grad.detach()
    baseline = score.item()
    norm_squared = grad.square().sum().item()
    trials = []
    with torch.no_grad():
        for alpha in (0.001, 0.01, 0.1, 1.0):
            delta = alpha * grad
            plus = reward_fn(clean + delta).item()
            minus = reward_fn(clean - delta).item()
            base = step['base_next'].to(captured_dtype)
            applied = (base + delta.to(captured_dtype)).float() - base.float()
            editable = grad != 0
            trials.append(dict(
                strength=alpha, reward_plus=plus, reward_minus=minus,
                central_slope=(plus - minus) / (2 * alpha),
                expected_slope=norm_squared,
                captured_applied_norm=applied.norm().item(),
                captured_changed_fraction=(applied[editable] != 0).float().mean().item()
                if editable.any() else 0.0,
            ))
    return dict(reward=baseline, gradient_norm=norm_squared ** 0.5,
                captured_dtype=str(captured_dtype), trials=trials)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--capture', required=True)
    parser.add_argument('--model', required=True)
    parser.add_argument('--dino', required=True)
    references = parser.add_mutually_exclusive_group(required=True)
    references.add_argument('--reference', help='Original human MP4')
    references.add_argument('--reference-latent', help='Cached human .pth; use reconstructed frames')
    parser.add_argument('--reference-frames', default='0', help='Explicit comma-separated human RGB indices (default: 0)')
    parser.add_argument('--strength', type=float, default=0.01)
    parser.add_argument('--output', required=True)
    parser.add_argument('--dtype', choices=['float32', 'captured'], default='float32',
                        help='Full precision diagnostic by default; does not change server inference')
    args = parser.parse_args()
    if args.strength <= 0:
        parser.error('strength must be positive')
    reference_path = args.reference_latent or args.reference
    if not Path(reference_path).is_file():
        parser.error(f'Reference file not found: {reference_path}')
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    step = torch.load(args.capture, map_location='cuda', weights_only=True)
    captured_dtype = step['sample'].dtype
    observed = step.pop('observed', None)
    dtype = torch.float32 if args.dtype == 'float32' else captured_dtype
    for key, value in step.items():
        if torch.is_tensor(value) and value.is_floating_point():
            step[key] = value.to(dtype)
    if observed is not None:
        observed = observed.to(dtype)
    # Avoid TF32 approximation while checking small directional changes.
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    vae = load_vae(resolve_model_component(args.model, 'vae'),
                   torch_dtype=step['sample'].dtype, torch_device='cuda')
    reward = DinoReward(vae, args.dino, torch.device('cuda'))
    if args.reference_latent:
        print(f'Using reconstructed human reference: {args.reference_latent}', flush=True)
        with decoder_on_cpu(vae):
            reference = reconstructed_reference(
                vae, args.reference_latent,
            )
        torch.cuda.empty_cache()
    else:
        reader = imageio.get_reader(args.reference)
        fps = float(reader.get_meta_data().get('fps', 12))
        frames = list(reader)
        reader.close()
        # Match server reference indexing after 12 FPS resampling.
        indices = torch.arange(max(1, int(len(frames) / fps * 12))) * fps / 12
        indices = indices.long().clamp(max=len(frames) - 1).unique()
        selected = indices
        import numpy as np
        reference = torch.from_numpy(np.stack([frames[i] for i in selected.tolist()]))
        reference = reference.permute(0, 3, 1, 2).float() / 255
    selected_frames = FixedFrameSelector(len(reference), args.reference_frames).select(0)
    reference = reference[selected_frames]
    reward.set_reference(reference)
    preview = (reference[0].detach().cpu() * 255).clamp(0, 255).byte()
    Image.fromarray(preview.permute(1, 2, 0).numpy()).save(output / 'reference.png')
    def reward_fn(clean):
        if observed is not None:
            clean = clean.clone()
            clean[:, :, :1] = observed
        return reward(clean)

    direction_check = None
    if args.dtype == 'float32':
        direction_check = check_direction(step, reward_fn, captured_dtype)
        (output / 'direction_check.json').write_text(json.dumps(direction_check, indent=2))
        print('Float32 direction check:', json.dumps(direction_check, indent=2), flush=True)

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
            stats['perturbed_reward'] = reward_fn(clean + direction).item()
            stats['applied_correction_norm'] = direction.float().norm().item()
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
        dict(arguments=vars(args), selected_reference_frames=selected_frames,
             direction_check=direction_check, measurements=reports), indent=2))
    print(json.dumps(reports, indent=2))


if __name__ == '__main__':
    main()
