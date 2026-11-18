"""One GPU DINO-distance rollout paired with scale +4 pilot job 29326930.

Dry-run by default. No CPU replay or baseline rerun is launched.
"""
import argparse
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]


def command(manifest, submit=False, account=None, budget=None):
    cmd = [sys.executable, str(ROOT/'experiments/scripts/robotwin_suite.py'), 'guidance',
        '--tasks', 'place_object_scale', '--guidance-modes', 'guidance',
        '--guidance-selectors', 'window', '--seeds', '0', '--sampling-seed', '1234',
        '--episodes', '1', '--spp', '8', '--phase-separation', '--compact-cache',
        '--window', '24', '--lookahead', '4', '--lookahead-mode', 'dino_distance',
        '--distance-smoothing', '3',
        '--max-lookahead-frames', '24', '--strength', '0.01',
        '--reference-source', 'latent', '--partition', 'gpu_a100',
        '--time', '01:00:00', '--client-timeout', '3000', '--minutes-per-episode', '20',
        '--manifest', str(manifest)]
    if submit: cmd.append('--submit')
    if account: cmd += ['--account', account]
    if budget is not None: cmd += ['--budget-sbu', str(budget)]
    return cmd


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--submit',action='store_true')
    p.add_argument('--manifest',type=Path,default=ROOT/'experiments/stages/03_temporal_guidance_same_seed/manifest.json')
    p.add_argument('--account');p.add_argument('--budget-sbu',type=float)
    args=p.parse_args()
    print('One A100 episode: scale, seed group 0 (expected simulator seed 100000), noise seed 1234.',flush=True)
    print('Compare with +4 pilot job 29326930; verify actual simulator seed and reference path match in config.json.',flush=True)
    subprocess.run(command(args.manifest,args.submit,args.account,args.budget_sbu),cwd=ROOT,check=True)


if __name__=='__main__': main()
