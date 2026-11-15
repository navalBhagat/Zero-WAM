"""Snapshot experiment/runtime source and reject code changes while jobs queue."""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[2]


def files():
    paths = []
    for folder in ('wan_va', 'evaluation'):
        paths += [p for p in (ROOT/folder).rglob('*') if p.suffix in
                  ('.py', '.sh', '.json', '.yml', '.yaml') and '__pycache__' not in p.parts]
    for folder in ('scripts', 'analysis', 'jobs', 'tests'):
        paths += [p for p in (ROOT/'experiments'/folder).glob('*') if p.suffix in ('.py', '.sbatch')]
    paths += [ROOT/'tests/test_guidance_temporal.py']
    paths += [ROOT/p for p in ('pyproject.toml', 'requirements.inference.txt')]
    return sorted(set(p for p in paths if p.is_file()))


def fingerprint():
    hashes = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files()}
    digest = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()
    return digest, hashes


def snapshot(destination):
    digest, hashes = fingerprint()
    with zipfile.ZipFile(destination, 'w', zipfile.ZIP_DEFLATED) as archive:
        for name in hashes:
            archive.write(ROOT/name, name)
        for extra in ('experiments/README.md', 'experiments/analysis/requirements.txt'):
            if (ROOT/extra).is_file(): archive.write(ROOT/extra,extra)
        archive.writestr('source_hashes.json', json.dumps(hashes, indent=2))
    return digest


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--verify', required=True)
    a = p.parse_args()
    actual, _ = fingerprint()
    if actual != a.verify:
        raise SystemExit('Source changed after planning: use a fresh manifest or restore the stage source snapshot.')
