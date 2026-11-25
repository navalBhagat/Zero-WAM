"""Content hashes cover executable source, never generated results or datasets."""
import hashlib
import json
import zipfile
from experiments.paths import ROOT


def files():
    paths = []
    for folder in ("wan_va", "evaluation", "experiments/noiseicl",
                   "experiments/runtime", "experiments/scripts", "experiments/jobs"):
        paths.extend(p for p in (ROOT / folder).rglob("*")
                     if p.suffix in (".py", ".sh", ".sbatch", ".json", ".yaml", ".yml")
                     and "__pycache__" not in p.parts)
    paths.extend((ROOT / "experiments").glob("*.py"))
    paths.extend(ROOT / name for name in ("pyproject.toml", "requirements.inference.txt"))
    return sorted(set(p for p in paths if p.is_file()))


def fingerprint():
    hashes = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files()}
    return hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest(), hashes


def snapshot(destination):
    digest, hashes = fingerprint()
    with zipfile.ZipFile(destination, "w", zipfile.ZIP_DEFLATED) as archive:
        for name in hashes:
            archive.write(ROOT / name, name)
        archive.writestr("source_hashes.json", json.dumps(hashes, sort_keys=True))
    return digest
