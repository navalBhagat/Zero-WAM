"""Snapshot experiment/runtime source and reject code changes while jobs queue."""

import hashlib
import json
import zipfile

from experiments.paths import ROOT


def files() -> list:
    """Enumerate runtime and nested experiment code; exclude all generated runs."""
    paths = []
    for folder in ("wan_va", "evaluation"):
        paths += [
            p
            for p in (ROOT / folder).rglob("*")
            if p.suffix in (".py", ".sh", ".json", ".yml", ".yaml")
            and "__pycache__" not in p.parts
        ]
    for folder in ("scripts", "analysis", "jobs", "tests", "campaign", "replay", "noise_inversion"):
        paths += [
            p
            for p in (ROOT / "experiments" / folder).rglob("*")
            if p.suffix in (".py", ".sbatch", ".sh") and "__pycache__" not in p.parts
        ]
    paths += list((ROOT / "experiments").glob("*.py"))
    paths += list((ROOT / "experiments/tests/fixtures").rglob("*.json"))
    paths += [ROOT / "tests/test_guidance_temporal.py"]
    paths += [ROOT / p for p in ("pyproject.toml", "requirements.inference.txt")]
    return sorted(set(p for p in paths if p.is_file()))


def fingerprint() -> tuple[str, dict[str, str]]:
    """Return an ordered file-hash map and its aggregate digest."""
    hashes = {
        str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in files()
    }
    digest = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()
    return digest, hashes


def snapshot(destination) -> str:
    """Archive the same files used by the queued-job digest, plus operator guides."""
    digest, hashes = fingerprint()
    with zipfile.ZipFile(destination, "w", zipfile.ZIP_DEFLATED) as archive:
        for name in hashes:
            archive.write(ROOT / name, name)
        for extra in ("experiments/README.md", "experiments/analysis/requirements.txt"):
            if (ROOT / extra).is_file():
                archive.write(ROOT / extra, extra)
        for doc in sorted((ROOT / "experiments/docs").glob("*.md")):
            archive.write(doc, str(doc.relative_to(ROOT)))
        archive.writestr("source_hashes.json", json.dumps(hashes, indent=2))
    return digest
