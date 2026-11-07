"""Apply the pinned RoboTwin installer patches, with backups and validation."""
import argparse
import importlib.util
from pathlib import Path
import re
import shutil


def patch_text(path, pattern, replacement, already_present):
    original = path.read_text()
    updated, count = re.subn(pattern, replacement, original)
    if count == 0:
        if already_present not in original:
            raise RuntimeError(f"Unrecognized source; refusing to patch {path}")
        print(f"Already patched: {path}")
        return
    backup = path.with_name(path.name + ".zerowam-upstream")
    if not backup.exists():
        shutil.copy2(path, backup)
    path.write_text(updated)
    print(f"Patched {path} ({count} replacements)")


def package_root(name):
    spec = importlib.util.find_spec(name)
    if spec is None or spec.origin is None:
        raise RuntimeError(f"Package not found: {name}")
    return Path(spec.origin).parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--robotwin-root", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, required=True)
    args = parser.parse_args()
    patch_text(
        package_root("sapien") / "wrapper/urdf_loader.py",
        r'("r")(\))( as)', r'\1, encoding="utf-8") as',
        'encoding="utf-8"',
    )
    patch_text(
        package_root("mplib") / "planner.py",
        r'(if np.linalg.norm\(delta_twist\) < 1e-4 )(or collide )(or not within_joint_limit:)',
        r'\1\3',
        'if np.linalg.norm(delta_twist) < 1e-4 or not within_joint_limit:',
    )
    for task in ("move_stapler_pad", "stamp_seal"):
        source = args.project_root / "evaluation/robotwin/envs" / f"{task}.py"
        target = args.robotwin_root / "envs" / f"{task}.py"
        if not source.is_file() or not target.is_file():
            raise FileNotFoundError(f"Missing evaluation patch source or target for {task}")
        backup = target.with_name(target.name + ".zerowam-upstream")
        if not backup.exists():
            shutil.copy2(target, backup)
        shutil.copy2(source, target)
        print(f"Installed evaluation criteria: {task}")


if __name__ == "__main__":
    main()
