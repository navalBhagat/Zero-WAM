"""Plan/resume Zero-WAM downloads without importing torch or requiring a GPU."""
import argparse
from fnmatch import fnmatchcase
import json
import os
from pathlib import Path
import runpy
import shutil
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed


def download_selected_files(download_file, *, repo_id, repo_type, revision, local_dir,
                            allow_patterns, max_workers):
    """Use the already resolved file list instead of enumerating the repo again."""
    def fetch(name):
        return download_file(repo_id=repo_id, repo_type=repo_type, revision=revision,
                             local_dir=local_dir, filename=name)

    print(f"Downloading {len(allow_patterns)} explicitly selected files; "
          "skipping snapshot repository enumeration.", flush=True)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        pending = {executor.submit(fetch, name): name for name in allow_patterns}
        for count, future in enumerate(as_completed(pending), 1):
            future.result()
            print(f"[file {count}/{len(pending)}] Ready: {pending[future]}", flush=True)


def download_with_status(download, *, sizes, **kwargs):
    """Report durable file progress even when Hub progress bars are suppressed."""
    destination = Path(kwargs["local_dir"])
    stopped = threading.Event()
    started = time.monotonic()

    def status():
        complete = 0
        stored = 0
        for name, expected in sizes.items():
            try:
                actual = (destination / name).stat().st_size
            except FileNotFoundError:
                continue
            complete += actual == expected
            stored += min(actual, expected)
        print(f"[download {time.monotonic() - started:.0f}s] "
              f"{complete}/{len(sizes)} selected files present at expected size; "
              f"{stored / GIB:.3f}/{sum(sizes.values()) / GIB:.3f} GiB in destination. "
              "Temporary transfers/Hub metadata requests are not counted.", flush=True)

    def heartbeat():
        while not stopped.wait(15):
            status()

    print("Starting Hugging Face download (may first fetch metadata or wait for cache locks).", flush=True)
    status()
    worker = threading.Thread(target=heartbeat, daemon=True)
    worker.start()
    try:
        return download(**kwargs)
    finally:
        stopped.set()
        worker.join()
        status()

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_REPO = "robbyant-research/zero-wam-posttrain-robotwin"
DATA_REPO = "Robbyant-Research/HumanGen"
GIB = 1024 ** 3


def reference_files(seen=False, tasks=None):
    mapping = runpy.run_path(str(
        PROJECT_ROOT / "evaluation/robotwin/robotwin_icl_human_videos.py"
    ))["ROBOTWIN_ICL_SEEN_HUMAN_VIDEOS" if seen else "ROBOTWIN_ICL_UNSEEN_HUMAN_VIDEOS"]
    if tasks:
        unknown = set(tasks) - set(mapping)
        if unknown:
            raise ValueError(f"Not seen RoboTwin tasks: {sorted(unknown)}")
        mapping = {task: mapping[task] for task in tasks}
    root = PROJECT_ROOT / "data/HumanGen"
    files = []
    for videos in mapping.values():
        for video in videos:
            relative = Path(video).relative_to(root)
            files.append(relative.as_posix())
            latent = Path("human_latents", *relative.parts[1:]).with_suffix(".pth")
            files.append(latent.as_posix())
    return sorted(set(files))


def evaluation_files():
    return reference_files()


def select_files(names, component, profile, tasks=None):
    if component == "checkpoint":
        return sorted(names)
    if profile in ("eval", "seen"):
        references = set(reference_files(seen=profile == "seen", tasks=tasks))
        required = {n for n in references if n.startswith("human_latents/")}
        missing = required - set(names)
        if missing:
            raise RuntimeError(f"Expected {profile} references missing from Hub: {sorted(missing)}")
        optional = references - required
        unavailable = optional - set(names)
        if unavailable and profile == "eval":
            print(f"Note: {len(unavailable)} reference MP4s unavailable; using required latents. "
                  "Use GUIDANCE_REFERENCE_SOURCE=latent or reconstruct videos from latents.")
        selected = required | {n for n in names if fnmatchcase(n, "icl_configs/*")}
        if profile == "eval":
            selected |= optional & set(names)
        if profile == "seen":
            mapping = runpy.run_path(str(PROJECT_ROOT / "evaluation/robotwin/robotwin_icl_human_videos.py"))[
                "ROBOTWIN_ICL_SEEN_HUMAN_VIDEOS"]
            chosen = set(tasks or mapping)
            robot_files = {n for n in names if n.startswith("robotwin_data/") and
                           len(Path(n).parts) > 2 and Path(n).parts[1].split("-")[0] in chosen
                           and Path(n).parts[2] in ("latents", "data", "meta")
                           and Path(n).suffix.lower() not in (".mp4", ".avi", ".webm", ".mov", ".mkv")}
            for task in chosen:
                if not any(Path(n).parts[1].split("-")[0] == task for n in robot_files):
                    raise RuntimeError(f"No recorded robot files found for seen task: {task}")
            selected |= robot_files
            selected |= {n for n in names if n.startswith("robotwin_data/meta/")}
        return sorted(selected)
    patterns = ("icl_configs/*", "robotwin_data/**",
                "human_latents/robotwin/**", "human_data/robotwin/**")
    return sorted(n for n in names if any(fnmatchcase(n, p) for p in patterns))


def additional_bytes(files, destination):
    total = 0
    for name, size in files.items():
        if size is None:
            raise RuntimeError(f"No size metadata for {name}; cannot plan quota safely")
        target = destination / name
        if not target.is_file() or target.stat().st_size != size:
            total += size
    return total


def link_destination(link, target):
    # Never replace a real directory or an unrelated/broken symlink.
    if link.is_symlink() or link.exists():
        if link.resolve() != target.resolve():
            raise RuntimeError(f"Refusing to replace {link}; move existing assets manually first")
        return
    link.parent.mkdir(parents=True, exist_ok=True)
    link.symlink_to(target, target_is_directory=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("component", choices=("checkpoint", "dataset", "all"))
    parser.add_argument("--storage-root", type=Path, required=True,
                        help="Explicit external storage path; never defaults to home")
    parser.add_argument("--profile", choices=("eval", "seen", "robotwin"), default="eval",
                        help="eval: held-out human references; seen: 43 training tasks; robotwin: full subset")
    parser.add_argument("--tasks", nargs="+", help="With --profile seen: download only these seen tasks")
    parser.add_argument("--download", action="store_true", help="Default is metadata-only planning")
    parser.add_argument("--quota-free-gib", type=float,
                        help="Your currently remaining personal quota on this filesystem, in GiB")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--link", action="store_true", help="Link downloaded content into repository paths")
    parser.add_argument("--checkpoint-revision", default="main")
    parser.add_argument("--dataset-revision", default="main")
    args = parser.parse_args()
    if args.tasks and (args.profile != "seen" or args.component == "checkpoint"):
        parser.error("--tasks requires a dataset download with --profile seen")
    if args.workers < 1:
        parser.error("--workers must be positive")
    if args.download and (args.quota_free_gib is None or args.quota_free_gib < 0):
        parser.error("Downloads require --quota-free-gib from your quota report; df free space is insufficient")
    if args.link and not args.download:
        parser.error("--link requires --download")

    root = args.storage_root.expanduser().resolve()
    # Redirect potentially large Xet chunks without relocating login credentials.
    os.environ.setdefault("HF_XET_CACHE", str(root / ".xet-cache"))
    from huggingface_hub import HfApi, hf_hub_download
    api = HfApi()
    # Use saved hf auth login credentials (or HF_TOKEN); do not print tokens.
    try:
        api.whoami()
    except Exception as error:
        raise RuntimeError("Authentication check failed. Run hf auth login and check connectivity.") from error

    root.mkdir(parents=True, exist_ok=True)
    manifest_path = root / "download-manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    components = ("checkpoint", "dataset") if args.component == "all" else (args.component,)
    plans = []
    for component in components:
        repo = MODEL_REPO if component == "checkpoint" else DATA_REPO
        repo_type = "model" if component == "checkpoint" else "dataset"
        requested = getattr(args, component + "_revision")
        previous = manifest.get(component)
        # Resume the originally resolved commit, even if upstream main has changed.
        revision = previous["revision"] if previous else requested
        if previous and requested not in ("main", revision, previous.get("requested_revision")):
            raise RuntimeError("Revision differs from saved manifest. Use a new storage root for a new revision.")
        info = api.repo_info(repo, repo_type=repo_type, revision=revision, files_metadata=True)
        sizes = {s.rfilename: s.size for s in info.siblings}
        names = select_files(sizes, component, args.profile, args.tasks)
        if not names:
            raise RuntimeError(f"Empty download selection for {component}")
        destination = root / ("zero-wam-posttrain-robotwin" if component == "checkpoint" else "HumanGen")
        selected = {name: sizes[name] for name in names}
        remaining = additional_bytes(selected, destination)
        print(f"{component}: {len(names)} files, {sum(selected.values()) / GIB:.2f} GiB total; "
              f"conservative additional content {remaining / GIB:.2f} GiB")
        print(f"  Commit: {info.sha}\n  Destination: {destination}")
        plans.append((component, repo, repo_type, info.sha, destination, names, selected, remaining, requested))

    remaining = sum(p[7] for p in plans)
    required = remaining + 2 * GIB  # temporary/cache margin, not a guarantee
    filesystem_free = shutil.disk_usage(root).free
    print(f"Required allowance including 2 GiB margin: {required / GIB:.2f} GiB")
    print(f"Filesystem free space: {filesystem_free / GIB:.2f} GiB (not your quota)")
    if not args.download:
        print("PLAN ONLY: no file content downloaded. Rerun with --download and --quota-free-gib.")
        return
    if required > min(args.quota_free_gib * GIB, filesystem_free):
        raise RuntimeError("Insufficient declared quota or filesystem space. Request storage before downloading.")

    # Check link conflicts before any expensive transfer.
    if args.link:
        for component, _, _, _, destination, *_ in plans:
            link = PROJECT_ROOT / ("checkpoints/zero-wam-posttrain-robotwin" if component == "checkpoint" else "data/HumanGen")
            if (link.exists() or link.is_symlink()) and link.resolve() != destination.resolve():
                raise RuntimeError(f"Existing path conflicts with download destination: {link}")

    for component, repo, repo_type, sha, destination, names, sizes, _, requested in plans:
        manifest[component] = dict(repo=repo, revision=sha, requested_revision=requested,
                                   profile=args.profile if component == "dataset" else None,
                                   tasks=args.tasks if component == "dataset" else None,
                                   files=names, complete=False)
        temporary = manifest_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(manifest, indent=2) + "\n")
        temporary.replace(manifest_path)
        download_with_status(lambda **kwargs: download_selected_files(hf_hub_download, **kwargs),
                             sizes=sizes, repo_id=repo, repo_type=repo_type, revision=sha,
                             local_dir=str(destination), allow_patterns=names, max_workers=args.workers)
        bad = [name for name, size in sizes.items()
               if not (destination / name).is_file() or (destination / name).stat().st_size != size]
        if bad:
            raise RuntimeError(f"Missing or size-mismatched files after download: {bad[:10]}")
        manifest[component]["complete"] = True
        temporary.write_text(json.dumps(manifest, indent=2) + "\n")
        temporary.replace(manifest_path)
        if args.link:
            link = PROJECT_ROOT / ("checkpoints/zero-wam-posttrain-robotwin" if component == "checkpoint" else "data/HumanGen")
            link_destination(link, destination)
    print("Requested downloads completed and file sizes verified. No inference was run.")


if __name__ == "__main__":
    main()
