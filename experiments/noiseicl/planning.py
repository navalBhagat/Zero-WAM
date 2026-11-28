"""Build campaign manifests; planning never submits jobs."""

from pathlib import Path

from experiments.io import read_json
from experiments.runtime.records import save_json
from experiments.runtime.provenance import snapshot
from wan_va.noise_inversion.dataset import gaussian_seed, stable_id
from .config import STAGES, UNSEEN_TASKS


def evaluation_jobs(settings, learned=False):
    methods = ["icl", "control", "oracle"]
    if learned:
        methods += ["learned", "learned_shuffled"]
    return [
        dict(
            name=f"{task}_{scene}",
            kind="evaluate",
            task=task,
            scene=scene,
            methods=methods,
            depends_on="train" if learned else None,
            submissions=[],
        )
        for task in UNSEEN_TASKS
        for scene in range(
            settings.scene_start, settings.scene_start + settings.scene_count
        )
    ]


def collection_jobs(settings, contexts):
    if not contexts:
        raise ValueError("Collection requires --contexts pointing to index.json")
    entries = read_json(Path(contexts))["contexts"]
    if any(row["task"] in UNSEEN_TASKS for row in entries):
        raise ValueError("Unseen tasks cannot enter collection")
    pairs = []
    for row in entries:
        for replica in range(settings.seeds_per_point):
            seed = gaussian_seed(row["id"], settings.sampling_seed, replica)
            pairs.append(
                dict(
                    id=stable_id(row["id"], seed),
                    context=row["path"],
                    context_sha256=row["sha256"],
                    split=row["split"],
                    seed=seed,
                )
            )
    if settings.stage == 3:
        pairs = pairs[:100]
    if not pairs:
        raise ValueError("No collection pairs planned")
    return [
        dict(
            name=f"collect_{start:06d}",
            kind="collect",
            submissions=[],
            pairs=pairs[start : start + settings.pairs_per_job],
        )
        for start in range(0, len(pairs), settings.pairs_per_job)
    ]


def completed_collection(dataset):
    if not dataset:
        raise ValueError(
            "Training requires --dataset pointing to a completed collection campaign"
        )
    root = Path(dataset).resolve()
    manifest = read_json(root / "manifest.json")
    if not manifest or manifest.get("stage") not in ("collect", "benchmark"):
        raise ValueError("Training input must be a collection campaign")
    for job in manifest["jobs"]:
        summary = read_json(root / "jobs" / job["name"] / "summary.json", {})
        if summary.get("status") != "completed":
            raise ValueError("Finish the collection campaign before planning training")
    return root


def seen_evaluation_jobs(dataset):
    seen, reserved, demo_pool = {}, {}, set()
    for path in sorted(dataset.glob("pairs/*/record.json")):
        pair = read_json(path)
        if pair.get("status") != "completed" or pair.get("fallback"):
            continue
        context = read_json(Path(pair["context"]))
        if context.get("simulator_seed") is None:
            raise ValueError(
                "Seen-scene evaluation requires verified simulator seeds. "
                "Use --training-only for recorded HumanGen data."
            )
        demo_pool.add(context["demo_latent"])
        reserved.setdefault(context["task"], set()).add(int(context["simulator_seed"]))
        if context["split"] == "test":
            seen[(context["task"], int(context["simulator_seed"]))] = context
    if not seen:
        raise ValueError("Dataset has no held-out seen scenes for evaluation")
    jobs = []
    for (task, scene), context in sorted(seen.items()):
        alternatives = sorted(demo_pool - {context["demo_latent"]})
        if not alternatives:
            raise ValueError("Shuffled-demo evaluation needs at least two human demos")
        jobs.append(
            dict(
                name=f"seen_{task}_{scene}",
                kind="evaluate",
                task=task,
                scene=scene,
                demo_latent=context["demo_latent"],
                reserved_scenes=sorted(reserved[task]),
                methods=["icl", "control", "oracle", "learned", "learned_shuffled"],
                shuffled_demo_latent=alternatives[0],
                depends_on="train",
                submissions=[],
            )
        )
    return jobs


def training_jobs(settings, dataset):
    root = completed_collection(dataset)
    jobs = [dict(name="train", kind="train", dataset=str(root), submissions=[])]
    if settings.training_only:
        from wan_va.noise_inversion.quality import inspect_dataset

        quality = inspect_dataset(root, check_tensors=False)
        if not quality["passed"]:
            raise ValueError(
                "Dataset quality checks failed: " + "; ".join(quality["errors"][:5])
            )
        if settings.overfit_pairs > quality["splits"]["train"]:
            raise ValueError(
                "Requested overfit subset is larger than the training split"
            )
        return jobs
    return jobs + evaluation_jobs(settings, learned=True) + seen_evaluation_jobs(root)


def plan(directory, settings, contexts=None, dataset=None):
    settings.validate()
    directory = Path(directory).resolve()
    if settings.stage in (1, 2):
        jobs = evaluation_jobs(settings)
    elif settings.stage in (3, 4):
        jobs = collection_jobs(settings, contexts)
    else:
        jobs = training_jobs(settings, dataset)

    directory.mkdir(parents=True, exist_ok=False)
    digest = snapshot(directory / "source.zip")
    stage = "train" if settings.training_only else STAGES[settings.stage]
    manifest = dict(
        schema_version=1,
        stage=stage,
        settings=settings.to_dict(),
        source_digest=digest,
        jobs=jobs,
    )
    save_json(directory / "manifest.json", manifest)
    for job in jobs:
        save_json(directory / "jobs" / job["name"] / "settings.json", job)
    return manifest
