"""Create five explicit campaign types; no jobs are submitted during planning."""
from pathlib import Path

from experiments.io import read_json
from experiments.runtime.records import save_json
from experiments.runtime.provenance import snapshot
from .config import STAGES, UNSEEN_TASKS
from wan_va.noise_inversion.dataset import stable_id, gaussian_seed


def evaluation_jobs(settings, learned=False):
    return [dict(name=f"{task}_{scene}", kind="evaluate", task=task, scene=scene,
                 methods=["icl", "control", "oracle"] + (["learned", "learned_shuffled"] if learned else []),
                 depends_on="train" if learned else None, submissions=[])
            for task in UNSEEN_TASKS for scene in range(settings.scene_start,
                                                       settings.scene_start + settings.scene_count)]


def plan(directory, settings, contexts=None, dataset=None):
    settings.validate()
    directory = Path(directory).resolve()
    jobs = []
    if settings.stage in (1, 2):
        jobs = evaluation_jobs(settings)
    elif settings.stage in (3, 4):
        if not contexts:
            raise ValueError("Collection requires --contexts pointing to index.json")
        entries = read_json(Path(contexts))["contexts"]
        if any(row["task"] in UNSEEN_TASKS for row in entries):
            raise ValueError("Unseen tasks cannot enter collection")
        pairs = []
        for row in entries:
            for replica in range(settings.seeds_per_point):
                seed = gaussian_seed(row["id"], settings.sampling_seed, replica)
                pairs.append(dict(id=stable_id(row["id"], seed), context=row["path"],
                                  context_sha256=row["sha256"], split=row["split"], seed=seed))
        if settings.stage == 3:
            pairs = pairs[:100]
        if not pairs:
            raise ValueError("No collection pairs planned")
        for start in range(0, len(pairs), settings.pairs_per_job):
            jobs.append(dict(name=f"collect_{start:06d}", kind="collect",
                             pairs=pairs[start:start + settings.pairs_per_job], submissions=[]))
    else:
        if not dataset:
            raise ValueError("Training requires --dataset pointing to a completed collection campaign")
        dataset_root = Path(dataset).resolve()
        collection = read_json(dataset_root / "manifest.json")
        if not collection or collection.get("stage") not in ("collect", "benchmark"):
            raise ValueError("Training input must be a collection campaign")
        if any(read_json(dataset_root / "jobs" / item["name"] / "summary.json", {}).get("status") != "completed"
               for item in collection["jobs"]):
            raise ValueError("Finish the collection campaign before planning training")
        jobs = [dict(name="train", kind="train", dataset=str(Path(dataset).resolve()), submissions=[])]
        jobs += evaluation_jobs(settings, learned=True)
        seen = {}
        demo_pool = set()
        reserved = {}
        for record in Path(dataset).glob("pairs/*/record.json"):
            pair = read_json(record)
            if pair.get("status") != "completed":
                continue
            context = read_json(Path(pair["context"]))
            if context.get("simulator_seed") is None:
                raise ValueError("Stage-5 seen-scene evaluation requires verified simulator seeds; "
                                 "recorded HumanGen episode IDs are not simulator seeds")
            demo_pool.add(context["demo_latent"])
            reserved.setdefault(context["task"], set()).add(int(context["simulator_seed"]))
            if context["split"] == "test":
                seen[(context["task"], int(context["simulator_seed"]))] = context
        for (task, scene), context in seen.items():
            jobs.append(dict(name=f"seen_{task}_{scene}", kind="evaluate", task=task, scene=scene,
                             demo_latent=context["demo_latent"], reserved_scenes=sorted(reserved[task]),
                             methods=["icl", "control", "oracle", "learned", "learned_shuffled"],
                             shuffled_demo_latent=next(demo for demo in sorted(demo_pool) if demo != context["demo_latent"]),
                             depends_on="train", submissions=[]))
        if not seen:
            raise ValueError("Dataset has no held-out seen scenes for evaluation")
    directory.mkdir(parents=True, exist_ok=False)
    digest = snapshot(directory / "source.zip")
    manifest = dict(schema_version=1, stage=STAGES[settings.stage], settings=settings.to_dict(),
                    source_digest=digest, jobs=jobs)
    save_json(directory / "manifest.json", manifest)
    for job in jobs:
        save_json(directory / "jobs" / job["name"] / "settings.json", job)
    return manifest
