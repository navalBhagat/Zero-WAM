"""Bounded feasible-scene search, shared across every noise in a scene slot."""
from dataclasses import dataclass
import fcntl
from pathlib import Path
from typing import Callable

from experiments.io import read_json
from .records import save_json


@dataclass(frozen=True)
class ProbeResult:
    feasible: bool
    episode_info: dict
    reason: str = ""


@dataclass(frozen=True)
class SceneSearch:
    registry: Path
    requested_seed: int
    reserved_seeds: tuple[int, ...]
    max_attempts: int = 20

    def __post_init__(self):
        if self.requested_seed not in self.reserved_seeds or self.max_attempts < 1:
            raise ValueError("Scene search requires a reserved slot and positive attempt limit")

    def environment(self) -> dict[str, str]:
        return dict(NOISE_INVERSION_SCENE_REGISTRY=str(self.registry),
                    NOISE_INVERSION_RESERVED_SCENES=",".join(map(str, self.reserved_seeds)),
                    NOISE_INVERSION_MAX_SCENE_ATTEMPTS=str(self.max_attempts))

    def accepted_seed(self) -> int:
        entry = read_json(self.registry, {}).get("slots", {}).get(str(self.requested_seed), {})
        if entry.get("accepted_seed") is None:
            raise ValueError("Captured source has no recorded accepted scene")
        return entry["accepted_seed"]


def _replacement_seed(data: dict, reserved: tuple[int, ...]) -> int:
    used = set(reserved)
    for entry in data["slots"].values():
        used.update(attempt["seed"] for attempt in entry["attempts"])
    seed = max(reserved) + 1
    while seed in used:
        seed += 1
    return seed


def select_scene(search: SceneSearch, probe: Callable[[int], ProbeResult]) -> tuple[int, dict]:
    """Only explicit infeasibility advances a seed; crashes stay retryable errors.

    Hold a per-task lock through feasibility checking so concurrent noise jobs
    cannot assign different accepted scenes to the same slot.
    """
    search.registry.parent.mkdir(parents=True, exist_ok=True)
    with search.registry.with_suffix(".lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        data = read_json(search.registry, dict(schema_version=1, slots={}))
        identity = dict(reserved_seeds=list(search.reserved_seeds), max_attempts=search.max_attempts)
        if data.get("search_settings", identity) != identity:
            raise ValueError("Scene-search settings changed; use a new campaign")
        data["search_settings"] = identity
        entry = data["slots"].setdefault(str(search.requested_seed), dict(
            accepted_seed=None, candidate_seed=search.requested_seed, attempts=[]))
        accepted = entry["accepted_seed"]
        if accepted is not None:
            # Never replace a scene after another noise has already used it.
            attempt = dict(seed=accepted, status="rechecking")
            entry["attempts"].append(attempt)
            save_json(search.registry, data)
            try:
                result = probe(accepted)
                if not result.feasible:
                    raise ValueError(f"Accepted scene {accepted} is no longer feasible: {result.reason}")
            except Exception as error:
                attempt.update(status="error", reason=str(error))
                save_json(search.registry, data)
                raise
            attempt["status"] = "rechecked"
            save_json(search.registry, data)
            return accepted, result.episode_info

        while sum(a["status"] not in ("rechecked", "rechecking") for a in entry["attempts"]) < search.max_attempts:
            seed = entry["candidate_seed"]
            attempt = dict(seed=seed, status="checking")
            entry["attempts"].append(attempt)
            save_json(search.registry, data)
            try:
                result = probe(seed)
            except Exception as error:
                attempt.update(status="error", reason=str(error))
                save_json(search.registry, data)
                raise
            if result.feasible:
                attempt["status"] = "accepted"
                entry["accepted_seed"] = seed
                save_json(search.registry, data)
                return seed, result.episode_info
            attempt.update(status="infeasible", reason=result.reason)
            entry["candidate_seed"] = _replacement_seed(data, search.reserved_seeds)
            save_json(search.registry, data)
        raise ValueError(f"Scene slot {search.requested_seed} exhausted {search.max_attempts} feasibility attempts")
