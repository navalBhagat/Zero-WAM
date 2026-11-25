"""Bounded feasible-scene search, shared across every noise in a scene slot."""
from __future__ import annotations
from dataclasses import dataclass
import fcntl
from pathlib import Path
from typing import Callable

from experiments.io import read_json
from .records import save_json, read_episode


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
    receipt: Path | None = None

    def __post_init__(self):
        if self.requested_seed not in self.reserved_seeds or self.max_attempts < 1:
            raise ValueError("Scene search requires a reserved slot and positive attempt limit")

    def environment(self) -> dict[str, str]:
        return dict(NOISE_INVERSION_SCENE_REGISTRY=str(self.registry),
                    NOISE_INVERSION_RESERVED_SCENES=",".join(map(str, self.reserved_seeds)),
                    NOISE_INVERSION_MAX_SCENE_ATTEMPTS=str(self.max_attempts))

    def accepted_seed(self) -> int:
        if self.receipt is not None and self.receipt.exists():
            receipt = read_json(self.receipt)
            if receipt["requested_seed"] != self.requested_seed:
                raise ValueError("Scene receipt belongs to a different slot")
            return receipt["accepted_seed"]
        entry = read_json(self.registry, {}).get("slots", {}).get(str(self.requested_seed), {})
        if entry.get("accepted_seed") is None:
            # Older captures predate receipts. A completed source is independent
            # evidence of the actual scene; do not throw away its saved tensors.
            if self.receipt is not None and (self.receipt.parent / "config.json").exists():
                source = read_episode(self.receipt.parent)
                seed = source["simulator_seed"]
                if seed in self.reserved_seeds and seed != self.requested_seed:
                    raise ValueError("Captured scene conflicts with another reserved slot")
                _save_receipt(self, seed)
                return seed
            raise ValueError("Captured source has no recorded accepted scene")
        return entry["accepted_seed"]


def setup_rejection(error: BaseException) -> bool:
    """Recognize only the observed expert missing-grasp-pose assertion."""
    return isinstance(error, AssertionError) and str(error) == "target_pose cannot be None for move action."


def _restore_receipts(data: dict, registry: Path) -> None:
    """Capture-local receipts retain accepted scenes even if a registry snapshot is lost."""
    for path in list(registry.parent.glob("scene_*/capture/scene_selection.json")) + list(registry.parent.glob("scene_*/*/noise/scene_selection.json")):

        receipt = read_json(path)
        slot, accepted = str(receipt["requested_seed"]), receipt["accepted_seed"]
        if any(other_slot != slot and other["accepted_seed"] == accepted
               for other_slot, other in data["slots"].items()):
            raise ValueError(f"Accepted scene {accepted} belongs to multiple slots")
        entry = data["slots"].setdefault(slot, dict(accepted_seed=None, candidate_seed=accepted, attempts=[]))
        if entry["accepted_seed"] not in (None, accepted):
            raise ValueError(f"Conflicting scene receipts for slot {slot}")
        entry["accepted_seed"] = accepted


def _save_receipt(search: SceneSearch, seed: int) -> None:
    if search.receipt is not None:
        save_json(search.receipt, dict(requested_seed=search.requested_seed, accepted_seed=seed))


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
        _restore_receipts(data, search.registry)
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
            _save_receipt(search, accepted)
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
                _save_receipt(search, seed)
                return seed, result.episode_info
            attempt.update(status="infeasible", reason=result.reason)
            entry["candidate_seed"] = _replacement_seed(data, search.reserved_seeds)
            save_json(search.registry, data)
        raise ValueError(f"Scene slot {search.requested_seed} exhausted {search.max_attempts} feasibility attempts")
