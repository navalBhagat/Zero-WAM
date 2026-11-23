"""Small, atomic records; completed task failures are valid experimental results."""
import json
from pathlib import Path


def save_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def read_episode(directory: Path) -> dict:
    episodes = json.loads((directory / "config.json").read_text())["episodes"]
    if len(episodes) != 1 or episodes[0].get("status") != "completed":
        raise ValueError(f"Expected one completed episode in {directory}")
    episode = episodes[0]
    if not isinstance(episode.get("success"), bool):
        raise ValueError(f"Missing Boolean task outcome in {directory}")
    return episode


def validate_chunks(directory: Path, count: int) -> None:
    """Require a complete contiguous tensor sequence before reusing a phase."""
    expected = [f"chunk_{index:04d}.pt" for index in range(count)]
    actual = [path.name for path in sorted(directory.glob("chunk_*.pt"))]
    if count < 1 or actual != expected:
        raise ValueError(f"Incomplete tensor sequence in {directory}; expected {count} chunks")


def validate_source(source: dict, trial) -> None:
    if source["task"] != trial.task or source["simulator_seed"] != trial.scene_seed:
        raise ValueError("Source task/scene differs from the requested trial")
    if source["server"]["sampling_seed"] != trial.sampling_seed or source["icl_seed"] != trial.icl_seed:
        raise ValueError("Source sampling/demo seed differs from the requested trial")
    if source["rendering"]["samples_per_pixel_effective"] != trial.render_samples:
        raise ValueError("Source rendering differs from the requested trial")
    server = source["server"]
    if (not server["use_icl"] or server["use_guidance"] or
            not server.get("cached_caption_used") or server.get("target_text_source") != "empty"):
        raise ValueError("Source must use native ICL with cached caption, empty target text and no reward")


def validate_pair(source: dict, other: dict) -> None:
    """Reject successful-looking comparisons with different effective settings."""
    for key in ("task", "simulator_seed", "task_instruction", "stopping_policy"):
        if source.get(key) != other.get(key):
            raise ValueError(f"Paired episode mismatch: {key}")
    if source["rendering"] != other["rendering"]:
        raise ValueError("Paired episode mismatch: rendering")
    if "simulator_action_limit" in source and source["simulator_action_limit"] != other.get("simulator_action_limit"):
        raise ValueError("Paired episode mismatch: simulator_action_limit")
    keys = ("model_path", "video_state_dtype", "transformer_dtype", "video_inference_steps",
            "action_inference_steps", "frame_chunk_size", "sampling_seed",
            "compact_observation_cache", "phase_separation", "phase_offload_transformer")
    for key in keys:
        if key not in source["server"] or source["server"][key] != other["server"].get(key):
            raise ValueError(f"Paired server mismatch: {key}")
    if any(other["server"].get(key) is not False for key in ("use_text", "use_icl", "use_guidance")):
        raise ValueError("Control/replay must disable language, ICL and reward guidance")
    if other["server"].get("model_text") != "":
        raise ValueError("Control/replay must use empty model text")


def summarize(root: Path) -> dict:
    source = read_episode(root / "capture")
    control = read_episode(root / "control")
    replay = read_episode(root / "replay")
    validate_pair(source, control)
    validate_pair(source, replay)
    horizon = source["rollout_chunks"]
    result = dict(status="completed", task=source["task"], simulator_seed=source["simulator_seed"],
                  sampling_seed=source["server"]["sampling_seed"], source_success=source["success"],
                  control_success=control["success"], replay_success=replay["success"],
                  source_chunks=horizon, replay_use_text=False, replay_use_icl=False)
    for name, episode in (("control", control), ("replay", replay)):
        extended = episode["rollout_chunks"] > horizon
        result.update({f"{name}_chunks": episode["rollout_chunks"],
                       f"{name}_success_within_source_horizon": episode["success"] and not extended,
                       f"{name}_extended_beyond_saved_noise": extended,
                       f"{name}_fallback_chunks": episode.get("noise_fallback_chunks", 0),
                       f"{name}_first_fallback_chunk": episode.get("first_noise_fallback_chunk"),
                       f"{name}_actions": episode.get("simulator_action_count")})
    result["source_actions"] = source.get("simulator_action_count")
    result["reference"] = source.get("selected_reference")
    result["rescue"] = not control["success"] and replay["success"]
    result["regression"] = control["success"] and not replay["success"]
    result["inversion"] = json.loads((root / "inversion.json").read_text())
    return result
