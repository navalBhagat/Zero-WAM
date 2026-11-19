"""Expand CLI options into ordered run dictionaries and exported environments.

The dictionary shape is the public manifest contract. Seed order, CFG settings,
render repetition rotation and conditioning defaults are intentionally stable.
"""

from argparse import Namespace
from typing import NamedTuple
from experiments.paths import ROOT
from .config import MODES


class Variant(NamedTuple):
    """One condition before task/seed expansion; ordering is significant."""

    label: str
    mode: str
    spp: int
    selector: str
    decoder_dtype: str
    repeat: int


def specifications(options: Namespace) -> list[dict]:
    """Build planned runs without touching files, importing models or submitting jobs."""
    tasks = options.tasks or (
        ["place_object_scale"]
        if options.stage == "render"
        else ["place_object_scale", "place_bread_basket"]
    )
    seeds = options.seeds or ([0] if options.stage == "render" else [0, 1, 2])
    specs = []
    for task in tasks:
        for seed in seeds:
            variants = stage_variants(options)
            for label, mode, spp, selector, dtype, repeat in variants:
                if selector == "window" and options.lookahead_mode == "dino_distance":
                    label += "_dino_distance"
                icl, text, guide = MODES[mode]
                environment = dict(
                    ZERO_WAM_ROOT=str(ROOT),
                    TASK=task,
                    SEED=str(seed),
                    ICL_SEED=str(seed),
                    SAMPLING_SEED=str(options.sampling_seed),
                    TEST_NUM=str(1 if options.stage == "render" else options.episodes),
                    USE_ICL=str(icl),
                    USE_TEXT=str(int(bool(text or icl))),
                    USE_GUIDANCE=str(guide),
                    ICL_TEXT_SOURCE="cached" if icl else "target",
                    TARGET_TEXT_SOURCE="instruction" if text else "empty",
                    COMPACT_OBSERVATION_CACHE=str(int(options.compact_cache)),
                    ICL_GUIDANCE_SCALE="5",
                    TEXT_GUIDANCE_SCALE="5" if mode == "text" else "1",
                    RENDER_SAMPLES_PER_PIXEL=str(spp),
                    PROFILE_TIMING="1",
                    PROFILE_SYNC_GPU="0",
                    PHASE_SEPARATION=str(
                        int(getattr(options, "phase_separation", False))
                    ),
                    PHASE_OFFLOAD_TRANSFORMER="1",
                    STAGE_REQUIRE_SUCCESS="0",
                    MONITOR_GPU="1",
                    TEXT_ENCODER_OFFLOAD="1",
                    VIDEO_STATE_FP32="1",
                    GUIDANCE_DECODER_DTYPE=dtype,
                    GUIDANCE_SELECTION=selector,
                    GUIDANCE_MATCH_WINDOW=str(options.window),
                    GUIDANCE_LOOKAHEAD=str(options.lookahead),
                    GUIDANCE_REFERENCE_FRAMES=str(options.fixed_frame),
                    GUIDANCE_STRENGTH=str(options.strength),
                    GUIDANCE_EVERY=str(options.guidance_every),
                    GUIDANCE_MAX_NORM="1",
                    GUIDANCE_REFERENCE_SOURCE=options.reference_source,
                    CLIENT_TIMEOUT=str(options.client_timeout),
                    EXPERIMENT_LABEL=label,
                    EXPERIMENT_STAGE=options.stage,
                    EXPERIMENT_REPEAT=str(repeat),
                    EXPERIMENT_OUTPUT_ROOT=str(
                        options.manifest.resolve().parent / "runs"
                    ),
                )
                environment.update(
                    COUNT_POLICY_FLOPS=str(int(not options.no_flop_count)),
                    RUN_DIAGNOSTICS=str(int(options.diagnostics)),
                    SAVE_GUIDANCE_CAPTURE=str(int(options.capture_guidance)),
                )
                if options.lookahead_mode != "frames":
                    environment.update(
                        GUIDANCE_LOOKAHEAD_MODE=options.lookahead_mode,
                        GUIDANCE_DISTANCE_INCREMENT=str(options.distance_increment)
                        if options.distance_increment is not None
                        else "",
                        GUIDANCE_DISTANCE_SMOOTHING=str(options.distance_smoothing),
                        GUIDANCE_MAX_LOOKAHEAD_FRAMES=str(options.max_lookahead_frames),
                    )
                specs.append(
                    dict(
                        task=task,
                        seed=seed,
                        label=label,
                        mode=mode,
                        repeat=repeat,
                        environment=environment,
                        status="planned",
                    )
                )
    return specs


def stage_variants(options: Namespace) -> list[Variant]:
    """List conditions in their existing execution order, including render rotation."""
    if options.stage == "render":
        # Rotate submission order; paired settings when multiple repetitions are requested.
        variants = [
            (f"spp{s}_r{r}", "icl", s, "fixed", "float32", r)
            for r in range(options.repeats)
            for s in ((32, 16, 8)[r % 3 :] + (32, 16, 8)[: r % 3])
        ]
    elif options.stage == "baselines":
        variants = [(m, m, options.spp, "fixed", "float32", 0) for m in ("text", "icl")]
    elif options.stage == "guidance":
        variants = [
            (f"{m}_{sel}", m, options.spp, sel, "float32", 0)
            for m in (options.guidance_modes or ["guidance", "icl_guidance"])
            for sel in (options.guidance_selectors or ["fixed", "window"])
        ]
    else:
        variants = [
            (f"{m}_{dtype}", m, options.spp, "window", dtype, 0)
            for m in ("guidance", "icl_guidance")
            for dtype in ("float32", "float16")
        ]
    return [Variant(*values) for values in variants]
