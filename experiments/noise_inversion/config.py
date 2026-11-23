"""Explicit settings and stable identities for paired noise-inversion trials."""
from dataclasses import asdict, dataclass
from pathlib import Path

TASKS = (
    "place_bread_basket", "place_object_scale", "open_microwave", "place_empty_cup",
    "stamp_seal", "move_stapler_pad", "stack_blocks_three",
)
PILOT_TASKS = TASKS[:2]


@dataclass(frozen=True)
class Trial:
    task: str
    scene_seed: int
    sampling_seed: int
    icl_seed: int = 0
    render_samples: int = 16

    def __post_init__(self):
        if self.task not in TASKS:
            raise ValueError(f"Unknown task: {self.task}")
        if min(self.scene_seed, self.sampling_seed, self.icl_seed) < 0 or self.render_samples < 1:
            raise ValueError("Seeds must be nonnegative and render samples positive")

    @property
    def name(self) -> str:
        return f"scene_{self.scene_seed}_noise_{self.sampling_seed}"

    def directory(self, campaign: Path) -> Path:
        return campaign / "trials" / self.task / self.name

    def to_dict(self) -> dict:
        return asdict(self)
