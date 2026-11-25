"""Explicit campaign sizes and reproducible sampling settings."""
from dataclasses import asdict, dataclass

UNSEEN_TASKS = ("place_bread_basket", "place_object_scale", "open_microwave", "place_empty_cup",
                "stamp_seal", "move_stapler_pad", "stack_blocks_three")
STAGES = {
    1: "smoke", 2: "signal", 3: "benchmark", 4: "collect", 5: "train_evaluate",
}


@dataclass(frozen=True)
class Settings:
    stage: int
    cluster: str = "das6"
    scene_count: int = 1
    sampling_seed: int = 1234
    scene_start: int = 100000
    hours: float = 3.0
    pairs_per_job: int = 25
    points_per_trajectory: int = 10
    seeds_per_point: int = 2
    train_steps: int = 5000
    effective_batch: int = 64
    architecture: str = "vit"
    width: int = 512
    layers: int = 8
    heads: int = 8
    overfit_pairs: int = 0

    def validate(self):
        if self.stage not in STAGES or self.cluster not in ("das6", "snellius"):
            raise ValueError("Invalid stage or cluster")
        if min(self.scene_count, self.hours, self.pairs_per_job, self.train_steps,
               self.effective_batch, self.points_per_trajectory, self.seeds_per_point) <= 0:
            raise ValueError("Counts, steps and walltime must be positive")
        if min(self.sampling_seed, self.scene_start) < 0:
            raise ValueError("Seeds must be nonnegative")
        if min(self.width, self.layers, self.heads) < 1 or self.architecture not in ("vit", "mlp") or self.width % self.heads:
            raise ValueError("Invalid adapter architecture")
        if self.overfit_pairs < 0:
            raise ValueError("overfit_pairs cannot be negative")
        if self.hours * 60 != int(self.hours * 60):
            raise ValueError("Walltime must be a whole number of minutes")

    def to_dict(self):
        return asdict(self)
