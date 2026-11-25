"""Headless actual-task camera diagnostic; run in the robotwin environment."""
import importlib
import os
from pathlib import Path
import numpy as np
from PIL import Image
import yaml

import argparse
import faulthandler
import sys

parser = argparse.ArgumentParser(description="Capture real RoboTwin task cameras without loading Zero-WAM.")
parser.add_argument("--robotwin-root", type=Path, default=Path(os.environ.get("ROBOTWIN_ROOT", "~/Robotwin")).expanduser())
parser.add_argument("--output", type=Path, default=Path("results/camera-check"))
parser.add_argument("--task", choices=("move_stapler_pad", "place_object_scale", "place_empty_cup"), default="place_empty_cup")
parser.add_argument("--shader", choices=("rt", "default"), default="rt")
parser.add_argument("--denoiser", choices=("oidn", "none"), default="oidn")
parser.add_argument("--seed", type=int, default=100000)
parser.add_argument("--frames", type=int, default=3)
parser.add_argument("--timeout", type=int, default=480, help="Exit with a stack dump after this many seconds")
options = parser.parse_args()
if options.timeout < 1 or options.frames < 1:
    parser.error("--timeout must be positive")
output = options.output.expanduser().resolve()
output.mkdir(parents=True, exist_ok=True)
root = options.robotwin_root.expanduser().resolve()
sys.path.insert(0, str(root))
os.chdir(root)
faulthandler.enable()
faulthandler.dump_traceback_later(options.timeout, exit=True)
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from evaluation.robotwin.rendering import configure_rendering
configure_rendering()
from sapien import render
from evaluation.robotwin.camera_diagnostics import configure_diagnostic_denoiser, configure_diagnostic_shader
configure_diagnostic_denoiser(render, options.denoiser)
configure_diagnostic_shader(render, options.shader)
import json
(output / "diagnostic_settings.json").write_text(json.dumps(dict(
    task=options.task, seed=options.seed, frames=options.frames,
    shader=options.shader, denoiser=options.denoiser, samples_per_pixel=int(os.environ.get("RENDER_SAMPLES_PER_PIXEL", "32")),
    model_loaded=os.environ.get("CAMERA_LOAD_MODEL", "0") == "1",
), indent=2) + "\n")
from envs import CONFIGS_PATH
configs = Path(CONFIGS_PATH)
with open("task_config/demo_clean.yml") as f:
    args = yaml.safe_load(f)
with open(configs / "_embodiment_config.yml") as f:
    embodiments = yaml.safe_load(f)
with open(configs / "_camera_config.yml") as f:
    cameras = yaml.safe_load(f)

embodiment = args["embodiment"]
if len(embodiment) != 1:
    raise RuntimeError("This diagnostic expects the single-embodiment demo_clean config")
robot_file = embodiments[embodiment[0]]["file_path"]
with open(Path(robot_file) / "config.yml") as f:
    robot_config = yaml.safe_load(f)
head = cameras[args["camera"]["head_camera_type"]]
args.update(
    task_name=options.task, task_config="demo_clean",
    left_robot_file=robot_file, right_robot_file=robot_file,
    dual_arm_embodied=True,
    left_embodiment_config=robot_config, right_embodiment_config=robot_config,
    head_camera_h=head["h"], head_camera_w=head["w"],
    eval_mode=True, render_freq=0, save_data=False,
    save_root=str(output), policy_name="ACT", ckpt_setting=0,
)
# Suppress only recording, not cameras, physics, or planner initialization.
args["eval_video_log"] = False
module = importlib.import_module(f"envs.{options.task}")
task = getattr(module, options.task)()
print(f"Loading actual {options.task} scene, seed {options.seed}.", flush=True)
print("CuRobo may initialize here; Zero-WAM is not loaded.")
task.setup_demo(now_ep_num=0, seed=options.seed, is_test=True, **args)

(output / "scene_info.json").write_text(json.dumps(dict(
    info=getattr(task, "info", {}),
    static_cameras=[dict(index=i, name=getattr(camera, "name", None))
                    for i, camera in enumerate(task.cameras.static_camera_list)],
), indent=2, default=str) + "\n")
print("Task initialized. Capturing observations through task.get_obs().")
# Do not retry setup failures: preserve the original traceback and stop.
from evaluation.robotwin.camera_diagnostics import trace_cameras
os.environ["NOISE_INVERSION_STEP_TIMEOUT"] = "90"
trace_stream = (output / "camera_timing.jsonl").open("a", buffering=1)
trace_cameras(task.cameras, trace_stream)
for frame in range(options.frames):
    print(f"Capturing task observation {frame}")
    obs = task.get_obs()
    for name in ("head_camera", "left_camera", "right_camera"):
        rgb = np.asarray(obs["observation"][name]["rgb"])
        if rgb.ndim != 3 or not np.isfinite(rgb).all():
            raise RuntimeError(f"Invalid image for {name}: {rgb.shape}")
        if rgb.dtype != np.uint8:
            raise RuntimeError(f"Unexpected RGB dtype for {name}: {rgb.dtype}")
        if frame in (0, options.frames - 1):
            Image.fromarray(rgb[..., :3]).save(output / f"{name}_{frame}.png")
        print(f"Checked {name} frame={frame}; shape={rgb.shape}")
    task.scene.step()
print(f"PASS: captured {options.frames} observations; see camera_timing.jsonl.", flush=True)
trace_stream.close()
task.close_env()
faulthandler.cancel_dump_traceback_later()
