"""Matched-history teacher reversal and frozen-backbone adapter inference."""
import json
from pathlib import Path
import time

import torch

from .artifacts import ChunkStore
from .solver import integrate, relative_error


class OnlineSession:
    def __init__(self, server, request):
        self.server = server
        self.request = request
        self.settings = request["noise_session"]
        self.method = self.settings["method"]
        if self.method not in ("icl", "control", "oracle", "learned"):
            raise ValueError(f"Unknown noise method: {self.method}")
        if not server.job_config.video_state_fp32:
            raise ValueError("Online reversal requires FP32 video state")
        self.root = Path(self.settings["directory"])
        self.root.mkdir(parents=True, exist_ok=True)
        self.history = []
        self.initial_frame = None
        self.encoder_cache = None
        self.demo, cached_text = server._load_or_encode_icl(request["icl_video_path"], request["icl_latent_path"])
        if self.method in ("icl", "oracle") and cached_text is None:
            raise ValueError("ICL teacher requires cached text_emb in the human latent file")
        if self.settings.get("save_demo", True):
            torch.save(dict(demo=self.demo.cpu()), self.root / "demo.pt")
        server.online_demo = (request["icl_latent_path"], request["icl_video_path"], self.demo, cached_text)
        self.adapter = None
        if self.method == "learned":
            from .adapter import load_adapter
            self.adapter = load_adapter(self.settings["adapter"], server.device)
        self.build_context(self.method == "icl")

    def build_context(self, use_demo):
        """Rebuild caches from actual observations; never reuse teacher predictions."""
        server = self.server
        server._reset(prompt=self.request.get("prompt", ""), use_icl=use_demo,
                      use_text=use_demo, target_text_source="empty",
                      icl_text_source="cached" if use_demo else "target",
                      icl_video_path=self.request["icl_video_path"],
                      icl_latent_path=self.request["icl_latent_path"],
                      video_guidance_scale=1.0, icl_guidance_scale=5.0 if use_demo else 1.0)
        server.init_latent = self.initial_frame.to(server.device) if self.initial_frame is not None else None
        from ..modules.icl_model import OBSERVATION_CACHE_TYPE
        for index, history in enumerate(self.history):
            server.chunk_idx = index
            for mode, key in (("video", "video"), ("action", "actions")):
                payload = server._prepare_icl_stream(history[key].to(server.device), 0, mode,
                                                   OBSERVATION_CACHE_TYPE)
                server.transformer(payload, update_cache=1, cache_name=server.cache_name,
                                   mode="forward_latent_only" if mode == "video" else "forward_action_only",
                                   clean_window_cache=mode == "action")
        server.chunk_idx = len(self.history)
        # _reset replaces this list. Preserve the actual streaming encoder state,
        # which must not be reset by a teacher/student cache rebuild.
        if self.encoder_cache is not None:
            server.streaming_vae.feat_cache = self.encoder_cache

    def noise(self, seed):
        server = self.server
        generator = torch.Generator(device=server.device).manual_seed(seed)
        video = torch.randn(1, 48, server.job_config.frame_chunk_size,
                            server.latent_height, server.latent_width,
                            device=server.device, dtype=server.dtype, generator=generator).float()
        action = torch.randn(1, server.job_config.action_dim, server.job_config.frame_chunk_size,
                             server.action_per_frame, 1, device=server.device,
                             dtype=server.dtype, generator=generator)
        return video, action

    def context(self):
        if self.history:
            return (torch.cat([h["video"] for h in self.history], dim=2),
                    torch.cat([h["actions"] for h in self.history], dim=2))
        return self.initial_frame, torch.zeros(1, self.server.job_config.action_dim, 1,
                                              self.server.action_per_frame, 1)

    def reverse(self, target):
        server = self.server
        server.scheduler.set_timesteps(server.job_config.num_inference_steps)
        sigmas = torch.cat((server.scheduler.sigmas, server.scheduler.sigmas.new_zeros(1)))
        observed = server.init_latent if not self.history else None
        def velocity(value, sigma):
            return server._video_velocity(value, sigma * server.scheduler.num_train_timesteps)
        recovered = integrate(target.float(), sigmas.flip(0), velocity, observed)
        reconstruction = integrate(recovered, sigmas, velocity, observed)
        return recovered, relative_error(reconstruction, target, not self.history)

    def sample(self, observation):
        started = time.perf_counter()
        server = self.server
        index = len(self.history)
        if self.initial_frame is None:
            from ..modules.phase_separation import encode_with_phase_separation
            self.initial_frame = encode_with_phase_separation(
                server, server._encode_initial_obs, observation, "initial_observation").detach().cpu()
            self.encoder_cache = server.streaming_vae.feat_cache
            torch.save(dict(initial_frame=self.initial_frame), self.root / "initial.pt")
        seed = int(self.settings["seed"]) + index
        original, action_noise = self.noise(seed)
        error = None
        target = None
        recovered = original
        if self.method in ("oracle", "icl"):
            self.build_context(True)
            teacher_action = server._infer_icl(dict(observation, video_only=self.method == "oracle", initial_latent=self.initial_frame, video_noise=original, action_noise=action_noise))
            target = server.last_predicted_latents.detach().float()
            if not torch.isfinite(target).all():
                raise RuntimeError("Teacher produced a nonfinite video latent")
            if self.method == "oracle":
                self.build_context(False)
                recovered, error = self.reverse(target)
        if self.method != "icl":
            if self.method != "oracle":
                self.build_context(False)
            if self.method == "learned":
                history, actions = self.context()
                with torch.no_grad():
                    recovered = self.adapter(original, self.demo.to(server.device),
                                             history.to(server.device), actions.to(server.device), index)
            result = server._infer_icl(dict(observation, initial_latent=self.initial_frame, video_noise=recovered, action_noise=action_noise))
        else:
            result = teacher_action
        if self.method == "oracle":
            mask = torch.ones_like(original)
            if index == 0:
                mask[:, :, :1] = 0
            ChunkStore(self.root / "pairs").save(index, original=original, recovered=recovered,
                target=target, mask=mask, action_noise=action_noise)
        if not torch.isfinite(server.last_predicted_latents).all() or not torch.isfinite(server.last_predicted_actions).all():
            raise RuntimeError("Policy produced nonfinite latents/actions")
        self.encoder_cache = server.streaming_vae.feat_cache
        record = dict(chunk=index, seed=seed, method=self.method, reconstruction_error=error,
                      history_chunks=index, seconds=time.perf_counter() - started,
                      teacher_uses_text=self.method in ("icl", "oracle"),
                      teacher_caption_source="cached" if self.method in ("icl", "oracle") else None,
                      target_text_source="empty", student_uses_text=False,
                      solver_steps=server.job_config.num_inference_steps,
                      dtype=str(server.dtype), model_path=server.job_config.model_path,
                      student_uses_icl=self.method == "icl", fallback=False)
        with (self.root / "chunks.jsonl").open("a") as stream:
            stream.write(json.dumps(record) + "\n")
        return result

    def observe(self, observation):
        server = self.server
        server.streaming_vae.feat_cache = self.encoder_cache
        actions = server.last_predicted_actions.detach().cpu()
        server._compute_icl_kv_cache(observation)
        history = dict(video=server.last_observation_latents, actions=actions)
        ChunkStore(self.root / "history").save(len(self.history), **history)
        self.history.append(history)
        self.encoder_cache = server.streaming_vae.feat_cache
