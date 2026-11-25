"""CPU checks for matched histories and streaming encoder state isolation."""
import json
import sys
from types import SimpleNamespace

import torch

from wan_va.noise_inversion.online import OnlineSession


class Transformer:
    def __init__(self):
        self.cache = []

    def __call__(self, payload, **kwargs):
        self.cache.append(payload)


class Server:
    def __init__(self):
        self.device = "cpu"
        self.dtype = torch.float32
        self.job_config = SimpleNamespace(video_state_fp32=True, frame_chunk_size=2,
            action_dim=30, num_inference_steps=2, model_path="fake")
        self.streaming_vae = SimpleNamespace(feat_cache=[])
        self.scheduler = SimpleNamespace(sigmas=torch.tensor([1., .5]), num_train_timesteps=1000,
                                         set_timesteps=lambda _: None)
        self.transformer = Transformer()
        self.cache_name = "test"
        self.calls = []

    def _load_or_encode_icl(self, *args):
        return torch.zeros(1, 48, 2, 4, 4), torch.ones(1, 2, 8)

    def _reset(self, **kwargs):
        self.use_icl = kwargs["use_icl"]
        assert kwargs["use_text"] == kwargs["use_icl"]
        assert kwargs["icl_text_source"] == ("cached" if kwargs["use_icl"] else "target")
        assert kwargs["target_text_source"] == "empty"
        self.chunk_idx = 0
        self.init_latent = None
        self.streaming_vae.feat_cache = []
        self.transformer.cache = []
        self.latent_height = self.latent_width = 4
        self.action_per_frame = 16

    def _prepare_icl_stream(self, data, time, mode, kind):
        return mode, data.clone()

    def _infer_icl(self, observation):
        self.init_latent = observation["initial_latent"]
        video = observation["video_noise"].clone() - (3 if self.use_icl else 1)
        if self.chunk_idx == 0:
            video[:, :, :1] = self.init_latent
        self.last_predicted_latents = video
        self.last_predicted_actions = observation["action_noise"].clone()
        self.calls.append((self.use_icl, self.chunk_idx, len(self.transformer.cache)))
        return video

    def _video_velocity(self, value, timestep):
        return torch.ones_like(value)

    def _compute_icl_kv_cache(self, observation):
        self.last_observation_latents = observation["video"]
        self.chunk_idx += 1
        self.streaming_vae.feat_cache.append("actual")


def test_online_reversal_uses_same_actual_history_without_demo_leakage(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "wan_va.modules.icl_model", SimpleNamespace(OBSERVATION_CACHE_TYPE=0))
    server = Server()
    session = OnlineSession(server, dict(prompt="", icl_video_path="", icl_latent_path="demo",
        noise_session=dict(method="oracle", directory=str(tmp_path), seed=1234)))
    session.initial_frame = torch.zeros(1, 48, 1, 4, 4)
    session.encoder_cache = ["initial"]
    session.sample({})
    session.observe(dict(video=torch.ones(1, 48, 2, 4, 4)))
    session.sample({})
    assert server.calls == [(True, 0, 0), (False, 0, 0), (True, 1, 2), (False, 1, 2)]
    assert session.encoder_cache == ["initial", "actual"]
    rows = [json.loads(line) for line in (tmp_path / "chunks.jsonl").read_text().splitlines()]
    assert all(row["reconstruction_error"] < 1e-6 for row in rows)
    assert all(row["teacher_uses_text"] and row["teacher_caption_source"] == "cached" for row in rows)
    assert all(not row["student_uses_text"] for row in rows)
    first = torch.load(tmp_path / "pairs/chunk_0000.pt", weights_only=True)
    assert first["mask"][:, :, :1].sum() == 0
    assert first["mask"][:, :, 1:].sum() > 0


def test_episode_noise_matches_original_draw_order(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "wan_va.modules.icl_model", SimpleNamespace(OBSERVATION_CACHE_TYPE=0))
    session = OnlineSession(Server(), dict(prompt="", icl_video_path="", icl_latent_path="demo",
        noise_session=dict(method="control", directory=str(tmp_path), seed=1234)))
    expected = torch.Generator(device="cpu").manual_seed(1234)
    for _ in range(3):
        video = torch.randn(1, 48, 2, 4, 4, generator=expected)
        action = torch.randn(1, 30, 2, 16, 1, generator=expected)
        torch.randn(action.shape, generator=expected)
        # Cache rebuilds and extra inference passes must not advance this stream.
        torch.randn(1000)
        session.build_context(False)
        actual_video, actual_action = session.noise()
        assert torch.equal(actual_video, video)
        assert torch.equal(actual_action, action)


def test_demo_cache_key_includes_video_path_when_latent_path_is_empty():
    import ast
    import os
    from pathlib import Path
    source = (Path(__file__).resolve().parents[1] / "wan_va/wan_va_server.py").read_text()
    server = next(node for node in ast.parse(source).body if isinstance(node, ast.ClassDef) and node.name == "VA_Server")
    method = next(node for node in server.body if isinstance(node, ast.FunctionDef) and node.name == "_load_or_encode_icl")
    scope = dict(os=os, torch=torch)
    exec(compile(ast.Module(body=[method], type_ignores=[]), "<demo-cache>", "exec"), scope)
    latent = torch.zeros(1)
    caption = torch.ones(1)
    mock = SimpleNamespace(online_demo=("", "/missing/a.mp4", latent, caption))
    assert scope["_load_or_encode_icl"](mock, "/missing/a.mp4", "") == (latent, caption)
    import pytest
    with pytest.raises(FileNotFoundError):
        scope["_load_or_encode_icl"](mock, "/missing/b.mp4", "")


def test_teacher_fails_explicitly_if_cached_caption_is_missing(tmp_path):
    import pytest
    server = Server()
    server._load_or_encode_icl = lambda *args: (torch.zeros(1, 48, 2, 4, 4), None)
    with pytest.raises(ValueError, match="requires cached text_emb"):
        OnlineSession(server, dict(prompt="", icl_video_path="", icl_latent_path="demo",
            noise_session=dict(method="oracle", directory=str(tmp_path), seed=1234)))
