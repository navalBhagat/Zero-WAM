"""Condition isolation checks without importing heavyweight inference dependencies."""
import ast
import importlib.util
import itertools
import logging
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('conditioning', ROOT / 'wan_va/conditioning.py')
conditioning = importlib.util.module_from_spec(spec)
spec.loader.exec_module(conditioning)


def test_switches_are_independent_of_parameters_and_legacy_flags():
    for icl, text, guidance in itertools.product((0, 1), repeat=3):
        env = dict(USE_ICL=str(icl), USE_TEXT=str(text), USE_GUIDANCE=str(guidance),
                   GUIDANCE_STRENGTH='0.1', TARGET_TEXT_CFG='5', EXTERNAL_STRENGTH='1')
        with pytest.warns(FutureWarning, match='Ignoring legacy settings'):
            config = conditioning.read_conditioning(env)
        assert (config['use_icl'], config['use_text'], config['use_guidance']) == (icl, text, guidance)
    defaults = conditioning.read_conditioning({'GUIDANCE_STRENGTH': '1'})
    assert defaults['use_guidance'] is False
    assert defaults['use_text'] is False
    with pytest.raises(ValueError):
        conditioning.read_conditioning({'USE_TEXT': '-1'})


@pytest.mark.parametrize('use_text', [False, True])
def test_cached_caption_never_leaks_into_icl_context(use_text):
    # Exercise the actual cache-building method, with only the transformer mocked.
    server = next(n for n in ast.parse((ROOT / 'wan_va/wan_va_server.py').read_text()).body
                  if isinstance(n, ast.ClassDef) and n.name == 'VA_Server')
    method = next(n for n in server.body if isinstance(n, ast.FunctionDef) and n.name == '_cache_icl_context')
    scope = dict(torch=torch, ICL_CACHE_TYPE=2, logger=logging.getLogger('test'),
                 get_mesh_id=lambda *args: torch.zeros(4, 2, dtype=torch.int64))
    exec(compile(ast.Module(body=[method], type_ignores=[]), '<cache-method>', 'exec'), scope)
    empty, prompt, stored_caption = (torch.full((1, 2, 8), value) for value in (0., 13., 77.))
    captured = []
    class Transformer:
        patch_size = (1, 1, 1)
        def __call__(self, payload, **kwargs):
            captured.append(payload)
        def cache_counts(self):
            return {}
    mock = SimpleNamespace(
        _load_or_encode_icl=lambda *args: (torch.ones(1, 2, 2, 1, 1), stored_caption),
        vae=SimpleNamespace(config=SimpleNamespace(latents_mean=[0., 0.])),
        target_prompt_embeds=conditioning.select_text(use_text, prompt, empty),
        transformer=Transformer(), dtype=torch.float32, device=torch.device('cpu'),
        job_config=SimpleNamespace(icl_rope_h=24), cache_name='test',
    )
    scope['_cache_icl_context'](mock, 'video', 'latent')
    assert captured[0]['text_emb'] is mock.target_prompt_embeds
    assert not torch.equal(captured[0]['text_emb'], stored_caption)
    torch.testing.assert_close(captured[0]['text_emb'], prompt if use_text else empty)


def test_unsupported_double_cfg_is_rejected_instead_of_silently_ignored():
    with pytest.raises(ValueError, match='both exceed'):
        conditioning.read_conditioning(dict(USE_ICL='1', USE_TEXT='1', TEXT_GUIDANCE_SCALE='5'))


@pytest.mark.parametrize('use_text', [False, True])
def test_reset_text_off_never_encodes_episode_instruction(use_text, monkeypatch):
    import sys
    import types
    package = types.ModuleType('_policy_test')
    package.__path__ = [str(ROOT / 'wan_va')]
    monkeypatch.setitem(sys.modules, package.__name__, package)
    server = next(n for n in ast.parse((ROOT / 'wan_va/wan_va_server.py').read_text()).body
                  if isinstance(n, ast.ClassDef) and n.name == 'VA_Server')
    method = next(n for n in server.body if isinstance(n, ast.FunctionDef) and n.name == '_reset_icl')
    scope = dict(torch=torch, logger=logging.getLogger('test'), __package__='_policy_test')
    exec(compile(ast.Module(body=[method], type_ignores=[]), '<reset-method>', 'exec'), scope)
    empty, prompt = torch.zeros(1, 2, 8), torch.ones(1, 2, 8)
    encoded = []
    def encode_prompt(**kwargs):
        encoded.append(kwargs['prompt'])
        return prompt, empty
    mock = SimpleNamespace(
        job_config=SimpleNamespace(use_text=use_text, use_guidance=False,
            action_per_frame=16, height=224, width=288, obs_cam_keys=['a','b','c'],
            action_dim=4, used_action_channel_ids=[0],
            norm_stat={'q01':[0]*4, 'q99':[1]*4}, action_norm_method='quantiles'),
        env_type='franka', dtype=torch.float32, device=torch.device('cpu'),
        transformer=SimpleNamespace(clear_cache=lambda *_: None), cache_name='test',
        streaming_vae=SimpleNamespace(clear_cache=lambda: None), streaming_vae_half=None,
        empty_text_emb=empty, encode_prompt=encode_prompt,
        text_encoder=torch.nn.Linear(1,1),
    )
    cached=[]
    mock._cache_icl_context=lambda *args: cached.append(mock.target_prompt_embeds.clone())
    monkeypatch.setattr(torch.cuda, 'memory_allocated', lambda *_: 0)
    monkeypatch.setattr(torch.cuda, 'memory_reserved', lambda *_: 0)
    scope['_reset_icl'](mock, 'private episode instruction', True, '', '', 1, 5,
                        use_text=use_text, use_guidance=False)
    assert encoded == (['private episode instruction'] if use_text else [])
    torch.testing.assert_close(cached[0], prompt if use_text else empty)
    assert mock.external_strength == 0
