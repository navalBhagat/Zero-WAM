"""Legacy-caption diagnostic stays opt-in and cannot bypass the text switch."""
import importlib.util
from pathlib import Path
import pytest

spec=importlib.util.spec_from_file_location('legacy_conditioning',
    Path(__file__).resolve().parents[1]/'wan_va/conditioning.py')
conditioning=importlib.util.module_from_spec(spec); spec.loader.exec_module(conditioning)


def test_default_ignores_caption_and_legacy_explicitly_uses_it():
    target,cached=object(),object()
    assert conditioning.select_icl_text('target',False,target,cached) is target
    assert conditioning.select_icl_text('cached',True,target,cached) is cached


def test_missing_caption_and_disabled_text_fail_instead_of_silent_fallback():
    with pytest.raises(ValueError,match='USE_TEXT'):
        conditioning.select_icl_text('cached',False,object(),object())
    with pytest.raises(ValueError,match='text_emb'):
        conditioning.select_icl_text('cached',True,object(),None)


def test_actual_cache_builder_receives_cached_embedding(monkeypatch):
    import ast
    import logging
    import sys
    import types
    from types import SimpleNamespace
    import torch
    root=Path(__file__).resolve().parents[1]
    package=types.ModuleType('_legacy_policy_test'); package.__path__=[str(root/'wan_va')]
    monkeypatch.setitem(sys.modules,package.__name__,package)
    cls=next(n for n in ast.parse((root/'wan_va/wan_va_server.py').read_text()).body
             if isinstance(n,ast.ClassDef) and n.name=='VA_Server')
    method=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='_cache_icl_context')
    scope=dict(torch=torch,__package__=package.__name__,logger=logging.getLogger('test'),
               ICL_CACHE_TYPE=2,get_mesh_id=lambda *args:torch.zeros(4,2,dtype=torch.int64))
    exec(compile(ast.Module(body=[method],type_ignores=[]),'<cache>','exec'),scope)
    payloads=[]
    class Transformer:
        patch_size=(1,1,1)
        def __call__(self,payload,**kwargs): payloads.append(payload)
        def cache_counts(self): return {}
    cached=torch.full((1,2,8),77.)
    mock=SimpleNamespace(icl_text_source='cached',use_text=True,
        _load_or_encode_icl=lambda *a:(torch.ones(1,2,2,1,1),cached),
        vae=SimpleNamespace(config=SimpleNamespace(latents_mean=[0.,0.])),
        target_prompt_embeds=torch.zeros_like(cached),transformer=Transformer(),
        dtype=torch.float32,device='cpu',job_config=SimpleNamespace(icl_rope_h=24),cache_name='test')
    scope['_cache_icl_context'](mock,'video','latent')
    assert mock.cached_caption_used
    torch.testing.assert_close(payloads[0]['text_emb'],cached)


def test_original_target_empty_and_caption_are_independent():
    target,empty,cached=object(),object(),object()
    assert conditioning.select_target_text(True,'empty',target,empty) is empty
    assert conditioning.select_icl_text('cached',True,target,cached) is cached
    assert conditioning.select_target_text(True,'instruction',target,empty) is target
