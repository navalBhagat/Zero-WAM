"""Operator arithmetic tests without importing WAM/transformer dependencies."""
import importlib.util
from pathlib import Path
import sys
import types
from types import SimpleNamespace

import pytest
import torch

PACKAGE='_flop_test'
pkg=types.ModuleType(PACKAGE);pkg.__path__=[str(Path(__file__).resolve().parents[1]/'wan_va/guidance')]
sys.modules[PACKAGE]=pkg
spec=importlib.util.spec_from_file_location(PACKAGE+'.flops',Path(pkg.__path__[0])/'flops.py')
flops=importlib.util.module_from_spec(spec);sys.modules[spec.name]=flops;spec.loader.exec_module(flops)


def test_frozen_linear_forward_backward_count_matches_actual_paths():
    weight=torch.randn(4,3,requires_grad=False)
    x=torch.randn(2,3,requires_grad=True)
    counter=flops.ArithmeticCounter()
    with counter:
        y=x@weight.T
        torch.autograd.grad(y.sum(),x)
    ops=counter.operations
    # Forward: [2,3]@[3,4]; backward input: [2,4]@[4,3]. No weight gradient.
    assert ops['aten.mm']==2*(2*3*4)*2
    assert not counter.unknown


def test_conv3d_frozen_backward_is_counted():
    x=torch.randn(1,2,3,4,4,requires_grad=True)
    weight=torch.randn(3,2,1,1,1)
    counter=flops.ArithmeticCounter()
    with counter:
        y=torch.nn.functional.conv3d(x,weight)
        torch.autograd.grad(y.sum(),x)
    assert counter.operations['aten.convolution']>0
    assert counter.operations['aten.convolution_backward']>0


def test_counter_preserves_result_and_rng():
    def run():
        x=torch.randn(2,3,requires_grad=True)
        y=(x@torch.ones(3,4)).square().sum()
        return torch.autograd.grad(y,x)[0]
    torch.manual_seed(5);expected=run()
    torch.manual_seed(5)
    with flops.ArithmeticCounter():actual=run()
    assert torch.equal(actual,expected)


def test_unknown_float_operation_is_explicit():
    with flops.ArithmeticCounter() as counter:
        torch.linalg.svd(torch.eye(3))
    assert counter.unknown
    assert counter.summary()['status']=='partial_unsupported_ops'


def test_request_counts_reset_and_backward_and_disabled_mode(tmp_path,monkeypatch):
    monkeypatch.setenv("COUNT_POLICY_FLOPS", "1")
    class Server:
        save_root=str(tmp_path)
        workload_episode=0
        chunk_idx=0
        @flops.profile_policy_call
        def infer(self,obs):
            if obs.get('reset'):
                with flops.component('text_encoder'):
                    _=torch.ones(2,3)@torch.ones(3,4)
            else:
                x=torch.ones(2,3,requires_grad=True)
                with flops.component('reward_vae_decode'):
                    y=x@torch.ones(3,4)
                with flops.component('guidance_backward'):
                    torch.autograd.grad(y.sum(),x)
            return {}
    s=Server();reset=s.infer({'reset':True})['compute'];total=s.infer({})['compute']
    assert reset['components_flops']['text_encoder']==48
    assert total['components_flops']['reward_vae_decode']==48
    assert total['components_flops']['guidance_backward']>=48
    assert total['counted_flops']>reset['counted_flops']
    assert len((tmp_path/'diagnostics/flops.jsonl').read_text().splitlines())==2
    monkeypatch.setenv('COUNT_POLICY_FLOPS','0')
    assert 'compute' not in s.infer({})


def test_compiled_attention_counts_dense_once():
    counter=flops.ArithmeticCounter();token=flops._ACTIVE.set(counter)
    q=torch.ones(1,2,3,4);k=torch.ones(1,2,5,4);v=torch.ones(1,2,5,6)
    try:
        with counter,flops.compiled_attention(q,k,v):
            result=(q@k.transpose(-1,-2))@v
    finally:flops._ACTIVE.reset(token)
    assert result.shape==(1,2,3,6)
    assert counter.summary()['counted_flops']==2*1*2*3*5*(4+6)
    assert counter.dense_attention and not counter.unknown


def test_operator_counting_defaults_off(monkeypatch):
    monkeypatch.delenv("COUNT_POLICY_FLOPS", raising=False)
    class Server:
        @flops.profile_policy_call
        def infer(self, obs):
            return {"ok": True}
    assert Server().infer({}) == {"ok": True}
