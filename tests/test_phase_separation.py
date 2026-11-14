"""Lifecycle and exception safety without loading WAM; optional real CUDA transfer."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import pytest
import torch

spec=importlib.util.spec_from_file_location('phase_module',Path(__file__).resolve().parents[1]/'wan_va/modules/phase_separation.py')
phase=importlib.util.module_from_spec(spec);spec.loader.exec_module(phase)


class Model(torch.nn.Module):
    def __init__(self,device='cpu'):
        super().__init__()
        self.linear=torch.nn.Linear(4,4,dtype=torch.bfloat16,device=device)
        self.register_buffer('freq',torch.ones(4,dtype=torch.bfloat16,device=device))
        self.blocks=[SimpleNamespace(attn1=SimpleNamespace(attn_caches={
            'pos':dict(k=torch.ones(1,4,1,1,device=device),v=torch.zeros(1,4,1,1,device=device))}))]
        self.frame_ids_cache=torch.arange(4,device=device)
        self.moves=[]
    def to(self,*args,**kwargs):
        self.moves.append(str(kwargs.get('device')))
        return super().to(*args,**kwargs)


def server(tmp_path,device='cpu'):
    return SimpleNamespace(transformer=Model(device),device=torch.device(device),
        save_root=str(tmp_path),chunk_idx=3,phase_separation=True,phase_offload_transformer=True)


def mock_cuda(monkeypatch):
    for name in ('synchronize','empty_cache','reset_peak_memory_stats'):
        monkeypatch.setattr(torch.cuda,name,lambda *a:None)
    monkeypatch.setattr(phase,'memory',lambda *a:dict(allocated_bytes=10,reserved_bytes=20,free_bytes=30,total_bytes=50))
    monkeypatch.setattr(torch.cuda,'max_memory_allocated',lambda *a:15)


def test_restore_and_cache_identity_and_dtype(tmp_path,monkeypatch):
    mock_cuda(monkeypatch);s=server(tmp_path)
    original=phase.cache_state(s.transformer)[0]
    value=phase.encode_with_phase_separation(s,lambda obs:obs+1,2,'observation_update')
    assert value==3 and s.transformer.moves==['cpu','cpu']
    assert phase.cache_state(s.transformer)[0]==original
    assert s.transformer.freq.dtype==s.transformer.linear.weight.dtype==torch.bfloat16
    row=json.loads((tmp_path/'phase_separation.jsonl').read_text())
    assert row['status']=='encoded' and row['kv_and_metadata_bytes']>0


def test_encoding_exception_still_restores_and_logs(tmp_path,monkeypatch):
    mock_cuda(monkeypatch);s=server(tmp_path)
    def fail(_):raise RuntimeError('encoding failed')
    with pytest.raises(RuntimeError,match='encoding failed'):
        phase.encode_with_phase_separation(s,fail,None,'observation_update')
    assert s.transformer.moves==['cpu','cpu']
    assert json.loads((tmp_path/'phase_separation.jsonl').read_text())['status']=='failed'


def test_disabled_is_original_call_and_control_never_moves(tmp_path,monkeypatch):
    mock_cuda(monkeypatch);s=server(tmp_path);s.phase_separation=False
    assert phase.encode_with_phase_separation(s,lambda x:x,4,'obs')==4
    assert not s.transformer.moves and not (tmp_path/'phase_separation.jsonl').exists()
    s.phase_separation=True;s.phase_offload_transformer=False
    assert phase.encode_with_phase_separation(s,lambda x:x,4,'obs')==4
    assert not s.transformer.moves


@pytest.mark.skipif(not torch.cuda.is_available(),reason='Needs CUDA')
def test_real_cuda_transfer_preserves_kv_and_values(tmp_path):
    s=server(tmp_path,'cuda:0');original=phase.cache_state(s.transformer)[0]
    weights=s.transformer.linear.weight.detach().cpu().clone()
    def encode(_):
        assert s.transformer.linear.weight.device.type=='cpu'
        assert s.transformer.blocks[0].attn1.attn_caches['pos']['k'].device.type=='cuda'
        return torch.ones(16,device=s.device)
    phase.encode_with_phase_separation(s,encode,None,'observation_update')
    assert s.transformer.linear.weight.device.type=='cuda'
    torch.testing.assert_close(s.transformer.linear.weight.cpu(),weights)
    assert phase.cache_state(s.transformer)[0]==original


def test_action_hash_supports_actual_numpy_return_and_tensor_mocks():
    import numpy as np
    array=np.arange(12,dtype=np.float32).reshape(3,4)
    assert phase.action_digest(array)==phase.action_digest(torch.from_numpy(array))
    assert phase.action_digest(array[:,::2])==phase.action_digest(array[:,::2].copy())
    changed=array.copy();changed[0,0]=1
    assert phase.action_digest(changed)!=phase.action_digest(array)
