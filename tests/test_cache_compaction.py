"""Exercise the actual cache pruning code on CPU without loading checkpoints."""
import ast
from pathlib import Path
from types import SimpleNamespace
from typing import Optional
import torch


def model(compact):
    source=Path(__file__).resolve().parents[1]/'wan_va/modules/icl_model.py'
    tree=ast.parse(source.read_text());scope=dict(torch=torch,Optional=Optional,
        ICL_CACHE_TYPE=2,OBSERVATION_CACHE_TYPE=0,PREDICTION_CACHE_TYPE=1)
    methods={}
    for cls in tree.body:
        if isinstance(cls,ast.ClassDef):
            for node in cls.body:
                if isinstance(node,ast.FunctionDef) and node.name in ('prune_cache','_prune_cache','clear_prediction_cache'):
                    exec(compile(ast.Module(body=[node],type_ignores=[]),str(source),'exec'),scope)
                    methods[node.name]=scope[node.name]
    class Attention: prune_cache=methods['prune_cache']
    class Model:
        _prune_cache=methods['_prune_cache']
        clear_prediction_cache=methods['clear_prediction_cache']
    m=Model();m.attn_window=64;m.compact_observation_cache=compact
    for name in ('type_ids_cache','seq_ids_cache','frame_ids_cache','cache_type_ids_cache'):
        setattr(m,name,torch.empty(0,dtype=torch.int))
    a=Attention();a.attn_caches={'pos':dict(k=torch.empty(1,0,1,1),v=torch.empty(1,0,1,1))}
    m.blocks=[SimpleNamespace(attn1=a)]
    return m


def append(m,valid,frame,kind,modality=0):
    pad=128-valid%128
    for name,data in [('type_ids_cache',[modality]*valid+[-1]*pad),
                      ('seq_ids_cache',[0]*valid+[-1]*pad),
                      ('frame_ids_cache',[frame]*valid+[-1]*pad),
                      ('cache_type_ids_cache',[kind]*(valid+pad))]:
        setattr(m,name,torch.cat([getattr(m,name),torch.tensor(data,dtype=torch.int)]))
    for name in ('k','v'):
        cache=m.blocks[0].attn1.attn_caches['pos']
        cache[name]=torch.cat([cache[name],torch.full((1,valid+pad,1,1),float(frame))],dim=1)


def test_compaction_preserves_valid_kv_and_plateaus_without_padding():
    regular,compact=model(False),model(True)
    for m in (regular,compact):append(m,2660,0,2)
    lengths=[]
    for chunk in range(45):
        for m in (regular,compact):
            append(m,756,2*chunk,1);m.clear_prediction_cache(-1)
            append(m,756,2*chunk,0);append(m,32,2*chunk+1,0,1)
            m.clear_prediction_cache()
        valid0=regular.frame_ids_cache>=0;valid1=compact.frame_ids_cache>=0
        for name in ('type_ids_cache','seq_ids_cache','frame_ids_cache','cache_type_ids_cache'):
            assert torch.equal(getattr(regular,name)[valid0],getattr(compact,name)[valid1])
        for name in ('k','v'):
            a=regular.blocks[0].attn1.attn_caches['pos'][name]
            b=compact.blocks[0].attn1.attn_caches['pos'][name]
            torch.testing.assert_close(a[:,valid0],b[:,valid1],rtol=0,atol=0)
            assert b.untyped_storage().nbytes()==b.numel()*b.element_size()
        assert not (((compact.cache_type_ids_cache==0)&(compact.frame_ids_cache==-1)).any())
        assert (compact.cache_type_ids_cache==2).sum()==2688  # Human prefix including its padding unchanged.
        lengths.append(compact.frame_ids_cache.numel())
    assert len(set(lengths[31:]))==1
    assert lengths[-1]==27904
