"""Per-request theoretical arithmetic accounting, including real input backward.

Two operations per multiply-add. Native PyTorch formulas count matmul/conv/SDPA
and backward; scalar formulas cover common elementwise/reduction operations.
Compiled FlexAttention is counted as a dense-equivalent bound outside dispatch,
without changing its kernel or mask. Unknown floating-point ops are reported.
These are mathematical estimates, NOT hardware counters or elapsed-time proxies.
"""
from collections import Counter, defaultdict
from contextlib import contextmanager
from contextvars import ContextVar
from functools import wraps
import json
import os
from pathlib import Path

import torch
from torch.utils._python_dispatch import TorchDispatchMode, _disable_current_modes
from torch.utils.flop_counter import flop_registry
from torch.utils._pytree import tree_flatten

_ACTIVE = ContextVar('policy_flops', default=None)
_LABEL = ContextVar('policy_flops_component', default='other_policy')

# Shape/data-movement operations contain no floating-point arithmetic. Integer
# mask/cache construction is also excluded by checking tensor dtypes below.
STRUCTURAL = set('alias detach view _unsafe_view reshape transpose t permute expand expand_as '
    'unsqueeze squeeze slice select narrow split split_with_sizes unbind cat stack '
    'contiguous clone copy_ _to_copy to empty empty_like empty_strided zeros zeros_like '
    'ones ones_like full full_like new_empty new_zeros new_ones new_full resize_ '
    'as_strided as_strided_ view_as_real view_as_complex real imag conj _conj lift_fresh lift_fresh_copy resolve_conj resolve_neg '
    'embedding index index_select gather scatter scatter_ index_put_ index_copy_ '
    'masked_select masked_fill masked_fill_ where pad constant_pad_nd reflection_pad2d '
    'replication_pad2d replication_pad3d repeat repeat_interleave flip roll '
    'arange linspace scalar_tensor _local_scalar_dense item equal is_same_size '
    'isfinite isnan isinf eq ne lt le gt ge logical_and logical_or logical_not '
    'bitwise_and bitwise_or bitwise_not any all nonzero count_nonzero argmax argmin '
    'max_pool2d_with_indices max_pool2d_with_indices_backward '
    'rand randn rand_like randn_like normal_ uniform_ bernoulli_ dropout native_dropout '
    'record_stream _assert_async _assert_scalar fill_ zero_ set_'.split())
ELEMENTWISE = set('add add_ sub sub_ mul mul_ div div_ true_divide floor_divide '
    'neg abs sign reciprocal sqrt rsqrt square pow exp exp2 log log2 log10 '
    'sin cos tan atan2 erf tanh sigmoid relu relu_ threshold clamp clamp_ clamp_min '
    'clamp_max maximum minimum fmod remainder round floor ceil trunc '
    'threshold_backward'.split())


def _float_tensors(value):
    return [x for x in tree_flatten(value)[0] if torch.is_tensor(x) and (x.is_floating_point() or x.is_complex())]


class ArithmeticCounter(TorchDispatchMode):
    def __init__(self):
        super().__init__()
        self.components = defaultdict(int)
        self.operations = Counter()
        self.unknown = Counter()
        self.approximations = set()
        self.dense_attention = False

    def add(self, value, operation, approximation=None):
        self.components[_LABEL.get()] += int(value)
        self.operations[operation] += int(value)
        if approximation: self.approximations.add(approximation)

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        out = func(*args, **kwargs)
        packet = getattr(func, '_overloadpacket', None)
        name = str(packet or func)
        short = name.split('.')[-1]
        floats = _float_tensors((args, kwargs, out))
        if not floats or short in STRUCTURAL:
            return out
        if packet in flop_registry:
            try:
                value=flop_registry[packet](*args, **kwargs, out_val=out)
                # PyTorch's standard matmul/conv formulas omit bias additions.
                if short=='addmm' or (short in ('convolution','_convolution') and len(args)>2 and args[2] is not None):
                    value+=sum(x.numel() for x in _float_tensors(out))
                self.add(value, name)
                if 'attention' in name:
                    self.dense_attention = True
                    self.approximations.add('SDPA uses dense mathematical attention; masks/kernel sparsity not modeled')
                return out
            except (TypeError, ValueError, RuntimeError, AssertionError) as error:
                self.unknown[name+':formula_'+type(error).__name__] += 1
                return out
        outputs = _float_tensors(out)
        n = sum(x.numel() for x in outputs)
        if short in ELEMENTWISE:
            complex_output=any(x.is_complex() for x in outputs)
            factor=(6 if short.startswith('mul') else 11 if short.startswith('div') else 2) if complex_output else 1
            self.add(factor*n, name, 'scalar real ops use one operation per output element; complex add/mul/div use 2/6/11')
        elif short in ('sum','mean','prod','amax','amin','max','min'):
            self.add(sum(x.numel() for x in _float_tensors(args)), name, 'reductions use one operation per input element')
        elif short in ('native_layer_norm','native_group_norm','layer_norm','group_norm','rms_norm'):
            self.add(8*args[0].numel(), name, 'normalization uses eight operations per input element')
        elif short in ('native_layer_norm_backward','native_group_norm_backward'):
            self.add(12*args[0].numel(), name, 'normalization backward uses twelve operations per gradient element')
        elif short in ('gelu','gelu_backward','silu','silu_backward','sigmoid_backward','tanh_backward'):
            self.add(10*n, name, 'nonlinear activations use ten operations per output element')
        elif short in ('_softmax','_safe_softmax','_log_softmax','softmax','log_softmax'):
            self.add(5*n,name,'softmax uses five operations per element')
        elif short in ('_softmax_backward_data','_log_softmax_backward_data'):
            self.add(5*n,name,'softmax backward uses five operations per element')
        elif short in ('linalg_vector_norm','norm'):
            self.add(3*args[0].numel(),name,'vector norm uses three operations per input element')
        elif short.startswith('upsample_'):
            self.add((32 if 'bicubic' in short else 8)*n,name,'interpolation uses a stencil-based operation estimate')
        elif short in ('scatter_add','scatter_add_','index_add','index_add_'):
            self.add(_float_tensors(args)[-1].numel(),name)
        elif short in ('addcmul','addcmul_','addcdiv','addcdiv_','lerp','lerp_'):
            self.add(3*n,name)
        else:
            self.unknown[name] += 1
        return out

    def summary(self):
        return dict(counted_flops=sum(self.components.values()),
            components_flops=dict(self.components), operations_flops=dict(self.operations),
            unsupported_ops=dict(self.unknown), approximations=sorted(self.approximations),
            status='partial_unsupported_ops' if self.unknown else 'estimated_all_observed_ops',
            dense_attention_bound=self.dense_attention, multiply_add_flops=2)


@contextmanager
def component(name):
    token=_LABEL.set(name)
    try: yield
    finally: _LABEL.reset(token)


def flop_component(name):
    def decorate(function):
        @wraps(function)
        def wrapped(*args, **kwargs):
            with component(name): return function(*args, **kwargs)
        return wrapped
    return decorate


@contextmanager
def compiled_attention(query, key, value):
    """Count once and run original compiled attention without an active dispatch mode.

    A dispatch mode can change compiler tracing. Keep the original kernel path;
    record a dense upper-bound formula instead of claiming sparse kernel FLOPs.
    """
    counter=_ACTIVE.get()
    if counter is None:
        yield
        return
    b,h,q,d=query.shape; k=key.shape[-2]; v=value.shape[-1]
    counter.add(2*b*h*q*k*(d+v),'compiled_flex_attention_dense_equivalent')
    counter.dense_attention=True
    counter.approximations.add('compiled FlexAttention uses dense QK/AV upper bound; masking/softmax kernel work not counted exactly')
    with _disable_current_modes(): yield


@contextmanager
def mask_construction():
    # Mask construction is integer/boolean bookkeeping, not policy arithmetic.
    if _ACTIVE.get() is None: yield
    else:
        with _disable_current_modes(): yield


def _merge(server, record):
    if not hasattr(server,'policy_flops_totals'):
        server.policy_flops_totals=dict(counted_flops=0,components_flops={},unsupported_ops={},
            approximations=[],requests=0,dense_attention_bound=False)
    total=server.policy_flops_totals
    total['counted_flops']+=record['counted_flops']; total['requests']+=1
    for key in ('components_flops','unsupported_ops'):
        for k,v in record[key].items(): total[key][k]=total[key].get(k,0)+v
    total['approximations']=sorted(set(total['approximations'])|set(record['approximations']))
    total['dense_attention_bound'] |= record['dense_attention_bound']
    total['status']='partial_unsupported_ops' if total['unsupported_ops'] else 'estimated_all_observed_ops'
    total['scope']='all executed Torch policy calls: reset, observation/cache, video/action inference, text/ICL and inversion forward calls; excludes checkpoint initialization and simulator/rendering'
    return json.loads(json.dumps(total))


def profile_policy_call(function):
    @wraps(function)
    def wrapped(server, obs, *args, **kwargs):
        if os.environ.get('COUNT_POLICY_FLOPS','0')!='1':
            return function(server,obs,*args,**kwargs)
        reset=bool(obs.get('reset'))
        if reset:
            server.policy_flops_totals=dict(counted_flops=0,components_flops={},unsupported_ops={},
                approximations=[],requests=0,dense_attention_bound=False)
        phase='reset' if reset else 'observation_cache' if obs.get('compute_kv_cache') else 'sampling'
        counter=ArithmeticCounter(); token=_ACTIVE.set(counter)
        result=None; failed=True
        try:
            with counter, component(phase):
                result=function(server,obs,*args,**kwargs)
            failed=False
            return result
        finally:
            _ACTIVE.reset(token)
            record=counter.summary()
            record.update(phase=phase,episode=getattr(server,'workload_episode',0),
                chunk=getattr(server,'chunk_idx',0),failed=failed)
            total=_merge(server,record)
            if isinstance(result,dict): result['compute']=total
            root=Path(server.save_root)/'diagnostics'; root.mkdir(parents=True,exist_ok=True)
            with (root/'flops.jsonl').open('a') as stream:
                stream.write(json.dumps(dict(**record,cumulative=total))+'\n')
    return wrapped

