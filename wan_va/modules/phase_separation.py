"""Opt-in single-GPU transformer offload around observation VAE encoding.

KV dictionaries/metadata stay on GPU. Registered parameters and buffers move to
CPU then back without changing dtype. Requires an unsharded frozen transformer.
"""
import json
import os
import logging
from pathlib import Path
import time

import torch


logger = logging.getLogger(__name__)


def cache_state(model):
    """Track cache identities/devices without copying tensor contents."""
    tensors = []
    for index, block in enumerate(model.blocks):
        for name, entry in block.attn1.attn_caches.items():
            for kind in ('k', 'v'):
                tensor = entry[kind]
                tensors.append((f'{index}:{name}:{kind}', tensor))
    for name in ('type_ids_cache', 'seq_ids_cache', 'frame_ids_cache', 'cache_type_ids_cache'):
        tensor = getattr(model, name, None)
        if tensor is not None:
            tensors.append((name, tensor))
    signature = [(name, id(t), t.data_ptr(), str(t.device), str(t.dtype), tuple(t.shape)) for name,t in tensors]
    return signature, sum(t.numel()*t.element_size() for _,t in tensors)


def memory(device):
    free,total = torch.cuda.mem_get_info(device)
    return dict(allocated_bytes=torch.cuda.memory_allocated(device),
                reserved_bytes=torch.cuda.memory_reserved(device), free_bytes=free,total_bytes=total)


def encode_with_phase_separation(server, function, observation, phase):
    if not getattr(server, 'phase_separation', False):
        return function(observation)
    model,device = server.transformer,server.device
    offload = getattr(server, 'phase_offload_transformer', True)
    signature,cache_bytes = cache_state(model)
    torch.cuda.synchronize(device)
    started = time.perf_counter()
    row = dict(phase=phase,chunk=server.chunk_idx,offload=offload,kv_and_metadata_bytes=cache_bytes,
               before=memory(device),status='started')
    transfer_started = started
    moved = False
    try:
        if offload:
            # finally restores even if transfer, cache verification, or encoding raises.
            moved = True
            model.to(device='cpu')
            torch.cuda.synchronize(device)
            torch.cuda.empty_cache()
        row['offload_seconds'] = time.perf_counter()-transfer_started
        if cache_state(model)[0] != signature:
            raise RuntimeError('Transformer movement changed KV cache/metadata identity, dtype or device')
        row['encoding_start'] = memory(device)
        torch.cuda.reset_peak_memory_stats(device)
        encoding_started = time.perf_counter()
        encoded = function(observation)
        torch.cuda.synchronize(device)
        row.update(encoding_seconds=time.perf_counter()-encoding_started,
                   encoding_peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
                   encoding_end=memory(device),status='encoded')
    except BaseException as error:
        row.update(status='failed',error=repr(error))
        raise
    finally:
        try:
            restore_started = time.perf_counter()
            if moved:
                # Drop unused encoding workspace before weights return. Live VAE state stays.
                torch.cuda.empty_cache()
                model.to(device=device)
                torch.cuda.synchronize(device)
            row['restore_seconds'] = time.perf_counter()-restore_started
            row['after'] = memory(device)
            if cache_state(model)[0] != signature:
                raise RuntimeError('KV cache changed while the transformer was offloaded')
        except BaseException as error:
            row.update(status='restore_failed',restore_error=repr(error))
            raise
        finally:
            row['total_seconds'] = time.perf_counter()-started
            logger.info('[phase-separation] %s',row)
            if os.environ.get('RUN_DIAGNOSTICS', '0') == '1':
                output = Path(server.save_root)/'diagnostics'/'phase_separation.jsonl'
                output.parent.mkdir(parents=True, exist_ok=True)
                with output.open('a') as stream:
                    stream.write(json.dumps(row)+'\n')
    return encoded


def action_digest(action):
    """Hash the returned action array without dropping dtype/shape information."""
    import hashlib
    import numpy as np
    if torch.is_tensor(action):
        action = action.detach().cpu()
        if action.dtype == torch.bfloat16:
            action = action.float()
        action = action.numpy()
    array = np.ascontiguousarray(action)
    header = f"{array.dtype}:{array.shape}:".encode()
    return hashlib.sha256(header + array.tobytes()).hexdigest()
