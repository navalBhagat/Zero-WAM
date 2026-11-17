"""Small workload records and wall timers; no model execution or CUDA sync."""
import logging
import time


class ResetTimer:
    def __init__(self, server):
        self.server = server
        self.enabled = bool(getattr(server.job_config, 'profile_timing', False))
        self.started = self.previous = time.perf_counter()
        self.parts = {}
        server.compute_components = {} if self.enabled else None
        # Same ordinal as config.json episode metadata, including failed resets.
        server.workload_episode = getattr(server, 'workload_episode', -1) + 1
        transformer = getattr(server, 'transformer', None)
        if transformer is not None:
            transformer.compute_workload = {} if self.enabled else None
        reward = getattr(server, 'external_reward', None)
        if reward is not None:
            reward.compute_workload = {} if self.enabled else None

    def mark(self, name):
        now = time.perf_counter()
        if self.enabled:
            self.parts[name] = self.parts.get(name, 0.) + now-self.previous
        self.previous = now

    def finish(self, **metadata):
        self.mark('finalization')
        record = dict(episode=self.server.workload_episode,
                      total_seconds=time.perf_counter()-self.started,
                      components_seconds=self.parts, timing='unsynchronized_wall', **metadata)
        self.server.last_reset_profile = record if self.enabled else None
        if self.enabled:
            logging.getLogger(__name__).info('[reset-profile] %s', record)
        flush_workload(self.server, 'reset')


def record_transformer_work(model, mode, query_tokens, padded_queries,
                            key_tokens, text_tokens):
    """Aggregate actual shapes. Dense attention is a bound, not executed sparse FLOPs."""
    ledger = getattr(model, 'compute_workload', None)
    if ledger is None:
        return
    block = model.blocks[0]
    width = block.attn1.to_q.in_features if mode == 'video' else block.attn1.action_to_q.in_features
    inner = block.attn1.inner_dim
    ffn = block.ffn if mode == 'video' else block.action_ffn
    # FeedForward's first projection is GELU's linear layer; no weight reads/copies.
    ffn_width = ffn.net[0].proj.out_features
    key = (mode, query_tokens, padded_queries, key_tokens, text_tokens,
           width, inner, ffn_width, len(model.blocks))
    ledger[key] = ledger.get(key, 0) + 1


def flush_workload(server, phase):
    model = getattr(server, 'transformer', None)
    ledger = getattr(model, 'compute_workload', None)
    reward = getattr(server, 'external_reward', None)
    components = getattr(reward, 'compute_workload', None)
    server_components = getattr(server, 'compute_components', None)
    combined = dict(server_components or {})
    for key, count in (components or {}).items(): combined[key] = combined.get(key, 0)+count
    if ledger or combined:
        keys = ('mode', 'query_tokens', 'padded_queries', 'key_tokens', 'text_tokens',
                'width', 'inner', 'ffn_width', 'layers')
        records = [dict(zip(keys, shape), calls=count) for shape, count in (ledger or {}).items()]
        logging.getLogger(__name__).info('[compute-workload] %s', dict(
            episode=server.workload_episode, chunk=getattr(server, 'chunk_idx', 0),
            phase=phase, transformer=records, components=[dict(component=k[0],
                input_shape=k[1], backward_to_input=k[2], calls=v)
                for k,v in combined.items()]))
        if ledger is not None: ledger.clear()
        if components is not None: components.clear()
        if server_components is not None: server_components.clear()


def record_reward_component(reward, component, tensor):
    ledger = getattr(reward, 'compute_workload', None)
    if ledger is None:
        return
    import torch
    key = (component, tuple(tensor.shape), bool(torch.is_grad_enabled() and tensor.requires_grad))
    ledger[key] = ledger.get(key, 0) + 1


def record_server_component(server, component, tensor):
    ledger = getattr(server, 'compute_components', None)
    if ledger is not None:
        key = (component, tuple(tensor.shape), False)
        ledger[key] = ledger.get(key, 0)+1
