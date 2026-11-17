"""Retrospective arithmetic accounting. Never convert elapsed time to FLOPs.

Two FLOPs per multiply-add. Transformer block linear layers are a subtotal;
attention is reported separately as a dense-equivalent bound. Neither is total
policy compute: VAE, DINO backward, conditioning, embedding, norms, MCP, etc.
are deliberately not silently assigned zero cost.
"""
import ast
import json
from pathlib import Path


DEFAULT_ARCHITECTURE = dict(num_attention_heads=24, attention_head_dim=128,
    num_layers=30, ffn_dim=14336, action_inner_dim=3072, action_ffn_dim=14336)


def core_flops(work):
    q, k, t = work['query_tokens'], work['key_tokens'], work['text_tokens']
    d, h, f, layers = (work[x] for x in ('width', 'inner', 'ffn_width', 'layers'))
    # Self Q/K/V/out; cross Q/out and text K/V; two FFN linear layers.
    linear = layers * (12*q*d*h + 4*t*h*h + 4*q*d*f)
    # Padding participates in attention, but not the block linear projections.
    attention = layers * 4 * work['padded_queries'] * (k+t) * h
    calls = work.get('calls', 1)
    return linear*calls, attention*calls


def records(path, marker):
    if not path.is_file():
        return []
    result = []
    for line in path.read_text(errors='replace').splitlines():
        if marker in line:
            try:
                entry = ast.literal_eval(line.split(marker, 1)[1].strip())
                if isinstance(entry, dict): result.append(entry)
            except (SyntaxError, ValueError): pass
    return result


def legacy_workloads(folder, config, architecture=None):
    """Recover sampling projection workloads from actual logged prediction tokens.

    No historical cache-update/reset arithmetic is inferred from incomplete shapes.
    Text length and architecture are explicitly assumptions unless provided.
    """
    architecture = architecture or DEFAULT_ARCHITECTURE
    metadata = [e for e in config.get('episodes', []) if e.get('server')]
    if not metadata:
        return []
    episodes = {int(e['episode_id']): e for e in metadata}
    ordinal = -1
    result = []
    for line in (folder/'server.log').read_text(errors='replace').splitlines() if (folder/'server.log').is_file() else []:
        if '[conditioning]' in line: ordinal += 1
        if '[ICL] cache after infer:' not in line: continue
        try: counts = ast.literal_eval(line.split('[ICL] cache after infer:', 1)[1].strip())
        except (ValueError, SyntaxError): continue
        # Reset ordinals follow config insertion order, not absolute resumed episode IDs.
        if ordinal < 0 or ordinal >= len(config.get('episodes', [])): continue
        episode_id = int(config['episodes'][ordinal]['episode_id'])
        episode = episodes.get(episode_id)
        if not episode: continue
        server = episode['server']
        if any(server.get(k) is None for k in ('video_inference_steps', 'action_inference_steps', 'frame_chunk_size')): continue
        q = counts.get(1, 0)
        if q <= 0: continue
        cfg = bool(server.get('use_icl') and (server.get('icl_guidance_scale') or 1)>1 or
                   server.get('use_text') and (server.get('text_guidance_scale') or 1)>1)
        h = architecture['num_attention_heads']*architecture['attention_head_dim']
        # Text sequence length (512 per branch) and action_per_frame=16 are
        # historical released Robotwin defaults, not measured tensor shapes.
        for mode, tokens, text, steps in [('video', q, 1024 if cfg else 512, server['video_inference_steps']+1),
                ('action', server['frame_chunk_size']*16, 512, server['action_inference_steps'])]:
            d = h if mode=='video' else architecture.get('action_inner_dim', h)
            if d <= 0: d = h
            result.append(dict(episode=ordinal, phase='sampling', transformer=[dict(
                mode=mode, query_tokens=tokens, padded_queries=tokens+128-tokens%128,
                key_tokens=0, text_tokens=text, width=d, inner=h,
                ffn_width=architecture['ffn_dim' if mode=='video' else 'action_ffn_dim'],
                layers=architecture['num_layers'], calls=steps)], historical=True))
    return result


def guidance_activity(folder, config):
    ordinal = -1
    activity = {}
    path = folder/'server.log'
    for line in path.read_text(errors='replace').splitlines() if path.is_file() else []:
        if '[conditioning]' in line: ordinal += 1
        if '[external-guidance]' not in line or ordinal < 0: continue
        try: entry=ast.literal_eval(line.split('[external-guidance]',1)[1].strip())
        except (ValueError,SyntaxError): continue
        if isinstance(entry,dict) and 'reward' in entry:
            activity[ordinal]=activity.get(ordinal,0)+1
    return activity


def estimate(folder, config, model_config=None):
    workload = records(folder/'server.log', '[compute-workload]')
    measured = bool(workload)
    architecture = None
    if model_config:
        architecture = DEFAULT_ARCHITECTURE | json.loads(Path(model_config).read_text())
    if not workload:
        workload = legacy_workloads(folder, config, architecture)
    activity = guidance_activity(folder, config)
    output = {}
    for ordinal, episode in enumerate(config.get('episodes', [])):
        rows = [w for w in workload if w.get('episode')==ordinal]
        linear = attention = sampling_linear = 0
        sampling_calls = 0
        for row in rows:
            for work in row['transformer']:
                a,b = core_flops(work); linear += a; attention += b
                if row['phase']=='sampling': sampling_linear += a
                if row['phase']=='sampling' and work['mode']=='video':
                    sampling_calls += work['calls']
        guided = episode.get('server', {}).get('use_guidance')
        updates = activity.get(ordinal, 0) if guided is not None else None
        output[episode['episode_id']] = dict(
            guidance_updates=updates,
            reward_decoder_forwards=updates,
            reward_dino_forwards=updates,
            reward_backward_passes=updates,
            reference_frames=(episode.get('server', {}).get('guidance') or {}).get('reference_frame_count'),
            transformer_core_linear_tflops=linear/1e12 if rows else None,
            sampling_transformer_core_linear_tflops=sampling_linear/1e12 if rows else None,
            transformer_dense_attention_tflops=attention/1e12 if rows and measured else None,
            compute_status=('partial_recorded_shapes' if measured else 'partial_assumed_architecture') if rows else 'unavailable',
            compute_scope='transformer block projections and FFN only; NOT total policy FLOPs',
            compute_assumptions=('actual shapes; dense attention is an upper bound on attention matmuls only' if measured else
                ('provided architecture' if model_config else 'constructor default architecture') +
                '; released Robotwin text=512/action_per_frame=16; sampling only; excludes reset/cache updates'),
            total_policy_tflops=None,
            compute_omissions='VAE encode/decode and backward; DINO forward/backward; prompt/conditioning; MCP; embeddings; norms; sparse attention execution',
            sampling_video_calls=sampling_calls,
            recorded_transformer_calls=sum(w['calls'] for r in rows for w in r['transformer']))
    return output
