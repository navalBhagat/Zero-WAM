"""Explicit inference switches. Parameters never implicitly enable a modality."""
import math


def flag(value, name):
    if value not in (False, True, 0, 1, '0', '1'):
        raise ValueError(f'{name} must be 0 or 1')
    return str(value) in ('1', 'True')


def read_conditioning(env):
    """Canonical names only: old negative-scale/text conventions are not inferred."""
    values = {
        'use_icl': flag(env.get('USE_ICL', '1'), 'USE_ICL'),
        'use_text': flag(env.get('USE_TEXT', '0'), 'USE_TEXT'),
        'icl_guidance_scale': float(env.get('ICL_GUIDANCE_SCALE', '5')),
        'text_guidance_scale': float(env.get('TEXT_GUIDANCE_SCALE', '1')),
    }
    for name in ('icl_guidance_scale', 'text_guidance_scale'):
        if not math.isfinite(values[name]) or values[name] < 0:
            raise ValueError(f'{name} must be finite and nonnegative')
    if values['use_text'] and values['text_guidance_scale'] < 1:
        raise ValueError('TEXT_GUIDANCE_SCALE must be >= 1 when USE_TEXT=1')
    if (values['use_icl'] and values['use_text'] and
            values['icl_guidance_scale'] > 1 and values['text_guidance_scale'] > 1):
        raise ValueError('Separate ICL and text CFG cannot both exceed 1 in the two-branch sampler; '
                         'set ICL_GUIDANCE_SCALE=1 or TEXT_GUIDANCE_SCALE=1')
    return values


def select_text(use_text, prompt_embedding, empty_embedding):
    """Used for both target and human-context inputs; ignore stored human captions."""
    return prompt_embedding if use_text else empty_embedding


def select_icl_text(source, use_text, target_embedding, cached_embedding):
    """Explicit legacy-caption diagnostic; ordinary runs keep the target embedding."""
    if source not in ('target', 'cached'):
        raise ValueError('ICL_TEXT_SOURCE must be target or cached')
    if source == 'target':
        return target_embedding
    if not use_text:
        raise ValueError('ICL_TEXT_SOURCE=cached requires USE_TEXT=1')
    if cached_embedding is None:
        raise ValueError('Legacy text test requires text_emb in the human latent file')
    return cached_embedding


def select_target_text(use_text, source, prompt_embedding, empty_embedding):
    """Explicit empty-target option for reproducing pre-switch ICL conditioning."""
    if source not in ('instruction', 'empty'):
        raise ValueError('TARGET_TEXT_SOURCE must be instruction or empty')
    return select_text(use_text and source == 'instruction', prompt_embedding, empty_embedding)
