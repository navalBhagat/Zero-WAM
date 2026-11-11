"""Explicit inference switches. Parameters never implicitly enable a modality."""
import math
import warnings


def flag(value, name):
    if value not in (False, True, 0, 1, '0', '1'):
        raise ValueError(f'{name} must be 0 or 1')
    return str(value) in ('1', 'True')


def read_conditioning(env):
    """Canonical names only: old negative-scale/text conventions are not inferred."""
    legacy = sorted(name for name in env if name in {
        'ICL_CFG', 'TARGET_TEXT_CFG', 'EXTERNAL_STRENGTH', 'EXTERNAL_EVERY',
        'EXTERNAL_MAX_NORM', 'DINO_CHECKPOINT', 'REFERENCE_START', 'REFERENCE_STOP',
        'EXTERNAL_REFERENCE_SOURCE', 'EXTERNAL_DECODER_DTYPE',
    })
    if legacy:
        warnings.warn('Ignoring legacy settings: ' + ', '.join(legacy) +
                      '. Use USE_ICL/USE_TEXT/USE_GUIDANCE and the documented '
                      'ICL_*, TEXT_*, GUIDANCE_* parameters.', FutureWarning, stacklevel=2)
    values = {
        'use_icl': flag(env.get('USE_ICL', '1'), 'USE_ICL'),
        'use_text': flag(env.get('USE_TEXT', '0'), 'USE_TEXT'),
        'use_guidance': flag(env.get('USE_GUIDANCE', '0'), 'USE_GUIDANCE'),
        'icl_guidance_scale': float(env.get('ICL_GUIDANCE_SCALE', '5')),
        'text_guidance_scale': float(env.get('TEXT_GUIDANCE_SCALE', '1')),
        'guidance_strength': float(env.get('GUIDANCE_STRENGTH', '0.01')),
    }
    for name in ('icl_guidance_scale', 'text_guidance_scale', 'guidance_strength'):
        if not math.isfinite(values[name]) or values[name] < 0:
            raise ValueError(f'{name} must be finite and nonnegative')
    if values['use_text'] and values['text_guidance_scale'] < 1:
        raise ValueError('TEXT_GUIDANCE_SCALE must be >= 1 when USE_TEXT=1')
    if values['use_guidance'] and values['guidance_strength'] <= 0:
        raise ValueError('GUIDANCE_STRENGTH must be positive when USE_GUIDANCE=1')
    if (values['use_icl'] and values['use_text'] and
            values['icl_guidance_scale'] > 1 and values['text_guidance_scale'] > 1):
        raise ValueError('Separate ICL and text CFG cannot both exceed 1 in the two-branch sampler; '
                         'set ICL_GUIDANCE_SCALE=1 or TEXT_GUIDANCE_SCALE=1')
    return values


def select_text(use_text, prompt_embedding, empty_embedding):
    """Used for both target and human-context inputs; ignore stored human captions."""
    return prompt_embedding if use_text else empty_embedding
