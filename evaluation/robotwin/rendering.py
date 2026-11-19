"""Enforce samples-per-pixel across SAPIEN setup calls in this process.

RoboTwin hardcodes 32 in Base_Task.setup_scene. Wrapping that one global setter
keeps external checkout files untouched and prevents later scenes resetting it.
Path depth and denoiser are deliberately unchanged.
"""
import os

_STATE_ATTRIBUTE = '_zerowam_samples_per_pixel'


def requested_samples(value=None):
    value = os.environ.get('RENDER_SAMPLES_PER_PIXEL', '32') if value is None else value
    try:
        samples = int(str(value))
    except (TypeError, ValueError) as error:
        raise ValueError('RENDER_SAMPLES_PER_PIXEL must be a positive integer') from error
    if samples < 1:
        raise ValueError('RENDER_SAMPLES_PER_PIXEL must be a positive integer')
    return samples


def configure_rendering(render=None, samples=None):
    if render is None:
        from sapien import render
    count = requested_samples(samples)
    if samples is not None:
        os.environ['RENDER_SAMPLES_PER_PIXEL'] = str(count)
    state = getattr(render, _STATE_ATTRIBUTE, None)
    if state is None:
        state = dict(requested=count, effective=None, original=render.set_ray_tracing_samples_per_pixel)
        def enforce(_scene_default):
            result = state['original'](state['requested'])
            state['effective'] = state['requested']
            return result
        render.set_ray_tracing_samples_per_pixel = enforce
        setattr(render, _STATE_ATTRIBUTE, state)
    state['requested'] = count
    render.set_ray_tracing_samples_per_pixel(count)
    return render_settings(render)


def render_settings(render=None):
    if render is None:
        from sapien import render
    state = getattr(render, _STATE_ATTRIBUTE, None)
    return dict(samples_per_pixel_requested=requested_samples() if state is None else state['requested'],
                samples_per_pixel_effective=None if state is None else state['effective'],
                verification='value applied through the native SAPIEN setter')
