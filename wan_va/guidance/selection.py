"""Explicit human-frame selection; replace this policy later, not the reward code."""
import operator


def validate_frames(indices, frame_count):
    if not indices:
        raise ValueError('Select at least one human RGB frame')
    result = []
    for value in indices:
        if isinstance(value, bool):
            raise ValueError('Human frame indices must be integers')
        try:
            index = operator.index(value)
        except TypeError as error:
            raise ValueError('Human frame indices must be integers') from error
        if index < 0 or index >= frame_count:
            raise ValueError(f'Human frame {index} outside [0, {frame_count - 1}]')
        result.append(index)
    if len(set(result)) != len(result):
        raise ValueError('Human frame indices must not repeat')
    return result


class FixedFrameSelector:
    """Fixed explicit frames, with an optional explicit override per robot chunk.

    The caller invokes select once per chunk and holds its result while denoising.
    No execution-progress inference or similarity search is performed here.
    """
    def __init__(self, frame_count, frames='0'):
        self.frame_count = frame_count
        try:
            fixed = [int(value.strip()) for value in frames.split(',')]
        except ValueError as error:
            raise ValueError('GUIDANCE_REFERENCE_FRAMES must be comma-separated integers') from error
        self.fixed_frames = validate_frames(fixed, frame_count)

    def select(self, chunk_index, frame_indices=None):
        # chunk_index is intentionally unused until a temporal selector is added.
        return validate_frames(self.fixed_frames if frame_indices is None else frame_indices,
                               self.frame_count)
