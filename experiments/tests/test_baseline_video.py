"""Reject incompatible composite layouts rather than embedding wrist/title panels."""

import numpy as np
import pytest
from experiments.replay.baseline_video import crop_external


def test_crop_uses_external_camera_without_title_or_wrists():
    frame = np.zeros((624, 960, 3), dtype=np.uint8)
    frame[40:280, :320] = 17
    crop = crop_external(frame)
    assert crop.shape == (240, 320, 3)
    assert (crop == 17).all()
    crop[:] = 0
    assert (frame[40:280, :320] == 17).all()


def test_unknown_layout_is_rejected():
    with pytest.raises(ValueError, match="Unexpected composite layout"):
        crop_external(np.zeros((240, 320, 3), dtype=np.uint8))
