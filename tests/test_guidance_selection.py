"""Explicit selection and cached-feature behavior, without downloading DINO."""
import importlib
from pathlib import Path
import sys
import types

import pytest
import torch
import torch.nn.functional as F

package = types.ModuleType('_selection_test')
package.__path__ = [str(Path(__file__).resolve().parents[1] / 'wan_va/guidance')]
sys.modules[package.__name__] = package
selection = importlib.import_module('_selection_test.selection')
DinoReward = importlib.import_module('_selection_test.dino_reward').DinoReward


def test_fixed_selector_and_explicit_per_chunk_override():
    selector = selection.FixedFrameSelector(20, frames="10,11")
    assert selector.select(0) == [10, 11]
    assert selector.select(5) == [10, 11]
    assert selector.select(5, [3]) == [3]
    assert selector.select(6) == [10, 11]  # override does not mutate the default
    assert selection.FixedFrameSelector(20, frames='2, 9').select(0) == [2, 9]


@pytest.mark.parametrize('frames', [[], [-1], [20], [1.5], [True], [2, 2]])
def test_explicit_indices_are_validated(frames):
    selector = selection.FixedFrameSelector(20)
    with pytest.raises(ValueError):
        selector.select(0, frames)


def test_feature_bank_matches_old_fixed_segment_reward_and_is_reused(tmp_path):
    frames = torch.rand(10, 3, 4, 6)
    calls = []
    def features(batch):
        calls.append(len(batch))
        return F.normalize(batch.mean((2, 3)), dim=-1)
    old = DinoReward.__new__(DinoReward)
    old.device, old.features = torch.device('cpu'), features
    old.set_reference(frames[2:4])
    expected = old.reference.clone()
    calls.clear()
    reward = DinoReward.__new__(DinoReward)
    reward.device, reward.features = torch.device('cpu'), features
    reward.prepare_references(frames)
    assert calls == [4, 4, 2]
    reward.select_reference([2, 3])
    torch.testing.assert_close(reward.reference, expected)
    reward.select_reference([7])
    assert calls == [4, 4, 2]  # selecting never re-encodes the human video
    assert reward.selected_frames == [7]
    paths = reward.save_reference_frames(tmp_path)
    assert Path(paths[0]).name == 'frame_0007.png'
    assert Path(paths[0]).is_file()
