"""Scientific boundaries of the offline correspondence filter."""

import numpy as np
import pytest
from experiments.replay.progress import track, read_labels


def test_future_visual_alias_cannot_skip_transition_bound():
    scores = np.zeros((3, 20))
    scores[:, 19] = 1
    rows, posterior = track(scores, max_advance=2)
    assert all(row["nearest_frame"] == 19 for row in rows)
    assert rows[0]["progress_frame"] == 0
    assert not posterior[2, 5:].any()
    np.testing.assert_allclose(posterior.sum(1), 1)


def test_reachable_visual_evidence_orders_sequence():
    scores = np.zeros((4, 10))
    for i in range(4):
        scores[i, i] = 1
    rows, _ = track(scores, max_advance=2, temperature=0.02)
    assert [r["progress_frame"] for r in rows] == [0, 1, 2, 3]


def test_ambiguous_images_do_not_prove_progress():
    rows, _ = track(np.zeros((4, 20)), max_advance=4)
    assert rows[-1]["interval_high"] > rows[-1]["interval_low"]
    assert rows[-1]["nearest_margin"] == 0


def test_invalid_scores_and_anchor():
    with pytest.raises(ValueError):
        track([[float("nan")]])
    with pytest.raises(ValueError):
        track([[0]], start_frame=1)


def test_labels_are_optional_evaluation_only(tmp_path):
    path = tmp_path / "labels.csv"
    path.write_text(
        "kind,index,stage\nreference,0,pre_grasp\nrobot,0,failed_grasp\nreference,1,\n"
    )
    assert read_labels(path, 2, [0]) == {
        "reference": {0: "pre_grasp"},
        "robot": {0: "failed_grasp"},
    }
