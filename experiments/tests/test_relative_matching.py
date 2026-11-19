"""Check zero-change abstention and relative correspondence independently of time."""

import numpy as np
from experiments.replay.relative_matching import matching_scores, evaluate


def test_relative_l2_matches_change_despite_different_initial_offsets():
    reference = np.array([[1.0, 0.0], [1.0, 1.0], [1.0, 2.0]])
    queries = np.array([[10.0, 0.0], [10.0, 2.0], [10.0, 1.0]])
    scores, valid, _ = matching_scores(reference, queries)
    assert valid.tolist() == [False, True, True]
    assert scores["relative_l2"].argmax(1).tolist() == [0, 2, 1]
    assert np.isneginf(scores["relative_cosine"][:, 0]).all()


def test_stationary_queries_abstain_and_do_not_get_free_accuracy():
    reference = np.eye(3)
    queries = np.repeat(reference[:1], 3, axis=0)
    labels = {
        "reference": {0: "pre_grasp", 1: "holding", 2: "placed"},
        "robot": {0: "pre_grasp", 1: "pre_grasp", 2: "placed"},
    }
    rows, _ = evaluate(reference, queries, [0, 1, 2], labels, "test")
    relative = [r for r in rows if r["method"] == "relative_cosine"]
    assert not relative[0]["evaluated"]
    assert all(
        r["abstained"] and r["evaluated"] and not r["stage_correct"]
        for r in relative[1:]
    )
    control = [r for r in rows if r["method"] == "stay_at_start" and r["evaluated"]]
    assert [r["stage_correct"] for r in control] == [True, False]
