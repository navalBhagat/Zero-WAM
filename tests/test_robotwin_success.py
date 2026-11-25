from types import SimpleNamespace

from evaluation.robotwin.success import update_success


def test_success_is_latched_before_a_later_state_can_undo_it():
    states = iter([False, True, False])
    task = SimpleNamespace(eval_success=False, check_success=lambda: next(states))
    assert not update_success(task)
    assert update_success(task)
    assert update_success(task)
    assert next(states) is False
