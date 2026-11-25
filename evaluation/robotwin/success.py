"""Latch task success before a later action can undo the completed state."""


def update_success(task):
    if not task.eval_success:
        task.eval_success = bool(task.check_success())
    return bool(task.eval_success)
