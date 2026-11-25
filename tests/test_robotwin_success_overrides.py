"""Test released task criteria without importing RoboTwin/SAPIEN."""
import ast
from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def success_check(task):
    path = ROOT / "evaluation/robotwin/envs" / f"{task}.py"
    tree = ast.parse(path.read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == task)
    method = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "check_success")
    scope = dict(np=np)
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(path), "exec"), scope)
    return scope["check_success"]


def actor(position):
    return SimpleNamespace(get_pose=lambda: SimpleNamespace(p=np.array(position)), get_name=lambda: "actor")


class SuccessCriteriaTests(unittest.TestCase):
    def robot(self, opened=True):
        return SimpleNamespace(is_left_gripper_open=lambda: opened, is_right_gripper_open=lambda: opened)

    def test_stapler_requires_pad_contact_and_release(self):
        check = success_check("move_stapler_pad")
        task = SimpleNamespace(stapler=actor([.07, .05, 0]), pad=actor([0, 0, 0]),
            check_actors_contact=lambda *args: True, robot=self.robot())
        self.assertTrue(check(task))
        task.check_actors_contact = lambda *args: False
        self.assertFalse(check(task))
        task.check_actors_contact = lambda *args: True
        task.robot = self.robot(opened=False)
        self.assertFalse(check(task))
        task.robot = self.robot()
        task.stapler = actor([.09, 0, 0])
        self.assertFalse(check(task))

    def test_seal_requires_xy_tolerance_and_open_grippers(self):
        check = success_check("stamp_seal")
        task = SimpleNamespace(seal=actor([.02, -.02, 0]), target=actor([0, 0, 0]), robot=self.robot())
        self.assertTrue(check(task))
        task.seal = actor([.04, 0, 0])
        self.assertFalse(check(task))
        task.seal = actor([0, 0, 0])
        task.robot = self.robot(opened=False)
        self.assertFalse(check(task))


if __name__ == "__main__":
    unittest.main()
