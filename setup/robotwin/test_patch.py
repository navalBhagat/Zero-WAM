"""CPU-only checks: python setup/robotwin/test_patch.py."""
import tempfile
import unittest
from pathlib import Path

from patch import patch_text


class PatchTests(unittest.TestCase):
    def test_patch_preserves_backup_and_is_idempotent(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "planner.py"
            old = "if np.linalg.norm(delta_twist) < 1e-4 or collide or not within_joint_limit:\n"
            new = "if np.linalg.norm(delta_twist) < 1e-4 or not within_joint_limit:\n"
            path.write_text(old)
            pattern = r'(if np.linalg.norm\(delta_twist\) < 1e-4 )(or collide )(or not within_joint_limit:)'
            for _ in range(2):
                patch_text(path, pattern, r'\1\3', new.strip())
            self.assertEqual(path.read_text(), new)
            self.assertEqual(path.with_name("planner.py.zerowam-upstream").read_text(), old)

    def test_unknown_source_fails_without_modification(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "unknown.py"
            path.write_text("unrecognized source")
            with self.assertRaises(RuntimeError):
                patch_text(path, "old", "new", "already patched")
            self.assertEqual(path.read_text(), "unrecognized source")

    def test_utf8_patch_is_idempotent(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "loader.py"
            path.write_text('with open(urdf_file, "r") as f:\n    pass\n')
            for _ in range(2):
                patch_text(path, r'("r")(\))( as)', r'\1, encoding="utf-8") as', 'encoding="utf-8"')
            self.assertEqual(path.read_text().count('encoding="utf-8"'), 1)


if __name__ == "__main__":
    unittest.main()
