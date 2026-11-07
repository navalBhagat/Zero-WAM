"""CPU-only tests without Hub credentials/network: python setup/zerowam/test_download.py."""
from pathlib import Path
import tempfile
import unittest
import sys
from types import SimpleNamespace
from unittest.mock import patch

from download import additional_bytes, evaluation_files, link_destination, select_files, main


class DownloadTests(unittest.TestCase):
    def test_eval_only_selects_fixed_references(self):
        refs = evaluation_files()
        self.assertEqual(len(refs), 14)
        names = refs + ["icl_configs/ICL_config_robotwin.json", "robotwin_data/task/latents/x.pth",
                        "human_latents/robotwin/unneeded.pth", "agibot_data/data.tar.zst"]
        selected = select_files(names, "dataset", "eval")
        self.assertEqual(set(selected), set(refs + ["icl_configs/ICL_config_robotwin.json"]))

    def test_missing_reference_is_an_error(self):
        with self.assertRaises(RuntimeError):
            select_files([], "dataset", "eval")

    def test_robotwin_profile_excludes_external_data(self):
        names = ["robotwin_data/task/latents/x.pth", "human_data/robotwin/a.mp4", "agibot_data/a"]
        self.assertEqual(select_files(names, "dataset", "robotwin"), sorted(names[:2]))

    def test_completed_files_are_not_budgeted_twice(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "ok").write_bytes(b"abc")
            (root / "partial").write_bytes(b"x")
            self.assertEqual(additional_bytes({"ok": 3, "partial": 5, "new": 7}, root), 12)
            with self.assertRaises(RuntimeError):
                additional_bytes({"unknown": None}, root)

    def test_links_never_replace_existing_content(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = root / "target"
            target.mkdir()
            link = root / "repo/link"
            link_destination(link, target)
            link_destination(link, target)
            with self.assertRaises(RuntimeError):
                link_destination(link, root / "different")
            existing = root / "existing"
            existing.mkdir()
            with self.assertRaises(RuntimeError):
                link_destination(existing, target)

    def test_plan_never_transfers_and_download_resumes_saved_revision(self):
        with tempfile.TemporaryDirectory() as directory:
            revisions = []
            transfers = []
            class API:
                def whoami(self):
                    return {"name": "test"}
                def repo_info(self, repo, **kwargs):
                    revisions.append(kwargs["revision"])
                    return SimpleNamespace(sha="fixed-commit", siblings=[
                        SimpleNamespace(rfilename="config.json", size=3)])
            def snapshot(**kwargs):
                transfers.append(kwargs)
                destination = Path(kwargs["local_dir"])
                destination.mkdir(exist_ok=True)
                (destination / "config.json").write_bytes(b"abc")
            fake = SimpleNamespace(HfApi=API, snapshot_download=snapshot)
            argv = ["download.py", "checkpoint", "--storage-root", directory]
            with patch.dict(sys.modules, {"huggingface_hub": fake}), patch.dict("os.environ"):
                with patch.object(sys, "argv", argv):
                    main()
                self.assertFalse(transfers)
                with patch.object(sys, "argv", argv + ["--download", "--quota-free-gib", "0"]):
                    with self.assertRaises(RuntimeError):
                        main()
                self.assertFalse(transfers)
                with patch.object(sys, "argv", argv + ["--download", "--quota-free-gib", "3"]):
                    main()
                with patch.object(sys, "argv", argv):
                    main()
            self.assertEqual(revisions[-1], "fixed-commit")
            self.assertEqual(len(transfers), 1)


if __name__ == "__main__":
    unittest.main()
