import importlib.util
from pathlib import Path
import unittest


spec = importlib.util.spec_from_file_location(
    "setup_download", Path(__file__).resolve().parents[1] / "setup/zerowam/download.py"
)
download = importlib.util.module_from_spec(spec)
spec.loader.exec_module(download)


class EvaluationDownloadTests(unittest.TestCase):
    def setUp(self):
        self.references = set(download.evaluation_files())
        self.latents = {n for n in self.references if n.startswith("human_latents/")}

    def test_latent_only_hub_is_supported(self):
        names = self.latents | {"icl_configs/test.json", "robotwin_data/unused.pth"}
        selected = download.select_files(names, "dataset", "eval")
        self.assertEqual(set(selected), self.latents | {"icl_configs/test.json"})

    def test_available_videos_are_included(self):
        selected = download.select_files(self.references, "dataset", "eval")
        self.assertEqual(set(selected), self.references)

    def test_video_does_not_replace_missing_latent(self):
        missing = next(iter(self.latents))
        with self.assertRaisesRegex(RuntimeError, "references missing"):
            download.select_files(self.references - {missing}, "dataset", "eval")


if __name__ == "__main__":
    unittest.main()
