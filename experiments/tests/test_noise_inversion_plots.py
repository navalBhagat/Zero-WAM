"""Numerical plot contracts: matched denominators, fallback and missing data."""
import tempfile
import unittest
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from experiments.noise_inversion.plots import success_figure, transfer_figure, write_plots


def row(task="place_object_scale"):
    return dict(task=task, valid_pairs=4, incomplete=1, pending=2,
                source_successes=3, control_successes=2, replay_successes=3,
                replay_success_within_noise_horizon=2,
                control_failures=2, rescues=2, regressions=1)


class PlotTests(unittest.TestCase):
    def tearDown(self):
        plt.close("all")

    def test_success_heights_and_fallback_overlay(self):
        figure = success_figure(plt, [row()])
        heights = [bar.get_height() for bar in figure.axes[0].patches]
        self.assertEqual(heights, [75, 50, 75, 25])
        self.assertEqual(figure.axes[0].patches[-1].get_y(), 50)

    def test_transfer_uses_eligible_denominators(self):
        figure = transfer_figure(plt, [row()])
        self.assertEqual(figure.axes[0].patches[0].get_height(), 100)
        self.assertEqual(figure.axes[1].patches[0].get_height(), 50)

    def test_missing_trials_are_not_zero_success_bars(self):
        missing = row()
        missing.update(valid_pairs=0, control_failures=0, control_successes=0)
        figure = success_figure(plt, [missing])
        self.assertFalse(figure.axes[0].patches)
        self.assertEqual([text.get_text() for text in figure.axes[0].texts], ["N/A"] * 3)
        figure = transfer_figure(plt, [missing])
        self.assertTrue(all(not axis.patches for axis in figure.axes))

    def test_write_plots_produces_nonempty_files(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            write_plots(directory, [row(), row("place_bread_basket")])
            for name in ("success.png", "rescue_regression.png"):
                self.assertGreater((directory / "plots" / name).stat().st_size, 1000)


if __name__ == "__main__":
    unittest.main()
