"""Real subprocess check: a stalled call exits and leaves a diagnostic stack."""
import os
import subprocess
import sys
import unittest
from unittest.mock import patch

from evaluation.robotwin.watchdog import guarded_call


class WatchdogTests(unittest.TestCase):
    def test_timeout_dumps_stack_and_terminates_client(self):
        environment = dict(os.environ, NOISE_INVERSION_STEP_TIMEOUT="0.05")
        program = ('import time; from evaluation.robotwin.watchdog import guarded_call; '
                   'guarded_call("simulated action hang", time.sleep, 10)')
        result = subprocess.run([sys.executable, "-c", program], env=environment,
                                capture_output=True, text=True, timeout=5)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("simulated action hang", result.stdout)
        self.assertIn("Timeout", result.stderr)
        self.assertIn("guarded_call", result.stderr)

    def test_completed_call_cancels_watchdog(self):
        with patch.dict(os.environ, NOISE_INVERSION_STEP_TIMEOUT="10"), \
             patch("evaluation.robotwin.watchdog.faulthandler.dump_traceback_later") as start, \
             patch("evaluation.robotwin.watchdog.faulthandler.cancel_dump_traceback_later") as cancel:
            self.assertEqual(guarded_call("short", lambda: 7), 7)
            start.assert_called_once()
            cancel.assert_called_once()


if __name__ == "__main__":
    unittest.main()
