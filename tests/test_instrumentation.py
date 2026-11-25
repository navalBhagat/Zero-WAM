from types import SimpleNamespace
from unittest.mock import patch

import torch

import importlib.util
from pathlib import Path
spec=importlib.util.spec_from_file_location('instrumentation',Path(__file__).resolve().parents[1]/'wan_va/diagnostics/instrumentation.py')
instrumentation=importlib.util.module_from_spec(spec);spec.loader.exec_module(instrumentation)
ResetTimer=instrumentation.ResetTimer
flush_workload=instrumentation.flush_workload


def test_reset_timer_separates_parts_and_disabled_mode_records_nothing():
    server=SimpleNamespace(job_config=SimpleNamespace(profile_timing=True),transformer=SimpleNamespace())
    with patch.object(instrumentation.time, 'perf_counter',side_effect=[0.,2.,5.,5.]):
        timer=ResetTimer(server);timer.mark('reference');timer.finish(first_context_use=True)
    assert server.last_reset_profile['components_seconds']==dict(reference=2.,finalization=3.)
    assert server.last_reset_profile['total_seconds']==5.
    assert server.last_reset_profile['first_context_use']
    server.job_config.profile_timing=False
    timer=ResetTimer(server);timer.finish()
    assert server.last_reset_profile is None and server.transformer.compute_workload is None

