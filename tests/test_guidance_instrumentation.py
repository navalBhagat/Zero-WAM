from types import SimpleNamespace
from unittest.mock import patch

import torch

import importlib.util
from pathlib import Path
spec=importlib.util.spec_from_file_location('instrumentation',Path(__file__).resolve().parents[1]/'wan_va/guidance/instrumentation.py')
instrumentation=importlib.util.module_from_spec(spec);spec.loader.exec_module(instrumentation)
ResetTimer=instrumentation.ResetTimer
flush_workload=instrumentation.flush_workload
record_reward_component=instrumentation.record_reward_component


def test_reset_timer_separates_parts_and_disabled_mode_records_nothing():
    server=SimpleNamespace(job_config=SimpleNamespace(profile_timing=True),transformer=SimpleNamespace())
    with patch.object(instrumentation.time, 'perf_counter',side_effect=[0.,2.,5.,5.]):
        timer=ResetTimer(server);timer.mark('reference');timer.finish(first_reward_use=True)
    assert server.last_reset_profile['components_seconds']==dict(reference=2.,finalization=3.)
    assert server.last_reset_profile['total_seconds']==5.
    assert server.last_reset_profile['first_reward_use']
    server.job_config.profile_timing=False
    timer=ResetTimer(server);timer.finish()
    assert server.last_reset_profile is None and server.transformer.compute_workload is None


def test_reward_workload_distinguishes_reference_and_gradient_inputs(caplog):
    reward=SimpleNamespace(compute_workload={})
    record_reward_component(reward,'dino_features',torch.zeros(1,3,224,224))
    record_reward_component(reward,'dino_features',torch.zeros(1,3,224,224,requires_grad=True))
    server=SimpleNamespace(workload_episode=0,chunk_idx=2,external_reward=reward)
    with caplog.at_level('INFO'):
        flush_workload(server,'sampling')
    assert "'backward_to_input': True" in caplog.text
    assert "'backward_to_input': False" in caplog.text
    assert not reward.compute_workload
