"""Compatibility boundaries for the experiments package refactor."""

import hashlib
import json
from pathlib import Path
import subprocess
import sys
import zipfile

import pytest

from experiments.paths import ROOT
from experiments.campaign import provenance
from experiments.campaign.config import parser
from experiments.campaign.plans import specifications
from experiments.campaign.job_report import record_timing, check_completion


@pytest.mark.parametrize(
    "case", ["render", "baselines", "guidance", "precision", "distance"]
)
def test_campaign_settings_match_pre_refactor_fixture(case):
    """Protect every exported setting, not only selected flags or run counts."""
    fixtures = json.loads(
        (Path(__file__).parent / "fixtures/campaign_settings.json").read_text()
    )
    expected = fixtures[case]
    options = parser().parse_args(expected["argv"])
    runs = specifications(options)
    canonical = json.dumps(runs, sort_keys=True).replace(str(ROOT), "<ROOT>")
    assert len(runs) == expected["runs"]
    assert hashlib.sha256(canonical.encode()).hexdigest() == expected["digest"]


def test_nested_sources_and_root_helpers_are_snapshotted(tmp_path, monkeypatch):
    nested = tmp_path / "experiments/replay/nested/helper.py"
    nested.parent.mkdir(parents=True)
    nested.write_text("VALUE = 1\n")
    shared = tmp_path / "experiments/io.py"
    shared.write_text("VALUE = 2\n")
    generated = tmp_path / "experiments/stages/job/generated.py"
    generated.parent.mkdir(parents=True)
    generated.write_text("SHOULD_NOT_BE_ARCHIVED = True\n")
    monkeypatch.setattr(provenance, "ROOT", tmp_path)
    before, hashes = provenance.fingerprint()
    assert set(hashes) == {"experiments/replay/nested/helper.py", "experiments/io.py"}
    archive = tmp_path / "snapshot.zip"
    assert provenance.snapshot(archive) == before
    with zipfile.ZipFile(archive) as contents:
        assert "experiments/replay/nested/helper.py" in contents.namelist()
        assert "experiments/io.py" in contents.namelist()
        assert not any("stages/" in name for name in contents.namelist())
        assert json.loads(contents.read("source_hashes.json")) == hashes
    nested.write_text("VALUE = 3\n")
    assert provenance.fingerprint()[0] != before


@pytest.mark.parametrize(
    "entrypoint",
    [
        "experiments/scripts/03_temporal_guidance_test.py",
        "experiments/scripts/precision_replay.py",
    ],
)
def test_replay_help_never_imports_model_stack(entrypoint):
    """Help must work without model/checkpoint/GPU dependencies installed."""
    code = (
        "import runpy, sys; "
        f'sys.argv=[{entrypoint!r}, "--help"]; '
        '\ntry: runpy.run_path(sys.argv[0],run_name="__main__")'
        "\nexcept SystemExit as error: assert error.code==0"
        '\nassert "torch" not in sys.modules'
        '\nassert "transformers" not in sys.modules'
        '\nassert "wan_va" not in sys.modules'
    )
    subprocess.run(
        [sys.executable, "-c", code],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )


def test_job_accounting_preserves_partial_timing_and_completion_checks(tmp_path):
    record = record_timing([1, 100, 110, 130, 140, 0, 200])
    assert record["allocated_seconds"] == 100
    assert record["client_seconds"] is None and not record["client_completed"]
    config = tmp_path / "config.json"
    config.write_text(
        json.dumps(dict(episodes=[dict(status="completed", success=False)]))
    )
    check_completion(tmp_path, 1)
    with pytest.raises(SystemExit, match="did not succeed"):
        check_completion(tmp_path, 1, require_success=True)
    with pytest.raises(SystemExit, match="differs"):
        check_completion(tmp_path, 2)


def test_capture_loader_preserves_rgb_and_promotes_state_without_wam(tmp_path):
    import torch
    from experiments.replay.artifacts import load_capture

    diagnostic = tmp_path / "diagnostics"
    diagnostic.mkdir()
    sample = torch.ones(1, 1, 2, 2, 2, dtype=torch.bfloat16)
    capture = dict(
        sample=sample,
        velocity=sample * 2,
        sigma=torch.tensor(0.4),
        base_next=sample * 3,
        fixed_first_frame=True,
        observed=sample[:, :, :1],
        reference_frames=[7],
        reference_rgb=torch.zeros(1, 3, 8, 8),
    )
    torch.save(capture, diagnostic / "capture_step.pt")
    (tmp_path / "config.json").write_text(
        json.dumps(
            dict(
                episodes=[
                    dict(selected_reference=dict(latent="/missing/reference.pth"))
                ]
            )
        )
    )
    loaded = load_capture(tmp_path, device="cpu")
    assert loaded.step["sample"].dtype == torch.float32
    assert torch.equal(loaded.step["sample"], sample.float())
    assert loaded.reference_indices == [7] and torch.equal(
        loaded.reference_rgb, capture["reference_rgb"]
    )
    assert loaded.observed.dtype == torch.float32
    with pytest.raises(ValueError, match="No captured reference RGB"):
        load_capture(tmp_path, frames_override="8", device="cpu")


def test_strength_measurement_preserves_zero_control_and_failure_rows(monkeypatch):
    import torch
    from experiments.replay.artifacts import CapturedStep
    from experiments.replay.precision import measure_strength

    for name in ("synchronize", "reset_peak_memory_stats"):
        monkeypatch.setattr(torch.cuda, name, lambda: None)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda: 0)
    sample = torch.ones(1, 1, 2, 2, 2)
    capture = CapturedStep(
        step=dict(
            sample=sample,
            velocity=sample * 0.5,
            sigma=torch.tensor(0.4),
            base_next=sample * 0.95,
            fixed_first_frame=True,
        ),
        observed=None,
        reference_rgb=None,
        reference_path=Path("/unused"),
        reference_indices=[7],
    )
    zero = measure_strength(capture, lambda clean: clean.sum(), 0, "float32", [7], 0)
    small = measure_strength(
        capture, lambda clean: clean.sum(), 0.01, "float32", [7], 0
    )
    larger = measure_strength(
        capture, lambda clean: clean.sum(), 0.03, "float32", [7], 0
    )
    assert zero["status"] == "ok" and zero["reward_gain"] == 0
    assert zero["gradient_norm"] is None and zero["applied_correction_norm"] == 0
    assert small["reward_gain"] > 0 and larger["reward_gain"] > small["reward_gain"]
    assert larger["applied_correction_norm"] == pytest.approx(
        3 * small["applied_correction_norm"], rel=1e-4
    )
    failed = measure_strength(
        capture, lambda clean: clean.sum() * torch.nan, 0.01, "float32", [7], 0
    )
    assert failed["status"] == "failed" and "Non-finite" in failed["error"]
